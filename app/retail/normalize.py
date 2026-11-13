"""
RETAIL NORMALIZATION - fixes names that only a shop-aware program understands.

  "Tee" / "tshirt" -> "T-Shirt"     "Bangalore" -> "Bengaluru"     "Yes" / "1" -> "Y"

Every change is a DECISION with a confidence number:
  confidence >= 0.85  -> applied
  confidence <  0.85  -> needs_review (value is left alone, our best guess is shown)
  unknown value in a closed list -> needs_review with a spelling suggestion (never applied)

Source is always "code" (a fixed alias table). A model could later add suggestions,
but they would be marked source="ai" and still go through the same confidence rule.

Rules match COLUMN NAMES (category, city, returned), not file names,
so they also work on hidden datasets with different file names.

Run:  python -m app.retail.normalize data/cleaned --out data/retail
"""
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

from app.cleaning.profiler import cut_text, load_csv
from app.errors import DataLoadError

MIN_CONFIDENCE = 0.85


def _key(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


# ------------------------------------------------------------ alias tables
# key (lowercase, letters+digits only) -> (canonical value, confidence, why)
CATEGORY_CANONICAL = ["T-Shirt", "Jeans", "Jacket", "Shoes", "Dress"]
CATEGORY_ALIASES = {
    "tee": ("T-Shirt", 0.95, "'Tee' is the common short name for a T-Shirt"),
    "tees": ("T-Shirt", 0.95, "plural of Tee"),
    "tshirts": ("T-Shirt", 0.98, "plural of T-Shirt"),
    "jackets": ("Jacket", 0.98, "plural of Jacket"),
    "dresses": ("Dress", 0.98, "plural of Dress"),
    "footwear": ("Shoes", 0.90, "Footwear is the broader word for Shoes"),
    "denim": ("Jeans", 0.75, "Denim could also mean denim jackets, so a person should confirm"),
}

CITY_CANONICAL = ["Delhi", "Bengaluru", "Kolkata", "Mumbai", "Pune", "Chennai", "Hyderabad", "Ahmedabad"]
CITY_ALIASES = {
    "bangalore": ("Bengaluru", 0.98, "older English name of Bengaluru"),
    "bombay": ("Mumbai", 0.98, "older name of Mumbai"),
    "calcutta": ("Kolkata", 0.98, "older name of Kolkata"),
    "madras": ("Chennai", 0.98, "older name of Chennai"),
    "newdelhi": ("Delhi", 0.80, "New Delhi is one part of Delhi, so a person should confirm"),
}

FLAG_CANONICAL = ["Y", "N"]
FLAG_ALIASES = {
    "yes": ("Y", 1.0, "yes means Y"), "true": ("Y", 1.0, "true means Y"),
    "1": ("Y", 0.90, "1 in a returned column means yes"),
    "no": ("N", 1.0, "no means N"), "false": ("N", 1.0, "false means N"),
    "0": ("N", 0.90, "0 in a returned column means no"),
}

RULES = [
    {"name": "category", "columns": re.compile(r"^(product_)?category$"), "canonical": CATEGORY_CANONICAL,
     "aliases": CATEGORY_ALIASES, "closed": True},
    {"name": "city", "columns": re.compile(r"^(store_|customer_)?city$"), "canonical": CITY_CANONICAL,
     "aliases": CITY_ALIASES, "closed": False},       # open list: unknown cities are left alone
    {"name": "returned_flag", "columns": re.compile(r"^(is_)?returned$|^return(ed)?_flag$"),
     "canonical": FLAG_CANONICAL, "aliases": FLAG_ALIASES, "closed": True},
]


# ------------------------------------------------------------ the logic
def _decision(dataset, column, raw, target, conf, how, why, status, rows):
    tag = re.sub(r"[^A-Za-z0-9]+", "_", raw)[:20] + "_" + hashlib.md5(raw.encode()).hexdigest()[:6]
    return {
        "decision_id": f"NORM:{dataset}:{column}:{tag}",
        "dataset": dataset, "column": column,
        "from": cut_text(raw), "to": target, "confidence": conf,
        "rule": how, "why": why, "source": "code", "status": status, "rows_affected": int(rows),
    }


def _normalize_series(dataset, column, series, rule, min_conf):
    canon_by_key = {_key(c): c for c in rule["canonical"]}
    mapping, decisions = {}, []
    for raw, n in series.value_counts().items():
        if raw.strip() == "":
            continue                                      # empty stays empty
        k = _key(raw)
        if k in canon_by_key:
            target, conf, how, why = canon_by_key[k], 1.0, "exact", "same value ignoring case and punctuation"
        elif k in rule["aliases"]:
            target, conf, why = rule["aliases"][k]
            how = "alias_table"
        elif rule["closed"]:
            close = difflib.get_close_matches(k, list(canon_by_key), n=1, cutoff=0.75)
            if close:
                target = canon_by_key[close[0]]
                conf = round(difflib.SequenceMatcher(None, k, close[0]).ratio(), 2)
                how, why = "spelling_suggestion", "looks like a misspelling; code only suggests, a person confirms"
            else:
                decisions.append(_decision(dataset, column, raw, None, 0.0, "no_match",
                                           "value is not in the known list", "needs_review", n))
                continue
        else:
            continue                                      # open list, unknown value: leave alone
        if target == raw:
            continue                                      # already correct
        applied = conf >= min_conf and how != "spelling_suggestion"
        if applied:
            mapping[raw] = target
        decisions.append(_decision(dataset, column, raw, target, conf, how, why,
                                   "applied" if applied else "needs_review", n))
    return series.map(lambda v: mapping.get(v, v)), decisions


def normalize_dataframe(dataset, df, min_conf=MIN_CONFIDENCE):
    out, decisions = df.copy(), []
    for col in df.columns:
        rule = next((r for r in RULES if r["columns"].match(str(col).strip().lower())), None)
        if rule is None:
            continue
        out[col], decs = _normalize_series(dataset, col, out[col], rule, min_conf)
        decisions += decs
    decisions.sort(key=lambda d: (d["column"], d["from"]))
    return out, decisions


# ------------------------------------------------------------ file level
def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_file(path, out_dir, min_conf=MIN_CONFIDENCE):
    path, out_dir = Path(path), Path(out_dir)
    base = re.sub(r"_clean$", "", path.stem)
    df, meta = load_csv(path)
    out, decisions = normalize_dataframe(base, df, min_conf)
    again, _ = normalize_dataframe(base, out, min_conf)

    changed_cols = [c for c in df.columns if not df[c].equals(out[c])]
    rule_cols = [c for c in df.columns if any(r["columns"].match(str(c).strip().lower()) for r in RULES)]
    cells = int(sum((df[c] != out[c]).sum() for c in df.columns))
    expected_cells = sum(d["rows_affected"] for d in decisions if d["status"] == "applied")
    checks = {
        "rows_unchanged": len(df) == len(out),
        "schema_unchanged": list(df.columns) == list(out.columns),
        "only_rule_columns_changed": set(changed_cols) <= set(rule_cols),
        "changes_match_decisions": cells == expected_cells,
        "idempotent": bool(again.equals(out)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base}_retail.csv"
    out.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    result = {
        "dataset": {"name": base, "rows": len(out), "columns": len(out.columns),
                    "version": f"{base}_retail@{_sha(out_path)[:12]}", "built_from": path.name},
        "decisions": decisions,
        "validation": {"passed": all(checks.values()), "checks": checks, "cells_changed": cells},
        "artifacts": {"input": str(path), "retail": str(out_path), "retail_sha256": _sha(out_path)},
    }
    (out_dir / f"{base}_retail_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def normalize_folder(folder, out_dir, min_conf=MIN_CONFIDENCE):
    results = []
    for path in sorted(Path(folder).glob("*_clean.csv")):
        try:
            results.append(normalize_file(path, out_dir, min_conf))
        except DataLoadError as err:
            results.append({"dataset": {"name": path.stem}, "status": "error", "error": err.to_dict()})
        except Exception as exc:
            results.append({"dataset": {"name": path.stem}, "status": "error",
                            "error": {"code": "UNEXPECTED_ERROR", "message": f"{type(exc).__name__}: {exc}"[:200]}})
    return results


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "data/retail"
    args = [a for a in args if a != out]
    if not args:
        print("Usage: python -m app.retail.normalize <folder with *_clean.csv> [--out data/retail]")
        sys.exit(1)
    for r in normalize_folder(args[0], out):
        if r.get("status") == "error":
            print(f"{r['dataset']['name']:<10} ERROR {r['error']['code']}")
            continue
        applied = [d for d in r["decisions"] if d["status"] == "applied"]
        review = [d for d in r["decisions"] if d["status"] == "needs_review"]
        mark = "PASS" if r["validation"]["passed"] else "FAIL"
        print(f"{r['dataset']['name']:<10} {len(applied)} applied, {len(review)} needs_review | "
              f"validation {mark} | {r['dataset']['version']}")
        for d in r["decisions"]:
            tag = "OK    " if d["status"] == "applied" else "REVIEW"
            print(f"    [{tag}] {d['column']}: '{d['from']}' -> {d['to']!r} "
                  f"(confidence {d['confidence']}, {d['rows_affected']} rows) {d['why']}")
