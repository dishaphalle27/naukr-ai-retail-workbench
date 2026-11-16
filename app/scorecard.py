"""
SCORECARD - grades the whole cleaning pipeline against the GROUND TRUTH, cell by cell.

Most projects say "the cleaner works". This one can prove how well, because the sample data was
generated from a known clean version (data/sample_truth) and then deliberately damaged.
For every cell we know three things: the TRUTH, the MESSY value, and what the pipeline produced.

    python -m app.scorecard                       # grades data/sample against data/sample_truth

Every cell where messy != truth is a "corrupted cell". The pipeline's result for it is one of:

  REPAIRED     output equals the truth                          (good)
  LEFT AS-IS   output equals the messy value: not repaired      (honest: needs a person, or cannot be recovered)
  WRONG FIX    output changed the value to something that is still wrong      (must be 0)
  DAMAGED      the messy value was already right and we broke it              (must be 0)

Also checked: rows lost (must be 0), rows invented (must be 0), and "silent leftovers": cells left as-is in a
column that the reports do NOT flag (must be 0). So nothing is left broken without being visible.

Exit code 0 only if wrong fixes, damage, lost rows, invented rows and silent leftovers are all zero.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

from app.chat.catalog import Catalog
from app.cleaning.pipeline import clean_folder
from app.retail.normalize import normalize_folder

KEYS = {"stores": ["store_id"], "products": ["product_id"], "customers": ["customer_id"],
        "orders": ["order_id"], "inventory": ["product_id", "store_id"]}


def _load(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _keyed(df, keys):
    """key -> row, first occurrence wins (the same rule the cleaner uses for duplicates)."""
    out = {}
    for rec in df.to_dict("records"):
        out.setdefault(tuple(str(rec[k]).strip().lower() for k in keys), rec)
    return out


def score_table(truth, messy, cleaned, keys):
    """Classify every cell of one table. Returns counts plus per-column 'left as-is' counts."""
    t, m, c = _keyed(truth, keys), _keyed(messy, keys), _keyed(cleaned, keys)
    cols = list(truth.columns)
    r = {"cells": 0, "corrupted": 0, "repaired": 0, "left_as_is": 0, "wrong_fix": 0, "damaged": 0,
         "changed": 0, "changed_correct": 0, "rows_truth": len(t), "rows_lost": 0, "rows_invented": 0,
         "left_by_column": {}, "wrong_by_column": {}}
    r["rows_lost"] = sum(1 for k in t if k not in c)
    r["rows_invented"] = sum(1 for k in c if k not in t)
    for k, trow in t.items():
        if k not in c or k not in m:
            continue
        for col in cols:
            tv, mv, cv = trow[col], m[k][col], c[k][col]
            r["cells"] += 1
            if cv != mv:
                r["changed"] += 1
                r["changed_correct"] += cv == tv
            if mv == tv:
                if cv != tv:
                    r["damaged"] += 1
                continue
            r["corrupted"] += 1
            if cv == tv:
                r["repaired"] += 1
            elif cv == mv:
                r["left_as_is"] += 1
                r["left_by_column"][col] = r["left_by_column"].get(col, 0) + 1
            else:
                r["wrong_fix"] += 1
                r["wrong_by_column"][col] = r["wrong_by_column"].get(col, 0) + 1
    return r


def _visible_columns(cleaned_results, normalized_results, catalog):
    """Which (dataset, column) pairs do the reports flag? Left-as-is cells outside these are 'silent'."""
    flagged = {}

    def add(ds, col, why):
        flagged.setdefault((ds, col), set()).add(why)

    for res in cleaned_results:
        if res.get("status") == "error":
            continue
        name = res["dataset"]["name"]
        for issue in res["issues_after"]:
            if issue["column"]:
                add(name, issue["column"], issue["type"])
    for res in normalized_results:
        if res.get("status") == "error":
            continue
        for d in res["decisions"]:
            if d["status"] == "needs_review":
                add(res["dataset"]["name"], d["column"], "needs_review alias")
    for view, reports in catalog.join_reports.items():
        for rep in reports:
            if rep["orphan_rows"] or rep["blank_key_rows"]:
                add(view.replace("_wide", ""), rep["left_key"], "unmatched join key")
    return flagged


def score(messy_dir, truth_dir, workdir):
    messy_dir, truth_dir, workdir = Path(messy_dir), Path(truth_dir), Path(workdir)
    cleaned = clean_folder(messy_dir, workdir / "clean")
    normalized = normalize_folder(workdir / "clean", workdir / "retail")
    catalog = Catalog.from_folder(workdir / "retail")
    flagged = _visible_columns(cleaned, normalized, catalog)

    tables, totals = {}, {k: 0 for k in ("cells", "corrupted", "repaired", "left_as_is", "wrong_fix", "damaged",
                                         "changed", "changed_correct", "rows_truth", "rows_lost", "rows_invented", "silent")}
    for name, keys in KEYS.items():
        truth_path, messy_path, out_path = truth_dir / f"{name}.csv", messy_dir / f"{name}.csv", workdir / "retail" / f"{name}_retail.csv"
        if not (truth_path.exists() and messy_path.exists() and out_path.exists()):
            continue
        t = score_table(_load(truth_path), _load(messy_path), _load(out_path), keys)
        t["left_explained"] = {col: sorted(flagged.get((name, col), [])) for col in t["left_by_column"]}
        t["silent"] = sum(n for col, n in t["left_by_column"].items() if not flagged.get((name, col)))
        tables[name] = t
        for k in totals:
            totals[k] += t.get(k, 0)
    totals["repair_rate"] = round(totals["repaired"] / totals["corrupted"], 4) if totals["corrupted"] else None
    totals["repair_precision"] = round(totals["changed_correct"] / totals["changed"], 4) if totals["changed"] else None
    totals["passed"] = all(totals[k] == 0 for k in ("wrong_fix", "damaged", "rows_lost", "rows_invented", "silent"))
    return {"totals": totals, "tables": tables}


def format_report(result):
    t = result["totals"]
    L = ["SCORECARD  (pipeline output vs the clean truth the sample data was made from)", ""]
    L.append(f"  cells compared                {t['cells']:>7,}")
    L.append(f"  corrupted cells (messy!=truth){t['corrupted']:>7,}")
    L.append(f"    REPAIRED exactly            {t['repaired']:>7,}   ({t['repair_rate']:.1%} of corrupted)")
    L.append(f"    LEFT AS-IS (see below)      {t['left_as_is']:>7,}")
    L.append(f"    WRONG FIX                   {t['wrong_fix']:>7,}   <- must be 0")
    L.append(f"  DAMAGED (was right, broke it) {t['damaged']:>7,}   <- must be 0")
    L.append(f"  rows lost / rows invented     {t['rows_lost']:>3} / {t['rows_invented']:<3}  <- must be 0 / 0")
    L.append(f"  repair precision              {t['repair_precision']:.1%}   (of all cells the cleaner changed, how many now equal the truth)")
    L.append(f"  silent leftovers              {t['silent']:>7,}   <- must be 0 (left-as-is cells in a column no report flags)")
    L.append("")
    L.append("  Left as-is, and why a person can see it:")
    any_left = False
    for name, tb in result["tables"].items():
        for col, n in sorted(tb["left_by_column"].items()):
            any_left = True
            why = ", ".join(tb["left_explained"][col]) or "NOT FLAGGED"
            L.append(f"    {name}.{col:<18}{n:>4} cells   flagged as: {why}")
    if not any_left:
        L.append("    (none)")
    L.append("")
    L.append("  RESULT: " + ("PASS" if t["passed"] else "FAIL"))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m app.scorecard", description="Grade the pipeline against the ground truth.")
    ap.add_argument("--messy", default="data/sample")
    ap.add_argument("--truth", default="data/sample_truth")
    ap.add_argument("--json", help="also write the full result to this file")
    args = ap.parse_args(argv)
    if not (Path(args.messy).is_dir() and Path(args.truth).is_dir()):
        print("Need the sample data and its truth. Run: python scripts/generate_messy_data.py")
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        result = score(args.messy, args.truth, tmp)
    print(format_report(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["totals"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
