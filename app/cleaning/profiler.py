"""
PROFILER - looks at a CSV and reports what is inside. It NEVER changes the data.

Main ideas
1. We read every column as TEXT. If pandas guessed types, it could silently
   change "007" into 7 or hide "Rs. 1,299". Text keeps the truth.
2. The profiler only reports FACTS (counts, formats, examples).
   Deciding "is this a problem?" is the job of the issue detector (next step).
3. Personal data (emails, phones...) is never copied into the profile as examples.
   Text that looks like a prompt injection is counted, never copied.

Use in code / notebook:
    from app.cleaning.profiler import profile_file
    profile = profile_file("data/sample/orders.csv")

Use in terminal:
    python -m app.cleaning.profiler data/sample/orders.csv
"""
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from app.errors import DataLoadError

# ----------------------------------------------------------------- settings
MAX_FILE_MB = 50                 # refuse bigger files (protects memory)
MAX_EXAMPLES = 5                 # never put more than this many examples in a profile
MAX_EXAMPLE_LEN = 40             # long text is cut
MAX_VARIANT_GROUPS = 10
LOW_CARDINALITY_LIMIT = 60       # "few distinct values" = category-like column
SMALL_DISTINCT_LIMIT = 12        # list all values if there are this few

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-"}

# Column names that probably hold personal data -> we never show example values
PII_NAMES = {"name", "first_name", "last_name", "customer_name", "full_name",
             "email", "phone", "mobile", "address", "dob"}
PII_PARTS = ("email", "phone", "mobile", "address")

# Text that tries to give instructions to an AI
INJECTION_RE = re.compile(
    r"ignore (?:all )?(?:previous|prior|above)|system prompt|disregard .{0,30}instructions"
    r"|reveal .{0,30}(?:prompt|secret|key|password)",
    re.IGNORECASE,
)

# Known date "shapes" (the look of the text, not yet the meaning)
DATE_SHAPES = {
    "YYYY-MM-DD": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "NN/NN/YYYY": re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$"),
    "NN-NN-YYYY": re.compile(r"^\d{1,2}-\d{1,2}-\d{4}$"),
    "DD Mon YYYY": re.compile(r"^\d{1,2} [A-Za-z]{3,9} \d{4}$"),
}


# ------------------------------------------------------------------ loading
def load_csv(path, max_mb=MAX_FILE_MB):
    """Read a CSV as text. Returns (DataFrame, meta). Raises DataLoadError."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise DataLoadError("FILE_NOT_FOUND", f"File not found: {path.name}")
    if path.suffix.lower() != ".csv":
        raise DataLoadError("BAD_FILE_TYPE", "Only .csv files are accepted",
                            {"file_name": path.name})
    size = path.stat().st_size
    if size > max_mb * 1024 * 1024:
        raise DataLoadError("FILE_TOO_LARGE", f"File is larger than {max_mb} MB",
                            {"size_bytes": size})
    if size == 0:
        raise DataLoadError("EMPTY_FILE", "File is empty (no header row)")

    meta = {"file_name": path.name, "size_bytes": size, "encoding": "utf-8",
            "encoding_fallback": False}
    options = dict(dtype=str, keep_default_na=False)
    try:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", **options)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1", **options)
            meta["encoding"] = "latin-1"
            meta["encoding_fallback"] = True
    except pd.errors.EmptyDataError:
        raise DataLoadError("EMPTY_FILE", "File has no readable header row")
    except pd.errors.ParserError as exc:
        # We do NOT skip bad lines: that would silently lose rows.
        raise DataLoadError("MALFORMED_CSV", "CSV structure is broken", {"detail": str(exc)[:200]})
    return df, meta


# ------------------------------------------------------------ small helpers
def _pct(part, total):
    return round(100.0 * part / total, 2) if total else 0.0


def _cut(text):
    text = str(text)
    return text if len(text) <= MAX_EXAMPLE_LEN else text[:MAX_EXAMPLE_LEN] + "..."


def _is_pii_column(name):
    low = str(name).strip().lower()
    return low in PII_NAMES or any(part in low for part in PII_PARTS)


def _null_mask(series):
    return series.str.strip().str.lower().isin(NULL_TOKENS)


def _valid_date(y, m, d):
    try:
        date(y, m, d)
        return True
    except ValueError:
        return False


def _date_shape(value):
    for name, rx in DATE_SHAPES.items():
        if rx.match(value):
            return name
    return None


def _check_date(value, shape):
    """Returns (is_real_calendar_date, is_ambiguous_day_month)."""
    try:
        if shape == "YYYY-MM-DD":
            datetime.strptime(value, "%Y-%m-%d")
            return True, False
        if shape == "DD Mon YYYY":
            for fmt in ("%d %b %Y", "%d %B %Y"):
                try:
                    datetime.strptime(value, fmt)
                    return True, False
                except ValueError:
                    pass
            return False, False
        sep = "/" if shape == "NN/NN/YYYY" else "-"
        a, b, y = (int(x) for x in value.split(sep))
        as_day_first = _valid_date(y, b, a)     # a = day, b = month
        as_month_first = _valid_date(y, a, b)   # a = month, b = day
        ambiguous = as_day_first and as_month_first and a != b
        return (as_day_first or as_month_first), ambiguous
    except ValueError:
        return False, False


def _order_proof(value, shape):
    """For NN/NN/YYYY style dates: does this value PROVE day-first or month-first?
    13/01/2026 can only be day-first (there is no month 13)."""
    sep = "/" if shape == "NN/NN/YYYY" else "-"
    try:
        a, b, y = (int(x) for x in value.split(sep))
    except ValueError:
        return None
    if a > 12 and b <= 12 and _valid_date(y, b, a):
        return "day_first_proof"
    if b > 12 and a <= 12 and _valid_date(y, a, b):
        return "month_first_proof"
    if a <= 12 and b <= 12 and a != b:
        return "ambiguous"
    return None


def _profile_dates(values):
    """values = non-null stripped text. Counts date shapes without parsing to dates."""
    formats, matched, invalid, ambiguous, evidence = {}, 0, 0, 0, {}
    for raw, n in values.value_counts().items():
        shape = _date_shape(raw)
        if shape is None:
            continue
        matched += n
        formats[shape] = formats.get(shape, 0) + int(n)
        ok, amb = _check_date(raw, shape)
        if not ok:
            invalid += int(n)
        if amb:
            ambiguous += int(n)
        if shape in ("NN/NN/YYYY", "NN-NN-YYYY"):
            proof = _order_proof(raw, shape)
            if proof:
                ev = evidence.setdefault(shape, {"day_first_proof": 0, "month_first_proof": 0, "ambiguous": 0})
                ev[proof] += int(n)
    return {"matched": int(matched), "formats": formats, "order_evidence": evidence,
            "invalid_calendar_dates": invalid, "ambiguous_day_month": ambiguous}


def _to_number(stripped):
    """Turn 'Rs. 1,299' into 1299. Returns (parsed, plain_parsed)."""
    plain = pd.to_numeric(stripped, errors="coerce")
    cleaned = (stripped.str.replace(r"(?i)^\s*(?:rs\.?|inr|₹|\$)\s*", "", regex=True)
                       .str.replace(",", "", regex=False).str.strip())
    return pd.to_numeric(cleaned, errors="coerce"), plain


def _spelling_variants(values):
    """Groups of different spellings of the same thing: 'West', 'west ', 'WEST'."""
    groups = {}
    for raw in values.unique():
        key = re.sub(r"[^a-z0-9]", "", raw.lower())
        if key:
            groups.setdefault(key, set()).add(raw)
    found = [sorted(raws)[:6] for raws in groups.values() if len(raws) > 1]
    return [[_cut(r) for r in g] for g in found[:MAX_VARIANT_GROUPS]]


def _shape(value):
    """Layout of a code, ignoring length but NOT case:
    'SKU-0007' -> 'A-9', 'sku-0007' -> 'a-9', 'SKU0007' -> 'A9', 'O5' == 'O10'."""
    v = re.sub(r"[A-Z]+", "A", value.strip())
    v = re.sub(r"[a-z]+", "a", v)
    return re.sub(r"\d+", "9", v)


# ---------------------------------------------------------- column profile
def profile_column(name, series):
    total = len(series)
    null_mask = _null_mask(series) if total else pd.Series([], dtype=bool)
    nulls = int(null_mask.sum())
    values = series[~null_mask] if total else series
    stripped = values.str.strip()
    non_null = len(values)
    pii = _is_pii_column(name)

    prof = {
        "null_count": nulls,
        "null_pct": _pct(nulls, total),
        "all_null": bool(total > 0 and nulls == total),
        "unique_count": int(values.nunique()),
        "unique_after_trim_lower": int(stripped.str.lower().nunique()),
        "values_with_edge_spaces": int((values != stripped).sum()),
        "values_with_double_spaces": int(values.str.contains(r"\s{2,}", regex=True).sum()),
        "pii_suspected": pii,
    }

    if non_null == 0:
        prof["kind"] = "empty"
        return prof

    dates = _profile_dates(stripped)
    numbers, plain = _to_number(stripped)
    numeric_ok = numbers.notna()
    date_share = dates["matched"] / non_null
    numeric_share = float(numeric_ok.mean())

    if date_share >= 0.5:
        prof["kind"] = "date"
        prof["date_formats"] = dates["formats"]
        prof["mixed_date_formats"] = len(dates["formats"]) > 1
        prof["ambiguous_day_month"] = dates["ambiguous_day_month"]
        prof["date_order_evidence"] = dates["order_evidence"]
        prof["unparsable_count"] = int(non_null - dates["matched"] + dates["invalid_calendar_dates"])
    elif numeric_share >= 0.8:
        prof["kind"] = "numeric"
        failed = stripped[~numeric_ok]
        prof["parse_failures"] = int(len(failed))
        prof["parse_failure_examples"] = [_cut(v) for v in failed.unique()[:MAX_EXAMPLES]]
        prof["numeric_with_symbols"] = int((plain.isna() & numeric_ok).sum())
        good = numbers.dropna()
        if len(good):
            prof["stats"] = {
                "min": float(good.min()), "max": float(good.max()),
                "mean": round(float(good.mean()), 2), "median": float(good.median()),
                "negatives": int((good < 0).sum()), "zeros": int((good == 0).sum()),
            }
    else:
        prof["kind"] = "text"
        inj = stripped.str.contains(INJECTION_RE, regex=True)
        prof["injection_like_values"] = int(inj.sum())
        if inj.any():
            # row numbers only (0-based, among non-null rows). The text is NOT copied.
            prof["injection_like_rows"] = [int(i) for i in values.index[inj.values][:MAX_EXAMPLES]]
        safe = stripped[~inj] if inj.any() else stripped
        if "email" in str(name).lower():
            prof["values_with_uppercase"] = int((stripped != stripped.str.lower()).sum())
        if not pii:
            distinct = safe.unique()
            prof["sample_values"] = [_cut(v) for v in distinct[:MAX_EXAMPLES]]
            if len(distinct) <= SMALL_DISTINCT_LIMIT:
                prof["distinct_values"] = sorted(_cut(v) for v in distinct)
            if prof["unique_count"] <= LOW_CARDINALITY_LIMIT:
                variants = _spelling_variants(values[~inj] if inj.any() else values)
                if variants:
                    prof["spelling_variants"] = variants
            # ID-like columns (mostly contain digits): check for mixed formats like SKU-001 / sku001
            digit_share = float(stripped.str.contains(r"\d", regex=True).mean())
            space_share = float(stripped.str.contains(r"\s", regex=True).mean())   # codes have no inner spaces
            if digit_share >= 0.9 and space_share <= 0.05 and non_null >= 8:
                shapes = stripped.map(_shape).value_counts()
                if 2 <= len(shapes) <= 6 and shapes.iloc[0] / non_null >= 0.5:
                    prof["format_patterns"] = {k: int(v) for k, v in shapes.items()}
    return prof


# ------------------------------------------------------- dataset profile
def _key_candidate(df):
    """First *_id column that is almost unique -> probably the row's key."""
    for col in df.columns:
        low = str(col).lower()
        if low == "id" or low.endswith("_id"):
            vals = df[col][~_null_mask(df[col])]
            if len(vals) and vals.nunique() / len(vals) >= 0.9:
                return col
    return None


def profile_dataframe(df, meta=None):
    df = df.fillna("").astype(str)            # work on text copy; caller's data untouched
    rows, cols = df.shape

    exact = df.duplicated() if rows else pd.Series([], dtype=bool)
    if rows:
        normalized = df.apply(lambda c: c.str.strip().str.lower())
        near = normalized.duplicated()
    else:
        near = exact

    key = _key_candidate(df) if rows else None
    key_info = None
    if key:
        key_nonnull = ~_null_mask(df[key])
        key_dup = df[key].duplicated() & key_nonnull
        key_info = {
            "column": key,
            "repeated_key_rows": int(key_dup.sum()),
            "conflicting_key_rows": int((key_dup & ~exact).sum()),   # same id, different content
        }

    header_issues = [str(c) for c in df.columns if str(c) != str(c).strip() or str(c).startswith("Unnamed")]

    return {
        "meta": meta or {},
        "row_count": int(rows),
        "column_count": int(cols),
        "column_names": [str(c) for c in df.columns],
        "header_issues": header_issues,
        "duplicates": {
            "exact_duplicate_rows": int(exact.sum()),
            "extra_after_trim_and_lowercase": int(near.sum() - exact.sum()),
            "key": key_info,
        },
        "columns": {str(c): profile_column(c, df[c]) for c in df.columns},
    }


def profile_file(path, max_mb=MAX_FILE_MB):
    df, meta = load_csv(path, max_mb=max_mb)
    return profile_dataframe(df, meta)


# ------------------------------------------------------- human-readable
def column_notes(col):
    """Short warnings for one column profile (used by the report)."""
    notes = []
    if col.get("all_null"):
        notes.append("ALL values empty")
    elif col["null_count"]:
        notes.append(f"{col['null_count']} empty ({col['null_pct']}%)")
    if col["values_with_edge_spaces"]:
        notes.append(f"{col['values_with_edge_spaces']} with leading/trailing spaces")
    if col.get("mixed_date_formats"):
        notes.append(f"mixed date formats {col['date_formats']}")
    if col.get("ambiguous_day_month"):
        notes.append(f"{col['ambiguous_day_month']} dates could be day/month OR month/day")
    if col.get("unparsable_count"):
        notes.append(f"{col['unparsable_count']} unreadable dates")
    if col.get("numeric_with_symbols"):
        notes.append(f"{col['numeric_with_symbols']} numbers stored as text (Rs., commas)")
    if col.get("parse_failures"):
        notes.append(f"{col['parse_failures']} not numbers, e.g. {col['parse_failure_examples']}")
    if col.get("stats", {}).get("negatives"):
        notes.append(f"{col['stats']['negatives']} negative values")
    if col.get("spelling_variants"):
        notes.append(f"spelling variants {col['spelling_variants'][:3]}")
    if col.get("format_patterns"):
        notes.append(f"mixed ID formats {col['format_patterns']}")
    if col.get("injection_like_values"):
        notes.append(f"{col['injection_like_values']} value(s) look like AI instructions (treated as plain data)")
    return notes


def format_report(profile):
    m = profile["meta"]
    d = profile["duplicates"]
    lines = [f"FILE {m.get('file_name', '?')} | {profile['row_count']} rows | "
             f"{profile['column_count']} columns | encoding {m.get('encoding', '?')}"]
    lines.append(f"DUPLICATES exact={d['exact_duplicate_rows']} "
                 f"extra-after-trim/lowercase={d['extra_after_trim_and_lowercase']}")
    if d["key"]:
        k = d["key"]
        lines.append(f"KEY '{k['column']}': {k['repeated_key_rows']} repeated, "
                     f"{k['conflicting_key_rows']} with conflicting content")
    for name, col in profile["columns"].items():
        lines.append(f"  - {name} [{col['kind']}] unique={col['unique_count']}")
        for note in column_notes(col):
            lines.append(f"      ! {note}")
    return "\n".join(lines)


# Public names so other modules (cleaner) can reuse these helpers
shape_of = _shape
date_shape = _date_shape
is_pii_column = _is_pii_column
null_mask = _null_mask
cut_text = _cut


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.cleaning.profiler <file.csv> [--json]")
        sys.exit(1)
    try:
        result = profile_file(sys.argv[1])
    except DataLoadError as err:
        print(json.dumps(err.to_dict(), indent=2))
        sys.exit(2)
    print(json.dumps(result, indent=2) if "--json" in sys.argv else format_report(result))
