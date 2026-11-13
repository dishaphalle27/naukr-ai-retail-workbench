"""
SAFE JOINS - connect files without ever multiplying or losing rows by accident.

The danger (called "fan-out"): if you join orders to a table where the key repeats,
every order row is copied once per match and revenue gets inflated.

Rules enforced here
  1. Join keys are explicit. Nothing is guessed at query time.
  2. The "one" side must really be unique, or the join is REFUSED (FAN_OUT_RISK).
  3. After the join, row count must equal the left table's row count, or we refuse.
  4. Left joins keep every left row. Rows with no match ("orphans") are counted and
     reported, never dropped. (So revenue is never lost.)
  5. Many-to-many joins are always refused.
"""
import pandas as pd

from app.cleaning.profiler import is_pii_column
from app.errors import DataLoadError

MAX_SAMPLE_KEYS = 5


def _blank(series):
    return series.isna() | (series.astype(str).str.strip() == "")


def _nonblank(series):
    return series[~_blank(series)].astype(str).str.strip()


def _is_unique(series):
    vals = _nonblank(series)
    return len(vals) > 0 and vals.is_unique


def discover_joins(tables):
    """Find possible joins between tables from shared *_id column names.
    Returns a list of {"left", "right", "key", "cardinality", "safe"}."""
    found = []
    names = sorted(tables)
    for a in names:
        for b in names:
            if a == b:
                continue
            for col in tables[a].columns:
                low = str(col).lower()
                if col not in tables[b].columns or not (low == "id" or low.endswith("_id")):
                    continue
                a_unique, b_unique = _is_unique(tables[a][col]), _is_unique(tables[b][col])
                if b_unique and not a_unique:
                    found.append({"left": a, "right": b, "key": col, "cardinality": "many_to_one", "safe": True})
                elif a_unique and b_unique and a < b:
                    found.append({"left": a, "right": b, "key": col, "cardinality": "one_to_one", "safe": True})
                elif not a_unique and not b_unique and a < b:
                    found.append({"left": a, "right": b, "key": col, "cardinality": "many_to_many", "safe": False})
    return found


def safe_join(left, right, left_key, right_key, left_name="left", right_name="right"):
    """Left join, many-to-one. Returns (joined_df, report). Raises DataLoadError if unsafe."""
    for df, key, label in ((left, left_key, left_name), (right, right_key, right_name)):
        if key not in df.columns:
            raise DataLoadError("JOIN_KEY_MISSING", f"Column '{key}' not found in {label}",
                                {"table": label, "key": key})

    r_keys = _nonblank(right[right_key])
    dup_mask = r_keys.duplicated(keep=False)
    if dup_mask.any():
        raise DataLoadError(
            "FAN_OUT_RISK",
            f"Join refused: '{right_key}' is not unique in {right_name}, so rows in {left_name} would be copied.",
            {"duplicate_key_values": int(r_keys[dup_mask].nunique()),
             "example_keys": sorted(r_keys[dup_mask].unique())[:MAX_SAMPLE_KEYS],
             "left": left_name, "right": right_name})

    l = left.copy()
    r = right.copy()
    l["_k"] = l[left_key].astype(str).str.strip()
    r["_k"] = r[right_key].astype(str).str.strip()
    renamed = {}
    for col in r.columns:
        if col in ("_k", right_key):
            continue
        if col in l.columns:
            renamed[col] = f"{right_name}_{col}"
    r = r.rename(columns=renamed)
    keep_cols = [c for c in r.columns if c != right_key]      # the key already exists on the left
    joined = l.merge(r[keep_cols], on="_k", how="left", validate="many_to_one", indicator=True)

    if len(joined) != len(left):
        raise DataLoadError("FAN_OUT_DETECTED", "Join changed the number of rows; result discarded.",
                            {"left_rows": len(left), "joined_rows": len(joined)})

    left_blank = _blank(left[left_key]).to_numpy()
    matched = (joined["_merge"] == "both").to_numpy()
    orphan_mask = (~matched) & (~left_blank)
    orphan_keys = joined.loc[orphan_mask, "_k"]
    joined = joined.drop(columns=["_k", "_merge"])

    report = {
        "left": left_name, "right": right_name, "left_key": left_key, "right_key": right_key,
        "cardinality": "many_to_one", "how": "left",
        "left_rows": len(left), "right_rows": len(right), "joined_rows": len(joined),
        "rows_match": len(joined) == len(left),
        "matched_rows": int(matched.sum()),
        "orphan_rows": int(orphan_mask.sum()),
        "orphan_key_sample": sorted(orphan_keys.unique())[:MAX_SAMPLE_KEYS],
        "blank_key_rows": int(left_blank.sum()),
        "renamed_columns": renamed,
        "status": "ok" if not orphan_mask.any() and not left_blank.any() else "ok_with_unmatched_rows",
    }
    return joined, report


def join_orders_wide(tables, base="orders", exclude_pii=True):
    """Join the base table (orders) to every table it safely points at.
    Personal columns (name, email, phone...) are left out by default (privacy).
    Returns (wide_df, list_of_reports, list_of_skipped_joins)."""
    wide = tables[base]
    reports, skipped = [], []
    for j in discover_joins(tables):
        if j["left"] != base:
            continue
        if j["cardinality"] != "many_to_one":
            skipped.append({**j, "reason": "not many-to-one"})
            continue
        right = tables[j["right"]]
        pii = [c for c in right.columns if exclude_pii and c != j["key"] and is_pii_column(c)]
        try:
            wide, rep = safe_join(wide, right.drop(columns=pii), j["key"], j["key"], base, j["right"])
            rep["excluded_pii_columns"] = pii
            reports.append(rep)
        except DataLoadError as err:
            skipped.append({**j, "reason": err.to_dict()})
    return wide, reports, skipped
