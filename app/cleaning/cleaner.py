"""
CLEANER - applies the cleaning plan using plain, testable Python. No AI here.

Rules this file follows
  * Only steps with status "planned" AND an allowed, changing action are run.
    needs_review / skipped steps are logged but never touch the data.
  * An unknown action is refused (allow-list in rules.py).
  * Each step works on a copy. If it crashes, the data from before that step is
    kept, the step is marked "failed", and the next steps still run.
  * The original row numbers are kept as the index, so the change log can say
    "row 1234 of your file" even after duplicates are removed.
  * Columns are never added, dropped, renamed or re-ordered.
"""
import re
from datetime import date, datetime

import pandas as pd

from app.cleaning.profiler import cut_text, date_shape, is_pii_column, null_mask, shape_of
from app.cleaning.rules import MUTATING_ACTIONS, NUMBER_WORDS

MAX_LISTED_ROWS = 50        # how many row numbers we list in a log entry
MAX_SAMPLES = 3             # how many before/after examples we show per column

CURRENCY_PREFIX = re.compile(r"^\s*(?:rs\.?|inr|₹|\$)\s*", re.IGNORECASE)
GROUPED_NUMBER = re.compile(r"^-?\d{1,3}(?:,\d{2,3})+(?:\.\d+)?$")    # 1,299  or  1,23,456
PLAIN_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")


# ---------------------------------------------------------------- helpers
def _record(info, col, old, new):
    """Count changed cells and keep a few safe examples (never for personal-data columns)."""
    changed = old != new
    n = int(changed.sum())
    info["cells_changed"] = info.get("cells_changed", 0) + n
    if n and not is_pii_column(col):
        rows = old.index[changed][:MAX_SAMPLES]
        info.setdefault("samples", {})[col] = [
            {"row": int(i) + 1, "before": cut_text(old[i]), "after": cut_text(new[i])} for i in rows
        ]
    return n


# ---------------------------------------------------------------- actions
def _trim_whitespace(df, step):
    out, info = df.copy(), {}
    for col in step["columns"]:
        old = out[col]
        new = old.str.strip().str.replace(r"\s{2,}", " ", regex=True)
        _record(info, col, old, new)
        out[col] = new
    return out, info


def _strip_money_value(v):
    s = v.strip()
    t = CURRENCY_PREFIX.sub("", s)
    if GROUPED_NUMBER.match(t):
        t = t.replace(",", "")
    # change only if the result is a clean number; otherwise leave the value alone
    return t if (t != s and PLAIN_NUMBER.match(t)) else v


def _strip_currency(df, step):
    out, info = df.copy(), {}
    for col in step["columns"]:
        old = out[col]
        new = old.map(_strip_money_value)
        _record(info, col, old, new)
        out[col] = new
    return out, info


def _convert_number_words(df, step):
    out, info = df.copy(), {}
    for col in step["columns"]:
        old = out[col]
        new = old.map(lambda v: str(NUMBER_WORDS[v.strip().lower()]) if v.strip().lower() in NUMBER_WORDS else v)
        _record(info, col, old, new)
        out[col] = new
    return out, info


def _to_iso(value, shape, order):
    """Returns (iso_text, was_ambiguous) or None if it is not a real date."""
    try:
        if order == "named_month":
            for fmt in ("%d %b %Y", "%d %B %Y"):
                try:
                    return datetime.strptime(value, fmt).date().isoformat(), False
                except ValueError:
                    pass
            return None
        sep = "/" if shape == "NN/NN/YYYY" else "-"
        a, b, y = (int(x) for x in value.split(sep))
        d = date(y, b, a) if order == "day_first" else date(y, a, b)
        return d.isoformat(), (a <= 12 and b <= 12 and a != b)
    except (ValueError, TypeError):
        return None


def _convert_dates(df, step):
    col, shape, order = step["columns"][0], step["params"]["shape"], step["params"]["order"]
    out = df.copy()
    old = out[col]
    stripped = old.str.strip()
    in_shape = stripped.map(lambda v: date_shape(v) == shape)
    sub = stripped[in_shape]
    mapping = {v: _to_iso(v, shape, order) for v in sub.unique()}
    iso = sub.map(lambda v: mapping[v][0] if mapping[v] else None)
    ok = iso.notna()
    new = old.copy()
    new.loc[iso.index[ok]] = iso[ok]
    ambiguous = sub[ok].map(lambda v: mapping[v][1])
    info = {"order_used": order, "not_real_dates_left_unchanged": int((~ok).sum()),
            "ambiguous_values_converted": int(ambiguous.sum()),
            "ambiguous_source_rows": [int(i) + 1 for i in ambiguous.index[ambiguous][:MAX_LISTED_ROWS]]}
    _record(info, col, old, new)
    out[col] = new
    return out, info


def _canonicalize(df, step):
    out, info = df.copy(), {"merged": []}
    for col in step["columns"]:
        old = out[col]
        counts = old[~null_mask(old)].value_counts()
        groups = {}
        for raw, n in counts.items():
            key = re.sub(r"[^a-z0-9]", "", raw.lower())
            if key:
                groups.setdefault(key, []).append((raw, int(n)))
        mapping = {}
        for members in groups.values():
            if len(members) > 1:
                # most frequent spelling wins; tie -> looks like Title Case; then alphabetical
                members.sort(key=lambda rn: (-rn[1], 0 if rn[0] == rn[0].title() else 1, rn[0]))
                canon = members[0][0]
                for raw, _ in members[1:]:
                    mapping[raw] = canon
        new = old.map(lambda v: mapping.get(v, v))
        _record(info, col, old, new)
        info["merged"] += [{"from": cut_text(k), "to": cut_text(v)} for k, v in list(mapping.items())[:10]]
        out[col] = new
    return out, info


def _normalize_ids(df, step):
    col, dominant = step["columns"][0], step["params"]["dominant_pattern"]
    m = re.fullmatch(r"[Aa]+([^Aa9]*)9", dominant)
    if not m:
        raise ValueError(f"unsupported pattern {dominant}")
    sep = m.group(1)
    out = df.copy()
    old = out[col]
    stripped = old.str.strip()
    parts = stripped.map(lambda v: re.fullmatch(r"([A-Za-z]+)[\W_]*(\d+)", v))
    conforming = stripped[stripped.map(shape_of) == dominant]
    prefix_counts = {}
    for v in conforming:
        p = re.match(r"[A-Za-z]+", v)
        if p:
            prefix_counts.setdefault(p.group(0).lower(), {}).setdefault(p.group(0), 0)
            prefix_counts[p.group(0).lower()][p.group(0)] += 1
    canon_prefix = {k: max(v, key=v.get) for k, v in prefix_counts.items()}

    def fix(v, s, match):
        if match is None or shape_of(s) == dominant:
            return v                      # already in the right format, or not a recognisable code
        prefix = canon_prefix.get(match.group(1).lower(), match.group(1).upper())
        return prefix + sep + match.group(2)

    new = pd.Series([fix(v, s, mt) for v, s, mt in zip(old, stripped, parts)], index=old.index)
    info = {}
    _record(info, col, old, new)
    out[col] = new
    return out, info


def _lowercase_email(df, step):
    out, info = df.copy(), {}
    for col in step["columns"]:
        old = out[col]
        new = old.str.strip().str.lower()
        _record(info, col, old, new)
        out[col] = new
    return out, info


def _drop_duplicates(df, step):
    dup = df.duplicated(keep="first")
    info = {"rows_removed": int(dup.sum()),
            "removed_source_rows": [int(i) + 1 for i in df.index[dup][:MAX_LISTED_ROWS]]}
    return df[~dup].copy(), info


ACTIONS = {
    "trim_whitespace": _trim_whitespace,
    "strip_currency_and_commas": _strip_currency,
    "convert_number_words": _convert_number_words,
    "convert_date_to_iso": _convert_dates,
    "canonicalize_spelling_variants": _canonicalize,
    "normalize_id_format": _normalize_ids,
    "lowercase_email": _lowercase_email,
    "drop_exact_duplicates": _drop_duplicates,
}
assert set(ACTIONS) == MUTATING_ACTIONS, "every changing action must have an implementation"


# ------------------------------------------------------------- main entry
def clean_dataframe(dataset, df, plan):
    """Apply the plan. Returns (cleaned_df, change_log). The input df is not modified."""
    work = df.fillna("").astype(str).copy()
    log = []
    for step in plan:
        entry = {"step_id": step["step_id"], "order": step["order"], "action": step["action"],
                 "columns": step["columns"], "rows_before": len(work), "rows_after": len(work),
                 "cells_changed": 0, "details": {}}

        if step["status"] != "planned":
            entry["status"] = step["status"]          # needs_review / skipped: data untouched
            log.append(entry)
            continue
        if step["action"] not in ACTIONS:
            entry.update(status="failed", error=f"action '{step['action']}' is not allowed")
            log.append(entry)
            continue
        try:
            missing = [c for c in step["columns"] if c != "*" and c not in work.columns]
            if missing:
                raise KeyError(f"column not found: {missing}")
            new_work, info = ACTIONS[step["action"]](work, step)
            work = new_work
            entry["cells_changed"] = info.pop("cells_changed", 0)
            entry["rows_after"] = len(work)
            entry["details"] = info
            entry["status"] = "applied"
        except Exception as exc:                      # one bad step must not stop the others
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:200])
        log.append(entry)
    return work, log
