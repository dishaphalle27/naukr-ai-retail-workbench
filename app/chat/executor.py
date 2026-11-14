"""
EXECUTOR - runs a plan that has ALREADY passed the validator. Plain pandas, no eval, no SQL.

Every answer comes back with EVIDENCE:
  dataset + version, columns used, filters, joins behind the dataset, the ordered list of
  operations (with row counts after each), rows considered, result size, formulas, and
  anything that was ignored (so no number is ever unexplained).
"""
import re
import time

import numpy as np
import pandas as pd

from app.chat.plan_schema import QUERY_SECONDS
from app.errors import DataLoadError
from app.retail.metrics import NO_MATCH, SALES_FORMULAS, sales_metrics

PLAIN_AGG = {"sum": "sum", "mean": "mean", "median": "median", "min": "min", "max": "max",
             "count": "count", "count_distinct": "nunique"}
PLAIN_WORDS = {"sum": "sum of {f}", "mean": "average of {f}", "median": "median of {f}", "min": "smallest {f}",
               "max": "largest {f}", "count": "number of rows with a value in {f}",
               "count_distinct": "number of different values in {f}"}


class Deadline:
    """Soft time budget, checked between steps (pandas calls themselves cannot be interrupted)."""

    def __init__(self, seconds=QUERY_SECONDS):
        self.start, self.seconds = time.monotonic(), seconds

    def check(self, where):
        if time.monotonic() - self.start > self.seconds:
            raise DataLoadError("QUERY_TOO_SLOW", f"Stopped: the question took longer than {self.seconds} seconds",
                                {"stopped_at": where})


def _py(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if pd.isna(v) else round(float(v), 2)
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return str(v)[:10]
    return v


def _rows(df):
    return [{c: _py(v) for c, v in rec.items()} for rec in df.to_dict(orient="records")]


def _mask(df, f, kind):
    s = df[f.field]
    op, v = f.op, f.value
    if op == "is_null":
        return s.isna()
    if op == "not_null":
        return s.notna()
    if kind == "text":
        low = s.str.strip().str.lower()           # text is compared ignoring case
        if op == "eq":
            return low == v.strip().lower()
        if op == "ne":
            return low.notna() & (low != v.strip().lower())
        if op == "in":
            return low.isin([x.strip().lower() for x in v])
        if op == "not_in":
            return low.notna() & ~low.isin([x.strip().lower() for x in v])
        if op == "contains":
            return low.str.contains(re.escape(v.strip().lower()), na=False)
    if kind == "date":
        t = s.fillna("")
        ok = s.notna()
        return {"eq": ok & (t == v), "ne": ok & (t != v), "gt": ok & (t > v), "gte": ok & (t >= v),
                "lt": ok & (t < v), "lte": ok & (t <= v)}[op] if op != "between" else ok & (t >= v[0]) & (t <= v[1])
    # numeric
    if op == "between":
        return (s >= v[0]) & (s <= v[1])
    if op == "in":
        return s.isin(v)
    if op == "not_in":
        return s.notna() & ~s.isin(v)
    return {"eq": s == v, "ne": s.notna() & (s != v), "gt": s > v, "gte": s >= v, "lt": s < v, "lte": s <= v}[op]


def _describe_filter(f):
    return {"field": f.field, "op": f.op, "value": f.value}


def execute(plan, catalog, warnings=None, deadline=None):
    deadline = deadline or Deadline()
    ds = plan.dataset
    df = catalog.typed(ds)
    kinds = {c: i["kind"] for c, i in catalog.columns(ds).items()}
    total_rows = len(df)
    ops, notes, used = [], [], set()

    # ------------------------------------------------------------ filters
    for f in plan.filters:
        deadline.check("filter")
        df = df[_mask(df, f, kinds[f.field])]
        used.add(f.field)
        compare = " (text compared ignoring upper/lower case)" if kinds[f.field] == "text" else ""
        ops.append({"op": "filter", "detail": f"{f.field} {f.op} {f.value!r}{compare}", "rows_after": len(df)})
    after_filters = len(df)

    result_cols, rows, groups_total = [], [], None

    # ------------------------------------------------------------ aggregate
    if plan.intent == "aggregate":
        deadline.check("aggregate")
        work = df.copy()
        keys = list(plan.group_by)
        used.update(keys)
        if plan.time_bucket:
            tb = plan.time_bucket
            name = f"{tb.field}_{tb.unit}"
            work[name] = work[tb.field].str[: 7 if tb.unit == "month" else 4]
            keys.append(name)
            used.add(tb.field)
            ops.append({"op": "time_bucket", "detail": f"{tb.field} grouped by {tb.unit}"})
        missing_key_rows = 0
        for k in keys:
            if kinds.get(k, "text") != "numeric":
                work[k] = work[k].where(work[k].notna(), NO_MATCH)
            else:
                missing_key_rows += int(work[k].isna().sum())
        if missing_key_rows:
            work = work.dropna(subset=[k for k in keys if kinds.get(k) == "numeric"])
            notes.append(f"{missing_key_rows} rows had no value in a numeric group column and were left out")
        if keys:
            ops.append({"op": "group_by", "detail": ", ".join(keys)})

        plain = [m for m in plan.metrics if m.agg != "named"]
        named = [m for m in plan.metrics if m.agg == "named"]
        formulas = {}

        spec, size_aliases = {}, []
        for m in plain:
            if m.agg == "count" and m.field is None:
                size_aliases.append(m.as_)
                formulas[m.as_] = "number of rows"
            else:
                spec[m.as_] = (m.field, PLAIN_AGG[m.agg])
                used.add(m.field)
                formulas[m.as_] = PLAIN_WORDS[m.agg].format(f=m.field)
                ignored = int(work[m.field].isna().sum())
                if ignored:
                    notes.append(f"{ignored} rows with a missing or non-numeric '{m.field}' were ignored by {m.as_}"
                                 if kinds[m.field] == "numeric" else f"{ignored} rows with no '{m.field}' were ignored by {m.as_}")
                ops.append({"op": "metric", "detail": f"{m.as_} = {formulas[m.as_]}"})

        grouped_key = keys if keys else ["_all"]
        base = work.assign(_all="all") if not keys else work
        frames = []
        if plain:
            g = base.groupby(grouped_key, dropna=False)
            parts = []
            if spec:
                parts.append(g.agg(**spec))
            for a in size_aliases:
                parts.append(g.size().rename(a))
            frames.append(pd.concat(parts, axis=1).reset_index())
        if named:
            nrows, _ = sales_metrics(work, keys)
            raw_named = pd.DataFrame(nrows)
            ndf = raw_named[keys].copy() if keys else pd.DataFrame(index=raw_named.index)
            for m in named:
                ndf[m.as_] = raw_named[m.name]
                formulas[m.as_] = SALES_FORMULAS[m.name]
                ops.append({"op": "metric", "detail": f"{m.as_} = {SALES_FORMULAS[m.name]}"})
            used.update(["revenue"] + (["returned"] if any("return" in m.name for m in named) else [])
                        + (["quantity"] if any(m.name == "units" for m in named) else []))
            frames.append(ndf)
        if len(frames) == 2:
            out = frames[0].merge(frames[1], on=keys, how="outer") if keys else pd.concat(
                [frames[0].drop(columns=["_all"], errors="ignore"), frames[1]], axis=1)
        else:
            out = frames[0]
        out = out.drop(columns=["_all"], errors="ignore")
        result_cols = keys + [m.as_ for m in plan.metrics]
        out = out[result_cols]
        groups_total = len(out)
        if not keys:
            groups_total = None

    # ---------------------------------------------------------------- list
    elif plan.intent == "list":
        deadline.check("list")
        used.update(plan.select)
        out = df[plan.select]
        if plan.sort:
            used.update(s.field for s in plan.sort)
            order = df.sort_values([s.field for s in plan.sort], ascending=[s.dir == "asc" for s in plan.sort],
                                   na_position="last", kind="mergesort").index
            out = out.loc[order]
        ops.append({"op": "select", "detail": ", ".join(plan.select)})
        groups_total = len(out)
        result_cols = list(plan.select)

    # ------------------------------------------------------------ distinct
    elif plan.intent == "distinct":
        deadline.check("distinct")
        field = plan.select[0]
        used.add(field)
        vc = df[field].dropna().value_counts()
        out = pd.DataFrame({"value": vc.index, "count": vc.values})
        out = out.sort_values(["count", "value"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
        ops.append({"op": "distinct", "detail": f"different values of {field}"})
        if df[field].isna().any():
            notes.append(f"{int(df[field].isna().sum())} rows with no {field} are not counted")
        groups_total = len(out)
        result_cols = ["value", "count"]

    # ------------------------------------------------------------ describe
    else:
        deadline.check("describe")
        field = plan.select[0]
        used.add(field)
        s = df[field].dropna()
        out = pd.DataFrame([{
            "count": len(s), "missing": int(df[field].isna().sum()), "sum": s.sum(), "mean": s.mean(),
            "std": s.std(), "min": s.min(), "p25": s.quantile(0.25), "median": s.median(),
            "p75": s.quantile(0.75), "max": s.max()}])
        ops.append({"op": "describe", "detail": f"descriptive statistics of {field}"})
        result_cols = list(out.columns)
        groups_total = None

    # -------------------------------------------------- sort + limit (shared)
    if plan.intent in ("aggregate", "distinct") and plan.sort:
        out = out.sort_values([s.field for s in plan.sort], ascending=[s.dir == "asc" for s in plan.sort],
                              na_position="last", kind="mergesort")
        ops.append({"op": "sort", "detail": ", ".join(f"{s.field} {s.dir}" for s in plan.sort)})
    elif plan.intent == "list" and plan.sort:
        ops.append({"op": "sort", "detail": ", ".join(f"{s.field} {s.dir}" for s in plan.sort)})
    truncated = plan.intent != "describe" and len(out) > plan.limit
    out = out.head(plan.limit) if plan.intent != "describe" else out
    if plan.intent != "describe":
        ops.append({"op": "limit", "detail": f"first {plan.limit} rows", "rows_after": len(out)})

    result_rows = _rows(out[result_cols] if result_cols else out)
    if any(NO_MATCH in map(str, r.values()) for r in result_rows):
        notes.append("'(no match)' means those rows have no matching record in a joined table; they are kept, not dropped")

    joins = [{"table": r["right"], "key": r["left_key"], "rows_before": r["left_rows"], "rows_after": r["joined_rows"],
              "unmatched_rows": r["orphan_rows"], "blank_key_rows": r["blank_key_rows"]}
             for r in catalog.join_reports.get(ds, [])]
    evidence = {
        "dataset": ds, "version": catalog.version(ds), "stage": catalog.stage,
        "columns_used": sorted(used),
        "filters": [_describe_filter(f) for f in plan.filters],
        "joins": joins,
        "operations": ops,
        "rows_in_dataset": total_rows, "rows_after_filters": after_filters,
        "result_rows": len(result_rows), "result_groups_total": groups_total, "truncated": bool(truncated),
        "preview_limit": plan.limit,
        "formulas": (formulas if plan.intent == "aggregate" else {}),
        "notes": notes,
    }
    return {"columns": result_cols, "rows": result_rows}, evidence
