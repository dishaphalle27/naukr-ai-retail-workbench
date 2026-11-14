"""
ANSWER BUILDER - turns an EXECUTED result into readable text.

Rule: every number in the text is read from result["rows"] and only formatted here
(rupee signs, % and Indian digit grouping). Nothing is calculated, guessed or remembered.
Because of this, a later AI can rephrase the text but can never change a number:
the numbers always travel next to it in answer["rows"].
"""
import re


def inr(x, decimals=0):
    """Indian digit grouping: 4602503 -> 46,02,503"""
    neg = x < 0
    s = f"{abs(x):.{decimals}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        whole = head + "," + tail
    return ("-" if neg else "") + whole + ("." + frac if frac else "")


def fmt_value(v, unit=None):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    if unit == "percent":
        return f"{v:.1f}%"
    if unit == "money":
        return "₹" + inr(v, 0)
    if float(v).is_integer():
        return inr(int(v))
    text = inr(v, 2)
    return text.rstrip("0").rstrip(".") if "." in text else text


def filter_text(f):
    v = f["value"]
    if f["op"] == "between":
        return f"{f['field']} from {v[0]} to {v[1]}"
    if f["op"] in ("is_null", "not_null"):
        return f"{f['field']} {'is empty' if f['op'] == 'is_null' else 'is filled in'}"
    words = {"eq": "=", "ne": "is not", "gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "in": "is one of",
             "not_in": "is none of", "contains": "contains"}
    shown = ", ".join(map(str, v)) if isinstance(v, list) else (int(v) if isinstance(v, float) and v.is_integer() else v)
    return f"{f['field']} {words.get(f['op'], f['op'])} {shown}"


def _dim_name(col):
    for unit in ("month", "year"):
        if col.endswith(f"_{unit}"):
            return unit
    return col.replace("_", " ")


def build_answer(plan, result, evidence, meta, warnings):
    intent, rows, cols = plan["intent"], result["rows"], result["columns"]
    labels, units = meta.get("labels", {}), meta.get("units", {})
    unit_cols = meta.get("units_by_column", {})
    lines, headline = [], ""

    if intent == "aggregate":
        aliases = [m["as"] for m in plan["metrics"]]
        keys = [c for c in cols if c not in aliases]

        def show(row, only=None):
            use = only or aliases
            if len(use) == 1:                       # one measure: the heading already names it
                return fmt_value(row.get(use[0]), units.get(use[0]))
            return ", ".join(f"{labels.get(a, a)} {fmt_value(row.get(a), units.get(a))}" for a in use)

        if not rows:
            headline = "No rows matched."
        elif not keys:
            headline = "; ".join(f"{labels.get(a, a)}: {fmt_value(rows[0].get(a), units.get(a))}" for a in aliases)
        else:
            dim_text = " and ".join(_dim_name(k) for k in keys)
            ranking = meta.get("ranking")
            first = rows[0]
            who = " / ".join(str(first[k]) for k in keys)
            if ranking and plan.get("limit") == 1:
                headline = f"{ranking.capitalize()} {labels.get(aliases[0], aliases[0]).lower()}: {who} ({show(first)})"
            elif ranking:
                headline = f"{ranking.capitalize()} {labels.get(aliases[0], aliases[0]).lower()} by {dim_text}:"
            else:
                headline = f"{labels.get(aliases[0], aliases[0])} by {dim_text}:"
            if not (ranking and plan.get("limit") == 1) or len(rows) > 1:
                for r in rows[:15]:
                    lines.append(f"  - {' / '.join(str(r[k]) for k in keys)}: {show(r)}")
    elif intent == "distinct":
        total = evidence.get("result_groups_total") or len(rows)
        headline = f"{total} different value(s) of {plan['select'][0]}:"
        lines = [f"  - {r['value']} ({fmt_value(r['count'])} rows)" for r in rows[:30]]
    elif intent == "describe":
        r = rows[0] if rows else {}
        f = plan["select"][0]
        u = unit_cols.get(f)
        headline = (f"{f}: {fmt_value(r.get('count'))} values, average {fmt_value(r.get('mean'), u)}, "
                    f"median {fmt_value(r.get('median'), u)}, from {fmt_value(r.get('min'), u)} to {fmt_value(r.get('max'), u)}")
        lines = [f"  - {k}: {fmt_value(r.get(k), u if k not in ('count', 'missing') else None)}"
                 for k in ("count", "missing", "sum", "mean", "std", "min", "p25", "median", "p75", "max")]
    else:  # list
        headline = f"{len(rows)} row(s) shown" + (f" (more exist, limit {plan.get('limit')})" if evidence.get("truncated") else "") + ":"
        for r in rows[:30]:
            lines.append("  - " + ", ".join(f"{k} {fmt_value(v, unit_cols.get(k))}" for k, v in r.items()))

    extra = []
    if plan.get("filters"):
        extra.append("Scope: " + "; ".join(filter_text(f) for f in plan["filters"]))
    for w in warnings or []:
        hint = f" Did you mean {', '.join(w['did_you_mean'])}?" if w.get("did_you_mean") else ""
        extra.append(f"Note: '{w['value']}' does not exist in {w['field']}.{hint}")
    extra += meta.get("notes", []) + evidence.get("notes", [])
    extra.append(f"Based on {fmt_value(evidence['rows_after_filters'])} of {fmt_value(evidence['rows_in_dataset'])} rows "
                 f"in {evidence['dataset']} ({evidence['version']}).")
    text = "\n".join([headline] + lines + extra)
    return {"text": text, "headline": headline, "columns": cols, "rows": rows, "assumptions": meta.get("notes", [])}
