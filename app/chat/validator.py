"""
PLAN VALIDATOR - the gate. Nothing runs until a plan passes every check here.

It treats the plan as UNTRUSTED input (it may come from a model that was tricked).
It collects ALL problems, not just the first, so the reason for a rejection is clear.

Each problem: {"code": ..., "path": ..., "message": ...}. Main codes:
  UNKNOWN_FIELD      a key that is not part of the plan format (e.g. "sql", "join")
  INVALID_VALUE      wrong type or a value not on the allow-list
  UNKNOWN_DATASET    dataset is not in the catalog
  UNKNOWN_COLUMN     column does not exist (with "did you mean")
  COLUMN_NOT_AVAILABLE  column exists but is hidden (personal data)
  BAD_OPERATOR       operator not valid for that column type
  LIMIT_TOO_LARGE / TOO_MANY_*   hard limits
  GROUPS_TOO_LARGE   the result would have too many groups (expensive)
  MISSING_REQUIRED   the intent needs something that is missing
"""
import difflib
import re
from datetime import date

from pydantic import ValidationError

from app.chat.plan_schema import (
    MAX_FILTERS, MAX_GROUP_BY, MAX_GROUPS, MAX_IN_VALUES, MAX_LIMIT, MAX_METRICS, MAX_SELECT, MAX_SORT,
    MAX_TEXT_VALUE_LEN, NAMED_METRICS, Plan,
)
from app.errors import DataLoadError

ALIAS_RE = re.compile(r"^[a-z_][a-z0-9_]{0,30}$")
NUMERIC_OPS = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "between", "is_null", "not_null"}
DATE_OPS = {"eq", "ne", "gt", "gte", "lt", "lte", "between", "is_null", "not_null"}
TEXT_OPS = {"eq", "ne", "in", "not_in", "contains", "is_null", "not_null"}
OPS_BY_KIND = {"numeric": NUMERIC_OPS, "date": DATE_OPS, "text": TEXT_OPS}
AGG_KINDS = {"sum": {"numeric"}, "mean": {"numeric"}, "median": {"numeric"},
             "min": {"numeric", "date"}, "max": {"numeric", "date"}}


class PlanRejected(DataLoadError):
    def __init__(self, problems):
        first = problems[0]["message"] if problems else "plan rejected"
        super().__init__("PLAN_REJECTED", f"Plan rejected: {first}", {"problems": problems})
        self.problems = problems


def _p(code, path, message):
    return {"code": code, "path": path, "message": message}


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_iso_date(v):
    if not isinstance(v, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
        return False
    try:
        date.fromisoformat(v)
        return True
    except ValueError:
        return False


def _suggest(name, options):
    return difflib.get_close_matches(str(name), list(options), n=3, cutoff=0.6)


def _schema_problems(exc):
    out = []
    for e in exc.errors():
        path = ".".join(str(x) for x in e["loc"])
        if e["type"] == "extra_forbidden":
            out.append(_p("UNKNOWN_FIELD", path, f"'{path}' is not allowed in a plan"))
        elif e["type"] == "literal_error":
            allowed = e.get("ctx", {}).get("expected", "")
            out.append(_p("INVALID_VALUE", path, f"'{path}' must be one of {allowed}"))
        else:
            out.append(_p("INVALID_VALUE", path, f"'{path}': {e['msg']}"))
    return out


def validate_plan(raw, catalog):
    """Returns (plan, warnings). Raises PlanRejected with every problem found."""
    if not isinstance(raw, dict):
        raise PlanRejected([_p("INVALID_VALUE", "", "a plan must be a JSON object")])
    try:
        plan = Plan.model_validate(raw)
    except ValidationError as exc:
        raise PlanRejected(_schema_problems(exc))

    problems, warnings = [], []
    if plan.dataset not in catalog.datasets:
        raise PlanRejected([_p("UNKNOWN_DATASET", "dataset", f"dataset '{plan.dataset}' does not exist. "
                               f"Available: {', '.join(catalog.datasets)}")])
    ds = plan.dataset
    allcols = catalog.columns(ds)

    def col(name, path):
        """Return the column info, or record a problem and return None."""
        if name not in allcols:
            near = _suggest(name, catalog.visible_columns(ds))
            hint = f" Did you mean: {', '.join(near)}?" if near else ""
            problems.append(_p("UNKNOWN_COLUMN", path, f"column '{name}' does not exist in {ds}.{hint}"))
            return None
        if allcols[name]["hidden"]:
            problems.append(_p("COLUMN_NOT_AVAILABLE", path,
                               f"column '{name}' holds personal data and is not available for questions"))
            return None
        return allcols[name]

    # ------------------------------------------------------------ hard limits
    if plan.limit < 1:
        problems.append(_p("INVALID_VALUE", "limit", "limit must be at least 1"))
    if plan.limit > MAX_LIMIT:
        problems.append(_p("LIMIT_TOO_LARGE", "limit", f"limit {plan.limit} is above the maximum of {MAX_LIMIT}"))
    for name, items, cap in (("filters", plan.filters, MAX_FILTERS), ("group_by", plan.group_by, MAX_GROUP_BY),
                             ("metrics", plan.metrics, MAX_METRICS), ("sort", plan.sort, MAX_SORT),
                             ("select", plan.select, MAX_SELECT)):
        if len(items) > cap:
            problems.append(_p(f"TOO_MANY_{name.upper()}", name, f"at most {cap} {name} are allowed"))

    # ---------------------------------------------------------------- filters
    for i, f in enumerate(plan.filters[:MAX_FILTERS]):
        path = f"filters.{i}"
        info = col(f.field, f"{path}.field")
        if not info:
            continue
        kind = info["kind"]
        if f.op not in OPS_BY_KIND[kind]:
            problems.append(_p("BAD_OPERATOR", f"{path}.op",
                               f"operator '{f.op}' cannot be used on {kind} column '{f.field}'"))
            continue
        v = f.value
        if f.op in ("is_null", "not_null"):
            if v is not None:
                problems.append(_p("INVALID_VALUE", f"{path}.value", f"'{f.op}' takes no value"))
            continue
        values = v if f.op in ("in", "not_in", "between") else [v]
        if f.op in ("in", "not_in", "between") and not isinstance(v, list):
            problems.append(_p("INVALID_VALUE", f"{path}.value", f"'{f.op}' needs a list of values"))
            continue
        if f.op == "between" and len(values) != 2:
            problems.append(_p("INVALID_VALUE", f"{path}.value", "'between' needs exactly 2 values"))
            continue
        if f.op in ("in", "not_in") and not (1 <= len(values) <= MAX_IN_VALUES):
            problems.append(_p("INVALID_VALUE", f"{path}.value", f"'{f.op}' needs 1 to {MAX_IN_VALUES} values"))
            continue
        ok = True
        for x in values:
            if kind == "numeric" and not _is_number(x):
                ok = False
            elif kind == "date" and not _is_iso_date(x):
                ok = False
            elif kind == "text" and not (isinstance(x, str) and 0 < len(x.strip()) <= MAX_TEXT_VALUE_LEN):
                ok = False
        if not ok:
            problems.append(_p("INVALID_VALUE", f"{path}.value",
                               f"value for {kind} column '{f.field}' must be "
                               f"{'a number' if kind == 'numeric' else 'a date like 2026-03-31' if kind == 'date' else 'a short text'}"))
            continue
        if kind == "text" and f.op in ("eq", "in", "ne", "not_in"):
            known = catalog.distinct_values(ds, f.field)
            if known is not None:
                lowered = {k.lower() for k in known}
                for x in values:
                    if x.strip().lower() not in lowered:
                        warnings.append({"code": "VALUE_NOT_FOUND", "path": f"{path}.value", "field": f.field,
                                         "value": x, "did_you_mean": _suggest(x, known)})

    # ------------------------------------------------------------- group_by
    group_kinds = {}
    for i, g in enumerate(plan.group_by[:MAX_GROUP_BY]):
        info = col(g, f"group_by.{i}")
        if info:
            group_kinds[g] = info["kind"]
    if plan.time_bucket:
        info = col(plan.time_bucket.field, "time_bucket.field")
        if info and info["kind"] != "date":
            problems.append(_p("BAD_OPERATOR", "time_bucket.field",
                               f"time_bucket needs a date column, '{plan.time_bucket.field}' is {info['kind']}"))
    if plan.intent != "aggregate" and (plan.group_by or plan.time_bucket):
        problems.append(_p("INVALID_VALUE", "group_by", "group_by and time_bucket are only for intent 'aggregate'"))

    # -------------------------------------------------------------- metrics
    aliases = []
    if plan.intent == "aggregate":
        if not plan.metrics:
            problems.append(_p("MISSING_REQUIRED", "metrics", "an aggregate question needs at least one metric"))
        for i, m in enumerate(plan.metrics[:MAX_METRICS]):
            path = f"metrics.{i}"
            if m.agg == "named":
                if m.name not in NAMED_METRICS:
                    problems.append(_p("INVALID_VALUE", f"{path}.name",
                                       f"named metric must be one of {', '.join(NAMED_METRICS)}"))
                else:
                    missing = [c for c in NAMED_METRICS[m.name] if c not in allcols]
                    if missing:
                        problems.append(_p("UNKNOWN_COLUMN", path,
                                           f"metric '{m.name}' needs column(s) {', '.join(missing)} which {ds} does not have"))
                if m.field is not None:
                    problems.append(_p("INVALID_VALUE", f"{path}.field", "named metrics take no field"))
                m.as_ = m.as_ or (m.name or "metric")
            else:
                if m.name is not None:
                    problems.append(_p("INVALID_VALUE", f"{path}.name", "'name' is only for agg 'named'"))
                if m.agg == "count" and m.field is None:
                    m.as_ = m.as_ or "row_count"
                else:
                    if m.field is None:
                        problems.append(_p("MISSING_REQUIRED", f"{path}.field", f"'{m.agg}' needs a field"))
                    else:
                        info = col(m.field, f"{path}.field")
                        if info and m.agg in AGG_KINDS and info["kind"] not in AGG_KINDS[m.agg]:
                            problems.append(_p("BAD_OPERATOR", f"{path}.agg",
                                               f"'{m.agg}' cannot be used on {info['kind']} column '{m.field}'"))
                    m.as_ = m.as_ or f"{m.agg}_{m.field}"
            if m.as_ and not ALIAS_RE.match(m.as_):
                problems.append(_p("INVALID_VALUE", f"{path}.as",
                                   "'as' must be lowercase letters, digits or underscore (max 31 characters)"))
            aliases.append(m.as_)
        reserved = set(plan.group_by) | ({f"{plan.time_bucket.field}_{plan.time_bucket.unit}"} if plan.time_bucket else set())
        if len(set(aliases)) != len(aliases) or (set(aliases) & reserved):
            problems.append(_p("INVALID_VALUE", "metrics", "metric names ('as') must be unique and not equal a group column"))

        # expensive-query guard: estimate how many groups the answer would have
        est = 1
        for g in plan.group_by:
            if g in allcols and not allcols[g]["hidden"]:
                est *= max(1, catalog.nunique(ds, g))
        if est > MAX_GROUPS:
            problems.append(_p("GROUPS_TOO_LARGE", "group_by",
                               f"this grouping could create about {est} groups; the maximum is {MAX_GROUPS}"))
    elif plan.metrics:
        problems.append(_p("INVALID_VALUE", "metrics", f"metrics are only for intent 'aggregate', not '{plan.intent}'"))

    # ---------------------------------------------------- other intents + sort
    if plan.intent == "list":
        if not plan.select:
            problems.append(_p("MISSING_REQUIRED", "select", "a list question needs 'select' (which columns to show)"))
        for i, c in enumerate(plan.select[:MAX_SELECT]):
            col(c, f"select.{i}")
        out_cols = set(plan.select) | set(catalog.visible_columns(ds))
    elif plan.intent == "distinct":
        if len(plan.select) != 1:
            problems.append(_p("MISSING_REQUIRED", "select", "a distinct question needs exactly one column in 'select'"))
        for i, c in enumerate(plan.select[:1]):
            col(c, f"select.{i}")
        out_cols = {"value", "count"}
    elif plan.intent == "describe":
        if len(plan.select) != 1:
            problems.append(_p("MISSING_REQUIRED", "select", "a describe question needs exactly one numeric column in 'select'"))
        for i, c in enumerate(plan.select[:1]):
            info = col(c, f"select.{i}")
            if info and info["kind"] != "numeric":
                problems.append(_p("BAD_OPERATOR", f"select.{i}", f"describe needs a numeric column, '{c}' is {info['kind']}"))
        out_cols = set()
    else:
        out_cols = set(plan.group_by) | set(a for a in aliases if a)
        if plan.time_bucket:
            out_cols.add(f"{plan.time_bucket.field}_{plan.time_bucket.unit}")
    if plan.intent == "describe" and plan.sort:
        problems.append(_p("INVALID_VALUE", "sort", "describe returns one row; sorting is not allowed"))
    for i, s in enumerate(plan.sort[:MAX_SORT]):
        if s.field not in out_cols:
            problems.append(_p("UNKNOWN_COLUMN", f"sort.{i}.field",
                               f"cannot sort by '{s.field}'. Sortable now: {', '.join(sorted(out_cols)) or 'nothing'}"))
        elif s.field in allcols and allcols[s.field]["hidden"]:
            problems.append(_p("COLUMN_NOT_AVAILABLE", f"sort.{i}.field", f"column '{s.field}' is not available"))

    if problems:
        raise PlanRejected(problems)
    return plan, warnings
