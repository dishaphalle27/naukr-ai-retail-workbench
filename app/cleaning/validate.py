"""
VALIDATION - checks the cleaner's own work. We never just trust it.

Checks
  schema_unchanged     same columns, same order
  rows_as_planned      rows removed is inside the range the plan promised
  rows_are_survivors   every output row came from an input row (nothing invented, same order)
  untouched_columns    columns no applied step mentioned are identical to the input
  no_failed_steps      no step crashed
  no_unexpected_issues nothing is left that a changing step should have fixed
  idempotent           cleaning the cleaned data again changes nothing
"""


def validate(before, after, plan, log, issues_after, idempotent):
    applied_ids = {e["step_id"] for e in log if e["status"] == "applied"}
    failed = [e["step_id"] for e in log if e["status"] == "failed"]

    # rows the plan allows us to remove (only from steps that were really applied)
    lo = sum(s["expected_rows_removed"]["min"] for s in plan if s["step_id"] in applied_ids)
    hi = sum(s["expected_rows_removed"]["max"] for s in plan if s["step_id"] in applied_ids)
    removed = len(before) - len(after)

    touched = {c for s in plan if s["step_id"] in applied_ids for c in s["columns"] if c != "*"}
    survivors = before.loc[after.index] if after.index.isin(before.index).all() else None
    untouched = [c for c in before.columns if c not in touched]
    untouched_ok = bool(survivors is not None and all(
        c in after.columns and survivors[c].equals(after[c]) for c in untouched))

    # issues still present that no "needs_review"/"skipped" step explains
    accepted = {i for s in plan if s["status"] in ("needs_review", "skipped") for i in s["issue_ids"]}
    unexpected = [i["id"] for i in issues_after if i["id"] not in accepted]
    left_for_people = sorted({i["id"] for i in issues_after if i["id"] in accepted})

    checks = {
        "schema_unchanged": list(before.columns) == list(after.columns),
        "rows_as_planned": lo <= removed <= hi,
        "rows_are_survivors": bool(after.index.isin(before.index).all() and after.index.is_monotonic_increasing),
        "untouched_columns_identical": untouched_ok,
        "no_failed_steps": not failed,
        "no_unexpected_issues": not unexpected,
        "idempotent": bool(idempotent),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "rows": {"before": len(before), "after": len(after), "removed": removed,
                 "allowed_removed_min": lo, "allowed_removed_max": hi},
        "columns": {"before": len(before.columns), "after": len(after.columns)},
        "failed_steps": failed,
        "unexpected_issue_ids": unexpected,
        "left_for_review": left_for_people,
    }
