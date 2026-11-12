"""
CLEANING PLAN - the to-do list written BEFORE any data is changed.

Every step has:
  step_id   stable name (same data -> same id)
  action    one of the ALLOWED_ACTIONS (rules.py)
  reason    why, in plain words
  columns   which columns it touches
  risk      low | medium | high
  source    "code" (a fixed rule) or "ai" (a suggestion; none yet)
  status    planned | needs_review | skipped   (the cleaner later adds applied / failed)
  expected_rows_removed  {"min": x, "max": y}  -> the cleaner is checked against this

Safety rules baked in:
  * Missing values are NEVER invented (leave_missing).
  * Columns are NEVER dropped or renamed (schema stays the same).
  * Anything uncertain becomes needs_review and is NOT changed automatically.
"""
import json
import sys

from app.cleaning.issues import detect_issues
from app.cleaning.profiler import profile_file
from app.cleaning.rules import ALLOWED_ACTIONS, NUMBER_WORDS, PHASES
from app.errors import DataLoadError

NO_ROWS_REMOVED = {"min": 0, "max": 0}
SHAPE_SLUG = {"NN/NN/YYYY": "nn-slash", "NN-NN-YYYY": "nn-dash", "DD Mon YYYY": "dd-mon-yyyy"}
HIGH_CONFIDENCE_PROOFS = 10     # at least this many proving dates -> "high" confidence


def _step(dataset, columns, action, phase, reason, risk, issue_ids, status="planned",
          params=None, expected=None, slug=None, confidence=None, review_note=None):
    assert action in ALLOWED_ACTIONS, f"unknown action {action}"
    cols = list(columns)
    sid = f"STEP:{dataset}:{cols[0] if len(cols) == 1 else '_dataset'}:{action}"
    if slug:
        sid += f":{slug}"
    return {
        "step_id": sid, "order": None, "phase": phase, "dataset": dataset,
        "columns": cols, "action": action, "reason": reason, "risk": risk,
        "source": "code", "status": status, "issue_ids": list(issue_ids),
        "params": params or {}, "expected_rows_removed": expected or dict(NO_ROWS_REMOVED),
        "confidence": confidence, "review_note": review_note,
    }


def _date_steps(dataset, name, col, iss):
    """One step per non-ISO date shape. Decide day-first vs month-first from PROOF in the data."""
    steps = []
    evidence = col.get("date_order_evidence", {})
    for shape, count in col["date_formats"].items():
        if shape == "YYYY-MM-DD":
            continue
        ids = [i for i in (iss(name, "DATE_FORMAT"),) if i]
        if shape == "DD Mon YYYY":
            steps.append(_step(dataset, [name], "convert_date_to_iso", "dates",
                               f"{count} dates written like '05 Jan 2026'. The month name makes them unambiguous.",
                               "low", ids, params={"shape": shape, "order": "named_month"}, slug=SHAPE_SLUG[shape]))
            continue
        ev = evidence.get(shape, {"day_first_proof": 0, "month_first_proof": 0, "ambiguous": 0})
        day, month, amb = ev["day_first_proof"], ev["month_first_proof"], ev["ambiguous"]
        if amb and iss(name, "AMBIGUOUS_DATES"):
            ids.append(iss(name, "AMBIGUOUS_DATES"))
        base = {"shape": shape, "proof_day_first": day, "proof_month_first": month, "ambiguous_values": amb}
        if day and not month:
            steps.append(_step(
                dataset, [name], "convert_date_to_iso", "dates",
                f"{count} dates in {shape}. {day} of them can only be day-first (e.g. a day above 12), "
                f"and none can only be month-first, so {amb} ambiguous ones are read day-first too.",
                "medium" if amb else "low", ids, params={**base, "order": "day_first"}, slug=SHAPE_SLUG[shape],
                confidence="high" if day >= HIGH_CONFIDENCE_PROOFS else "medium",
                review_note="Assumption: one date order per format. Ambiguous rows are listed in the change log."
                if amb else None))
        elif month and not day:
            steps.append(_step(
                dataset, [name], "convert_date_to_iso", "dates",
                f"{count} dates in {shape}. {month} of them can only be month-first, "
                f"and none can only be day-first, so {amb} ambiguous ones are read month-first too.",
                "medium" if amb else "low", ids, params={**base, "order": "month_first"}, slug=SHAPE_SLUG[shape],
                confidence="high" if month >= HIGH_CONFIDENCE_PROOFS else "medium",
                review_note="Assumption: one date order per format. Ambiguous rows are listed in the change log."
                if amb else None))
        else:
            why = ("both day-first and month-first proofs exist" if day and month
                   else "no value proves the order")
            steps.append(_step(
                dataset, [name], "flag_for_review", "dates",
                f"{count} dates in {shape} cannot be converted safely: {why}.",
                "high", ids, status="needs_review", params={**base, "order": "unknown"}, slug=SHAPE_SLUG[shape],
                review_note="Left unchanged. A person must say which order this format uses."))
    return steps


def build_plan(dataset, profile, issues=None):
    issues = issues if issues is not None else detect_issues(dataset, profile)
    lookup = {(i["column"], i["type"]): i["id"] for i in issues}

    def iss(column, itype):
        return lookup.get((column, itype))

    def ids(*pairs):
        return [x for x in (iss(c, t) for c, t in pairs) if x]

    steps = []
    d = profile["duplicates"]

    if profile["row_count"] == 0:
        return steps

    for h in profile["header_issues"]:
        steps.append(_step(dataset, [h], "flag_for_review", "headers",
                           "Column name looks wrong, but renaming would change the schema.", "medium",
                           ids((h, "HEADER_ISSUE")), status="needs_review",
                           review_note="Left unchanged."))

    for name, col in profile["columns"].items():
        kind = col["kind"]

        if col["all_null"]:
            steps.append(_step(dataset, [name], "flag_for_review", "review",
                               "Column is completely empty. Dropping it would change the schema.",
                               "medium", ids((name, "ALL_NULL_COLUMN")), status="needs_review",
                               review_note="Column kept as is."))
        elif col["null_count"]:
            steps.append(_step(dataset, [name], "leave_missing", "missing",
                               f"{col['null_count']} empty values. There is no safe way to invent them, so they stay empty.",
                               "low", ids((name, "MISSING_VALUES")), status="skipped"))

        if iss(name, "WHITESPACE"):
            steps.append(_step(dataset, [name], "trim_whitespace", "whitespace",
                               "Remove spaces at the start and end, and collapse double spaces.",
                               "low", ids((name, "WHITESPACE"))))

        if kind == "numeric":
            if col.get("numeric_with_symbols"):
                steps.append(_step(dataset, [name], "strip_currency_and_commas", "numbers",
                                   f"{col['numeric_with_symbols']} numbers have 'Rs.', a currency sign or commas. "
                                   "Only the symbols are removed; the number is unchanged.",
                                   "low", ids((name, "NUMBERS_AS_TEXT"))))
            if col.get("parse_failures"):
                examples = col.get("parse_failure_examples", [])
                known = sorted({e.lower() for e in examples if e.strip().lower() in NUMBER_WORDS})
                unknown = [e for e in examples if e.strip().lower() not in NUMBER_WORDS]
                if known:
                    steps.append(_step(dataset, [name], "convert_number_words", "numbers",
                                       f"Numbers written as words ({', '.join(known)}) become digits.",
                                       "low", ids((name, "NON_NUMERIC_IN_NUMERIC")),
                                       params={"words": known}))
                if unknown or not known:
                    steps.append(_step(dataset, [name], "flag_for_review", "review",
                                       "Some non-numeric values in a number column are not known number words.",
                                       "medium", ids((name, "NON_NUMERIC_IN_NUMERIC")), status="needs_review",
                                       review_note="Left unchanged."))
            if col.get("stats", {}).get("negatives"):
                steps.append(_step(dataset, [name], "flag_for_review", "review",
                                   f"{col['stats']['negatives']} negative values. They could be real (backorders, "
                                   "returns) or errors, so we do not change them.",
                                   "medium", ids((name, "NEGATIVE_VALUES")), status="needs_review",
                                   review_note="Left unchanged."))

        if kind == "date":
            steps.extend(_date_steps(dataset, name, col, iss))
            if col.get("unparsable_count"):
                steps.append(_step(dataset, [name], "flag_for_review", "review",
                                   f"{col['unparsable_count']} values are not real dates.",
                                   "high", ids((name, "UNPARSABLE_DATES")), status="needs_review",
                                   review_note="Left unchanged."))

        if kind == "text":
            if col.get("spelling_variants"):
                safe = all(len({v.strip().lower() for v in g}) == 1 for g in col["spelling_variants"])
                steps.append(_step(dataset, [name], "canonicalize_spelling_variants", "variants",
                                   "Different spellings of the same value are merged into the most common one "
                                   "(ties: Title Case, then alphabetical).",
                                   "low" if safe else "medium", ids((name, "SPELLING_VARIANTS")),
                                   params={"rule": "most_frequent_spelling"}))
            if col.get("format_patterns"):
                dominant = max(col["format_patterns"], key=col["format_patterns"].get)
                steps.append(_step(dataset, [name], "normalize_id_format", "ids",
                                   f"Codes use {len(col['format_patterns'])} formats. All are rewritten to the "
                                   f"most common one ({dominant}).",
                                   "medium", ids((name, "MIXED_ID_FORMAT")),
                                   params={"dominant_pattern": dominant}))
            if col.get("values_with_uppercase"):
                steps.append(_step(dataset, [name], "lowercase_email", "variants",
                                   "Email addresses are written in lowercase.", "low", ids((name, "EMAIL_CASE"))))
            if col.get("injection_like_values"):
                steps.append(_step(dataset, [name], "treat_as_data", "security",
                                   f"{col['injection_like_values']} value(s) look like instructions to an AI. "
                                   "They are kept as plain text and never obeyed.",
                                   "high", ids((name, "INJECTION_TEXT")), status="skipped",
                                   review_note="Excluded from any text sent to a model."))

    key = d["key"]
    if key and key["conflicting_key_rows"]:
        steps.append(_step(dataset, [key["column"]], "flag_for_review", "review",
                           f"{key['conflicting_key_rows']} rows share a {key['column']} but have different content.",
                           "high", ids((key["column"], "CONFLICTING_KEY")), status="needs_review",
                           review_note="Not removed. A person must choose which row is right."))

    exact, near = d["exact_duplicate_rows"], d["extra_after_trim_and_lowercase"]
    if exact or near:
        steps.append(_step(dataset, ["*"], "drop_exact_duplicates", "duplicates",
                           (f"Remove {exact} exact copies (the first copy is kept). "
                            + (f"Up to {near} more rows may match after spaces are trimmed." if near else "")).strip(),
                           "medium" if near else "low",
                           ids((None, "DUPLICATE_ROWS"), (None, "NEAR_DUPLICATE_ROWS")),
                           expected={"min": exact, "max": exact + near}))

    order = {c: i for i, c in enumerate(profile["column_names"])}
    steps.sort(key=lambda s: (PHASES[s["phase"]], order.get(s["columns"][0], 999), s["step_id"]))
    for i, s in enumerate(steps, start=1):
        s["order"] = i
    return steps


def format_plan(plan):
    lines = []
    for s in plan:
        lines.append(f"{s['order']:>2}. [{s['status']:<12}] risk={s['risk']:<6} {s['action']:<31} "
                     f"{','.join(s['columns'])}")
        lines.append(f"      {s['reason']}")
        if s["review_note"]:
            lines.append(f"      note: {s['review_note']}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.cleaning.plan <file.csv> [--json]")
        sys.exit(1)
    try:
        prof = profile_file(sys.argv[1])
    except DataLoadError as err:
        print(json.dumps(err.to_dict(), indent=2))
        sys.exit(2)
    name = prof["meta"]["file_name"].rsplit(".", 1)[0]
    found = detect_issues(name, prof)
    result = build_plan(name, prof, found)
    if "--json" in sys.argv:
        print(json.dumps({"issues": found, "cleaning_plan": result}, indent=2))
    else:
        print(f"{name}: {len(found)} issues -> {len(result)} planned steps\n")
        print(format_plan(result))
