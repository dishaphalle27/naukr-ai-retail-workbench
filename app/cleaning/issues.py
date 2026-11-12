"""
ISSUE DETECTOR - reads a profile and lists the problems found.

Input : the profile (facts) from profiler.py
Output: a list of issues. Each issue has a STABLE id, so the same data always
        gives the same ids (important for reproducible evaluation).

Only counts and safe evidence are used. No cell values are copied, so nothing
private or prompt-injection-like can leak from here.
"""

def _issue(dataset, column, itype, severity, count, message, evidence=None):
    return {
        "id": f"ISS:{dataset}:{column or '_dataset'}:{itype}",
        "dataset": dataset,
        "column": column,
        "type": itype,
        "severity": severity,            # low | medium | high
        "affected_count": int(count),
        "message": message,
        "evidence": evidence or {},
    }


def _only_punctuation_or_case_differs(group):
    """True if variants differ only by case/spaces (very safe to merge)."""
    return len({v.strip().lower() for v in group}) == 1


def detect_issues(dataset, profile):
    issues = []
    add = issues.append
    d = profile["duplicates"]

    if profile["row_count"] == 0:
        add(_issue(dataset, None, "EMPTY_DATASET", "high", 0, "File has a header but no data rows."))
        return issues

    for h in profile["header_issues"]:
        add(_issue(dataset, h, "HEADER_ISSUE", "medium", 1,
                   "Column name has stray spaces or no name. Renaming would change the schema."))

    if d["exact_duplicate_rows"]:
        add(_issue(dataset, None, "DUPLICATE_ROWS", "medium", d["exact_duplicate_rows"],
                   f"{d['exact_duplicate_rows']} rows are exact copies of another row."))
    if d["extra_after_trim_and_lowercase"]:
        add(_issue(dataset, None, "NEAR_DUPLICATE_ROWS", "medium", d["extra_after_trim_and_lowercase"],
                   "Rows that become copies after removing spaces and ignoring upper/lower case."))
    key = d["key"]
    if key and key["conflicting_key_rows"]:
        add(_issue(dataset, key["column"], "CONFLICTING_KEY", "high", key["conflicting_key_rows"],
                   f"Same {key['column']} appears with different content. Cannot safely choose one."))

    for name, col in profile["columns"].items():
        n = col["null_count"]
        if col["all_null"]:
            add(_issue(dataset, name, "ALL_NULL_COLUMN", "medium", n, "Every value in this column is empty."))
        elif n:
            sev = "medium" if col["null_pct"] >= 20 else "low"
            add(_issue(dataset, name, "MISSING_VALUES", sev, n,
                       f"{n} empty values ({col['null_pct']}%).", {"null_pct": col["null_pct"]}))

        edge, dbl = col["values_with_edge_spaces"], col["values_with_double_spaces"]
        if edge or dbl:
            add(_issue(dataset, name, "WHITESPACE", "low", max(edge, dbl),
                       "Extra spaces at the edges or inside values.", {"edge": edge, "double": dbl}))

        kind = col["kind"]
        if kind == "date":
            formats = col["date_formats"]
            non_iso = sum(v for k, v in formats.items() if k != "YYYY-MM-DD")
            if non_iso:
                add(_issue(dataset, name, "DATE_FORMAT", "medium", non_iso,
                           "Dates are not all written as YYYY-MM-DD.", {"formats": formats}))
            if col.get("ambiguous_day_month"):
                add(_issue(dataset, name, "AMBIGUOUS_DATES", "medium", col["ambiguous_day_month"],
                           "Some dates can be read as day/month or month/day.",
                           {"order_evidence": col.get("date_order_evidence", {})}))
            if col.get("unparsable_count"):
                add(_issue(dataset, name, "UNPARSABLE_DATES", "high", col["unparsable_count"],
                           "Some values are not real dates in any known format."))
        elif kind == "numeric":
            if col.get("numeric_with_symbols"):
                add(_issue(dataset, name, "NUMBERS_AS_TEXT", "low", col["numeric_with_symbols"],
                           "Numbers written with currency signs or commas."))
            if col.get("parse_failures"):
                add(_issue(dataset, name, "NON_NUMERIC_IN_NUMERIC", "medium", col["parse_failures"],
                           "Some values in a number column are not numbers.",
                           {"examples": col["parse_failure_examples"]}))
            if col.get("stats", {}).get("negatives"):
                add(_issue(dataset, name, "NEGATIVE_VALUES", "medium", col["stats"]["negatives"],
                           "Negative numbers found. They may be real (returns, backorders) or errors."))
        elif kind == "text":
            if col.get("injection_like_values"):
                add(_issue(dataset, name, "INJECTION_TEXT", "high", col["injection_like_values"],
                           "Text that looks like instructions to an AI. It will be kept as plain data."))
            variants = col.get("spelling_variants")
            if variants:
                safe = all(_only_punctuation_or_case_differs(g) for g in variants)
                add(_issue(dataset, name, "SPELLING_VARIANTS", "low" if safe else "medium", len(variants),
                           "Same value written in different ways (case, spaces or punctuation).",
                           {"groups": variants, "case_or_space_only": safe}))
            if col.get("format_patterns"):
                add(_issue(dataset, name, "MIXED_ID_FORMAT", "low", sum(col["format_patterns"].values()),
                           "Codes written in more than one format.", {"patterns": col["format_patterns"]}))
            if col.get("values_with_uppercase"):
                add(_issue(dataset, name, "EMAIL_CASE", "low", col["values_with_uppercase"],
                           "Email addresses with capital letters."))

    issues.sort(key=lambda i: (i["column"] or "", i["type"]))
    return issues
