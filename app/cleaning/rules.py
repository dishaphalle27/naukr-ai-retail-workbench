"""Shared rules. One place, so the detector, the planner and the cleaner agree.

ALLOWED_ACTIONS is an allow-list: the cleaner will refuse any step whose action
is not in this list. (Same idea we will use later for chat query plans.)
"""

# Actions that CHANGE data
MUTATING_ACTIONS = {
    "trim_whitespace",                 # remove edge spaces, collapse double spaces
    "strip_currency_and_commas",       # "Rs. 1,299" -> 1299
    "convert_number_words",            # "two" -> 2
    "convert_date_to_iso",             # any known date shape -> YYYY-MM-DD
    "canonicalize_spelling_variants",  # "west " / "WEST" -> "West"
    "normalize_id_format",             # "sku0007" -> "SKU-0007"
    "lowercase_email",                 # "A@B.COM" -> "a@b.com"
    "drop_exact_duplicates",           # remove copies, keep first
}

# Actions that change NOTHING (they only record a decision)
NON_MUTATING_ACTIONS = {
    "flag_for_review",     # a human must decide
    "leave_missing",       # empty stays empty (we never invent values)
    "treat_as_data",       # suspicious text is kept as plain data, never obeyed
}

ALLOWED_ACTIONS = MUTATING_ACTIONS | NON_MUTATING_ACTIONS

# Order in which steps run. Trim first (so spaces do not hide duplicates),
# remove duplicates last.
PHASES = {
    "headers": 0,
    "whitespace": 1,
    "numbers": 2,
    "dates": 3,
    "variants": 4,
    "ids": 5,
    "duplicates": 6,
    "review": 7,
    "missing": 8,
    "security": 9,
}

RISKS = ("low", "medium", "high")
STATUSES = ("planned", "needs_review", "skipped")   # cleaner later adds: applied / failed

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
