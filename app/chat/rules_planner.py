"""
RULE-BASED PLANNER - turns an English question into a query plan WITHOUT any AI.

Why it exists
  * The app works with no API key and no internet (a reviewer can run everything).
  * If the AI is down or returns broken JSON, this is the fallback.
  * It is deliberately limited: it understands a defined set of question shapes and
    says so when it does not understand (it never guesses a column).

What parse_question() returns - one of:
  {"kind": "plan",     "plan": {...}, "meta": {...}}          run this plan
  {"kind": "clarify",  "question": "...", "options": [...]}   ask the user first
  {"kind": "refuse",   "code": "...", "message": "...", "suggestions": [...]}
  {"kind": "command",  "command": "clear_filters"}
  {"kind": "insight",  "insight": {...}}                       product health questions

Order of checks (the first that matches wins)
  1. prompt-injection / personal-data requests  -> refuse
  2. product-health questions                    -> insight
  3. things the data does not contain            -> refuse
  4. filters (values, numbers, dates), measure, "by <dimension>", top/bottom
  5. follow-ups reuse the previous filters / measure / grouping
"""
import calendar
import difflib
import re

from app.chat.answers import filter_text as _fmt_filter

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10}
NUM_ALT = "|".join(NUM_WORDS)
MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
               "october", "november", "december"]
MONTHS = {n: i for i, n in enumerate(MONTH_NAMES, 1)}
MONTHS.update({n[:3]: i for i, n in enumerate(MONTH_NAMES, 1)})
MONTHS["sept"] = 9
MONTH_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))

EXAMPLES = [
    "What is the total revenue?", "Revenue by category", "Top 5 stores by revenue",
    "Which region has the highest return rate?", "How many orders were there in March?",
    "Compare West and East revenue", "Monthly revenue trend", "Which products need attention?",
]

# ------------------------------------------------------------------ safety screens
INJECTION_RE = re.compile(
    r"ignore (?:all |any )?(?:the )?(?:previous|prior|above|earlier)|disregard .{0,30}instructions|system prompt"
    r"|your (?:hidden |secret )?instructions|reveal .{0,30}(?:prompt|secret|key|password)|api[ _-]?key"
    r"|jailbreak|developer mode|pretend (?:you are|to be)|\bact as\b|override (?:the )?(?:rules|safety)")
PII_RE = re.compile(r"\b(?:e-?mails?|phone|mobile|addresses|address|customer names?|names? of (?:the )?customers?"
                    r"|who (?:is|are) (?:the )?customers?|contact details|passwords?)\b")

UNANSWERABLE = [
    ("NO_PROFIT_DATA", r"\bprofit\w*",
     "There is no profit figure in the data. Orders record revenue only, and cost is stored per product, not per order.",
     ["Revenue by category", "Average price by category", "Average cost by category"]),
    ("NO_MARGIN_COLUMN", r"\bmargins?\b",
     "Margin is not a column I can group or filter by. It is only used inside the product health score.",
     ["Which products need attention?", "Average price by category", "Average cost by category"]),
    ("NO_TRAFFIC_DATA", r"\b(?:conversion|footfall|foot traffic|traffic|visitors?|page views?|bounce)\b",
     "The data has orders, products, customers, stores and stock only. There is no visitor or conversion information.",
     ["Orders by month", "Average order value by store"]),
    ("NO_FORECAST", r"\b(?:forecast\w*|predict\w*|projection\w*|next (?:month|year|quarter|week)|will be)\b",
     "I only report what is in the data. I cannot predict future values.",
     ["Monthly revenue trend", "Revenue by month"]),
    ("NO_SUCH_DATA", r"\b(?:competitors?|market share|suppliers?|employees?|staff|salary|salaries|discounts?|coupons?"
                     r"|promotions?|campaigns?|reviews?|ratings?|feedback|seasons?|gender|age|religion)\b",
     "The data has no information about that.",
     ["Revenue by category", "Return rate by region"]),
    ("CAUSAL_QUESTION", r"\bwhy\b|\breason for\b|\bwhat caused\b",
     "The data can show what changed (by month, category, region or store) but it cannot tell me why it changed.",
     ["Monthly revenue trend", "Revenue by category", "Return rate by region"]),
]
RELATIVE_TIME_RE = re.compile(r"\b(?:last|this|previous|current) (?:month|year|week|quarter)\b|\byesterday\b|\btoday\b"
                              r"|\brecent(?:ly)?\b|\bso far\b")
DOMAIN_RE = re.compile(
    r"revenue|sales|sold|selling|order|unit|quantit|return|aov|average|avg|mean|median|categor|region|store|product|brand"
    r"|customer|stock|inventory|price|cost|month|year|city|cities|sku|top|bottom|list|show|how many|count|total|compare"
    r"| vs |describe|statistic|distinct|unique|different|flag|health|restock|attention|filter|those|them|only|highest"
    r"|lowest|best|worst|most|least|number|rate|trend|what about|how about")
CLEAR_RE = re.compile(r"\b(?:clear|reset|remove|drop|forget) (?:all |the |my |any )?(?:filters?|scope|selection)\b"
                      r"|\bstart over\b|\bnew topic\b")
FOLLOWUP_RE = re.compile(r"^(?:only|and|now|then|also|what about|how about|same|just|instead|among|of those|of these|"
                         r"for those|for them|in those|out of those|but)\b|\b(?:among|of|for|in|from|out of) (?:those|these|them)\b"
                         r"|\bthose\b|\bthem\b|\bthat one\b")

# --------------------------------------------------------------------- vocabulary
METRIC_PATTERNS = [
    ("return_rate_pct", r"return(?:ed)?[- ]rate|rate of returns?|returns? (?:percentage|ratio)"
                        r"|percentage of (?:the )?(?:orders )?(?:that )?(?:were |are |got )?returned|% returned"),
    ("returned_revenue", r"returned revenue|revenue (?:lost )?(?:to|from) returns?"),
    ("returned_orders", r"returned orders?|orders? (?:that were |which were )?returned|(?:number of |how many )returns?"
                        r"|returns?|returned"),
    ("aov", r"average order (?:value|size)|avg\.? order (?:value|size)|aov|average basket|basket size"),
    ("units", r"units?(?: sold)?|quantit(?:y|ies)|items? sold|pieces"),
    ("orders", r"orders?|transactions?"),
    ("revenue", r"revenue|turnover|income|money (?:made|earned)"),
]
LABELS = {"revenue": "Revenue", "orders": "Orders", "units": "Units sold", "aov": "Average order value",
          "return_rate_pct": "Return rate", "returned_orders": "Returned orders", "returned_revenue": "Returned revenue"}
UNITS = {"revenue": "money", "aov": "money", "returned_revenue": "money", "return_rate_pct": "percent"}
SALES_RE = re.compile(r"\b(?:sales|sold|selling|best[- ]?sellers?|best[- ]?selling)\b")
AVG_RE = re.compile(r"\b(average|avg|mean)\s+(price|cost)\b")
DESCRIBE_RE = re.compile(r"\b(?:describe|statistics|stats|summary statistics|distribution|spread)\b")
DESCRIBE_FIELDS = {"revenue": ("orders", "revenue"), "quantity": ("orders", "quantity"),
                   "price": ("products", "price"), "cost": ("products", "cost"), "stock": ("inventory", "stock_on_hand")}

DIM_WORD = (r"(categor(?:y|ies)|regions?|stores?|products?|brands?|cit(?:y|ies)|customers?|months?|years?"
            r"|storecity|customercity)")
DIM_CUE = r"(?:by|per|for each|each|across|which|what|top|bottom|highest|lowest|best|worst|most|least)"
DIM_RE = re.compile(rf"\b{DIM_CUE}(?:\s+(?:\d+|{NUM_ALT}))?(?:\s+(?:selling|performing))?\s+{DIM_WORD}\b")
DIM_WITH_RE = re.compile(rf"\b{DIM_WORD}\s+(?:with|has|have|had)\s+(?:the\s+)?(?:highest|lowest|most|least|best|worst|largest|smallest|fewest)\b")
EXTRA_DIM_RE = re.compile(rf"\s*(?:,|and|then)?\s*{DIM_WORD}\b")
TREND_RE = re.compile(r"\b(?:monthly|yearly|annual|trend|over time)\b")
DIM_KIND = {"category": "category", "categories": "category", "region": "region", "regions": "region",
            "store": "store", "stores": "store", "product": "product", "products": "product", "brand": "brand",
            "brands": "brand", "city": "city", "cities": "city", "customer": "customer", "customers": "customer",
            "month": "month", "months": "month", "year": "year", "years": "year",
            "storecity": "storecity", "customercity": "customercity"}

HIGH_RE = re.compile(r"\b(?:highest|most|best|largest|biggest|maximum|max|greatest|top)\b")
LOW_RE = re.compile(r"\b(?:lowest|least|fewest|smallest|worst|minimum|min|bottom)\b")
TOPN_RE = re.compile(rf"\b(?:top|bottom|first|last)\s+(\d+|{NUM_ALT})\b")
NUM_FILTER_RE = re.compile(
    r"\b(revenue|quantity|price|cost|stock)\s*(?:of|is|was|are)?\s*(above|over|greater than|more than|at least|below|under"
    r"|less than|at most|equal to|equals)\s*(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)")
NUM_OPS = {"above": "gt", "over": "gt", "greater than": "gt", "more than": "gt", "at least": "gte", "below": "lt",
           "under": "lt", "less than": "lt", "at most": "lte", "equal to": "eq", "equals": "eq"}
NUM_FIELD = {"revenue": "revenue", "quantity": "quantity", "price": "price", "cost": "cost", "stock": "stock_on_hand"}
ISO_RE = r"\d{4}-\d{2}-\d{2}"
DATE_BETWEEN_RE = re.compile(rf"\bbetween\s+({ISO_RE})\s+and\s+({ISO_RE})\b")
DATE_ONE_RE = re.compile(rf"\b(since|after|before|until|on|from)\s+({ISO_RE})\b")
MONTH_RE = re.compile(rf"\b(in|during|for|of|since|from|after|before|until|through)\s+({MONTH_ALT})(?:\s+(\d{{4}}))?\b")
YEAR_RE = re.compile(r"\b(in|during|for|of)\s+(20\d{2})\b")

ENTITY_COUNT_RE = re.compile(r"\bhow many (?:different |distinct |unique )?(products|customers|stores|categories|brands|regions|cities)\b"
                             r"|\bnumber of (?:different |distinct |unique )?(products|customers|stores|categories|brands|regions|cities)\b")
ENTITY_COUNT = {"products": ("products", "count", None), "customers": ("customers", "count", None),
                "stores": ("stores", "count", None), "categories": ("products", "count_distinct", "category"),
                "brands": ("products", "count_distinct", "brand"), "regions": ("stores", "count_distinct", "region")}
DISTINCT_RE = re.compile(
    r"\b(?:what|which)\s+(categories|brands|regions|cities|stores)\b.*\b(?:exist|are there|do we have|available|sell|carry)\b"
    r"|\b(?:list|show|give me)\s+(?:me\s+)?(?:all\s+|the\s+)?(?:different\s+|distinct\s+|unique\s+)?(categories|brands|regions|cities)\b"
    r"|\b(?:distinct|unique|different)\s+(categories|brands|regions|cities|stores|products)\b")
DISTINCT_FIELDS = {"categories": ("products", "category"), "brands": ("products", "brand"),
                   "regions": ("stores", "region"), "stores": ("stores", "store_name"), "products": ("products", "product_id")}
LIST_TOP_RE = re.compile(rf"\b(top|bottom|biggest|largest|smallest|highest|lowest)\s*(\d+|{NUM_ALT})?\s*(orders|products)\s+by\s+(revenue|price|cost|quantity)\b")
LIST_RE = re.compile(r"\b(?:list|show|give me|display)\b(?: me)?(?: all| the)? (orders|products|stores|customers)\b")
LIST_COLUMNS = {"orders": ["order_id", "order_date", "product_id", "store_id", "quantity", "revenue"],
                "products": ["product_id", "category", "brand", "price", "cost"],
                "stores": ["store_id", "store_name", "city", "region"],
                "customers": ["customer_id", "city", "signup_date"]}
STOCK_RE = re.compile(r"\b(?:stock|inventory|stocked|in stock|out of stock|stock-?outs?)\b")
OUT_OF_STOCK_RE = re.compile(r"\b(?:out of stock|stock-?outs?|no stock|zero stock|sold out)\b")
INSIGHT_RE = re.compile(r"\b(?:flagged|health score|unhealthy|needs? attention|need restock\w*|restock\w*|underperform\w*"
                        r"|worst products?|weakest products?|product health|at risk)\b")
EXPLAIN_RE = re.compile(r"\b(?:why|explain|flag\w*|health|score|attention|tell me about|details?|status)\b")
METRIC_WORD_RE = re.compile(r"revenue|orders?|units?|quantit|return rate|aov|sales")


# ---------------------------------------------------------------------- helpers
def _refuse(code, message, suggestions=None):
    return {"kind": "refuse", "code": code, "message": message, "suggestions": suggestions or EXAMPLES[:3]}


def _clarify(question, options, term, original):
    return {"kind": "clarify", "question": question, "options": options, "term": term, "original": original}


def _blank(work, span):
    return work[:span[0]] + " " * (span[1] - span[0]) + work[span[1]:]


def _num(tok):
    return int(tok) if tok.isdigit() else NUM_WORDS[tok]


def _value_pattern(v):
    parts = [p for p in re.split(r"[-\s]+", v.strip().lower()) if p]
    body = r"[-\s]?".join(re.escape(p) for p in parts)
    return re.compile(r"(?<![a-z0-9])" + body + r"(?:e?s)?(?![a-z0-9])")


SKIP_VALUE_COLUMNS = {"returned", "product_name", "sku", "order_id", "order_date", "signup_date", "last_restock_date"}


def _value_candidates(catalog, ds):
    out = []
    for col, info in catalog.visible_columns(ds).items():
        if info["kind"] != "text" or col in SKIP_VALUE_COLUMNS:
            continue
        vals = catalog.distinct_values(ds, col, limit=60)
        for v in vals or []:
            if len(v) >= 3:
                out.append((col, v))
    return out


def _extract_values(work, ds, catalog, forced):
    """Find known values (West, Jeans, P007...) in the question. Returns (filters, compare_cols, ambiguity, work)."""
    spans = {}
    for col, v in _value_candidates(catalog, ds):
        for m in _value_pattern(v).finditer(work):
            spans.setdefault((m.start(), m.end()), []).append((col, v))
    used, per_col, ambiguity = [], {}, None
    for span in sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0])):
        if any(not (span[1] <= u[0] or span[0] >= u[1]) for u in used):
            continue
        used.append(span)
        options = spans[span]
        cols = sorted({c for c, _ in options})
        value = options[0][1]
        if len(cols) > 1:
            pick = forced.get(f"value:{value.lower()}")
            if pick in cols:
                options = [(c, v) for c, v in options if c == pick]
            elif ambiguity is None:
                ambiguity = {"value": value, "columns": cols}
                continue
            else:
                continue
        col, value = options[0]
        per_col.setdefault(col, [])
        if value not in per_col[col]:
            per_col[col].append(value)
    for span in used:
        work = _blank(work, span)
    filters, compare = [], []
    for col, values in per_col.items():
        if len(values) == 1:
            filters.append({"field": col, "op": "eq", "value": values[0]})
        else:
            filters.append({"field": col, "op": "in", "value": values})
            compare.append(col)
    return filters, compare, ambiguity, work


def _data_year(catalog):
    s = catalog.typed("orders")["order_date"].dropna()
    return int(s.max()[:4]) if len(s) else None


UNKNOWN_VALUE_RE = re.compile(r"\b(?:in|for|at|of|from) (?:the )?([a-z][a-z\-]{2,}) (region|category|brand)\b")
NOT_A_VALUE = {"each", "every", "this", "that", "which", "what", "all", "any", "the", "same", "whole", "one", "per",
               "our", "these", "those", "other", "each"}


def _unknown_value(work, original, ds, catalog):
    """'revenue in the Wset region': the word before 'region' is not a known value. Ask, do not silently ignore."""
    m = UNKNOWN_VALUE_RE.search(work)
    if not m or m.group(1) in NOT_A_VALUE or m.group(2) not in catalog.columns(ds):
        return None
    word, field = m.group(1), m.group(2)
    known = catalog.distinct_values(ds, field) or []
    if not known or word in {k.lower() for k in known}:
        return None
    close = difflib.get_close_matches(word, [k.lower() for k in known], n=3, cutoff=0.6)
    close = [k for k in known if k.lower() in close]
    if not close:
        return _refuse("UNKNOWN_VALUE", f"There is no {field} called '{word}'. Available {field} values: {', '.join(known[:12])}.",
                       [f"Revenue by {field}"])
    options = [{"label": v, "keywords": [v.lower()], "forced": {},
                "rewrite": re.sub(re.escape(word), v, original, count=1, flags=re.IGNORECASE)} for v in close]
    return _clarify(f"I cannot find a {field} called '{word}'. Did you mean one of these?", options, word, original)


def _extract_dates(work, ds, catalog):
    """Returns (filters, notes, work, error_message)."""
    filters, notes = [], []
    if ds not in ("orders", "orders_wide"):
        return filters, notes, work, None
    year = _data_year(catalog)
    m = DATE_BETWEEN_RE.search(work)
    if m:
        filters.append({"field": "order_date", "op": "between", "value": [m.group(1), m.group(2)]})
        work = _blank(work, m.span())
    for m in list(DATE_ONE_RE.finditer(work)):
        word, d = m.group(1), m.group(2)
        op = {"since": "gte", "from": "gte", "after": "gt", "before": "lt", "until": "lte", "on": "eq"}[word]
        filters.append({"field": "order_date", "op": op, "value": d})
        work = _blank(work, m.span())
    for m in list(MONTH_RE.finditer(work)):
        word, month, yr = m.group(1), MONTHS[m.group(2)], m.group(3)
        y = int(yr) if yr else year
        if not yr:
            notes.append(f"No year was given, so I used {year}, the year of the latest orders.")
        last = calendar.monthrange(y, month)[1]
        first_day, last_day = f"{y}-{month:02d}-01", f"{y}-{month:02d}-{last:02d}"
        if word in ("in", "during", "for", "of"):
            filters.append({"field": "order_date", "op": "between", "value": [first_day, last_day]})
        elif word in ("since", "from"):
            filters.append({"field": "order_date", "op": "gte", "value": first_day})
        elif word == "after":
            filters.append({"field": "order_date", "op": "gt", "value": last_day})
        elif word == "before":
            filters.append({"field": "order_date", "op": "lt", "value": first_day})
        else:
            filters.append({"field": "order_date", "op": "lte", "value": last_day})
        work = _blank(work, m.span())
    for m in list(YEAR_RE.finditer(work)):
        y = int(m.group(2))
        filters.append({"field": "order_date", "op": "between", "value": [f"{y}-01-01", f"{y}-12-31"]})
        work = _blank(work, m.span())
    return filters, notes, work, None


def _extract_numbers(work, ds, catalog):
    filters = []
    for m in list(NUM_FILTER_RE.finditer(work)):
        field = NUM_FIELD[m.group(1)]
        if field in catalog.columns(ds):
            filters.append({"field": field, "op": NUM_OPS[m.group(2)], "value": float(m.group(3).replace(",", ""))})
            work = _blank(work, m.span())
    return filters, work


def _detect_metrics(work):
    found = []
    for name, pattern in METRIC_PATTERNS:
        m = re.search(rf"\b(?:{pattern})\b", work)
        if m:
            found.append(name)
            work = _blank(work, m.span())
    return found, work


def _dim_field(ds, kind, catalog):
    cols = catalog.columns(ds)
    options = {"category": ["category"], "brand": ["brand"], "region": ["region"], "store": ["store_name"],
               "product": ["product_id"], "customer": ["customer_id"],
               "city": ["city"], "customercity": ["city"],
               "storecity": ["stores_city"] if ds == "orders_wide" else ["city"]}.get(kind, [])
    for field in options:
        if field in cols:
            return field
    return None


def _is_plural(word):
    return word.endswith("s") and word not in ("storecity", "customercity")


def _find_dims(work):
    """Find 'by region', 'which categories', 'by region and category'... Returns ([(kind, plural)], work)."""
    dims = []
    for regex in (DIM_RE, DIM_WITH_RE):
        pos = 0
        while True:
            m = regex.search(work, pos)
            if not m:
                break
            word = m.group(1)
            dims.append((m.start(1), DIM_KIND[word], _is_plural(word)))
            work = _blank(work, m.span())
            end = m.end()
            while True:                                   # "... by region AND category"
                x = EXTRA_DIM_RE.match(work, end)
                if not x:
                    break
                w = x.group(1)
                dims.append((x.start(1), DIM_KIND[w], _is_plural(w)))
                work = _blank(work, x.span())
                end = x.end()
            pos = end
    dims.sort()
    return [(k, p) for _, k, p in dims], work


def unmatched_guard(plan, catalog):
    """For rankings: leave out rows that have no match in a group column (a pile of unmatched rows is not a real group).
    Only acts when such rows exist. Adds a visible 'not_null' filter to the plan. Returns (filters_added, note)."""
    ds = plan["dataset"]
    have = {f["field"] for f in plan.get("filters", []) if f["op"] in ("eq", "in", "not_null")}
    auto = [{"field": g, "op": "not_null"} for g in plan.get("group_by", [])
            if g not in have and g in catalog.columns(ds) and catalog.typed(ds)[g].isna().any()]
    if not auto:
        return [], None
    plan["filters"] = plan.get("filters", []) + auto
    names = ", ".join(f["field"] for f in auto)
    return auto, (f"Rows with no match in {names} are left out of rankings (a group of unmatched rows is not a real group). "
                  "Ask without 'highest' or 'top' to see them.")


def normalize_question(question):
    """Returns (text, low): whitespace-cleaned original and a lowercase version with 'store city' tokens."""
    text = " ".join(question.strip().split())
    low = text.lower()
    low = re.sub(r"\bstore(?:'s)? cit(?:y|ies)\b", "storecity", low)
    low = re.sub(r"\bcustomer(?:'s)? cit(?:y|ies)\b", "customercity", low)
    return text, low


def screen(text, low, catalog):
    """Deterministic checks that run BEFORE any planner (rules or AI).
    Returns a refuse / insight / command outcome, or None to continue.
    Injection and personal-data requests are stopped here, so they never reach a model."""
    # 1. injection and personal data
    if INJECTION_RE.search(low):
        return _refuse("PROMPT_INJECTION", "I cannot follow instructions that change how I work. I only answer questions about "
                       "the loaded retail data, using checked calculations.")
    if PII_RE.search(low):
        return _refuse("PERSONAL_DATA", "Personal details such as names, emails, phone numbers and addresses are not available for questions.",
                       ["How many customers are there?", "Customers by city", "Revenue by customer city"])

    # 2. product-health questions
    product_ids = ({p.lower(): p for p in (catalog.distinct_values("products", "product_id") or [])}
                   if "products" in catalog.datasets else {})
    mentioned = [product_ids[t] for t in re.findall(r"\bp\d+\b", low) if t in product_ids]
    unknown_ids = [t.upper() for t in re.findall(r"\bp\d+\b", low) if t not in product_ids]
    if (INSIGHT_RE.search(low) or ((mentioned or unknown_ids) and EXPLAIN_RE.search(low))) and not METRIC_WORD_RE.search(low):
        if mentioned or unknown_ids:
            return {"kind": "insight", "insight": {"type": "explain", "product_id": (mentioned or unknown_ids)[0]}}
        m = re.search(rf"\b(\d+|{NUM_ALT})\b", low)
        return {"kind": "insight", "insight": {"type": "attention", "limit": min(_num(m.group(1)), 20) if m else 5}}

    # 3. things the data does not contain, and questions outside the retail data
    for code, pattern, message, suggestions in UNANSWERABLE:
        if re.search(pattern, low):
            return _refuse(code, message, suggestions)
    if RELATIVE_TIME_RE.search(low):
        s = catalog.typed("orders")["order_date"].dropna() if "orders" in catalog.datasets else []
        span = f" The orders in this data run from {s.min()} to {s.max()}." if len(s) else ""
        return _refuse("RELATIVE_TIME", "I cannot tell what 'last month' or 'recent' means for this data. "
                       "Please give a month name or dates." + span,
                       ["Revenue in March", "Orders between 2026-02-01 and 2026-02-28"])
    if not DOMAIN_RE.search(low):
        return _refuse("OUT_OF_SCOPE", "I can only answer questions about the loaded retail data: orders, products, "
                       "customers, stores and stock.", EXAMPLES[:4])
    if CLEAR_RE.search(low):
        return {"kind": "command", "command": "clear_filters"}
    return None


def screen_question(question, catalog):
    text, low = normalize_question(question)
    return screen(text, low, catalog)


# --------------------------------------------------------------------- main
def parse_question(question, catalog, state, forced=None):
    forced = forced or {}
    state = state or {}
    text, low = normalize_question(question)

    screened = screen(text, low, catalog)
    if screened:
        return screened

    followup = bool(FOLLOWUP_RE.search(low)) and bool(state.get("last_plan"))
    notes, work = [], low

    # 4. pick the dataset family -------------------------------------------------------
    asks_describe = bool(DESCRIBE_RE.search(low))
    describe_field = next((w for w in DESCRIBE_FIELDS if re.search(rf"\b{w}\b", low)), None)
    avg = AVG_RE.search(low)
    stock = bool(STOCK_RE.search(low))
    list_top = LIST_TOP_RE.search(low)
    entity_count = ENTITY_COUNT_RE.search(low)

    if asks_describe and describe_field:
        ds, field = DESCRIBE_FIELDS[describe_field]
        filters, compare, amb, work = _extract_values(work, ds, catalog, forced)
        return _finish_simple(catalog, ds, {"intent": "describe", "dataset": ds, "select": [field], "filters": filters},
                              {"units_by_column": {field: "money" if field in ("revenue", "price", "cost") else None}}, [])

    if list_top:
        direction, n, ent, field = list_top.groups()
        ds = ent
        limit = _num(n) if n else 5
        order = "asc" if direction in ("bottom", "smallest", "lowest") else "desc"
        cols = list(LIST_COLUMNS[ent])
        if field not in cols:
            cols.append(field)
        plan = {"intent": "list", "dataset": ds, "select": cols, "sort": [{"field": field, "dir": order}], "limit": min(limit, 100)}
        return _finish_simple(catalog, ds, plan, {"units_by_column": {"revenue": "money", "price": "money", "cost": "money"}}, [])

    if stock:
        return _plan_stock(work, low, catalog, state, forced, followup, bool(OUT_OF_STOCK_RE.search(low)))

    ds = "products" if avg else "orders_wide"
    if "orders_wide" not in catalog.datasets and not avg:
        ds = "orders" if "orders" in catalog.datasets else catalog.datasets[0]

    # 5. filters -----------------------------------------------------------------------
    filters, compare_cols, ambiguity, work = _extract_values(work, ds, catalog, forced)
    if ambiguity:
        v, cols = ambiguity["value"], ambiguity["columns"]
        labels = {"city": "customer city", "stores_city": "store city"}
        options = [{"label": labels.get(c, c.replace("_", " ")), "keywords": [labels.get(c, c), c], "forced": {f"value:{v.lower()}": c}} for c in cols]
        names = " or ".join(o["label"] for o in options)
        return _clarify(f"'{v}' appears in more than one column. Do you mean {names}?", options, v, text)
    unknown = _unknown_value(work, text, ds, catalog)
    if unknown:
        return unknown
    date_filters, date_notes, work, _ = _extract_dates(work, ds, catalog)
    num_filters, work = _extract_numbers(work, ds, catalog)
    filters = filters + date_filters + num_filters
    notes += date_notes

    # 6. entity counts / distinct values / list ---------------------------------------
    if entity_count and not avg:
        key = next(g for g in entity_count.groups() if g)
        if key == "cities":
            options = [{"label": "store cities", "keywords": ["store", "stores"], "forced": {"cities_means": "stores"}},
                       {"label": "customer cities", "keywords": ["customer", "customers"], "forced": {"cities_means": "customers"}}]
            if forced.get("cities_means") is None:
                return _clarify("Do you mean the cities of the stores or the cities of the customers?", options, "cities", text)
            tbl = forced["cities_means"]
            spec = (tbl, "count_distinct", "city")
        else:
            spec = ENTITY_COUNT[key]
        tbl, agg, field = spec
        f2, _, _, _ = _extract_values(low, tbl, catalog, forced)
        metric = {"agg": agg, "as": key}
        if field:
            metric["field"] = field
        plan = {"intent": "aggregate", "dataset": tbl, "filters": f2, "metrics": [metric]}
        return _finish_simple(catalog, tbl, plan, {"labels": {key: key.capitalize()}, "units": {}}, [])

    found, work = _detect_metrics(work)
    sales_word = bool(SALES_RE.search(work))

    if not found and not sales_word and not avg:
        m = DISTINCT_RE.search(low)
        if m:
            key = next(g for g in m.groups() if g)
            if key == "cities":
                if forced.get("cities_means") is None:
                    options = [{"label": "store cities", "keywords": ["store", "stores"], "forced": {"cities_means": "stores"}},
                               {"label": "customer cities", "keywords": ["customer", "customers"], "forced": {"cities_means": "customers"}}]
                    return _clarify("Do you mean the cities of the stores or the cities of the customers?", options, "cities", text)
                tbl, field = forced["cities_means"], "city"
            else:
                tbl, field = DISTINCT_FIELDS[key]
            return _finish_simple(catalog, tbl, {"intent": "distinct", "dataset": tbl, "select": [field]}, {}, [])
        m = LIST_RE.search(low)
        if m:
            ent = m.group(1)
            f2, _, _, _ = _extract_values(low, ent, catalog, forced)
            nf, _ = _extract_numbers(low, ent, catalog)
            top = TOPN_RE.search(low)
            plan = {"intent": "list", "dataset": ent, "select": LIST_COLUMNS[ent], "filters": f2 + nf,
                    "limit": min(_num(top.group(1)), 100) if top else 10}
            return _finish_simple(catalog, ent, plan, {"units_by_column": {"revenue": "money", "price": "money", "cost": "money"}}, [])

    # 7. aggregate: measure, dimensions, ranking --------------------------------------
    dims, work = _find_dims(work)
    tb_unit = None
    if TREND_RE.search(work):
        tb_unit = "year" if re.search(r"\b(?:yearly|annual)\b", work) else "month"
    group_fields, plural_dim = [], False
    for kind, plural in dims:
        if kind in ("month", "year"):
            tb_unit = kind
            continue
        field = _dim_field(ds, kind, catalog)
        if field is None:
            return _refuse("CANNOT_GROUP", f"I cannot break these numbers down by {kind} for this data.", ["Revenue by category", "Revenue by region"])
        if field not in group_fields:
            group_fields.append(field)
        plural_dim = plural_dim or plural
    group_fields = group_fields[:3]

    metrics = []
    labels, units = {}, {}
    if avg:
        which = avg.group(2)
        alias = f"avg_{which}"
        metrics.append({"agg": "mean", "field": which, "as": alias})
        labels[alias], units[alias] = f"Average {which}", "money"
    for name in found:
        metrics.append({"agg": "named", "name": name, "as": name})
        labels[name], units[name] = LABELS[name], UNITS.get(name)
    if not metrics and sales_word:
        choice = forced.get("sales_means")
        if choice is None and followup and state.get("metrics"):
            metrics = state["metrics"]
            labels, units = dict(state.get("labels", {})), dict(state.get("units", {}))
        elif choice is None:
            return _clarify("'Sales' can mean different things. Which measure do you want?",
                            [{"label": "Revenue (money)", "keywords": ["revenue", "money"], "forced": {"sales_means": "revenue"}},
                             {"label": "Units sold", "keywords": ["units", "quantity", "items"], "forced": {"sales_means": "units"}},
                             {"label": "Number of orders", "keywords": ["orders", "order count"], "forced": {"sales_means": "orders"}}],
                            "sales", text)
        else:
            metrics.append({"agg": "named", "name": choice, "as": choice})
            labels[choice], units[choice] = LABELS[choice], UNITS.get(choice)
            notes.append(f"'Sales' was read as {LABELS[choice].lower()}, as you chose.")

    reused = []
    if not metrics:
        if followup and state.get("metrics"):
            metrics = state["metrics"]
            labels, units = dict(state.get("labels", {})), dict(state.get("units", {}))
            reused.append("measure")
        elif group_fields or tb_unit or HIGH_RE.search(low) or LOW_RE.search(low):
            return _clarify("What should I measure?",
                            [{"label": "Revenue (money)", "keywords": ["revenue", "money"], "forced": {"sales_means": "revenue"}},
                             {"label": "Units sold", "keywords": ["units", "quantity", "items"], "forced": {"sales_means": "units"}},
                             {"label": "Number of orders", "keywords": ["orders", "order count"], "forced": {"sales_means": "orders"}}],
                            "measure", text)
        elif filters:
            for name in ("revenue", "orders"):
                metrics.append({"agg": "named", "name": name, "as": name})
                labels[name], units[name] = LABELS[name], UNITS.get(name)
            notes.append("You did not ask for a specific measure, so I am showing revenue and orders for this scope.")
        else:
            return _clarify("I did not understand that. Which of these would you like?",
                            [{"label": e, "keywords": [e.lower()], "forced": {}, "rewrite": e} for e in EXAMPLES[:4]], "not_understood", text)

    if followup and not group_fields and not tb_unit and not dims:
        group_fields = list(state.get("group_by", []))
        tb_unit = tb_unit or (state.get("time_bucket") or {}).get("unit")
        if group_fields or tb_unit:
            reused.append("grouping")

    # carry filters from the previous question (follow-ups only)
    carried, dropped = [], []
    if followup:
        new_fields = {f["field"] for f in filters}
        for old in state.get("filters", []):
            if old["field"] in new_fields:
                continue
            (carried if old["field"] in catalog.columns(ds) else dropped).append(old)
        if carried:
            notes.append("Carried over from your earlier question: " + "; ".join(_fmt_filter(f) for f in carried) + ".")
        if dropped:
            notes.append("Not carried over (this dataset has no such column): " + "; ".join(_fmt_filter(f) for f in dropped) + ".")
        filters = carried + filters
    elif state.get("filters"):
        notes.append("Earlier filters were not carried over because this looks like a new question. "
                     "Start with 'among those' to keep them.")
    if reused:
        notes.append("Reused from the previous question: " + " and ".join(reused) + ".")

    # compare "West and East" -> group by that column
    if compare_cols and not group_fields and not tb_unit:
        group_fields = [compare_cols[0]]

    plan = {"intent": "aggregate", "dataset": ds, "filters": filters, "group_by": group_fields, "metrics": metrics}
    date_field = "order_date"
    if tb_unit:
        plan["time_bucket"] = {"field": date_field, "unit": tb_unit}
    ranking = None
    topn = TOPN_RE.search(low)
    asc, desc = bool(LOW_RE.search(low)), bool(HIGH_RE.search(low))
    if group_fields or tb_unit:
        first = metrics[0]["as"]
        bottom_word = bool(topn and topn.group(0).startswith(("bottom", "last"))) or bool(re.search(r"\bbottom\b", low))
        direction = "asc" if (bottom_word or (asc and not desc)) else "desc"
        explicit_rank = asc or desc
        if explicit_rank:
            ranking = "lowest" if direction == "asc" else "highest"
        if topn:
            limit = _num(topn.group(1))
        elif explicit_rank and re.search(r"\b(?:top|bottom)\b", low):
            limit = 5
        elif explicit_rank:
            limit = 3 if plural_dim else 1
        else:
            limit = 10
        if tb_unit and not explicit_rank and not topn:
            bucket = f"{date_field}_{tb_unit}"
            plan["sort"] = [{"field": bucket, "dir": "asc"}]
            limit = 36 if tb_unit == "month" else 10
        else:
            plan["sort"] = [{"field": first, "dir": direction}]
        plan["limit"] = min(limit, 100)

    auto_filters = []
    if ranking and group_fields:
        auto_filters, guard_note = unmatched_guard(plan, catalog)
        if guard_note:
            notes.append(guard_note)
    meta = {"labels": labels, "units": units, "ranking": ranking, "notes": notes, "auto_filters": auto_filters,
            "dims": group_fields + ([f"{date_field}_{tb_unit}"] if tb_unit else [])}
    return {"kind": "plan", "plan": plan, "meta": meta}


def _finish_simple(catalog, ds, plan, meta, notes):
    plan.setdefault("filters", [])
    meta = dict(meta)
    meta.setdefault("labels", {})
    meta.setdefault("units", {})
    meta.setdefault("ranking", None)
    meta["notes"] = meta.get("notes", []) + notes
    meta.setdefault("dims", [])
    return {"kind": "plan", "plan": plan, "meta": meta}


def _plan_stock(work, low, catalog, state, forced, followup, out_of_stock):
    ds = "inventory_wide" if "inventory_wide" in catalog.datasets else "inventory"
    filters, _, ambiguity, work = _extract_values(work, ds, catalog, forced)
    if ambiguity:
        v, cols = ambiguity["value"], ambiguity["columns"]
        options = [{"label": c, "keywords": [c], "forced": {f"value:{v.lower()}": c}} for c in cols]
        return _clarify(f"'{v}' appears in more than one column. Which do you mean?", options, v, low)
    num_filters, work = _extract_numbers(work, ds, catalog)
    filters += num_filters
    dims, work = _find_dims(work)
    group_fields = []
    for kind, _ in dims:
        field = _dim_field(ds, kind, catalog)
        if field and field not in group_fields:
            group_fields.append(field)
    notes = []
    if followup:
        new_fields = {f["field"] for f in filters}
        carried = [f for f in state.get("filters", []) if f["field"] not in new_fields and f["field"] in catalog.columns(ds)]
        if carried:
            notes.append("Carried over from your earlier question: " + "; ".join(_fmt_filter(f) for f in carried) + ".")
        filters = carried + filters
    if out_of_stock:
        filters.append({"field": "stock_on_hand", "op": "eq", "value": 0})
        metric = {"agg": "count", "as": "out_of_stock"}
        label = "Product-store combinations out of stock"
        notes.append("Each stock row is one product in one store; 'out of stock' means stock on hand is 0.")
    else:
        metric = {"agg": "sum", "field": "stock_on_hand", "as": "total_stock"}
        label = "Units in stock"
    plan = {"intent": "aggregate", "dataset": ds, "filters": filters, "group_by": group_fields[:3], "metrics": [metric]}
    if group_fields:
        plan["sort"] = [{"field": metric["as"], "dir": "asc" if LOW_RE.search(low) else "desc"}]
        plan["limit"] = 10
    meta = {"labels": {metric["as"]: label}, "units": {}, "ranking": None, "notes": notes, "dims": group_fields}
    return {"kind": "plan", "plan": plan, "meta": meta}
