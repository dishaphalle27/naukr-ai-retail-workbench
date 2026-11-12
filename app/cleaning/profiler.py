"""
PROFILER - looks at a CSV and reports what is inside. It NEVER changes the data.

Main ideas
1. We read every column as TEXT. If pandas guessed types, it could silently
   change "007" into 7 or hide "Rs. 1,299". Text keeps the truth.
2. The profiler only reports FACTS (counts, formats, examples).
   Deciding "is this a problem?" is the job of the issue detector (next step).
3. Personal data (emails, phones...) is never copied into the profile as examples.
   Text that looks like a prompt injection is counted, never copied.

Use in code / notebook:
    from app.cleaning.profiler import profile_file
    profile = profile_file("data/sample/orders.csv")

Use in terminal:
    python -m app.cleaning.profiler data/sample/orders.csv
"""
import json
import re
import sys
from pathlib import Path
import pandas as pd

WORD_NUMBERS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}

CATEGORY_ALIASES = {"tee", "tshirt", "t shirt", "t-shirt", "denim", "jackets", "footwear", "dresses"}
CITY_ALIASES = {"bangalore", "bombay"}


def build_cleaning_plan(file_path: str) -> dict:
    """Scans a messy CSV and creates a step-by-step cleaning plan before changing data."""
    path = Path(file_path)
    # Read everything as plain text first so nothing is altered automatically
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    plan = []

    # 1. Exact duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        plan.append({
            "id": "STEP_01_REMOVE_DUPLICATES",
            "reason": f"Found {dup_count} exact duplicate rows that would inflate totals.",
            "affected_fields": ["all_columns"],
            "risk": "low",
            "source": "code",
            "status": "pending"
        })

    # 2. Extra whitespace and inconsistent uppercase/lowercase flags or IDs
    messy_text_cols = []
    for col in df.columns:
        has_extra_spaces = df[col].apply(lambda x: x != x.strip() or "  " in x).any()
        unique_lower = df[col].str.strip().str.lower().nunique()
        unique_raw = df[col].nunique()
        is_id_or_flag = col in {"store_id", "region", "returned", "email"}
        if has_extra_spaces or (is_id_or_flag and unique_lower < unique_raw):
            messy_text_cols.append(col)

    if messy_text_cols:
        plan.append({
            "id": "STEP_02_TRIM_AND_STANDARDIZE_TEXT",
            "reason": "Found extra spaces or inconsistent uppercase/lowercase values (e.g., 'Yes' vs 'yes', 's01' vs 'S01').",
            "affected_fields": messy_text_cols,
            "risk": "low",
            "source": "code",
            "status": "pending"
        })

    # 3. Currency symbols or commas inside money columns (e.g., 'Rs. 1,299')
    currency_cols = [
        col for col in df.columns
        if any(k in col for k in ["price", "revenue", "cost"])
        and df[col].str.contains(r"Rs\.|,", regex=True).any()
    ]
    if currency_cols:
        plan.append({
            "id": "STEP_03_CLEAN_CURRENCY_TEXT",
            "reason": "Found money amounts stored as text with 'Rs.' or commas.",
            "affected_fields": currency_cols,
            "risk": "low",
            "source": "code",
            "status": "pending"
        })

    # 4. Numbers written as English words (e.g., 'one', 'two')
    word_num_cols = [
        col for col in df.columns
        if df[col].str.strip().str.lower().isin(WORD_NUMBERS).any()
    ]
    if word_num_cols:
        plan.append({
            "id": "STEP_04_CONVERT_WORD_NUMBERS",
            "reason": "Found numbers written as English words ('one', 'two') instead of digits.",
            "affected_fields": word_num_cols,
            "risk": "low",
            "source": "code",
            "status": "pending"
        })

    # 5. Mixed and ambiguous date formats (e.g., 05/01/2026)
    date_cols = [col for col in df.columns if "date" in col.lower()]
    if date_cols:
        has_non_iso = any(
            not df[col].str.match(r"^\d{4}-\d{2}-\d{2}$").all()
            for col in date_cols
        )
        if has_non_iso:
            plan.append({
                "id": "STEP_05_NORMALIZE_DATES_DAY_FIRST",
                "reason": "Found mixed date formats including ambiguous DD/MM vs MM/DD dates. Policy: parse slash dates as DD/MM/YYYY.",
                "affected_fields": date_cols,
                "risk": "medium",
                "source": "code",
                "status": "pending"
            })

    # 6. Retail SKU formatting variants (e.g., 'sku0001' or 'SKU_0001' -> 'SKU-0001')
    if "sku" in df.columns:
        non_standard_sku = not df["sku"].str.strip().str.match(r"^SKU-\d{4}$").all()
        if non_standard_sku:
            plan.append({
                "id": "STEP_06_STANDARDIZE_SKU_FORMAT",
                "reason": "Found inconsistent SKU formats (missing hyphens or lowercase prefixes). Policy: normalize to 'SKU-XXXX'.",
                "affected_fields": ["sku"],
                "risk": "low",
                "source": "code",
                "status": "pending"
            })

    # 7. Retail category or city aliases (e.g., 'Tee' -> 'T-Shirt', 'Bombay' -> 'Mumbai')
    alias_cols = []
    if "category" in df.columns and df["category"].str.strip().str.lower().isin(CATEGORY_ALIASES).any():
        alias_cols.append("category")
    if "city" in df.columns and df["city"].str.strip().str.lower().isin(CITY_ALIASES).any():
        alias_cols.append("city")

    if alias_cols:
        plan.append({
            "id": "STEP_07_NORMALIZE_RETAIL_ALIASES",
            "reason": "Found retail category or city aliases (e.g., 'Tee'/'T-Shirt', 'Bombay'/'Mumbai'). Policy: map to canonical names.",
            "affected_fields": alias_cols,
            "risk": "low",
            "source": "code",
            "status": "pending"
        })

    # 8. Negative stock values in inventory
    if "stock_on_hand" in df.columns:
        has_negative = df["stock_on_hand"].str.strip().str.startswith("-").any()
        if has_negative:
            plan.append({
                "id": "STEP_08_FLAG_NEGATIVE_STOCK",
                "reason": "Found negative stock_on_hand values. Policy: clamp to 0 and flag for audit instead of deleting rows.",
                "affected_fields": ["stock_on_hand"],
                "risk": "medium",
                "source": "code",
                "status": "pending"
            })

    # 9. Empty / missing cells
    null_cols = [col for col in df.columns if (df[col].str.strip() == "").any()]
    if null_cols:
        plan.append({
            "id": "STEP_09_HANDLE_MISSING_VALUES",
            "reason": "Found empty cells. Policy: fill missing IDs with 'UNKNOWN' and keep numeric nulls explicit without dropping rows.",
            "affected_fields": null_cols,
            "risk": "low",
            "source": "code",
            "status": "pending"
        })

    return {
        "file": path.name,
        "total_steps": len(plan),
        "cleaning_plan": plan
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.cleaning.planner <path_to_csv>")
        sys.exit(1)

    result = build_cleaning_plan(sys.argv[1])
    print(json.dumps(result, indent=2))