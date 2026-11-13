"""
RETAIL METRICS - every number comes from plain pandas arithmetic, never from an AI.

Each function returns the numbers PLUS the evidence:
  formulas        how each number is calculated, in words
  rows_considered how many rows went in
  excluded        rows left out and why (so nothing disappears silently)
  columns_used    which columns were read
"""
import pandas as pd

from app.errors import DataLoadError

NO_MATCH = "(no match)"


def num(series):
    """Text -> numbers. Anything that is not a number becomes NaN (and is counted by callers)."""
    clean = series.where(series.notna(), "").astype(str).str.strip()
    return pd.to_numeric(clean, errors="coerce")


def _need(df, cols, table="table"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataLoadError("MISSING_COLUMN", f"{table} has no column(s): {', '.join(missing)}",
                            {"missing": missing, "available": list(df.columns)})


def _round(x, nd=2):
    return None if x is None or pd.isna(x) else round(float(x), nd)


# ---------------------------------------------------------------- sales
SALES_FORMULAS = {
    "revenue": "sum of revenue",
    "orders": "number of distinct order_id",
    "units": "sum of quantity",
    "aov": "revenue / orders  (average order value)",
    "returned_orders": "distinct orders where returned = Y",
    "return_rate_pct": "100 * returned_orders / orders",
    "returned_revenue": "sum of revenue where returned = Y",
}


def _sales_row(part):
    rev = part["_rev"]
    oid = part["order_id"] if "order_id" in part.columns else pd.Series(part.index.astype(str), index=part.index)
    orders = int(oid.nunique())
    is_ret = part["_ret"]
    returned_orders = int(oid[is_ret].nunique())
    revenue = rev.sum()
    return {
        "revenue": _round(revenue),
        "orders": orders,
        "units": _round(part["_qty"].sum()) if part["_qty"].notna().any() else None,
        "aov": _round(revenue / orders) if orders else None,
        "returned_orders": returned_orders,
        "return_rate_pct": _round(100 * returned_orders / orders) if orders else None,
        "returned_revenue": _round(rev[is_ret].sum()),
    }


def sales_metrics(orders, group_by=None):
    """Revenue, AOV, return rate... overall or per group (e.g. category, region).
    Returns (rows, meta). Rows with no match in a joined table are grouped as '(no match)'."""
    _need(orders, ["revenue"], "orders")
    group_by = list(group_by or [])
    _need(orders, group_by, "orders")

    df = orders.copy()
    df["_rev"] = num(df["revenue"])
    df["_qty"] = num(df["quantity"]) if "quantity" in df.columns else float("nan")
    df["_ret"] = (df["returned"].astype(str).str.strip().str.upper() == "Y") if "returned" in df.columns else False
    for g in group_by:
        df[g] = df[g].where(df[g].notna() & (df[g].astype(str).str.strip() != ""), NO_MATCH)

    if group_by:
        rows = []
        for key, part in df.groupby(group_by, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            rows.append({**dict(zip(group_by, key)), **_sales_row(part)})
    else:
        rows = [_sales_row(df)]

    meta = {
        "formulas": SALES_FORMULAS,
        "rows_considered": len(df),
        "group_by": group_by,
        "columns_used": [c for c in ["order_id", "revenue", "quantity", "returned"] + group_by if c in orders.columns],
        "excluded": {"revenue_not_a_number": int(df["_rev"].isna().sum()),
                     "quantity_not_a_number": int(df["_qty"].isna().sum())},
        "missing_evidence": [] if "returned" in orders.columns else ["no 'returned' column: return rate not available"],
    }
    return rows, meta


# ------------------------------------------------------------ inventory
STOCK_FORMULAS = {
    "stock_out_rate_pct": "100 * (product-store rows with stock = 0) / (rows with a valid stock number)",
    "low_stock_rate_pct": "100 * (rows with stock <= reorder_level) / (rows with a valid stock number)",
    "total_units_in_stock": "sum of stock_on_hand (valid rows only)",
}


def stock_metrics(inventory):
    _need(inventory, ["stock_on_hand"], "inventory")
    stock = num(inventory["stock_on_hand"])
    missing = int(stock.isna().sum())
    negative = int((stock < 0).sum())
    valid = stock[stock.notna() & (stock >= 0)]
    n = len(valid)
    out = {"stock_out_rate_pct": _round(100 * (valid == 0).sum() / n) if n else None,
           "total_units_in_stock": _round(valid.sum()), "rows_with_valid_stock": int(n)}
    if "reorder_level" in inventory.columns:
        reorder = num(inventory["reorder_level"])[valid.index]
        out["low_stock_rate_pct"] = _round(100 * (valid <= reorder).sum() / n) if n else None
    meta = {"formulas": STOCK_FORMULAS, "rows_considered": len(inventory),
            "excluded": {"missing_stock": missing, "negative_stock_treated_as_unreliable": negative},
            "columns_used": [c for c in ["stock_on_hand", "reorder_level"] if c in inventory.columns]}
    return out, meta


def period_days(orders):
    """Number of days between the first and last order (inclusive), or None."""
    if "order_date" not in orders.columns:
        return None
    d = pd.to_datetime(orders["order_date"].where(orders["order_date"].notna(), ""),
                       format="%Y-%m-%d", errors="coerce").dropna()
    return int((d.max() - d.min()).days) + 1 if len(d) else None


def product_stock_position(orders, inventory):
    """Per product: units sold, stock, sell-through %, daily velocity, stock cover in days."""
    _need(orders, ["product_id", "quantity"], "orders")
    _need(inventory, ["product_id", "stock_on_hand"], "inventory")
    days = period_days(orders)
    sold = num(orders["quantity"]).groupby(orders["product_id"]).sum()
    st = num(inventory["stock_on_hand"])
    stock = st[st >= 0].groupby(inventory.loc[st[st >= 0].index, "product_id"]).sum()
    out = {}
    for pid in sorted(set(sold.index) | set(stock.index)):
        u, s = float(sold.get(pid, 0.0)), float(stock.get(pid, 0.0))
        velocity = u / days if days else None
        out[pid] = {
            "units_sold": _round(u), "stock_units": _round(s),
            "sell_through_pct": _round(100 * u / (u + s)) if (u + s) else None,
            "daily_velocity": _round(velocity, 3),
            "stock_cover_days": _round(s / velocity) if velocity else None,
        }
    meta = {"period_days": days,
            "formulas": {"sell_through_pct": "100 * units_sold / (units_sold + stock_units)",
                         "daily_velocity": "units_sold / period_days",
                         "stock_cover_days": "stock_units / daily_velocity"}}
    return out, meta
