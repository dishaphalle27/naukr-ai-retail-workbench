"""
PRODUCT HEALTH SCORE (0-100, higher = healthier). Fully explainable and reproducible.

Four explicit signals, each turned into a 0-1 number by a fixed rule:

  margin_pct        (price - cost) / price * 100       0% -> 0,  50% or more -> 1
  velocity_per_30d  units sold per 30 days             0 -> 0,   75th percentile of all products or more -> 1
  return_rate_pct   returned orders / orders * 100     5% or less -> 1,  25% or more -> 0   (needs >= 5 orders)
  stock_cover_days  stock / daily sales                under 7 days -> days/7,  7 to 60 -> 1,
                                                       60 to 120 -> falls to 0.4,  over 120 -> 0.4

Weights: margin 0.30, velocity 0.30, return rate 0.20, stock cover 0.20.

If a signal cannot be calculated (e.g. cost is missing) it is LEFT OUT, the remaining
weights are scaled up to 100%, and the gap is listed under "missing_evidence".
"evidence_coverage" tells you how much of the full weight was available.
The AI never changes a score. It may only explain it afterwards.
"""
import pandas as pd

from app.errors import DataLoadError
from app.retail.metrics import num, period_days

WEIGHTS = {"margin_pct": 0.30, "velocity_per_30d": 0.30, "return_rate_pct": 0.20, "stock_cover_days": 0.20}
MIN_ORDERS_FOR_RETURN_RATE = 5
ATTENTION_BELOW = 0.4        # a signal at or under this value is reported as a reason


def _clamp(x):
    return max(0.0, min(1.0, x))


def _norm_margin(v):
    return _clamp(v / 50.0)


def _norm_return(v):
    return _clamp(1 - (v - 5) / 20.0)


def _norm_cover(v):
    if v < 7:
        return _clamp(v / 7.0)
    if v <= 60:
        return 1.0
    if v <= 120:
        return 1.0 - 0.6 * (v - 60) / 60.0
    return 0.4


def _percentile75(values):
    vals = sorted(v for v in values if v is not None and v > 0)
    if not vals:
        return None
    idx = 0.75 * (len(vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)


def _recommend(sig, norm):
    cover, ret, margin = sig.get("stock_cover_days"), sig.get("return_rate_pct"), sig.get("margin_pct")
    if cover is not None and cover < 7 and norm.get("velocity_per_30d", 0) >= 0.5:
        return "Restock soon: selling well but stock covers less than a week"
    if ret is not None and ret > 15:
        return "Investigate returns: return rate is high"
    if margin is not None and margin < 30:
        return "Review price or cost: margin is low"
    if cover is not None and cover > 120:
        return "Overstocked: consider a promotion"
    return "No action needed"


def product_scores(orders, products, inventory):
    """Returns {"scores": [...], "meta": {...}}. Scores are sorted worst first."""
    for df, cols, name in ((orders, ["product_id", "quantity"], "orders"),
                           (products, ["product_id", "price", "cost"], "products"),
                           (inventory, ["product_id", "stock_on_hand"], "inventory")):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise DataLoadError("MISSING_COLUMN", f"{name} has no column(s): {', '.join(missing)}", {"missing": missing})

    days = period_days(orders)
    qty = num(orders["quantity"])
    sold = qty.groupby(orders["product_id"]).sum()
    oid = orders["order_id"] if "order_id" in orders.columns else pd.Series(orders.index.astype(str), index=orders.index)
    n_orders = oid.groupby(orders["product_id"]).nunique()
    if "returned" in orders.columns:
        is_ret = orders["returned"].astype(str).str.strip().str.upper() == "Y"
        n_ret = oid[is_ret].groupby(orders.loc[is_ret, "product_id"]).nunique()
    else:
        n_ret = None
    st = num(inventory["stock_on_hand"])
    ok = st.notna() & (st >= 0)
    stock = st[ok].groupby(inventory.loc[ok, "product_id"]).sum()

    velocity = {}
    for pid in products["product_id"]:
        velocity[pid] = (float(sold.get(pid, 0.0)) * 30.0 / days) if days else None
    ref = _percentile75(velocity.values())

    scores = []
    for _, p in products.iterrows():
        pid = p["product_id"]
        price, cost = num(pd.Series([p["price"]])).iloc[0], num(pd.Series([p["cost"]])).iloc[0]
        sig, norm, missing = {}, {}, []

        if pd.notna(price) and pd.notna(cost) and price > 0:
            sig["margin_pct"] = round(100 * (price - cost) / price, 1)
            norm["margin_pct"] = _norm_margin(sig["margin_pct"])
        else:
            missing.append("price or cost missing: margin not scored")

        v = velocity.get(pid)
        if v is not None and ref:
            sig["velocity_per_30d"] = round(v, 2)
            norm["velocity_per_30d"] = _clamp(v / ref)
        else:
            missing.append("sales period unknown or no sales in the data: velocity not scored")

        orders_n = int(n_orders.get(pid, 0))
        if n_ret is None:
            missing.append("no 'returned' column: return rate not scored")
        elif orders_n < MIN_ORDERS_FOR_RETURN_RATE:
            missing.append(f"only {orders_n} orders (need {MIN_ORDERS_FOR_RETURN_RATE}): return rate not scored")
        else:
            sig["return_rate_pct"] = round(100 * int(n_ret.get(pid, 0)) / orders_n, 1)
            norm["return_rate_pct"] = _norm_return(sig["return_rate_pct"])

        daily = (float(sold.get(pid, 0.0)) / days) if days else None
        if pid not in stock.index:
            missing.append("no valid stock rows: stock cover not scored")
        elif not daily:
            missing.append("no sales: stock cover not scored")
        else:
            sig["stock_cover_days"] = round(float(stock[pid]) / daily, 1)
            norm["stock_cover_days"] = _norm_cover(sig["stock_cover_days"])

        avail = sum(WEIGHTS[k] for k in norm)
        if avail == 0:
            scores.append({"product_id": pid, "score": None, "evidence_coverage": 0.0, "contributions": [],
                           "reasons": [], "recommendation": "Not enough evidence", "missing_evidence": missing})
            continue
        contrib, reasons = [], []
        for k in WEIGHTS:
            if k in norm:
                pts = round(100 * WEIGHTS[k] * norm[k] / avail, 1)
                contrib.append({"signal": k, "value": sig[k], "normalized": round(norm[k], 2),
                                "weight": WEIGHTS[k], "points": pts})
                if norm[k] <= ATTENTION_BELOW:
                    reasons.append({"margin_pct": f"low margin ({sig[k]}%)",
                                    "velocity_per_30d": f"slow sales ({sig[k]} units per 30 days)",
                                    "return_rate_pct": f"high return rate ({sig[k]}%)",
                                    "stock_cover_days": f"stock cover of {sig[k]} days is outside the healthy 7 to 60"}[k])
        score = round(sum(c["points"] for c in contrib), 1)
        scores.append({"product_id": pid, "score": score, "evidence_coverage": round(avail, 2),
                       "signals": sig, "contributions": contrib, "reasons": reasons,
                       "recommendation": _recommend(sig, norm), "missing_evidence": missing})

    scores.sort(key=lambda s: (s["score"] is None, s["score"] if s["score"] is not None else 0, s["product_id"]))
    orders_unmatched = sorted(set(orders["product_id"]) - set(products["product_id"]))
    meta = {"weights": WEIGHTS, "period_days": days, "velocity_reference_75th_percentile": _round_or_none(ref),
            "products_scored": sum(1 for s in scores if s["score"] is not None), "products_total": len(scores),
            "order_product_ids_not_in_products": orders_unmatched[:5]}
    return {"scores": scores, "meta": meta}


def _round_or_none(x):
    return None if x is None else round(x, 3)
