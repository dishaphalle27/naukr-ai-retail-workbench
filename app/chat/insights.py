"""
RETAIL INSIGHTS - product-health questions ("which products need attention?", "why is P012 flagged?").

This is the "retail conversational intelligence" flow: it explains a score using the
explicit signals from app/retail/scoring.py. Nothing here is invented:
every number comes from product_scores(), and the evidence lists the formulas and sources.
"""
from app.chat.answers import fmt_value
from app.errors import DataLoadError
from app.retail.scoring import WEIGHTS, product_scores

NEEDED = ("orders", "products", "inventory")
SIGNAL_NAMES = {"margin_pct": "margin", "velocity_per_30d": "sales speed", "return_rate_pct": "return rate",
                "stock_cover_days": "stock cover"}
SIGNAL_UNITS = {"margin_pct": "%", "velocity_per_30d": " units/30 days", "return_rate_pct": "%", "stock_cover_days": " days"}
FORMULAS = {
    "margin_pct": "(price - cost) / price * 100;  0% scores 0, 50% or more scores 1",
    "velocity_per_30d": "units sold * 30 / days covered by the orders;  scored against the 75th percentile of all products",
    "return_rate_pct": "returned orders / orders * 100;  5% or less scores 1, 25% or more scores 0;  needs at least 5 orders",
    "stock_cover_days": "stock units / daily sales;  under 7 days scores days/7, 7 to 60 scores 1, over 120 scores 0.4",
    "score": "sum of (weight x signal score) x 100, rescaled when a signal is missing",
}


def _evidence(catalog, result, shown, extra_notes):
    versions = {n: catalog.version(n) for n in NEEDED}
    meta = result["meta"]
    return {
        "dataset": "product_health", "version": "; ".join(versions.values()), "stage": catalog.stage,
        "datasets": versions,
        "columns_used": ["orders.product_id", "orders.order_id", "orders.quantity", "orders.returned", "orders.order_date",
                         "products.product_id", "products.price", "products.cost", "inventory.product_id",
                         "inventory.stock_on_hand"],
        "filters": [], "joins": [],
        "operations": [
            {"op": "signal", "detail": "margin = (price - cost) / price * 100 from products"},
            {"op": "signal", "detail": f"sales speed = units sold per 30 days over {meta['period_days']} days of orders"},
            {"op": "signal", "detail": "return rate = returned orders / orders per product (needs 5+ orders)"},
            {"op": "signal", "detail": "stock cover = stock on hand (valid rows only) / daily sales"},
            {"op": "score", "detail": f"weights {WEIGHTS}; missing signals are left out and weights rescaled"},
        ],
        "rows_in_dataset": catalog.row_count("orders"), "rows_after_filters": catalog.row_count("orders"),
        "result_rows": shown, "result_groups_total": meta["products_total"], "truncated": False,
        "preview_limit": shown, "formulas": FORMULAS,
        "notes": extra_notes + ([f"Orders mention product ids that are not in the products table: "
                                 f"{', '.join(meta['order_product_ids_not_in_products'])}. They cannot be scored."]
                                if meta["order_product_ids_not_in_products"] else []),
    }


def _load(catalog):
    missing = [n for n in NEEDED if n not in catalog.datasets]
    if missing:
        raise DataLoadError("MISSING_DATASET", f"Product health needs {', '.join(NEEDED)}; missing: {', '.join(missing)}")
    return product_scores(catalog.table_text("orders"), catalog.table_text("products"), catalog.table_text("inventory"))


def _reasons(s):
    return "; ".join(s["reasons"]) if s["reasons"] else "no signal is below the warning level"


def attention_list(catalog, limit=5):
    result = _load(catalog)
    scored = [s for s in result["scores"] if s["score"] is not None][:limit]
    rows = [{"product_id": s["product_id"], "score": s["score"], "recommendation": s["recommendation"],
             "reasons": _reasons(s)} for s in scored]
    lines = [f"  - {s['product_id']}: {fmt_value(s['score'])}/100 -> {s['recommendation']}. Why: {_reasons(s)}." for s in scored]
    e = _evidence(catalog, result, len(rows), ["Lowest health scores first. Ask 'why is P012 flagged?' for the full breakdown."])
    text = "\n".join([f"{len(rows)} products with the lowest health score (0 to 100, higher is healthier):"] + lines +
                     ["Score = margin 30%, sales speed 30%, return rate 20%, stock cover 20%. Missing evidence is left out, never guessed.",
                      f"Based on {fmt_value(e['rows_in_dataset'])} orders over {result['meta']['period_days']} days ({e['version']})."])
    return {"status": "ok", "answer": {"text": text, "headline": text.split("\n")[0], "columns": list(rows[0]) if rows else [],
                                       "rows": rows, "assumptions": []},
            "plan": {"intent": "insight", "type": "attention", "limit": limit}, "evidence": e, "warnings": []}


def explain_product(catalog, product_id):
    result = _load(catalog)
    match = next((s for s in result["scores"] if s["product_id"] == product_id), None)
    if match is None:
        orphan = product_id in result["meta"]["order_product_ids_not_in_products"]
        msg = (f"{product_id} appears in orders but not in the products table, so it has no price, cost or stock to score."
               if orphan else f"I cannot find a product with id {product_id}.")
        return {"status": "refused", "refusal": {"code": "UNKNOWN_PRODUCT", "message": msg,
                                                 "suggestions": ["Which products need attention?"]}}
    if match["score"] is None:
        text = f"{product_id} cannot be scored: " + "; ".join(match["missing_evidence"]) + "."
        rows = []
    else:
        lines = [f"  - {SIGNAL_NAMES[c['signal']]}: {fmt_value(c['value'])}{SIGNAL_UNITS[c['signal']]} -> signal score {c['normalized']} "
                 f"x weight {c['weight']} = {fmt_value(c['points'])} points" for c in match["contributions"]]
        parts = [f"{product_id} has a health score of {fmt_value(match['score'])}/100 "
                 f"(evidence available: {int(match['evidence_coverage'] * 100)}% of the weights).",
                 *lines,
                 f"Flagged because: {_reasons(match)}.",
                 f"Recommendation: {match['recommendation']}."]
        if match["missing_evidence"]:
            parts.append("Missing evidence: " + "; ".join(match["missing_evidence"]) + ".")
        parts.append(f"Based on {fmt_value(catalog.row_count('orders'))} orders over {result['meta']['period_days']} days.")
        text = "\n".join(parts)
        rows = [{"signal": SIGNAL_NAMES[c["signal"]], "value": c["value"], "signal_score": c["normalized"],
                 "weight": c["weight"], "points": c["points"]} for c in match["contributions"]]
    e = _evidence(catalog, result, len(rows), [])
    return {"status": "ok", "answer": {"text": text, "headline": text.split("\n")[0], "columns": list(rows[0]) if rows else [],
                                       "rows": rows, "assumptions": []},
            "plan": {"intent": "insight", "type": "explain", "product_id": product_id}, "evidence": e, "warnings": []}
