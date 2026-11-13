"""
RETAIL REPORT - shows everything from Step 5 on the retail datasets.

Run:  python -m app.retail.report data/retail
"""
import sys
from pathlib import Path

import pandas as pd

from app.retail.joins import discover_joins, join_orders_wide
from app.retail.metrics import product_stock_position, sales_metrics, stock_metrics
from app.retail.scoring import product_scores


def load_tables(folder):
    tables = {}
    for path in sorted(Path(folder).glob("*_retail.csv")):
        tables[path.stem.replace("_retail", "")] = pd.read_csv(path, dtype=str, keep_default_na=False)
    return tables


def money(x):
    return "n/a" if x is None else f"Rs {x:,.0f}"


def main(folder):
    t = load_tables(folder)
    print("=== 1. JOINS FOUND (and whether they are safe) ===")
    for j in discover_joins(t):
        flag = "safe  " if j["safe"] else "REFUSE"
        print(f"  [{flag}] {j['left']}.{j['key']} -> {j['right']}  ({j['cardinality']})")

    wide, reports, skipped = join_orders_wide(t)
    print("\n=== 2. JOIN CHECKS (orders as the base table) ===")
    for r in reports:
        print(f"  orders -> {r['right']:<10} rows {r['left_rows']} -> {r['joined_rows']} "
              f"| unmatched {r['orphan_rows']} {r['orphan_key_sample']} | blank keys {r['blank_key_rows']}"
              f"{' | personal columns left out: ' + str(r['excluded_pii_columns']) if r['excluded_pii_columns'] else ''}")

    rows, meta = sales_metrics(wide)
    s = rows[0]
    print("\n=== 3. SALES (all orders) ===")
    print(f"  revenue {money(s['revenue'])} | orders {s['orders']} | AOV {money(s['aov'])} "
          f"| returned {s['returned_orders']} ({s['return_rate_pct']}%) | rows read {meta['rows_considered']}")

    for col in ("category", "region"):
        if col in wide.columns:
            rows, _ = sales_metrics(wide, [col])
            print(f"\n  by {col}:")
            for r in rows:
                print(f"    {r[col]:<12} revenue {money(r['revenue']):>13} | orders {r['orders']:>4} "
                      f"| AOV {money(r['aov']):>8} | return rate {r['return_rate_pct']}%")

    if "inventory" in t:
        st, smeta = stock_metrics(t["inventory"])
        print(f"\n=== 4. STOCK === stock-out rate {st['stock_out_rate_pct']}% | low-stock rate "
              f"{st.get('low_stock_rate_pct')}% | units {st['total_units_in_stock']} | left out: {smeta['excluded']}")

    if {"products", "inventory"} <= set(t):
        result = product_scores(t["orders"], t["products"], t["inventory"])
        m = result["meta"]
        print(f"\n=== 5. PRODUCT HEALTH SCORE === {m['products_scored']}/{m['products_total']} scored "
              f"over {m['period_days']} days; order product ids not in products: {m['order_product_ids_not_in_products']}")
        for sc in result["scores"][:5]:
            print(f"\n  {sc['product_id']}  score {sc['score']}  (evidence {int(sc['evidence_coverage'] * 100)}%)"
                  f"  -> {sc['recommendation']}")
            for c in sc["contributions"]:
                print(f"      {c['signal']:<17} value {c['value']:<7} score {c['normalized']:<5} "
                      f"x weight {c['weight']} = {c['points']} pts")
            if sc["reasons"]:
                print(f"      flagged because: {'; '.join(sc['reasons'])}")
            if sc["missing_evidence"]:
                print(f"      missing evidence: {'; '.join(sc['missing_evidence'])}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/retail")
