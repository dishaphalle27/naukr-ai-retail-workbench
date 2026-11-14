"""Retail tests: normalization decisions, safe joins, metric arithmetic, score explanations."""
import json
from pathlib import Path

import pandas as pd
import pytest

from app.cleaning.pipeline import clean_folder
from app.errors import DataLoadError
from app.retail.joins import discover_joins, join_orders_wide, safe_join
from app.retail.metrics import product_stock_position, sales_metrics, stock_metrics
from app.retail.normalize import normalize_dataframe, normalize_file, normalize_folder
from app.retail.scoring import WEIGHTS, product_scores

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"
needs_sample = pytest.mark.skipif(not (SAMPLE / "planted_issues.json").exists(),
                                  reason="run scripts/generate_messy_data.py first")


def T(**cols):
    return pd.DataFrame({k: [str(v) for v in vals] for k, vals in cols.items()})


# ------------------------------------------------------------ normalization
def test_known_aliases_are_applied_with_confidence():
    df = T(category=["Tee", "Jackets", "Footwear", "T-Shirt"])
    out, decs = normalize_dataframe("p", df)
    assert out["category"].tolist() == ["T-Shirt", "Jacket", "Shoes", "T-Shirt"]
    assert all(d["status"] == "applied" and d["confidence"] >= 0.85 and d["source"] == "code" for d in decs)


def test_low_confidence_alias_is_left_alone_and_sent_for_review():
    out, decs = normalize_dataframe("p", T(category=["Denim", "Jeans"]))
    assert out["category"].tolist() == ["Denim", "Jeans"]          # NOT changed
    d = decs[0]
    assert d["status"] == "needs_review" and d["to"] == "Jeans" and d["confidence"] < 0.85


def test_misspelling_is_only_suggested_never_applied():
    out, decs = normalize_dataframe("p", T(category=["Jaket", "Dress"]))
    assert out["category"][0] == "Jaket"
    assert decs[0]["rule"] == "spelling_suggestion" and decs[0]["status"] == "needs_review"
    assert decs[0]["to"] == "Jacket"


def test_unknown_category_is_flagged_without_a_guess():
    out, decs = normalize_dataframe("p", T(category=["Spaceship"]))
    assert out["category"][0] == "Spaceship" and decs[0]["to"] is None and decs[0]["status"] == "needs_review"


def test_returned_flags_become_y_n_and_empty_stays_empty():
    out, _ = normalize_dataframe("o", T(returned=["Yes", "no", "1", "0", "", "Y", "N"]))
    assert out["returned"].tolist() == ["Y", "N", "Y", "N", "", "Y", "N"]


def test_open_list_city_unknown_values_untouched():
    out, decs = normalize_dataframe("c", T(city=["Bangalore", "Bombay", "Nashik", "pune"]))
    assert out["city"].tolist() == ["Bengaluru", "Mumbai", "Nashik", "Pune"]
    assert not [d for d in decs if d["from"] == "Nashik"]


def test_rules_follow_column_names_not_file_names():
    out, _ = normalize_dataframe("anything_at_all", T(product_category=["Tee"], notes=["Tee"]))
    assert out["product_category"][0] == "T-Shirt" and out["notes"][0] == "Tee"


def test_normalization_is_idempotent_and_keeps_shape(tmp_path):
    f = tmp_path / "products_clean.csv"
    T(id=range(6), category=["Tee", "Denim", "Jackets", "T-Shirt", "Jeans", "Shoes"]).to_csv(f, index=False)
    r = normalize_file(f, tmp_path / "out")
    v = r["validation"]
    assert v["passed"] and all(v["checks"].values())
    assert r["dataset"]["version"].startswith("products_retail@")


# ------------------------------------------------------------------- joins
def test_join_keeps_row_count_and_reports_unmatched_rows():
    orders = T(order_id=range(5), product_id=["P1", "P2", "P1", "P999", ""])
    products = T(product_id=["P1", "P2"], category=["A", "B"])
    joined, rep = safe_join(orders, products, "product_id", "product_id", "orders", "products")
    assert len(joined) == 5 and rep["rows_match"]
    assert rep["orphan_rows"] == 1 and rep["orphan_key_sample"] == ["P999"] and rep["blank_key_rows"] == 1
    assert rep["status"] == "ok_with_unmatched_rows"


def test_join_refused_when_right_key_repeats():
    orders = T(order_id=range(3), product_id=["P1", "P1", "P2"])
    products = T(product_id=["P1", "P1", "P2"], category=["A", "A2", "B"])
    with pytest.raises(DataLoadError) as err:
        safe_join(orders, products, "product_id", "product_id", "orders", "products")
    assert err.value.code == "FAN_OUT_RISK" and err.value.details["duplicate_key_values"] == 1


def test_fan_out_would_really_inflate_revenue_without_the_check():
    orders = T(order_id=range(2), product_id=["P1", "P2"], revenue=[100, 200])
    products = T(product_id=["P1", "P1", "P2"], category=["A", "A2", "B"])
    naive = orders.merge(products, on="product_id")
    assert len(naive) == 3                       # what a careless join would do: rows copied
    with pytest.raises(DataLoadError):
        safe_join(orders, products, "product_id", "product_id")


def test_missing_join_key_gives_structured_error():
    with pytest.raises(DataLoadError) as err:
        safe_join(T(a=[1]), T(b=[1]), "a", "zzz")
    assert err.value.code == "JOIN_KEY_MISSING"


def test_many_to_many_is_discovered_as_unsafe():
    tables = {"orders": T(order_id=range(4), product_id=["P1", "P1", "P2", "P2"]),
              "inventory": T(product_id=["P1", "P1", "P2", "P2"], stock=[1, 2, 3, 4]),
              "products": T(product_id=["P1", "P2"], price=[1, 2])}
    found = {(j["left"], j["right"], j["cardinality"]): j["safe"] for j in discover_joins(tables)}
    assert found[("inventory", "orders", "many_to_many")] is False
    assert found[("orders", "products", "many_to_one")] is True


def test_wide_table_never_contains_personal_columns_and_keeps_rows():
    tables = {"orders": T(order_id=range(3), customer_id=["C1", "C2", "C1"], revenue=[1, 2, 3]),
              "customers": T(customer_id=["C1", "C2"], name=["Asha", "Ravi"], email=["a@x", "b@x"], city=["P", "Q"])}
    wide, reports, _ = join_orders_wide(tables)
    assert len(wide) == 3 and "name" not in wide.columns and "email" not in wide.columns
    assert reports[0]["excluded_pii_columns"] == ["name", "email"]


def test_same_column_name_in_two_tables_is_renamed_not_overwritten():
    tables = {"orders": T(order_id=range(4), store_id=["S1", "S2", "S1", "S2"], customer_id=["C1", "C2", "C1", "C2"]),
              "stores": T(store_id=["S1", "S2"], city=["Pune", "Delhi"]),
              "customers": T(customer_id=["C1", "C2"], city=["Mumbai", "Mumbai"])}
    wide, _, _ = join_orders_wide(tables)
    assert len(wide) == 4
    assert {"city", "stores_city"} <= set(wide.columns)          # both cities kept, nothing overwritten
    assert set(wide["city"]) == {"Mumbai"} and set(wide["stores_city"]) == {"Pune", "Delhi"}


# ----------------------------------------------------------------- metrics
def orders_df():
    return T(order_id=["O1", "O2", "O3", "O4"], revenue=[100, 300, 200, 400], quantity=[1, 3, 2, 4],
             returned=["N", "Y", "N", "Y"], category=["A", "A", "B", "B"])


def test_sales_numbers_match_hand_calculation():
    rows, meta = sales_metrics(orders_df())
    r = rows[0]
    assert r["revenue"] == 1000 and r["orders"] == 4 and r["units"] == 10
    assert r["aov"] == 250 and r["returned_orders"] == 2 and r["return_rate_pct"] == 50.0
    assert r["returned_revenue"] == 700
    assert meta["rows_considered"] == 4 and "aov" in meta["formulas"]


def test_group_by_sums_back_to_total():
    rows, _ = sales_metrics(orders_df(), ["category"])
    assert {r["category"]: r["revenue"] for r in rows} == {"A": 400, "B": 600}
    assert sum(r["revenue"] for r in rows) == 1000


def test_unmatched_rows_get_their_own_group_so_revenue_is_not_lost():
    df = orders_df()
    df.loc[3, "category"] = ""
    rows, _ = sales_metrics(df, ["category"])
    assert "(no match)" in {r["category"] for r in rows} and sum(r["revenue"] for r in rows) == 1000


def test_non_numbers_are_excluded_and_counted():
    df = orders_df()
    df.loc[0, "revenue"] = "oops"
    rows, meta = sales_metrics(df)
    assert rows[0]["revenue"] == 900 and meta["excluded"]["revenue_not_a_number"] == 1


def test_missing_column_gives_structured_error_not_a_crash():
    with pytest.raises(DataLoadError) as err:
        sales_metrics(T(order_id=[1]))
    assert err.value.code == "MISSING_COLUMN"


def test_stock_metrics_exclude_negative_and_missing():
    inv = T(stock_on_hand=[0, 5, 10, -2, "", 0], reorder_level=[10] * 6)
    out, meta = stock_metrics(inv)
    assert out["rows_with_valid_stock"] == 4 and out["stock_out_rate_pct"] == 50.0
    assert meta["excluded"] == {"missing_stock": 1, "negative_stock_treated_as_unreliable": 1}


def test_stock_cover_and_sell_through():
    orders = T(order_id=range(2), product_id=["P1", "P1"], quantity=[5, 5],
               order_date=["2026-01-01", "2026-01-10"])
    inv = T(product_id=["P1", "P1"], stock_on_hand=[10, 20])
    pos, meta = product_stock_position(orders, inv)
    assert meta["period_days"] == 10
    p = pos["P1"]
    assert p["units_sold"] == 10 and p["stock_units"] == 30
    assert p["sell_through_pct"] == 25.0 and p["daily_velocity"] == 1.0 and p["stock_cover_days"] == 30.0


# ------------------------------------------------------------------ scoring
def score_fixture(cost="40"):
    products = T(product_id=["P1", "P2"], price=[100, 100], cost=[cost, 40])
    dates = ["2026-01-01"] * 10 + ["2026-01-30"] * 10
    orders = T(order_id=range(20), product_id=["P1"] * 10 + ["P2"] * 10, quantity=[1] * 20,
               returned=["Y"] * 3 + ["N"] * 7 + ["N"] * 10, order_date=dates)
    inventory = T(product_id=["P1", "P2"], stock_on_hand=[2, 20])
    return orders, products, inventory


def test_score_contributions_add_up_and_follow_the_formula():
    res = product_scores(*score_fixture())
    p1 = next(s for s in res["scores"] if s["product_id"] == "P1")
    assert p1["signals"]["margin_pct"] == 60.0 and p1["signals"]["return_rate_pct"] == 30.0
    assert round(sum(c["points"] for c in p1["contributions"]), 1) == p1["score"]
    assert {c["weight"] for c in p1["contributions"]} <= set(WEIGHTS.values())
    assert p1["evidence_coverage"] == 1.0


def test_score_explains_why_a_product_is_flagged():
    res = product_scores(*score_fixture())
    p1 = next(s for s in res["scores"] if s["product_id"] == "P1")
    assert any("return rate" in r for r in p1["reasons"])
    assert "Investigate returns" in p1["recommendation"] or "Restock" in p1["recommendation"]
    assert res["scores"][0]["product_id"] == "P1"            # worst first


def test_missing_cost_is_reported_and_weights_are_rescaled():
    res = product_scores(*score_fixture(cost=""))
    p1 = next(s for s in res["scores"] if s["product_id"] == "P1")
    assert "margin_pct" not in p1["signals"]
    assert any("margin" in m for m in p1["missing_evidence"])
    assert p1["evidence_coverage"] == 0.7
    assert 0 <= p1["score"] <= 100 and round(sum(c["points"] for c in p1["contributions"]), 1) == p1["score"]


def test_few_orders_means_no_return_rate_instead_of_a_noisy_one():
    orders, products, inventory = score_fixture()
    res = product_scores(orders.iloc[:12], products, inventory)       # P2 has only 2 orders
    p2 = next(s for s in res["scores"] if s["product_id"] == "P2")
    assert "return_rate_pct" not in p2["signals"]
    assert any("return rate not scored" in m for m in p2["missing_evidence"])


def test_same_input_gives_identical_scores():
    a, b = product_scores(*score_fixture()), product_scores(*score_fixture())
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ----------------------------------------- real sample data vs the answer key
@pytest.fixture(scope="module")
def retail_tables(tmp_path_factory):
    root = tmp_path_factory.mktemp("retail")
    clean_folder(SAMPLE, root / "clean")
    normalize_folder(root / "clean", root / "retail")
    return {p.stem.replace("_retail", ""): pd.read_csv(p, dtype=str, keep_default_na=False)
            for p in (root / "retail").glob("*_retail.csv")}


@needs_sample
def test_pipeline_totals_match_the_answer_key(retail_tables):
    key = json.loads((SAMPLE / "planted_issues.json").read_text())
    wide, reports, _ = join_orders_wide(retail_tables)
    rows, _ = sales_metrics(wide)
    assert len(wide) == key["true_row_counts"]["orders"]                       # no fan-out, no loss
    assert rows[0]["revenue"] == key["true_totals"]["total_revenue_all_orders"]
    assert rows[0]["returned_orders"] == key["true_totals"]["returned_orders"]
    by_cat, _ = sales_metrics(wide, ["category"])
    assert sum(r["revenue"] for r in by_cat) == rows[0]["revenue"]


@needs_sample
def test_planted_orphans_are_reported_not_hidden(retail_tables):
    key = json.loads((SAMPLE / "planted_issues.json").read_text())["planted_issues"]
    _, reports, _ = join_orders_wide(retail_tables)
    products = next(r for r in reports if r["right"] == "products")
    assert products["orphan_rows"] == key["orphan_product_id"] and products["orphan_key_sample"] == ["P999"]
    customers = next(r for r in reports if r["right"] == "customers")
    assert customers["blank_key_rows"] == key["missing_customer_id"]
