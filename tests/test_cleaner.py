"""Cleaner tests: each one protects a promise (no silent row loss, stable output, safe failure)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from app.cleaning.cleaner import clean_dataframe
from app.cleaning.issues import detect_issues
from app.cleaning.pipeline import clean_file, clean_folder
from app.cleaning.plan import build_plan
from app.cleaning.profiler import load_csv, profile_dataframe
from app.cleaning.validate import validate

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"
needs_sample = pytest.mark.skipif(not (SAMPLE / "planted_issues.json").exists(),
                                  reason="run scripts/generate_messy_data.py first")


def df_of(**cols):
    return pd.DataFrame({k: [str(v) for v in vals] for k, vals in cols.items()})


def run(df, name="t"):
    prof = profile_dataframe(df)
    issues = detect_issues(name, prof)
    plan = build_plan(name, prof, issues)
    cleaned, log = clean_dataframe(name, df, plan)
    return cleaned, log, plan


def entry(log, action):
    return [e for e in log if e["action"] == action]


# ------------------------------------------------------------ small actions
def test_trim_collapses_spaces_and_leaves_other_columns_alone():
    df = df_of(name_x=["  Slim  Jeans ", "Tee", "Dress", "Shoes", "Coat", "Hat", "Bag", "Belt"],
               code=["A1"] * 8)
    cleaned, _, _ = run(df)
    assert cleaned["name_x"][0] == "Slim Jeans"
    assert cleaned["code"].equals(df["code"])


def test_money_symbols_removed_but_text_left_alone():
    vals = ["100", "Rs. 1,299", "₹ 2,50,000", "250", "400", "500", "$75", "Rs. abc", "60"]
    cleaned, _, _ = run(df_of(price=vals))
    assert cleaned["price"].tolist()[:3] == ["100", "1299", "250000"]
    assert cleaned["price"][6] == "75"
    assert cleaned["price"][7] == "Rs. abc"          # not a number -> untouched


def test_number_words_become_digits():
    cleaned, _, _ = run(df_of(id=range(10), qty=["1", "2", "two", "3", "One", "1", "2", "3", "1", "2"]))
    assert cleaned["qty"].tolist() == ["1", "2", "2", "3", "1", "1", "2", "3", "1", "2"]


def dated(vals):
    return df_of(id=range(len(vals)), d=vals)      # unique id so rows are not duplicates


def test_day_first_dates_converted_and_ambiguous_rows_listed():
    vals = ["2026-01-01"] * 4 + ["13/01/2026", "20/02/2026", "05/01/2026", "07/03/2026"]
    cleaned, log, _ = run(dated(vals))
    assert cleaned["d"].tolist()[4:] == ["2026-01-13", "2026-02-20", "2026-01-05", "2026-03-07"]
    d = entry(log, "convert_date_to_iso")[0]["details"]
    assert d["ambiguous_values_converted"] == 2
    assert d["ambiguous_source_rows"] == [7, 8]           # 1-based data rows in the original file


def test_month_first_dates():
    vals = ["2026-01-01"] * 4 + ["01-13-2026", "02-20-2026", "01-05-2026"]
    cleaned, _, _ = run(dated(vals))
    assert cleaned["d"].tolist()[4:] == ["2026-01-13", "2026-02-20", "2026-01-05"]


def test_dates_with_no_proof_are_left_unchanged():
    vals = ["2026-01-01"] * 4 + ["05/01/2026", "07/03/2026", "02/04/2026"]
    cleaned, log, _ = run(dated(vals))
    assert cleaned["d"].tolist() == vals                  # nothing guessed
    assert entry(log, "flag_for_review")[0]["status"] == "needs_review"


def test_spelling_variants_merge_to_most_common():
    cleaned, _, _ = run(df_of(region=["West", "West", "west ", "WEST", "East", "East", "North", "North"]))
    assert set(cleaned["region"]) == {"West", "East", "North"}


def test_mixed_sku_formats_become_one_format():
    skus = ["SKU-0001", "SKU-0002", "SKU-0003", "SKU-0004", "SKU-0005", "sku0006", "SKU_0007", "sku-0008"]
    cleaned, _, _ = run(df_of(sku=skus))
    assert cleaned["sku"].tolist() == [f"SKU-000{i}" for i in range(1, 9)]


def test_email_lowercase_and_trim():
    emails = [f"A{i}@EXAMPLE.COM" for i in range(7)] + ["  b@x.com"]
    cleaned, log, _ = run(df_of(email=emails))
    assert cleaned["email"][0] == "a0@example.com" and cleaned["email"][7] == "b@x.com"
    assert all("samples" not in e["details"] for e in log)      # no personal data in the log


# ------------------------------------------------------------ safety promises
def test_duplicates_removed_first_kept_and_rows_listed():
    ids = [f"A{i}" for i in range(10)] + ["A3", "A4"]
    df = df_of(id=ids, v=list(range(10)) + [3, 4])
    cleaned, log, _ = run(df)
    assert len(cleaned) == 10 and list(cleaned.index) == list(range(10))
    assert entry(log, "drop_exact_duplicates")[0]["details"]["removed_source_rows"] == [11, 12]


def test_needs_review_and_skipped_steps_never_change_data():
    df = df_of(stock=["5", "10", "-3", "", "8", "12", "7", "9"])
    cleaned, log, _ = run(df)
    assert cleaned.equals(df)
    assert {e["status"] for e in log} == {"needs_review", "skipped"}


def test_unknown_action_is_refused():
    df = df_of(a=["1", "2"])
    bad = [{"step_id": "S", "order": 1, "action": "run_python", "columns": ["a"], "status": "planned",
            "params": {}, "issue_ids": [], "expected_rows_removed": {"min": 0, "max": 0}}]
    cleaned, log = clean_dataframe("t", df, bad)
    assert log[0]["status"] == "failed" and "not allowed" in log[0]["error"]
    assert cleaned.equals(df)


def test_one_failed_step_does_not_stop_the_others():
    df = df_of(a=["  x ", "y", "z", "w", "v", "u", "t", "s"])
    plan = build_plan("t", profile_dataframe(df))
    broken = dict(plan[0], step_id="STEP:broken", columns=["no_such_column"])
    cleaned, log = clean_dataframe("t", df, [broken] + plan)
    assert log[0]["status"] == "failed"
    assert cleaned["a"][0] == "x"                          # the real step still ran


def test_input_dataframe_is_never_modified():
    df = df_of(a=["  x ", "y", "y", "z", "w", "v", "u", "t"])
    before = df.copy(deep=True)
    run(df)
    pd.testing.assert_frame_equal(df, before)


# ------------------------------------------------------------- validation
def test_validation_passes_on_a_good_clean():
    ids = [f"A{i}" for i in range(10)] + ["A3"]
    df = df_of(id=ids, v=[f" {i} " for i in range(10)] + [" 3 "])
    prof = profile_dataframe(df)
    plan = build_plan("t", prof)
    cleaned, log = clean_dataframe("t", df, plan)
    after = detect_issues("t", profile_dataframe(cleaned))
    v = validate(df, cleaned, plan, log, after, idempotent=True)
    assert v["passed"], v


def test_validation_catches_a_silently_lost_row():
    df = df_of(a=["1", "2", "3", "4"])
    cleaned = df.drop(index=2)                  # a buggy cleaner dropped a row nobody planned
    v = validate(df, cleaned, [], [], [], idempotent=True)
    assert v["checks"]["rows_as_planned"] is False and v["passed"] is False


def test_validation_catches_schema_change_and_hidden_edits():
    df = df_of(a=["1", "2", "3"], b=["x", "y", "z"])
    renamed = df.rename(columns={"b": "c"})
    assert validate(df, renamed, [], [], [], True)["checks"]["schema_unchanged"] is False
    edited = df.copy()
    edited.loc[0, "b"] = "SNEAKY"                # a column no step claimed to touch
    assert validate(df, edited, [], [], [], True)["checks"]["untouched_columns_identical"] is False


# ------------------------------------------------- end to end on sample data
@needs_sample
def test_cleaning_twice_changes_nothing_on_every_sample_file(tmp_path):
    for path in sorted(SAMPLE.glob("*.csv")):
        r = clean_file(path, tmp_path / "first")
        assert r["validation"]["checks"]["idempotent"], path.name
        again = clean_file(Path(r["artifacts"]["cleaned"]), tmp_path / "second", dataset_name=path.stem)
        assert again["artifacts"]["cleaned_sha256"] == r["artifacts"]["cleaned_sha256"], path.name
        assert again["validation"]["rows"]["removed"] == 0


@needs_sample
def test_sample_files_all_pass_validation(tmp_path):
    for r in clean_folder(SAMPLE, tmp_path):
        assert r["validation"]["passed"], (r["dataset"]["name"], r["validation"])


@needs_sample
def test_cleaned_orders_match_the_answer_key(tmp_path):
    key = json.loads((SAMPLE / "planted_issues.json").read_text())
    r = clean_file(SAMPLE / "orders.csv", tmp_path)
    cleaned, _ = load_csv(r["artifacts"]["cleaned"])
    assert len(cleaned) == key["true_row_counts"]["orders"]
    assert pd.to_numeric(cleaned["revenue"]).sum() == key["true_totals"]["total_revenue_all_orders"]
    assert cleaned["order_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    assert pd.to_numeric(cleaned["quantity"]).notna().all()


@needs_sample
def test_same_input_gives_same_file_hash(tmp_path):
    a = clean_file(SAMPLE / "products.csv", tmp_path / "a")
    b = clean_file(SAMPLE / "products.csv", tmp_path / "b")
    assert a["artifacts"]["cleaned_sha256"] == b["artifacts"]["cleaned_sha256"]


def test_bad_file_does_not_abort_the_folder(tmp_path):
    (tmp_path / "good.csv").write_text("id,v\nA1,1\nA2,2\n")
    (tmp_path / "empty.csv").write_text("")
    (tmp_path / "header_only.csv").write_text("a,b\n")
    results = {r["dataset"]["name"]: r for r in clean_folder(tmp_path, tmp_path / "out")}
    assert results["empty"]["error"]["code"] == "EMPTY_FILE"
    assert results["good"]["validation"]["passed"]
    assert results["header_only"]["validation"]["rows"]["after"] == 0
