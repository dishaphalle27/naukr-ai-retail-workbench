"""Tests for the 'proof' features: scorecard against ground truth, replay verification, receipts, original artifacts."""
import json
from pathlib import Path

import pandas as pd
import pytest

import app.scorecard as scorecard
from app.chat.answers import format_receipt
from app.chat.assistant import Assistant
from app.chat.catalog import Catalog
from app.chat.store import ChatStore
from app.chat.verify import verify_session
from app.config import Settings
from tests.conftest import _write
import app.evaluate as ev

ROOT = Path(__file__).resolve().parent.parent
needs_sample = pytest.mark.skipif(not ((ROOT / "data" / "sample_truth").exists() and (ROOT / "data" / "sample").exists()),
                                  reason="run scripts/generate_messy_data.py first")


def T(**cols):
    return pd.DataFrame({k: [str(v) for v in vals] for k, vals in cols.items()})


# ------------------------------------------------------------------ scorecard
def test_scorecard_classifies_every_kind_of_cell():
    truth = T(id=["1", "2", "3", "4", "5"], v=["a", "b", "c", "d", "e"])
    messy = T(id=["1", "2", "3", "4", "5"], v=["A", "B", "c", "D", "e"])        # 1,2,4 corrupted
    out = T(id=["1", "2", "3", "4", "5"], v=["a", "B", "c", "zzz", "BROKEN"])   # 1 repaired, 2 left, 4 wrong fix, 5 damaged
    r = scorecard.score_table(truth, messy, out, ["id"])
    assert (r["corrupted"], r["repaired"], r["left_as_is"], r["wrong_fix"], r["damaged"]) == (3, 1, 1, 1, 1)
    assert r["left_by_column"] == {"v": 1} and r["wrong_by_column"] == {"v": 1}
    assert r["changed"] == 3 and r["changed_correct"] == 1                       # precision is 1 of 3


def test_scorecard_counts_lost_and_invented_rows():
    truth = T(id=["1", "2", "3"], v=["a", "b", "c"])
    messy = T(id=["1", "2", "3", "3"], v=["a", "b", "c", "c"])
    lost = scorecard.score_table(truth, messy, T(id=["1", "2"], v=["a", "b"]), ["id"])
    invented = scorecard.score_table(truth, messy, T(id=["1", "2", "3", "9"], v=["a", "b", "c", "x"]), ["id"])
    assert lost["rows_lost"] == 1 and invented["rows_invented"] == 1


def test_scorecard_uses_the_first_duplicate_like_the_cleaner_does():
    truth = T(id=["1"], v=["a"])
    messy = T(id=["1", "1"], v=["a", "a"])
    assert scorecard.score_table(truth, messy, T(id=["1"], v=["a"]), ["id"])["corrupted"] == 0


@needs_sample
def test_the_real_pipeline_scores_perfectly_on_the_sample(tmp_path):
    result = scorecard.score(ROOT / "data" / "sample", ROOT / "data" / "sample_truth", tmp_path)
    t = result["totals"]
    assert t["passed"] and t["wrong_fix"] == 0 and t["damaged"] == 0 and t["rows_lost"] == 0 and t["rows_invented"] == 0
    assert t["silent"] == 0 and t["repair_precision"] == 1.0
    assert t["corrupted"] > 1400 and t["repair_rate"] > 0.97
    # every cell left alone is explained by a flag a person can see
    for name, table in result["tables"].items():
        for col, why in table["left_explained"].items():
            assert why, f"{name}.{col} was left as-is without any report flagging it"
    assert "RESULT: PASS" in scorecard.format_report(result)


@needs_sample
def test_the_scorecard_fails_loudly_when_a_cleaner_breaks_something(tmp_path, monkeypatch, capsys):
    from app.cleaning import cleaner
    real = cleaner.ACTIONS["strip_currency_and_commas"]

    def sabotage(df, step):
        out, info = real(df, step)
        col = step["columns"][0]
        out.loc[out.index[0], col] = "999999"                   # a silent wrong 'fix'
        return out, info
    monkeypatch.setitem(cleaner.ACTIONS, "strip_currency_and_commas", sabotage)
    result = scorecard.score(ROOT / "data" / "sample", ROOT / "data" / "sample_truth", tmp_path)
    assert result["totals"]["passed"] is False and (result["totals"]["wrong_fix"] + result["totals"]["damaged"]) > 0
    assert scorecard.main(["--messy", str(ROOT / "data" / "sample"), "--truth", str(ROOT / "data" / "sample_truth")]) == 2


def test_scorecard_needs_the_sample_data(tmp_path, capsys):
    assert scorecard.main(["--messy", str(tmp_path / "x"), "--truth", str(tmp_path / "y")]) == 2


# -------------------------------------------------------------------- replay
def test_a_saved_session_replays_exactly(shop):
    store = ChatStore()
    bot = Assistant(shop, store)
    bot.ask("s", "only stores in the West region")
    bot.ask("s", "among those, which categories had the highest return rate?")
    bot.ask("s", "profit by region")                         # refused: nothing to replay
    bot.ask("s", "Which products need attention?")           # insight: not a replayable plan
    results = verify_session(shop, store, "s")
    assert [r["verdict"] for r in results] == ["reproduced", "reproduced", "skipped", "skipped"]


def test_replay_notices_when_the_data_changed(tmp_path):
    def build(revenue):
        folder = tmp_path / f"d{revenue}"
        folder.mkdir(exist_ok=True)
        _write(folder, "orders", [{"order_id": "O1", "order_date": "2026-01-01", "product_id": "P1", "store_id": "S1",
                                   "quantity": 1, "revenue": revenue, "returned": "N"}])
        _write(folder, "products", [{"product_id": "P1", "category": "T-Shirt", "price": 1, "cost": 1}])
        return Catalog.from_folder(folder)
    store = ChatStore()
    Assistant(build(100), store).ask("s", "What is the total revenue?")
    same = verify_session(build(100), store, "s")[0]
    changed = verify_session(build(250), store, "s")[0]
    assert same["verdict"] == "reproduced"
    assert changed["verdict"] == "changed" and "data version changed" in changed["detail"]


def test_replay_notices_different_numbers_on_the_same_version(shop, monkeypatch):
    import app.chat.verify as verify_module
    store = ChatStore()
    Assistant(shop, store).ask("s", "What is the total revenue?")
    real = verify_module.run_plan

    def tampered(plan, catalog):
        out = real(plan, catalog)
        out["result"]["rows"][0]["revenue"] += 1
        return out
    monkeypatch.setattr(verify_module, "run_plan", tampered)
    assert verify_session(shop, store, "s")[0]["detail"] == "same data version but different numbers"


# ------------------------------------------------------------------- receipt
def test_receipt_shows_where_the_numbers_came_from(shop):
    reply = Assistant(shop, ChatStore()).ask("s", "Revenue by category in the West region")
    receipt = format_receipt(reply)
    for piece in ("RECEIPT", "orders_wide_retail@", "region = West", "products", "unmatched", "validator", "not by the model"):
        assert piece in receipt
    assert receipt.isascii()                                   # prints on any console
    assert format_receipt({"status": "refused"}) == ""


# ------------------------------------------------------------ evaluator originals
def test_evaluator_keeps_the_original_next_to_the_cleaned_artifact(tmp_path):
    src = tmp_path / "orders.csv"
    src.write_text("order_id,revenue\nO1,Rs. 100\nO2,200\n")
    out = ev.evaluate([{"id": "orig", "input": str(src)}], tmp_path / "art", settings=Settings(), rules_only=True)
    arts = out["cases"][0]["artifacts"]
    assert "orig/original/orders.csv" in arts and "orig/clean/orders_clean.csv" in arts
    assert (tmp_path / "art" / "orig" / "original" / "orders.csv").read_bytes() == src.read_bytes()
