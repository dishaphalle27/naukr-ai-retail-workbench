"""Evaluator tests: one failure never stops the batch, outputs are complete and repeatable, checks are real."""
import json
from pathlib import Path

import pytest

import app.evaluate as ev
from app.chat.assistant import Assistant
from app.config import Settings

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample"
GOLD = ROOT / "eval" / "cases.json"

ORDERS = ("order_id,order_date,customer_id,product_id,store_id,quantity,revenue,returned\n"
          "O1,2026-01-05,C1,P1,S1,2,200,N\nO2,2026-01-20,C2,P1,S2,1,100,Y\nO3,2026-02-03,C1,P2,S1,3,600,N\nO4,2026-02-10,C3,P2,S2,1,200,Y\n")
PRODUCTS = "product_id,category,price,cost\nP1,T-Shirt,100,40\nP2,Jeans,200,100\n"


@pytest.fixture
def files(tmp_path):
    (tmp_path / "orders.csv").write_text(ORDERS)
    (tmp_path / "products.csv").write_text(PRODUCTS)
    (tmp_path / "empty.csv").write_text("")
    (tmp_path / "no_date.csv").write_text("order_id,revenue,returned\nO1,100,N\nO2,200,Y\n")
    return tmp_path


def run(cases, tmp_path, **kw):
    return ev.evaluate(cases, tmp_path / "out", settings=Settings(), rules_only=True, **kw)


def by_id(out):
    return {c["id"]: c for c in out["cases"]}


# ------------------------------------------------------------ the batch survives
def test_one_failed_case_never_aborts_the_batch(files):
    cases = [{"id": "good1", "input": str(files / "orders.csv")},
             {"id": "empty", "input": str(files / "empty.csv")},
             {"id": "missing", "input": str(files / "nope.csv")},
             {"id": "good2", "datasets": {"orders": str(files / "orders.csv"), "products": str(files / "products.csv")}},
             "not even an object", {"no_id": True}, {"id": "nothing_to_load"}]
    out = run(cases, files)
    assert len(out["cases"]) == 7
    got = by_id(out)
    assert got["good1"]["status"] == "ok" and got["good2"]["status"] == "ok"
    assert got["empty"]["errors"][0]["code"] == "EMPTY_FILE" and got["missing"]["errors"][0]["code"] == "FILE_NOT_FOUND"
    assert got["nothing_to_load"]["errors"][0]["code"] == "NO_DATASETS"
    invalid = [c for c in out["cases"] if c["id"] is None]
    assert len(invalid) == 2 and all(c["error"]["code"] == "INVALID_CASE" for c in invalid)
    assert out["summary"]["cases"] == 7 and out["summary"]["cases_error"] >= 4


def test_one_bad_dataset_inside_a_case_gives_partial_and_the_chat_still_runs(files):
    case = {"id": "mixed", "datasets": {"orders": str(files / "orders.csv"), "products": str(files / "empty.csv")},
            "questions": [{"question": "What is the total revenue?", "expect": {"first_row": {"revenue": 1100}}}]}
    c = by_id(run([case], files))["mixed"]
    assert c["status"] == "partial" and c["errors"][0]["code"] == "EMPTY_FILE"
    assert c["errors"][0]["details"]["dataset"] == "products"
    assert c["chat_evaluation"][0]["passed"] is True


def test_one_failing_question_does_not_stop_the_others(files, monkeypatch):
    real = Assistant.ask

    def flaky(self, session, question):
        if "explode" in question:
            raise RuntimeError("boom")
        return real(self, session, question)
    monkeypatch.setattr(Assistant, "ask", flaky)
    case = {"id": "q", "input": str(files / "orders.csv"), "questions": ["explode please revenue", "What is the total revenue?"]}
    chat = by_id(run([case], files))["q"]["chat_evaluation"]
    assert chat[0]["status"] == "error" and chat[0]["error"]["code"] == "QUESTION_FAILED"
    assert chat[1]["status"] == "ok"


# ---------------------------------------------------------------- output shape
def test_single_dataset_case_has_the_flat_shape_from_the_assessment(files):
    c = by_id(run([{"id": "one", "input": str(files / "orders.csv"), "questions": ["What is the total revenue?"]}], files))["one"]
    for key in ("dataset", "profile_before", "issues", "cleaning_plan", "profile_after", "validation", "chat_evaluation", "artifacts"):
        assert key in c
    assert c["dataset"]["name"] == "orders" and c["dataset"]["version"].startswith("orders_clean@")
    assert c["profile_before"]["row_count"] == 4 and c["validation"]["passed"] is True
    q = c["chat_evaluation"][0]
    assert set(q) >= {"question", "plan", "answer", "trace", "status"} and q["status"] == "ok"
    assert q["answer"]["rows"][0]["revenue"] == 1100 and q["evidence"]["version"]


def test_several_datasets_are_keyed_by_name_with_flat_issue_and_plan_lists(files):
    c = by_id(run([{"id": "two", "datasets": {"orders": str(files / "orders.csv"), "products": str(files / "products.csv")}}], files))["two"]
    assert set(c["dataset"]) == {"orders", "products"} and set(c["validation"]) == {"orders", "products"}
    assert isinstance(c["issues"], list) and isinstance(c["cleaning_plan"], list)
    assert set(c["versions"]) == {"orders", "products"}


def test_header_only_data_gives_a_structured_answer_not_a_crash(files):
    (files / "h.csv").write_text("order_id,revenue,returned\n")
    c = by_id(run([{"id": "h", "input": str(files / "h.csv"), "questions": ["What is the total revenue?"]}], files))["h"]
    assert c["status"] == "ok" and c["chat_evaluation"][0]["status"] in ("ok", "refused", "clarify", "rejected", "error")


def test_missing_date_column_does_not_crash_the_chat(files):
    qs = [{"question": "What is the total revenue?", "expect": {"status": "ok", "first_row": {"revenue": 300}}},
          "Revenue by month", "Revenue in March", "Revenue last month"]
    chat = by_id(run([{"id": "nd", "input": str(files / "no_date.csv"), "questions": qs}], files))["nd"]["chat_evaluation"]
    assert chat[0]["passed"] is True
    assert all(c["status"] != "error" for c in chat)


# ------------------------------------------------------------------ repeatable
def test_two_runs_in_different_folders_give_identical_json(files, tmp_path_factory):
    cases = [{"id": "rep", "datasets": {"orders": str(files / "orders.csv"), "products": str(files / "products.csv")},
              "questions": ["Revenue by category", "How many orders were there?"]}]
    a = ev.evaluate(cases, tmp_path_factory.mktemp("a"), settings=Settings(), rules_only=True)
    b = ev.evaluate(cases, tmp_path_factory.mktemp("b") / "deeper", settings=Settings(), rules_only=True)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_nothing_volatile_is_in_the_output(files):
    text = json.dumps(run([{"id": "v", "input": str(files / "orders.csv"), "questions": ["What is the total revenue?"]}], files))
    for word in ("timing_ms", "elapsed_ms", "created_at"):
        assert word not in text


def test_rerun_removes_stale_files(files):
    case = {"id": "stale", "input": str(files / "orders.csv")}
    run([case], files)
    leftover = files / "out" / "stale" / "old_junk.txt"
    leftover.write_text("x")
    c = by_id(run([case], files))["stale"]
    assert not leftover.exists() and not any("old_junk" in k for k in c["artifacts"])


# ---------------------------------------------------------------------- safety
def test_a_case_id_cannot_escape_the_artifacts_folder(files):
    c = by_id(run([{"id": "../../escape", "input": str(files / "orders.csv")}], files))["../../escape"]
    assert c["status"] == "ok"
    out = (files / "out").resolve()
    assert all((out / k.split("/")[0]).resolve().parent == out for k in c["artifacts"])
    assert not (files.parent / "escape").exists()


def test_unsafe_dataset_names_are_made_safe(files):
    c = by_id(run([{"id": "n", "datasets": {"../evil name!": str(files / "orders.csv")}}], files))["n"]
    assert c["status"] == "ok" and all(".." not in k for k in c["artifacts"])


def test_rules_only_never_uses_a_model_even_if_a_key_is_configured(files, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    cfg = Settings(provider="gemini", api_key="k")
    out = ev.evaluate([{"id": "r", "input": str(files / "orders.csv"), "questions": ["Revenue by region"]}],
                      files / "out", settings=cfg, rules_only=True)
    assert out["cases"][0]["chat_evaluation"][0]["trace"]["planner"] in ("rules", "rules_screen")


# ---------------------------------------------------------- the checks are real
def test_a_wrong_expectation_fails_so_a_pass_means_something(files):
    case = {"id": "chk", "input": str(files / "orders.csv"), "questions": [
        {"question": "What is the total revenue?", "expect": {"status": "ok", "first_row": {"revenue": 1100}}},
        {"question": "What is the total revenue?", "expect": {"first_row": {"revenue": 999999}}},
        {"question": "What is the profit?", "expect": {"status": "ok"}},
        {"question": "What is the profit?", "expect": {"refusal_code": "NO_PROFIT_DATA", "contains_text": ["no profit"]}},
        {"question": "show sales", "expect": {"status": "clarify", "clarification_options_include": ["revenue"]}}]}
    c = by_id(run([case], files))["chk"]
    assert [q["passed"] for q in c["chat_evaluation"]] == [True, False, False, True, True]
    assert c["chat_evaluation"][1]["checks"][0]["detail"]
    s = ev.summarize([c])
    assert s["chat_with_expectations"] == 5 and s["chat_passed"] == 3 and s["chat_accuracy"] == 0.6


def test_expectation_numbers_compare_with_a_small_tolerance():
    reply = {"status": "ok", "answer": {"rows": [{"x": 11.6667}]}}
    assert ev.check_expectations(reply, {"first_row": {"x": 11.67}})[0]["passed"]
    assert not ev.check_expectations(reply, {"first_row": {"x": 11.9}})[0]["passed"]


# --------------------------------------------------------------- command line
def test_command_line_writes_results_and_returns_zero(files, capsys):
    cases = files / "cases.json"
    cases.write_text(json.dumps([{"id": "cli", "input": str(files / "orders.csv"), "questions": ["What is the total revenue?"]}]))
    code = ev.main(["--input", str(cases), "--output", str(files / "res" / "results.json"),
                    "--artifacts-dir", str(files / "art"), "--rules-only"])
    assert code == 0 and "Evaluated 1 case" in capsys.readouterr().out
    data = json.loads((files / "res" / "results.json").read_text())
    assert data["summary"]["cases_ok"] == 1 and data["cases"][0]["id"] == "cli"


@pytest.mark.parametrize("content,code", [("{not json", "INPUT_NOT_JSON"), ('{"cases": 5}', "INPUT_SHAPE"), (None, "INPUT_NOT_FOUND")])
def test_unreadable_input_gives_a_clear_error_and_exit_code_2(files, capsys, content, code):
    path = files / "bad.json"
    if content is not None:
        path.write_text(content)
    assert ev.main(["--input", str(path), "--output", str(files / "r.json")]) == 2
    assert code in capsys.readouterr().out


def test_cases_may_be_a_list_or_an_object_with_a_cases_key(files):
    p = files / "wrapped.json"
    p.write_text(json.dumps({"cases": [{"id": "w", "input": str(files / "orders.csv")}]}))
    assert len(ev.load_cases(p)) == 1


# --------------------------------------- the real sample data and gold questions
@pytest.mark.skipif(not (GOLD.exists() and (ROOT / "data" / "sample_truth").exists() and SAMPLE.exists()),
                    reason="run scripts/generate_messy_data.py and scripts/build_gold.py first")
def test_gold_questions_all_pass_on_the_messy_sample(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)                                    # the gold file uses paths relative to the repo
    cases = json.loads(GOLD.read_text())
    out = ev.evaluate(cases, tmp_path, settings=Settings(), rules_only=True)
    s = out["summary"]
    failed = [(q["question"], [c for c in q["checks"] if not c["passed"]]) for c in out["cases"] for q in c["chat_evaluation"] if q.get("passed") is False]
    assert not failed, failed
    assert s["chat_with_expectations"] >= 25 and s["chat_accuracy"] == 1.0
    assert s["datasets_validated"] == 5 and s["datasets_passed_validation"] == 5


@pytest.mark.skipif(not (ROOT / "eval" / "cases_edge.json").exists(), reason="run scripts/make_edge_cases.py first")
def test_edge_case_pack_never_crashes_and_expected_answers_hold(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    cases = json.loads((ROOT / "eval" / "cases_edge.json").read_text())
    out = ev.evaluate([c for c in cases if not (isinstance(c, dict) and c.get("id") == "edge_120k_rows")], tmp_path,
                      settings=Settings(), rules_only=True)
    assert len(out["cases"]) == len(cases) - (1 if any(isinstance(c, dict) and c.get("id") == "edge_120k_rows" for c in cases) else 0)
    assert out["summary"]["chat_accuracy"] == 1.0
    assert {c["status"] for c in out["cases"]} <= {"ok", "partial", "error"}
