import json

import app.prepare as prepare_module
import app.runs as runs_module
from app.chat.catalog import Catalog

ORDERS = ("order_id,order_date,customer_id,product_id,store_id,quantity,revenue,returned\n"
          "O1,2026-01-05,C1,P1,S1,2,Rs. 200,Yes\nO2,05/01/2026,C2,P1,S2,1,100,no\nO2,05/01/2026,C2,P1,S2,1,100,no\n")


def raw(tmp_path, **files):
    folder = tmp_path / "raw"
    folder.mkdir(exist_ok=True)
    for name, text in files.items():
        (folder / f"{name}.csv").write_text(text)
    return folder


def args(tmp_path, folder, *extra):
    return [str(folder), "--clean-dir", str(tmp_path / "c"), "--retail-dir", str(tmp_path / "r"),
            "--runs-dir", str(tmp_path / "runs"), *extra]


def test_prepare_creates_a_run_and_the_chat_can_load_the_result(tmp_path, capsys):
    code = prepare_module.main(args(tmp_path, raw(tmp_path, orders=ORDERS)))
    out = capsys.readouterr().out
    assert code == 0 and "status: succeeded" in out and "python -m app.chat.cli" in out
    assert Catalog.from_folder(tmp_path / "r").row_count("orders") == 2          # the duplicate was removed


def test_a_run_keeps_the_original_the_cleaned_file_status_and_hashes(tmp_path):
    folder = raw(tmp_path, orders=ORDERS)
    run = runs_module.create_run(folder, tmp_path / "runs")
    root = tmp_path / "runs" / run["run_id"]
    assert (root / "original" / "orders.csv").read_bytes() == (folder / "orders.csv").read_bytes()   # byte for byte
    assert (root / "cleaned" / "orders_clean.csv").exists() and (root / "retail" / "orders_retail.csv").exists()
    meta = json.loads((root / "run.json").read_text())
    assert meta["status"] == "succeeded" and meta["inputs"][0]["file"] == "orders.csv"
    assert meta["datasets"]["orders"]["rows_before"] == 3 and meta["datasets"]["orders"]["rows_after"] == 2
    assert set(meta["artifacts"]) >= {"original/orders.csv", "cleaned/orders_clean.csv", "retail/orders_retail.csv"}
    assert (tmp_path / "runs" / "LATEST").read_text() == run["run_id"]


def test_same_input_reuses_the_run_and_a_changed_byte_makes_a_new_one(tmp_path):
    folder = raw(tmp_path, orders=ORDERS)
    first = runs_module.create_run(folder, tmp_path / "runs")
    marker = tmp_path / "runs" / first["run_id"] / "run.json"
    before = marker.read_bytes()
    again = runs_module.create_run(folder, tmp_path / "runs")
    assert again["reused"] is True and again["run_id"] == first["run_id"] and marker.read_bytes() == before
    (folder / "orders.csv").write_text(ORDERS.replace("Rs. 200", "Rs. 201"))
    changed = runs_module.create_run(folder, tmp_path / "runs")
    assert changed["run_id"] != first["run_id"] and changed["reused"] is False
    assert len(runs_module.list_runs(tmp_path / "runs")) == 2


def test_a_bad_file_gives_a_partial_run_with_the_structured_error(tmp_path):
    run = runs_module.create_run(raw(tmp_path, orders=ORDERS, empty=""), tmp_path / "runs")
    assert run["status"] == "partial" and run["errors"][0]["code"] == "EMPTY_FILE" and run["errors"][0]["dataset"] == "empty"
    assert "orders" in run["datasets"]


def test_a_crash_is_saved_as_a_failed_run_and_leaves_no_half_written_folder(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk exploded")
    monkeypatch.setattr(runs_module, "normalize_folder", boom)
    run = runs_module.create_run(raw(tmp_path, orders=ORDERS), tmp_path / "runs")
    assert run["status"] == "failed" and run["errors"][0]["code"] == "RUN_CRASHED"
    assert not list((tmp_path / "runs").glob(".tmp-*"))
    assert runs_module.load_run(run["run_id"], tmp_path / "runs")["status"] == "failed"
    monkeypatch.undo()                                          # a failed run is not "reused": it is retried
    assert runs_module.create_run(tmp_path / "raw", tmp_path / "runs")["status"] == "succeeded"


def test_prepare_reports_problems_with_a_nonzero_exit_code(tmp_path, capsys):
    assert prepare_module.main(args(tmp_path, raw(tmp_path, bad=""))) == 2
    assert prepare_module.main(args(tmp_path, tmp_path / "missing")) == 2
    assert "no .csv files" in capsys.readouterr().out.lower()


def test_the_runs_command_lists_and_shows(tmp_path, capsys, monkeypatch):
    run = runs_module.create_run(raw(tmp_path, orders=ORDERS), tmp_path / "runs")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "runs").rename(tmp_path / "data" / "runs")
    assert runs_module.main(["list"]) == 0 and run["run_id"] in capsys.readouterr().out
    assert runs_module.main(["show", run["run_id"]]) == 0 and '"status": "succeeded"' in capsys.readouterr().out
    assert runs_module.main(["show", "run-nope"]) == 2
