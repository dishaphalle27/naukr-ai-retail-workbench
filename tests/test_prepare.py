from pathlib import Path

import app.prepare as prepare_module
from app.chat.catalog import Catalog

ORDERS = ("order_id,order_date,customer_id,product_id,store_id,quantity,revenue,returned\n"
          "O1,2026-01-05,C1,P1,S1,2,Rs. 200,Yes\nO2,05/01/2026,C2,P1,S2,1,100,no\nO2,05/01/2026,C2,P1,S2,1,100,no\n")


def test_prepare_cleans_normalizes_and_the_chat_can_load_the_result(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "orders.csv").write_text(ORDERS)
    code = prepare_module.main([str(raw), "--clean-dir", str(tmp_path / "c"), "--retail-dir", str(tmp_path / "r")])
    out = capsys.readouterr().out
    assert code == 0 and "All checks passed" in out and "python -m app.chat.cli" in out
    assert Catalog.from_folder(tmp_path / "r").row_count("orders") == 2          # the duplicate was removed


def test_prepare_reports_problems_with_a_nonzero_exit_code(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "bad.csv").write_text("")
    assert prepare_module.main([str(raw), "--clean-dir", str(tmp_path / "c"), "--retail-dir", str(tmp_path / "r")]) == 2
    assert prepare_module.main([str(tmp_path / "missing")]) == 2
    assert "not found" in capsys.readouterr().out.lower()
