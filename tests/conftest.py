"""Shared test data: a tiny shop whose answers can be calculated by hand."""
import pandas as pd
import pytest

from app.chat.assistant import Assistant
from app.chat.catalog import Catalog
from app.chat.store import ChatStore


def _write(folder, name, rows):
    pd.DataFrame(rows).astype(str).to_csv(folder / f"{name}_retail.csv", index=False)


@pytest.fixture
def shop(tmp_path):
    """
    orders (8): O1..O8, revenue total 1850, returned O2 O4 O8 (3 of 8)
    West = store S1 (O1 O3 O5 O7 O8: revenue 1350, 5 orders, 1 returned)
    East = store S2 (O2 O4 O6: revenue 500)
    P9 in O7 is not in products (unmatched). Delhi is both a customer city and a store city.
    """
    cols = ["order_id", "order_date", "customer_id", "product_id", "store_id", "quantity", "revenue", "returned"]
    rows = [("O1", "2026-01-05", "C1", "P1", "S1", 2, 200, "N"), ("O2", "2026-01-20", "C2", "P1", "S2", 1, 100, "Y"),
            ("O3", "2026-02-03", "C1", "P2", "S1", 3, 600, "N"), ("O4", "2026-02-10", "C3", "P2", "S2", 1, 200, "Y"),
            ("O5", "2026-02-25", "C2", "P3", "S1", 4, 400, "N"), ("O6", "2026-03-01", "C3", "P3", "S2", 2, 200, "N"),
            ("O7", "2026-03-15", "C1", "P9", "S1", 1, 50, "N"), ("O8", "2026-03-20", "C2", "P1", "S1", 1, 100, "Y")]
    _write(tmp_path, "orders", [dict(zip(cols, r)) for r in rows])
    _write(tmp_path, "products", [{"product_id": "P1", "category": "T-Shirt", "brand": "Veda", "price": 100, "cost": 40},
                                  {"product_id": "P2", "category": "Jeans", "brand": "Lumo", "price": 200, "cost": 100},
                                  {"product_id": "P3", "category": "Jeans", "brand": "Veda", "price": 200, "cost": 120}])
    _write(tmp_path, "stores", [{"store_id": "S1", "store_name": "Mart Pune", "city": "Pune", "region": "West"},
                                {"store_id": "S2", "store_name": "Mart Delhi", "city": "Delhi", "region": "East"}])
    _write(tmp_path, "customers", [{"customer_id": "C1", "name": "Asha Rao", "email": "a@x.com", "city": "Pune"},
                                   {"customer_id": "C2", "name": "Ravi Das", "email": "b@x.com", "city": "Delhi"},
                                   {"customer_id": "C3", "name": "Meena Iyer", "email": "c@x.com", "city": "Pune"}])
    inv = [("P1", "S1", 0), ("P1", "S2", 5), ("P2", "S1", 0), ("P2", "S2", 10), ("P3", "S1", 3), ("P3", "S2", 0)]
    _write(tmp_path, "inventory", [{"product_id": p, "store_id": s, "stock_on_hand": n, "reorder_level": 10,
                                    "last_restock_date": "2026-03-01"} for p, s, n in inv])
    return Catalog.from_folder(tmp_path)


@pytest.fixture
def bot(shop):
    return Assistant(shop, ChatStore())


def rows_of(reply):
    return reply["answer"]["rows"]
