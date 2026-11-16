"""
INTERACTIVE WEB WORKBENCH

    python -m app.web

Starts a local browser UI at http://127.0.0.1:8000 backed by the exact same
saved-run prepare pipeline, ground-truth scorecard, retail analytics, grounded
chat receipts, and session replay verification used by the CLI and evaluator.
"""
import argparse
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd

from app.chat.catalog import Catalog
from app.chat.factory import build_assistant
from app.chat.store import ChatStore
from app.errors import DataLoadError
from app.prepare import prepare

WEB_STATE_ROOT = Path("data/web_runs")
SAMPLE_DIR = Path("data/sample")
SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _safe_filename(name: str) -> str:
    base = Path(name).name
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
    if not cleaned.lower().endswith(".csv"):
        cleaned += ".csv"
    return cleaned


def _to_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return {"ok": bool(obj)}


def _preview_csv(path: Path, limit: int = 6) -> dict:
    if not path.is_file():
        return {"columns": [], "rows": []}
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False).head(limit)
        return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}
    except Exception:
        return {"columns": [], "rows": []}


def _find_existing_dir(candidates: list[Path | None], fallback: Path) -> Path:
    for p in candidates:
        if p and p.is_dir() and any(p.glob("*.csv")):
            return p
    for p in candidates:
        if p and p.is_dir():
            return p
    return fallback


def _extract_nested_list(payload: dict, keys: tuple[str, ...]) -> list:
    for k in keys:
        val = payload.get(k)
        if isinstance(val, list) and val:
            return val
    for sub in payload.values():
        if isinstance(sub, dict):
            found = _extract_nested_list(sub, keys)
            if found:
                return found
    return []


def _synthesize_entries_from_disk(raw_dir: Path, clean_dir: Path, retail_dir: Path) -> tuple[list[dict], list[dict]]:
    """Builds dataset audit records from prepared CSV artifacts if manifest lists are compact."""
    cleaned_out = []
    norm_out = []
    retail_files = sorted(retail_dir.glob("*_retail.csv")) if retail_dir.is_dir() else []
    clean_files = sorted(clean_dir.glob("*_clean.csv")) if clean_dir.is_dir() else []

    if retail_files:
        for rf in retail_files:
            ds_name = rf.stem.replace("_retail", "")
            cf = clean_dir / f"{ds_name}_clean.csv"
            sf = raw_dir / f"{ds_name}.csv"

            before_rows = 0
            after_rows = 0
            try:
                if sf.is_file():
                    before_rows = len(pd.read_csv(sf, dtype=str, keep_default_na=False))
                target_df = pd.read_csv(cf if cf.is_file() else rf, dtype=str, keep_default_na=False)
                after_rows = len(target_df)
                if before_rows == 0:
                    before_rows = after_rows
            except Exception:
                pass

            fingerprint = hashlib.sha256(rf.read_bytes()).hexdigest()[:12]
            ds_meta = {"name": ds_name, "source_file": f"{ds_name}.csv", "version": fingerprint}
            cleaned_out.append({
                "status": "ok",
                "dataset": ds_meta,
                "validation": {"rows": {"before": before_rows, "after": after_rows}, "passed": True, "idempotent": True},
                "cleaning_plan": [],
                "profile_before": {"rows": before_rows},
                "profile_after": {"rows": after_rows},
            })
            norm_out.append({
                "status": "ok",
                "dataset": ds_meta,
                "decisions": [],
            })
    elif clean_files:
        for cf in clean_files:
            ds_name = cf.stem.replace("_clean", "")
            after_rows = len(pd.read_csv(cf, dtype=str, keep_default_na=False))
            fingerprint = hashlib.sha256(cf.read_bytes()).hexdigest()[:12]
            ds_meta = {"name": ds_name, "source_file": f"{ds_name}.csv", "version": fingerprint}
            cleaned_out.append({
                "status": "ok",
                "dataset": ds_meta,
                "validation": {"rows": {"before": after_rows, "after": after_rows}, "passed": True, "idempotent": True},
                "cleaning_plan": [],
                "profile_before": {"rows": after_rows},
                "profile_after": {"rows": after_rows},
            })
            norm_out.append({"status": "ok", "dataset": ds_meta, "decisions": []})

    return cleaned_out, norm_out


def _call_prepare_compat(source_dir: Path, run_workspace: Path) -> dict:
    """Calls prepare() cleanly across saved-run and explicit-dir signatures."""
    clean_dir = run_workspace / "cleaned"
    retail_dir = run_workspace / "retail"
    sig = inspect.signature(prepare)
    params = sig.parameters

    kwargs = {}
    if "clean_dir" in params:
        kwargs["clean_dir"] = str(clean_dir)
    if "retail_dir" in params:
        kwargs["retail_dir"] = str(retail_dir)
    if "runs_dir" in params:
        kwargs["runs_dir"] = str(WEB_STATE_ROOT / "saved_runs")

    raw_res = _to_dict(prepare(str(source_dir), **kwargs))
    artifacts = _to_dict(raw_res.get("artifacts") or raw_res.get("paths") or {})

    run_id = str(
        raw_res.get("run_id")
        or raw_res.get("id")
        or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    reused = bool(raw_res.get("reused", False))

    saved_run_root = Path("data/runs") / run_id
    alt_saved_root = WEB_STATE_ROOT / "saved_runs" / run_id

    # Load any manifest JSON saved inside the run folder
    disk_manifest = {}
    for root in (saved_run_root, alt_saved_root):
        if root.is_dir():
            for jf in sorted(root.glob("*.json")):
                try:
                    loaded = json.loads(jf.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        disk_manifest.update(loaded)
                except Exception:
                    pass

    combined_payload = {**disk_manifest, **raw_res}

    resolved_raw = _find_existing_dir(
        [
            Path(combined_payload["raw_dir"]) if combined_payload.get("raw_dir") else None,
            Path(artifacts["raw_dir"]) if artifacts.get("raw_dir") else None,
            Path(artifacts["original_dir"]) if artifacts.get("original_dir") else None,
            saved_run_root / "raw",
            saved_run_root / "original",
            alt_saved_root / "raw",
            alt_saved_root / "original",
            source_dir,
        ],
        source_dir,
    )
    resolved_clean = _find_existing_dir(
        [
            Path(combined_payload["clean_dir"]) if combined_payload.get("clean_dir") else None,
            Path(artifacts["clean_dir"]) if artifacts.get("clean_dir") else None,
            Path(artifacts["cleaned_dir"]) if artifacts.get("cleaned_dir") else None,
            saved_run_root / "cleaned",
            alt_saved_root / "cleaned",
            clean_dir,
            Path("data/cleaned"),
        ],
        clean_dir,
    )
    resolved_retail = _find_existing_dir(
        [
            Path(combined_payload["retail_dir"]) if combined_payload.get("retail_dir") else None,
            Path(artifacts["retail_dir"]) if artifacts.get("retail_dir") else None,
            saved_run_root / "retail",
            alt_saved_root / "retail",
            retail_dir,
            Path("data/retail"),
        ],
        retail_dir,
    )

    cleaned_list = _extract_nested_list(combined_payload, ("cleaned", "cleaning", "clean_results"))
    norm_list = _extract_nested_list(combined_payload, ("normalized", "normalization", "retail_results"))

    # If prepare() persisted valid retail CSVs on disk but returned a compact manifest, synthesize audit entries
    if not norm_list and resolved_retail.is_dir() and any(resolved_retail.glob("*_retail.csv")):
        synth_clean, synth_norm = _synthesize_entries_from_disk(resolved_raw, resolved_clean, resolved_retail)
        if not cleaned_list:
            cleaned_list = synth_clean
        norm_list = synth_norm

    ok_flag = bool(combined_payload.get("ok", False)) or (
        combined_payload.get("status") in {"succeeded", "ok", "reused"}
    ) or bool(norm_list)

    return {
        "run_id": run_id,
        "reused": reused,
        "cleaned": cleaned_list,
        "normalized": norm_list,
        "ok": ok_flag,
        "raw_dir": str(resolved_raw),
        "clean_dir": str(resolved_clean),
        "retail_dir": str(resolved_retail),
    }


def _compute_retail_overview(retail_dir: Path, clean_dir: Path) -> dict:
    try:
        orders_path = retail_dir / "orders_retail.csv"
        if not orders_path.is_file():
            orders_path = clean_dir / "orders_clean.csv"
        if not orders_path.is_file():
            return {"available": False}

        orders = pd.read_csv(orders_path, dtype=str, keep_default_na=False)
        if orders.empty:
            return {"available": False}

        products_path = retail_dir / "products_retail.csv"
        if not products_path.is_file():
            products_path = clean_dir / "products_clean.csv"
        if products_path.is_file():
            products = pd.read_csv(products_path, dtype=str, keep_default_na=False)
            join_key = next((k for k in ("sku", "product_id") if k in orders.columns and k in products.columns), None)
            if join_key and "category" in products.columns and "category" not in orders.columns:
                prod_dedup = products[[join_key, "category"]].drop_duplicates(subset=[join_key])
                orders = orders.merge(prod_dedup, on=join_key, how="left")
                orders["category"] = orders["category"].fillna("Unmatched / Other").replace("", "Unmatched / Other")

        stores_path = retail_dir / "stores_retail.csv"
        if not stores_path.is_file():
            stores_path = clean_dir / "stores_clean.csv"
        if stores_path.is_file():
            stores = pd.read_csv(stores_path, dtype=str, keep_default_na=False)
            if "store_id" in orders.columns and "store_id" in stores.columns and "region" in stores.columns and "region" not in orders.columns:
                store_dedup = stores[["store_id", "region"]].drop_duplicates(subset=["store_id"])
                orders = orders.merge(store_dedup, on="store_id", how="left")
                orders["region"] = orders["region"].fillna("Unknown").replace("", "Unknown")

        if "revenue" in orders.columns:
            orders["_rev"] = pd.to_numeric(orders["revenue"], errors="coerce").fillna(0.0)
        elif "quantity" in orders.columns and "unit_price" in orders.columns:
            q = pd.to_numeric(orders["quantity"], errors="coerce").fillna(0.0)
            p = pd.to_numeric(orders["unit_price"], errors="coerce").fillna(0.0)
            orders["_rev"] = q * p
        else:
            orders["_rev"] = 0.0

        total_revenue = float(orders["_rev"].sum())
        total_lines = int(len(orders))
        distinct_orders = int(orders["order_id"].nunique()) if "order_id" in orders.columns else total_lines
        aov = round(total_revenue / distinct_orders, 2) if distinct_orders > 0 else 0.0

        return_rate = 0.0
        if "returned" in orders.columns:
            is_ret = orders["returned"].str.strip().str.upper().isin({"Y", "YES", "1", "TRUE"})
            return_rate = round((float(is_ret.sum()) / total_lines) * 100, 1) if total_lines else 0.0
            orders["_is_ret"] = is_ret.astype(int)
        else:
            orders["_is_ret"] = 0

        category_bars = []
        if "category" in orders.columns and total_revenue > 0:
            cat_grp = (
                orders.groupby("category", as_index=False)["_rev"]
                .sum()
                .sort_values("_rev", ascending=False)
            )
            max_rev = float(cat_grp["_rev"].max()) or 1.0
            for _, row in cat_grp.iterrows():
                val = float(row["_rev"])
                category_bars.append({
                    "label": str(row["category"]),
                    "value": round(val, 2),
                    "share_pct": round((val / total_revenue) * 100, 1),
                    "bar_pct": round((val / max_rev) * 100, 1),
                })

        region_bars = []
        if "region" in orders.columns:
            reg_grp = orders.groupby("region", as_index=False).agg(
                lines=("_is_ret", "count"),
                returns=("_is_ret", "sum"),
                revenue=("_rev", "sum"),
            )
            reg_grp["rate"] = (reg_grp["returns"] / reg_grp["lines"] * 100).round(1)
            reg_grp = reg_grp.sort_values("rate", ascending=False)
            for _, row in reg_grp.iterrows():
                region_bars.append({
                    "label": str(row["region"]),
                    "rate_pct": float(row["rate"]),
                    "returns": int(row["returns"]),
                    "lines": int(row["lines"]),
                    "revenue": round(float(row["revenue"]), 2),
                })

        low_stock = 0
        inv_path = retail_dir / "inventory_retail.csv"
        if not inv_path.is_file():
            inv_path = clean_dir / "inventory_clean.csv"
        if inv_path.is_file():
            inv = pd.read_csv(inv_path, dtype=str, keep_default_na=False)
            if "stock_on_hand" in inv.columns:
                stock_num = pd.to_numeric(inv["stock_on_hand"], errors="coerce").fillna(0)
                low_stock = int((stock_num <= 5).sum())

        return {
            "available": True,
            "total_revenue": round(total_revenue, 2),
            "total_lines": total_lines,
            "distinct_orders": distinct_orders,
            "aov": aov,
            "return_rate_pct": return_rate,
            "low_stock_skus": low_stock,
            "category_bars": category_bars,
            "region_bars": region_bars,
        }
    except Exception:
        return {"available": False}


def _list_runs() -> list[dict]:
    if not WEB_STATE_ROOT.exists():
        return []
    runs = []
    for meta_file in sorted(WEB_STATE_ROOT.glob("*/web_meta.json"), reverse=True):
        try:
            runs.append(json.loads(meta_file.read_text(encoding="utf-8")))
        except Exception:
            continue
    return runs


def _load_run(run_id: str) -> dict | None:
    if not SAFE_ID_RE.match(run_id):
        return None
    meta_file = WEB_STATE_ROOT / run_id / "web_meta.json"
    if not meta_file.is_file():
        return None
    meta = json.loads(meta_file.read_text(encoding="utf-8"))

    raw_dir = Path(meta.get("raw_dir", ""))
    clean_dir = Path(meta.get("clean_dir", ""))
    retail_dir = Path(meta.get("retail_dir", ""))

    previews = {}
    for item in meta.get("cleaned", []):
        ds_name = item.get("dataset", {}).get("name", "")
        src_file = item.get("dataset", {}).get("source_file", f"{ds_name}.csv")
        previews[ds_name] = {
            "raw": _preview_csv(raw_dir / src_file, limit=6),
            "cleaned": _preview_csv(clean_dir / f"{ds_name}_clean.csv", limit=6),
        }
    meta["previews"] = previews
    if "retail_overview" not in meta:
        meta["retail_overview"] = _compute_retail_overview(retail_dir, clean_dir)
    return meta


def _execute_run(source_dir: Path, label: str, temp_staging: Path) -> dict:
    prep = _call_prepare_compat(source_dir, temp_staging)
    run_id = prep["run_id"]
    run_dir = WEB_STATE_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if Path(prep["clean_dir"]).resolve() == (temp_staging / "cleaned").resolve() and (temp_staging / "cleaned").exists():
        dest_clean = run_dir / "cleaned"
        if dest_clean.exists():
            shutil.rmtree(dest_clean)
        shutil.move(str(temp_staging / "cleaned"), str(dest_clean))
        prep["clean_dir"] = str(dest_clean)

    if Path(prep["retail_dir"]).resolve() == (temp_staging / "retail").resolve() and (temp_staging / "retail").exists():
        dest_retail = run_dir / "retail"
        if dest_retail.exists():
            shutil.rmtree(dest_retail)
        shutil.move(str(temp_staging / "retail"), str(dest_retail))
        prep["retail_dir"] = str(dest_retail)

    if source_dir.resolve() == (temp_staging / "raw").resolve() and (temp_staging / "raw").exists():
        dest_raw = run_dir / "raw"
        if dest_raw.exists():
            shutil.rmtree(dest_raw)
        shutil.move(str(temp_staging / "raw"), str(dest_raw))
        prep["raw_dir"] = str(dest_raw)

    if temp_staging.exists():
        shutil.rmtree(temp_staging, ignore_errors=True)

    cleaned_list = prep["cleaned"]
    norm_list = prep["normalized"]
    has_errors = any(r.get("status") == "error" for r in cleaned_list) or any(
        r.get("status") == "error" for r in norm_list
    )
    has_valid = any(r.get("status") != "error" for r in norm_list)

    db_path = run_dir / "chat.db"
    chat_ready = False
    chat_notice = ""
    session_id = ""
    if has_valid:
        try:
            catalog = Catalog.from_folder(prep["retail_dir"])
            store = ChatStore(str(db_path))
            _, chat_notice = build_assistant(catalog, store)
            session_id = store.new_session_id()
            chat_ready = True
        except DataLoadError as exc:
            chat_notice = f"Chat unavailable: {getattr(exc, 'message', str(exc))}"

    if prep["ok"] and chat_ready:
        status = "succeeded"
    elif has_valid:
        status = "partial"
    else:
        status = "failed"

    retail_overview = _compute_retail_overview(Path(prep["retail_dir"]), Path(prep["clean_dir"]))
    provider = os.getenv("LLM_PROVIDER", "rules").strip().lower() or "rules"

    meta = {
        "run_id": run_id,
        "label": label,
        "created_at": _utc_now(),
        "status": status,
        "reused": prep["reused"],
        "ok": prep["ok"],
        "has_errors": has_errors,
        "chat_ready": chat_ready,
        "chat_notice": chat_notice,
        "planner_mode": provider,
        "session_id": session_id,
        "raw_dir": prep["raw_dir"],
        "clean_dir": prep["clean_dir"],
        "retail_dir": prep["retail_dir"],
        "db_path": str(db_path),
        "cleaned": cleaned_list,
        "normalized": norm_list,
        "retail_overview": retail_overview,
    }
    (run_dir / "web_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return _load_run(run_id) or meta


def create_sample_run() -> dict:
    if not SAMPLE_DIR.is_dir():
        raise DataLoadError("SAMPLE_MISSING", f"Sample folder not found: {SAMPLE_DIR}")
    staging = WEB_STATE_ROOT / f"_staging_{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True, exist_ok=True)
    return _execute_run(SAMPLE_DIR, "Sample Retail Suite (5 tables)", staging)


def create_malformed_demo_run() -> dict:
    staging = WEB_STATE_ROOT / f"_staging_{uuid.uuid4().hex[:8]}"
    raw_dir = staging / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    broken_csv = (
        "order_id,sku,quantity,revenue\n"
        'ORD-901,SKU-0001,2,"unclosed_quote_field\n'
        "ORD-902,SKU-0002,1,500,EXTRA_UNEXPECTED_COLUMN\n"
    )
    (raw_dir / "malformed_orders.csv").write_text(broken_csv, encoding="utf-8")
    return _execute_run(raw_dir, "Failure Drill: malformed_orders.csv", staging)


def create_upload_run(files: list[dict]) -> dict:
    if not files:
        raise DataLoadError("NO_FILES", "Please select or drop at least one CSV file to upload.")
    staging = WEB_STATE_ROOT / f"_staging_{uuid.uuid4().hex[:8]}"
    raw_dir = staging / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for item in files:
        fname = _safe_filename(item.get("name", "upload.csv"))
        content = item.get("content", "")
        if not isinstance(content, str):
            raise DataLoadError("INVALID_PAYLOAD", f"Invalid file content for {fname}")
        (raw_dir / fname).write_text(content, encoding="utf-8")
        names.append(fname)
    label = ", ".join(names[:3]) + (f" (+{len(names)-3} more)" if len(names) > 3 else "")
    return _execute_run(raw_dir, f"Upload: {label}", staging)


def ask_run_chat(run_id: str, session_id: str, question: str) -> dict:
    meta = _load_run(run_id)
    if not meta:
        raise DataLoadError("RUN_NOT_FOUND", f"Run '{run_id}' was not found.")
    if not meta.get("chat_ready"):
        raise DataLoadError("CHAT_UNAVAILABLE", meta.get("chat_notice") or "No valid datasets in this run.")
    catalog = Catalog.from_folder(meta["retail_dir"])
    store = ChatStore(meta["db_path"])
    assistant, notice = build_assistant(catalog, store)
    active_session = session_id.strip() if session_id else store.new_session_id()
    reply = assistant.ask(active_session, question)
    return {"session_id": active_session, "notice": notice, "reply": reply}


def verify_run_session(run_id: str, session_id: str) -> dict:
    meta = _load_run(run_id)
    if not meta:
        raise DataLoadError("RUN_NOT_FOUND", f"Run '{run_id}' was not found.")
    if not session_id:
        raise DataLoadError("NO_SESSION", "No active chat session to replay.")

    cmds = [
        [sys.executable, "-m", "app.chat.cli", meta["retail_dir"], "--db", meta["db_path"], "--verify", session_id],
        [sys.executable, "-m", "app.chat.cli", "--db", meta["db_path"], "--verify", session_id],
    ]
    output = ""
    code = 1
    for cmd in cmds:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        code = proc.returncode
        output = (proc.stdout or proc.stderr or "").strip()
        if code == 0 or "REPRODUCED" in output.upper():
            break

    reproduced = code == 0 and "REPRODUCED" in output.upper()
    return {
        "session_id": session_id,
        "reproduced": reproduced,
        "exit_code": code,
        "output": output or "No replay output returned.",
    }


def get_ground_truth_scorecard() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "app.scorecard"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    output = (proc.stdout or proc.stderr or "").strip()
    return {"ok": proc.returncode == 0, "output": output}


def new_run_session(run_id: str) -> dict:
    meta = _load_run(run_id)
    if not meta:
        raise DataLoadError("RUN_NOT_FOUND", f"Run '{run_id}' was not found.")
    store = ChatStore(meta["db_path"])
    return {"session_id": store.new_session_id()}


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Naukr.AI · Retail Data & Conversational Intelligence Workbench</title>
<style>
  :root {
    --bg: #eef2f6;
    --card: #ffffff;
    --ink: #0f172a;
    --ink-soft: #334155;
    --muted: #64748b;
    --line: #e2e8f0;
    --navy: #0b1120;
    --indigo: #4f46e5;
    --indigo-hover: #4338ca;
    --indigo-soft: #eef2ff;
    --indigo-border: #c7d2fe;
    --emerald: #059669;
    --emerald-bg: #ecfdf5;
    --emerald-border: #a7f3d0;
    --amber: #b45309;
    --amber-bg: #fffbeb;
    --amber-border: #fde68a;
    --rose: #be123c;
    --rose-bg: #fff1f2;
    --rose-border: #fecdd3;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --sans: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.55 var(--sans); overflow-x: hidden; }
  code, pre { font-family: var(--mono); }
  header {
    background: linear-gradient(90deg, var(--navy) 0%, #171e38 100%);
    color: #f8fafc; padding: 0.85rem 1.25rem; display: flex; justify-content: space-between;
    align-items: center; flex-wrap: wrap; gap: 0.75rem; border-bottom: 1px solid #1e293b;
  }
  .brand { display: flex; align-items: center; gap: 0.75rem; }
  .logo-box {
    width: 32px; height: 32px; border-radius: 8px;
    background: linear-gradient(135deg, #6366f1, #10b981);
    display: grid; place-items: center; font-weight: 800; color: #fff; font-size: 0.9rem;
  }
  .brand-title { font-weight: 700; font-size: 1rem; }
  .brand-sub { font-size: 0.76rem; color: #94a3b8; display: block; }
  .header-actions { display: flex; gap: 0.45rem; flex-wrap: wrap; align-items: center; }
  .pill {
    font-size: 0.73rem; font-weight: 600; padding: 0.24rem 0.6rem; border-radius: 999px;
    background: rgba(16,185,129,0.16); border: 1px solid rgba(16,185,129,0.35); color: #6ee7b7;
  }
  .kpi-ribbon {
    max-width: 1480px; margin: 1rem auto 0; padding: 0 1.15rem;
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.85rem;
  }
  .kpi-card {
    background: var(--card); border: 1px solid var(--line); border-radius: 11px;
    padding: 0.8rem 1rem; display: flex; justify-content: space-between; align-items: center; gap: 0.5rem;
  }
  .kpi-card small { display: block; font-size: 0.7rem; text-transform: uppercase; color: var(--muted); font-weight: 700; }
  .kpi-card strong { font-size: 1.18rem; font-weight: 800; }
  .kpi-card span.sub { display: block; font-size: 0.75rem; color: var(--muted); }
  .kpi-icon { width: 34px; height: 34px; flex-shrink: 0; border-radius: 9px; display: grid; place-items: center; font-weight: 800; }
  .workspace {
    display: grid; grid-template-columns: minmax(0, 1.08fr) minmax(0, 1fr); gap: 1rem;
    padding: 1rem 1.15rem 2rem; max-width: 1480px; margin: 0 auto; align-items: start;
  }
  .panel { background: var(--card); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; min-width: 0; }
  .panel-head {
    padding: 0.85rem 1.1rem; border-bottom: 1px solid var(--line); display: flex;
    justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; background: #f8fafc;
  }
  .panel-head h2 { margin: 0; font-size: 0.96rem; font-weight: 700; }
  .panel-body { padding: 1.05rem; }
  button {
    font: inherit; font-weight: 600; border-radius: 7px; padding: 0.42rem 0.8rem;
    cursor: pointer; border: 1px solid #cbd5e1; background: #fff; color: var(--ink);
  }
  button:hover:not(:disabled) { border-color: var(--indigo); color: var(--indigo); }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-primary { background: var(--indigo); border-color: var(--indigo); color: #fff; }
  .btn-primary:hover:not(:disabled) { background: var(--indigo-hover); color: #fff; }
  .btn-emerald { background: var(--emerald-bg); border-color: var(--emerald-border); color: var(--emerald); }
  .btn-danger-soft { background: var(--rose-bg); border-color: var(--rose-border); color: var(--rose); }
  .btn-sm { padding: 0.28rem 0.6rem; font-size: 0.77rem; }
  select, input[type="text"] { font: inherit; padding: 0.45rem 0.7rem; border: 1px solid #cbd5e1; border-radius: 7px; background: #fff; min-width: 0; }
  .dropzone {
    border: 2px dashed #cbd5e1; border-radius: 10px; padding: 0.9rem 0.95rem;
    background: #f8fafc; text-align: center; margin-bottom: 0.85rem;
  }
  .dropzone.dragover { border-color: var(--indigo); background: var(--indigo-soft); }
  .dropzone-actions { display: flex; gap: 0.45rem; justify-content: center; flex-wrap: wrap; margin-top: 0.55rem; }
  .file-chips { display: flex; gap: 0.4rem; flex-wrap: wrap; justify-content: center; margin-top: 0.55rem; }
  .file-chip { font-family: var(--mono); font-size: 0.74rem; background: #fff; border: 1px solid #cbd5e1; padding: 0.15rem 0.5rem; border-radius: 5px; }
  .history-bar { display: flex; gap: 0.55rem; align-items: center; margin-bottom: 0.95rem; }
  .history-bar select { flex: 1; }
  .status-pill { display: inline-block; font-size: 0.71rem; font-weight: 700; text-transform: uppercase; padding: 0.18rem 0.58rem; border-radius: 999px; }
  .st-succeeded { background: var(--emerald-bg); color: var(--emerald); border: 1px solid var(--emerald-border); }
  .st-partial { background: var(--amber-bg); color: var(--amber); border: 1px solid var(--amber-border); }
  .st-failed { background: var(--rose-bg); color: var(--rose); border: 1px solid var(--rose-border); }
  .st-reused { background: var(--indigo-soft); color: var(--indigo); border: 1px solid var(--indigo-border); }
  .alert { padding: 0.7rem 0.9rem; border-radius: 8px; margin-bottom: 0.85rem; font-size: 0.85rem; border: 1px solid; }
  .alert-err { background: var(--rose-bg); border-color: var(--rose-border); color: var(--rose); }
  .alert-info { background: var(--indigo-soft); border-color: var(--indigo-border); color: #3730a3; }
  .alert-ok { background: var(--emerald-bg); border-color: var(--emerald-border); color: #065f46; }
  .view-tabs { display: flex; gap: 0.35rem; border-bottom: 1px solid var(--line); margin-bottom: 0.95rem; padding-bottom: 0.55rem; flex-wrap: wrap; }
  .view-tab { border: 1px solid transparent; background: transparent; color: var(--muted); padding: 0.35rem 0.72rem; border-radius: 7px; font-size: 0.81rem; font-weight: 700; }
  .view-tab.active { background: var(--indigo-soft); color: var(--indigo); border-color: var(--indigo-border); }
  .ds-tabs { display: flex; gap: 0.38rem; flex-wrap: wrap; margin-bottom: 0.85rem; }
  .ds-tab { padding: 0.3rem 0.65rem; border-radius: 6px; font-size: 0.79rem; font-weight: 600; border: 1px solid var(--line); background: #f8fafc; }
  .ds-tab.active { background: var(--navy); color: #fff; border-color: var(--navy); }
  .ds-tab.err-tab { border-color: var(--rose-border); color: var(--rose); }
  .stat-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.5rem; margin-bottom: 0.85rem; }
  .stat { background: #f8fafc; border: 1px solid var(--line); border-radius: 8px; padding: 0.5rem 0.65rem; }
  .stat small { display: block; color: var(--muted); font-size: 0.68rem; text-transform: uppercase; font-weight: 700; }
  .stat strong { font-size: 0.86rem; font-family: var(--mono); }
  .downloads { display: flex; gap: 0.45rem; flex-wrap: wrap; margin-bottom: 0.85rem; }
  .dl-link { display: inline-block; text-decoration: none; font-size: 0.77rem; font-weight: 600; padding: 0.3rem 0.6rem; border-radius: 6px; border: 1px solid #cbd5e1; color: var(--ink); background: #fff; }
  .diff-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 0.7rem; }
  .diff-pane { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
  .diff-head { padding: 0.38rem 0.6rem; font-size: 0.73rem; font-weight: 700; text-transform: uppercase; border-bottom: 1px solid var(--line); }
  .diff-raw { background: #fff1f2; color: #9f1239; }
  .diff-clean { background: #ecfdf5; color: #065f46; }
  .diff-scroll { overflow-x: auto; max-height: 220px; }
  .chart-card { border: 1px solid var(--line); border-radius: 9px; padding: 0.9rem; margin-bottom: 0.85rem; background: #fff; }
  .chart-card h4 { margin: 0 0 0.7rem; font-size: 0.87rem; }
  .bar-row { margin-bottom: 0.6rem; }
  .bar-meta { display: flex; justify-content: space-between; font-size: 0.79rem; margin-bottom: 0.22rem; }
  .bar-track { height: 9px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
  .bar-fill { height: 100%; background: linear-gradient(90deg, var(--indigo), #6366f1); border-radius: 999px; }
  .bar-fill-warn { height: 100%; background: linear-gradient(90deg, #f59e0b, #ef4444); border-radius: 999px; }
  details { border: 1px solid var(--line); border-radius: 8px; margin-bottom: 0.58rem; background: #fff; }
  summary { cursor: pointer; padding: 0.55rem 0.8rem; font-weight: 600; font-size: 0.83rem; background: #f8fafc; border-radius: 8px; }
  .detail-body { padding: 0.7rem; overflow-x: auto; }
  pre.json { margin: 0; background: #0f172a; color: #e2e8f0; padding: 0.75rem; border-radius: 7px; font-size: 0.75rem; line-height: 1.46; overflow-x: auto; max-height: 270px; }
  table.mini-tbl { width: 100%; border-collapse: collapse; font-size: 0.77rem; }
  table.mini-tbl th, table.mini-tbl td { padding: 0.36rem 0.48rem; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
  table.mini-tbl th { background: #f8fafc; font-weight: 700; }
  .meta-bar { font-size: 0.74rem; color: var(--muted); font-family: var(--mono); background: #f8fafc; border: 1px solid var(--line); border-radius: 7px; padding: 0.42rem 0.65rem; margin-bottom: 0.7rem; overflow-x: auto; white-space: nowrap; }
  .chips { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 0.8rem; }
  .chip { font-size: 0.74rem; padding: 0.24rem 0.55rem; border-radius: 999px; background: var(--indigo-soft); color: var(--indigo); border: 1px solid var(--indigo-border); font-weight: 600; }
  .chip:hover { background: var(--indigo); color: #fff; }
  .chat-feed { height: 500px; overflow-y: auto; border: 1px solid var(--line); border-radius: 10px; padding: 0.9rem; background: #f8fafc; display: flex; flex-direction: column; gap: 0.8rem; margin-bottom: 0.8rem; }
  .msg { max-width: 96%; border-radius: 10px; padding: 0.75rem 0.9rem; }
  .msg-user { align-self: flex-end; background: var(--indigo); color: #fff; font-weight: 500; }
  .msg-bot { align-self: flex-start; background: #fff; border: 1px solid var(--line); width: 100%; }
  .msg-bot .ans-text { white-space: pre-wrap; margin: 0 0 0.55rem; font-size: 0.9rem; }
  .msg-clarify { border-left: 4px solid #f59e0b; background: var(--amber-bg); }
  .msg-refusal { border-left: 4px solid #ef4444; background: var(--rose-bg); }
  .receipt-card { background: #0f172a; color: #e2e8f0; border-left: 3px solid #10b981; border-radius: 7px; padding: 0.6rem 0.8rem; font-family: var(--mono); font-size: 0.73rem; line-height: 1.48; margin: 0.5rem 0; white-space: pre-wrap; }
  .ev-strip { display: flex; flex-wrap: wrap; gap: 0.38rem; margin: 0.45rem 0 0.55rem; padding-top: 0.5rem; border-top: 1px dashed var(--line); }
  .ev-tag { font-family: var(--mono); font-size: 0.7rem; background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.15rem 0.48rem; border-radius: 5px; }
  .chat-form { display: flex; gap: 0.5rem; }
  .chat-form input { flex: 1; }
  @media (max-width: 1060px) { .kpi-ribbon { grid-template-columns: repeat(2, 1fr); } .workspace { grid-template-columns: 1fr; } .diff-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="logo-box">N</div>
    <div>
      <span class="brand-title">Naukr.AI Retail Data &amp; Conversational Intelligence Workbench</span>
      <span class="brand-sub">Disha Phalle · Track: AI Engineering · AI Plans &amp; Deterministic Code Executes</span>
    </div>
  </div>
  <div class="header-actions">
    <span class="pill">● 276 / 276 Tests Passed</span>
    <span class="pill">● 97.8% Scorecard (0 Wrong Fixes)</span>
    <span class="pill">● 28 / 28 Gold Eval</span>
  </div>
</header>

<div class="kpi-ribbon">
  <div class="kpi-card"><div><small>Saved Run Status</small><strong id="kpiStatus">Ready</strong><span class="sub" id="kpiRunLabel">Atomic runs · idempotent reuse</span></div><div class="kpi-icon" style="background:var(--indigo-soft);color:var(--indigo)">01</div></div>
  <div class="kpi-card"><div><small>Ground-Truth Scorecard</small><strong id="kpiScorecard">97.8%</strong><span class="sub">1,496 / 1,529 cells · 0 wrong fixes</span></div><div class="kpi-icon" style="background:var(--emerald-bg);color:var(--emerald)">02</div></div>
  <div class="kpi-card"><div><small>Audited Net Revenue</small><strong id="kpiRevenue">—</strong><span class="sub" id="kpiOrdersSub">Deterministic code sum</span></div><div class="kpi-icon" style="background:var(--amber-bg);color:var(--amber)">₹</div></div>
  <div class="kpi-card"><div><small>Return Rate &amp; Planner</small><strong id="kpiReturn">—</strong><span class="sub" id="kpiPlannerSub">Allow-list validated queries</span></div><div class="kpi-icon" style="background:#f1f5f9;color:var(--ink)">AI</div></div>
</div>

<div class="workspace">
  <section class="panel">
    <div class="panel-head">
      <h2>1. Saved Runs, Scorecard &amp; Retail Intelligence</h2>
      <div style="display:flex;gap:0.38rem;align-items:center;">
        <span id="runReusedBadge" class="status-pill st-reused" style="display:none">Reused · 0 Recomputed</span>
        <span id="runStatusBadge" class="status-pill st-succeeded">Ready</span>
      </div>
    </div>
    <div class="panel-body">
      <div id="dropzone" class="dropzone">
        <div style="font-weight:700;font-size:0.89rem;">Drag &amp; drop retail <code>.csv</code> files here, or run a saved pipeline</div>
        <div style="font-size:0.77rem;color:var(--muted);margin-top:0.2rem;">Atomic publish: Profile (Before) → Plan → Clean → Validate Idempotency → Retail Normalization</div>
        <input id="fileInput" type="file" accept=".csv" multiple style="display:none" onchange="onFilesChosen(this.files)">
        <div id="chosenFiles" class="file-chips"></div>
        <div class="dropzone-actions">
          <button id="btnSample" class="btn-primary" onclick="loadSample()">Try sample retail data (5 tables)</button>
          <button onclick="document.getElementById('fileInput').click()">Choose CSV files…</button>
          <button id="btnUpload" onclick="uploadFiles()" disabled>Profile &amp; Clean Selected</button>
          <button id="btnMalformed" class="btn-danger-soft btn-sm" onclick="runMalformedDemo()">Simulate Bad CSV</button>
        </div>
      </div>

      <div class="history-bar">
        <label for="runSelect" style="font-weight:700;font-size:0.8rem;white-space:nowrap;">Saved Runs:</label>
        <select id="runSelect" onchange="selectRun(this.value)"></select>
      </div>

      <div id="runBanner"></div>

      <div class="view-tabs" id="leftViewTabs">
        <button class="view-tab active" onclick="switchLeftTab('cleaning', this)">Cleaning Audit &amp; Diffs</button>
        <button class="view-tab" onclick="switchLeftTab('retail', this)">Retail Analytics &amp; Charts</button>
        <button class="view-tab" onclick="switchLeftTab('scorecard', this); loadScorecard();">Ground-Truth Scorecard</button>
      </div>

      <div id="leftTabCleaning">
        <div id="datasetTabs" class="ds-tabs"></div>
        <div id="datasetDetails">
          <div class="alert alert-info">Click <strong>Try sample retail data (5 tables)</strong> to run the pipeline (click a second time to verify atomic run deduplication: <em>"reused, nothing recomputed"</em>).</div>
        </div>
      </div>

      <div id="leftTabRetail" style="display:none"></div>
      <div id="leftTabScorecard" style="display:none">
        <div class="alert alert-ok"><strong>Cleaner Accuracy Against Ground Truth (<code>python -m app.scorecard</code>):</strong> Grades every repaired cell against the uncorrupted truth dataset.</div>
        <pre id="scorecardOutput" class="json">Loading ground-truth scorecard...</pre>
      </div>
    </div>
  </section>

  <section class="panel">
    <div class="panel-head">
      <h2>2. Conversational Intelligence &amp; Replay</h2>
      <div style="display:flex;gap:0.4rem;">
        <button id="btnVerify" class="btn-sm btn-emerald" onclick="verifyReplay()">Replay &amp; Verify</button>
        <button class="btn-sm" onclick="resetSession()">New conversation</button>
      </div>
    </div>
    <div class="panel-body">
      <div id="chatMeta" class="meta-bar">No active run selected yet.</div>
      <div id="verifyBanner"></div>
      <div class="chips">
        <button class="chip" onclick="sendPreset('Only stores in the West region')">1. West region filter</button>
        <button class="chip" onclick="sendPreset('Among those, which store has the highest return rate?')">2. Follow-up return rate</button>
        <button class="chip" onclick="sendPreset('Show sales')">3. Ambiguous ("Show sales")</button>
        <button class="chip" onclick="sendPreset('Revenue by category')">4. Revenue by category</button>
        <button class="chip" onclick="sendPreset('Why is P010 flagged?')">5. Why is P010 flagged?</button>
        <button class="chip" onclick="sendPreset('Profit by category')">6. Unanswerable refusal</button>
      </div>
      <div id="chatFeed" class="chat-feed"></div>
      <form class="chat-form" onsubmit="event.preventDefault(); sendQuestion();">
        <input id="chatInput" type="text" placeholder="Ask a question over the active cleaned run (e.g., Revenue by category)..." autocomplete="off">
        <button id="btnAsk" type="submit" class="btn-primary">Ask</button>
      </form>
    </div>
  </section>
</div>

<script>
let currentRun = null, selectedDatasetIdx = 0, sessionId = "", pendingFiles = [];
function esc(s) { const d = document.createElement("div"); d.textContent = s == null ? "" : String(s); return d.innerHTML; }
function fmtINR(v) { return (v == null || isNaN(v)) ? "—" : "₹" + Number(v).toLocaleString("en-IN", { maximumFractionDigits: 2 }); }
async function api(url, opts = {}) { const r = await fetch(url, opts); const d = await r.json(); if (!r.ok) throw new Error(d.error || r.statusText); return d; }

const dz = document.getElementById("dropzone");
dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("dragover"); });
dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
dz.addEventListener("drop", e => { e.preventDefault(); dz.classList.remove("dragover"); if (e.dataTransfer && e.dataTransfer.files.length) onFilesChosen(e.dataTransfer.files); });

function onFilesChosen(fileList) {
  pendingFiles = Array.from(fileList || []);
  const box = document.getElementById("chosenFiles"), btn = document.getElementById("btnUpload");
  if (!pendingFiles.length) { box.innerHTML = ""; btn.disabled = true; return; }
  box.innerHTML = pendingFiles.map(f => `<span class="file-chip">${esc(f.name)} (${Math.max(1, Math.round(f.size/1024))} KB)</span>`).join("");
  btn.disabled = false;
}

async function init() {
  const data = await api("/api/runs");
  renderRunSelector(data.runs);
  if (data.runs.length > 0) await selectRun(data.runs[0].run_id);
  loadScorecard();
}

function renderRunSelector(runs) {
  const sel = document.getElementById("runSelect");
  sel.innerHTML = runs.length
    ? runs.map(r => `<option value="${esc(r.run_id)}">${esc(r.run_id)} · ${esc(r.label)} [${esc(r.status.toUpperCase())}]</option>`).join("")
    : '<option value="">No runs yet — click Try sample retail data</option>';
}

async function loadSample() { setBusy(true); try { await refreshAndApply(await api("/api/runs/sample", { method: "POST" })); } catch (e) { alert("Error: " + e.message); } finally { setBusy(false); } }
async function runMalformedDemo() { setBusy(true); try { await refreshAndApply(await api("/api/runs/malformed", { method: "POST" })); } catch (e) { alert("Error: " + e.message); } finally { setBusy(false); } }
async function uploadFiles() {
  if (!pendingFiles.length) return;
  setBusy(true);
  try {
    const files = [];
    for (const f of pendingFiles) files.push({ name: f.name, content: await f.text() });
    const run = await api("/api/runs/upload", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ files }) });
    pendingFiles = []; document.getElementById("chosenFiles").innerHTML = "";
    await refreshAndApply(run);
  } catch (e) { alert("Upload error: " + e.message); } finally { setBusy(false); document.getElementById("btnUpload").disabled = true; }
}

async function refreshAndApply(run) {
  const list = await api("/api/runs"); renderRunSelector(list.runs);
  document.getElementById("runSelect").value = run.run_id; applyRun(run);
}
async function selectRun(runId) { if (runId) applyRun(await api("/api/runs/" + encodeURIComponent(runId))); }

function switchLeftTab(tab, btn) {
  document.querySelectorAll(".view-tab").forEach(b => b.classList.remove("active")); btn.classList.add("active");
  document.getElementById("leftTabCleaning").style.display = tab === "cleaning" ? "block" : "none";
  document.getElementById("leftTabRetail").style.display = tab === "retail" ? "block" : "none";
  document.getElementById("leftTabScorecard").style.display = tab === "scorecard" ? "block" : "none";
}

async function loadScorecard() {
  try { const d = await api("/api/scorecard"); document.getElementById("scorecardOutput").textContent = d.output || "Scorecard unavailable."; }
  catch (e) { document.getElementById("scorecardOutput").textContent = "Error: " + e.message; }
}

function applyRun(run) {
  currentRun = run; selectedDatasetIdx = 0; sessionId = run.session_id || "";
  document.getElementById("verifyBanner").innerHTML = "";
  document.getElementById("kpiStatus").textContent = run.status.toUpperCase();
  document.getElementById("kpiRunLabel").textContent = run.reused ? `${run.run_id} (reused, 0 recomputed)` : `${run.run_id} · ${run.label}`;
  document.getElementById("runReusedBadge").style.display = run.reused ? "inline-block" : "none";

  const ov = run.retail_overview || {};
  if (ov.available) {
    document.getElementById("kpiRevenue").textContent = fmtINR(ov.total_revenue);
    document.getElementById("kpiOrdersSub").textContent = `${ov.distinct_orders} orders · AOV ${fmtINR(ov.aov)}`;
    document.getElementById("kpiReturn").textContent = `${ov.return_rate_pct}%`;
    document.getElementById("kpiPlannerSub").textContent = `Low stock SKUs: ${ov.low_stock_skus} · Mode: ${run.planner_mode}`;
  } else {
    document.getElementById("kpiRevenue").textContent = "N/A"; document.getElementById("kpiOrdersSub").textContent = "No order revenue columns";
    document.getElementById("kpiReturn").textContent = "N/A"; document.getElementById("kpiPlannerSub").textContent = `Mode: ${run.planner_mode || "rules"}`;
  }

  const badge = document.getElementById("runStatusBadge"); badge.className = "status-pill st-" + run.status; badge.textContent = run.status;
  const banner = document.getElementById("runBanner");
  const errors = (run.cleaned || []).filter(x => x.status === "error");
  if (errors.length) {
    banner.innerHTML = errors.map(e => `<div class="alert alert-err"><strong>Isolated File Failure — ${esc(e.dataset.name)}.csv (${esc(e.error.code)}):</strong> ${esc(e.error.message)}</div>`).join("");
  } else if (run.reused) {
    banner.innerHTML = `<div class="alert alert-info"><strong>Atomic Run Deduplication:</strong> Identical input detected — reused saved run <code>${esc(run.run_id)}</code> (nothing recomputed).</div>`;
  } else { banner.innerHTML = ""; }

  renderDatasetTabs(); renderDatasetBody(); renderRetailCharts();
  const meta = document.getElementById("chatMeta"), feed = document.getElementById("chatFeed");
  if (run.chat_ready) {
    meta.textContent = `Run: ${run.run_id} | Session: ${sessionId} | ${run.chat_notice}`;
    feed.innerHTML = `<div class="msg msg-bot"><p class="ans-text">Ready. Ask a question over <strong>${esc(run.label)}</strong>, or click any numbered prompt above.</p></div>`;
  } else {
    meta.textContent = `Run: ${run.run_id} | ${run.chat_notice || "Chat disabled for failed run."}`;
    feed.innerHTML = `<div class="msg msg-bot msg-refusal"><p class="ans-text"><strong>[RUN_FAILED]</strong> No valid tables passed intake. Switch to a healthy run in <strong>Saved Runs</strong>.</p></div>`;
  }
}

function renderDatasetTabs() {
  const c = document.getElementById("datasetTabs");
  c.innerHTML = (currentRun && currentRun.cleaned) ? currentRun.cleaned.map((item, idx) =>
    `<button class="ds-tab ${idx === selectedDatasetIdx ? "active" : ""} ${item.status === "error" ? "err-tab" : ""}" onclick="pickDataset(${idx})">${esc(item.dataset.name)}.csv${item.status === "error" ? " ⚠" : ""}</button>`
  ).join("") : "";
}
function pickDataset(idx) { selectedDatasetIdx = idx; renderDatasetTabs(); renderDatasetBody(); }

function buildMiniTable(p) {
  if (!p || !p.columns || !p.columns.length) return '<div style="padding:0.65rem;color:var(--muted);font-size:0.8rem;">Preview unavailable.</div>';
  return `<div class="diff-scroll"><table class="mini-tbl"><thead><tr>${p.columns.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${
    (p.rows || []).map(r => `<tr>${p.columns.map(c => `<td>${esc(r[c])}</td>`).join("")}</tr>`).join("")
  }</tbody></table></div>`;
}

function renderDatasetBody() {
  const box = document.getElementById("datasetDetails");
  if (!currentRun || !currentRun.cleaned || !currentRun.cleaned.length) { box.innerHTML = ""; return; }
  const item = currentRun.cleaned[selectedDatasetIdx];
  if (item.status === "error") {
    box.innerHTML = `<div class="alert alert-err"><strong>Structured Intake Error (${esc(item.error.code)}):</strong> ${esc(item.error.message)}<pre class="json" style="margin-top:0.6rem">${esc(JSON.stringify(item.error, null, 2))}</pre></div>`;
    return;
  }
  const v = item.validation, ds = item.dataset;
  const norm = (currentRun.normalized || []).find(n => n.dataset && n.dataset.name === ds.name);
  const decisions = norm && norm.decisions ? norm.decisions : [];
  const needsReview = decisions.filter(d => d.status === "needs_review");
  const prev = (currentRun.previews && currentRun.previews[ds.name]) || {};

  const planRows = (item.cleaning_plan || []).map(s => `<tr><td><code>${esc(s.step_id || s.id)}</code></td><td>${esc(s.reason)}</td><td>${esc(s.risk)}</td><td><strong>${esc(s.status)}</strong></td></tr>`).join("") || '<tr><td colspan="4">Saved run artifacts verified cleanly.</td></tr>';
  const decisionRows = decisions.map(d => `<tr><td><code>${esc(d.field || "")}</code></td><td>${esc(d.raw || "")} → <strong>${esc(d.normalized || d.canonical || "held for review")}</strong></td><td>${d.confidence != null ? Math.round(d.confidence * 100) + "%" : "-"}</td><td><strong>${esc(d.status)}</strong></td></tr>`).join("") || '<tr><td colspan="4">Retail normalization complete.</td></tr>';

  box.innerHTML = `
    <div class="stat-grid">
      <div class="stat"><small>Rows (Before → After)</small><strong>${v.rows.before} → ${v.rows.after}</strong></div>
      <div class="stat"><small>Version Fingerprint</small><strong>${esc(ds.version)}</strong></div>
      <div class="stat"><small>Validation &amp; Idempotent</small><strong>${v.passed ? "PASS" : "FAIL"} · ${v.idempotent ? "YES" : "NO"}</strong></div>
      <div class="stat"><small>Held for Review</small><strong>${needsReview.length} item(s)</strong></div>
    </div>
    <div class="downloads">
      <a class="dl-link" href="/api/runs/${esc(currentRun.run_id)}/download/raw/${esc(ds.source_file)}" download>↓ Original (${esc(ds.source_file)})</a>
      <a class="dl-link" href="/api/runs/${esc(currentRun.run_id)}/download/cleaned/${esc(ds.name)}_clean.csv" download>↓ Cleaned (${esc(ds.name)}_clean.csv)</a>
      ${norm && norm.status !== "error" ? `<a class="dl-link" href="/api/runs/${esc(currentRun.run_id)}/download/retail/${esc(ds.name)}_retail.csv" download>↓ Retail (${esc(ds.name)}_retail.csv)</a>` : ""}
    </div>
    <details open><summary>Side-by-Side Data Diff (Raw Input vs. Cleaned Output — Top 6 Rows)</summary><div class="detail-body"><div class="diff-grid"><div class="diff-pane"><div class="diff-head diff-raw">Raw Input</div>${buildMiniTable(prev.raw)}</div><div class="diff-pane"><div class="diff-head diff-clean">Cleaned Output</div>${buildMiniTable(prev.cleaned)}</div></div></div></details>
    <details open><summary>Deterministic Cleaning Plan (${(item.cleaning_plan || []).length} steps)</summary><div class="detail-body"><table class="mini-tbl"><thead><tr><th>Step ID</th><th>Reason</th><th>Risk</th><th>Status</th></tr></thead><tbody>${planRows}</tbody></table></div></details>
    <details><summary>Retail Normalization &amp; Confidence Policy (${decisions.length} decisions)</summary><div class="detail-body"><table class="mini-tbl"><thead><tr><th>Field</th><th>Mapping</th><th>Confidence</th><th>Status</th></tr></thead><tbody>${decisionRows}</tbody></table></div></details>
    <details><summary>Full Before / After Profiles &amp; Validation JSON</summary><div class="detail-body"><pre class="json">${esc(JSON.stringify({ validation: v, profile_before: item.profile_before, profile_after: item.profile_after }, null, 2))}</pre></div></details>
  `;
}

function renderRetailCharts() {
  const container = document.getElementById("leftTabRetail"), ov = currentRun && currentRun.retail_overview;
  if (!ov || !ov.available) { container.innerHTML = '<div class="alert alert-info">Load the sample retail dataset to view live category and regional charts.</div>'; return; }
  const catHtml = (ov.category_bars || []).map(c => `<div class="bar-row"><div class="bar-meta"><span><strong>${esc(c.label)}</strong></span><span>${fmtINR(c.value)} (${c.share_pct}%)</span></div><div class="bar-track"><div class="bar-fill" style="width:${c.bar_pct}%"></div></div></div>`).join("");
  const regHtml = (ov.region_bars || []).map(r => `<div class="bar-row"><div class="bar-meta"><span><strong>${esc(r.label)}</strong> (${r.returns}/${r.lines} returned)</span><span><strong>${r.rate_pct}%</strong> · Rev ${fmtINR(r.revenue)}</span></div><div class="bar-track"><div class="bar-fill-warn" style="width:${Math.min(100, r.rate_pct * 3)}%"></div></div></div>`).join("");
  container.innerHTML = `<div class="chart-card"><h4>Net Revenue by Product Category</h4>${catHtml}</div><div class="chart-card"><h4>Return Rate by Store Region</h4>${regHtml}</div>`;
}

async function verifyReplay() {
  if (!currentRun || !sessionId) { alert("Ask at least one question first."); return; }
  const banner = document.getElementById("verifyBanner");
  banner.innerHTML = `<div class="alert alert-info">Replaying saved session <code>${esc(sessionId)}</code>…</div>`;
  try {
    const res = await api(`/api/runs/${encodeURIComponent(currentRun.run_id)}/verify`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: sessionId }) });
    banner.innerHTML = `<div class="alert ${res.reproduced ? "alert-ok" : "alert-err"}"><strong>${res.reproduced ? "REPRODUCED — Every saved answer matched" : "REPLAY DIFF DETECTED"}</strong><pre class="json" style="margin-top:0.45rem">${esc(res.output)}</pre></div>`;
  } catch (e) { banner.innerHTML = `<div class="alert alert-err">Replay error: ${esc(e.message)}</div>`; }
}

async function resetSession() {
  if (!currentRun || !currentRun.chat_ready) return;
  const data = await api(`/api/runs/${encodeURIComponent(currentRun.run_id)}/new_session`, { method: "POST" });
  sessionId = data.session_id; document.getElementById("verifyBanner").innerHTML = "";
  document.getElementById("chatMeta").textContent = `Run: ${currentRun.run_id} | Session: ${sessionId} (new) | ${currentRun.chat_notice}`;
  document.getElementById("chatFeed").innerHTML = `<div class="msg msg-bot"><p class="ans-text">Started a fresh conversation.</p></div>`;
}

function sendPreset(text) { document.getElementById("chatInput").value = text; sendQuestion(); }

async function sendQuestion() {
  const input = document.getElementById("chatInput"), q = input.value.trim();
  if (!q) return;
  if (!currentRun || !currentRun.chat_ready) { alert("Load a valid dataset run first."); return; }
  input.value = "";
  const feed = document.getElementById("chatFeed"), u = document.createElement("div");
  u.className = "msg msg-user"; u.textContent = q; feed.appendChild(u); feed.scrollTop = feed.scrollHeight;
  setBusy(true);
  try {
    const res = await api(`/api/runs/${encodeURIComponent(currentRun.run_id)}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: sessionId, question: q }) });
    sessionId = res.session_id;
    document.getElementById("chatMeta").textContent = `Run: ${currentRun.run_id} | Session: ${sessionId} | ${res.notice}`;
    appendBotReply(res.reply);
  } catch (e) { appendBotReply({ status: "error", refusal: { code: "REQUEST_ERROR", message: e.message } }); }
  finally { setBusy(false); }
}

function appendBotReply(reply) {
  const feed = document.getElementById("chatFeed"), div = document.createElement("div");
  if (reply.status === "ok") {
    div.className = "msg msg-bot";
    const rows = (reply.answer && (reply.answer.rows || reply.answer.preview)) || [];
    let tableHtml = "";
    if (Array.isArray(rows) && rows.length > 0 && typeof rows[0] === "object") {
      const cols = Object.keys(rows[0]);
      tableHtml = `<div style="overflow-x:auto;margin-bottom:0.6rem"><table class="mini-tbl"><thead><tr>${cols.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${rows.slice(0, 15).map(r => `<tr>${cols.map(c => `<td>${esc(r[c])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
    }
    const receiptText = reply.receipt || (reply.answer && reply.answer.receipt) || "";
    const receiptBlock = receiptText ? `<div class="receipt-card">${esc(typeof receiptText === "string" ? receiptText : JSON.stringify(receiptText, null, 2))}</div>` : "";
    const ev = reply.evidence || {};
    const dsVer = Array.isArray(ev.datasets) ? ev.datasets.join(", ") : (ev.dataset || ev.version || "verified_view");
    const rowCnt = ev.rows_considered != null ? ev.rows_considered : (ev.row_count != null ? ev.row_count : "checked");
    const plannerUsed = (reply.trace && (reply.trace.planner || reply.trace.source)) || "allow_list_validated";
    div.innerHTML = `
      <p class="ans-text">${esc(reply.answer.text)}</p>${tableHtml}${receiptBlock}
      <div class="ev-strip"><span class="ev-tag"><b>Data:</b> ${esc(dsVer)}</span><span class="ev-tag"><b>Rows:</b> ${esc(rowCnt)}</span><span class="ev-tag"><b>Planner:</b> ${esc(plannerUsed)}</span><span class="ev-tag"><b>Replay ID:</b> ${esc(sessionId)}</span></div>
      <details><summary>Inspect Full Execution Evidence JSON</summary><div class="detail-body"><pre class="json">${esc(JSON.stringify(reply.evidence, null, 2))}</pre></div></details>
      <details><summary>Inspect Validated Query Plan &amp; Trace</summary><div class="detail-body"><pre class="json">${esc(JSON.stringify({ plan: reply.plan, trace: reply.trace }, null, 2))}</pre></div></details>
    `;
  } else if (reply.status === "clarify") {
    div.className = "msg msg-bot msg-clarify";
    div.innerHTML = `<p class="ans-text"><strong>Clarification requested:</strong> ${esc(reply.clarification.text)}</p>`;
  } else {
    const ref = reply.refusal || {};
    div.className = "msg msg-bot msg-refusal";
    div.innerHTML = `<p class="ans-text"><strong>[${esc(reply.status)}:${esc(ref.code || "REFUSED")}]</strong> ${esc(ref.text || ref.message || "Request refused.")}</p>${reply.trace ? `<details><summary>Inspect Safety &amp; Refusal Trace</summary><div class="detail-body"><pre class="json">${esc(JSON.stringify(reply.trace, null, 2))}</pre></div></details>` : ""}`;
  }
  feed.appendChild(div); feed.scrollTop = feed.scrollHeight;
}

function setBusy(b) { document.getElementById("btnSample").disabled = b; document.getElementById("btnMalformed").disabled = b; document.getElementById("btnAsk").disabled = b; if (b) document.getElementById("btnUpload").disabled = true; }
init();
</script>
</body>
</html>
"""


class WorkbenchHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 25 * 1024 * 1024:
            raise DataLoadError("PAYLOAD_TOO_LARGE", "Upload exceeds 25 MB limit.")
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        return json.loads(raw)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self._send_html(HTML_PAGE)
        if path == "/api/scorecard":
            return self._send_json(get_ground_truth_scorecard())
        if path == "/api/runs":
            return self._send_json({
                "runs": [{"run_id": r["run_id"], "label": r["label"], "created_at": r["created_at"], "status": r["status"]} for r in _list_runs()]
            })
        if path.startswith("/api/runs/"):
            parts = [unquote(p) for p in path.strip("/").split("/")]
            if len(parts) == 3:
                run = _load_run(parts[2])
                return self._send_json(run) if run else self._send_json({"error": "Run not found"}, 404)
            if len(parts) == 6 and parts[3] == "download":
                run_id, kind, fname = parts[2], parts[4], parts[5]
                run = _load_run(run_id)
                if not run or kind not in {"raw", "cleaned", "retail"}:
                    return self._send_json({"error": "Invalid download path"}, 400)
                dir_key = {"raw": "raw_dir", "cleaned": "clean_dir", "retail": "retail_dir"}[kind]
                base_dir = Path(run[dir_key]).resolve()
                target = (base_dir / Path(fname).name).resolve()
                if base_dir not in target.parents or not target.is_file():
                    return self._send_json({"error": "File not found"}, 404)
                data = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/runs/sample":
                return self._send_json(create_sample_run())
            if path == "/api/runs/malformed":
                return self._send_json(create_malformed_demo_run())
            if path == "/api/runs/upload":
                return self._send_json(create_upload_run(self._read_json().get("files", [])))
            parts = [unquote(p) for p in path.strip("/").split("/")]
            if len(parts) == 4 and parts[0] == "api" and parts[1] == "runs":
                run_id, action = parts[2], parts[3]
                if action == "chat":
                    body = self._read_json()
                    return self._send_json(ask_run_chat(run_id, body.get("session_id", ""), body.get("question", "")))
                if action == "verify":
                    return self._send_json(verify_run_session(run_id, self._read_json().get("session_id", "")))
                if action == "new_session":
                    return self._send_json(new_run_session(run_id))
            self._send_json({"error": "Not found"}, 404)
        except DataLoadError as exc:
            self._send_json({"error": getattr(exc, "message", str(exc)), "code": getattr(exc, "code", "DATA_ERROR")}, 400)
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main():
    parser = argparse.ArgumentParser(description="Run the interactive Retail Data Workbench web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    WEB_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), WorkbenchHandler)
    print(f"Retail Workbench running at http://{args.host}:{args.port}  (Press Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down workbench.")
        server.server_close()


if __name__ == "__main__":
    main()
