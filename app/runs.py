"""
RUNS - every preparation of a set of CSVs is saved as a RUN you can inspect later.

    data/runs/run-3f9a1c0d22ab/
        run.json        status, metadata, inputs (with sha256), per-dataset results, artifact hashes
        original/       byte-for-byte copies of the files you gave us
        cleaned/        *_clean.csv  and reports
        retail/         *_retail.csv and reports   <- what the chat reads

Rules
  * The run id is a fingerprint of the input FILES (names + bytes) and the pipeline version.
    Same input -> same run id -> the existing run is REUSED (idempotent). Change one byte -> a new run.
  * A run is built in a hidden temp folder and moved into place in one step. A crash can never leave
    a half-written run that looks finished.
  * Failures are saved too (status "failed" with the structured error), so they can be inspected.

Status:  succeeded  = every file cleaned, validated and normalized
         partial    = some files succeeded, some produced a structured error
         failed     = nothing usable, or an unexpected crash

    python -m app.runs list
    python -m app.runs show RUN_ID
"""
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.cleaning.pipeline import clean_folder
from app.errors import DataLoadError
from app.retail.normalize import normalize_folder

PIPELINE_VERSION = "1"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_id_for(files):
    pairs = [[p.name, _sha256(p)] for p in sorted(files)]
    digest = hashlib.sha256(json.dumps([PIPELINE_VERSION, pairs]).encode()).hexdigest()
    return "run-" + digest[:12]


def _artifact_hashes(root):
    return {p.relative_to(root).as_posix(): _sha256(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name != "run.json"}


def _summarize(cleaned, normalized):
    datasets, errors = {}, []
    by_norm = {r["dataset"]["name"]: r for r in normalized if r.get("status") != "error"}
    for r in cleaned:
        name = r["dataset"]["name"]
        if r.get("status") == "error":
            errors.append({**r["error"], "dataset": name})
            continue
        counts = {}
        for e in r["change_log"]:
            counts[e["status"]] = counts.get(e["status"], 0) + 1
        n = by_norm.get(name)
        datasets[name] = {
            "rows_before": r["validation"]["rows"]["before"], "rows_after": r["validation"]["rows"]["after"],
            "columns": r["dataset"]["columns"], "steps": counts,
            "validation_passed": r["validation"]["passed"],
            "cleaned_version": r["dataset"]["version"],
            "retail_version": n["dataset"]["version"] if n else None,
            "retail_validation_passed": n["validation"]["passed"] if n else False,
            "needs_review_decisions": sum(1 for d in (n["decisions"] if n else []) if d["status"] == "needs_review"),
        }
        if not n:
            errors.append({"code": "NORMALIZATION_FAILED", "message": "retail normalization did not run", "dataset": name})
    return datasets, errors


def create_run(folder, runs_dir="data/runs", force=False):
    """Prepare every CSV in `folder` as a persisted run. Returns the run.json dict (+ 'reused' and 'path')."""
    folder, runs_dir = Path(folder), Path(runs_dir)
    files = sorted(folder.glob("*.csv")) if folder.is_dir() else []
    if not files:
        raise DataLoadError("NO_DATASETS", f"No .csv files found in {folder}")
    run_id = run_id_for(files)
    final = runs_dir / run_id
    if final.exists() and not force:
        meta = json.loads((final / "run.json").read_text(encoding="utf-8"))
        if meta.get("status") in ("succeeded", "partial"):
            return {**meta, "reused": True, "path": str(final)}

    runs_dir.mkdir(parents=True, exist_ok=True)
    tmp = runs_dir / f".tmp-{run_id}"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "original").mkdir(parents=True)
    for p in files:
        shutil.copyfile(p, tmp / "original" / p.name)

    meta = {"run_id": run_id, "status": "failed", "pipeline_version": PIPELINE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "inputs": [{"file": p.name, "bytes": p.stat().st_size, "sha256": _sha256(p)} for p in files],
            "datasets": {}, "errors": []}
    try:
        cleaned = clean_folder(tmp / "original", tmp / "cleaned")
        normalized = normalize_folder(tmp / "cleaned", tmp / "retail")
        datasets, errors = _summarize(cleaned, normalized)
        meta["datasets"], meta["errors"] = datasets, errors
        all_ok = datasets and all(d["validation_passed"] and d["retail_validation_passed"] for d in datasets.values())
        if datasets and not errors and all_ok:
            meta["status"] = "succeeded"
        elif datasets:
            meta["status"] = "partial"
    except Exception as exc:                                    # a crash is saved as a failed run, not lost
        meta["errors"].append({"code": "RUN_CRASHED", "message": f"{type(exc).__name__}: {exc}"[:200]})
    meta["artifacts"] = _artifact_hashes(tmp)
    (tmp / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if final.exists():
        shutil.rmtree(final)
    tmp.replace(final)                                          # published in one step
    if meta["status"] in ("succeeded", "partial"):
        (runs_dir / "LATEST").write_text(run_id, encoding="utf-8")
    return {**meta, "reused": False, "path": str(final)}


def load_run(run_id, runs_dir="data/runs"):
    path = Path(runs_dir) / run_id / "run.json"
    if not path.is_file():
        raise DataLoadError("RUN_NOT_FOUND", f"No run called {run_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_runs(runs_dir="data/runs"):
    root = Path(runs_dir)
    runs = []
    for p in sorted(root.glob("run-*/run.json")) if root.is_dir() else []:
        m = json.loads(p.read_text(encoding="utf-8"))
        runs.append({"run_id": m["run_id"], "status": m["status"], "created_at": m["created_at"],
                     "files": [i["file"] for i in m["inputs"]]})
    return runs


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    runs_dir = "data/runs"
    if argv[:1] == ["list"]:
        for r in list_runs(runs_dir):
            print(f"{r['run_id']}  {r['status']:<10} {r['created_at']}  {', '.join(r['files'])}")
        return 0
    if argv[:1] == ["show"] and len(argv) > 1:
        try:
            print(json.dumps(load_run(argv[1], runs_dir), indent=2))
            return 0
        except DataLoadError as err:
            print(err.message)
            return 2
    print("Usage: python -m app.runs list | show RUN_ID")
    return 1


if __name__ == "__main__":
    sys.exit(main())
