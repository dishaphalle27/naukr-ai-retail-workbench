"""
PREPARE - one command that gets raw CSVs ready for chat, and saves the work as a RUN.

    python -m app.prepare data/sample

  1. fingerprints the input files and creates (or reuses) a run:  data/runs/<run-id>/
        original/  cleaned/  retail/  run.json (status, metadata, hashes)
  2. copies the finished files to  data/cleaned  and  data/retail  so the chat can read them
  3. prints a short summary

Same input twice = the existing run is reused (nothing is recomputed). Use --force to rebuild.
Exit code 0 = succeeded. 2 = partial / failed (see the lines above).
"""
import argparse
import shutil
import sys
from pathlib import Path

from app.errors import DataLoadError
from app.runs import create_run


def _publish(run_path, name, target):
    target = Path(target)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(Path(run_path) / name, target)


def prepare(folder, clean_dir="data/cleaned", retail_dir="data/retail", runs_dir="data/runs", force=False):
    run = create_run(folder, runs_dir, force=force)
    if run["status"] in ("succeeded", "partial"):
        _publish(run["path"], "cleaned", clean_dir)
        _publish(run["path"], "retail", retail_dir)
    return run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m app.prepare", description="Clean and normalize every CSV in a folder, as a saved run.")
    ap.add_argument("folder", help="folder with raw .csv files")
    ap.add_argument("--clean-dir", default="data/cleaned")
    ap.add_argument("--retail-dir", default="data/retail")
    ap.add_argument("--runs-dir", default="data/runs")
    ap.add_argument("--force", action="store_true", help="rebuild even if this exact input was already prepared")
    args = ap.parse_args(argv)
    try:
        run = prepare(args.folder, args.clean_dir, args.retail_dir, args.runs_dir, args.force)
    except DataLoadError as err:
        print(err.message)
        return 2
    print(f"RUN {run['run_id']}  status: {run['status']}  " + ("(same input as before: reused, nothing recomputed)" if run["reused"] else "(new)"))
    print(f"    {run['path']}/  original/  cleaned/  retail/  run.json")
    for name, d in sorted(run["datasets"].items()):
        steps = ", ".join(f"{n} {s}" for s, n in sorted(d["steps"].items()))
        mark = "PASS" if d["validation_passed"] and d["retail_validation_passed"] else "FAIL"
        print(f"  {name:<10} rows {d['rows_before']:>5} -> {d['rows_after']:>5} | {steps or 'no steps'} | "
              f"{d['needs_review_decisions']} alias(es) need review | validation {mark} | {d['retail_version']}")
    for e in run["errors"]:
        print(f"  ERROR {e.get('dataset', '')}: [{e['code']}] {e['message']}")
    if run["status"] == "succeeded":
        print(f"\nAll checks passed. Start the chat with:  python -m app.chat.cli {args.retail_dir}")
        return 0
    print("\nThis run is not fully clean (see lines above). Files that worked are still available to chat.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
