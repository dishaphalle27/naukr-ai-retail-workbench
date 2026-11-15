"""
PREPARE - one command that gets raw CSVs ready for chat.

    python -m app.prepare data/sample

It runs, in order:   clean (profile, plan, fix, validate)  ->  retail normalization
and writes   data/cleaned/*_clean.csv   and   data/retail/*_retail.csv   (plus JSON reports).
Then start the chat with:   python -m app.chat.cli data/retail

Exit code 0 = every file cleaned and passed validation. 2 = something needs attention.
"""
import argparse
import sys
from pathlib import Path

from app.cleaning.pipeline import clean_folder, summarize
from app.retail.normalize import normalize_folder


def prepare(folder, clean_dir="data/cleaned", retail_dir="data/retail"):
    cleaned = clean_folder(folder, clean_dir)
    normalized = normalize_folder(clean_dir, retail_dir)
    ok = bool(cleaned) and all(r.get("status") != "error" and r["validation"]["passed"] for r in cleaned) \
        and all(r.get("status") != "error" and r["validation"]["passed"] for r in normalized)
    return {"cleaned": cleaned, "normalized": normalized, "ok": ok}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m app.prepare", description="Clean and normalize every CSV in a folder.")
    ap.add_argument("folder", help="folder with raw .csv files")
    ap.add_argument("--clean-dir", default="data/cleaned")
    ap.add_argument("--retail-dir", default="data/retail")
    args = ap.parse_args(argv)
    if not Path(args.folder).is_dir():
        print(f"Folder not found: {args.folder}")
        return 2
    result = prepare(args.folder, args.clean_dir, args.retail_dir)
    if not result["cleaned"]:
        print(f"No .csv files found in {args.folder}")
        return 2
    print("CLEANING")
    for r in result["cleaned"]:
        print("  " + summarize(r))
    print("RETAIL NORMALIZATION")
    for r in result["normalized"]:
        if r.get("status") == "error":
            print(f"  {r['dataset']['name']:<10} ERROR {r['error']['code']}")
            continue
        review = [d for d in r["decisions"] if d["status"] == "needs_review"]
        applied = [d for d in r["decisions"] if d["status"] == "applied"]
        print(f"  {r['dataset']['name']:<10} {len(applied)} applied, {len(review)} needs_review | {r['dataset']['version']}")
    print(f"\nAll checks passed. Start the chat with:  python -m app.chat.cli {args.retail_dir}" if result["ok"]
          else "\nSome files need attention (see lines above).")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
