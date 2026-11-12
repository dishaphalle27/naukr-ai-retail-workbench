"""
PIPELINE - the full cleaning flow for one CSV (or a folder of CSVs).

    load -> profile (before) -> find issues -> make plan -> clean
         -> profile (after) -> validate -> clean AGAIN to prove idempotency -> save

The evaluator and the app will both call clean_file(). One pipeline, no copies.

Run from the terminal:
    python -m app.cleaning.pipeline data/sample --out data/cleaned
    python -m app.cleaning.pipeline data/sample/orders.csv
"""
import hashlib
import json
import sys
from pathlib import Path

from app.cleaning.cleaner import clean_dataframe
from app.cleaning.issues import detect_issues
from app.cleaning.plan import build_plan
from app.cleaning.profiler import load_csv, profile_dataframe
from app.cleaning.validate import validate
from app.errors import DataLoadError


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_file(path, out_dir, dataset_name=None, max_mb=50):
    """Clean one CSV. Returns a result dict. Raises DataLoadError for unreadable files."""
    path, out_dir = Path(path), Path(out_dir)
    name = dataset_name or path.stem
    df, meta = load_csv(path, max_mb=max_mb)

    profile_before = profile_dataframe(df, meta)
    issues = detect_issues(name, profile_before)
    plan = build_plan(name, profile_before, issues)

    cleaned, log = clean_dataframe(name, df, plan)

    profile_after = profile_dataframe(cleaned, {"file_name": f"{name}_clean.csv"})
    issues_after = detect_issues(name, profile_after)

    # Second pass: clean the cleaned data. It must come out identical.
    plan2 = build_plan(name, profile_after, issues_after)
    cleaned2, _ = clean_dataframe(name, cleaned, plan2)
    idempotent = cleaned2.equals(cleaned)

    validation = validate(df, cleaned, plan, log, issues_after, idempotent)
    validation["second_pass_changing_steps"] = [s["step_id"] for s in plan2 if s["status"] == "planned"]

    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = out_dir / f"{name}_clean.csv"
    cleaned.to_csv(cleaned_path, index=False, encoding="utf-8", lineterminator="\n")

    result = {
        "dataset": {"name": name, "source_file": path.name, "rows": len(df), "columns": len(df.columns),
                    "version": f"{name}_clean@{_sha256(cleaned_path)[:12]}"},
        "profile_before": profile_before,
        "issues": issues,
        "cleaning_plan": plan,
        "change_log": log,
        "profile_after": profile_after,
        "issues_after": issues_after,
        "validation": validation,
        "artifacts": {"original": str(path), "original_sha256": _sha256(path),
                      "cleaned": str(cleaned_path), "cleaned_sha256": _sha256(cleaned_path)},
    }
    report_path = out_dir / f"{name}_report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["artifacts"]["report"] = str(report_path)
    return result


def clean_folder(folder, out_dir):
    """Clean every .csv in a folder. One bad file never stops the others."""
    results = []
    for path in sorted(Path(folder).glob("*.csv")):
        try:
            results.append(clean_file(path, out_dir))
        except DataLoadError as err:
            results.append({"dataset": {"name": path.stem}, "status": "error", "error": err.to_dict()})
        except Exception as exc:     # unexpected bug: report it, keep going
            results.append({"dataset": {"name": path.stem}, "status": "error",
                            "error": {"code": "UNEXPECTED_ERROR", "message": f"{type(exc).__name__}: {exc}"[:200]}})
    return results


def summarize(result):
    if result.get("status") == "error":
        e = result["error"]
        return f"{result['dataset']['name']:<10} ERROR {e['code']}: {e['message']}"
    v = result["validation"]
    counts = {}
    for e in result["change_log"]:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    steps = ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
    mark = "PASS" if v["passed"] else "FAIL"
    return (f"{result['dataset']['name']:<10} rows {v['rows']['before']:>5} -> {v['rows']['after']:>5} | "
            f"steps: {steps or 'none'} | validation {mark} | {result['dataset']['version']}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "data/cleaned"
    if "--out" in sys.argv:
        args = [a for a in args if a != out]
    if not args:
        print("Usage: python -m app.cleaning.pipeline <file.csv | folder> [--out data/cleaned]")
        sys.exit(1)
    target = Path(args[0])
    if target.is_dir():
        outcomes = clean_folder(target, out)
    else:
        try:
            outcomes = [clean_file(target, out)]
        except DataLoadError as err:
            outcomes = [{"dataset": {"name": target.stem}, "status": "error", "error": err.to_dict()}]
    for r in outcomes:
        print(summarize(r))
    sys.exit(0 if all(r.get("status") != "error" and r["validation"]["passed"] for r in outcomes) else 2)
