"""
BATCH EVALUATOR - the command reviewers run on hidden datasets, no UI needed.

    python -m app.evaluate --input cases.json --output results.json --artifacts-dir ./evaluation_output
    optional:  --rules-only   never call a model, even if a key is configured

It calls the SAME functions the app uses:  clean_file -> normalize_file -> Catalog -> Assistant.

cases.json is a list of cases (or {"cases": [...]}). A case:

  {"id": "shop_1",
   "datasets": {"orders": "path/orders.csv", "products": "path/products.csv"},   # or "input": "one.csv", or "folder": "dir"
   "questions": ["Revenue by region",
                 {"question": "among those, which has the highest return rate?", "session": "main",
                  "expect": {"status": "ok", "first_row": {"region": "West"}}}]}

Per-case output (the PDF's minimum shape, plus extras). With ONE dataset the keys are flat
(dataset, profile_before, ...). With several datasets each key holds a map {dataset_name: ...};
issues and cleaning_plan stay flat lists because every id already contains the dataset name.

  dataset, profile_before, issues, cleaning_plan, profile_after, validation,
  chat_evaluation [{question, plan, answer, trace, status, evidence, checks, passed}],
  artifacts {relative_path: sha256}, retail_decisions, errors, status

Guarantees
  * One failed case (or dataset, or question) never aborts the batch: it becomes a structured error.
  * Output has no timestamps or timings, so a repeat run on the same data (and the same model replies)
    gives the same JSON.
  * Every case writes only inside its own folder under --artifacts-dir.
"""
import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, ValidationError

from app.chat.catalog import Catalog
from app.chat.factory import build_assistant
from app.chat.store import ChatStore
from app.cleaning.pipeline import clean_file
from app.config import Settings, load_settings
from app.errors import DataLoadError
from app.retail.normalize import normalize_file

EVALUATOR_VERSION = "1.0"
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
VOLATILE_KEYS = {"timing_ms", "elapsed_ms", "time", "created_at"}


class CaseSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")          # extra keys such as "description" are fine
    id: str
    datasets: Optional[Dict[str, str]] = None
    input: Optional[str] = None
    folder: Optional[str] = None
    questions: List[Union[str, Dict[str, Any]]] = []


# ------------------------------------------------------------------- helpers
def strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: strip_volatile(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [strip_volatile(v) for v in obj]
    return obj


def safe_case_dir_name(case_id):
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", case_id) and case_id not in (".", ".."):
        return case_id
    return "case_" + hashlib.sha256(case_id.encode()).hexdigest()[:12]


def error_dict(code, message, **details):
    return {"code": code, "message": message, "details": details}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_datasets(spec):
    """Returns ({name: Path}, [errors]). Names are restricted so they are safe in file names."""
    found, errors = {}, []
    if spec.datasets:
        items = list(spec.datasets.items())
    elif spec.input:
        items = [(Path(spec.input).stem, spec.input)]
    elif spec.folder:
        folder = Path(spec.folder)
        items = [(p.stem, str(p)) for p in sorted(folder.glob("*.csv"))] if folder.is_dir() else []
        if not items:
            errors.append(error_dict("NO_DATASETS", f"No .csv files found in folder '{spec.folder}'"))
    else:
        items = []
        errors.append(error_dict("NO_DATASETS", "A case needs 'datasets', 'input' or 'folder'"))
    for name, path in items:
        clean_name = re.sub(r"[^a-z0-9_]", "_", str(name).lower())
        clean_name = re.sub(r"^[^a-z]+", "d_", clean_name)[:41]
        if not NAME_RE.match(clean_name):
            errors.append(error_dict("BAD_DATASET_NAME", f"Dataset name '{name}' cannot be used", dataset=str(name)[:40]))
        elif clean_name in found:
            errors.append(error_dict("DUPLICATE_DATASET", f"Two datasets map to the name '{clean_name}'"))
        else:
            found[clean_name] = Path(path)
    return found, errors


# -------------------------------------------------------------- expectations
def _close(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=0.011)
    return a == b


def _subset(actual, wanted):
    return all(k in actual and _close(actual[k], v) for k, v in wanted.items())


def check_expectations(reply, expect):
    """Compare a chat reply with what the case author expected. Returns a list of named checks."""
    checks = []

    def add(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    rows = (reply.get("answer") or {}).get("rows", []) if reply.get("status") == "ok" else []
    if "status" in expect:
        add("status", reply.get("status") == expect["status"], f"got {reply.get('status')}, wanted {expect['status']}")
    if "refusal_code" in expect:
        got = (reply.get("refusal") or {}).get("code")
        add("refusal_code", got == expect["refusal_code"], f"got {got}")
    if "first_row" in expect:
        ok = bool(rows) and _subset(rows[0], expect["first_row"])
        add("first_row", ok, f"got {rows[0] if rows else None}")
    if "rows_include" in expect:
        for want in expect["rows_include"]:
            add("rows_include", any(_subset(r, want) for r in rows), f"wanted {want}")
    if "row_count" in expect:
        add("row_count", len(rows) == expect["row_count"], f"got {len(rows)}")
    if "contains_text" in expect:
        text = ((reply.get("answer") or {}).get("text") or (reply.get("refusal") or {}).get("message")
                or (reply.get("clarification") or {}).get("text") or "").lower()
        for piece in expect["contains_text"]:
            add("contains_text", piece.lower() in text, f"wanted '{piece}'")
    if "plan_filters_include" in expect:
        filters = (reply.get("plan") or {}).get("filters", [])
        for want in expect["plan_filters_include"]:
            add("plan_filters_include", any(_subset({k: v for k, v in f.items() if v is not None}, want) for f in filters),
                f"wanted {want}")
    if "evidence_dataset" in expect:
        got = (reply.get("evidence") or {}).get("dataset")
        add("evidence_dataset", got == expect["evidence_dataset"], f"got {got}")
    if "rows_after_filters" in expect:
        got = (reply.get("evidence") or {}).get("rows_after_filters")
        add("rows_after_filters", got == expect["rows_after_filters"], f"got {got}")
    if "clarification_options_include" in expect:
        opts = [o.lower() for o in (reply.get("clarification") or {}).get("options", [])]
        for want in expect["clarification_options_include"]:
            add("clarification_option", any(want.lower() in o for o in opts), f"options {opts}")
    return checks


# ------------------------------------------------------------------- chat
def run_chat(case_id, questions, catalog, settings, rules_only):
    """Run the questions through the real Assistant. Returns the chat_evaluation list."""
    store = ChatStore(":memory:")
    if rules_only:
        settings = Settings()
    assistant, notice = build_assistant(catalog, store, settings=settings)
    out = []
    for i, q in enumerate(questions, 1):
        spec = q if isinstance(q, dict) else {"question": q}
        text = spec.get("question")
        entry = {"question": text if isinstance(text, str) else str(text)[:200]}
        try:
            if not isinstance(text, str):
                raise ValueError("question must be text")
            session = f"eval-{case_id}-{spec.get('session', 'main')}"
            reply = assistant.ask(session, text)
            answer = reply.get("answer") or reply.get("refusal") or reply.get("clarification")
            entry.update({"plan": reply.get("plan"), "answer": strip_volatile(answer), "evidence": strip_volatile(reply.get("evidence")),
                          "trace": strip_volatile(reply.get("trace")), "status": reply["status"],
                          "warnings": reply.get("warnings", [])})
            if isinstance(spec.get("expect"), dict):
                checks = check_expectations(reply, spec["expect"])
                entry["checks"] = checks
                entry["passed"] = all(c["passed"] for c in checks)
        except Exception as exc:                          # one bad question never stops the others
            entry.update({"plan": None, "answer": None, "evidence": None, "trace": {}, "status": "error",
                          "error": error_dict("QUESTION_FAILED", f"{type(exc).__name__}: {exc}"[:200])})
            if isinstance(spec.get("expect"), dict):
                entry["checks"], entry["passed"] = [], False
        out.append(entry)
    return out


# -------------------------------------------------------------------- one case
def run_case(raw_case, artifacts_root, settings, rules_only=False):
    case_id = str(raw_case.get("id", "")) if isinstance(raw_case, dict) else ""
    try:
        if not isinstance(raw_case, dict):
            raise ValueError("a case must be a JSON object")
        spec = CaseSpec.model_validate(raw_case)
    except (ValidationError, ValueError) as exc:
        return {"id": case_id or None, "status": "error",
                "error": error_dict("INVALID_CASE", "The case is not in the expected format", detail=str(exc)[:300])}
    try:
        return _run_valid_case(spec, artifacts_root, settings, rules_only)
    except Exception as exc:                              # last line of defence: the batch continues
        return {"id": spec.id, "status": "error",
                "error": error_dict("CASE_FAILED", f"{type(exc).__name__}: {exc}"[:300])}


def _run_valid_case(spec, artifacts_root, settings, rules_only):
    case_dir = (artifacts_root / safe_case_dir_name(spec.id)).resolve()
    if artifacts_root.resolve() not in case_dir.parents:
        raise ValueError("case folder would be outside the artifacts folder")
    if case_dir.exists():
        shutil.rmtree(case_dir)                           # no stale files from an earlier run
    case_dir.mkdir(parents=True)

    datasets, errors = resolve_datasets(spec)
    cleaned, ds_info, before, after, validation, versions, decisions = {}, {}, {}, {}, {}, {}, []
    issues, plan = [], []
    for name, path in datasets.items():
        try:
            r = clean_file(path, case_dir / "clean", dataset_name=name)
        except DataLoadError as err:
            errors.append({**err.to_dict(), "details": {**err.details, "dataset": name}})
            continue
        except Exception as exc:
            errors.append(error_dict("CLEANING_FAILED", f"{type(exc).__name__}: {exc}"[:200], dataset=name))
            continue
        ds_info[name], before[name], after[name] = r["dataset"], r["profile_before"], r["profile_after"]
        validation[name] = r["validation"]
        issues += r["issues"]
        plan += r["cleaning_plan"]
        cleaned[name] = Path(r["artifacts"]["cleaned"])
        try:
            n = normalize_file(cleaned[name], case_dir / "retail")
            versions[name] = n["dataset"]["version"]
            decisions += [{**d, "dataset": name} for d in n["decisions"]]
            validation[name]["retail_normalization"] = n["validation"]
        except Exception as exc:
            errors.append(error_dict("NORMALIZATION_FAILED", f"{type(exc).__name__}: {exc}"[:200], dataset=name))

    chat = []
    if spec.questions:
        try:
            catalog = Catalog.from_folder(case_dir / "retail")
            chat = run_chat(safe_case_dir_name(spec.id), spec.questions, catalog, settings, rules_only)
        except Exception as exc:
            err = error_dict("CHAT_UNAVAILABLE", f"{type(exc).__name__}: {exc}"[:200])
            errors.append(err)
            chat = [{"question": (q if isinstance(q, str) else str(q.get("question")))[:200], "plan": None, "answer": None,
                     "trace": {}, "status": "error", "error": err} for q in spec.questions]
        (case_dir / "chat_evaluation.json").write_text(json.dumps(chat, indent=2), encoding="utf-8")

    artifacts = {}
    for path in sorted(case_dir.rglob("*")):
        if path.is_file():
            artifacts[path.relative_to(artifacts_root.resolve()).as_posix()] = sha256_file(path)

    single = len(datasets) == 1 and len(ds_info) == 1
    pick = (lambda d: next(iter(d.values()))) if single else (lambda d: d)
    status = "ok" if ds_info and not errors else ("partial" if ds_info else "error")
    result = {
        "id": spec.id, "status": status,
        "dataset": pick(ds_info) if ds_info else {},
        "profile_before": pick(before) if before else {},
        "issues": issues,
        "cleaning_plan": plan,
        "profile_after": pick(after) if after else {},
        "validation": pick(validation) if validation else {},
        "chat_evaluation": chat,
        "artifacts": artifacts,
        "retail_decisions": decisions,
        "versions": versions,
        "errors": errors,
    }
    return strip_volatile(result)


# ------------------------------------------------------------------ the batch
def _validations(results):
    """All per-dataset validation dicts, whether the case output is flat (one dataset) or keyed by name."""
    found = []
    for r in results:
        v = r.get("validation") or {}
        if "passed" in v:
            found.append(v)
        else:
            found += [x for x in v.values() if isinstance(x, dict) and "passed" in x]
    return found


def summarize(results):
    chat = [c for r in results for c in r.get("chat_evaluation", [])]
    checked = [c for c in chat if "passed" in c]
    passed = sum(1 for c in checked if c["passed"])
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "cases": len(results),
        "cases_ok": sum(1 for r in results if r["status"] == "ok"),
        "cases_partial": sum(1 for r in results if r["status"] == "partial"),
        "cases_error": sum(1 for r in results if r["status"] == "error"),
        "datasets_validated": len(_validations(results)),
        "datasets_passed_validation": sum(1 for v in _validations(results) if v.get("passed")),
        "chat_questions": len(chat),
        "chat_with_expectations": len(checked),
        "chat_passed": passed,
        "chat_accuracy": round(passed / len(checked), 4) if checked else None,
    }


def evaluate(cases, artifacts_dir, settings=None, rules_only=False):
    artifacts_root = Path(artifacts_dir)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    settings = settings or load_settings()
    results = [run_case(c, artifacts_root, settings, rules_only) for c in cases]
    return {"summary": summarize(results), "cases": results}


def load_cases(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise DataLoadError("INPUT_NOT_FOUND", f"Input file not found: {path}")
    except (ValueError, UnicodeDecodeError) as exc:
        raise DataLoadError("INPUT_NOT_JSON", f"Input file is not valid JSON: {str(exc)[:120]}")
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list):
        raise DataLoadError("INPUT_SHAPE", "Input must be a JSON list of cases, or an object with a 'cases' list")
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m app.evaluate", description="Run evaluation cases through the production pipeline.")
    ap.add_argument("--input", required=True, help="cases.json")
    ap.add_argument("--output", required=True, help="where to write results.json")
    ap.add_argument("--artifacts-dir", default="./evaluation_output", help="folder for cleaned files and reports")
    ap.add_argument("--rules-only", action="store_true", help="never call a model")
    args = ap.parse_args(argv)
    try:
        cases = load_cases(args.input)
    except DataLoadError as err:
        print(json.dumps(err.to_dict(), indent=2))
        return 2
    output = evaluate(cases, args.artifacts_dir, rules_only=args.rules_only)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    s = output["summary"]
    print(f"Evaluated {s['cases']} case(s): {s['cases_ok']} ok, {s['cases_partial']} partial, {s['cases_error']} error.")
    if s["chat_with_expectations"]:
        print(f"Chat questions with expected answers: {s['chat_passed']}/{s['chat_with_expectations']} passed "
              f"(accuracy {s['chat_accuracy']:.0%}).")
    print(f"Results: {args.output}   Artifacts: {args.artifacts_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
