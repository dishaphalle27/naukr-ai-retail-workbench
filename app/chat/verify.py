"""
VERIFY - replay a saved chat session and check that every computed answer REPRODUCES.

For each saved turn that computed something, the stored (validated) plan is run again against the
data that is loaded now, and compared with what was answered back then:

  reproduced   same data version, same rows           -> the answer can be trusted and re-checked
  changed      the data version differs, or the numbers differ -> says exactly which
  skipped      nothing was computed (a clarification, a refusal, a product-health explanation)

This is the practical meaning of "the answer is reproducible": not a promise, a check anyone can run.
"""
from app.chat.engine import run_plan

COMPUTED = {"aggregate", "list", "distinct", "describe"}


def verify_session(catalog, store, session_id):
    results = []
    for turn in store.trace(session_id):
        base = {"turn": turn["turn"], "question": turn["question"]}
        plan = turn.get("plan") or {}
        if turn["status"] != "ok" or plan.get("intent") not in COMPUTED:
            results.append({**base, "verdict": "skipped", "detail": f"nothing computed ({turn['status']})"})
            continue
        old_rows = (turn.get("answer") or {}).get("rows", [])
        old_version = (turn.get("evidence") or {}).get("version")
        out = run_plan(plan, catalog)
        if out["status"] != "ok":
            results.append({**base, "verdict": "changed", "detail": f"the saved plan no longer runs: {out['error']['code']}"})
            continue
        new_version = out["evidence"]["version"]
        if new_version != old_version:
            results.append({**base, "verdict": "changed", "detail": f"data version changed: {old_version} -> {new_version}"})
        elif out["result"]["rows"] != old_rows:
            results.append({**base, "verdict": "changed", "detail": "same data version but different numbers"})
        else:
            results.append({**base, "verdict": "reproduced", "detail": f"{len(old_rows)} row(s) identical on {new_version}"})
    return results
