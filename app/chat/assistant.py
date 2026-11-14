"""
ASSISTANT - answers one question at a time and remembers the conversation.

    assistant = Assistant(catalog, store)
    reply = assistant.ask(session_id, "Revenue by category")

Every reply has the same shape (also used by the evaluator later):
  status        ok | clarify | refused | rejected | error
  answer        {"text", "headline", "columns", "rows", "assumptions"}   (when ok)
  plan          the exact plan that was validated and executed
  evidence      dataset + version, columns, filters, joins, operations, row counts, formulas
  clarification {"question", "options"}                                  (when clarify)
  refusal       {"code", "message", "suggestions"}                       (when refused)
  trace         how the question was understood, step by step

The flow:   question -> planner (rules now, an AI later) -> plan
            plan -> VALIDATOR -> EXECUTOR -> result -> answer text built from the result
"""
import copy
import re

from app.chat.answers import build_answer
from app.chat.engine import run_plan
from app.chat.insights import attention_list, explain_product
from app.chat.rules_planner import EXAMPLES, parse_question
from app.errors import DataLoadError

MAX_QUESTION_CHARS = 500


def _clean_state(state):
    s = copy.deepcopy(state)
    s["pending"] = None
    return s


def _match_option(reply, options):
    """Which clarification option did the user pick? By number, or by keyword.
    The most specific (longest) matching keyword wins. If two options tie, return None (ask again)."""
    r = reply.strip().lower()
    if r.isdigit() and 1 <= int(r) <= len(options):
        return options[int(r) - 1]
    best, best_len, tie = None, 0, False
    for o in options:
        words = [o["label"].lower()] + [k.lower() for k in o.get("keywords", [])]
        lengths = [len(w) for w in words if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", r)]
        if not lengths:
            continue
        score = max(lengths)
        if score > best_len:
            best, best_len, tie = o, score, False
        elif score == best_len:
            tie = True
    return None if tie else best


class Assistant:
    def __init__(self, catalog, store, planner=None):
        self.catalog = catalog
        self.store = store
        self.planner = planner or parse_question          # a model-based planner can be plugged in here

    # ---------------------------------------------------------------- public
    def ask(self, session_id, question):
        self.store.ensure_session(session_id, {n: self.catalog.version(n) for n in self.catalog.datasets})
        state = self.store.get_state(session_id)
        question = question if isinstance(question, str) else ""
        trace = {"planner": "rules", "steps": []}

        if len(question) > MAX_QUESTION_CHARS:
            response = self._refused("QUESTION_TOO_LONG", f"Please keep questions under {MAX_QUESTION_CHARS} characters.", [], trace)
            return self._save(session_id, question[:80] + "...", response, state)
        if not question.strip():
            response = self._refused("EMPTY_QUESTION", "Please type a question about the retail data.", EXAMPLES[:4], trace)
            return self._save(session_id, "", response, state)

        effective, forced = question, None
        pending = state.get("pending")
        if pending:
            choice = _match_option(question, pending["options"])
            if choice is not None:
                effective = choice.get("rewrite") or pending["original"]
                forced = choice.get("forced") or {}
                state = {**state, **pending["state_before"]}
                trace["steps"].append(f"Clarification answered: '{choice['label']}'. Re-reading the question: {effective!r}")
            else:
                trace["steps"].append("The reply did not match any option, so it was treated as a new question.")
            state["pending"] = None

        outcome = self.planner(effective, self.catalog, state, forced)
        trace["steps"].append(f"Planner result: {outcome['kind']}")
        kind = outcome["kind"]

        if kind == "plan":
            response = self._run(outcome, state, trace)
        elif kind == "clarify":
            response = self._clarify(outcome, state, trace)
        elif kind == "refuse":
            response = self._refused(outcome["code"], outcome["message"], outcome["suggestions"], trace)
        elif kind == "command":
            response = self._command(outcome, state, trace)
        else:
            response = self._insight(outcome["insight"], state, trace)
        new_state = response.pop("_state", state)
        return self._save(session_id, question, response, new_state)

    # --------------------------------------------------------------- handlers
    def _run(self, outcome, state, trace):
        plan, meta = outcome["plan"], outcome["meta"]
        trace["steps"] += meta.get("notes", [])
        out = run_plan(plan, self.catalog)
        trace["validation"] = "passed" if out["status"] != "rejected" else out["error"]["details"]["problems"]
        trace["timing_ms"] = out["timing_ms"]
        if out["status"] == "rejected":
            problems = out["error"]["details"]["problems"]
            text = "I could not run that safely: " + "; ".join(p["message"] for p in problems)
            return {"status": "rejected", "answer": None, "plan": out["plan"], "evidence": None, "warnings": [],
                    "refusal": {"code": "PLAN_REJECTED", "message": text, "suggestions": EXAMPLES[:3]}, "trace": trace}
        if out["status"] == "error":
            return {"status": "error", "answer": None, "plan": out["plan"], "evidence": None, "warnings": [],
                    "refusal": {"code": out["error"]["code"], "message": out["error"]["message"], "suggestions": []},
                    "trace": trace}
        answer = build_answer(out["plan"], out["result"], out["evidence"], meta, out["warnings"])
        auto = meta.get("auto_filters", [])
        remembered = [f for f in out["plan"].get("filters", []) if f not in auto]
        new_state = {**state, "filters": remembered, "metrics": out["plan"].get("metrics"),
                     "labels": meta.get("labels", {}), "units": meta.get("units", {}),
                     "group_by": out["plan"].get("group_by", []), "time_bucket": out["plan"].get("time_bucket"),
                     "dataset": out["plan"]["dataset"], "last_plan": out["plan"], "pending": None}
        return {"status": "ok", "answer": answer, "plan": out["plan"], "evidence": out["evidence"],
                "warnings": out["warnings"], "trace": trace, "_state": new_state}

    def _clarify(self, outcome, state, trace):
        options = outcome["options"]
        text = outcome["question"] + "\n" + "\n".join(f"  {i}. {o['label']}" for i, o in enumerate(options, 1))
        pending = {"original": outcome["original"], "options": options, "state_before": _clean_state(state)}
        return {"status": "clarify", "answer": None, "plan": None, "evidence": None, "warnings": [],
                "clarification": {"question": outcome["question"], "options": [o["label"] for o in options], "text": text},
                "trace": trace, "_state": {**state, "pending": pending}}

    def _refused(self, code, message, suggestions, trace):
        text = message + (("\nYou could ask:\n" + "\n".join(f"  - {s}" for s in suggestions)) if suggestions else "")
        return {"status": "refused", "answer": None, "plan": None, "evidence": None, "warnings": [],
                "refusal": {"code": code, "message": message, "suggestions": suggestions, "text": text}, "trace": trace}

    def _command(self, outcome, state, trace):
        new_state = {**state, "filters": [], "pending": None}
        text = "Filters cleared. Your next question starts from all the data."
        return {"status": "ok", "answer": {"text": text, "headline": text, "columns": [], "rows": [], "assumptions": []},
                "plan": {"intent": "command", "command": outcome["command"]}, "evidence": None, "warnings": [],
                "trace": trace, "_state": new_state}

    def _insight(self, insight, state, trace):
        try:
            if insight["type"] == "attention":
                out = attention_list(self.catalog, insight["limit"])
            else:
                out = explain_product(self.catalog, insight["product_id"])
        except DataLoadError as err:
            return self._refused(err.code, err.message, [], trace)
        if out["status"] == "refused":
            r = out["refusal"]
            return self._refused(r["code"], r["message"], r["suggestions"], trace)
        out["trace"] = trace
        trace["steps"].append("Answered from the product health score (explicit signals, fixed weights).")
        return out

    # ------------------------------------------------------------------ save
    def _save(self, session_id, question, response, state):
        turn = self.store.save_turn(session_id, question, response, state)
        response["session_id"], response["turn"], response["question"] = session_id, turn, question
        return response
