"""
CHAT STORE - saves every conversation so a reviewer can inspect it later:

      question  ->  plan  ->  execution evidence  ->  answer

Uses SQLite (built into Python, no server). Each session is separate: its filters,
last measure and pending clarification live in its own row, so one user can never see
another user's state.

Never stored: prompts to a model, API keys, or anything outside the response itself.
(The rule-based planner has no hidden prompt. When a model is added later, its prompt
will not be written here.)
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_STATE = {"filters": [], "metrics": None, "labels": {}, "units": {}, "group_by": [], "time_bucket": None,
                 "dataset": None, "last_plan": None, "pending": None}

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, versions TEXT NOT NULL, state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS turns (
    session_id TEXT NOT NULL, turn_no INTEGER NOT NULL, created_at TEXT NOT NULL,
    question TEXT NOT NULL, status TEXT NOT NULL, plan TEXT, answer TEXT, evidence TEXT, trace TEXT,
    PRIMARY KEY (session_id, turn_no));
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ChatStore:
    def __init__(self, path=":memory:"):
        self.path = str(path)
        self._memory = None
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        else:
            self._memory = sqlite3.connect(":memory:")
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        conn = self._memory or sqlite3.connect(self.path)
        return _Ctx(conn, keep_open=self._memory is not None)

    # ------------------------------------------------------------ sessions
    def new_session_id(self):
        return uuid.uuid4().hex[:12]

    def ensure_session(self, session_id, versions):
        with self._conn() as c:
            row = c.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                c.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)",
                          (session_id, _now(), json.dumps(versions), json.dumps(DEFAULT_STATE)))

    def get_state(self, session_id):
        with self._conn() as c:
            row = c.execute("SELECT state FROM sessions WHERE id = ?", (session_id,)).fetchone()
        state = json.loads(row[0]) if row else {}
        return {**DEFAULT_STATE, **state}

    def save_turn(self, session_id, question, response, state):
        with self._conn() as c:
            n = (c.execute("SELECT COALESCE(MAX(turn_no), 0) FROM turns WHERE session_id = ?", (session_id,)).fetchone()[0]) + 1
            c.execute("INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (session_id, n, _now(), question, response["status"],
                       json.dumps(response.get("plan")), json.dumps(response.get("answer") or response.get("refusal") or response.get("clarification")),
                       json.dumps(response.get("evidence")), json.dumps(response.get("trace"))))
            c.execute("UPDATE sessions SET state = ? WHERE id = ?", (json.dumps(state), session_id))
        return n

    def trace(self, session_id):
        """Everything a reviewer needs: for each turn the question, plan, evidence, answer and trace."""
        with self._conn() as c:
            rows = c.execute("SELECT turn_no, created_at, question, status, plan, answer, evidence, trace FROM turns "
                             "WHERE session_id = ? ORDER BY turn_no", (session_id,)).fetchall()
        return [{"turn": r[0], "time": r[1], "question": r[2], "status": r[3], "plan": json.loads(r[4]),
                 "answer": json.loads(r[5]), "evidence": json.loads(r[6]), "trace": json.loads(r[7])} for r in rows]

    def sessions(self):
        with self._conn() as c:
            return [{"id": r[0], "created_at": r[1], "versions": json.loads(r[2])}
                    for r in c.execute("SELECT id, created_at, versions FROM sessions ORDER BY created_at")]


class _Ctx:
    """Open, commit and close a connection (memory connections stay open)."""

    def __init__(self, conn, keep_open):
        self.conn, self.keep_open = conn, keep_open

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        if not self.keep_open:
            self.conn.close()
        return False
