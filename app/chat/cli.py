"""
CHAT IN THE TERMINAL

    python -m app.chat.cli data/retail                    interactive chat
    python -m app.chat.cli data/retail --ask "Revenue by category"
    python -m app.chat.cli data/retail --trace SESSION_ID  show a saved conversation

    python -m app.chat.cli data/retail --verify SESSION_ID   re-run a saved session and check every number reproduces

Inside the chat:  :plan  :evidence  :trace  :new  :quit
"""
import json
import sys

from app.chat.answers import format_receipt
from app.chat.factory import build_assistant
from app.chat.catalog import Catalog
from app.chat.store import ChatStore
from app.chat.verify import verify_session
from app.errors import DataLoadError


def show(reply):
    if reply["status"] == "ok":
        print(reply["answer"]["text"])
        receipt = format_receipt(reply)
        if receipt:
            print(receipt)
    elif reply["status"] == "clarify":
        print("? " + reply["clarification"]["text"])
    else:
        r = reply["refusal"]
        print(f"[{reply['status']}:{r['code']}] {r.get('text') or r['message']}")


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")      # rupee signs print on any Windows console
    except (AttributeError, ValueError):
        pass
    positional = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and not sys.argv[i - 1].startswith("--")]
    if not positional:
        print(__doc__)
        return 1
    try:
        catalog = Catalog.from_folder(positional[0])
    except DataLoadError as err:
        print(json.dumps(err.to_dict(), indent=2))
        return 2
    store = ChatStore(arg("--db", "data/chat/chat.db"))
    if "--verify" in sys.argv:
        results = verify_session(catalog, store, arg("--verify"))
        for r in results:
            print(f"turn {r['turn']}: {r['verdict'].upper():<10} {r['question'][:60]!r}  {r['detail']}")
        bad = [r for r in results if r["verdict"] == "changed"]
        print("\nEvery computed answer reproduced exactly." if not bad else f"\n{len(bad)} answer(s) no longer reproduce.")
        return 2 if bad else 0
    if "--trace" in sys.argv:
        print(json.dumps(store.trace(arg("--trace")), indent=2))
        return 0
    assistant, notice = build_assistant(catalog, store)
    print(notice)
    session = arg("--session") or store.new_session_id()
    if "--ask" in sys.argv:
        reply = assistant.ask(session, arg("--ask"))
        show(reply)
        print(f"\n(session {session}, turn {reply['turn']})")
        return 0
    print(f"Retail chat. Session {session}. Type a question, or :quit\n")
    last = None
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in (":quit", ":q"):
            break
        if q == ":new":
            session, last = store.new_session_id(), None
            print(f"New session {session}")
            continue
        if q in (":plan", ":evidence", ":trace") and last:
            print(json.dumps({":plan": last["plan"], ":evidence": last["evidence"], ":trace": last["trace"]}[q], indent=2))
            continue
        if not q:
            continue
        last = assistant.ask(session, q)
        show(last)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
