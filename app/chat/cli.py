"""
CHAT IN THE TERMINAL

    python -m app.chat.cli data/retail                    interactive chat
    python -m app.chat.cli data/retail --ask "Revenue by category"
    python -m app.chat.cli data/retail --trace SESSION_ID  show a saved conversation

Inside the chat:  :plan  :evidence  :trace  :new  :quit
"""
import json
import sys

from app.chat.factory import build_assistant
from app.chat.catalog import Catalog
from app.chat.store import ChatStore
from app.errors import DataLoadError


def show(reply):
    if reply["status"] == "ok":
        print(reply["answer"]["text"])
    elif reply["status"] == "clarify":
        print("? " + reply["clarification"]["text"])
    else:
        r = reply["refusal"]
        print(f"[{reply['status']}:{r['code']}] {r.get('text') or r['message']}")


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
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
