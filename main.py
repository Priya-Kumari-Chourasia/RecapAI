"""Command-line version of Recap AI.

    python main.py add <file-or-youtube-url> [--project P] [--title T] [--date YYYY-MM-DD] [--engine groq|sarvam|local]
    python main.py add-pdf <file.pdf> [--project P] [--title T] [--date YYYY-MM-DD]
    python main.py ask [--project P]          (no --project = search all projects)
    python main.py list [--project P]

Uses the same workspace and storage as the Streamlit app (set RECAP_WORKSPACE, default "demo").
"""
import argparse
import os

from dotenv import load_dotenv

load_dotenv()

from core import rag_engine, vector_store  # noqa: E402
from core.documents import process_pdf  # noqa: E402
from core.pipeline import process_recording  # noqa: E402
from core.transcriber import format_timestamp  # noqa: E402

WS = vector_store.workspace_id(os.getenv("RECAP_WORKSPACE", os.getenv("DEFAULT_WORKSPACE", "demo")))


def cmd_add(args):
    out = process_recording(
        args.source, workspace=WS, project=args.project or vector_store.DEFAULT_PROJECT,
        engine=args.engine, title=args.title, meeting_date=args.date,
    )
    m = out["meeting"]
    print("\n" + "=" * 60)
    print(f"📌 {m['title']} ({m['date']})\n\n{m['summary']}\n")
    print("✅ Action items:")
    for a in m["action_items"]:
        print(f"  - {a['task']} — {a['owner']} (deadline: {a['deadline']})")
    if out["followups"]:
        print("\n🔁 Updated from earlier meetings:")
        for c in out["followups"]:
            print(f"  - [{c['status']}] {c['task']}: “{c['evidence']}”")
    print("=" * 60)


def cmd_add_pdf(args):
    d = process_pdf(args.source, WS, project=args.project or vector_store.DEFAULT_PROJECT,
                    title=args.title, doc_date=args.date)
    print(f"\n📄 {d['title']} ({d['page_count']} pages)\n\n{d['summary']}")


def cmd_list(args):
    for m in vector_store.list_sources(WS, project=args.project):
        open_items = sum(1 for a in m.get("action_items", []) if a["status"] != "done")
        kind = "PDF    " if m.get("source_type") == "pdf" else "meeting"
        print(f"{m['date']}  {kind}  [{m.get('project')}]  {m['title']}  ({open_items} open action items)")


def cmd_ask(args):
    print("💬 Ask across your meetings and documents (type 'exit' to quit)\n")
    history = []
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        result = rag_engine.answer_question(WS, question, history=history, project=args.project)
        print(f"\n🤖 {result['answer']}")
        for s in result["sources"]:
            print(f"   [{s['ref']}] {rag_engine.describe(s)}")
        print()
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": result["answer"]}]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recap AI — meeting memory")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="Transcribe a recording and save it to memory")
    add.add_argument("source")
    add.add_argument("--project", help="Project name (default: General)")
    add.add_argument("--title")
    add.add_argument("--date")
    add.add_argument("--engine", default="groq", choices=["groq", "sarvam", "local"])
    add.set_defaults(func=cmd_add)
    pdf = sub.add_parser("add-pdf", help="Add a PDF (paper, slides, notes) to a project")
    pdf.add_argument("source")
    pdf.add_argument("--project", help="Project name (default: General)")
    pdf.add_argument("--title")
    pdf.add_argument("--date")
    pdf.set_defaults(func=cmd_add_pdf)
    for name, func, help_text in [("ask", cmd_ask, "Chat across meetings"), ("list", cmd_list, "List saved meetings")]:
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--project", help="Limit to one project (default: all projects)")
        cmd.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)
