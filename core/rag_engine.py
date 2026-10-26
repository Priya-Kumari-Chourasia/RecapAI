"""Ask questions across every meeting (and document) in a workspace.

Retrieval is filtered to the workspace (and optionally to chosen meetings),
and each retrieved chunk is labelled with its meeting, date and timestamp so
the model can cite it as [1], [2], ... and compare what was said over time.
"""
import re
from datetime import date, datetime

from core import vector_store
from core.llm import complete
from core.transcriber import format_timestamp

SYSTEM = """You are a research-meeting assistant. Answer the question using ONLY the numbered
excerpts below, which come from the user's recorded meetings and documents.

Rules:
- Put the excerpt number in square brackets after every claim, e.g. "Use a 70/30 split [2]."
- If excerpts from different dates disagree, say what changed and when; the newer one wins.
- If the excerpts don't contain the answer, reply exactly:
  "I couldn't find this in your meetings."
- Be concise. Today's date is {today}.

Excerpts:
{context}"""


def describe(hit: dict) -> str:
    """Human-readable location of a chunk, e.g. 'Thesis sync · 2026-10-03 · 12:34'."""
    where = f"p. {hit['page']}" if hit.get("page") else format_timestamp(hit.get("start", 0))
    try:
        when = datetime.fromisoformat(hit.get("date", "")).strftime("%d %b %Y")
    except ValueError:
        when = hit.get("date", "")
    return f"{hit.get('title', 'Untitled')} · {when} · {where}"


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, 1):
        kind = "Document" if hit.get("source_type") == "pdf" else "Meeting"
        project = f" (project: {hit['project']})" if hit.get("project") else ""
        blocks.append(f"[{i}] {kind}{project}: {describe(hit)}\n{hit['text']}")
    return "\n\n".join(blocks)


def cited_numbers(answer: str, n_hits: int) -> list[int]:
    nums = {int(n) for group in re.findall(r"\[([\d,\s]+)\]", answer) for n in re.findall(r"\d+", group)}
    return sorted(n for n in nums if 1 <= n <= n_hits)


def answer_question(
    workspace: str,
    question: str,
    source_ids: list[str] | None = None,
    history: list[dict] | None = None,
    k: int = 6,
    project: str | None = None,
) -> dict:
    """Returns {"answer": str, "sources": [hit, ...]} where sources are the excerpts the answer cited."""
    # Short follow-ups ("and who owns it?") retrieve better with the previous question attached.
    last_user = next((m["content"] for m in reversed(history or []) if m["role"] == "user"), "")
    retrieval_query = f"{last_user}\n{question}" if last_user and len(question.split()) < 6 else question

    hits = vector_store.search(workspace, retrieval_query, k=k, source_ids=source_ids, project=project)
    if not hits:
        return {"answer": "I couldn't find this in your meetings.", "sources": []}

    hits.sort(key=lambda h: (h.get("date", ""), h.get("start", 0) or h.get("page", 0)))
    recent = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in (history or [])[-4:])
    human = f"Conversation so far:\n{recent}\n\nQuestion: {question}" if recent else question

    answer = complete(SYSTEM.format(today=date.today().isoformat(), context=build_context(hits)), human)
    cited = cited_numbers(answer, len(hits))
    sources = [{**hits[n - 1], "ref": n} for n in cited]
    return {"answer": answer, "sources": sources}
