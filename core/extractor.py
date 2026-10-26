"""Action items, key decisions and open questions as validated, structured data.

One JSON call per transcript chunk (instead of three free-text calls), validated
with Pydantic, so the follow-up tracker can work with real fields
(task / owner / deadline) rather than parsing numbered lists.
"""
import re
import uuid

from pydantic import BaseModel, Field

from core.llm import complete_json
from core.summarizer import split_transcript

EXTRACT_CHUNK_CHARS = 12000


class ActionItem(BaseModel):
    task: str = Field(description="What needs to be done, as a short imperative sentence")
    owner: str = Field(default="Not specified", description="Who is responsible, or 'Not specified'")
    deadline: str = Field(default="Not specified", description="Deadline as said in the meeting, or 'Not specified'")


class MeetingInsights(BaseModel):
    action_items: list[ActionItem] = Field(default_factory=list)
    key_decisions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


SYSTEM = (
    "You are an expert meeting analyst. The transcript is usually a meeting but may be a lecture, "
    "video or other recording. From it, extract:\n"
    "- action_items: concrete tasks someone committed to or was asked to do\n"
    "- key_decisions: things that were agreed or decided\n"
    "- open_questions: unresolved questions or topics that need follow-up\n"
    "Only include items actually present in the transcript. Use empty lists when there are none "
    "(a lecture or a song usually has no action items)."
)


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _dedupe(items: list, key) -> list:
    seen, out = set(), []
    for item in items:
        k = _norm(key(item))
        if k and k not in seen:
            seen.add(k)
            out.append(item)
    return out


def extract_insights(transcript: str) -> dict:
    merged = MeetingInsights()
    for chunk in split_transcript(transcript, EXTRACT_CHUNK_CHARS):
        part = complete_json(SYSTEM, chunk, MeetingInsights)
        merged.action_items += part.action_items
        merged.key_decisions += part.key_decisions
        merged.open_questions += part.open_questions

    action_items = [
        {**a.model_dump(), "id": uuid.uuid4().hex[:12], "status": "open", "updates": []}
        for a in _dedupe(merged.action_items, lambda a: a.task)
    ]
    return {
        "action_items": action_items,
        "key_decisions": _dedupe(merged.key_decisions, str),
        "open_questions": _dedupe(merged.open_questions, str),
    }
