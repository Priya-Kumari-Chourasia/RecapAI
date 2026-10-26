"""Follow-up tracker: carry action items across meetings.

When a new meeting is added, every still-open action item from earlier
meetings is checked against it. For each item we retrieve the most relevant
moments of the *new* meeting (RAG scoped to that one meeting), then one LLM
call decides per item: done / in_progress / not_mentioned, with a quote as
evidence. Status changes are written back to the meeting the item came from.
"""
import re
from typing import Literal

from pydantic import BaseModel, Field

from core import vector_store
from core.llm import complete_json
from core.transcriber import format_timestamp

MAX_ITEMS = 25


class ItemUpdate(BaseModel):
    item_id: str
    status: Literal["done", "in_progress", "not_mentioned"]
    evidence: str = Field(default="", description="Short quote from the new meeting supporting the status, or empty")
    excerpt: int | None = Field(default=None, description="Number of the excerpt the evidence came from")


class Duplicate(BaseModel):
    new_item: int = Field(description="Number of the NEW ITEM")
    item_id: str = Field(description="Id of the OPEN ITEM it repeats")


class TrackerResult(BaseModel):
    updates: list[ItemUpdate] = Field(default_factory=list)
    duplicates: list[Duplicate] = Field(
        default_factory=list, description="NEW ITEMS that are the same task as an OPEN ITEM (re-mentioned or re-assigned)"
    )


SYSTEM = (
    "You track action items across a series of meetings. For each OPEN ITEM below you get excerpts "
    "from the NEWEST meeting that might mention it. Decide its status in that meeting:\n"
    "- done: the meeting says it was completed\n"
    "- in_progress: it was discussed as started/ongoing, or re-assigned\n"
    "- not_mentioned: the excerpts don't really talk about this item\n"
    "Be strict: similar words are not enough, the excerpt must be about the same task. "
    "Return one update per OPEN ITEM.\n\n"
    "You also get the NEW ITEMS extracted from the newest meeting. If a new item is really the same task "
    "as an open item (mentioned again, re-assigned, or reported on), list it in `duplicates`. "
    "Different tasks that only sound similar (e.g. the same experiment on a different dataset) are NOT duplicates."
)


def open_items(workspace: str, exclude_source: str, up_to_date: str, project: str | None = None) -> list[tuple[dict, dict]]:
    """(source, item) pairs for unfinished items from meetings on or before `up_to_date`,
    only from the same project, so one project's tasks are never matched against another's meetings."""
    pairs = []
    for src in vector_store.list_sources(workspace, project=project):
        if src["source_id"] == exclude_source or src.get("source_type") != "meeting":
            continue
        if src.get("date", "") > up_to_date:
            continue
        for item in src.get("action_items", []):
            if item.get("status") != "done":
                pairs.append((src, item))
    return pairs[:MAX_ITEMS]  # list_sources is newest-first


def track_followups(workspace: str, new_meeting: dict) -> list[dict]:
    """Update earlier action items from `new_meeting`. Returns a list of changes for the UI."""
    pairs = open_items(workspace, new_meeting["source_id"], new_meeting["date"], new_meeting.get("project"))
    if not pairs:
        return []

    blocks, excerpt_hits = [], []
    for src, item in pairs:
        hits = vector_store.search(
            workspace, f"{item['task']} {item.get('owner', '')}", k=2, source_ids=[new_meeting["source_id"]]
        )
        lines = []
        for hit in hits:
            excerpt_hits.append(hit)
            lines.append(f"  ({len(excerpt_hits)}) [{format_timestamp(hit['start'])}] {hit['text']}")
        blocks.append(
            f"OPEN ITEM {item['id']}: {item['task']} (owner: {item.get('owner')}, "
            f"from '{src['title']}' on {src['date']})\nExcerpts:\n" + ("\n".join(lines) or "  (none)")
        )

    new_items = new_meeting.get("action_items", [])
    if new_items:
        blocks.append("NEW ITEMS from the newest meeting:\n" + "\n".join(
            f"  NEW ITEM {n}: {i['task']} (owner: {i.get('owner')})" for n, i in enumerate(new_items, 1)
        ))

    result = complete_json(SYSTEM, "\n\n".join(blocks), TrackerResult)
    by_id = {u.item_id: u for u in result.updates}

    changes, touched = [], {}
    for src, item in pairs:
        update = by_id.get(item["id"])
        if not update or update.status == "not_mentioned":
            continue
        hit = excerpt_hits[update.excerpt - 1] if update.excerpt and 0 < update.excerpt <= len(excerpt_hits) else None
        entry = {
            "source_id": new_meeting["source_id"],
            "title": new_meeting["title"],
            "date": new_meeting["date"],
            "status": update.status,
            "evidence": update.evidence,
            "start": hit["start"] if hit else None,
        }
        item["status"] = update.status
        item.setdefault("updates", []).append(entry)
        touched[src["source_id"]] = src
        changes.append({"task": item["task"], "owner": item.get("owner"), "from_title": src["title"], **entry})

    for source_id, src in touched.items():
        vector_store.update_source(workspace, source_id, {"action_items": src["action_items"]})

    # A new meeting often re-mentions old tasks, so the extractor lists them again.
    # Keep the original item as the single tracked copy: drop new items the LLM
    # matched to an open item, plus near-identical wording as a safety net.
    open_ids = {item["id"] for _, item in pairs}
    repeated = {d.new_item for d in result.duplicates if d.item_id in open_ids}
    earlier = [item["task"] for _, item in pairs]
    kept = [
        i for n, i in enumerate(new_items, 1)
        if n not in repeated and not any(_same_task(i["task"], t) for t in earlier)
    ]
    if len(kept) != len(new_items):
        new_meeting["action_items"] = kept
        vector_store.update_source(workspace, new_meeting["source_id"], {"action_items": kept})
    return changes


_STOP = {"the", "a", "an", "to", "of", "on", "for", "and", "in", "by", "with", "is", "it"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


def _same_task(a: str, b: str, threshold: float = 0.85) -> bool:
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= threshold


def set_item_status(workspace: str, source_id: str, item_id: str, status: str) -> None:
    """Manual override from the Follow-ups table."""
    src = vector_store.get_source(workspace, source_id)
    if not src:
        return
    for item in src.get("action_items", []):
        if item["id"] == item_id:
            item["status"] = status
    vector_store.update_source(workspace, source_id, {"action_items": src["action_items"]})
