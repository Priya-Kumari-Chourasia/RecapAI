"""Fact-check: is every summary point (or chat answer) actually backed by the source?

For each claim we retrieve the most relevant passages of the *same* meeting or
document (RAG scoped to one source), then a single LLM call acts as a strict
judge and labels every claim:

    supported      the passages clearly state it
    partial        some of it is there, but a detail is missing or different
    unsupported    the passages don't say it (possible hallucination)

with a short quote as evidence and where it was said (timestamp or page).
Results are stored on the meeting, so the check runs once, and the evaluation
script (eval/faithfulness.py) aggregates them into one faithfulness score.
"""
import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from core import vector_store
from core.llm import complete_json
from core.transcriber import format_timestamp

EVIDENCE_PER_CLAIM = 3
BATCH = 10  # claims per LLM call; keeps prompts small on Groq's free tier

STATUS_ICON = {"supported": "✅", "partial": "⚠️", "unsupported": "❌"}


class Verdict(BaseModel):
    claim: int = Field(description="Number of the CLAIM")
    status: Literal["supported", "partial", "unsupported"]
    evidence: str = Field(default="", description="Short exact quote from the passages, or empty if unsupported")
    passage: int | None = Field(default=None, description="Number of the passage the quote comes from")
    note: str = Field(default="", description="For partial/unsupported: what is missing or different, in a few words")


class Verdicts(BaseModel):
    verdicts: list[Verdict] = Field(default_factory=list)


SYSTEM = (
    "You are a strict fact-checker. Each CLAIM was written by an AI about a recording or document. "
    "Judge each claim ONLY against the numbered PASSAGES listed under it, which are taken from the source.\n"
    "- supported: the passages clearly state it (paraphrase is fine)\n"
    "- partial: the main idea is there but a detail (number, name, date, owner) is missing or different\n"
    "- unsupported: the passages do not say it\n"
    "Do not use outside knowledge. Quote the evidence exactly and keep quotes short. "
    "Return one verdict per claim."
)


def split_claims(text: str) -> list[str]:
    """Summary bullets become claims; prose is split into sentences. Markdown and citations are stripped."""
    text = re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", text or "")
    bullets = [m.group(1) for m in re.finditer(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+(.+)$", text)]
    pieces = bullets or re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])", text)
    claims = []
    for piece in pieces:
        clean = re.sub(r"[*_`#]+", "", piece).strip(" -–—:;")
        if len(clean.split()) >= 3:
            claims.append(clean)
    return claims


def _where(hit: dict) -> dict:
    if hit.get("page"):
        return {"page": hit["page"], "start": None}
    return {"page": None, "start": hit.get("start")}


def _passage_line(n: int, hit: dict) -> str:
    where = f"p. {hit['page']}" if hit.get("page") else format_timestamp(hit.get("start", 0))
    return f"  PASSAGE {n} [{hit.get('title', '')} · {where}]: {hit['text']}"


def _judge(claims: list[str], evidence: list[list[dict]] | None = None, shared: list[dict] | None = None) -> list[dict]:
    """One LLM call for a batch of claims. Either each claim has its own passages (`evidence`),
    or all claims are judged against one shared list (`shared`, used for chat answers)."""
    blocks, flat = [], []
    if shared is not None:
        flat = list(shared)
        blocks.append("PASSAGES (for every claim):\n" + "\n".join(_passage_line(n, h) for n, h in enumerate(flat, 1)))
        blocks.append("\n".join(f"CLAIM {i}: {c}" for i, c in enumerate(claims, 1)))
        evidence = [flat for _ in claims]
    else:
        for i, (claim, hits) in enumerate(zip(claims, evidence), 1):
            lines = []
            for hit in hits:
                flat.append(hit)
                lines.append(_passage_line(len(flat), hit))
            blocks.append(f"CLAIM {i}: {claim}\n" + ("\n".join(lines) or "  (no passages found)"))

    result = complete_json(SYSTEM, "\n\n".join(blocks), Verdicts)
    by_claim = {v.claim: v for v in result.verdicts}

    out = []
    for i, claim in enumerate(claims, 1):
        v = by_claim.get(i)
        if v is None:  # the model skipped it; don't count it as verified
            out.append({"claim": claim, "status": "unsupported", "evidence": "", "note": "not checked", "page": None, "start": None})
            continue
        hit = flat[v.passage - 1] if v.passage and 0 < v.passage <= len(flat) else (evidence[i - 1][0] if evidence[i - 1] else None)
        supported_by = hit if hit and v.status != "unsupported" else None
        out.append({
            "claim": claim,
            "status": v.status,
            "evidence": v.evidence if supported_by else "",
            "note": v.note,
            "title": supported_by.get("title") if supported_by else None,
            **(_where(supported_by) if supported_by else {"page": None, "start": None}),
        })
    return out


def score(results: list[dict]) -> dict:
    counts = {s: sum(1 for r in results if r["status"] == s) for s in STATUS_ICON}
    total = len(results)
    return {
        "total": total,
        **counts,
        # partial counts as half: the gist is right but a detail isn't
        "faithfulness": round((counts["supported"] + 0.5 * counts["partial"]) / total, 3) if total else None,
    }


def check_claims(workspace: str, claims: list[str], source_ids: list[str] | None = None,
                 passages: list[dict] | None = None) -> dict:
    """Check claims against retrieved passages (from `source_ids`) or against given `passages`."""
    results = []
    for start in range(0, len(claims), BATCH):
        batch = claims[start : start + BATCH]
        if passages is not None:
            results += _judge(batch, shared=passages)
        else:
            evidence = [vector_store.search(workspace, c, k=EVIDENCE_PER_CLAIM, source_ids=source_ids) for c in batch]
            results += _judge(batch, evidence=evidence)
    return {"claims": results, "score": score(results),
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def check_source_summary(workspace: str, source: dict, save: bool = True) -> dict:
    """Fact-check a meeting's or document's summary against that same meeting/document."""
    report = check_claims(workspace, split_claims(source.get("summary", "")), source_ids=[source["source_id"]])
    if save:
        vector_store.update_source(workspace, source["source_id"], {"fact_check": report})
    return report


def check_answer(workspace: str, answer: str, passages: list[dict]) -> dict:
    """Fact-check a chat answer against the excerpts it was generated from."""
    if answer.startswith("I couldn't find this"):
        return {"claims": [], "score": score([]), "checked_at": None}
    # Every claim is judged against the same excerpts the answer saw (deduped, capped for prompt size).
    seen, unique = set(), []
    for p in passages:
        key = (p.get("source_id"), p.get("start"), p.get("page"), p["text"][:50])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return check_claims(workspace, split_claims(answer), passages=unique[:8])


def run_fact_check(record: dict, progress=print) -> None:
    """Verify a new meeting's or document's summary right after it is saved. A failure here
    (e.g. a rate limit) must not lose the item, which is already stored; the check can be
    re-run later from the Summary tab."""
    progress("🔍 Fact-checking the summary against the source…")
    try:
        record["fact_check"] = check_source_summary(record["workspace"], record)
    except Exception as err:  # noqa: BLE001
        progress(f"⚠️ Fact-check skipped ({err}); you can run it later from the Summary tab.")
