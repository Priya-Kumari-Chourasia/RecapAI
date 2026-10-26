"""Title + summary (map-reduce over long transcripts).

Recordings aren't always meetings (lectures, podcasts, videos, even music), so
the prompts describe whatever was recorded and never ask the user questions.
"""
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm import complete

MAP_CHUNK_CHARS = 6000

CONTEXT = (
    "The text is an automatic transcript of a recording. It is usually a meeting, but it can also be a "
    "lecture, talk, interview, video or anything else with audio. Work with whatever it is. "
    "Never ask the user for more information and never reply that the text isn't a meeting."
)


def split_transcript(transcript: str, chunk_size: int = MAP_CHUNK_CHARS) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    chunks = split_transcript(transcript)
    if len(chunks) == 1:
        partials = chunks
    else:
        partials = [
            complete(f"{CONTEXT}\nSummarize this portion concisely, keeping names, numbers and decisions.", c)
            for c in chunks
        ]
    return complete(
        f"{CONTEXT}\nWrite the final summary as 4-8 markdown bullet points. Lead with the most important "
        "point. Keep names, numbers, deadlines and technical terms exact. Do not invent anything. "
        "If it is not a meeting (e.g. a lecture or a song), summarize what it covers.",
        "\n\n".join(partials),
    )


def clean_title(title: str) -> str | None:
    """Reject chatty or broken titles (e.g. the model asking a question instead of answering)."""
    title = re.sub(r"\s+", " ", title or "").strip().strip('"\'*#').strip()
    if not title or title.endswith("?") or len(title.split()) > 12 or len(title) > 90:
        return None
    if re.search(r"\b(sorry|please provide|could you|i can't|i cannot)\b", title, re.I):
        return None
    return title


def generate_title(summary: str, fallback: str = "Untitled recording") -> str:
    title = complete(
        f"{CONTEXT}\nWrite a short, specific title (max 8 words) for this recording based on its summary. "
        "Return only the title, no quotes.",
        summary[:3000],
    )
    return clean_title(title) or fallback
