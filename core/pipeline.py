"""End-to-end: recording -> transcript -> insights -> stored in meeting memory -> follow-ups."""
import os
from datetime import date as date_cls
from datetime import datetime, timezone

from core import vector_store
from core.extractor import extract_insights
from core.summarizer import generate_title, summarize
from core.tracker import track_followups
from core.transcriber import segments_to_text, transcribe_all
from utils.audio_processor import is_youtube_url, process_input


def _cleanup(chunks):
    for path, _ in chunks:
        try:
            os.remove(path)
        except OSError:
            pass


def process_recording(
    source: str,
    workspace: str,
    project: str = vector_store.DEFAULT_PROJECT,
    engine: str = "groq",
    title: str | None = None,
    meeting_date: str | None = None,
    origin: str | None = None,
    vocabulary: str | None = None,
    progress=print,
) -> dict:
    """Process a YouTube URL or local audio/video file and save it to the workspace.

    Returns {"meeting": record, "followups": [changes to earlier action items]}.
    """
    progress("📥 Preparing audio…")
    chunks, duration = process_input(source)
    try:
        segments = transcribe_all(chunks, engine=engine, progress=progress, vocabulary=vocabulary)
    finally:
        _cleanup(chunks)
    if not segments:
        raise RuntimeError("No speech was detected in this recording.")

    record = build_meeting(
        segments,
        workspace=workspace,
        project=project,
        engine=engine,
        title=title,
        meeting_date=meeting_date,
        duration=duration,
        origin=origin or ("youtube" if is_youtube_url(source) else "upload"),
        origin_ref=source if is_youtube_url(source) else os.path.basename(source),
        progress=progress,
    )
    return save_meeting(record, progress=progress)


def build_meeting(segments, *, workspace, engine, project=vector_store.DEFAULT_PROJECT, title=None, meeting_date=None, duration=0.0,
                  origin="upload", origin_ref="", progress=print) -> dict:
    transcript = segments_to_text(segments)
    progress("📋 Writing summary…")
    summary = summarize(transcript)
    progress("✅ Extracting action items, decisions and open questions…")
    insights = extract_insights(transcript)
    if not title:
        progress("📌 Generating title…")
        stem = os.path.splitext(os.path.basename(origin_ref or ""))[0]
        fallback = stem if stem and origin in ("upload", "youtube") else f"Recording {meeting_date or date_cls.today().isoformat()}"
        title = generate_title(summary, fallback=fallback)

    return {
        "workspace": workspace,
        "project": project,
        "source_id": vector_store.new_source_id(),
        "source_type": "meeting",
        "title": title,
        "date": meeting_date or date_cls.today().isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origin": origin,
        "origin_ref": origin_ref,
        "engine": engine,
        "duration": round(duration or (segments[-1]["end"] if segments else 0), 1),
        "word_count": len(transcript.split()),
        "summary": summary,
        **insights,
        "segments": segments,
    }


def save_meeting(record: dict, progress=print) -> dict:
    progress("🧠 Saving to your meeting memory…")
    vector_store.save_source(record, vector_store.chunk_segments(record["segments"]))
    progress("🔁 Checking earlier action items against this meeting…")
    followups = track_followups(record["workspace"], record)
    return {"meeting": record, "followups": followups}
