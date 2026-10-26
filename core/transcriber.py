"""Speech-to-text with timestamps.

Every engine returns the same thing: a list of segments
    {"start": seconds, "end": seconds, "text": "..."}
measured from the start of the full recording. Those timestamps flow all the
way through to chat citations like [12:34].

Engines:
- groq    — Whisper large-v3-turbo on Groq's API (default: fast, no GPU, works when hosted)
- sarvam  — Sarvam AI speech-to-text-translate (Hindi / Hinglish audio -> English text)
- local   — openai-whisper on your own machine (optional; needs requirements-local.txt)
"""
import os

import requests
from dotenv import load_dotenv
from pydub import AudioSegment

from utils.ffmpeg_path import ensure_ffmpeg_on_path

ensure_ffmpeg_on_path()
load_dotenv()

ENGINES = {
    "groq": "Whisper (Groq API)",
    "sarvam": "Hinglish → English (Sarvam AI)",
    "local": "Whisper (local, offline)",
}

GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")
# Sarvam's sync API rejects audio longer than 30s, so we send 25s pieces.
SARVAM_PIECE_SECONDS = 25


def local_whisper_available() -> bool:
    try:
        import whisper  # noqa: F401
    except ImportError:
        return False
    return True


def engine_ready(engine: str) -> tuple[bool, str]:
    """(ready, reason) — used by the UI to show which engines can run."""
    if engine == "groq":
        return bool(os.getenv("GROQ_API_KEY")), "needs GROQ_API_KEY"
    if engine == "sarvam":
        return bool(os.getenv("SARVAM_API_KEY")), "needs SARVAM_API_KEY"
    if engine == "local":
        return local_whisper_available(), "pip install -r requirements-local.txt"
    return False, "unknown engine"


def _shift(segments: list[dict], offset: float) -> list[dict]:
    return [
        {"start": round(s["start"] + offset, 2), "end": round(s["end"] + offset, 2), "text": s["text"].strip()}
        for s in segments
        if s.get("text", "").strip()
    ]


# --- Groq -----------------------------------------------------------------

_groq_client = None


def _groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"), max_retries=3)
    return _groq_client


def transcribe_chunk_groq(chunk_path: str, prompt: str | None = None) -> list[dict]:
    with open(chunk_path, "rb") as f:
        kwargs = dict(
            file=(os.path.basename(chunk_path), f.read()),
            model=GROQ_STT_MODEL,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            temperature=0.0,
        )
    if prompt:
        kwargs["prompt"] = prompt
    response = _groq().audio.transcriptions.create(**kwargs)
    data = response if isinstance(response, dict) else response.model_dump()
    segments = data.get("segments") or []
    if not segments and data.get("text"):
        segments = [{"start": 0.0, "end": float(data.get("duration") or 0), "text": data["text"]}]
    return [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"]} for s in segments]


# --- Sarvam ---------------------------------------------------------------


def _send_to_sarvam(piece_path: str) -> str:
    headers = {"api-subscription-key": os.getenv("SARVAM_API_KEY")}
    with open(piece_path, "rb") as f:
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files={"file": (os.path.basename(piece_path), f, "audio/wav")},
            data={"model": SARVAM_MODEL},
            timeout=120,
        )
    if not response.ok:
        raise RuntimeError(f"Sarvam returned {response.status_code}: {response.text[:300]}")
    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> list[dict]:
    """One segment per 25s piece — coarse, but enough to cite where something was said."""
    if not os.getenv("SARVAM_API_KEY"):
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")
    audio = AudioSegment.from_file(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    segments = []
    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start : start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")
        try:
            text = _send_to_sarvam(piece_path)
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)
        segments.append({"start": start / 1000, "end": (start + len(piece)) / 1000, "text": text})
    return segments


# --- Local Whisper ----------------------------------------------------------

_local_model = None


def transcribe_chunk_local(chunk_path: str, prompt: str | None = None) -> list[dict]:
    global _local_model
    import whisper

    if _local_model is None:
        print(f"Loading local Whisper model: {WHISPER_MODEL} ...")
        _local_model = whisper.load_model(WHISPER_MODEL)
    result = _local_model.transcribe(chunk_path, task="transcribe", initial_prompt=prompt)
    return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in result["segments"]]


# --- Public API -------------------------------------------------------------


def vocabulary_prompt(vocabulary: str | None) -> str | None:
    """Whisper spells names and jargon correctly when they appear in its prompt
    (e.g. 'pre-training' instead of 'free training'). Keep it short: Whisper reads ~224 tokens."""
    terms = ", ".join(t.strip() for t in (vocabulary or "").replace("\n", ",").split(",") if t.strip())
    return f"Glossary: {terms}."[:600] if terms else None


def transcribe_chunk(chunk_path: str, engine: str = "groq", prompt: str | None = None) -> list[dict]:
    if engine == "sarvam":
        return transcribe_chunk_sarvam(chunk_path)
    if engine == "local":
        return transcribe_chunk_local(chunk_path, prompt)
    return transcribe_chunk_groq(chunk_path, prompt)


def transcribe_all(
    chunks: list[tuple[str, float]], engine: str = "groq", progress=None, vocabulary: str | None = None
) -> list[dict]:
    """Transcribe (chunk_path, offset_seconds) pairs into one timestamped segment list."""
    prompt = vocabulary_prompt(vocabulary)
    segments: list[dict] = []
    for i, (chunk_path, offset) in enumerate(chunks):
        if progress:
            progress(f"Transcribing part {i + 1}/{len(chunks)} with {ENGINES[engine]}…")
        segments.extend(_shift(transcribe_chunk(chunk_path, engine, prompt), offset))
    return segments


def segments_to_text(segments: list[dict]) -> str:
    return " ".join(s["text"] for s in segments).strip()


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds or 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def segments_to_timestamped_text(segments: list[dict]) -> str:
    return "\n".join(f"[{format_timestamp(s['start'])}] {s['text']}" for s in segments)
