# Recap AI

Turn any video into a conversation. Paste a YouTube link or upload a file and
get a clean transcript, an AI summary, extracted action items / decisions /
open questions, and a chat you can ask anything about the video — grounded in
its transcript via RAG.

## Features

- **Two input sources** — YouTube URL (downloaded via `yt-dlp`) or a local
  audio/video file upload (`mp4`, `mp3`, `wav`, `m4a`, `webm`, `mov`).
- **Two transcription engines**
  - **English** — [OpenAI Whisper](https://github.com/openai/whisper), running locally.
  - **Hinglish → English** — [Sarvam AI](https://www.sarvam.ai/)'s speech-to-text-translate API.
- **AI-generated insights** (via Mistral, through LangChain) — title, summary,
  action items (with owner/deadline), key decisions, and open questions.
- **Chat with the video** — a retrieval-augmented chat (Chroma vector store +
  HuggingFace sentence embeddings) that answers questions using only the
  transcript.
- **Streamlit UI** — process a video from the sidebar, browse results in
  tabs, download the transcript/summary as text files.

## Project structure

```
app.py                   Streamlit UI (main entry point)
main.py                  CLI entry point — same pipeline, no UI
core/
  transcriber.py         Whisper / Sarvam transcription
  summarizer.py          Title + summary generation
  extractor.py            Action items / key decisions / open questions
  rag_engine.py           RAG chat chain (build + query)
  vector_store.py         Chroma vector store + embeddings
utils/
  audio_processor.py      YouTube download, format conversion, chunking
  ffmpeg_path.py           Locates a local ffmpeg install for pydub/whisper
downloads/                YouTube audio downloads (created at runtime)
uploads/                  Uploaded files (created at runtime)
vector_db/                Persisted Chroma vector store (created at runtime)
```

## Setup

**Requirements:** Python 3.10+, [FFmpeg](https://ffmpeg.org/) installed and
on `PATH` (or discoverable via winget on Windows).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key       # required — summary, extraction & chat
SARVAM_API_KEY=your_sarvam_api_key         # required only for Hinglish transcription
SARVAM_STT_MODEL=saaras:v2.5               # optional, defaults to saaras:v2.5
```

## Running

**Streamlit app (recommended):**

```bash
streamlit run app.py
```

**CLI:**

```bash
python main.py
```

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (already done if you're reading this from there).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick
   this repo/branch and set the main file to `app.py`.
3. Under **Advanced settings → Secrets**, add your keys in TOML form:
   ```toml
   MISTRAL_API_KEY = "your_mistral_api_key"
   SARVAM_API_KEY = "your_sarvam_api_key"
   ```
4. Deploy. `packages.txt` (ffmpeg) and `requirements.txt` are picked up
   automatically.

> **Note:** local Whisper transcription is CPU/RAM-heavy (it pulls in
> PyTorch). On the free Community Cloud tier this can be slow or hit memory
> limits — set `WHISPER_MODEL = "tiny"` or `"base"` in the app's secrets if
> you run into that.

## Notes

- The local Whisper model size can be changed via the `WHISPER_MODEL` env var
  (defaults to `small`).
- Sarvam's sync API only accepts clips ≤30s, so Hinglish audio is
  automatically sliced into 25s pieces before being sent.
- The vector store is rebuilt per video (in-memory build via
  `build_rag_chain`), backed by a local Chroma collection.
