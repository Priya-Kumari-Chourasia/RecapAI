# Recap AI — memory for every meeting you record

Most "AI video summarizers" forget a video the moment you close the tab. Recap AI
keeps **every** meeting or lecture in a searchable workspace, answers questions
**across all of them** with dated, timestamped sources, and **tracks action items
from one meeting to the next**.

I built it for my M.Tech thesis meetings: guidance comes fast, and replaying an
hour of audio to find one comment isn't practical.

## Features

- **Three ways in:** record in the browser, upload audio/video, or paste a YouTube link.
- **Timestamped transcription**
  - **Whisper large-v3-turbo on Groq** (default): fast, no GPU, works when hosted.
  - **Sarvam AI** for Hindi / Hinglish speech, translated to English.
  - **Local Whisper** (optional, offline): `pip install -r requirements-local.txt`.
- **Structured insights:** summary, plus action items (task / owner / deadline),
  key decisions and open questions, extracted as Pydantic-validated JSON.
- **Meeting memory:** every meeting is stored in **Qdrant** (Qdrant Cloud when
  hosted, a local folder otherwise) with its date and per-chunk start/end times.
- **Ask across meetings:** RAG over the whole workspace, or only the meetings you
  pick. Answers cite their excerpts as `[1]`, `[2]`, and each source shows the
  meeting, date and time, e.g. `Thesis sync · 03 Oct 2026 · 12:34`. When meetings
  disagree, the newer one wins and the answer says what changed.
- **Follow-up tracker:** when a new meeting is added, every open action item from
  earlier meetings is checked against it. Per item, the most relevant moments of
  the new meeting are retrieved and an LLM decides *done / in progress / not
  mentioned*, quoting the evidence. Re-mentioned tasks aren't duplicated, and
  statuses can be edited by hand in the Follow-ups tab.
- **PDF chat:** add papers, slides or notes (several at once) to a project. Text is
  extracted page by page and stored next to the meeting transcripts, so one
  question is answered from both, with sources like `Bonsai paper · p. 4` next to
  `Thesis sync · 03 Oct 2026 · 12:34`. If a paper and a meeting disagree, the
  answer reports both. Scanned PDFs without a text layer get a clear message.
- **Fact-check (hallucination detection):** every summary point is checked against
  its own meeting or PDF. For each point the most relevant passages are retrieved
  and an LLM judge labels it ✅ supported / ⚠️ partly supported / ❌ not in the
  source, with the supporting quote and its timestamp or page. Summaries get a
  faithfulness score (e.g. "7/8 points verified · 94% faithful"), and any chat
  answer can be checked the same way with **Check this answer**.
- **Projects:** keep each project's meetings separate (e.g. *Thesis*, *BrickMind*).
  Meetings, chat and follow-ups are scoped to the selected project, so one
  project's tasks are never matched against another's meetings. Turn on
  *Search all projects* in Ask for questions like "what's due this week everywhere?".
- **Private workspace key:** all projects live under a workspace key that works
  like a password (it's hashed, and every query filters on it), so different
  people using the hosted app never see each other's projects.
- **YouTube on hosted servers:** YouTube blocks most cloud IPs. The app detects
  this and tells the user to upload or record instead of failing with a stack trace.

## Architecture

```
record / upload / YouTube
        │  pydub: 16 kHz mono, 10-min MP3 chunks (+ start offsets)
        ▼
transcriber.py ── Groq Whisper | Sarvam | local Whisper
        │  segments [{start, end, text}] on the full-recording timeline
        ▼
summarizer.py (map-reduce)    extractor.py (JSON → Pydantic)
        │
        ▼
vector_store.py ── Qdrant: recap_chunks (text + start/end) and recap_sources (meeting records)
        │                  embeddings: fastembed BAAI/bge-small-en-v1.5 (ONNX, no PyTorch)
        ├── factcheck.py   verify summary points / answers against the source
        ├── tracker.py     carry action items forward across meetings
        └── rag_engine.py  Q&A across meetings + PDFs with [n] citations
PDF ── documents.py ── pypdf page text → page-tagged chunks ──┘
```

```
app.py              Streamlit UI (Add · Library · Ask · Follow-ups)
main.py             CLI: add / ask / list
core/
  pipeline.py       end-to-end processing of one recording
  transcriber.py    timestamped speech-to-text (Groq / Sarvam / local Whisper)
  summarizer.py     title + map-reduce summary
  extractor.py      action items / decisions / questions as validated JSON
  documents.py      PDF upload: page extraction, page-tagged chunks, summary
  factcheck.py      claim-level fact-check of summaries and answers (LLM judge + retrieval)
eval/
  faithfulness.py   faithfulness score over all summaries in a workspace
  vector_store.py   Qdrant collections, chunking, workspace isolation
  rag_engine.py     cross-meeting retrieval + cited answers
  tracker.py        follow-up tracking across meetings
  llm.py            Groq LLM wrapper (LangChain), JSON mode with repair retry
utils/
  audio_processor.py  YouTube download (+ blocked-IP detection), conversion, chunking
  llm_retry.py        retries Groq 429s using the server's retry-after
tests/              offline tests (fake LLM + in-memory Qdrant), no API keys needed
```

## Setup

**Requirements:** Python 3.10+ and [FFmpeg](https://ffmpeg.org/) on `PATH`.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Create `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key          # required: LLM + Whisper transcription
SARVAM_API_KEY=your_sarvam_api_key      # optional: Hinglish transcription

# Optional: Qdrant Cloud. Without these, meetings are stored in ./qdrant_data
QDRANT_URL=https://xxxx.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# Optional
GROQ_MODEL=openai/gpt-oss-120b          # any Groq chat model
DEFAULT_WORKSPACE=demo
```

### Qdrant Cloud (needed for memory to survive restarts on Streamlit Cloud)

1. Create a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io).
2. Copy the cluster URL and create an API key.
3. Put them in `.env` locally, and in the Streamlit **Secrets** box when deployed.

Collections and indexes are created automatically on first run.

## Running

```bash
streamlit run app.py          # web app
python main.py add meeting.mp3 --project Thesis --title "Thesis sync" --date 2026-10-03
python main.py add-pdf bonsai.pdf --project Thesis
python main.py ask --project Thesis   # omit --project to search every project
python main.py list
python -m eval.faithfulness --project Thesis --csv claims.csv   # faithfulness report
pytest -q                     # offline tests
```

> Local Qdrant mode lets only one process open `./qdrant_data` at a time, so
> close the Streamlit app before using the CLI (or use Qdrant Cloud).

## Deploying to Streamlit Community Cloud

1. **New app** → pick this repo and set the main file to `app.py`.
2. **Advanced settings → Secrets** (top level, no `[section]` header):
   ```toml
   GROQ_API_KEY = "..."
   SARVAM_API_KEY = "..."
   QDRANT_URL = "https://xxxx.cloud.qdrant.io"
   QDRANT_API_KEY = "..."
   ```
3. Deploy. `packages.txt` installs ffmpeg.

### YouTube links on hosted servers

YouTube blocks most datacenter IPs ("Sign in to confirm you're not a bot"), so
links that work locally often fail on Streamlit Cloud. The app catches this and
asks the user to upload or record instead. Optional workarounds:

- `YTDLP_COOKIES`: the contents of a `cookies.txt` export, added as a secret.
  Use a throwaway Google account, because accounts used this way can be banned,
  and cookies expire.
- `YTDLP_PROXY`: route downloads through a (residential) proxy.

## Evaluation

`python -m eval.faithfulness` fact-checks every summary in a workspace (results
are cached on each meeting, `--recheck` re-runs them) and prints per-meeting and
overall faithfulness, defined as (supported + 0.5 × partial) / all summary points,
plus the unsupported points so you can see what the summarizer gets wrong.
`--csv` exports every claim with its verdict and evidence.

## Notes

- Groq's free tier limits tokens per minute; the app waits for the time Groq
  asks for (`retry-after`) and retries. For very long recordings, a smaller
  model such as `llama-3.1-8b-instant` (`GROQ_MODEL`) has higher limits.
- Sarvam's sync API accepts ≤30 s clips, so Hinglish audio is sent in 25 s
  pieces, which also gives 25 s timestamp resolution.
