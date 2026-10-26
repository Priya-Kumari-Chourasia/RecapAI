import html
import os
import re
from datetime import date, datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# On Streamlit Community Cloud, secrets come from st.secrets, not .env. Mirror
# them into os.environ before importing core modules (some read env at import).
# Flattens one level of [section] nesting, a common way keys silently go missing.
_secrets_seen = []
_secrets_error = None
try:
    def _flatten_secrets(mapping, out):
        for k, v in mapping.items():
            if hasattr(v, "items"):
                _flatten_secrets(v, out)
            else:
                out[k] = str(v)

    _flat_secrets = {}
    _flatten_secrets(st.secrets, _flat_secrets)
    for _key, _value in _flat_secrets.items():
        os.environ.setdefault(_key, _value)
        _secrets_seen.append(_key)
except Exception as _e:  # noqa: BLE001 - no secrets file locally is normal
    _secrets_error = str(_e)

from core import rag_engine, tracker, vector_store  # noqa: E402
from core.documents import NoTextInPdfError, process_pdf  # noqa: E402
from core.pipeline import process_recording  # noqa: E402
from core.transcriber import ENGINES, engine_ready, format_timestamp, segments_to_timestamped_text  # noqa: E402
from utils.audio_processor import YouTubeBlockedError  # noqa: E402

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Recap AI", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer { visibility: hidden; height: 0; }
    header[data-testid="stHeader"] { background: transparent; }

    :root {
        --accent: #6c5ce7; --accent-dark: #5b4bd6; --accent-soft: #f1effe;
        --ink: #1a1a2e; --muted: #6b7280; --border: #edeef2;
        --success: #1e8e5a; --success-soft: #eafaf1;
        --warn: #b7791f; --warn-soft: #fdf6e3;
        --danger: #c0392b; --danger-soft: #fdecea;
    }

    section[data-testid="stSidebar"] { background: #fafafa; border-right: 1px solid #ececec; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.75rem; }

    .brand-row { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 1.3rem; }
    .brand-badge {
        flex-shrink: 0; width: 42px; height: 42px; border-radius: 12px;
        background: linear-gradient(135deg, var(--accent), #a29bfe);
        display: flex; align-items: center; justify-content: center; font-size: 1.25rem;
        box-shadow: 0 4px 12px rgba(108, 92, 231, 0.35);
    }
    .brand-title { font-size: 1.12rem; font-weight: 800; color: var(--ink); line-height: 1.25; }
    .brand-sub { font-size: 0.78rem; color: var(--muted); }
    .sidebar-section-label {
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase;
        color: var(--muted); margin: 1.15rem 0 0.5rem 0;
    }

    .chip-row { margin: 0.4rem 0 0.2rem 0; }
    .chip {
        display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.28rem 0.65rem;
        border-radius: 999px; font-size: 0.74rem; font-weight: 600; margin: 0 0.4rem 0.4rem 0;
    }
    .chip-ok { background: var(--success-soft); color: var(--success); }
    .chip-warn { background: var(--danger-soft); color: var(--danger); }
    .chip-info { background: var(--accent-soft); color: var(--accent-dark); }

    .meta-chip {
        display: inline-flex; align-items: center; padding: 0.32rem 0.8rem; border-radius: 999px;
        background: var(--accent-soft); color: var(--accent-dark); font-size: 0.8rem; font-weight: 600;
        margin: 0 0.4rem 0.5rem 0;
    }
    .stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.4rem; }
    .stat {
        background: #fff; border: 1px solid var(--border); border-radius: 12px; padding: 0.55rem 0.6rem;
    }
    .stat b { display: block; font-size: 1.25rem; color: var(--ink); }
    .stat span { font-size: 0.72rem; color: var(--muted); }

    .status { padding: 0.15rem 0.55rem; border-radius: 999px; font-size: 0.72rem; font-weight: 700; }
    .status-open { background: var(--danger-soft); color: var(--danger); }
    .status-in_progress { background: var(--warn-soft); color: var(--warn); }
    .status-done { background: var(--success-soft); color: var(--success); }
    .update-line { font-size: 0.82rem; color: var(--muted); margin: 0.15rem 0 0 1.2rem; }

    .stButton > button { border-radius: 10px; font-weight: 600; }
    .stButton > button[kind="primary"] {
        background: var(--accent); border: none; box-shadow: 0 4px 14px rgba(108, 92, 231, 0.35);
    }
    .stButton > button[kind="primary"]:disabled { background: #d8d5f5; box-shadow: none; }

    .empty-hero { text-align: center; padding: 2.5rem 2rem 1.5rem 2rem; max-width: 680px; margin: 0 auto; }
    .empty-hero h1 { font-size: 2rem; font-weight: 800; color: var(--ink); margin-bottom: 0.6rem; }
    .empty-hero p { font-size: 1rem; color: var(--muted); line-height: 1.6; }
    .empty-hero .emoji { font-size: 2.6rem; margin-bottom: 0.8rem; }
    .feature-card {
        background: #fff; border: 1px solid var(--border); border-radius: 14px;
        padding: 1.3rem 1.1rem; text-align: center; height: 100%;
    }
    .fc-icon { font-size: 1.6rem; margin-bottom: 0.5rem; }
    .fc-title { font-weight: 700; color: var(--ink); margin-bottom: 0.3rem; font-size: 0.96rem; }
    .fc-desc { font-size: 0.84rem; color: var(--muted); line-height: 1.55; }

    .video-title { font-size: 1.7rem; font-weight: 800; color: var(--ink); line-height: 1.3; margin-bottom: 0.4rem; }
    div[data-testid="stTabs"] button[role="tab"] { font-weight: 600; font-size: 0.92rem; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent) !important; }
    div[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
    .project-pill {
        display: inline-block; padding: 0.25rem 0.8rem; border-radius: 999px; margin-bottom: 0.8rem;
        background: var(--accent-soft); color: var(--accent-dark); font-size: 0.8rem; font-weight: 700;
    }
    .source-line { font-size: 0.84rem; color: var(--muted); margin-bottom: 0.35rem; }
    .source-line b { color: var(--ink); }
    </style>
    """,
    unsafe_allow_html=True,
)

STATUS_LABEL = {"open": "Open", "in_progress": "In progress", "done": "Done"}


def esc(text) -> str:
    return html.escape(str(text or ""))


def chip(label: str, ok: bool | None) -> str:
    cls = "chip-info" if ok is None else ("chip-ok" if ok else "chip-warn")
    icon = "" if ok is None else ("✓ " if ok else "✕ ")
    return f'<span class="chip {cls}">{icon}{esc(label)}</span>'


def status_badge(status: str) -> str:
    return f'<span class="status status-{status}">{STATUS_LABEL.get(status, status)}</span>'


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text or "").strip().lower()
    return re.sub(r"[\s_-]+", "-", text)[:60] or "meeting"


def nice_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return iso or ""


def meeting_report(m: dict) -> str:
    lines = [f"# {m['title']}", f"_{nice_date(m['date'])} · {format_timestamp(m.get('duration', 0))}_", "",
             "## Summary", m.get("summary", ""), "", "## Action items"]
    for a in m.get("action_items", []):
        lines.append(f"- [{'x' if a['status'] == 'done' else ' '}] {a['task']} — {a['owner']} (deadline: {a['deadline']})")
    lines += ["", "## Key decisions", *[f"- {d}" for d in m.get("key_decisions", [])],
              "", "## Open questions", *[f"- {q}" for q in m.get("open_questions", [])],
              "", "## Transcript", segments_to_timestamped_text(m.get("segments", []))]
    return "\n".join(lines)


@st.cache_resource(show_spinner="Loading the embedding model (first run only)…")
def warm_up_memory():
    vector_store.get_embedder()
    vector_store.get_client()
    return True


# ---------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-badge">🎬</div>
            <div>
                <div class="brand-title">Recap AI</div>
                <div class="brand-sub">Memory for every meeting you record</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Containers keep the visual order (project first, key last) while the code
    # reads the workspace key first, since the project list depends on it.
    project_box = st.container()
    stats_box = st.container()
    status_box = st.container()
    key_box = st.expander("🔐 Workspace key")

with key_box:
    workspace_name = st.text_input(
        "Workspace key",
        value=st.session_state.get("workspace_name", os.getenv("DEFAULT_WORKSPACE", "demo")),
        type="password",
        help="All your projects are stored under this key.",
    ).strip()
    st.caption(
        "Like a password for your meetings: anyone who enters the same key sees the same projects. "
        "Running only on your laptop? You can leave it as it is. On the hosted app, pick a private key."
    )
st.session_state.workspace_name = workspace_name

groq_ok = bool(os.getenv("GROQ_API_KEY"))
sarvam_ok = bool(os.getenv("SARVAM_API_KEY"))
with status_box:
    st.markdown('<div class="sidebar-section-label">Services</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="chip-row">'
        + chip("Groq (LLM + Whisper)", groq_ok)
        + chip("Sarvam (Hinglish)", sarvam_ok)
        + chip(vector_store.storage_label(), None)
        + "</div>",
        unsafe_allow_html=True,
    )
    if not groq_ok:
        st.caption("⚠️ GROQ_API_KEY is required.")
        with st.expander("Why am I seeing this?"):
            if _secrets_error:
                st.caption(f"Couldn't read st.secrets: {_secrets_error}")
            elif _secrets_seen:
                st.caption("Keys detected: " + ", ".join(sorted(_secrets_seen)) + ". Check for a typo.")
            else:
                st.caption(
                    "No keys detected. Locally: check your .env file. On Streamlit Cloud: "
                    "Manage app → Settings → Secrets, paste keys at the top level, save, reboot."
                )

if not workspace_name:
    with project_box:
        st.info("Enter a workspace key (🔐 below) to start.")
    st.stop()

warm_up_memory()
WS = vector_store.workspace_id(workspace_name)
projects = vector_store.list_projects(WS)

with project_box:
    st.markdown('<div class="sidebar-section-label">Project</div>', unsafe_allow_html=True)
    wanted = st.session_state.pop("switch_to_project", None) or st.session_state.get("project")
    PROJECT = st.selectbox(
        "Project",
        projects,
        index=projects.index(wanted) if wanted in projects else 0,
        label_visibility="collapsed",
        help="Each project has its own meetings, chat and follow-ups.",
    )
    if PROJECT != st.session_state.get("project"):
        st.session_state.pop("last_result", None)
        st.session_state.pop("selected_meeting", None)
    st.session_state.project = PROJECT

    with st.popover("➕ New project", width="stretch"):
        new_name = st.text_input("Project name", placeholder="e.g. Thesis, BrickMind", key="new_project_name")
        if st.button("Create project", type="primary", disabled=not new_name.strip()):
            if new_name.strip().lower() in {p.lower() for p in projects}:
                st.warning("A project with that name already exists.")
            else:
                st.session_state.switch_to_project = vector_store.create_project(WS, new_name)
                st.rerun()

sources = vector_store.list_sources(WS, project=PROJECT)
meetings = [s for s in sources if s.get("source_type") == "meeting"]
docs = [s for s in sources if s.get("source_type") == "pdf"]
ICON = {"meeting": "🎙️", "pdf": "📄"}


def source_label(i: str) -> str:
    src = next(x for x in sources if x["source_id"] == i)
    return f"{ICON.get(src.get('source_type'), '')} {src['title']} — {nice_date(src['date'])}"


all_items = [(m, a) for m in meetings for a in m.get("action_items", [])]
open_count = sum(1 for _, a in all_items if a["status"] != "done")

with stats_box:
    st.markdown(
        f"""<div class="stat-grid">
            <div class="stat"><b>{len(meetings)}</b><span>meetings</span></div>
            <div class="stat"><b>{len(docs)}</b><span>PDFs</span></div>
            <div class="stat"><b>{open_count}</b><span>open tasks</span></div>
        </div>""",
        unsafe_allow_html=True,
    )
    if sources:
        st.markdown('<div class="sidebar-section-label">Recent</div>', unsafe_allow_html=True)
        for m in sources[:5]:
            st.caption(f"{ICON.get(m.get('source_type'), '')} **{m['title']}** · {nice_date(m['date'])}")
    if PROJECT != vector_store.DEFAULT_PROJECT:
        with st.popover(f"🗑️ Delete “{PROJECT}”", width="stretch"):
            st.write(f"This permanently deletes **{PROJECT}**, its {len(meetings)} meeting(s) and {len(docs)} document(s).")
            if st.button("Yes, delete project", type="primary"):
                vector_store.delete_project(WS, PROJECT)
                st.session_state.switch_to_project = vector_store.DEFAULT_PROJECT
                st.rerun()

# ---------------------------------------------------------------------------- tabs

tab_add, tab_meetings, tab_ask, tab_follow = st.tabs(
    ["➕ Add", f"📚 Library ({len(sources)})", "💬 Ask", f"✅ Follow-ups ({open_count})"]
)

# ---- Add meeting ------------------------------------------------------------

with tab_add:
    if not sources:
        st.markdown(
            f"""
            <div class="empty-hero">
                <div class="project-pill">Project: {esc(PROJECT)}</div>
                <div class="emoji">🧠</div>
                <h1>Never lose what was said in a meeting</h1>
                <p>Record or upload your meetings and lectures, and add papers or slides as PDFs.
                Recap AI transcribes them, pulls out action items and decisions, remembers <b>everything</b>
                in this project, and tracks whether each task got done in later meetings.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        for col, (icon, title, desc) in zip(cols, [
            ("🎙️", "Meetings & PDFs", "Record in the browser, upload audio/video, paste a YouTube link, or add PDFs."),
            ("💬", "Ask across everything", "“What did we decide about the baseline?” — answered from meetings and PDFs, with sources."),
            ("🔁", "Follow-ups tracked", "Each new meeting checks earlier action items and marks them done or in progress."),
        ]):
            col.markdown(
                f'<div class="feature-card"><div class="fc-icon">{icon}</div>'
                f'<div class="fc-title">{title}</div><div class="fc-desc">{desc}</div></div>',
                unsafe_allow_html=True,
            )
        st.write("")

    with st.container(border=True):
        source_type = st.segmented_control(
            "Source", ["Upload file", "Record", "YouTube URL", "PDF document"], default="Upload file", required=True
        )
        is_pdf = source_type == "PDF document"
        source, origin, pdf_files = None, None, []
        if is_pdf:
            pdf_files = st.file_uploader(
                "PDF files (papers, slides, notes)", type=["pdf"], accept_multiple_files=True
            ) or []
        elif source_type == "Upload file":
            uploaded = st.file_uploader(
                "Audio or video file", type=["mp4", "mp3", "wav", "m4a", "webm", "mov", "ogg", "flac"]
            )
            if uploaded is not None:
                source = os.path.join(UPLOAD_DIR, os.path.basename(uploaded.name))
                with open(source, "wb") as f:
                    f.write(uploaded.getbuffer())
                origin = "upload"
        elif source_type == "Record":
            recording = st.audio_input("Record a meeting (allow microphone access when your browser asks)")
            if recording is not None:
                source = os.path.join(UPLOAD_DIR, f"recording-{datetime.now():%Y%m%d-%H%M%S}.wav")
                with open(source, "wb") as f:
                    f.write(recording.getbuffer())
                origin = "recording"
        else:
            source = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...").strip() or None
            origin = "youtube"
            st.caption(
                "ℹ️ YouTube links work when you run Recap AI on your own computer. Hosted servers are "
                "often blocked by YouTube; if that happens, upload the file instead."
            )

        if is_pdf:
            c1, c2 = st.columns([3, 2])
            title = c1.text_input(
                "Title (optional)", placeholder="From the PDF or its file name",
                disabled=len(pdf_files) > 1, help="Only for a single PDF; with several, each keeps its own title.",
            )
            meeting_date = c2.date_input("Date", value=date.today(), max_value=date.today())
            engine, engine_ok, vocabulary = None, True, None
            process_clicked = st.button(
                f"📄 Add {len(pdf_files) or ''} PDF{'s' if len(pdf_files) > 1 else ''} to “{PROJECT}”",
                type="primary", width="stretch", disabled=not (pdf_files and groq_ok),
            )
        else:
            c1, c2, c3 = st.columns([3, 2, 3])
            title = c1.text_input("Title (optional)", placeholder="Generated automatically if left empty")
            meeting_date = c2.date_input("Meeting date", value=date.today(), max_value=date.today())
            engine_options = [e for e in ENGINES if engine_ready(e)[0] or e != "local"]
            engine = c3.selectbox(
                "Transcription",
                engine_options,
                format_func=lambda e: ENGINES[e] + ("" if engine_ready(e)[0] else f" — {engine_ready(e)[1]}"),
            )
            engine_ok = engine_ready(engine)[0]
            vocabulary = st.text_input(
                "Names & terms (optional)",
                placeholder="e.g. Bonsai, XMC, pre-training, Dr. Verma",
                help="Comma-separated names and technical words. Whisper uses them to spell jargon correctly "
                "(e.g. 'pre-training' instead of 'free training'). Not used by Sarvam.",
            )
            process_clicked = st.button(
                f"🚀 Process & save to “{PROJECT}”",
                type="primary",
                width="stretch",
                disabled=not (source and engine_ok and groq_ok),
            )

    if process_clicked and is_pdf and pdf_files:
        added, failed = [], []
        with st.status(f"Adding {len(pdf_files)} PDF(s)…", expanded=True) as status:
            for f in pdf_files:
                path = os.path.join(UPLOAD_DIR, os.path.basename(f.name))
                with open(path, "wb") as out_file:
                    out_file.write(f.getbuffer())
                try:
                    added.append(process_pdf(
                        path, WS, project=PROJECT, title=(title or None) if len(pdf_files) == 1 else None,
                        doc_date=meeting_date.isoformat(), progress=lambda msg, n=f.name: status.write(f"{n}: {msg}"),
                    ))
                except NoTextInPdfError as err:
                    failed.append((f.name, str(err)))
                except Exception as err:  # noqa: BLE001
                    failed.append((f.name, f"Something went wrong: {err}"))
            status.update(
                label=f"Added {len(added)} PDF(s)" + (f", {len(failed)} failed" if failed else ""),
                state="error" if failed and not added else "complete", expanded=bool(failed),
            )
        for name, msg in failed:
            st.warning(f"**{name}** — {msg}", icon="📄")
        if added:
            st.session_state.last_docs = added
            st.session_state.pop("last_result", None)
            st.session_state.selected_meeting = added[-1]["source_id"]
            if not failed:
                st.rerun()

    if process_clicked and not is_pdf and source:
        with st.status("Processing…", expanded=True) as status:
            try:
                out = process_recording(
                    source,
                    workspace=WS,
                    project=PROJECT,
                    engine=engine,
                    title=title or None,
                    meeting_date=meeting_date.isoformat(),
                    origin=origin,
                    vocabulary=vocabulary or None,
                    progress=status.write,
                )
                st.session_state.last_result = out
                st.session_state.pop("last_docs", None)
                st.session_state.selected_meeting = out["meeting"]["source_id"]
                status.update(label="Saved to your meeting memory", state="complete", expanded=False)
            except YouTubeBlockedError as err:
                status.update(label="YouTube blocked the download", state="error", expanded=False)
                st.warning(str(err), icon="🚫")
            except Exception as err:  # noqa: BLE001 - show any failure in the UI
                status.update(label="Processing failed", state="error")
                st.error(f"Something went wrong: {err}")
            else:
                st.rerun()

    for d in st.session_state.get("last_docs", []):
        with st.container(border=True):
            st.markdown(f"#### 📄 Added: {d['title']}")
            st.caption(f"{d['page_count']} pages · {d['word_count']} words — ask about it in the **Ask** tab.")
            st.markdown(d["summary"])

    last = st.session_state.get("last_result")
    if last:
        m = last["meeting"]
        with st.container(border=True):
            st.markdown(f"#### ✅ Saved: {m['title']}")
            st.markdown(m["summary"])
            st.caption(
                f"{len(m['action_items'])} new action items · {len(m['key_decisions'])} decisions · "
                f"{len(m['open_questions'])} open questions — open the **Library** tab for details."
            )
            if last["followups"]:
                st.markdown(f"**🔁 Updated {len(last['followups'])} action item(s) from earlier meetings**")
                for c in last["followups"]:
                    at = f" at {format_timestamp(c['start'])}" if c.get("start") is not None else ""
                    st.markdown(
                        f"{status_badge(c['status'])} **{esc(c['task'])}** <span class='update-line'>"
                        f"(from “{esc(c['from_title'])}”) — said{at}: “{esc(c['evidence'])}”</span>",
                        unsafe_allow_html=True,
                    )

# ---- Meetings -----------------------------------------------------------------

with tab_meetings:
    chosen, m = None, None
    if not sources:
        st.info(f"Nothing in **{PROJECT}** yet. Add a meeting or a PDF in the **Add** tab.")
    else:
        ids = [x["source_id"] for x in sources]
        current = st.session_state.get("selected_meeting")
        chosen = st.selectbox(
            "Item",
            ids,
            index=ids.index(current) if current in ids else 0,
            format_func=source_label,
            label_visibility="collapsed",
        )
        st.session_state.selected_meeting = chosen
        m = vector_store.get_source(WS, chosen)

    if m and m.get("source_type") == "pdf":
        st.markdown(f'<div class="video-title">📄 {esc(m["title"])}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="meta-chip">📅 {nice_date(m["date"])}</span>'
            f'<span class="meta-chip">📑 {m.get("page_count", 0)} pages</span>'
            f'<span class="meta-chip">📝 {m.get("word_count", 0)} words</span>'
            f'<span class="meta-chip">🗂️ {esc(m.get("origin_ref", ""))}</span>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown(m["summary"])
        c1, c2, _ = st.columns([2, 2, 4])
        c1.download_button(
            "⬇️ Download summary (.md)", f"# {m['title']}\n\n{m['summary']}\n",
            file_name=f"{slugify(m['title'])}-summary.md", mime="text/markdown", width="stretch",
        )
        with c2.popover("🗑️ Delete document", width="stretch"):
            st.write("This removes the document from this project's memory.")
            if st.button("Yes, delete it", type="primary", key="del_doc"):
                vector_store.delete_source(WS, chosen)
                st.session_state.pop("selected_meeting", None)
                st.session_state.pop("last_docs", None)
                st.rerun()
    elif m:

        st.markdown(f'<div class="video-title">📌 {esc(m["title"])}</div>', unsafe_allow_html=True)
        done = sum(1 for a in m["action_items"] if a["status"] == "done")
        st.markdown(
            f'<span class="meta-chip">📅 {nice_date(m["date"])}</span>'
            f'<span class="meta-chip">⏱️ {format_timestamp(m.get("duration", 0))}</span>'
            f'<span class="meta-chip">📝 {m.get("word_count", 0)} words</span>'
            f'<span class="meta-chip">✅ {done}/{len(m["action_items"])} action items done</span>',
            unsafe_allow_html=True,
        )

        t_sum, t_act, t_dec, t_q, t_tr = st.tabs(
            ["📋 Summary", "✅ Action items", "🔑 Decisions", "❓ Open questions", "📝 Transcript"]
        )
        with t_sum:
            with st.container(border=True):
                st.markdown(m["summary"])
        with t_act:
            if not m["action_items"]:
                st.caption("No action items in this meeting.")
            for a in m["action_items"]:
                st.markdown(
                    f"{status_badge(a['status'])} **{esc(a['task'])}** — {esc(a['owner'])}"
                    f" <span style='color:var(--muted)'>· deadline: {esc(a['deadline'])}</span>",
                    unsafe_allow_html=True,
                )
                for u in a.get("updates", []):
                    at = f" at {format_timestamp(u['start'])}" if u.get("start") is not None else ""
                    st.markdown(
                        f"<div class='update-line'>↳ {STATUS_LABEL[u['status']]} in “{esc(u['title'])}” "
                        f"({nice_date(u['date'])}{at}): “{esc(u['evidence'])}”</div>",
                        unsafe_allow_html=True,
                    )
        with t_dec:
            st.markdown("\n".join(f"- {d}" for d in m["key_decisions"]) or "_No key decisions found._")
        with t_q:
            st.markdown("\n".join(f"- {q}" for q in m["open_questions"]) or "_No open questions found._")
        with t_tr:
            st.text_area(
                "Transcript", segments_to_timestamped_text(m["segments"]), height=420, label_visibility="collapsed"
            )

        c1, c2, _ = st.columns([2, 2, 4])
        c1.download_button(
            "⬇️ Download report (.md)", meeting_report(m), file_name=f"{slugify(m['title'])}.md",
            mime="text/markdown", width="stretch",
        )
        with c2.popover("🗑️ Delete meeting", width="stretch"):
            st.write("This removes the meeting and its transcript from memory.")
            if st.button("Yes, delete it", type="primary"):
                vector_store.delete_source(WS, chosen)
                st.session_state.pop("selected_meeting", None)
                st.session_state.pop("last_result", None)
                st.rerun()

# ---- Ask ------------------------------------------------------------------------

with tab_ask:
    all_projects = st.toggle(
        "Search all projects", value=False, help="Off: only this project. On: every project in your workspace."
    )
    if not sources and not all_projects:
        st.info(f"Nothing in **{PROJECT}** yet. Add a meeting or PDF first, or turn on *Search all projects*.")
    else:
        history_key = f"chat_{WS}_{'*' if all_projects else PROJECT}"
        history = st.session_state.setdefault(history_key, [])
        f1, f2 = st.columns([5, 1], vertical_alignment="bottom")
        if all_projects:
            scope = []
            f1.caption(f"Searching every project: {', '.join(projects)}")
        else:
            scope = f1.multiselect(
                f"Search in ({PROJECT})",
                [x["source_id"] for x in sources],
                format_func=source_label,
                placeholder=f"Everything in {PROJECT}",
            )
        if f2.button("Clear chat", width="stretch"):
            st.session_state[history_key] = []
            st.rerun()

        if not history:
            st.caption("💡 Try asking")
            suggestions = [
                "What did we decide across all meetings?",
                "What changed between the last two meetings?",
                "What deadlines were mentioned?",
            ]
            for col, s in zip(st.columns(3), suggestions):
                if col.button(s, width="stretch", key=f"sugg_{s}"):
                    st.session_state.pending_question = s

        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander(f"Sources ({len(msg['sources'])})"):
                        for s in msg["sources"]:
                            st.markdown(
                                f"<div class='source-line'><b>[{s['ref']}] {esc(s.get('project', ''))} › {esc(rag_engine.describe(s))}</b><br>"
                                f"{esc(s['text'][:300])}…</div>",
                                unsafe_allow_html=True,
                            )

        typed = st.chat_input("Ask anything about your meetings and documents…")
        question = typed or st.session_state.pop("pending_question", None)
        if question:
            with st.spinner("Searching your meetings…"):
                try:
                    result = rag_engine.answer_question(
                        WS, question, source_ids=scope or None, history=history,
                        project=None if all_projects else PROJECT,
                    )
                except Exception as err:  # noqa: BLE001
                    result = {"answer": f"⚠️ Something went wrong: {err}", "sources": []}
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
            st.rerun()

# ---- Follow-ups -------------------------------------------------------------------

with tab_follow:
    if not all_items:
        st.info("Action items from your meetings will appear here, and later meetings will update their status.")
    else:
        show_done = st.toggle("Show completed items", value=False)
        rows, keys = [], []
        for m, a in sorted(all_items, key=lambda p: p[0]["date"], reverse=True):
            if a["status"] == "done" and not show_done:
                continue
            last_update = a["updates"][-1] if a.get("updates") else None
            rows.append({
                "Status": STATUS_LABEL[a["status"]],
                "Task": a["task"],
                "Owner": a["owner"],
                "Deadline": a["deadline"],
                "From meeting": f"{m['title']} ({nice_date(m['date'])})",
                "Latest update": (
                    f"{nice_date(last_update['date'])}: {last_update['evidence']}" if last_update else "—"
                ),
            })
            keys.append((m["source_id"], a["id"], a["status"]))

        if not rows:
            st.success("🎉 Every action item is done.")
        else:
            st.caption("Statuses update automatically when a later meeting mentions a task. You can also change them here.")
            edited = st.data_editor(
                rows,
                hide_index=True,
                width="stretch",
                disabled=["Task", "Owner", "Deadline", "From meeting", "Latest update"],
                column_config={
                    "Status": st.column_config.SelectboxColumn(options=list(STATUS_LABEL.values()), required=True, width="small"),
                    "Task": st.column_config.TextColumn(width="large"),
                    "Latest update": st.column_config.TextColumn(width="large"),
                },
                key=f"followups_{WS}_{PROJECT}_{show_done}",
            )
            reverse = {v: k for k, v in STATUS_LABEL.items()}
            changed = False
            for row, (source_id, item_id, old) in zip(edited, keys):
                new = reverse[row["Status"]]
                if new != old:
                    tracker.set_item_status(WS, source_id, item_id, new)
                    changed = True
            if changed:
                # Edits are stored by row position; drop them so they can't land on
                # a different row once the table is rebuilt from the saved data.
                del st.session_state[f"followups_{WS}_{PROJECT}_{show_done}"]
                st.rerun()
