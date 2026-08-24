import os
import re

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import ask_question, build_rag_chain
from core.summarizer import generate_title, summarize
from core.transcriber import transcribe_all
from utils.audio_processor import process_input

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(
    page_title="Recap AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Menu/deploy clutter is hidden via [client] toolbarMode = "minimal" in
       .streamlit/config.toml, which leaves the sidebar toggle untouched. */
    #MainMenu, footer { visibility: hidden; height: 0; }
    header[data-testid="stHeader"] { background: transparent; }

    :root {
        --accent: #6c5ce7;
        --accent-dark: #5b4bd6;
        --accent-soft: #f1effe;
        --ink: #1a1a2e;
        --muted: #6b7280;
        --border: #edeef2;
        --success: #1e8e5a;
        --success-soft: #eafaf1;
        --danger: #c0392b;
        --danger-soft: #fdecea;
    }

    /* ---------- Sidebar shell ---------- */
    section[data-testid="stSidebar"] {
        background: #fafafa;
        border-right: 1px solid #ececec;
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.75rem; }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 1.5rem;
    }
    .brand-badge {
        flex-shrink: 0;
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--accent), #a29bfe);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        box-shadow: 0 4px 12px rgba(108, 92, 231, 0.35);
    }
    .brand-title {
        font-size: 1.12rem;
        font-weight: 800;
        color: var(--ink);
        line-height: 1.25;
    }
    .brand-sub {
        font-size: 0.78rem;
        color: var(--muted);
    }

    .sidebar-section-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 1.15rem 0 0.5rem 0;
    }

    /* ---------- Status chips ---------- */
    .chip-row { margin: 0.6rem 0 0.2rem 0; }
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 600;
        margin: 0 0.4rem 0.4rem 0;
    }
    .chip-ok { background: var(--success-soft); color: var(--success); }
    .chip-warn { background: var(--danger-soft); color: var(--danger); }

    .meta-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.32rem 0.8rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent-dark);
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0 0.4rem 0.5rem 0;
    }
    .chips-row { margin-bottom: 1.3rem; }

    /* ---------- Current video card (sidebar) ---------- */
    .current-video-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.9rem 1rem;
    }
    .cvc-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.3rem;
    }
    .cvc-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: var(--ink);
        line-height: 1.35;
        margin-bottom: 0.2rem;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .cvc-meta { font-size: 0.78rem; color: var(--muted); }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: none;
        transition: transform 0.05s ease-in-out;
    }
    .stButton > button:active { transform: scale(0.98); }
    .stButton > button[kind="primary"] {
        background: var(--accent);
        box-shadow: 0 4px 14px rgba(108, 92, 231, 0.35);
    }
    .stButton > button[kind="primary"]:disabled {
        background: #d8d5f5;
        box-shadow: none;
    }

    /* ---------- Empty state ---------- */
    .empty-hero {
        text-align: center;
        padding: 3.5rem 2rem 2.5rem 2rem;
        max-width: 640px;
        margin: 0 auto;
    }
    .empty-hero h1 { font-size: 2.1rem; font-weight: 800; color: var(--ink); margin-bottom: 0.6rem; }
    .empty-hero p { font-size: 1.02rem; color: var(--muted); line-height: 1.6; }
    .empty-hero .emoji { font-size: 3rem; margin-bottom: 1rem; }

    .feature-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.5rem 1.2rem;
        text-align: center;
        height: 100%;
    }
    .fc-icon { font-size: 1.7rem; margin-bottom: 0.55rem; }
    .fc-title { font-weight: 700; color: var(--ink); margin-bottom: 0.35rem; font-size: 0.98rem; }
    .fc-desc { font-size: 0.85rem; color: var(--muted); line-height: 1.55; }

    /* ---------- Result header ---------- */
    .video-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: var(--ink);
        line-height: 1.3;
        margin-bottom: 0.6rem;
    }

    div[data-testid="stTabs"] button[role="tab"] {
        font-weight: 600;
        font-size: 0.92rem;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent) !important; }
    div[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }

    .content-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }

    .stChatInput { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text or "").strip().lower()
    return re.sub(r"[\s_-]+", "-", text)[:60] or "video"


def count_list_items(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"(?m)^\s*\d+[.)]\s+", text))


def chip(label: str, ok: bool) -> str:
    cls = "chip-ok" if ok else "chip-warn"
    icon = "✓" if ok else "✕"
    return f'<span class="chip {cls}">{icon} {label}</span>'


with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-badge">🎬</div>
            <div>
                <div class="brand-title">Recap AI</div>
                <div class="brand-sub">Transcript, summary &amp; chat</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-label">1 · Add a video</div>', unsafe_allow_html=True)
    source_type = st.segmented_control(
        "Source",
        ["YouTube URL", "Upload file"],
        default="YouTube URL",
        required=True,
        label_visibility="collapsed",
    )

    source = None
    if source_type == "Upload file":
        uploaded = st.file_uploader(
            "Upload audio or video", type=["mp4", "mp3", "wav", "m4a", "webm", "mov"], label_visibility="collapsed"
        )
        if uploaded is not None:
            source = os.path.join(UPLOAD_DIR, uploaded.name)
            with open(source, "wb") as f:
                f.write(uploaded.getbuffer())
    else:
        source = st.text_input(
            "YouTube URL", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed"
        ).strip() or None

    st.markdown('<div class="sidebar-section-label">2 · Language</div>', unsafe_allow_html=True)
    language = st.selectbox(
        "Language",
        ["english", "hinglish"],
        format_func=lambda x: "English (Whisper, local)" if x == "english" else "Hinglish → English (Sarvam API)",
        label_visibility="collapsed",
    )

    engine_label = "Sarvam AI" if language == "hinglish" else "Whisper (local)"
    engine_ok = bool(os.getenv("SARVAM_API_KEY")) if language == "hinglish" else True
    mistral_ok = bool(os.getenv("MISTRAL_API_KEY"))

    st.markdown(
        f'<div class="chip-row">{chip(engine_label, engine_ok)}{chip("Mistral LLM", mistral_ok)}</div>',
        unsafe_allow_html=True,
    )

    missing_key = not engine_ok or not mistral_ok
    if missing_key:
        missing = []
        if not engine_ok:
            missing.append("SARVAM_API_KEY")
        if not mistral_ok:
            missing.append("MISTRAL_API_KEY")
        st.caption(f"⚠️ Missing from .env: {', '.join(missing)}")

    st.divider()
    process_clicked = st.button(
        "🚀 Process video",
        use_container_width=True,
        type="primary",
        disabled=not source or missing_key,
        help="Add a source and configure the required API key(s) above." if (not source or missing_key) else None,
    )

    if st.session_state.result:
        st.divider()
        r = st.session_state.result
        st.markdown(
            f"""
            <div class="current-video-card">
                <div class="cvc-label">Now viewing</div>
                <div class="cvc-title">{r['title']}</div>
                <div class="cvc-meta">{len(r['transcript'].split())} words</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Clear & start over", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.rerun()

if process_clicked and source:
    st.session_state.chat_history = []
    with st.status("Processing your video…", expanded=True) as status:
        try:
            status.write("📥 Downloading & preparing audio…")
            chunks = process_input(source)

            status.write("🎙️ Transcribing audio…this is the slow part, hang tight")
            transcript = transcribe_all(chunks, language=language)

            status.write("📌 Generating title…")
            title = generate_title(transcript)

            status.write("📋 Writing summary…")
            summary = summarize(transcript)

            status.write("✅ Extracting action items…")
            action_items = extract_action_items(transcript)

            status.write("🔑 Extracting key decisions…")
            decisions = extract_key_decisions(transcript)

            status.write("❓ Extracting open questions…")
            questions = extract_questions(transcript)

            status.write("🧠 Building knowledge base for chat…")
            rag_chain = build_rag_chain(transcript)

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            status.update(label="Done!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Processing failed", state="error")
            st.error(f"Something went wrong: {e}")

result = st.session_state.result

if not result:
    st.markdown(
        """
        <div class="empty-hero">
            <div class="emoji">🎬</div>
            <h1>Turn any video into a conversation</h1>
            <p>Paste a YouTube link or upload a file in the sidebar. You'll get a clean transcript,
            a summary, extracted action items and decisions — and a chat you can ask anything about
            the video.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    features = [
        ("📝", "Clean transcript", "Local Whisper or Sarvam AI turns speech into accurate, readable text."),
        ("📋", "Instant insights", "Auto-generated summary, action items, decisions and open questions."),
        ("💬", "Chat with it", "Ask anything about the video — answers are grounded in the transcript."),
    ]
    cols = st.columns(3)
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="fc-icon">{icon}</div>
                    <div class="fc-title">{title}</div>
                    <div class="fc-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    slug = slugify(result["title"])
    word_count = len(result["transcript"].split())
    ai_count = count_list_items(result["action_items"])
    kd_count = count_list_items(result["key_decisions"])
    oq_count = count_list_items(result["open_questions"])

    header_col, home_col = st.columns([6, 1], vertical_alignment="center")
    with header_col:
        st.markdown(f'<div class="video-title">📌 {result["title"]}</div>', unsafe_allow_html=True)
    with home_col:
        if st.button("🏠 Home", use_container_width=True, help="Clear this video and start over"):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.rerun()

    st.markdown(
        f"""
        <div class="chips-row">
            <span class="meta-chip">📝 {word_count} words</span>
            <span class="meta-chip">✅ {ai_count} action items</span>
            <span class="meta-chip">🔑 {kd_count} decisions</span>
            <span class="meta-chip">❓ {oq_count} open questions</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_summary, tab_transcript, tab_actions, tab_decisions, tab_questions, tab_chat = st.tabs(
        ["📋 Summary", "📝 Transcript", "✅ Action Items", "🔑 Key Decisions", "❓ Open Questions", "💬 Chat"]
    )

    with tab_summary:
        st.markdown(f'<div class="content-card">{result["summary"]}</div>', unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download summary", result["summary"], file_name=f"{slug}-summary.txt", mime="text/plain"
        )

    with tab_transcript:
        st.text_area("Transcript", result["transcript"], height=500, label_visibility="collapsed")
        st.download_button(
            "⬇️ Download transcript", result["transcript"], file_name=f"{slug}-transcript.txt", mime="text/plain"
        )

    with tab_actions:
        st.markdown(f'<div class="content-card">{result["action_items"]}</div>', unsafe_allow_html=True)

    with tab_decisions:
        st.markdown(f'<div class="content-card">{result["key_decisions"]}</div>', unsafe_allow_html=True)

    with tab_questions:
        st.markdown(f'<div class="content-card">{result["open_questions"]}</div>', unsafe_allow_html=True)

    with tab_chat:
        if not st.session_state.chat_history:
            st.caption("💡 Try asking")
            suggestions = ["Summarize the key takeaways", "What decisions were made?", "List any deadlines mentioned"]
            sugg_cols = st.columns(3)
            for col, s in zip(sugg_cols, suggestions):
                with col:
                    if st.button(s, use_container_width=True, key=f"sugg_{s}"):
                        st.session_state.pending_question = s

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        typed_question = st.chat_input("Ask something about this video…")
        question = typed_question or st.session_state.pop("pending_question", None)

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("Thinking…"):
                answer = ask_question(result["rag_chain"], question)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()
