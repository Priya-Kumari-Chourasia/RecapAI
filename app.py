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

from core import factcheck, rag_engine, tracker, vector_store  # noqa: E402
from core.documents import NoTextInPdfError, process_pdf  # noqa: E402
from core.pipeline import process_recording  # noqa: E402
from core.sample_data import DEMO_KEY, DEMO_PROJECT, ensure_demo_project  # noqa: E402
from core.transcriber import ENGINES, engine_ready, format_timestamp, segments_to_timestamped_text  # noqa: E402
from utils.audio_processor import YouTubeBlockedError  # noqa: E402

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Recap AI", page_icon="📓", layout="wide", initial_sidebar_state="expanded")

# Visual identity: a lab notebook. Deep notebook-green ink on cool paper, a serif for
# titles (Literata, made for long reading), Plex Sans for everything else. The one loud
# element is the highlighter: evidence quotes are "marked" in yellow, the way you'd
# mark the line that proves a point.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Literata:opsz,wght@7..72,500;7..72,650&display=swap');

    :root {
        --paper: #f5f7f4; --sheet: #ffffff; --line: #dde3dd; --line-soft: #e9eee9;
        --ink: #1c2b2a; --ink-soft: #55635f; --ink-faint: #8a958f;
        --green: #2f5d50; --green-deep: #23473d; --green-soft: #e3ece7;
        --marker: #fbe68a; --amber: #9a6410; --amber-soft: #fbf1dc; --red: #a93a32; --red-soft: #f8e4e1;
    }
    html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'IBM Plex Sans', sans-serif; }
    #MainMenu, footer { visibility: hidden; height: 0; }
    header[data-testid="stHeader"] { background: transparent; height: 2rem; }
    .block-container, [data-testid="stMainBlockContainer"], [data-testid="stAppViewBlockContainer"] {
        padding-top: 1.6rem !important; max-width: 1180px; }
    h1, h2, h3, .serif { font-family: 'Literata', serif !important; letter-spacing: -0.01em; }

    /* sidebar */
    section[data-testid="stSidebar"] { background: var(--sheet); border-right: 1px solid var(--line); }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
    .wordmark { font-family: 'Literata', serif; font-size: 1.45rem; font-weight: 650; color: var(--ink); line-height: 1; }
    .wordmark small { font-family: 'IBM Plex Sans', sans-serif; font-size: 0.72rem; font-weight: 600; color: var(--green);
        margin-left: 0.3rem; vertical-align: 0.35rem; }
    .tagline { color: var(--ink-soft); font-size: 0.82rem; margin: 0.35rem 0 1.4rem; }
    .side-label { font-size: 0.78rem; font-weight: 600; color: var(--ink-soft); margin: 1.1rem 0 0.35rem; }
    .svc { font-size: 0.8rem; color: var(--ink-soft); line-height: 1.75; }
    .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 0.45rem; vertical-align: 1px; }
    .dot.on { background: #3b8a5c; } .dot.off { background: #c9b7b3; } .dot.info { background: var(--green); }

    /* page header */
    .page-head { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
        border-bottom: 1px solid var(--line); padding-bottom: 0.9rem; margin-bottom: 0.4rem; }
    .page-head h1 { font-size: 2.15rem; margin: 0; padding: 0; color: var(--ink); font-weight: 650; }
    .page-head .facts { color: var(--ink-soft); font-size: 0.92rem; }
    .page-head .facts b { color: var(--ink); font-weight: 600; }

    /* tabs */
    div[data-testid="stTabs"] button[role="tab"] p { font-size: 0.95rem; font-weight: 500; }
    div[data-testid="stTabs"] button[aria-selected="true"] p { color: var(--green-deep); font-weight: 600; }
    div[data-baseweb="tab-highlight"] { background-color: var(--green) !important; height: 2px; }
    div[data-baseweb="tab-border"] { background-color: var(--line); }

    /* buttons */
    .stButton > button, .stDownloadButton > button, div[data-testid="stPopover"] > div > button {
        border-radius: 8px; font-weight: 500; border: 1px solid var(--line); }
    .stButton > button[kind="primary"] { background: var(--green); border-color: var(--green); color: #fff; }
    .stButton > button[kind="primary"]:hover { background: var(--green-deep); border-color: var(--green-deep); }
    .stButton > button[kind="primary"]:disabled { background: #c7d3cd; border-color: #c7d3cd; }

    /* library list: a radio styled as an index of entries */
    /* Works on Streamlit 1.62 (options are <label>s) and 1.66+ (options are <div>s wrapping a <label>). */
    .st-key-library_index [role="radiogroup"] { gap: 0.15rem; }
    .st-key-library_index [role="radiogroup"] > * {
        width: 100%; padding: 0.55rem 0.75rem; margin: 0; border-left: 3px solid transparent; border-radius: 0 8px 8px 0; }
    .st-key-library_index [role="radiogroup"] > *:hover { background: var(--line-soft); }
    .st-key-library_index [role="radiogroup"] > [data-selected="true"] {
        background: var(--green-soft); border-left-color: var(--green); }
    .st-key-library_index [data-testid="stRadioOption"] { width: 100%; cursor: pointer; }
    .st-key-library_index [data-testid="stRadioOption"] > div { align-items: flex-start; text-align: left; }
    /* hide only the radio circle: the sibling just before the title text */
    .st-key-library_index div:has(> [data-testid="stMarkdownContainer"]) > div:first-child:not([data-testid]) { display: none; }
    .st-key-library_index [data-testid="stRadioOption"] p { font-weight: 500; color: var(--ink); line-height: 1.35; }
    .st-key-library_index [data-testid="stCaptionContainer"] p,
    .st-key-library_index [data-testid="stRadioCaption"] p { font-weight: 400; color: var(--ink-soft); font-size: 0.82rem; }
    .st-key-library_index div:has(> [data-testid="stCaptionContainer"]) { padding-left: 0 !important; margin-left: 0 !important; }
    .st-key-library_index [data-testid="stRadioCaption"] { padding-left: 0; margin-top: 0.1rem; }
    .st-key-library_index [data-testid="stRadioOption"]:focus-visible { outline: 2px solid var(--green); outline-offset: 2px; }

    /* detail header */
    .entry-kind { color: var(--green); font-size: 0.85rem; font-weight: 600; margin-bottom: 0.1rem; }
    .entry-title { font-family: 'Literata', serif; font-size: 1.6rem; font-weight: 650; color: var(--ink); line-height: 1.25; }
    .entry-meta { color: var(--ink-soft); font-size: 0.88rem; margin: 0.35rem 0 0.9rem; }

    /* timestamps and pages read like a tape counter */
    .stamp { display: inline-block; font-variant-numeric: tabular-nums; font-size: 0.78rem; font-weight: 600;
        color: var(--green-deep); background: var(--green-soft); border-radius: 4px; padding: 0 0.35rem; white-space: nowrap; }

    /* fact-check */
    .fc-head { display: flex; align-items: center; gap: 0.9rem; flex-wrap: wrap; margin-bottom: 0.6rem; }
    .fc-head .verdict { font-size: 0.95rem; color: var(--ink); }
    .fc-head .verdict b { font-weight: 600; }
    .fc-bar { flex: 1; min-width: 140px; height: 6px; border-radius: 3px; display: flex; overflow: hidden; background: var(--line-soft); }
    .fc-bar span:nth-child(1) { background: #3b8a5c; } .fc-bar span:nth-child(2) { background: #d2a03c; }
    .fc-bar span:nth-child(3) { background: #c0574e; }
    .fc-pct { font-size: 0.85rem; font-weight: 600; color: var(--green-deep); }
    .claim { border-left: 3px solid var(--line); padding: 0.45rem 0 0.45rem 0.85rem; margin: 0.35rem 0; }
    .claim.supported { border-left-color: #3b8a5c; } .claim.partial { border-left-color: #d2a03c; }
    .claim.unsupported { border-left-color: #c0574e; }
    .claim .tag { font-size: 0.74rem; font-weight: 600; margin-right: 0.4rem; }
    .claim.supported .tag { color: #2f6e4a; } .claim.partial .tag { color: var(--amber); } .claim.unsupported .tag { color: var(--red); }
    .claim .text { color: var(--ink); }
    .claim .ev { margin-top: 0.3rem; font-size: 0.86rem; color: var(--ink-soft); }
    .stMarkdown mark { background: linear-gradient(transparent 12%, var(--marker) 12%, var(--marker) 88%, transparent 88%);
        color: var(--ink); padding: 0 0.15rem; }
    .claim .note { font-size: 0.82rem; color: var(--amber); margin-top: 0.2rem; }
    .claim.unsupported .note { color: var(--red); }

    /* tasks */
    .status { font-size: 0.74rem; font-weight: 600; padding: 0.1rem 0.45rem; border-radius: 4px; white-space: nowrap; }
    .status-open { background: var(--red-soft); color: var(--red); }
    .status-in_progress { background: var(--amber-soft); color: var(--amber); }
    .status-done { background: var(--green-soft); color: #2f6e4a; }
    .task { padding: 0.55rem 0; border-bottom: 1px solid var(--line-soft); }
    .task .who { color: var(--ink-soft); font-size: 0.86rem; }
    .task .hist { color: var(--ink-soft); font-size: 0.84rem; margin: 0.25rem 0 0 0.2rem; }
    .task-counts { display: flex; gap: 1.6rem; margin: 0.3rem 0 0.9rem; flex-wrap: wrap; }
    .task-counts div { font-size: 0.88rem; color: var(--ink-soft); }
    .task-counts b { font-family: 'Literata', serif; font-size: 1.5rem; color: var(--ink); margin-right: 0.35rem; }

    /* add tab */
    .steps { counter-reset: s; padding: 0.2rem 0 0 0; }
    .steps div { counter-increment: s; position: relative; padding: 0 0 1rem 2.1rem; color: var(--ink-soft); font-size: 0.9rem; }
    .steps div::before { content: counter(s); position: absolute; left: 0; top: 0; width: 1.45rem; height: 1.45rem;
        border-radius: 50%; border: 1.5px solid var(--green); color: var(--green); font-size: 0.78rem; font-weight: 600;
        display: flex; align-items: center; justify-content: center; }
    .steps b { color: var(--ink); font-weight: 600; display: block; }
    .recent-row { padding: 0.45rem 0; border-bottom: 1px solid var(--line-soft); font-size: 0.9rem; color: var(--ink); }
    .recent-row span { color: var(--ink-faint); font-size: 0.8rem; display: block; }
    .empty { padding: 2.2rem 0 1rem; color: var(--ink-soft); max-width: 560px; }
    .empty h3 { color: var(--ink); font-size: 1.35rem; margin-bottom: 0.4rem; }

    /* ask: chat avatars in the notebook palette instead of Streamlit's red/orange */
    [data-testid="stChatMessageAvatarUser"] { background: var(--line) !important; color: var(--ink) !important; }
    [data-testid="stChatMessageAvatarAssistant"] { background: var(--green) !important; color: #fff !important; }
    [data-testid="stChatMessage"] { background: transparent; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: var(--sheet); border: 1px solid var(--line-soft); }
    /* ask */
    .src { padding: 0.45rem 0; border-bottom: 1px solid var(--line-soft); font-size: 0.86rem; color: var(--ink-soft); }
    .src b { color: var(--ink); font-weight: 600; }
    .scope-note { color: var(--ink-soft); font-size: 0.88rem; }

    /* welcome page */
    .welcome-top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 2rem; }
    .welcome-top a { color: var(--ink-soft); font-size: 0.9rem; text-decoration: none; border-bottom: 1px solid var(--line); }
    .welcome-top a:hover { color: var(--green-deep); border-bottom-color: var(--green); }
    .welcome h1 { font-size: 3rem; line-height: 1.08; font-weight: 650; color: var(--ink); margin: 0 0 1.1rem; max-width: 17ch; }
    .welcome p.lede { font-size: 1.12rem; line-height: 1.6; color: var(--ink-soft); max-width: 46ch; margin-bottom: 1.6rem; }
    .welcome .built { color: var(--ink-faint); font-size: 0.84rem; margin-top: 1.2rem; }
    .specimen { background: var(--sheet); border: 1px solid var(--line); border-radius: 10px; padding: 1.2rem 1.3rem 0.8rem;
        box-shadow: 0 18px 40px -28px rgba(28, 43, 42, 0.45); }
    .specimen .entry-title { font-size: 1.2rem; }
    .specimen .entry-meta { margin-bottom: 0.7rem; }
    .specimen-caption { color: var(--ink-faint); font-size: 0.8rem; margin-top: 0.6rem; text-align: right; }
    .uses { display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; margin-top: 3.2rem;
        border-top: 1px solid var(--line); padding-top: 1.4rem; }
    .uses h4 { font-family: 'Literata', serif; font-size: 1.05rem; color: var(--ink); margin: 0 0 0.35rem; }
    .uses p { color: var(--ink-soft); font-size: 0.92rem; line-height: 1.55; margin: 0; }
    .uses .eg { margin-top: 0.55rem; font-size: 0.86rem; color: var(--ink); }
    @media (max-width: 900px) { .uses { grid-template-columns: 1fr; gap: 1.2rem; } .welcome h1 { font-size: 2.3rem; } }
    @media (max-width: 640px) {
        .page-head h1 { font-size: 1.7rem; }
        .entry-title { font-size: 1.35rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

STATUS_LABEL = {"open": "Not started", "in_progress": "In progress", "done": "Done"}
CLAIM_TAG = {"supported": "Verified", "partial": "Partly true", "unsupported": "Not in source"}


def esc(text) -> str:
    return html.escape(str(text or ""))


def status_badge(status: str) -> str:
    return f'<span class="status status-{status}">{STATUS_LABEL.get(status, status)}</span>'


def stamp(text: str) -> str:
    return f'<span class="stamp">{esc(text)}</span>' if text else ""


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text or "").strip().lower()
    return re.sub(r"[\s_-]+", "-", text)[:60] or "meeting"


def nice_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return iso or ""


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def where_label(c: dict) -> str:
    if c.get("page"):
        return f"p. {c['page']}"
    return format_timestamp(c["start"]) if c.get("start") is not None else ""


def render_fact_check(report: dict, show_title: bool = False) -> None:
    """Score bar, then each claim with its verdict, the highlighted evidence and where it was said."""
    sc = report["score"]
    if not sc["total"]:
        st.caption("Nothing to check.")
        return
    widths = [100 * sc[k] / sc["total"] for k in ("supported", "partial", "unsupported")]
    extra = []
    if sc["partial"]:
        extra.append(f"{sc['partial']} partly true")
    if sc["unsupported"]:
        extra.append(f"{sc['unsupported']} not in the source")
    bar = "".join(f"<span style='width:{w:.1f}%'></span>" for w in widths)
    st.markdown(
        f"<div class='fc-head'><div class='verdict'><b>{sc['supported']} of {sc['total']}</b> points verified"
        f"{(' (' + ', '.join(extra) + ')') if extra else ''}</div>"
        f"<div class='fc-bar'>{bar}</div>"
        f"<div class='fc-pct'>{round(100 * sc['faithfulness'])}% faithful</div></div>",
        unsafe_allow_html=True,
    )
    for c in report["claims"]:
        ev = ""
        if c.get("evidence"):
            src = f"{esc(c['title'])} " if show_title and c.get("title") else ""
            ev = f"<div class='ev'><mark>{esc(c['evidence'])}</mark> {src}{stamp(where_label(c))}</div>"
        note = f"<div class='note'>{esc(c['note'])}</div>" if c.get("note") and c["status"] != "supported" else ""
        st.markdown(
            f"<div class='claim {c['status']}'><span class='tag'>{CLAIM_TAG[c['status']]}</span>"
            f"<span class='text'>{esc(c['claim'])}</span>{ev}{note}</div>",
            unsafe_allow_html=True,
        )


def meeting_report(m: dict) -> str:
    lines = [f"# {m['title']}", f"_{nice_date(m['date'])}, {format_timestamp(m.get('duration', 0))}_", "",
             "## Summary", m.get("summary", ""), ""]
    if m.get("fact_check"):
        sc = m["fact_check"]["score"]
        lines += [f"## Fact-check: {sc['supported']}/{sc['total']} points verified "
                  f"({round(100 * (sc['faithfulness'] or 0))}% faithful)"]
        for c in m["fact_check"]["claims"]:
            where = f" ({format_timestamp(c['start'])})" if c.get("start") is not None else ""
            lines.append(f"- {factcheck.STATUS_ICON[c['status']]} {c['claim']}"
                         + (f"  \n  > “{c['evidence']}”{where}" if c.get("evidence") else "")
                         + (f"  \n  _{c['note']}_" if c.get("note") and c["status"] != "supported" else ""))
        lines.append("")
    lines += ["## Action items"]
    for a in m.get("action_items", []):
        lines.append(f"- [{'x' if a['status'] == 'done' else ' '}] {a['task']} ({a['owner']}, deadline: {a['deadline']})")
    lines += ["", "## Key decisions", *[f"- {d}" for d in m.get("key_decisions", [])],
              "", "## Open questions", *[f"- {q}" for q in m.get("open_questions", [])],
              "", "## Transcript", segments_to_timestamped_text(m.get("segments", []))]
    return "\n".join(lines)


@st.cache_resource(show_spinner="Loading the embedding model (first run only)…")
def warm_up_memory():
    vector_store.get_embedder()
    vector_store.get_client()
    return True


@st.dialog("Delete")
def confirm_delete(kind: str, name: str, action) -> None:
    """A real dialog (not a popover) so it closes itself once the item is gone."""
    st.write(f"Delete the {kind} **{name}**? This can't be undone.")
    c1, c2 = st.columns(2)
    if c1.button(f"Delete {kind}", type="primary", width="stretch"):
        action()
        st.rerun()
    if c2.button("Cancel", width="stretch"):
        st.rerun()


# ---------------------------------------------------------------------------- welcome page

REPO_URL = "https://github.com/Priya-Kumari-Chourasia/RecapAI"


def specimen_html() -> str:
    """A real-looking slice of the app: the fact-checked summary of the sample meeting."""
    from core.sample_data import FACT_CHECKS, MEETING_1
    from core.factcheck import split_claims

    rows = []
    for claim, (status, evidence, start, note) in list(zip(split_claims(MEETING_1["summary"]), FACT_CHECKS["m1"]))[:4]:
        ev = f"<div class='ev'><mark>{esc(evidence)}</mark> {stamp(format_timestamp(start))}</div>" if evidence else ""
        nt = f"<div class='note'>{esc(note)}</div>" if note else ""
        rows.append(f"<div class='claim {status}'><span class='tag'>{CLAIM_TAG[status]}</span>"
                    f"<span class='text'>{esc(claim)}</span>{ev}{nt}</div>")
    return (
        "<div class='specimen'>"
        "<div class='entry-kind'>Meeting, 1 Oct</div>"
        f"<div class='entry-title'>{esc(MEETING_1['title'])}</div>"
        "<div class='entry-meta'>12:34 long, 3 tasks, every summary point checked</div>"
        + "".join(rows) +
        "</div><div class='specimen-caption'>From the demo project</div>"
    )


def render_welcome() -> None:
    from core.sample_data import FACT_CHECKS

    verdicts = [v[0] for checks in FACT_CHECKS.values() for v in checks]
    demo_verified, demo_total = verdicts.count("supported"), len(verdicts)
    st.markdown(
        """<style>
        section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
        [data-testid="stExpandSidebarButton"] { display: none; }
        </style>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='welcome-top'><div class='wordmark'>Recap<small>AI</small></div>"
        f"<a href='{REPO_URL}' target='_blank'>Source code on GitHub</a></div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([6, 5], gap="large")
    with left:
        st.markdown(
            "<div class='welcome'><h1>Remember what was said in every meeting.</h1>"
            "<p class='lede'>Record or upload your meetings, lectures and papers. Recap AI writes a summary you can "
            "check point by point, answers questions with the moment it was said, and keeps track of who "
            "promised what.</p></div>",
            unsafe_allow_html=True,
        )
        b1, b2 = st.columns(2)
        if b1.button("Open the demo", type="primary", width="stretch"):
            st.session_state.update(entered=True, workspace_name=DEMO_KEY, seed_demo=True,
                                    switch_to_project=DEMO_PROJECT)
            st.rerun()
        with b2.popover("Use my own workspace", width="stretch"):
            key = st.text_input("Workspace key", type="password",
                                help="Pick a private key and remember it. Entering the same key later, "
                                     "on any device, opens the same projects.")
            if st.button("Open workspace", type="primary", disabled=not key.strip()):
                st.session_state.update(entered=True, workspace_name=key.strip())
                st.rerun()
        st.markdown(
            "<div class='welcome'><div class='built'>Whisper on Groq for speech, Qdrant for memory, "
            "LangChain for the language model, Streamlit for this page.</div></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(specimen_html(), unsafe_allow_html=True)

    st.markdown(
        "<div class='uses'>"
        "<div><h4>Ask across meetings</h4><p>Questions are answered from every meeting and PDF in a project, "
        "with the date and the minute each answer comes from.</p>"
        "<div class='eg'>Which baseline did we pick, and why? <span class='stamp'>0:52</span></div></div>"
        "<div><h4>Tasks that follow up on themselves</h4><p>When a later meeting says a task is finished or under "
        "way, its status changes, with the sentence that shows it.</p>"
        "<div class='eg'><span class='status status-done'>Done</span> Finish the literature review</div></div>"
        "<div><h4>A summary you can trust</h4><p>Every summary point is checked against the transcript and marked "
        "verified, partly true or not in the source.</p>"
        f"<div class='eg'>{demo_verified} of {demo_total} points verified in the demo</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )


if not st.session_state.get("entered") and os.getenv("SKIP_WELCOME", "").lower() not in {"1", "true", "yes"}:
    render_welcome()
    st.stop()

# ---------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.markdown(
        "<div class='wordmark'>Recap<small>AI</small></div>"
        "<div class='tagline'>A searchable memory of every meeting you record.</div>",
        unsafe_allow_html=True,
    )
    # Containers keep the visual order (project first, key last) while the code
    # reads the workspace key first, since the project list depends on it.
    project_box = st.container()
    status_box = st.container()
    key_box = st.expander("Workspace key")

with key_box:
    workspace_name = st.text_input(
        "Workspace key",
        value=st.session_state.get("workspace_name", os.getenv("DEFAULT_WORKSPACE", "demo")),
        type="password",
        label_visibility="collapsed",
    ).strip()
    st.caption(
        "Works like a password: anyone who enters the same key sees the same projects. "
        "On your own laptop you can leave it as it is."
    )
    if st.button("Back to the welcome page", width="stretch"):
        st.session_state.entered = False
        st.rerun()
st.session_state.workspace_name = workspace_name

groq_ok = bool(os.getenv("GROQ_API_KEY"))
sarvam_ok = bool(os.getenv("SARVAM_API_KEY"))
with status_box:
    st.markdown("<div class='side-label'>Services</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='svc'>"
        f"<div><span class='dot {'on' if groq_ok else 'off'}'></span>Groq {'connected' if groq_ok else 'key missing'}</div>"
        f"<div><span class='dot {'on' if sarvam_ok else 'off'}'></span>Sarvam {'connected' if sarvam_ok else 'not set (Hinglish off)'}</div>"
        f"<div><span class='dot info'></span>{vector_store.storage_label()}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    if not groq_ok:
        with st.expander("Why is Groq missing?"):
            if _secrets_error:
                st.caption(f"Couldn't read st.secrets: {_secrets_error}")
            elif _secrets_seen:
                st.caption("Keys found: " + ", ".join(sorted(_secrets_seen)) + ". Check GROQ_API_KEY for a typo.")
            else:
                st.caption(
                    "No keys found. Locally: add GROQ_API_KEY to your .env file. On Streamlit Cloud: "
                    "Manage app, Settings, Secrets; paste keys at the top level, save and reboot."
                )

if not workspace_name:
    with project_box:
        st.info("Enter a workspace key below to start.")
    st.stop()

warm_up_memory()
WS = vector_store.workspace_id(workspace_name)
if st.session_state.pop("seed_demo", False):
    with st.spinner("Setting up the demo project…"):
        ensure_demo_project(WS)
projects = vector_store.list_projects(WS)

with project_box:
    st.markdown("<div class='side-label'>Project</div>", unsafe_allow_html=True)
    wanted = st.session_state.pop("switch_to_project", None) or st.session_state.get("project")
    PROJECT = st.selectbox(
        "Project",
        projects,
        index=projects.index(wanted) if wanted in projects else 0,
        label_visibility="collapsed",
    )
    if PROJECT != st.session_state.get("project"):
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_docs", None)
        st.session_state.pop("selected_meeting", None)
    st.session_state.project = PROJECT

    c_new, c_del = st.columns(2)
    with c_new.popover("New project", width="stretch"):
        new_name = st.text_input("Project name", placeholder="e.g. Thesis, BrickMind", key="new_project_name")
        if st.button("Create project", type="primary", disabled=not new_name.strip()):
            if new_name.strip().lower() in {p.lower() for p in projects}:
                st.warning("A project with that name already exists.")
            else:
                st.session_state.switch_to_project = vector_store.create_project(WS, new_name)
                st.rerun()
    if c_del.button("Delete", width="stretch", disabled=PROJECT == vector_store.DEFAULT_PROJECT,
                    help="Deletes this project and everything in it. The General project can't be deleted."):
        def _delete_project(name=PROJECT):
            vector_store.delete_project(WS, name)
            st.session_state.switch_to_project = vector_store.DEFAULT_PROJECT
        confirm_delete("project", f"{PROJECT} (and everything in it)", _delete_project)

sources = vector_store.list_sources(WS, project=PROJECT)
meetings = [s for s in sources if s.get("source_type") == "meeting"]
docs = [s for s in sources if s.get("source_type") == "pdf"]
all_items = [(m, a) for m in meetings for a in m.get("action_items", [])]
open_count = sum(1 for _, a in all_items if a["status"] != "done")


def source_label(i: str) -> str:
    src = next(x for x in sources if x["source_id"] == i)
    return f"{'PDF' if src.get('source_type') == 'pdf' else 'Meeting'}: {src['title']} ({nice_date(src['date'])})"


def source_caption(src: dict) -> str:
    parts = [nice_date(src["date"])]
    if src.get("source_type") == "pdf":
        parts.append(f"PDF, {plural(src.get('page_count', 0), 'page')}")
    else:
        parts.append(format_timestamp(src.get("duration", 0)))
    fc = src.get("fact_check")
    if fc and fc["score"]["total"]:
        parts.append(f"{fc['score']['supported']}/{fc['score']['total']} verified")
    return ", ".join(parts)


# ---------------------------------------------------------------------------- page header

facts = (f"<b>{len(meetings)}</b> {'meeting' if len(meetings) == 1 else 'meetings'}, "
         f"<b>{len(docs)}</b> {'PDF' if len(docs) == 1 else 'PDFs'}, "
         f"<b>{open_count}</b> unfinished {'task' if open_count == 1 else 'tasks'}")
st.markdown(
    f"<div class='page-head'><h1>{esc(PROJECT)}</h1><div class='facts'>{facts}</div></div>",
    unsafe_allow_html=True,
)

# Fixed labels: if a label changed (e.g. "Library (2)" -> "Library (1)"), Streamlit would treat the
# tabs as new and jump back to the first one. The counts are in the page header instead.
tab_add, tab_library, tab_ask, tab_follow = st.tabs(["Add", "Library", "Ask", "Follow-ups"])

# ---- Add ---------------------------------------------------------------------------

with tab_add:
    form_col, side_col = st.columns([3, 2], gap="large")

    with form_col:
        source_type = st.segmented_control(
            "What are you adding?", ["Audio or video", "Record now", "YouTube link", "PDF"],
            default="Audio or video", required=True,
        )
        is_pdf = source_type == "PDF"
        source, origin, pdf_files = None, None, []
        if is_pdf:
            pdf_files = st.file_uploader(
                "Papers, slides or notes", type=["pdf"], accept_multiple_files=True
            ) or []
        elif source_type == "Audio or video":
            uploaded = st.file_uploader(
                "Meeting or lecture recording", type=["mp4", "mp3", "wav", "m4a", "webm", "mov", "ogg", "flac"]
            )
            if uploaded is not None:
                source = os.path.join(UPLOAD_DIR, os.path.basename(uploaded.name))
                with open(source, "wb") as f:
                    f.write(uploaded.getbuffer())
                origin = "upload"
        elif source_type == "Record now":
            recording = st.audio_input("Record (your browser will ask for microphone access)")
            if recording is not None:
                source = os.path.join(UPLOAD_DIR, f"recording-{datetime.now():%Y%m%d-%H%M%S}.wav")
                with open(source, "wb") as f:
                    f.write(recording.getbuffer())
                origin = "recording"
        else:
            source = st.text_input("YouTube link", placeholder="https://youtube.com/watch?v=...").strip() or None
            origin = "youtube"
            st.caption("YouTube links work when Recap AI runs on your own computer. Hosted servers are often "
                       "blocked by YouTube; if that happens, download the video and upload it instead.")

        c1, c2 = st.columns([3, 2])
        title = c1.text_input(
            "Title (optional)",
            placeholder="Taken from the PDF" if is_pdf else "Written for you if left empty",
            disabled=is_pdf and len(pdf_files) > 1,
            help="With several PDFs, each keeps its own title." if is_pdf else None,
        )
        meeting_date = c2.date_input("Date" if is_pdf else "Meeting date", value=date.today(), max_value=date.today())

        c3, c4 = st.columns([3, 2])
        target = c3.selectbox("Save to project", projects, index=projects.index(PROJECT))
        if is_pdf:
            engine, engine_ok, vocabulary = None, True, None
        else:
            engine_options = [e for e in ENGINES if engine_ready(e)[0] or e != "local"]
            engine = c4.selectbox(
                "Transcription",
                engine_options,
                format_func=lambda e: ENGINES[e] + ("" if engine_ready(e)[0] else f" ({engine_ready(e)[1]})"),
            )
            engine_ok = engine_ready(engine)[0]
            vocabulary = st.text_input(
                "Names and terms (optional)",
                placeholder="Bonsai, XMC, pre-training, Dr. Rao",
                help="Comma-separated. Whisper uses them to spell names and jargon correctly "
                "(for example 'pre-training' instead of 'free training'). Not used by Sarvam.",
            )

        if is_pdf:
            label = f"Add {plural(len(pdf_files), 'PDF') if pdf_files else 'PDFs'} to {target}"
            ready = bool(pdf_files and groq_ok)
        else:
            label = f"Transcribe and save to {target}"
            ready = bool(source and engine_ok and groq_ok)
        process_clicked = st.button(label, type="primary", width="stretch", disabled=not ready)

    with side_col:
        if not sources:
            st.markdown(
                f"<h3 class='serif' style='font-size:1.15rem;margin:0.2rem 0 0.9rem'>{esc(PROJECT)} is empty. Here's how it fills up:</h3>"
                "<div class='steps'>"
                "<div><b>Add a meeting, lecture or PDF</b>Record it here, upload a file, or paste a YouTube link.</div>"
                "<div><b>Get a checked summary</b>Transcript with timestamps, action items, decisions, and every "
                "summary point verified against what was actually said.</div>"
                "<div><b>Ask and follow up</b>Ask questions across everything in the project. Tasks from earlier "
                "meetings are marked done when a later meeting says so.</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"<div class='side-label' style='margin-top:0.2rem'>Recently added to {esc(PROJECT)}</div>",
                        unsafe_allow_html=True)
            for s in sources[:6]:
                st.markdown(f"<div class='recent-row'>{esc(s['title'])}<span>{esc(source_caption(s))}</span></div>",
                            unsafe_allow_html=True)

    if process_clicked and is_pdf and pdf_files:
        added, failed = [], []
        with st.status(f"Adding {plural(len(pdf_files), 'PDF')}…", expanded=True) as status:
            for f in pdf_files:
                path = os.path.join(UPLOAD_DIR, os.path.basename(f.name))
                with open(path, "wb") as out_file:
                    out_file.write(f.getbuffer())
                try:
                    added.append(process_pdf(
                        path, WS, project=target, title=(title or None) if len(pdf_files) == 1 else None,
                        doc_date=meeting_date.isoformat(), progress=lambda msg, n=f.name: status.write(f"{n}: {msg}"),
                    ))
                except NoTextInPdfError as err:
                    failed.append((f.name, str(err)))
                except Exception as err:  # noqa: BLE001
                    failed.append((f.name, f"Couldn't add it: {err}"))
            status.update(
                label=f"Added {plural(len(added), 'PDF')}" + (f", {len(failed)} failed" if failed else ""),
                state="error" if failed and not added else "complete", expanded=bool(failed),
            )
        for name, msg in failed:
            st.warning(f"**{name}**: {msg}")
        if added:
            st.session_state.last_docs = added
            st.session_state.pop("last_result", None)
            st.session_state.selected_meeting = added[-1]["source_id"]
            st.session_state.switch_to_project = st.session_state.project = target
            if not failed:
                st.rerun()

    if process_clicked and not is_pdf and source:
        failure = None
        with st.status("Processing the recording…", expanded=True) as status:
            try:
                out = process_recording(
                    source, workspace=WS, project=target, engine=engine, title=title or None,
                    meeting_date=meeting_date.isoformat(), origin=origin, vocabulary=vocabulary or None,
                    progress=status.write,
                )
            except YouTubeBlockedError as err:
                status.update(label="YouTube blocked the download", state="error", expanded=False)
                failure = ("warning", str(err))
            except Exception as err:  # noqa: BLE001 - show any failure in the UI
                status.update(label="Processing stopped", state="error", expanded=False)
                failure = ("error", f"Processing stopped: {err}")
            else:
                st.session_state.last_result = out
                st.session_state.pop("last_docs", None)
                st.session_state.selected_meeting = out["meeting"]["source_id"]
                st.session_state.switch_to_project = st.session_state.project = target
                status.update(label=f"Saved to {target}", state="complete", expanded=False)
        # Shown outside the (collapsed) status box, so the explanation is always visible.
        if failure:
            (st.warning if failure[0] == "warning" else st.error)(failure[1])
        else:
            st.rerun()

    for d in st.session_state.get("last_docs", []):
        with st.container(border=True):
            st.markdown(f"<div class='entry-kind'>PDF added</div><div class='entry-title'>{esc(d['title'])}</div>"
                        f"<div class='entry-meta'>{plural(d['page_count'], 'page')}, {d['word_count']} words. "
                        "Ask about it in the Ask tab.</div>", unsafe_allow_html=True)
            if d.get("fact_check"):
                render_fact_check(d["fact_check"])
            else:
                st.markdown(d["summary"])

    last = st.session_state.get("last_result")
    if last:
        m = last["meeting"]
        with st.container(border=True):
            st.markdown(
                f"<div class='entry-kind'>Meeting saved</div><div class='entry-title'>{esc(m['title'])}</div>"
                f"<div class='entry-meta'>{plural(len(m['action_items']), 'new action item')}, "
                f"{plural(len(m['key_decisions']), 'decision')}, {plural(len(m['open_questions']), 'open question')}. "
                "Full details are in the Library tab.</div>",
                unsafe_allow_html=True,
            )
            if m.get("fact_check"):
                render_fact_check(m["fact_check"])
            else:
                st.markdown(m["summary"])
            if last["followups"]:
                st.markdown(f"<div class='side-label'>Updated {plural(len(last['followups']), 'task')} "
                            "from earlier meetings</div>", unsafe_allow_html=True)
                for c in last["followups"]:
                    at = stamp(format_timestamp(c["start"])) if c.get("start") is not None else ""
                    st.markdown(
                        f"<div class='task'>{status_badge(c['status'])} {esc(c['task'])}"
                        f"<div class='hist'>From {esc(c['from_title'])}: <mark>{esc(c['evidence'])}</mark> {at}</div></div>",
                        unsafe_allow_html=True,
                    )

# ---- Library ------------------------------------------------------------------------


def summary_view(m: dict, key: str) -> None:
    """Fact-checked summary if available; otherwise the plain summary with a button to check it."""
    report = m.get("fact_check")
    if report:
        render_fact_check(report)
    else:
        st.markdown(m["summary"])
        st.caption("This summary hasn't been checked against the source yet.")
    label = "Check this summary" if not report else "Check again"
    if st.button(label, key=f"fc_{key}", type="secondary" if report else "primary", disabled=not groq_ok):
        with st.spinner("Checking every point against the source…"):
            try:
                factcheck.check_source_summary(WS, m)
            except Exception as err:  # noqa: BLE001
                st.error(f"The check stopped: {err}")
            else:
                st.session_state.pop("last_result", None)
                st.rerun()


with tab_library:
    if not sources:
        st.markdown(
            f"<div class='empty'><h3>Nothing in {esc(PROJECT)} yet</h3>"
            "Meetings and PDFs you add will be listed here with their summaries, tasks and transcripts.</div>",
            unsafe_allow_html=True,
        )
    else:
        index_col, detail_col = st.columns([2, 5], gap="large")
        with index_col:
            ids = [x["source_id"] for x in sources]
            # The selection lives in session state under the radio's key. Passing a changing
            # `index` instead made Streamlit rebuild the widget and jump back to the first entry,
            # which also swallowed clicks (e.g. Delete) made on the selected entry.
            wanted = st.session_state.pop("selected_meeting", None)  # set after adding a meeting/PDF
            if wanted in ids:
                st.session_state.library_choice = wanted
            elif st.session_state.get("library_choice") not in ids:
                st.session_state.library_choice = ids[0]
            with st.container(key="library_index"):
                chosen = st.radio(
                    "Entries",
                    ids,
                    key="library_choice",
                    format_func=lambda i: next(x["title"] for x in sources if x["source_id"] == i),
                    captions=[source_caption(x) for x in sources],
                    label_visibility="collapsed",
                )
        m = vector_store.get_source(WS, chosen)

        with detail_col:
            if m and m.get("source_type") == "pdf":
                st.markdown(
                    f"<div class='entry-kind'>PDF document</div><div class='entry-title'>{esc(m['title'])}</div>"
                    f"<div class='entry-meta'>Dated {nice_date(m['date'])}, "
                    f"{plural(m.get('page_count', 0), 'page')}, {m.get('word_count', 0)} words, "
                    f"file {esc(m.get('origin_ref', ''))}</div>",
                    unsafe_allow_html=True,
                )
                summary_view(m, key=f"pdf_{chosen}")
                st.write("")
                c1, c2, _ = st.columns([2, 2, 3])
                c1.download_button(
                    "Download summary", f"# {m['title']}\n\n{m['summary']}\n",
                    file_name=f"{slugify(m['title'])}-summary.md", mime="text/markdown", width="stretch",
                )
                if c2.button("Delete", width="stretch", key="del_PDF_" + chosen):
                    def _delete(source_id=chosen):
                        vector_store.delete_source(WS, source_id)
                        for k in ('selected_meeting', 'last_docs'):
                            st.session_state.pop(k, None)
                    confirm_delete("PDF", m["title"], _delete)
            elif m:
                done = sum(1 for a in m["action_items"] if a["status"] == "done")
                n_tasks = len(m["action_items"])
                task_note = f"{done} of {plural(n_tasks, 'task')} done" if n_tasks else "no tasks"
                st.markdown(
                    f"<div class='entry-kind'>Meeting</div><div class='entry-title'>{esc(m['title'])}</div>"
                    f"<div class='entry-meta'>{nice_date(m['date'])}, {format_timestamp(m.get('duration', 0))} long, "
                    f"{m.get('word_count', 0)} words, {task_note}</div>",
                    unsafe_allow_html=True,
                )
                n_dq = len(m["key_decisions"]) + len(m["open_questions"])
                t_sum, t_act, t_dec, t_tr = st.tabs(
                    ["Summary", f"Tasks ({len(m['action_items'])})", f"Decisions and questions ({n_dq})", "Transcript"]
                )
                with t_sum:
                    summary_view(m, key=f"meeting_{chosen}")
                with t_act:
                    if not m["action_items"]:
                        st.caption("No tasks came up in this meeting.")
                    for a in m["action_items"]:
                        hist = "".join(
                            f"<div class='hist'>{STATUS_LABEL[u['status']]} in {esc(u['title'])}, {nice_date(u['date'])}: "
                            f"<mark>{esc(u['evidence'])}</mark> "
                            + (stamp(format_timestamp(u["start"])) if u.get("start") is not None else "")
                            + "</div>"
                            for u in a.get("updates", [])
                        )
                        st.markdown(
                            f"<div class='task'>{status_badge(a['status'])} {esc(a['task'])}"
                            f"<div class='who'>{esc(a['owner'])}, "
                            f"{'no deadline' if a['deadline'] == 'Not specified' else 'due ' + esc(a['deadline'])}</div>"
                            f"{hist}</div>",
                            unsafe_allow_html=True,
                        )
                with t_dec:
                    st.markdown("**Decisions**")
                    st.markdown("\n".join(f"- {d}" for d in m["key_decisions"]) or "_None recorded._")
                    st.markdown("**Open questions**")
                    st.markdown("\n".join(f"- {q}" for q in m["open_questions"]) or "_None recorded._")
                with t_tr:
                    st.text_area("Transcript", segments_to_timestamped_text(m["segments"]), height=420,
                                 label_visibility="collapsed")
                st.write("")
                c1, c2, _ = st.columns([2, 2, 3])
                c1.download_button(
                    "Download report", meeting_report(m), file_name=f"{slugify(m['title'])}.md",
                    mime="text/markdown", width="stretch",
                )
                if c2.button("Delete", width="stretch", key="del_meeting_" + chosen):
                    def _delete(source_id=chosen):
                        vector_store.delete_source(WS, source_id)
                        for k in ('selected_meeting', 'last_result'):
                            st.session_state.pop(k, None)
                    confirm_delete("meeting", m["title"], _delete)

# ---- Ask --------------------------------------------------------------------------------

with tab_ask:
    t1, t2, t3 = st.columns([4, 2, 1], vertical_alignment="bottom")
    all_projects = t2.toggle("Search all projects", value=False)
    if not sources and not all_projects:
        st.markdown(
            f"<div class='empty'><h3>Nothing to search in {esc(PROJECT)} yet</h3>"
            "Add a meeting or PDF first, or switch on <b>Search all projects</b>.</div>",
            unsafe_allow_html=True,
        )
    else:
        history_key = f"chat_{WS}_{'*' if all_projects else PROJECT}"
        history = st.session_state.setdefault(history_key, [])
        if all_projects:
            scope = []
            t1.markdown(f"<div class='scope-note'>Searching every project: {esc(', '.join(projects))}</div>",
                        unsafe_allow_html=True)
        else:
            scope = t1.multiselect(
                "Search in", [x["source_id"] for x in sources], format_func=source_label,
                placeholder=f"Everything in {PROJECT}",
            )
        if t3.button("Clear", width="stretch", disabled=not history):
            st.session_state[history_key] = []
            st.rerun()

        if not history:
            st.markdown("<div class='side-label'>Try asking</div>", unsafe_allow_html=True)
            suggestions = [
                "What did we decide, and when?",
                "What changed between the last two meetings?",
                "Which deadlines were mentioned?",
            ]
            for col, s in zip(st.columns(3), suggestions):
                if col.button(s, width="stretch", key=f"sugg_{s}"):
                    st.session_state.pending_question = s

        for i, msg in enumerate(history):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("check"):
                    with st.container(border=True):
                        render_fact_check(msg["check"], show_title=True)
                elif msg["role"] == "assistant" and msg.get("context"):
                    if st.button("Check this answer", key=f"check_{history_key}_{i}", disabled=not groq_ok):
                        with st.spinner("Checking each statement against the excerpts…"):
                            try:
                                msg["check"] = factcheck.check_answer(WS, msg["content"], msg["context"])
                            except Exception as err:  # noqa: BLE001
                                st.error(f"The check stopped: {err}")
                            else:
                                st.rerun()
                if msg.get("sources"):
                    with st.expander(f"{plural(len(msg['sources']), 'source')}"):
                        for s in msg["sources"]:
                            where = f"p. {s['page']}" if s.get("page") else format_timestamp(s.get("start", 0))
                            proj = f"{esc(s.get('project', ''))}, " if all_projects else ""
                            st.markdown(
                                f"<div class='src'><b>[{s['ref']}] {esc(s.get('title', ''))}</b> "
                                f"{proj}{nice_date(s.get('date', ''))} {stamp(where)}<br>{esc(s['text'][:300])}…</div>",
                                unsafe_allow_html=True,
                            )

        typed = st.chat_input(f"Ask about {'all projects' if all_projects else PROJECT}…")
        question = typed or st.session_state.pop("pending_question", None)
        if question:
            with st.spinner("Searching…"):
                try:
                    result = rag_engine.answer_question(
                        WS, question, source_ids=scope or None, history=history,
                        project=None if all_projects else PROJECT,
                    )
                except Exception as err:  # noqa: BLE001
                    result = {"answer": f"The search stopped: {err}", "sources": []}
            history.append({"role": "user", "content": question})
            history.append({
                "role": "assistant", "content": result["answer"],
                "sources": result["sources"], "context": result.get("context", []),
            })
            st.rerun()

# ---- Follow-ups ------------------------------------------------------------------------

with tab_follow:
    if not all_items:
        st.markdown(
            f"<div class='empty'><h3>No tasks in {esc(PROJECT)} yet</h3>"
            "Tasks mentioned in your meetings appear here. When a later meeting says one is finished "
            "or under way, its status updates by itself, with the quote that shows it.</div>",
            unsafe_allow_html=True,
        )
    else:
        counts = {s: sum(1 for _, a in all_items if a["status"] == s) for s in STATUS_LABEL}
        st.markdown(
            "<div class='task-counts'>" + "".join(
                f"<div><b>{counts[s]}</b>{STATUS_LABEL[s].lower()}</div>" for s in STATUS_LABEL
            ) + "</div>",
            unsafe_allow_html=True,
        )
        show_done = st.toggle("Show finished tasks", value=False)
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
                    f"{nice_date(last_update['date'])}: {last_update['evidence']}" if last_update else ""
                ),
            })
            keys.append((m["source_id"], a["id"], a["status"]))

        if not rows:
            st.success("Every task in this project is done.")
        else:
            st.caption("Change a status here if the automatic update got it wrong.")
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
