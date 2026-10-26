from core import rag_engine, tracker, vector_store
from core.pipeline import build_meeting, save_meeting
from core.transcriber import _shift, format_timestamp
from utils.llm_retry import _retry_after_seconds, is_rate_limit


def segs(*texts, step=30.0):
    return [{"start": i * step, "end": (i + 1) * step, "text": t} for i, t in enumerate(texts)]


MEETING_1 = segs(
    "Today we decided to use Bonsai as the baseline for the label tree.",
    "Priya, please finish the literature review by Friday.",
    "Also run the ablation study on the label embeddings.",
)
MEETING_2 = segs(
    "Good news, the literature review is finished and sent to the professor.",
    "The ablation study is still running on the cluster.",
)


def add(ws, segments, date, title=None):
    record = build_meeting(segments, workspace=ws, engine="groq", meeting_date=date, title=title, progress=lambda *_: None)
    return save_meeting(record, progress=lambda *_: None)


# --- small units -------------------------------------------------------------


def test_timestamps_shift_by_chunk_offset():
    shifted = _shift([{"start": 1.0, "end": 2.5, "text": " hi "}, {"start": 3, "end": 4, "text": "  "}], 600)
    assert shifted == [{"start": 601.0, "end": 602.5, "text": "hi"}]
    assert format_timestamp(754) == "12:34" and format_timestamp(3725) == "1:02:05"


def test_chunking_keeps_time_ranges_and_overlap():
    chunks = vector_store.chunk_segments(segs(*["word " * 60] * 6), target_chars=600)
    assert chunks[0]["start"] == 0 and chunks[-1]["end"] == 180
    assert chunks[1]["start"] == chunks[0]["end"] - 30  # one segment of overlap


def test_rate_limit_detection_and_wait_parsing():
    class Err(Exception):
        status_code = 429

    assert is_rate_limit(Err("Rate limit reached. Please try again in 1m2.5s."))
    assert _retry_after_seconds(Err("Please try again in 1m2.5s.")) == 62.5
    assert not is_rate_limit(ValueError("bad input"))


def test_workspace_id_is_stable_and_case_insensitive():
    assert vector_store.workspace_id(" Tilu-Thesis ") == vector_store.workspace_id("tilu-thesis")
    assert vector_store.workspace_id("a") != vector_store.workspace_id("b")


# --- meeting memory ----------------------------------------------------------


def test_meeting_is_saved_with_structured_insights(fake_llm, memory):
    ws = vector_store.workspace_id("thesis")
    out = add(ws, MEETING_1, "2026-10-01")
    meeting = out["meeting"]
    assert meeting["title"] == "Thesis Sync"
    assert [a["task"] for a in meeting["action_items"]] == ["Finish the literature review", "Run the ablation study"]
    assert all(a["status"] == "open" for a in meeting["action_items"])

    listed = vector_store.list_sources(ws)
    assert len(listed) == 1 and "segments" not in listed[0]
    assert vector_store.get_source(ws, meeting["source_id"])["segments"] == MEETING_1


def test_workspaces_are_isolated(fake_llm, memory):
    a, b = vector_store.workspace_id("alice"), vector_store.workspace_id("bob")
    add(a, MEETING_1, "2026-10-01")
    assert vector_store.list_sources(b) == []
    assert vector_store.search(b, "Bonsai baseline") == []
    assert vector_store.search(a, "Bonsai baseline")


def test_cross_meeting_answer_cites_dated_timestamped_excerpts(fake_llm, memory):
    ws = vector_store.workspace_id("thesis")
    add(ws, MEETING_1, "2026-10-01")
    add(ws, MEETING_2, "2026-10-08")

    result = rag_engine.answer_question(ws, "What baseline did we choose?")
    assert result["answer"].endswith("[1].")
    assert len(result["sources"]) == 1 and result["sources"][0]["ref"] == 1

    system_prompt = fake_llm.calls[-1][0]
    assert "01 Oct 2026" in system_prompt and "08 Oct 2026" in system_prompt  # both meetings retrieved
    assert "Thesis Sync · 01 Oct 2026 · 0:00" in system_prompt  # excerpt labelled with date + time


def test_search_can_be_limited_to_chosen_meetings(fake_llm, memory):
    ws = vector_store.workspace_id("thesis")
    first = add(ws, MEETING_1, "2026-10-01")["meeting"]
    add(ws, MEETING_2, "2026-10-08")
    hits = vector_store.search(ws, "literature review", source_ids=[first["source_id"]])
    assert hits and {h["source_id"] for h in hits} == {first["source_id"]}


def test_followups_update_earlier_action_items(fake_llm, memory):
    ws = vector_store.workspace_id("thesis")
    first = add(ws, MEETING_1, "2026-10-01")["meeting"]
    review, ablation = first["action_items"]

    fake_llm.tracker_reply = {"updates": [
        {"item_id": review["id"], "status": "done", "evidence": "the literature review is finished", "excerpt": 1},
        {"item_id": ablation["id"], "status": "in_progress", "evidence": "still running on the cluster", "excerpt": 3},
    ]}
    second = add(ws, MEETING_2, "2026-10-08")

    statuses = {c["task"]: c["status"] for c in second["followups"]}
    assert statuses == {"Finish the literature review": "done", "Run the ablation study": "in_progress"}

    stored = vector_store.get_source(ws, first["source_id"])["action_items"]
    assert stored[0]["status"] == "done"
    assert stored[0]["updates"][0]["date"] == "2026-10-08"
    assert stored[0]["updates"][0]["start"] is not None  # links to the moment in meeting 2

    # meeting 2 re-mentioned both tasks; they must not be duplicated as new items
    assert vector_store.get_source(ws, second["meeting"]["source_id"])["action_items"] == []
    # done items aren't re-checked by later meetings
    assert [i["task"] for _, i in tracker.open_items(ws, "x", "2026-12-31")] == ["Run the ablation study"]


def test_manual_status_override_and_delete(fake_llm, memory):
    ws = vector_store.workspace_id("thesis")
    meeting = add(ws, MEETING_1, "2026-10-01")["meeting"]
    item = meeting["action_items"][1]
    tracker.set_item_status(ws, meeting["source_id"], item["id"], "done")
    assert vector_store.get_source(ws, meeting["source_id"])["action_items"][1]["status"] == "done"

    vector_store.delete_source(ws, meeting["source_id"])
    assert vector_store.list_sources(ws) == [] and vector_store.search(ws, "Bonsai") == []


def test_youtube_block_is_reported_clearly(monkeypatch):
    import yt_dlp

    from utils import audio_processor

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            raise yt_dlp.utils.DownloadError("ERROR: [youtube] abc: Sign in to confirm you're not a bot.")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    try:
        audio_processor.download_youtube_audio("https://youtu.be/abc")
    except audio_processor.YouTubeBlockedError as err:
        assert "Upload file" in str(err)
    else:
        raise AssertionError("expected YouTubeBlockedError")


# --- projects ------------------------------------------------------------------


def add_to(ws, project, segments, date):
    record = build_meeting(segments, workspace=ws, engine="groq", project=project, meeting_date=date, progress=lambda *_: None)
    return save_meeting(record, progress=lambda *_: None)


def test_projects_are_listed_and_scoped(fake_llm, memory):
    ws = vector_store.workspace_id("me")
    vector_store.create_project(ws, "  BrickMind  ")
    add_to(ws, "Thesis", MEETING_1, "2026-10-01")
    assert vector_store.list_projects(ws) == ["General", "BrickMind", "Thesis"]

    assert len(vector_store.list_sources(ws, project="Thesis")) == 1
    assert vector_store.list_sources(ws, project="BrickMind") == []
    assert len(vector_store.list_sources(ws)) == 1  # registry entries aren't listed as meetings
    assert vector_store.search(ws, "Bonsai baseline", project="BrickMind") == []
    assert vector_store.search(ws, "Bonsai baseline", project="Thesis")


def test_followups_never_cross_projects(fake_llm, memory):
    ws = vector_store.workspace_id("me")
    thesis = add_to(ws, "Thesis", MEETING_1, "2026-10-01")["meeting"]
    fake_llm.tracker_reply = {"updates": [
        {"item_id": a["id"], "status": "done", "evidence": "x", "excerpt": 1} for a in thesis["action_items"]
    ]}
    factory = add_to(ws, "BrickMind", MEETING_2, "2026-10-08")
    assert factory["followups"] == []  # thesis items were not even considered
    assert all(a["status"] == "open" for a in vector_store.get_source(ws, thesis["source_id"])["action_items"])


def test_ask_all_projects_vs_one(fake_llm, memory):
    ws = vector_store.workspace_id("me")
    add_to(ws, "Thesis", MEETING_1, "2026-10-01")
    add_to(ws, "BrickMind", segs("Kiln temperature must stay below 1000 degrees."), "2026-10-02")

    rag_engine.answer_question(ws, "What should we do?", project="BrickMind")
    assert "Kiln" in fake_llm.calls[-1][0] and "Bonsai" not in fake_llm.calls[-1][0]
    rag_engine.answer_question(ws, "What should we do?")
    assert "Kiln" in fake_llm.calls[-1][0] and "Bonsai" in fake_llm.calls[-1][0]
    assert "(project: BrickMind)" in fake_llm.calls[-1][0]


def test_delete_project_removes_its_meetings_only(fake_llm, memory):
    ws = vector_store.workspace_id("me")
    add_to(ws, "Thesis", MEETING_1, "2026-10-01")
    add_to(ws, "BrickMind", MEETING_2, "2026-10-02")
    vector_store.delete_project(ws, "BrickMind")
    assert vector_store.list_projects(ws) == ["General", "Thesis"]
    assert [m["project"] for m in vector_store.list_sources(ws)] == ["Thesis"]


def test_meetings_saved_before_projects_show_under_general(fake_llm, memory):
    ws = vector_store.workspace_id("me")
    record = build_meeting(MEETING_1, workspace=ws, engine="groq", progress=lambda *_: None)
    del record["project"]  # simulate a meeting saved by the previous version
    vector_store.save_source(record, vector_store.chunk_segments(record["segments"]))
    vector_store.get_client().delete_payload(vector_store.CHUNKS, keys=["project"], points=vector_store.models.Filter())
    vector_store.get_client().delete_payload(vector_store.SOURCES, keys=["project"], points=vector_store.models.Filter())

    assert len(vector_store.list_sources(ws, project="General")) == 1
    assert vector_store.search(ws, "Bonsai baseline", project="General")
    assert vector_store.list_sources(ws, project="Thesis") == []


# --- robustness found while checking real runs -----------------------------------


def test_chatty_title_falls_back_to_file_name(fake_llm, memory):
    from core.summarizer import clean_title

    fake_llm.title_reply = "Could you please provide the meeting details so I can craft an appropriate title?"
    ws = vector_store.workspace_id("me")
    record = build_meeting(segs("You're glowing tonight"), workspace=ws, engine="groq", origin="upload",
                           origin_ref="Sapphire.mp3", progress=lambda *_: None)
    assert record["title"] == "Sapphire"

    record = build_meeting(segs("hello"), workspace=ws, engine="groq", origin="recording",
                           origin_ref="recording-1.wav", meeting_date="2026-10-26", progress=lambda *_: None)
    assert record["title"] == "Recording 2026-10-26"

    assert clean_title('"Pre-training vs Fine-tuning"') == "Pre-training vs Fine-tuning"
    assert clean_title("I'm sorry, I can't title this") is None


def test_summary_prompt_never_assumes_a_meeting(fake_llm, memory):
    build_meeting(segs("You're glowing tonight"), workspace="w", engine="groq", progress=lambda *_: None)
    summary_system = next(sys for sys, _ in fake_llm.calls if "final summary" in sys)
    assert "never reply that the text isn't a meeting" in summary_system.lower()


def test_llm_flags_reassigned_task_as_duplicate_but_keeps_different_task(fake_llm, memory):
    ws = vector_store.workspace_id("me")
    fake_llm.extract_reply = {"action_items": [{"task": "Run ablation on dataset A", "owner": "Priya"}]}
    first = add(ws, segs("Priya, run the ablation on dataset A."), "2026-10-01")["meeting"]
    old_id = first["action_items"][0]["id"]

    # meeting 2: the same task re-assigned in other words, plus a genuinely new, similar-sounding task
    fake_llm.extract_reply = {"action_items": [
        {"task": "Priya to continue the dataset A ablation", "owner": "Priya"},
        {"task": "Run ablation on dataset B", "owner": "Priya"},
    ]}
    fake_llm.tracker_reply = {
        "updates": [{"item_id": old_id, "status": "in_progress", "evidence": "keep going with dataset A", "excerpt": 1}],
        "duplicates": [{"new_item": 1, "item_id": old_id}],
    }
    second = add(ws, segs("Keep going with dataset A. Also run it on dataset B."), "2026-10-08")

    kept = [a["task"] for a in vector_store.get_source(ws, second["meeting"]["source_id"])["action_items"]]
    assert kept == ["Run ablation on dataset B"]  # re-assignment dropped, the new dataset-B task kept
    tracker_prompt = next(h for sys_, h in reversed(fake_llm.calls) if "track action items" in sys_)
    assert "NEW ITEM 1: Priya to continue the dataset A ablation" in tracker_prompt


def test_summary_prompt_forbids_padding_short_recordings(fake_llm, memory):
    # Real run: a one-sentence recording got an invented 4th bullet because the prompt asked for 4-8.
    build_meeting(segs("Please finish the literature review by Friday."), workspace="w", engine="groq",
                  progress=lambda *_: None)
    system = next(sys_ for sys_, _ in fake_llm.calls if "final summary" in sys_)
    assert "4-8" not in system and "never add background" in system
