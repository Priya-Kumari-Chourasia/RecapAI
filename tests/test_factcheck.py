from core import factcheck, rag_engine, vector_store
from tests.test_memory import MEETING_1, MEETING_2, add_to, segs


def test_split_claims_from_bullets_and_prose():
    summary = "- **Use Bonsai** as the baseline [1].\n* Priya finishes the review by Friday\n1. Ok\n- Ablation on label embeddings [2, 3]"
    assert factcheck.split_claims(summary) == [
        "Use Bonsai as the baseline.", "Priya finishes the review by Friday", "Ablation on label embeddings",
    ]
    prose = "You agreed to use Bonsai [1]. The paper reports 82.8 P@1 [2]! Yes."
    assert factcheck.split_claims(prose) == ["You agreed to use Bonsai.", "The paper reports 82.8 P@1!"]


def test_summary_is_checked_against_its_own_meeting_only(fake_llm, memory):
    ws = vector_store.workspace_id("me")
    other = add_to(ws, "Thesis", MEETING_2, "2026-10-08")["meeting"]
    meeting = add_to(ws, "Thesis", MEETING_1, "2026-10-01")["meeting"]
    meeting["summary"] = (
        "- We chose Bonsai as the baseline for the label tree\n"
        "- Priya must finish the literature review by Monday\n"
        "- The team will switch to a GPU cluster next month"
    )
    fake_llm.factcheck_reply = {"verdicts": [
        {"claim": 1, "status": "supported", "evidence": "use Bonsai as the baseline", "passage": 1},
        {"claim": 2, "status": "partial", "evidence": "finish the literature review by Friday", "passage": 4,
         "note": "deadline is Friday, not Monday"},
        {"claim": 3, "status": "unsupported", "note": "not mentioned"},
    ]}
    report = factcheck.check_source_summary(ws, meeting)

    prompt = fake_llm.calls[-1][1]
    assert "Good news, the literature review is finished" not in prompt  # other meeting never used as evidence
    assert [c["status"] for c in report["claims"]] == ["supported", "partial", "unsupported"]
    assert report["score"] == {"total": 3, "supported": 1, "partial": 1, "unsupported": 1, "faithfulness": 0.5}
    assert report["claims"][0]["start"] == 0 and report["claims"][0]["title"] == meeting["title"]
    assert report["claims"][1]["note"] == "deadline is Friday, not Monday"
    assert report["claims"][2]["evidence"] == "" and report["claims"][2]["start"] is None

    stored = vector_store.get_source(ws, meeting["source_id"])["fact_check"]
    assert stored["score"]["faithfulness"] == 0.5


def test_skipped_claim_is_not_counted_as_verified(fake_llm, memory):
    ws = vector_store.workspace_id("me")
    meeting = add_to(ws, "Thesis", MEETING_1, "2026-10-01")["meeting"]
    meeting["summary"] = "- We chose Bonsai as the baseline\n- Run the ablation study soon"
    fake_llm.factcheck_reply = {"verdicts": [{"claim": 1, "status": "supported", "evidence": "x", "passage": 1}]}
    report = factcheck.check_source_summary(ws, meeting, save=False)
    assert report["claims"][1]["status"] == "unsupported" and report["claims"][1]["note"] == "not checked"


def test_new_meeting_is_fact_checked_automatically_and_failure_does_not_lose_it(fake_llm, memory):
    ws = vector_store.workspace_id("me")
    meeting = add_to(ws, "Thesis", MEETING_1, "2026-10-01")["meeting"]
    assert meeting["fact_check"]["score"]["total"] == 1  # fake summary has one bullet
    assert vector_store.get_source(ws, meeting["source_id"])["fact_check"]["score"]["supported"] == 1

    fake_llm.factcheck_reply = "not json at all"  # judge returns garbage twice -> check fails
    second = add_to(ws, "Thesis", MEETING_2, "2026-10-08")["meeting"]
    assert "fact_check" not in second
    assert vector_store.get_source(ws, second["source_id"]) is not None  # meeting still saved


def test_chat_answer_is_checked_against_the_excerpts_it_saw(fake_llm, memory):
    ws = vector_store.workspace_id("me")
    add_to(ws, "Thesis", MEETING_1, "2026-10-01")
    result = rag_engine.answer_question(ws, "What baseline did we choose?", project="Thesis")
    assert result["context"]

    fake_llm.factcheck_reply = {"verdicts": [{"claim": 1, "status": "supported", "evidence": "use Bonsai", "passage": 1}]}
    report = factcheck.check_answer(ws, result["answer"], result["context"] + result["context"])
    prompt = fake_llm.calls[-1][1]
    assert prompt.count("PASSAGE 1 ") == 1 and "PASSAGE 2 " not in prompt  # passages listed once, deduped
    assert "CLAIM 1: You agreed to use Bonsai as the baseline." in prompt
    assert report["score"]["faithfulness"] == 1.0

    assert factcheck.check_answer(ws, "I couldn't find this in your meetings or documents.", [])["claims"] == []


def test_evaluation_script_reports_overall_faithfulness(fake_llm, memory, tmp_path):
    import io

    from eval import faithfulness

    ws = vector_store.workspace_id("me")
    add_to(ws, "Thesis", MEETING_1, "2026-10-01")  # auto-checked: 1/1 supported
    m2 = add_to(ws, "Thesis", MEETING_2, "2026-10-08")["meeting"]
    vector_store.update_source(ws, m2["source_id"], {"fact_check": None})  # pretend it was never checked

    fake_llm.factcheck_reply = {"verdicts": [{"claim": 1, "status": "unsupported", "note": "invented"}]}
    rows, claims, overall = faithfulness.evaluate(ws, project="Thesis", log=lambda *_: None)
    assert overall == {"total": 2, "supported": 1, "partial": 0, "unsupported": 1, "faithfulness": 0.5}
    assert {r["title"] for r in rows} == {"Thesis Sync"} and len(rows) == 2

    buf = io.StringIO()
    faithfulness.print_report(rows, overall, claims, out=buf)
    text = buf.getvalue()
    assert "OVERALL" in text and "50.0%" in text and "invented" in text
