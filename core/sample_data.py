"""A ready-made demo project, so a first-time visitor sees Recap AI working without uploading anything.

Everything here is written by hand (no LLM calls): two short thesis meetings a week
apart, with their summaries, tasks, follow-up updates and fact-checks. Seeding only
needs embeddings, so it is fast and free. All names are fictional.
"""
from core import vector_store

DEMO_KEY = "demo"
DEMO_PROJECT = "Sample: thesis meetings"
# Bump when the sample content changes; older copies already stored in Qdrant are replaced.
DEMO_VERSION = 2


def _segs(lines):
    return [{"start": float(s), "end": float(e), "text": t} for s, e, t in lines]


MEETING_1 = {
    "title": "Choosing a baseline for the label tree",
    "date": "2026-10-01",
    "duration": 754.0,
    "segments": _segs([
        (0, 14, "Okay, let's start. Last week you compared three tree-based methods for extreme multi-label classification."),
        (14, 31, "Yes. Bonsai, Parabel and a flat one-vs-rest baseline, on EURLex-4K and Wiki10-31K."),
        (31, 52, "Bonsai was the strongest on both datasets, about two points better in precision at one than Parabel."),
        (52, 70, "Then we use Bonsai as the main baseline. It is also the method your label representation idea builds on."),
        (70, 96, "Agreed. I will keep Parabel only as a secondary comparison in the results table."),
        (96, 128, "For the related work chapter, please finish the literature review by Friday, so I can read it over the weekend."),
        (128, 160, "I also want an ablation on the label embeddings. Compare the TF-IDF label features against the sentence-embedding features."),
        (160, 190, "Should the ablation use the full Wiki10 set? It takes almost a day on my laptop."),
        (190, 221, "Use the lab cluster for that. Ask Arjun for an account, he set it up last month."),
        (221, 260, "One open question is the branching factor. The paper uses one hundred, but with our label features a smaller tree might be better."),
        (260, 300, "Let's not decide that today. Try sixteen, thirty two and one hundred and we look at it next week."),
        (300, 330, "Okay. So by next meeting: the literature review, the cluster account, and the first ablation runs."),
    ]),
    "summary": (
        "- Bonsai was chosen as the main baseline; Parabel stays as a secondary comparison\n"
        "- Bonsai beat Parabel by about two points in precision at 1 on EURLex-4K and Wiki10-31K\n"
        "- Meera will finish the literature review by Friday\n"
        "- An ablation will compare TF-IDF label features with sentence-embedding features\n"
        "- The branching factor is still open; values 16, 32 and 100 will be tested"
    ),
    "action_items": [
        {"id": "demo-a1", "task": "Finish the literature review", "owner": "Meera", "deadline": "Friday",
         "status": "done", "updates": []},
        {"id": "demo-a2", "task": "Run the label-embedding ablation (TF-IDF vs sentence embeddings)", "owner": "Meera",
         "deadline": "Not specified", "status": "in_progress", "updates": []},
        {"id": "demo-a3", "task": "Get a lab cluster account from Arjun", "owner": "Meera", "deadline": "Not specified",
         "status": "done", "updates": []},
    ],
    "key_decisions": ["Use Bonsai as the main baseline", "Keep Parabel only as a secondary comparison"],
    "open_questions": ["Which branching factor works best with the new label features (16, 32 or 100)?"],
}

MEETING_2 = {
    "title": "Ablation results and branching factor",
    "date": "2026-10-08",
    "duration": 612.0,
    "segments": _segs([
        (0, 18, "So, how did the week go? Did you send me the literature review?"),
        (18, 40, "Yes, the literature review is finished, I sent it on Friday evening."),
        (40, 66, "I read it, it is good. Small comments on the section about label trees, I will mark them in the document."),
        (66, 95, "I also got the cluster account from Arjun on Monday, so the long runs are going there now."),
        (95, 130, "The ablation is still running on the cluster. TF-IDF is done, the sentence-embedding run finishes tomorrow."),
        (130, 165, "For the branching factor, thirty two gave the best precision on EURLex, slightly better than one hundred."),
        (165, 200, "Interesting. Then use thirty two for the main experiments, but report one hundred as well for comparison."),
        (200, 240, "For next week, write a short note with the ablation table, and start the method chapter."),
    ]),
    "summary": (
        "- The literature review is finished and was sent on Friday\n"
        "- The cluster account is set up and long runs now go there\n"
        "- The ablation is still running; the TF-IDF run is done\n"
        "- A branching factor of 32 gave the best precision on EURLex and will be used for the main experiments\n"
        "- Meera will write a note with the ablation table and start the method chapter"
    ),
    "action_items": [
        {"id": "demo-b1", "task": "Write a short note with the ablation table", "owner": "Meera",
         "deadline": "Next meeting", "status": "open", "updates": []},
        {"id": "demo-b2", "task": "Start the method chapter", "owner": "Meera", "deadline": "Not specified",
         "status": "open", "updates": []},
    ],
    "key_decisions": ["Use a branching factor of 32 for the main experiments, and report 100 for comparison"],
    "open_questions": [],
}

# Follow-up updates written onto meeting 1's tasks by meeting 2 (what the tracker would produce).
FOLLOWUPS = {
    "demo-a1": ("done", "the literature review is finished, I sent it on Friday evening", 18.0),
    "demo-a2": ("in_progress", "The ablation is still running on the cluster", 95.0),
    "demo-a3": ("done", "I also got the cluster account from Arjun on Monday", 66.0),
}


def _fact_check(meeting, verdicts):
    """verdicts: list of (status, evidence, start, note) in summary-bullet order."""
    from core.factcheck import score, split_claims

    claims = []
    for claim, (status, evidence, start, note) in zip(split_claims(meeting["summary"]), verdicts):
        claims.append({"claim": claim, "status": status, "evidence": evidence, "note": note,
                       "title": meeting["title"], "page": None, "start": start})
    return {"claims": claims, "score": score(claims), "checked_at": "2026-10-08T12:00:00+00:00"}


FACT_CHECKS = {
    "m1": [
        ("supported", "we use Bonsai as the main baseline", 52.0, ""),
        ("partial", "about two points better in precision at one than Parabel", 31.0,
         "the two-point gap was stated overall, not separately per dataset"),
        ("supported", "please finish the literature review by Friday", 96.0, ""),
        ("supported", "Compare the TF-IDF label features against the sentence-embedding features", 128.0, ""),
        ("supported", "Try sixteen, thirty two and one hundred", 260.0, ""),
    ],
    "m2": [
        ("supported", "the literature review is finished, I sent it on Friday evening", 18.0, ""),
        ("supported", "the long runs are going there now", 66.0, ""),
        ("supported", "TF-IDF is done, the sentence-embedding run finishes tomorrow", 95.0, ""),
        ("supported", "use thirty two for the main experiments", 165.0, ""),
        ("partial", "write a short note with the ablation table, and start the method chapter", 200.0,
         "no owner was named in the meeting"),
    ],
}


def ensure_demo_project(workspace: str) -> str:
    """Create the sample project in `workspace` if it isn't there, was deleted, or is an older
    version of the sample. Returns its name."""
    existing = vector_store.list_sources(workspace, project=DEMO_PROJECT)
    if len(existing) == 2 and all(m.get("sample_version") == DEMO_VERSION for m in existing):
        return DEMO_PROJECT
    if existing:
        vector_store.delete_project(workspace, DEMO_PROJECT)
    vector_store.create_project(workspace, DEMO_PROJECT)
    ids = {}
    for key, meeting, checks in (("m1", MEETING_1, FACT_CHECKS["m1"]), ("m2", MEETING_2, FACT_CHECKS["m2"])):
        record = {
            **{k: v for k, v in meeting.items()},
            "action_items": [dict(a, updates=list(a["updates"])) for a in meeting["action_items"]],
            "workspace": workspace,
            "project": DEMO_PROJECT,
            "source_id": vector_store.new_source_id(),
            "source_type": "meeting",
            "created_at": f"{meeting['date']}T10:00:00+00:00",
            "origin": "sample",
            "origin_ref": "sample meeting",
            "engine": "groq",
            "word_count": sum(len(s["text"].split()) for s in meeting["segments"]),
            "sample": True,
            "sample_version": DEMO_VERSION,
        }
        record["fact_check"] = _fact_check(meeting, checks)
        ids[key] = record
    for item in ids["m1"]["action_items"]:
        if item["id"] in FOLLOWUPS:
            status, evidence, start = FOLLOWUPS[item["id"]]
            item["updates"] = [{"source_id": ids["m2"]["source_id"], "title": MEETING_2["title"],
                                "date": MEETING_2["date"], "status": status, "evidence": evidence, "start": start}]
    for record in ids.values():
        vector_store.save_source(record, vector_store.chunk_segments(record["segments"]))
    return DEMO_PROJECT
