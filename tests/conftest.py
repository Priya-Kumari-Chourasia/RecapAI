"""Offline test fixtures: an in-memory Qdrant, a tiny hashing embedder and a scripted fake LLM.
No API keys or network needed: `pytest -q`."""
import hashlib
import json
import math
import os
import re
import sys

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import llm, vector_store  # noqa: E402


class HashEmbedder:
    """Bag-of-words hashed into 256 dims: crude, but similar text -> similar vectors."""

    dim = 256

    def _vec(self, text):
        v = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    """Routes on the system prompt and returns canned answers; records every call."""

    def __init__(self):
        self.calls = []
        self.tracker_reply = None
        self.title_reply = '"Thesis Sync"'
        self.extract_reply = None
        self.factcheck_reply = None  # None: mark every claim supported by passage 1

    def __call__(self, prompt_value):
        system, human = (m.content for m in prompt_value.to_messages())
        self.calls.append((system, human))
        if "strict fact-checker" in system:
            if self.factcheck_reply is not None:
                return AIMessage(content=json.dumps(self.factcheck_reply))
            n_claims = len(re.findall(r"(?m)^CLAIM \d+:", human))
            return AIMessage(content=json.dumps({"verdicts": [
                {"claim": i, "status": "supported", "evidence": "quote", "passage": 1} for i in range(1, n_claims + 1)
            ]}))
        if "track action items" in system:
            return AIMessage(content=json.dumps(self.tracker_reply or {"updates": []}))
        if "meeting analyst" in system and self.extract_reply is not None:
            return AIMessage(content=json.dumps(self.extract_reply))
        if "meeting analyst" in system:
            items = []
            if "literature review" in human.lower():
                items.append({"task": "Finish the literature review", "owner": "Priya", "deadline": "Friday"})
            if "ablation" in human.lower():
                items.append({"task": "Run the ablation study", "owner": "Priya", "deadline": "Not specified"})
            return AIMessage(content="```json\n" + json.dumps(
                {"action_items": items, "key_decisions": ["Use Bonsai as the baseline"], "open_questions": []}
            ) + "\n```")
        if "research-meeting assistant" in system:
            return AIMessage(content="You agreed to use Bonsai as the baseline [1].")
        if "title" in system:
            return AIMessage(content=self.title_reply)
        return AIMessage(content="- Discussed the thesis plan")


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "get_llm", lambda: RunnableLambda(fake))
    return fake


@pytest.fixture
def memory():
    vector_store.set_embedder(HashEmbedder())
    vector_store.set_client(QdrantClient(":memory:"))
    yield vector_store


def make_pdf(path, pages, title=None):
    """Write a minimal text PDF by hand (no extra dependency). `pages` is a list of strings;
    an empty string makes a page with no text, like a scanned image page."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 11 Tf 40 760 Td ({safe}) Tj ET" if text else ""
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        content_id = len(objects)
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
                       f"/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    info_id = None
    if title:
        objects.append(f"<< /Title ({title}) >>")
        info_id = len(objects)
    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    trailer = f"<< /Size {len(objects) + 1} /Root 1 0 R" + (f" /Info {info_id} 0 R" if info_id else "") + " >>"
    out += f"trailer\n{trailer}\nstartxref\n{xref}\n%%EOF\n".encode()
    with open(path, "wb") as f:
        f.write(out)
    return str(path)
