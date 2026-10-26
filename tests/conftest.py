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

    def __call__(self, prompt_value):
        system, human = (m.content for m in prompt_value.to_messages())
        self.calls.append((system, human))
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
