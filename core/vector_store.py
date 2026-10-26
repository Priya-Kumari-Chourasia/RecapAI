"""Persistent meeting memory on Qdrant.

Two collections:
- recap_chunks  — retrievable pieces of transcripts (with start/end seconds) and PDFs (with page)
- recap_sources — one record per meeting / document: title, date, summary, action items, ...

Uses Qdrant Cloud when QDRANT_URL is set (survives Streamlit Cloud restarts),
otherwise an on-disk local Qdrant in ./qdrant_data — same code either way.
Every point carries a `workspace` id, and every query filters on it, so
different workspaces never see each other's meetings.
"""
import hashlib
import os
import uuid
import warnings

from qdrant_client import QdrantClient, models

CHUNKS = "recap_chunks"
SOURCES = "recap_sources"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "qdrant_data")
TARGET_CHUNK_CHARS = 700
DEFAULT_PROJECT = "General"


# --- Embeddings ---------------------------------------------------------------


class FastEmbedder:
    """ONNX embeddings via fastembed — no PyTorch, so it fits on Streamlit Cloud."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name)
        self.dim = len(next(iter(self.model.embed(["dimension probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


_embedder = None
_client = None
_ready: set[str] = set()


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = FastEmbedder()
    return _embedder


def set_embedder(embedder) -> None:
    """Tests inject a small offline embedder here."""
    global _embedder
    _embedder = embedder
    _ready.clear()


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL")
        if url:
            _client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"), timeout=30)
        else:
            _client = QdrantClient(path=LOCAL_PATH)
    return _client


def set_client(client: QdrantClient) -> None:
    global _client
    _client = client
    _ready.clear()


def storage_label() -> str:
    return "Qdrant Cloud" if os.getenv("QDRANT_URL") else "Local Qdrant (this machine)"


def _ensure(name: str) -> None:
    if name in _ready:
        return
    client = get_client()
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config=models.VectorParams(size=get_embedder().dim, distance=models.Distance.COSINE),
        )
    # Qdrant Cloud needs payload indexes to filter efficiently; local mode ignores them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for field in ("workspace", "project", "source_id", "source_type"):
            try:
                client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)
            except Exception:  # noqa: BLE001 - already exists
                pass
    _ready.add(name)


# --- Helpers ------------------------------------------------------------------


def workspace_id(name: str) -> str:
    """Hash the workspace name so it works like a password: knowing the hash lists nothing."""
    return hashlib.sha256(name.strip().lower().encode()).hexdigest()[:24]


def new_source_id() -> str:
    return uuid.uuid4().hex


def _match(**fields) -> list[models.FieldCondition]:
    return [models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in fields.items()]


def _scope(workspace: str, project: str | None = None) -> list:
    """Filter conditions for a workspace, optionally one project. Meetings saved before
    projects existed have no `project` field; they belong to the default project."""
    must = _match(workspace=workspace)
    if project == DEFAULT_PROJECT:
        must.append(models.Filter(should=[
            *_match(project=DEFAULT_PROJECT),
            models.IsEmptyCondition(is_empty=models.PayloadField(key="project")),
        ]))
    elif project:
        must += _match(project=project)
    return must


def chunk_segments(segments: list[dict], target_chars: int = TARGET_CHUNK_CHARS) -> list[dict]:
    """Group consecutive transcript segments into ~700-char chunks that keep start/end times.
    The last segment of each chunk is repeated at the start of the next for context overlap."""
    chunks, current = [], []
    for seg in segments:
        current.append(seg)
        if sum(len(s["text"]) + 1 for s in current) >= target_chars:
            chunks.append(current)
            current = [current[-1]] if len(current) > 1 else []
    if current and (not chunks or current != chunks[-1][-1:]):
        chunks.append(current)
    return [
        {"text": " ".join(s["text"] for s in group), "start": group[0]["start"], "end": group[-1]["end"]}
        for group in chunks
    ]


# --- Writes -------------------------------------------------------------------


def save_source(record: dict, chunks: list[dict]) -> None:
    """Store one meeting/document record plus its retrievable chunks."""
    _ensure(SOURCES)
    _ensure(CHUNKS)
    client, embedder = get_client(), get_embedder()
    record.setdefault("project", DEFAULT_PROJECT)
    shared = {k: record[k] for k in ("workspace", "project", "source_id", "source_type", "title", "date")}

    batch = 64
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        vectors = embedder.embed_documents([c["text"] for c in part])
        client.upsert(
            CHUNKS,
            points=[
                models.PointStruct(id=uuid.uuid4().hex, vector=v, payload={**shared, **c, "chunk_index": i + j})
                for j, (c, v) in enumerate(zip(part, vectors))
            ],
        )

    summary_vector = embedder.embed_documents([f"{record['title']}\n{record.get('summary', '')}"])[0]
    client.upsert(SOURCES, points=[models.PointStruct(id=record["source_id"], vector=summary_vector, payload=record)])


def update_source(workspace: str, source_id: str, fields: dict) -> None:
    _ensure(SOURCES)
    get_client().set_payload(
        SOURCES,
        payload=fields,
        points=models.Filter(must=_match(workspace=workspace, source_id=source_id)),
    )


def delete_source(workspace: str, source_id: str) -> None:
    for name in (SOURCES, CHUNKS):
        _ensure(name)
        get_client().delete(
            name,
            points_selector=models.FilterSelector(filter=models.Filter(must=_match(workspace=workspace, source_id=source_id))),
        )


# --- Reads --------------------------------------------------------------------


def list_sources(workspace: str, project: str | None = None, include_segments: bool = False) -> list[dict]:
    """Meetings/documents in a workspace (optionally one project), newest first."""
    _ensure(SOURCES)
    payload = True if include_segments else models.PayloadSelectorExclude(exclude=["segments", "pages"])
    must = _scope(workspace, project)
    must_not = _match(source_type="project")  # project registry entries aren't meetings
    out, offset = [], None
    while True:
        points, offset = get_client().scroll(
            SOURCES,
            scroll_filter=models.Filter(must=must, must_not=must_not),
            with_payload=payload,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        out.extend(p.payload for p in points)
        if offset is None:
            break
    return sorted(out, key=lambda r: (r.get("date", ""), r.get("created_at", "")), reverse=True)


def get_source(workspace: str, source_id: str) -> dict | None:
    _ensure(SOURCES)
    points, _ = get_client().scroll(
        SOURCES,
        scroll_filter=models.Filter(must=_match(workspace=workspace, source_id=source_id)),
        with_payload=True,
        limit=1,
    )
    return points[0].payload if points else None


def search(
    workspace: str, query: str, k: int = 6, source_ids: list[str] | None = None, project: str | None = None
) -> list[dict]:
    """Semantic search over chunks in one workspace, optionally limited to a project or some sources."""
    _ensure(CHUNKS)
    must = _scope(workspace, project)
    if source_ids is not None:
        if not source_ids:
            return []
        must.append(models.FieldCondition(key="source_id", match=models.MatchAny(any=list(source_ids))))
    result = get_client().query_points(
        CHUNKS,
        query=get_embedder().embed_query(query),
        query_filter=models.Filter(must=must),
        limit=k,
        with_payload=True,
    )
    return [{**p.payload, "score": p.score} for p in result.points]


# --- Projects -----------------------------------------------------------------


def _project_point_id(workspace: str, name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"recap-project/{workspace}/{name.lower()}"))


def create_project(workspace: str, name: str) -> str:
    """Register a project so it shows up even before it has any meetings."""
    name = " ".join(name.split())
    _ensure(SOURCES)
    get_client().upsert(
        SOURCES,
        points=[models.PointStruct(
            id=_project_point_id(workspace, name),
            vector=get_embedder().embed_documents([name])[0],
            payload={"workspace": workspace, "source_type": "project", "project": name},
        )],
    )
    return name


def list_projects(workspace: str) -> list[str]:
    """Registered projects plus any project that has meetings; 'General' always exists."""
    _ensure(SOURCES)
    names, offset = set(), None
    while True:
        points, offset = get_client().scroll(
            SOURCES,
            scroll_filter=models.Filter(must=_match(workspace=workspace)),
            with_payload=["project"],
            limit=256,
            offset=offset,
        )
        names.update(p.payload.get("project") or DEFAULT_PROJECT for p in points)
        if offset is None:
            break
    return [DEFAULT_PROJECT] + sorted(n for n in names if n != DEFAULT_PROJECT)


def delete_project(workspace: str, name: str) -> None:
    """Delete a project and every meeting/document in it."""
    for name_ in (SOURCES, CHUNKS):
        _ensure(name_)
        get_client().delete(
            name_,
            points_selector=models.FilterSelector(filter=models.Filter(must=_match(workspace=workspace, project=name))),
        )
