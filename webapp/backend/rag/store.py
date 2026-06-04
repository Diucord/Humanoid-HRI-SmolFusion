"""RAG 벡터 스토어. 페르소나별 컬렉션 분리.

임베딩: bge-m3 (멀티링구얼, 한/영/일 강함).
벡터DB: ChromaDB (인메모리).
"""
import os
import uuid
import threading
from typing import List
from pypdf import PdfReader

from core import settings

_stores = {}
_lock = threading.Lock()

# bge-m3 임베딩 함수 (전 페르소나 공유, 지연 로딩)
_embed_fn = None


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        from chromadb.utils import embedding_functions
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBED_MODEL,
            device=settings.resolve_device(),
        )
    return _embed_fn


class RAGStore:
    def __init__(self, persona_id: str):
        import chromadb
        self.persona_id = persona_id
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=f"persona_{persona_id}",
            embedding_function=_get_embed_fn(),
        )

    def add_text(self, text: str):
        chunks = _chunk(text, settings.RAG_CHUNK_SIZE, settings.RAG_CHUNK_OVERLAP)
        if not chunks:
            return 0
        ids = [str(uuid.uuid4()) for _ in chunks]
        self._collection.add(documents=chunks, ids=ids)
        return len(chunks)

    def add_file(self, filename: str, data: bytes) -> int:
        ext = os.path.splitext(filename)[-1].lower()
        if ext == ".pdf":
            import io
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = data.decode("utf-8", errors="ignore")
        return self.add_text(text)

    def query(self, question: str, top_k: int = 3) -> str:
        n = self.count()
        if n == 0:
            return ""
        res = self._collection.query(query_texts=[question], n_results=min(top_k, n))
        docs = res.get("documents", [[]])[0]
        return "\n---\n".join(docs)

    def count(self) -> int:
        return self._collection.count()

    def clear(self):
        self._client.delete_collection(f"persona_{self.persona_id}")
        self._collection = self._client.get_or_create_collection(
            name=f"persona_{self.persona_id}",
            embedding_function=_get_embed_fn(),
        )


def get_store(persona_id: str) -> RAGStore:
    with _lock:
        if persona_id not in _stores:
            _stores[persona_id] = RAGStore(persona_id)
        return _stores[persona_id]


def _chunk(text: str, size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return [c for c in chunks if c.strip()]
