"""RAG 벡터 스토어. 페르소나별 컬렉션 분리.

Hybrid Retrieval:
  - Dense  : bge-m3 임베딩 + ChromaDB (의미 유사도)
  - Sparse : BM25 (키워드 정확 매칭)
  - Fusion : RRF(Reciprocal Rank Fusion)로 두 랭킹 병합

Dense 단독은 고유명사·모델명·수치 같은 정확 키워드에 약하고,
BM25 단독은 표현이 다른 동의어를 놓친다. 두 랭킹을 RRF로 합쳐
서로의 약점을 보완한다.

한국어 BM25 주의:
  공백 split은 조사 때문에 "로봇은/로봇이/로봇의"가 전부 다른 토큰이 되어
  매칭률이 급락한다. 형태소 분석기(konlpy 등) 추가 의존성 없이 처리하기 위해
  문자 n-gram(bigram) + 어절 원형을 함께 토큰으로 사용한다.
"""
import os
import re
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


_TOKEN_RE = re.compile(r"[0-9A-Za-z]+|[가-힣]+")


def _tokenize(text: str) -> List[str]:
    """한국어 대응 토크나이저.

    영문/숫자는 단어 단위로, 한글은 어절 + 문자 bigram으로 분해한다.
    bigram이 조사 변형("로봇은" vs "로봇이")을 흡수해 매칭률을 살린다.
    """
    tokens = []
    for word in _TOKEN_RE.findall(text.lower()):
        tokens.append(word)
        # 한글 어절은 bigram으로도 분해 (2글자 이하는 이미 원형으로 충분)
        if len(word) > 2 and re.match(r"^[가-힣]+$", word):
            tokens.extend(word[i:i + 2] for i in range(len(word) - 1))
    return tokens


class RAGStore:
    def __init__(self, persona_id: str):
        import chromadb
        self.persona_id = persona_id
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=f"persona_{persona_id}",
            embedding_function=_get_embed_fn(),
        )
        # BM25용 원문 청크 보관 (Chroma와 인덱스 순서 동기 유지)
        self._chunks: List[str] = []
        self._bm25 = None

    def _rebuild_bm25(self):
        """청크가 바뀔 때마다 BM25 인덱스 재구성.

        문서 업로드는 드물고 청크 수도 수백 단위라 전체 재구성이 저렴하다.
        """
        if not self._chunks:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi([_tokenize(c) for c in self._chunks])

    def add_text(self, text: str):
        chunks = _chunk(text, settings.RAG_CHUNK_SIZE, settings.RAG_CHUNK_OVERLAP)
        if not chunks:
            return 0
        ids = [str(uuid.uuid4()) for _ in chunks]
        self._collection.add(documents=chunks, ids=ids)
        self._chunks.extend(chunks)
        self._rebuild_bm25()
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

        # --- Dense: 의미 유사도 랭킹 ---
        # 각 검색기가 top_k보다 넓게 후보를 내야 융합에서 고를 여지가 생긴다.
        pool = min(max(top_k * 2, top_k), n)
        res = self._collection.query(query_texts=[question], n_results=pool)
        dense_docs = res.get("documents", [[]])[0]

        # --- Sparse: BM25 키워드 랭킹 ---
        sparse_docs = []
        if self._bm25 is not None:
            import numpy as np
            scores = self._bm25.get_scores(_tokenize(question))
            # 점수 0(= 매칭 토큰 전무)인 문서는 노이즈이므로 융합에서 제외
            ranked = np.argsort(scores)[::-1][:pool]
            sparse_docs = [self._chunks[i] for i in ranked if scores[i] > 0]

        # --- Fusion: RRF ---
        docs = _rrf_fuse(dense_docs, sparse_docs, top_k)
        return "\n---\n".join(docs)

    def count(self) -> int:
        return self._collection.count()

    def clear(self):
        self._client.delete_collection(f"persona_{self.persona_id}")
        self._collection = self._client.get_or_create_collection(
            name=f"persona_{self.persona_id}",
            embedding_function=_get_embed_fn(),
        )
        self._chunks = []
        self._bm25 = None


def _rrf_fuse(dense: List[str], sparse: List[str], top_k: int, k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion.

    score(d) = Σ 1/(k + rank_i(d))

    Dense의 거리와 BM25의 점수는 스케일이 전혀 다르므로 직접 더할 수 없다.
    RRF는 점수 대신 '순위'만 쓰기 때문에 정규화 없이 이종 검색기를 합칠 수 있다.
    k=60은 원 논문(Cormack et al., 2009)의 관례값으로, 상위 순위 쏠림을 완화한다.
    """
    scores = {}
    for ranking in (dense, sparse):
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:top_k]


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
