"""RAG engine: two-stage retrieval via fastembed bi-encoder + ONNX cross-encoder.

Stage 1 (bi-encoder recall): cosine similarity over dense embeddings to fetch
  ``fetch_k`` candidates.
Stage 2 (cross-encoder re-ranking): BAAI/bge-reranker-base scores each
  candidate against the query, producing the final top-k ranking.

Runs 100% locally on CPU via ONNX Runtime.  All methods are synchronous
CPU-bound functions — callers must invoke via ``loop.run_in_executor``.
"""

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"


class RAGEngine:
    def __init__(self, documents: list[tuple[str, str]]) -> None:
        self.doc_ids = [d[0] for d in documents]
        self.doc_texts = [d[1] for d in documents]
        self._model = TextEmbedding(model_name=EMBED_MODEL)
        self._reranker = TextCrossEncoder(model_name=RERANK_MODEL)
        self._embeddings: np.ndarray | None = None
        if documents:
            self._build_index()

    def _build_index(self) -> None:
        raw = list(self._model.embed(self.doc_texts, batch_size=32))
        self._embeddings = np.array(raw, dtype=np.float32)

    def _encode_query(self, query: str) -> np.ndarray:
        raw = list(self._model.embed([query], batch_size=1))
        return np.array(raw[0], dtype=np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def retrieve(
        self, query: str, top_k: int = 5, fetch_k: int = 20
    ) -> list[tuple[str, str, float]]:
        if self._embeddings is None or len(self.doc_texts) == 0:
            return []

        # ── Stage 1: bi-encoder recall ──────────────────────────
        query_vec = self._encode_query(query)
        scored = [
            (
                self.doc_ids[i],
                self.doc_texts[i],
                self._cosine(query_vec, self._embeddings[i]),
            )
            for i in range(len(self.doc_texts))
        ]
        scored.sort(key=lambda x: x[2], reverse=True)

        if len(scored) <= top_k:
            return scored[:top_k]

        candidates = scored[: min(fetch_k, len(scored))]

        # ── Stage 2: cross-encoder re-ranking ───────────────────
        candidate_texts = [c[1] for c in candidates]
        reranked = list(self._reranker.rerank(query, candidate_texts))

        results: list[tuple[str, str, float]] = [
            (candidates[i][0], candidates[i][1], float(score))
            for i, score in enumerate(reranked)
        ]

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]


def build_rag_from_chunks(chunks: dict[str, str]) -> RAGEngine:
    documents: list[tuple[str, str]] = []
    for section_name, text in chunks.items():
        raw_lines = [ln.strip() for ln in text.split("\n")]
        lines = [ln for ln in raw_lines if ln and len(ln) > 10]
        for i, line in enumerate(lines):
            doc_id = f"{section_name}_{i}"
            documents.append((doc_id, line))
        if len(text) > 20:
            documents.append((section_name, text[:500]))
    return RAGEngine(documents)
