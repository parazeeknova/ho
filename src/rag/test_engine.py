"""Tests for RAG engine: two-stage bi-encoder + cross-encoder retrieval."""

import numpy as np
import pytest

from src.rag.engine import RAGEngine, build_rag_from_chunks


@pytest.fixture
def mock_embed(mocker):
    embeddings = {
        "python django flask fastapi": [0.9, 0.1, 0.0, 0.0],
        "react typescript javascript frontend": [0.1, 0.9, 0.0, 0.0],
        "built rest apis with python and django": [0.8, 0.0, 0.0, 0.0],
        "developed react components for dashboard": [0.0, 0.8, 0.0, 0.0],
    }

    def _fake_embed(texts, batch_size=32):
        for t in texts:
            if t in embeddings:
                yield np.array(embeddings[t], dtype=np.float32)
            else:
                vec = np.zeros(4, dtype=np.float32)
                for i, word in enumerate(t.split()):
                    vec[i % 4] = float(hash(word) % 10) / 10
                yield vec

    mock_model = mocker.MagicMock()
    mock_model.embed.side_effect = _fake_embed
    mocker.patch("src.rag.engine.TextEmbedding", return_value=mock_model)

    def _fake_rerank(query: str, docs: list[str]):
        for doc in docs:
            tag = "python" if "python" in doc.lower() else "frontend"
            yield 0.9 if tag in query.lower() else 0.3

    mock_reranker = mocker.MagicMock()
    mock_reranker.rerank.side_effect = _fake_rerank
    mocker.patch("src.rag.engine.TextCrossEncoder", return_value=mock_reranker)

    return mock_model


class TestRAGEngine:
    def test_index_empty_docs(self, mock_embed) -> None:
        engine = RAGEngine([])
        assert len(engine.doc_texts) == 0
        assert engine.retrieve("anything") == []

    def test_index_and_retrieve(self, mock_embed) -> None:
        docs = [
            ("skills_0", "python django flask fastapi"),
            ("skills_1", "react typescript javascript frontend"),
            ("exp_0", "built rest apis with python and django"),
            ("exp_1", "developed react components for dashboard"),
        ]
        engine = RAGEngine(docs)
        assert engine._embeddings is not None
        assert engine._embeddings.shape[0] == 4

        results = engine.retrieve("python backend developer", top_k=2)
        assert len(results) == 2
        assert all(0 <= r[2] <= 1 for r in results)

    def test_retrieve_skips_rerank_when_single_doc(self, mock_embed) -> None:
        docs = [
            ("a", "machine learning tensorflow pytorch"),
        ]
        engine = RAGEngine(docs)
        results = engine.retrieve("react frontend developer", top_k=5)
        assert len(results) == 1
        assert 0 <= results[0][2] <= 1

    def test_retrieve_empty_query(self, mock_embed) -> None:
        docs = [("a", "some content here")]
        engine = RAGEngine(docs)
        results = engine.retrieve("", top_k=1)
        assert len(results) == 1

    def test_reranker_prefers_semantic_match(self, mock_embed) -> None:
        docs = [
            ("py_0", "python django backend api rest"),
            ("py_1", "python machine learning data science"),
            ("fe_0", "react frontend css html design"),
            ("fe_1", "typescript angular ui components"),
        ]
        engine = RAGEngine(docs)
        results = engine.retrieve("python api developer", top_k=2, fetch_k=4)
        assert len(results) == 2
        assert results[0][0] in ("py_0", "py_1")

    def test_build_from_chunks(self, mock_embed) -> None:
        chunks = {
            "skills": "python\ndjango\nreact\npostgresql",
            "experience": "built api with python and django\nled frontend team using react",
            "education": "bachelor of computer science",
        }
        engine = build_rag_from_chunks(chunks)
        assert len(engine.doc_texts) >= 4

        results = engine.retrieve("python backend engineer", top_k=3)
        assert len(results) == 3

    def test_large_document_count(self, mock_embed) -> None:
        docs = [
            (f"doc_{i}", f"content number {i} with unique terms {i}xyz")
            for i in range(100)
        ]
        engine = RAGEngine(docs)
        results = engine.retrieve("content 50xyz", top_k=3)
        assert len(results) == 3

    def test_fetch_k_limits_candidates(self, mock_embed, mocker) -> None:
        docs = [(f"doc_{i}", f"content number {i}") for i in range(50)]
        engine = RAGEngine(docs)

        rerank_spy = mocker.spy(engine._reranker, "rerank")
        engine.retrieve("content", top_k=3, fetch_k=10)
        rerank_spy.assert_called_once()
        args = rerank_spy.call_args[0]
        assert len(args[1]) == 10
