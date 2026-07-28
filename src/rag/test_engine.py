"""Tests for RAG engine: fastembed semantic retrieval, chunk building."""

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

    def test_retrieve_all(self, mock_embed) -> None:
        docs = [
            ("a", "machine learning tensorflow pytorch"),
            ("b", "react css html frontend design"),
        ]
        engine = RAGEngine(docs)
        results = engine.retrieve("react frontend developer", top_k=2)
        assert len(results) == 2

    def test_retrieve_empty_query(self, mock_embed) -> None:
        docs = [("a", "some content here")]
        engine = RAGEngine(docs)
        results = engine.retrieve("", top_k=1)
        assert len(results) == 1

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
        docs = [(f"doc_{i}", f"content number {i} with unique terms {i}xyz") for i in range(100)]
        engine = RAGEngine(docs)
        results = engine.retrieve("content 50xyz", top_k=3)
        assert len(results) == 3
