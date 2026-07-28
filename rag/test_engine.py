"""Tests for RAG engine: TF-IDF indexing, retrieval, revalidation."""

from rag.engine import RAGEngine, build_rag_from_chunks


class TestRAGEngine:
    def test_index_empty_docs(self) -> None:
        engine = RAGEngine([])
        assert len(engine.doc_texts) == 0
        assert engine.retrieve("anything") == []

    def test_index_and_retrieve(self) -> None:
        docs = [
            ("skills_0", "python django flask fastapi"),
            ("skills_1", "react typescript javascript frontend"),
            ("exp_0", "built rest apis with python and django"),
            ("exp_1", "developed react components for dashboard"),
        ]
        engine = RAGEngine(docs)
        assert len(engine.tfidf_vectors) == 4

        results = engine.retrieve("python backend developer", top_k=2)
        assert len(results) == 2
        assert "skills_0" in [r[0] for r in results]  # most relevant
        assert all(0 <= r[2] <= 1 for r in results)  # scores in [0,1]

    def test_retrieve_exact_match(self) -> None:
        docs = [
            ("a", "machine learning tensorflow pytorch"),
            ("b", "react css html frontend design"),
        ]
        engine = RAGEngine(docs)
        results = engine.retrieve("react frontend developer", top_k=1)
        assert results[0][0] == "b"

    def test_retrieve_no_match(self) -> None:
        docs = [("a", "python django"), ("b", "java spring")]
        engine = RAGEngine(docs)
        results = engine.retrieve("ruby rails", top_k=2)
        # Should still return results, just with low scores
        assert len(results) == 2
        assert all(r[2] < 0.5 for r in results)

    def test_cosine_same_vector(self) -> None:
        docs = [("a", "python django api"), ("b", "python django api")]
        engine = RAGEngine(docs)
        results = engine.retrieve("python django api", top_k=2)
        assert results[0][2] >= results[1][2]

    def test_build_from_chunks(self) -> None:
        chunks = {
            "skills": "python\ndjango\nreact\npostgresql",
            "experience": "built api with python and django\nled frontend team using react",
            "education": "bachelor of computer science",
        }
        engine = build_rag_from_chunks(chunks)
        assert len(engine.doc_texts) >= 5  # lines + section summaries

        results = engine.retrieve("python backend engineer", top_k=3)
        assert len(results) == 3
        assert any("skills" in r[0] for r in results)

    def test_retrieve_empty_query(self) -> None:
        docs = [("a", "some content here")]
        engine = RAGEngine(docs)
        results = engine.retrieve("", top_k=1)
        # Empty query should still return but with zero score
        assert len(results) == 1
        assert results[0][2] == 0.0

    def test_large_document_count(self) -> None:
        docs = [(f"doc_{i}", f"content number {i} with unique terms {i}xyz") for i in range(100)]
        engine = RAGEngine(docs)
        results = engine.retrieve("content 50xyz", top_k=3)
        assert len(results) == 3
        assert "doc_50" in [r[0] for r in results]
