"""RAG engine: TF-IDF embedding + cosine retrieval. No external deps."""

import json
import math
from collections import Counter


class RAGEngine:
    def __init__(self, documents: list[tuple[str, str]]) -> None:
        self.doc_ids = [d[0] for d in documents]
        self.doc_texts = [d[1] for d in documents]
        self.doc_freq: dict[str, int] = {}
        self.tfidf_vectors: list[dict[str, float]] = []
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in text.split() if len(t) > 1]

    def _build_index(self) -> None:
        tokenized_docs = [self._tokenize(d) for d in self.doc_texts]
        n_docs = len(tokenized_docs)

        for tokens in tokenized_docs:
            for term in set(tokens):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        for tokens in tokenized_docs:
            tf = Counter(tokens)
            vec: dict[str, float] = {}
            for term, count in tf.items():
                idf = math.log((n_docs + 1) / (self.doc_freq.get(term, 1) + 1)) + 1
                vec[term] = count * idf
            self.tfidf_vectors.append(vec)

    def _vectorize_query(self, query: str) -> dict[str, float]:
        tokens = self._tokenize(query)
        tf = Counter(tokens)
        n_docs = len(self.doc_texts)
        vec: dict[str, float] = {}
        for term, count in tf.items():
            doc_count = self.doc_freq.get(term, 0)
            idf = math.log((n_docs + 1) / (doc_count + 1)) + 1 if doc_count > 0 else 1.0
            vec[term] = count * idf
        return vec

    def _cosine(self, a: dict[str, float], b: dict[str, float]) -> float:
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in set(a) | set(b))
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, str, float]]:
        query_vec = self._vectorize_query(query)
        scores = [
            (
                self.doc_ids[i],
                self.doc_texts[i],
                self._cosine(query_vec, self.tfidf_vectors[i]),
            )
            for i in range(len(self.doc_texts))
        ]
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]

    def revalidate(self, job: dict, ctx) -> dict | None:
        """RAG revalidation: re-check matched job details against resume chunks.

        After LLM matching, run a second retrieval with the full JD + extracted
        role/skills as query, and verify extracted fields are consistent.
        """
        query = (
            f"{job.get('role', '')} {job.get('company', '')} "
            f"{' '.join(job.get('matching_skills', []))}"
        )
        chunks = self.retrieve(query, top_k=5)

        if not chunks or chunks[0][2] < 0.01:
            return None  # nothing relevant — likely hallucinated

        verify_prompt = (
            "Cross-check this job extraction against the candidate's relevant "
            "resume snippets. Confirm or correct: role, company, match_percent, "
            "shortlist_probability. Return corrected JSON with same fields, or "
            "set match_percent to 0 if this is clearly wrong.\n\n"
            f"Current extraction: {json.dumps(job, default=str)[:2000]}\n\n"
            "Relevant resume:\n"
            + "\n".join(f"[{c[0]}] {c[1]}" for c in chunks)
            + "\n\nReturn ONLY validated JSON. No markdown."
        )

        result = ctx.json_chat(verify_prompt)
        if isinstance(result, dict) and "match_percent" in result:
            result["match_percent"] = int(result["match_percent"])
            result["shortlist_probability"] = int(result.get("shortlist_probability", 0))
            result["_revalidated"] = True
            return result
        return job  # keep original if validation fails


def build_rag_from_chunks(chunks: dict[str, str]) -> RAGEngine:
    documents: list[tuple[str, str]] = []
    for section_name, text in chunks.items():
        raw_lines = [ln.strip() for ln in text.split("\n")]
        lines = [ln for ln in raw_lines if ln and len(ln) > 10]
        for i, line in enumerate(lines):
            doc_id = f"{section_name}_{i}"
            documents.append((doc_id, line))
        if len(text) > 20:
            documents.append((section_name, text[:300]))
    return RAGEngine(documents)
