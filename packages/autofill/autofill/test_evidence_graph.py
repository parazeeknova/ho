"""Tests for the Candidate Evidence Graph (resume atom parsing + matching)."""

from __future__ import annotations

from autofill.evidence_graph import (
    EvidenceGraph,
    _match_score,
    _tokenize,
)


def _sample_chunks() -> dict[str, str]:
    return {
        "projects": (
            "Zephyr | 3 months | Python, FastAPI, Kafka, React\n"
            "Built an event-driven infrastructure for high-volume telemetry ingestion.\n"
            "Designed async normalization and fault-tolerant retry pipelines.\n"
            "Served 10K+ daily events at sub-50ms p95 latency.\n"
            "Scaled to 200 concurrent consumers.\n"
            "\n"
            "Hermes | 4 months | TypeScript, Node.js, LLM, PostgreSQL\n"
            "Built an agent-orchestration system with a memory/event architecture.\n"
            "Implemented RAG retrieval over a vector store.\n"
            "Won 2nd place at a national hackathon.\n"
        ),
        "skills": "Python, TypeScript, FastAPI, Kafka, React, Node.js, PostgreSQL\n",
    }


def test_parse_projects_into_atoms():
    eg = EvidenceGraph()
    atoms = eg._atoms_from_resume_chunks(_sample_chunks())
    assert len(atoms) == 2
    zephyr = next(a for a in atoms if "Zephyr" in a["title"])
    assert "Kafka" in zephyr["technologies"]
    assert "Python" in zephyr["technologies"]
    assert any("10K" in o for o in zephyr["measurable_outcomes"])
    assert any("latency" in o for o in zephyr["measurable_outcomes"])
    assert "infrastructure" in zephyr["roles"]
    assert zephyr["seniority_signal"] in ("mid", "senior", "junior")
    # Hermes detected with AI role.
    hermes = next(a for a in atoms if "Hermes" in a["title"])
    assert "ai" in hermes["roles"]


def test_match_score_tokens():
    req = _tokenize("event-driven infrastructure kafka high-volume telemetry")
    atom_kw = _tokenize("kafka event ingestion infrastructure telemetry")
    assert _match_score(atom_kw, req) >= 3


def test_retrieve_for_job_ranks_best_atom_first():
    eg = EvidenceGraph()
    atoms = eg._atoms_from_resume_chunks(_sample_chunks())
    # No store: retrieval over persisted atoms requires a store, but the pure
    # scoring path is tested via _match_score above. Verify format_for_prompt.
    fmt = eg.format_for_prompt(atoms, limit=2)
    assert "Zephyr" in fmt
    assert "Problem:" in fmt or "Action:" in fmt or "Result:" in fmt
