"""Candidate Evidence Graph — structured atoms about the candidate's work.

The review's personalization pivot: a persona/RAG store produces generic but
factually-correct prose ("I am excited to apply... my experience with A, B, C
aligns well"). An Evidence Graph stores every real project/experience as a
structured atom (problem, actions, technologies, architecture, scale, outcomes,
decisions, lesson, ownership, roles, industries, seniority) and retrieves the
STRONGEST atoms for a given job's requirements, so answers/cover letters are
built as:

    claim  →  candidate evidence  →  job requirement  →  specific connection

instead of:

    job description  →  LLM  →  generic cover letter

Atoms are built from the resume's projects/experience and from learned Q&A over
time (question encountered -> answer -> evidence atom -> future application).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from src.logging import get_logger

logger = get_logger("evidence_graph")

# Role families (a superset of the radar's role_family set) used to tag atoms.
ROLE_FAMILIES = [
    "backend",
    "frontend",
    "fullstack",
    "infrastructure",
    "devops",
    "ai",
    "data",
    "mobile",
    "security",
    "platform",
    "developer_tools",
]

_INDUSTRY_HINTS = {
    "fintech": ["payment", "bank", "finance", "ledger", "fraud"],
    "health": ["health", "medical", "clinic", "patient"],
    "developer_tools": ["developer tool", "sdk", "cli", "api platform", "open source"],
    "ai_ml": ["llm", "language model", "machine learning", "transformer", "rag", "agent"],
    "infra_cloud": ["infrastructure", "kubernetes", "distributed", "cloud", "streaming"],
    "saas": ["saas", "subscription", "platform"],
    "ecommerce": ["ecommerce", "checkout", "cart", "merchant"],
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _hash_atom(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9_+.#-]{1,}", _norm(text)))


def _match_score(atom_kw: set[str], req_kw: set[str]) -> int:
    return len(atom_kw & req_kw)


class EvidenceGraph:
    """Keyword-scored retrieval over structured candidate evidence atoms."""

    def __init__(self, db: Any = None):
        self.db = db

    def _atoms_from_resume_chunks(self, chunks: dict[str, str]) -> list[dict[str, Any]]:
        """Parse resume projects/experience into structured evidence atoms.

        A resume project block is typically:

            Project Name | 2 months | React, FastAPI
            Built a realtime dashboard used by 200+ daily users.
            Achieved sub-50ms p95 API latency.

        We split the section on blank-ish boundaries, pull the header line for
        title/technologies/roles, and fill problem/actions/outcomes from the
        remaining lines. This is deliberately heuristic — the LLM (and learned
        Q&A) refine it later; the structure is what unlocks personalization.
        """
        atoms: list[dict[str, Any]] = []
        projects_text = chunks.get("projects") or chunks.get("experience") or ""
        blocks = self._split_blocks(projects_text)
        for block in blocks:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if not lines:
                continue
            title = self._extract_title(lines[0])
            techs = self._extract_technologies(lines[0])
            roles = self._detect_roles(lines[0] + " " + " ".join(lines[1:6]))
            outcome_lines = [ln for ln in lines[1:] if self._is_outcome_line(ln)]
            action_lines = [ln for ln in lines[1:] if not self._is_outcome_line(ln)]
            scale = self._extract_scale("\n".join(lines))
            industries = self._detect_industries("\n".join(lines))
            seniority = self._detect_seniority("\n".join(lines))
            keywords = set(techs)
            for ln in lines[1:]:
                keywords.update(_tokenize(ln))
            atom = {
                "kind": "project",
                "title": title,
                "problem": self._extract_problem(lines),
                "actions": action_lines[:8],
                "technologies": techs,
                "architecture": [],
                "scale": scale,
                "measurable_outcomes": outcome_lines[:6],
                "decisions": [],
                "failure": "",
                "lesson": "",
                "ownership": self._detect_ownership(lines),
                "evidence": "\n".join(lines),
                "roles": roles,
                "industries": industries,
                "seniority_signal": seniority,
                "keywords": sorted(keywords)[:60],
                "source": "resume",
            }
            if not title and not atom["actions"]:
                continue
            atom["atom_id"] = _hash_atom(atom)
            atoms.append(atom)
        return atoms

    @staticmethod
    def _split_blocks(text: str) -> list[str]:
        lines = text.splitlines()
        blocks: list[str] = []
        cur: list[str] = []
        for ln in lines:
            if not ln.strip() and cur:
                blocks.append("\n".join(cur))
                cur = []
            elif ln.strip():
                cur.append(ln)
        if cur:
            blocks.append("\n".join(cur))
        return blocks

    @staticmethod
    def _extract_title(header: str) -> str:
        # Title is the leading text before '|' or a date range.
        m = re.split(r"\s*[|–—]\s*", header, maxsplit=1)
        t = m[0].strip()
        t = re.sub(r"\b(\w{3,9}) \d{4}\b.*$", "", t)  # strip trailing date
        return t.strip()[:120]

    @staticmethod
    def _extract_technologies(header: str) -> list[str]:
        # After the first '|' (or comma) — tech stack tokens.
        parts = re.split(r"\s*[|]\s*", header, maxsplit=1)
        if len(parts) < 2:
            return []
        tail = parts[1]
        # Split on commas/pipes; keep known-looking tokens.
        toks = [t.strip() for t in re.split(r"[,|]", tail)]
        out = []
        for t in toks:
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9+.#_-]{0,28}", t) and len(t) > 1:
                out.append(t)
        return out[:15]

    @staticmethod
    def _detect_roles(text: str) -> list[str]:
        roles = []
        low = text.lower()
        mapping = {
            "backend": ["backend", "api", "server", "microservice"],
            "frontend": ["frontend", "react", "ui", "web app"],
            "fullstack": ["fullstack", "full-stack"],
            "infrastructure": ["infra", "kubernetes", "docker", "terraform", "deploy"],
            "devops": ["devops", "ci/cd", "jenkins", "pipeline"],
            "ai": ["llm", "language model", "machine learning", "rag", "neural", "agent"],
            "data": ["data pipeline", "etl", "spark", "database"],
            "security": ["security", "auth", "encryption"],
            "platform": ["platform", "sdk", "developer tool"],
        }
        for role, hints in mapping.items():
            if any(h in low for h in hints):
                roles.append(role)
        return roles

    @staticmethod
    def _is_outcome_line(line: str) -> bool:
        return bool(
            re.search(
                r"\d+%|\d+x|\$|reduced|improved|scaled|served|users?|latency|uptime|p9\d|downloads|hackathon|won|ranked|publish|stars|requests",
                line,
                re.I,
            )
        )

    @staticmethod
    def _extract_scale(text: str) -> str:
        m = re.search(
            r"(\d[\d,]*(?:\.\d+)?\s*(?:K|k|M|m|million|thousand)?\s*(?:users|requests|users|people|customers|downloads|stars))",
            text,
            re.I,
        )
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_problem(lines: list[str]) -> str:
        for ln in lines[1:]:
            if (
                re.search(
                    r"(built|created|designed|developed|architected|solved|engineered|shipped|implemented)",
                    ln,
                )
                and len(ln) < 300
            ):
                return ln.strip()[:300]
        return ""

    @staticmethod
    def _detect_ownership(lines: list[str]) -> str:
        for ln in lines[1:]:
            if re.search(
                r"(led|founded|owned|built from scratch|sole|single-handedly|"
                r"architected|designed and built)",
                ln,
            ):
                return ln.strip()[:200]
        return ""

    def _detect_industries(self, text: str) -> list[str]:
        low = text.lower()
        return [ind for ind, hints in _INDUSTRY_HINTS.items() if any(h in low for h in hints)]

    @staticmethod
    def _detect_seniority(text: str) -> str:
        low = text.lower()
        if re.search(r"(led|lead|founded|staff|principal|architect)", low):
            return "senior"
        if re.search(r"(intern|junior|associate)", low):
            return "junior"
        return "mid"

    # ---- persistence + retrieval -------------------------------------------------

    async def persist_atoms(self, store: Any, atoms: list[dict[str, Any]]) -> int:
        """Upsert evidence atoms into candidate_evidence (dedup by atom_id)."""
        if not atoms:
            return 0
        added = 0
        try:
            async with store._pool.acquire() as conn:
                for a in atoms:
                    aid = a.get("atom_id") or _hash_atom(a)
                    exists = await conn.fetchval(
                        "SELECT 1 FROM candidate_evidence WHERE atom_id=$1", aid
                    )
                    if exists:
                        continue
                    await conn.execute(
                        """
                        INSERT INTO candidate_evidence (
                            atom_id, kind, title, problem, actions, technologies,
                            architecture, scale, measurable_outcomes, decisions,
                            failure, lesson, ownership, evidence, roles, industries,
                            seniority_signal, keywords, source
                        )
                        VALUES (
                            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,
                            $15,$16,$17,$18,$19
                        )
                        """,
                        aid,
                        a.get("kind", "project"),
                        a.get("title", ""),
                        a.get("problem", ""),
                        json.dumps(a.get("actions", [])),
                        json.dumps(a.get("technologies", [])),
                        json.dumps(a.get("architecture", [])),
                        a.get("scale", ""),
                        json.dumps(a.get("measurable_outcomes", [])),
                        json.dumps(a.get("decisions", [])),
                        a.get("failure", ""),
                        a.get("lesson", ""),
                        a.get("ownership", ""),
                        a.get("evidence", ""),
                        json.dumps(a.get("roles", [])),
                        json.dumps(a.get("industries", [])),
                        a.get("seniority_signal", ""),
                        json.dumps(a.get("keywords", [])),
                        a.get("source", "resume"),
                    )
                    added += 1
        except Exception as e:
            logger.warning("evidence persist failed", error=str(e))
        if added:
            logger.info("evidence graph persisted atoms", count=added)
        return added

    async def build_from_resume(self, store: Any, chunks: dict[str, str]) -> int:
        """Parse resume chunks into atoms and persist them."""
        atoms = self._atoms_from_resume_chunks(chunks)
        return await self.persist_atoms(store, atoms)

    async def retrieve_for_job(
        self, store: Any, requirements: list[str] | str, limit: int = 6
    ) -> list[dict[str, Any]]:
        """Retrieve the strongest evidence atoms for a job's requirements.

        Requirement text is tokenized; atoms are keyword-scored on their
        keywords/technologies/roles; the top-`limit` are returned. This is the
        evidence->requirement matching step the review asked for.
        """
        req_text = requirements if isinstance(requirements, str) else " ".join(requirements)
        req_kw = _tokenize(req_text)
        try:
            async with store._pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM candidate_evidence ORDER BY created_at ASC")
        except Exception as e:
            logger.warning("evidence retrieve failed", error=str(e))
            return []
        scored: list[tuple[int, dict[str, Any]]] = []
        for r in rows:
            atom = dict(r)
            kw = set(atom.get("keywords") or [])
            kw |= set(atom.get("technologies") or [])
            kw |= set(atom.get("roles") or [])
            score = _match_score(kw, req_kw)
            if score > 0:
                scored.append((score, atom))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [dict(a) for _, a in scored[:limit]]

    async def retrieve_all(self, store: Any) -> list[dict[str, Any]]:
        try:
            async with store._pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM candidate_evidence ORDER BY created_at ASC")
                return [dict(r) for r in rows]
        except Exception as e:
            logger.warning("evidence retrieve all failed", error=str(e))
            return []

    def format_for_prompt(self, atoms: list[dict[str, Any]], limit: int = 4) -> str:
        """Render retrieved atoms as grounding text for the LLM."""
        if not atoms:
            return ""
        parts: list[str] = []
        for a in atoms[:limit]:
            bits = [f"- {a.get('title') or a.get('kind')}"]
            if a.get("problem"):
                bits.append(f"  Problem: {a['problem']}")
            for act in (a.get("actions") or [])[:3]:
                bits.append(f"  Action: {act}")
            for o in (a.get("measurable_outcomes") or [])[:3]:
                bits.append(f"  Result: {o}")
            techs = ", ".join((a.get("technologies") or [])[:8])
            if techs:
                bits.append(f"  Tech: {techs}")
            parts.append("\n".join(bits))
        return "\n".join(parts)
