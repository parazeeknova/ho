"""RAG & LLM Integration for answering custom job screener questions."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx

from autofill.profile import Profile
from src.configuration import get_config
from src.llm.context import ContextManager
from src.logging import get_logger
from src.memory.pgvector_store import MemoryStore

logger = get_logger("autofill.rag")

ROOT = Path(__file__).resolve().parent.parent
PERSONA_JSON = ROOT / "persona.json"
PERSONA_TXT = ROOT / "persona.txt"

# Sentinel returned when an answer is a personal fact we must not fabricate.
# The CLI prompts the user for these; the worker leaves them blank.
ASK_USER = "__ASK_USER__"

# Cosine distance (embedding <=>) below which a persona chunk is a confident match.
PERSONA_MATCH_THRESHOLD = 0.6

# Personal / knockout question categories. We never guess these - if the user
# hasn't configured an answer in profile.customAnswers or the persona store,
# we ask instead.
_PERSONAL_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"visa|sponsorship", re.I), "visa"),
    (
        re.compile(r"authorized to work|legally authorized|work authorization|right to work", re.I),
        "authorization",
    ),
    (
        re.compile(r"expected (annual )?(cash )?(salary|compensation)|salary expectation|expected comp", re.I),
        "expected_comp",
    ),
    (
        re.compile(r"current (annual )?(cash )?compensation|current salary|current comp", re.I),
        "current_comp",
    ),
    (re.compile(r"current location|currently (based|located|residing|living)", re.I), "current_location"),
    (re.compile(r"how soon.*join|when can you (start|join)|start date|availability|notice period", re.I), "start_date"),
    (re.compile(r"relocat", re.I), "relocation"),
    (re.compile(r"hybrid|in.?office|work model|office per week", re.I), "work_model"),
    (re.compile(r"equity|rsu|esop|stock options|hold any equity", re.I), "equity"),
]

# Personal-fact categories that resolve from configured data (no guessing, no prompting).
_DETERMINISTIC_ANSWERS = {
    "visa": "No",
    "authorization": "Yes",
}

# These still require the configured min-salary / are safe defaults.
_EXPECTED_COMP_KEYS = {"expected_comp"}

# Countries used to keep work-authorization answers country-specific.
_COUNTRY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("india", re.compile(r"\bindia\b", re.I)),
    ("united states", re.compile(r"united states|\busa?\b", re.I)),
    ("uk", re.compile(r"\buk\b|\bunited kingdom\b", re.I)),
    ("canada", re.compile(r"\bcanada\b", re.I)),
    ("australia", re.compile(r"\baustralia\b", re.I)),
]


def _mentioned_countries(text: str) -> set[str]:
    return {name for name, pat in _COUNTRY_PATTERNS if pat.search(text)}


async def _embed_text(text: str) -> list[float] | None:
    """Embed a single text via the configured local embedding server."""
    cfg = get_config().embed
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
            resp = await client.post(
                f"{cfg.url}/embeddings",
                json={"model": cfg.model, "input": [text[:2000]]},
            )
            resp.raise_for_status()
            emb = resp.json()["data"][0]["embedding"]
            if isinstance(emb, list) and len(emb) > 0:
                return [float(v) for v in emb]
    except Exception as e:
        logger.warning("Embedding lookup failed", error=str(e))
    return None


class ScreenerRAG:
    """Answers screener questions using candidate persona and GeneralCompute LLM.

    ``store`` is an optional MemoryStore used to retrieve grilled personal facts
    (persona_embeddings) and resume context (resume_embeddings). When no store is
    provided, persona/resume retrieval is skipped and only deterministic rules,
    customAnswers, and the LLM are used.
    """

    def __init__(
        self,
        context_manager: ContextManager | None = None,
        profile: Profile | None = None,
        store: MemoryStore | None = None,
    ) -> None:
        self.cm = context_manager or ContextManager()
        self.profile = profile or Profile()
        self.store = store

    async def close(self) -> None:
        if self.store is not None:
            await self.store.close()
            self.store = None

    async def _embed(self, text: str) -> list[float] | None:
        return await _embed_text(text)

    def _country_mismatch(self, q_lower: str, stored_lower: str) -> bool:
        qc = _mentioned_countries(q_lower)
        sc = _mentioned_countries(stored_lower)
        if qc and sc:
            return not (qc & sc)
        return False

    async def _lookup_persona(self, q: str, q_lower: str) -> str | None:
        """Retrieve a ground-truth persona answer for a question, if any."""
        if self.store is None:
            return None
        emb = await self._embed(q)
        if not emb:
            return None
        try:
            results = await self.store.search_similar_persona(emb, top_k=6)
        except Exception as e:
            logger.warning("Persona search failed", error=str(e))
            return None
        for r in results:
            # Keep work-authorization country-specific.
            if r["category"] == "work_authorization" and self._country_mismatch(
                q_lower, (r["question"] or "").lower()
            ):
                continue
            if r["distance"] <= PERSONA_MATCH_THRESHOLD:
                answer = (r.get("answer") or "").strip()
                if answer:
                    logger.info(
                        "Persona match",
                        category=r.get("category"),
                        distance=r["distance"],
                        question=q,
                    )
                    return answer
        return None

    async def _gather_context(self, questions: list[str]) -> str:
        """Collect grounding text from resume_embeddings for open-ended questions."""
        if self.store is None or not questions:
            return ""
        seen: set[str] = set()
        parts: list[str] = []
        for q in questions[:5]:
            emb = await self._embed(q)
            if not emb:
                continue
            try:
                rows = await self.store.search_similar_chunks(emb, top_k=3)
            except Exception as e:
                logger.warning("Resume context search failed", error=str(e))
                continue
            for r in rows:
                content = (r.get("content") or "").strip()
                if content and content not in seen:
                    seen.add(content)
                    parts.append(content)
        return "\n".join(parts[:12])

    def _match_custom_answer(self, q: str, q_lower: str) -> str | None:
        """Return a configured customAnswers value if it fuzzy-matches the question."""
        for custom_key, custom_val in self.profile.customAnswers.items():
            if custom_key.lower() in q_lower or q_lower in custom_key.lower():
                return custom_val
        return None

    async def answer_questions(self, questions: list[str]) -> dict[str, str]:
        """Generate answers for a list of screener questions.

        Resolution order per question:
        1. profile.customAnswers (explicit config)
        2. persona_embeddings (grilled ground truth)
        3. deterministic rules (visa/authorization/expected-comp fallback)
        4. LLM grounded in resume + persona context (open-ended)
        5. __ASK_USER__ when nothing grounds the answer.
        """
        if not questions:
            return {}

        logger.info("Generating RAG answers for questions", count=len(questions))

        cfg = get_config()
        persona_text = getattr(cfg.candidate, "persona", "") or (
            "Experienced Software Engineer with strong background in "
            "backend, Python, Node.js, and cloud systems."
        )
        min_salary = getattr(cfg.candidate, "min_salary", "Flexible / Open to discussion")

        answers: dict[str, str] = {}
        unresolved_questions: list[str] = []

        for q in questions:
            q_lower = q.lower()

            custom = self._match_custom_answer(q, q_lower)
            if custom is not None:
                answers[q] = custom
                continue

            persona_ans = await self._lookup_persona(q, q_lower)
            if persona_ans is not None:
                answers[q] = persona_ans
                continue

            matched_rule = next(((p, key) for p, key in _PERSONAL_RULES if p.search(q)), None)
            if matched_rule:
                _, key = matched_rule
                if key in _DETERMINISTIC_ANSWERS:
                    answers[q] = _DETERMINISTIC_ANSWERS[key]
                elif key in _EXPECTED_COMP_KEYS:
                    answers[q] = min_salary
                else:
                    answers[q] = ASK_USER
                continue

            unresolved_questions.append(q)

        if not unresolved_questions:
            return answers

        context = await self._gather_context(unresolved_questions)

        prompt = f"""
You are completing a job application form on behalf of the candidate.

Candidate Background & Persona:
{persona_text}

Candidate Name: {self.profile.firstName} {self.profile.lastName}
Candidate Email: {self.profile.email}
Candidate LinkedIn: {self.profile.linkedin}
Candidate GitHub: {self.profile.github}
"""
        if context:
            prompt += f"""
Verified facts retrieved from the candidate's resume:
{context}
"""
        prompt += f"""
Writing style rules (follow strictly for every answer):
- Direct and professional-casual. No "Dear Hiring Manager" tone, no corporate filler.
- Lead with concrete, quantified outcomes from the persona (metrics, numbers, specific tech) instead of generic claims like "passionate about" or "excited to leverage."
- Every sentence must earn its place. Cut anything that doesn't add information.
- Never use em dashes.
- No buzzwords, no vague enthusiasm statements, no restating the question back before answering.
- Answers must be strictly grounded in the persona and profile data above. Do not invent facts, projects, or numbers not present in the persona.
- Match answer length to the question: one or two tight sentences for short-answer fields, a short paragraph (3-5 sentences) for "why this role/company" style prompts. Never pad to sound more substantial.

CRITICAL RULE: If a question asks for a personal fact or detail that is NOT present in the
candidate persona or profile above (for example exact dates, precise numbers, compensation,
location, availability, or anything you would be guessing), do NOT invent an answer. Return
the exact literal string "__ASK_USER__" as that question's answer value instead.

Answer the following open-ended application questions concisely, professionally, and accurately as
the candidate, following the style rules above:
{json.dumps(unresolved_questions, indent=2)}

Return a JSON object mapping each question string to its generated answer string. Return only the JSON object, no preamble or explanation.
"""

        try:
            schema = {
                "type": "object",
                "additionalProperties": {"type": "string"}
            }
            raw_resp = await self.cm.chat(prompt, schema=schema)
            cleaned = raw_resp.strip()

            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()

            generated = json.loads(cleaned)
            for q, a in generated.items():
                if isinstance(a, str) and a.strip() and a.strip() != ASK_USER:
                    answers[q] = a.strip()
                else:
                    answers[q] = ASK_USER

            # Questions the LLM silently omitted are unknown, not "N/A".
            for q in unresolved_questions:
                if q not in answers:
                    answers[q] = ASK_USER

        except Exception as e:
            logger.exception("Failed to generate LLM RAG answers", error=str(e))
            for q in unresolved_questions:
                if q not in answers:
                    answers[q] = ASK_USER

        return answers

    async def learn(self, question: str, answer: str) -> bool:
        """Persist a user-provided answer into the persona knowledge base.

        Indexes the Q&A into ``persona_embeddings`` so future, semantically
        similar questions resolve to it, and appends it to ``persona.json``
        (and ``persona.txt``) so the knowledge survives rebuilds. Exact
        duplicate questions are skipped. Returns True when persisted.
        """
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not question or not answer:
            return False

        if self.store is not None:
            try:
                if await self.store.persona_question_exists(question):
                    logger.info("Learn skipped: question already known", question=question)
                    return False
            except Exception as e:
                logger.warning("Learn dedup check failed", error=str(e))

        matched_rule = next(((p, key) for p, key in _PERSONAL_RULES if p.search(question)), None)
        category = matched_rule[1] if matched_rule else "general"
        content = f"Q: {question}\nA: {answer}"

        indexed = False
        if self.store is not None:
            emb = await self._embed(content)
            if emb:
                try:
                    await self.store.index_persona_chunks(
                        [
                            {
                                "category": category,
                                "question": question,
                                "answer": answer,
                                "content": content,
                                "embedding": emb,
                            }
                        ]
                    )
                    indexed = True
                except Exception as e:
                    logger.warning("Learn indexing failed", error=str(e))
            else:
                logger.warning("Learn embedding failed; keeping persona.json record only")

        self._append_persona_json(question, answer, category)
        self._append_persona_txt(question, answer)
        logger.info(
            "Learned answer persisted",
            question=question,
            category=category,
            indexed=indexed,
        )
        return True

    def _append_persona_json(self, question: str, answer: str, category: str) -> None:
        """Durably append a learned Q&A to persona.json (atomic write)."""
        try:
            data = json.loads(PERSONA_JSON.read_text())
        except (OSError, json.JSONDecodeError):
            data = {"name": "", "version": 1, "answers": []}
        data["version"] = int(data.get("version", 1)) + 1
        data.setdefault("answers", []).append(
            {"category": category, "question": question, "answer": answer}
        )
        tmp = PERSONA_JSON.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(tmp, PERSONA_JSON)

    def _append_persona_txt(self, question: str, answer: str) -> None:
        """Insert the learned line into persona.txt before the resume section."""
        try:
            text = PERSONA_TXT.read_text()
        except OSError:
            return
        line = f"- {question}: {answer}"
        marker = "From Resume:"
        if marker in text:
            text = text.replace(marker, f"{line}\n{marker}", 1)
        else:
            text = text.rstrip("\n") + f"\n{line}\n"
        PERSONA_TXT.write_text(text)
