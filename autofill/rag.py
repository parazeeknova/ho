"""RAG & LLM Integration for answering custom job screener questions."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional
from src.llm.context import ContextManager
from src.configuration import get_config
from src.logging import get_logger
from autofill.profile import Profile

logger = get_logger("autofill.rag")


class ScreenerRAG:
    """Answers screener questions using candidate persona and GeneralCompute LLM."""

    def __init__(self, context_manager: Optional[ContextManager] = None) -> None:
        self.cm = context_manager or ContextManager()
        self.profile = Profile()

    async def answer_questions(self, questions: List[str]) -> Dict[str, str]:
        """Generate answers for a list of screener questions."""
        if not questions:
            return {}

        logger.info("Generating RAG answers for questions", count=len(questions))

        cfg = get_config()
        persona_text = (
            getattr(cfg.candidate, "candidate_persona", "")
            or "Experienced Software Engineer with strong background in backend, Python, Node.js, and cloud systems."
        )
        min_salary = getattr(cfg.candidate, "candidate_min_salary", "Flexible / Open to discussion")

        answers: Dict[str, str] = {}
        unresolved_questions: List[str] = []

        for q in questions:
            q_lower = q.lower()
            if "visa" in q_lower or "sponsorship" in q_lower:
                answers[q] = "No"
            elif "authorized to work" in q_lower or "legally authorized" in q_lower:
                answers[q] = "Yes"
            elif "salary" in q_lower or "compensation" in q_lower:
                answers[q] = min_salary
            else:
                # Fuzzy keyword lookup in customAnswers
                matched = False
                for custom_key, custom_val in self.profile.customAnswers.items():
                    if custom_key.lower() in q_lower or q_lower in custom_key.lower():
                        answers[q] = custom_val
                        matched = True
                        break
                if not matched:
                    unresolved_questions.append(q)

        if not unresolved_questions:
            return answers

        prompt = f"""
You are completing a job application form on behalf of the candidate.
Candidate Background & Persona:
{persona_text}

Candidate Name: {self.profile.firstName} {self.profile.lastName}
Candidate Email: {self.profile.email}
Candidate LinkedIn: {self.profile.linkedin}
Candidate GitHub: {self.profile.github}

Answer the following open-ended application questions concisely, professionally, and accurately as the candidate:
{json.dumps(unresolved_questions, indent=2)}

Return a JSON object mapping each question string to its generated answer string.
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
                answers[q] = a

        except Exception as e:
            logger.exception("Failed to generate LLM RAG answers", error=str(e))
            for q in unresolved_questions:
                if q not in answers:
                    answers[q] = "N/A"

        return answers
