"""Memory wizard: the ``init-memory`` flow, driven from a Discord thread.

Reuses the terminal grill's question set (``grill_persona``) and the persona
build/embed logic (``build_persona``), but instead of a rich prompt the host
(Discord thread) supplies answers one at a time:

- ``log(text)`` posts a status line.
- ``ask(question, meta) -> str | None`` asks one question and awaits the
  answer; ``None`` means the user skipped. ``meta`` carries ``buttons``
  (labels to render) and ``hint`` (current value to show).

The wizard checks infra (postgres + embedding server), indexes the resume,
grills only the persona data that is still missing, accepts optional extra
Q&A, rebuilds memory, and returns a summary line.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]  # packages/ingest
REPO = ROOT.parent.parent  # repo root
AUTOFILL = REPO / "packages" / "autofill"
AUTOFILL_SCRIPTS = AUTOFILL / "scripts"

for _p in (REPO, ROOT, AUTOFILL, AUTOFILL_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ["LOG_LEVEL"] = "WARNING"  # quiet JSON log spam in setup scripts

from build_persona import (  # type: ignore[import-not-found]  # noqa: E402
    embed_chunks,
    render_chunks,
    resume_summary,
)
from grill_persona import (  # type: ignore[import-not-found]  # noqa: E402
    CONTACT_FIELDS,
    CORE_QUESTIONS,
    LINK_FIELDS,
    PERSONA_JSON,
    _normalize_answer,
    _resume_defaults,
    _sanitize_link,
    build_question_set,
    identity_mismatches,
)
from init_memory import (  # type: ignore[import-not-found]  # noqa: E402
    _start_postgres,
    embed_server_ready,
)

AskFn = Callable[[str, dict[str, Any]], Awaitable[str | None]]
LogFn = Callable[[str], Awaitable[None]]

_SKIP_TEXT = {"skip", "skip it", "pass", "next", "done"}

_YES_NO_CATEGORIES = {
    "work_authorization",
    "visa_sponsorship",
    "disability",
    "veteran_status",
    "employee_relation",
}

_URL_RX = re.compile(r"https?://[^\s]+")


class MemoryWizardError(Exception):
    """Raised when the wizard cannot proceed (missing infra, bad resume)."""


def parse_instruction(text: str) -> dict[str, Any]:
    """Extract intent from the free-text part of ``/memory ...``.

    Handles the natural phrasing the wizard was built for:

      /memory
      /memory update this and add my resume and portfolio
      /memory resume https://example.com/resume.pdf
      /memory portfolio https://example.com
      /memory everything      (re-ask every persona question)
      /memory no resume       (skip the resume step)

    A URL is classified by what precedes it: a document extension
    (.pdf/.docx/.txt/.html) or the word "resume"/"cv" marks it as the resume;
    "portfolio"/"website" marks it as the website. A lone bare URL with no
    keyword is treated as the resume link.
    """
    raw = text or ""
    low = raw.strip().lower()
    urls = _URL_RX.findall(raw)
    resume_url: str | None = None
    resume_path: str | None = None
    website: str | None = None
    force_all = bool(re.search(r"\b(all|everything|re-grill|refresh)\b", low)) or "--all" in low

    for url in urls:
        if re.search(r"\.(?:pdf|docx|txt|html?)(?:\?|#|$)", url, re.I):
            resume_url = resume_url or url
            continue
        # Look at the words immediately before this URL to decide intent.
        pos = raw.find(url)
        prefix = raw[max(0, pos - 80) : pos].lower()
        if re.search(r"\b(portfolio|website|my site|web)\b", prefix):
            website = website or url
        elif re.search(r"\bresume\b|\bcv\b", prefix):
            resume_url = resume_url or url
        elif "portfolio" in low or "website" in low:
            website = website or url
        else:
            resume_url = resume_url or url

    if not urls and "resume" in low:
        path = re.search(
            r"(?:\b(?:resume|path)\s*[:\-]?\s*)?([~.\w/]+\.(?:pdf|docx|txt|html))",
            low,
        )
        if path:
            resume_path = path.group(1)
    if resume_path and resume_url is None and urls:
        resume_path = None
    return {
        "resume_url": resume_url,
        "resume_path": resume_path,
        "website": website,
        "force_all": force_all,
        "no_resume": bool(re.search(r"\b(no resume|skip resume)\b", low)),
    }


def _is_skip(answer: str | None) -> bool:
    if not answer:
        return True
    return answer.strip().lower() in _SKIP_TEXT


def format_persona() -> str:
    """Render persona.json as a compact, readable block for /persona."""
    try:
        data = json.loads(PERSONA_JSON.read_text())
    except Exception as e:
        return f"No persona.json yet ({e})"
    lines: list[str] = []
    name = data.get("name") or (data.get("identity") or {}).get("firstName") or ""
    if name:
        lines.append(f"# {name}")
    identity = data.get("identity") or {}
    contact = [f"{k}: {v}" for k, v in identity.items() if v and k in CONTACT_FIELDS]
    if contact:
        lines.append("**Contact**\n" + "\n".join(f"· {c}" for c in contact))
    answers = data.get("answers") or []
    if answers:
        blocks = [f"**Q:** {a['question']}\n**A:** {a['answer']}" for a in answers]
        lines.append("**Q&A**\n" + "\n\n".join(blocks))
    if data.get("resume_summary"):
        lines.append("**Resume summary**\n" + data["resume_summary"].strip())
    return "\n\n".join(lines) or "_persona.json is empty._"


class MemoryWizard:
    """Run the memory update flow with a host-supplied ask/log interface."""

    def __init__(
        self,
        ask: AskFn,
        log: LogFn,
        persona_json: Path | None = None,
        ctx: Any | None = None,
    ) -> None:
        self.ask = ask
        self.log = log
        self.persona_json = persona_json or PERSONA_JSON
        self.ctx = ctx

    # ── infra ─────────────────────────────────────────────────────────

    async def _ensure_infra(self) -> None:
        from src.memory.pgvector_store import MemoryStore

        await self.log("**Step 1 · Infrastructure**\nChecking Postgres and the embedding server...")
        try:
            store = await MemoryStore.create()
            await store.close()
        except Exception:
            await self.log("Postgres is down — starting `agent-memory-db` via docker compose...")
            if not await _start_postgres():
                raise MemoryWizardError(
                    "Could not start Postgres. Start it manually:\n"
                    "`docker compose -f packages/ingest/docker-compose.yaml up -d agent-memory-db`"
                ) from None
        if not await embed_server_ready():
            await self.log("Embedding server is down — spawning `scripts/serve.py`...")
            if not await self._ensure_embed_server():
                raise MemoryWizardError(
                    "Embedding server unavailable. Start it with "
                    "`uv run python packages/ingest/scripts/serve.py` and run /memory again."
                ) from None
        await self.log("Infra ready · Postgres :5433 · embeddings :8900")

    async def _ensure_embed_server(self) -> bool:
        log = Path("/tmp/opencode") if Path("/tmp/opencode").exists() else Path("/tmp")
        log = log / "embed_server.log"
        with open(log, "a") as out:
            proc = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts" / "serve.py")],
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        await self.log(f"Spawned serve.py (pid {proc.pid}); waiting for health...")
        for _ in range(60):
            if await embed_server_ready():
                return True
            await asyncio.sleep(1)
        return False

    # ── resume ────────────────────────────────────────────────────────

    async def _index_resume(self, resume_url: str | None, resume_path: str | None) -> bool:
        from src.memory.pgvector_store import MemoryStore
        from src.rag.loader import (
            chunk_resume,
            extract_text,
            index_resume_in_pgvector,
            load_resume,
        )

        if resume_path:
            path = Path(resume_path)
            if not path.exists():
                raise MemoryWizardError(f"Resume file not found: {path}") from None
            await self.log("Extracting resume text...")
            full_text = await asyncio.to_thread(extract_text, path)
            chunks = chunk_resume(full_text)
        elif resume_url:
            await self.log(f"Downloading resume from {resume_url}...")
            full_text, chunks = await asyncio.to_thread(load_resume, resume_url)
        else:
            return False
        await self.log("Embedding + indexing resume into memory...")
        store = await MemoryStore.create()
        try:
            await index_resume_in_pgvector(chunks, store)
        finally:
            await store.close()
        await self.log("✅ Resume indexed")
        return True

    # ── persona grill ─────────────────────────────────────────────────

    async def _grill_identity(self, data: dict[str, Any], resume: dict, force_all: bool) -> None:
        identity = dict(data.get("identity", {}) or {})
        existing_name = (
            data.get("name")
            or " ".join(filter(None, (identity.get("firstName"), identity.get("lastName")))).strip()
        )
        resume_name = resume.get("name", "")
        if force_all or not (existing_name or resume_name):
            hint = ""
            if resume_name and not existing_name:
                hint = f"from resume: {resume_name}"
            elif existing_name:
                hint = existing_name
            answer = await self.ask("What is your full name?", {"hint": hint, "buttons": ["skip"]})
            if answer and not _is_skip(answer):
                name = _normalize_answer(answer)
                data["name"] = name
                parts = name.split(None, 1)
                identity["firstName"] = parts[0]
                identity["lastName"] = parts[1] if len(parts) > 1 else ""

        for field in CONTACT_FIELDS:
            existing = identity.get(field, "")
            if existing and not force_all:
                continue
            hint = ""
            if existing:
                hint = f"current: {existing}"
            elif resume.get(field):
                hint = f"from resume: {resume.get(field)}"
            answer = await self.ask(f"What is your {field}?", {"hint": hint, "buttons": ["skip"]})
            if _is_skip(answer) or not answer:
                continue
            if field in LINK_FIELDS:
                answer = _sanitize_link(answer)
            elif field == "email":
                answer = _normalize_answer(answer).lower()
            else:
                answer = _normalize_answer(answer)
            if answer:
                identity[field] = answer

        if not identity.get("firstName") and not existing_name:
            first = await self.ask("First name?", {"buttons": ["skip"]})
            if first and not _is_skip(first):
                identity["firstName"] = first
            last = await self.ask("Last name?", {"buttons": ["skip"]})
            if last and not _is_skip(last):
                identity["lastName"] = last
        data["identity"] = identity

    @staticmethod
    def _previous_answer(
        category: str,
        question: str,
        by_question: dict[str, dict],
        by_category: dict[str, list[dict]],
    ) -> dict:
        if question in by_question:
            return by_question[question]
        matches = by_category.get(category, [])
        if len(matches) == 1:
            return matches[0]
        return {}

    async def _grill_answers(
        self, data: dict[str, Any], force_all: bool, questions: list[tuple[str, str]]
    ) -> int:
        stored = list(data.get("answers", []))
        by_question = {a["question"]: a for a in stored}
        by_category: dict[str, list[dict]] = {}
        for a in stored:
            by_category.setdefault(a["category"], []).append(a)

        result: list[dict] = []
        skipped = 0
        for category, question in questions:
            multi = sum(1 for c, _ in questions if c == category) > 1
            already = question in by_question or (not multi and bool(by_category.get(category)))
            if already and not force_all:
                skipped += 1
                entry = by_question.get(question)
                if entry is None:
                    cat = by_category.get(category) or []
                    entry = cat[0] if cat else None
                if entry is not None:
                    result.append(entry)
                continue
            prev = self._previous_answer(category, question, by_question, by_category)
            if category in _YES_NO_CATEGORIES:
                buttons = ["yes", "no", "pnta", "skip"]
            else:
                buttons = ["pnta", "skip"]
            meta: dict[str, Any] = {"buttons": buttons}
            if prev and prev.get("answer"):
                meta["hint"] = prev["answer"]
            answer = await self.ask(question, meta)
            if _is_skip(answer) or not answer:
                continue
            answer = _normalize_answer(answer, category)
            if not answer:
                continue
            if prev and answer == prev.get("answer", ""):
                final_q = prev.get("question", question)
            else:
                final_q = question
            result.append({"category": category, "question": final_q, "answer": answer})

        wizard_cats = {c for c, _ in questions}
        extra_kept = [a for a in stored if a.get("category") not in wizard_cats]
        data["answers"] = result + extra_kept
        return skipped

    async def _extra_qa(self, data: dict[str, Any]) -> None:
        await self.log("**Anything else?**\nAdd extra Q&A as `question | answer`, or press Done.")
        unparseable = 0
        while True:
            answer = await self.ask(
                "Extra Q&A (`question | answer`, or `done`)",
                {"buttons": ["done"]},
            )
            if _is_skip(answer) or not answer:
                break
            if "|" in answer:
                q, a = (p.strip() for p in answer.split("|", 1))
                if q and a:
                    unparseable = 0
                    data["answers"].append({"category": "general", "question": q, "answer": a})
                    await self.log(f"✅ Saved · **{q}** → {a}")
                else:
                    await self.log("Both a question and an answer are needed: `question | answer`.")
            else:
                unparseable += 1
                await self.log("Format it as `question | answer`, or press Done to finish.")
                # Never trap the user in a loop over the same prompt: after
                # two unparseable replies, accept it as "done".
                if unparseable >= 2:
                    break

    # ── save + build ──────────────────────────────────────────────────

    def _write(self, data: dict[str, Any]) -> None:
        self.persona_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.persona_json.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(tmp, self.persona_json)

    async def _save_and_build(self, data: dict[str, Any]) -> dict[str, int]:
        from src.memory.pgvector_store import MemoryStore

        data["version"] = int(data.get("version", 1)) + 1
        self._write(data)
        answers = data.get("answers", [])
        identity = data.get("identity", {}) or {}
        chunks = render_chunks(answers, identity)
        await self.log(
            f"Rebuilding persona memory from {len(answers)} answers "
            f"+ {len(identity)} identity fields..."
        )
        store = await MemoryStore.create()
        try:
            resume = await resume_summary(store)
            if resume:
                data["resume_summary"] = resume
                self._write(data)
            records = await embed_chunks(chunks)
            await store.clear_persona()
            await store.index_persona_chunks(records)
            count = await store.persona_chunk_count()
            resume_count = await store.chunk_count()
        finally:
            await store.close()
        return {"persona_chunks": count, "resume_chunks": resume_count}

    # ── flow ──────────────────────────────────────────────────────────

    async def _resolve_website(self, data: dict[str, Any], parsed: dict[str, Any], low: str) -> str:
        """Set persona identity.website.

        Source of truth: explicit command URL > PORTFOLIO_URL env > saved value.
        An explicit choice never gets overwritten by the env default. Returns
        the resolved website ("" if none).
        """
        explicit = parsed["website"]
        website = explicit or os.getenv("PORTFOLIO_URL") or ""
        if website:
            existing_website = (data.get("identity", {}) or {}).get("website", "")
            if explicit or not existing_website:
                data.setdefault("identity", {})["website"] = _sanitize_link(website)
                await self.log(f"✅ Portfolio/website set · {website}")
        elif "portfolio" in low or "website" in low:
            await self.log(
                "No portfolio URL provided — set PORTFOLIO_URL or pass one to `/memory`."
            )
        return website

    async def run(
        self,
        instruction: str = "",
        resume_url: str | None = None,
        resume_path: str | None = None,
        force_all: bool = False,
    ) -> str:
        parsed = parse_instruction(instruction)
        if resume_url is None and parsed["resume_url"]:
            resume_url = parsed["resume_url"]
        if resume_path is None and parsed["resume_path"]:
            resume_path = parsed["resume_path"]
        force_all = force_all or parsed["force_all"]
        low = (instruction or "").lower()

        await self._ensure_infra()

        await self.log("**Step 2 · Resume**")
        indexed = False
        if resume_url or resume_path or os.getenv("RESUME_URL") or os.getenv("RESUME_PATH"):
            indexed = await self._index_resume(
                resume_url or os.getenv("RESUME_URL"),
                resume_path or os.getenv("RESUME_PATH"),
            )
        elif "resume" in low and not parsed["no_resume"]:
            answer = await self.ask(
                "No resume source set. Drop the resume URL here, or press Skip.",
                {"buttons": ["skip"]},
            )
            if answer and not _is_skip(answer):
                indexed = await self._index_resume(answer, None)
        if not indexed and not parsed["no_resume"]:
            await self.log(
                "No resume indexed — set RESUME_URL/RESUME_PATH or pass one to `/memory`."
            )

        await self.log("**Step 3 · Persona Q&A**")
        if self.persona_json.exists():
            try:
                data = json.loads(self.persona_json.read_text())
            except Exception:
                data = {"name": "", "version": 1, "identity": {}, "answers": []}
        else:
            data = {"name": "", "version": 1, "identity": {}, "answers": []}

        await self._resolve_website(data, parsed, low)
        resume_defaults: dict = {}
        try:
            resume_defaults = await _resume_defaults()
        except Exception:
            resume_defaults = {}
        if resume_defaults:
            await self.log(
                f"Prefilled identity defaults from the resume ({len(resume_defaults)} fields)."
            )
            mismatches = identity_mismatches(data.get("identity", {}), resume_defaults)
            if mismatches:
                await self.log(
                    "⚠️ Resume disagrees with saved persona:\n"
                    + "\n".join(
                        f"· {field}: saved `{saved}`, resume `{resume_v}`"
                        for field, saved, resume_v in mismatches
                    )
                )
        await self._grill_identity(data, resume_defaults, force_all)

        # Dynamic question set: core application-form questions + LLM-generated
        # follow-ups tailored to the resume and what's already answered.
        try:
            questions = await build_question_set(
                self.ctx,
                str(data.get("resume_summary") or ""),
                data.get("answers", []),
                data.get("identity", {}),
            )
        except Exception:
            questions = list(CORE_QUESTIONS)
        if len(questions) > len(CORE_QUESTIONS):
            await self.log(
                f"Generated {len(questions) - len(CORE_QUESTIONS)} extra questions "
                "from your resume."
            )

        skipped = await self._grill_answers(data, force_all, questions)
        if skipped:
            await self.log(
                f"Skipped {skipped} already-answered question(s) — "
                "`/memory everything` re-asks all."
            )
        await self._extra_qa(data)

        await self.log("**Step 4 · Building memory**")
        counts = await self._save_and_build(data)
        return (
            f"Resume: `{counts['resume_chunks']}` chunks · "
            f"Persona: `{counts['persona_chunks']}` chunks · "
            f"{len(data.get('answers', []))} answers · "
            f"{len(data.get('identity', {}) or {})} identity fields"
        )
