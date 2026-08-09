"""End-to-end MemoryWizard integration test.

Runs the real wizard against live Postgres + embedding server + LLM, feeding
realistic answers to the dynamic (LLM-generated) questions, and verifies:
  1. the resume is indexed into resume_embeddings
  2. the portfolio URL lands in persona.json identity.website
  3. the answered dynamic questions are persisted in persona.json answers
  4. the persona is embedded and retrievable via pgvector similarity search
  5. /persona (format_persona) renders the whole thing

Writes to the REAL data/persona.json (the product file) by design — this is
the flow a real /memory run executes. Back up first.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
for _p in (str(ROOT), f"{ROOT}/packages/ingest", f"{ROOT}/packages/autofill"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.insert(0, f"{ROOT}/packages/autofill/scripts")

from src.agent.memory_wizard import MemoryWizard, format_persona  # noqa: E402
from src.llm.context import ContextManager  # noqa: E402
from src.memory.pgvector_store import MemoryStore  # noqa: E402

RESUME_URL = "https://f.example.com/raw/XghaIR.pdf"
PORTFOLIO = "https://example.com"

# Answers keyed by topic keywords; the LLM's generated questions vary run to
# run, so we answer by topic rather than exact question text.
TOPIC_ANSWERS: list[tuple[list[str], str]] = [
    (
        ["strongest", "programming language", "deepest"],
        "Rust for systems programming, TypeScript for product work",
    ),
    (
        ["proud", "amemorymachine", "verso", "complex infrastructure"],
        "Verso: self-hosted RAG knowledge base with OCR, real-time "
        "multiplayer markdown, long-term memory",
    ),
    (["timezone"], "IST (India), flexible for US overlap"),
    (["scale", "largest product"], "Asocialmedia served 10K+ page views with sub-50ms APIs"),
    (
        ["hardware", "workspace", "requirement"],
        "No special hardware; need full infra access and stable net",
    ),
    (["sponsorship"], "Open to visa sponsorship for the right role"),
    (
        ["full-stack", "area", "specialize"],
        "Backend + infrastructure, production TypeScript frontends",
    ),
    (["team size", "structure"], "Prefer small cross-functional teams"),
    (["research", "llm"], "CICBA 2025 Springer paper on fine-tuning LLMs for code benchmarks"),
    (["relocation", "region"], "Open to relocating for a strong opportunity"),
]


def answer_for(question: str) -> str:
    ql = question.lower()
    for keys, answer in TOPIC_ANSWERS:
        if any(k in ql for k in keys):
            return answer
    return ""


async def run_wizard() -> dict:
    asked: list[str] = []
    answered: list[tuple[str, str]] = []
    extra_sent = False

    async def fake_ask(question: str, meta: dict) -> str | None:
        nonlocal extra_sent
        asked.append(question)
        if "Extra Q&A" in question:
            # Submit one real extra Q&A, then finish on the next call.
            if not extra_sent:
                extra_sent = True
                return "portfolio | https://example.com is my portfolio"
            return "done"
        ans = answer_for(question)
        if ans:
            answered.append((question, ans))
            print(f"  ▶ {question[:70]}  =>  {ans}")
            return ans
        print(f"  · skip: {question[:60]}")
        return "skip"

    async def fake_log(text: str) -> None:
        print(f"  · {text.splitlines()[0][:100]}")

    ctx = ContextManager()
    wizard = MemoryWizard(
        ask=fake_ask, log=fake_log, ctx=ctx, persona_json=Path("data/persona.json")
    )
    result = await wizard.run(f"update this and add my resume and portfolio {PORTFOLIO}")
    await ctx.aclose()

    data = json.loads(Path("data/persona.json").read_text())
    return {
        "result": result,
        "asked": asked,
        "answered": answered,
        "data": data,
    }


async def verify() -> None:
    print("=== E2E: /memory update ... portfolio ===")
    outcome = await run_wizard()
    data = outcome["data"]

    print("\n--- Checks ---")
    ok = True

    # 1. portfolio in identity
    website = (data.get("identity") or {}).get("website", "")
    status = "PASS" if website == PORTFOLIO else f"FAIL (got {website!r})"
    ok &= website == PORTFOLIO
    print(f"[{status}] website = {website}")

    # 2. resume embeddings updated
    store = await MemoryStore.create()
    try:
        rc = await store.chunk_count()
        print(f"[{'PASS' if rc >= 50 else 'FAIL'}] resume_chunks = {rc}")
        ok &= rc >= 50

        # 3. dynamic answers persisted
        dyn = [
            a
            for a in data.get("answers", [])
            if a["category"] not in ("identity", "general")
            and a["answer"] != "Prefer not to answer"
        ]
        print(f"[{'PASS' if len(dyn) >= 5 else 'FAIL'}] dynamic answers saved = {len(dyn)}")
        ok &= len(dyn) >= 5
        for a in dyn:
            print(f"      [{a['category']}] {a['question'][:55]} => {a['answer'][:55]}")

        # 4. extra Q&A saved
        extra = [a for a in data.get("answers", []) if a["category"] == "general"]
        print(f"[{'PASS' if extra else 'FAIL'}] extra Q&A saved = {len(extra)}")
        ok &= bool(extra)
        for a in extra:
            print(f"      [{a['question'][:45]}] => {a['answer'][:45]}")

        # 5. persona retrievable by embedding similarity
        pc = await store.persona_chunk_count()
        print(f"[{'PASS' if pc > 0 else 'FAIL'}] persona_chunks = {pc}")
        ok &= pc > 0

        import httpx
        from src.configuration import get_config

        cfg = get_config().embed
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{cfg.url}/embeddings",
                json={"model": cfg.model, "input": ["What is Harsh's portfolio website?"]},
            )
            emb = r.json()["data"][0]["embedding"]
        hits = await store.search_similar_persona(emb, top_k=5)
        texts = [h.get("content", "") for h in hits]
        top_text = texts[0][:120] if texts else ""
        print(f"[{'PASS' if texts else 'FAIL'}] retrieval top hit: {top_text}")
        ok &= bool(texts)

        # The portfolio identity chunk should rank highly.
        portfolio_hit = any("portfolio" in t.lower() or "example.com" in t.lower() for t in texts)
        print(f"[{'PASS' if portfolio_hit else 'WARN'}] portfolio mentioned in top-5 retrieval")
    finally:
        await store.close()

    # 6. /persona renders
    persona_render = format_persona()
    print(f"[{'PASS' if PORTFOLIO in persona_render else 'FAIL'}] /persona shows portfolio")
    ok &= PORTFOLIO in persona_render

    print(f"\n{outcome['result']}")
    print("\n=== VERDICT:", "ALL PASS" if ok else "FAILURES PRESENT", "===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(verify())
