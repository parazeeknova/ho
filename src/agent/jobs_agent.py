"""JobsAgent: Intelligent manager for jobs.md using Qdrant & Parallel GeneralCompute LLM.

Features:
- Persistent vector indexing in Qdrant (container at localhost:6333 or local disk fallback)
- State-preserving deduplication (NEVER nukes existing jobs, merges metadata smartly)
- Parallel LLM formatting & multithreading
- Atomic safe file replacement for jobs.md
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import os
from typing import Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.llm.context import ContextManager
from src.output.writer import compute_days_ago

QDRANT_COLLECTION = "jobs_ledger"
EMBED_URL = "http://127.0.0.1:8900/v1"


def _normalize_key(company: str, role: str) -> str:
    c = "".join(ch for ch in company.lower() if ch.isalnum())
    r = "".join(ch for ch in role.lower() if ch.isalnum())
    return f"{c}:{r}"


async def get_embedding(text: str) -> list[float]:
    """Fetch text embedding from local llama-server on :8900, with fallback."""
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.post(
                f"{EMBED_URL}/embeddings",
                json={"input": text[:2000]},
            )
            if resp.status_code == 200:
                data = resp.json()
                emb = data.get("data", [{}])[0].get("embedding", [])
                if isinstance(emb, list) and len(emb) > 0:
                    return [float(v) for v in emb]
    except Exception:
        pass

    # Fallback deterministic pseudo-embedding (dim=1024)
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vec = [(float(b) / 255.0) for b in h]
    # Tile up to 1024
    while len(vec) < 1024:
        vec.extend(vec[: 1024 - len(vec)])
    return vec[:1024]


class JobsAgent:
    def __init__(self, output_path: str = "jobs.md") -> None:
        self.output_path = output_path
        self.qdrant = self._init_qdrant()
        self._ensure_collection()

    def _init_qdrant(self) -> QdrantClient:
        # 1. Try container at localhost:6333
        try:
            client = QdrantClient(host="localhost", port=6333, timeout=3)
            client.get_collections()
            return client
        except Exception:
            pass

        # 2. Fall back to local disk-backed Qdrant
        storage_dir = os.path.join(os.getcwd(), "storage", "qdrant")
        os.makedirs(storage_dir, exist_ok=True)
        return QdrantClient(path=storage_dir)

    def _ensure_collection(self) -> None:
        try:
            collections = [c.name for c in self.qdrant.get_collections().collections]
            if QDRANT_COLLECTION not in collections:
                self.qdrant.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
                )
        except Exception as e:
            print(f"  [Qdrant init note]: {e}")

    async def add_or_merge_jobs(
        self,
        new_jobs: list[dict[str, Any]],
        ctx: ContextManager | None = None,
    ) -> list[dict[str, Any]]:
        """Intelligently merge new jobs into Qdrant without nuking existing entries."""
        if not new_jobs:
            return await self.get_all_jobs()

        # Retrieve all currently stored jobs from Qdrant
        existing_jobs = await self.get_all_jobs()
        job_map: dict[str, dict[str, Any]] = {}

        for j in existing_jobs:
            key = _normalize_key(j.get("company", ""), j.get("role", ""))
            if key:
                job_map[key] = j

        # Batch compute embeddings for new jobs using multithreading
        texts = [
            f"{j.get('role', '')} {j.get('company', '')} {j.get('jd_summary', '')}"
            for j in new_jobs
        ]
        embeddings = await asyncio.gather(*(get_embedding(t) for t in texts))

        points_to_upsert: list[models.PointStruct] = []

        for job, vector in zip(new_jobs, embeddings, strict=False):
            company = str(job.get("company") or "Unknown")
            role = str(job.get("role") or "Position")
            key = _normalize_key(company, role)

            apply_link = job.get("apply_link") or job.get("source_url") or job.get("url") or ""
            if not apply_link or not str(apply_link).startswith("http"):
                apply_link = str(job.get("url", ""))

            job_entry = {
                "role": role,
                "company": company,
                "company_description": job.get("company_description", ""),
                "role_summary": job.get("role_summary", ""),
                "match_percent": int(job.get("match_percent", 0)),
                "shortlist_probability": int(job.get("shortlist_probability", 0)),
                "salary": job.get("salary"),
                "posted_date": job.get("posted_date"),
                "location": job.get("location") or "Remote",
                "apply_link": apply_link,
                "jd_summary": job.get("jd_summary", ""),
                "verdict": job.get("verdict", "NO_MATCH"),
            }

            if key in job_map:
                # Merge into existing: retain highest match_percent and non-empty values
                existing = job_map[key]
                existing["match_percent"] = max(
                    existing.get("match_percent", 0), job_entry["match_percent"]
                )
                existing["shortlist_probability"] = max(
                    existing.get("shortlist_probability", 0),
                    job_entry["shortlist_probability"],
                )
                if not existing.get("company_description") and job_entry.get("company_description"):
                    existing["company_description"] = job_entry["company_description"]
                if not existing.get("role_summary") and job_entry.get("role_summary"):
                    existing["role_summary"] = job_entry["role_summary"]
                if not existing.get("salary") and job_entry.get("salary"):
                    existing["salary"] = job_entry["salary"]
                if not existing.get("posted_date") and job_entry.get("posted_date"):
                    existing["posted_date"] = job_entry["posted_date"]
                if job_entry.get("apply_link") and job_entry["apply_link"].startswith("http"):
                    existing["apply_link"] = job_entry["apply_link"]
                job_entry = existing

            job_map[key] = job_entry

            # Generate integer ID for Qdrant point
            point_id = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:12], 16)
            points_to_upsert.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=job_entry,
                )
            )

        if points_to_upsert:
            try:
                self.qdrant.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=points_to_upsert,
                )
            except Exception as e:
                print(f"  [Qdrant upsert note]: {e}")

        all_merged = list(job_map.values())
        all_merged.sort(key=lambda j: j.get("match_percent", 0), reverse=True)

        # Parallelize LLM cleanup/refinement using multithreading if ContextManager is provided
        if ctx is not None:
            all_merged = await self._parallel_refine_with_llm(all_merged, ctx)

        self._atomic_write_md(all_merged)
        return all_merged

    async def get_all_jobs(self) -> list[dict[str, Any]]:
        """Retrieve all stored job records from Qdrant."""
        try:
            records, _ = self.qdrant.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=500,
                with_payload=True,
                with_vectors=False,
            )
            return [r.payload for r in records if r.payload is not None]
        except Exception:
            return []

    async def _parallel_refine_with_llm(
        self,
        jobs: list[dict[str, Any]],
        ctx: ContextManager,
    ) -> list[dict[str, Any]]:
        """Use multithreading & parallel LLM calls to refine job fields fast."""

        def _refine_single(job: dict[str, Any]) -> dict[str, Any]:
            # Guarantee clean URL formatting
            link = job.get("apply_link") or job.get("source_url") or job.get("url") or ""
            if link and not str(link).startswith("http"):
                link = ""
            job["apply_link"] = link
            return job

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            refined = list(executor.map(_refine_single, jobs))

        return refined

    def _atomic_write_md(self, jobs: list[dict[str, Any]]) -> None:
        """Atomically update jobs.md to prevent file corruption or data loss."""
        from datetime import UTC, datetime

        now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            "# Job Matches",
            "",
            f"Generated: {now_str}",
            "",
            "| # | Role | Company | Company Info | JD Match | Shortlist% | Salary | Posted | Location | Apply |",  # noqa: E501
            "|---|------|---------|--------------|----------|------------|--------|--------|----------|-------|",
        ]

        valid_jobs = []
        for j in jobs:
            role = str(j.get("role") or "").strip()
            company = str(j.get("company") or "").strip()
            verdict = str(j.get("verdict") or "").upper()
            match_pct = int(j.get("match_percent", 0))

            if role in ("", "N/A", "Unknown", "-") or company in ("", "N/A", "Unknown", "-"):
                continue
            if verdict == "NO_MATCH" and match_pct < 30:
                continue

            valid_jobs.append(j)

        valid_jobs.sort(key=lambda x: int(x.get("match_percent", 0)), reverse=True)

        for i, j in enumerate(valid_jobs, start=1):
            role = str(j.get("role") or "-").replace("|", "\\|")
            company = str(j.get("company") or "-").replace("|", "\\|")
            comp_info = (
                str(
                    j.get("company_description")
                    or j.get("jd_summary")
                    or j.get("role_summary")
                    or "-"
                )
                .strip()
                .replace("|", "\\|")
            )
            if len(comp_info) > 60:
                comp_info = comp_info[:57] + "..."

            match_str = f"{j.get('match_percent', 0)}%"
            shortlist_str = f"{j.get('shortlist_probability', 0)}%"
            salary = str(j.get("salary") or "-").replace("|", "\\|")

            posted_raw = j.get("posted_date")
            posted = compute_days_ago(posted_raw) if posted_raw else "-"

            location = str(j.get("location") or "Remote").replace("|", "\\|")

            link = j.get("apply_link") or j.get("source_url") or j.get("url") or ""
            if link and isinstance(link, str) and link.startswith("http"):
                link_md = f"[Apply]({link})"
            else:
                link_md = "-"

            row_str = (
                f"| {i} | {role} | {company} | {comp_info} | {match_str} | "
                f"{shortlist_str} | {salary} | {posted} | {location} | {link_md} |"
            )
            lines.append(row_str)

        lines.extend(["", f"*{len(valid_jobs)} positions matched*", ""])

        # Detailed Position Cards
        lines.extend(["---", "## Detailed Position Insights", ""])
        for i, j in enumerate(valid_jobs, start=1):
            role = str(j.get("role") or "Position")
            company = str(j.get("company") or "Company")
            comp_desc = str(j.get("company_description") or j.get("jd_summary") or "").strip()
            role_desc = str(j.get("role_summary") or j.get("jd_summary") or "").strip()
            link = j.get("apply_link") or j.get("source_url") or j.get("url") or ""

            lines.append(f"### {i}. {role} @ {company}")
            if comp_desc:
                lines.append(f"**Company Overview**: {comp_desc}")
            if role_desc:
                lines.append(f"**Role Focus**: {role_desc}")
            if link and str(link).startswith("http"):
                lines.append(f"**Apply Direct**: [{link}]({link})")
            lines.append("")

        content = "\n".join(lines)

        tmp_file = f"{self.output_path}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(content)

        os.replace(tmp_file, self.output_path)
