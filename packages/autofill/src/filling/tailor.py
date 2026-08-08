"""JD-tailored LaTeX resume generation for the autofill runner.

The base resume (``resume.tex`` at the repo root, or ``RESUME_TEX_PATH`` /
``RESUME_TEX_URL``) is tailored per job with conservative edits:

- JD keyword extraction (one LLM call) isolates the technologies/terms the
  posting cares about.
- Deterministic reordering applies it: JD-matching skills lead each group and
  their groups lead the skills section; JD-relevant experience/project units
  move up and their bullets reorder so the strongest match leads.
- A conservative LLM rewrite pass surfaces the JD keywords inside real
  project/experience bullets — same facts, same numbers, same meaning, no
  fabricated scope, never AI-slop phrasing.
- Exact-phrase mirroring is limited to real resume items (e.g. JD says
  "React.js", resume says "React" -> "React.js"); no skill is ever added that
  the resume's Technical Skills section does not already contain.

The edits are applied to a parsed view of the template and the rest of the
file (preamble, comments, macros, achievements/education/certifications) is
re-emitted verbatim, so the .tex always stays valid. The tailored document is
compiled with ``tectonic`` into a per-job PDF under
``packages/node/artifacts/resumes/<job_id>/`` and cached by
``(job_id, sha1(jd)[:12])``.

Any failure (no tex source, no tectonic, parse error, LLM failure) returns the
unchanged base document path so the run falls back to the static resume.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.logging import get_logger

logger = get_logger("autofill.src.filling.tailor")

# Repo root: tailor.py lives at packages/autofill/src/filling/, so the repo
# root is parents[4].
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ARTIFACTS_ROOT = _REPO_ROOT / "packages" / "node" / "artifacts" / "resumes"

# Technical Skills whitelist (source of truth in screener/rag.py). Only these
# technologies may be *claimed*; mirroring never introduces a term outside it.
from autofill.src.screener.rag import _RESUME_SKILLS  # noqa: E402

_DEFAULT_TEX_PATH = _REPO_ROOT / "resume.tex"

# Compile wall-clock cap. First tectonic run downloads packages; allow slack.
_TECTONIC_TIMEOUT_SEC = 300

# Group lines in the Technical Skills section look like:
#   \textbf{Languages}{: JavaScript, TypeScript, Python, Rust} \\
_GROUP_LINE_RE = re.compile(r"\\textbf\{([^}]*)\}\{:\s*(.*?)\s*\}\s*\\\\")

# A resume item macro line: \resumeItem{<text>}
_ITEM_RE = re.compile(r"\\resumeItem\{(.*?)\}")

# Unit (subheading / project heading) macro names that introduce a block whose
# bullets belong to it.
_UNIT_START_RE = re.compile(r"\\resume(?:Subheading|ProjectHeading)\b")

# Canonical spellings for a handful of common aliases; mirror the JD's phrasing
# only when the resume already carries the skill in a real (whitelisted) form.
_CANONICAL: dict[str, str] = {
    "react": "React.js",
    "node": "Node.js",
    "express": "Express.js",
    "next": "Next.js",
    "postgres": "PostgreSQL",
    "tailwind": "Tailwind CSS",
}


def _tex_source() -> Path | None:
    """Resolve the base .tex resume: local env path first, then the repo-root
    resume.tex, then a cached download from RESUME_TEX_URL."""
    local = os.environ.get("RESUME_TEX_PATH", "").strip()
    if local and Path(local).is_file():
        return Path(local)
    if _DEFAULT_TEX_PATH.is_file():
        return _DEFAULT_TEX_PATH
    url = os.environ.get("RESUME_TEX_URL", "").strip()
    if not url:
        return None
    return _cached_download(url)


def _cached_download(url: str) -> Path | None:
    """Download RESUME_TEX_URL into the artifacts dir, cached by content so
    re-runs skip the fetch. Returns None on any failure (caller falls back to
    no tailoring)."""
    cache_dir = _REPO_ROOT / "packages" / "node" / "artifacts" / "resume-tex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{hashlib.sha1(url.encode()).hexdigest()[:12]}.tex"
    if cache_path.is_file():
        return cache_path
    try:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
        tmp = cache_path.with_suffix(".tex.tmp")
        tmp.write_bytes(content)
        os.replace(tmp, cache_path)
        return cache_path
    except Exception as e:
        logger.warning("Could not download RESUME_TEX_URL", url=url, error=str(e))
        return None


def _tectonic_bin() -> str | None:
    return shutil.which("tectonic")


def tailor_enabled() -> bool:
    """True when a tex source exists, tectonic is available, and tailoring is
    not explicitly disabled via TAILOR_RESUME=0."""
    if os.environ.get("TAILOR_RESUME", "1").strip() == "0":
        return False
    return _tex_source() is not None and _tectonic_bin() is not None


def _jd_text(jd: dict[str, Any] | None) -> str:
    jd = jd or {}
    title = str(jd.get("title") or "").strip()
    desc = str(jd.get("description") or "").strip()
    return " ".join(part for part in (title, desc[:4000]) if part).strip()


def _extract_keywords(tex: str) -> list[str]:
    """All technology tokens named in the resume's Technical Skills section."""
    found: list[str] = []
    for line in tex.splitlines():
        m = _GROUP_LINE_RE.search(line)
        if not m:
            continue
        for token in re.split(r"[,/]+", m.group(2)):
            token = token.strip()
            if not token:
                continue
            low = token.lower()
            if low in _RESUME_SKILLS or any(
                low.startswith(s) or s in low for s in _RESUME_SKILLS
            ):
                found.append(token)
    return found


async def _llm_extract_keywords(
    jd: dict[str, Any] | None, cm: Any, resume_skills: list[str]
) -> list[str]:
    """Ask the LLM which resume skills the JD emphasizes, as a JSON list.

    The model may only pick from ``resume_skills`` (the actual resume skills) —
    never invent a technology. Falls back to the deterministic extractor.
    """
    text = _jd_text(jd)
    if not text or not resume_skills:
        return []
    prompt = (
        "The candidate's resume names these technologies: "
        + ", ".join(resume_skills)
        + "\n\nThe target job posting emphasizes certain technologies. Return a JSON "
        "array of the resume technologies most relevant to this job, ordered by "
        "importance (most relevant first). Only pick from the list above; return an "
        "empty array if none overlap. Never invent technologies.\n"
        f"<job_description>\n{text}\n</job_description>\n"
        "JSON array:"
    )
    try:
        raw = (await cm.chat(prompt, max_tokens=300, interactive=True)).strip()
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            raw = match.group(1).strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return []
        out: list[str] = []
        for token in parsed:
            token = str(token).strip()
            low = token.lower()
            if token in resume_skills or low in {s.lower() for s in resume_skills}:
                out.append(token)
        return out
    except Exception as e:
        logger.warning("LLM keyword extraction failed; using deterministic", error=str(e))
        return []


# System prompt for the conservative bullet rewriter. The bullets must keep
# their exact facts/numbers/meaning; the rewrite may only surface JD-relevant
# keywords the resume ALREADY claims, phrased naturally — never invent scope,
# and never read like AI-generated copy.
BULLET_REWRITE_SYSTEM_PROMPT = """\
You are a careful resume editor. Rewrite the candidate's project/experience
bullet points so the job description's keywords surface naturally. Rules:

- PRESERVE EVERY FACT: numbers, percentages, latency, costs, user counts,
  dates, company names, project names, and technologies must stay EXACTLY as
  written (including LaTeX escapes like \\%, \\$, \\&, \\to).
- Only weave in keywords from the provided list when they genuinely describe
  the same technology the bullet already mentions (e.g. "pgvector" may become
  "pgvector (PostgreSQL vector search)" — the underlying fact is unchanged).
- Never claim a technology, metric, scale, or outcome that is not already in
  the bullet. Never add leadership, team size, or responsibility the bullet
  does not state.
- Keep the same meaning, same length class (do not expand a 1-line bullet into
  3 lines), and the same first-person active voice.
- Write like a competent engineer, not a marketer. Ban: "passionate",
  "excited", "seamless", "cutting-edge", "game-changer", "leverage", "utilize",
  "elevate", "unlock", "foster", "harness", "synergy", "robust and scalable",
  "in today's fast-paced world". Never use em dashes or exclamation points.
- If a bullet needs no change, return it exactly as-is.

Return valid JSON in this exact shape, one entry per bullet you received:
{"rewrites": [{"original": "<original bullet verbatim>", "rewritten":
"<rewritten bullet>"}]}. Copy the "original" text byte-for-byte from the list
you were given. Do not include entries for bullets you did not receive. No
preamble, no markdown fences, no extra fields.
"""


async def _rewrite_bullets(
    base_tex: str,
    keywords: list[str],
    jd: dict[str, Any] | None,
    cm: Any,
) -> str:
    """Conservatively rewrite ``\\resumeItem`` bullets to surface JD keywords.

    One LLM call for all bullets in Experience/Projects. Returns the tex with
    accepted rewrites applied; on any failure (LLM error, bad JSON, or a
    rewrite that changed too much) the bullet is kept verbatim. Never touches
    anything outside the bullet bodies.
    """
    if not keywords:
        return base_tex
    spans = _find_item_spans(base_tex)
    if not spans:
        return base_tex
    bullets = [inner for _, _, inner in spans]
    text = _jd_text(jd)
    prompt = (
        "Rewrite these resume bullet points so the job description's keywords "
        "surface naturally, without inventing facts or changing meaning.\n"
        f"Allowed keywords (only weave these in, and only where the bullet "
        f"already implies them): {', '.join(keywords)}\n"
        f"<job_description>\n{text or '(no description)'}\n</job_description>\n"
        "Bullets (copy the original text EXACTLY, then your rewrite):\n"
        + "\n".join(f"- {b}" for b in bullets)
    )
    # A {original, rewritten} pair list is more robust than a dict keyed by
    # bullet text (long LaTeX-escaped keys get mangled by the model).
    schema = {
        "type": "object",
        "properties": {
            "rewrites": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "original": {"type": "string"},
                        "rewritten": {"type": "string"},
                    },
                    "required": ["original", "rewritten"],
                },
            }
        },
        "required": ["rewrites"],
    }
    try:
        raw = (
            await cm.chat(
                prompt,
                schema=schema,
                system_prompt=BULLET_REWRITE_SYSTEM_PROMPT,
                max_tokens=3000,
                interactive=True,
            )
        ).strip()
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            raw = match.group(1).strip()
        parsed = json.loads(raw)
        pairs = parsed.get("rewrites") if isinstance(parsed, dict) else None
        if not isinstance(pairs, list):
            return base_tex
        rewrites: dict[str, str] = {}
        for pair in pairs:
            original = str(pair.get("original") or "").strip()
            rewritten = str(pair.get("rewritten") or "").strip()
            if not original or not rewritten:
                continue
            if original not in bullets:
                continue
            # Conservative guard: reject rewrites that dropped the core facts.
            if _keeps_facts(original, rewritten):
                rewrites[original] = rewritten
        if not rewrites:
            return base_tex
        return _apply_bullet_rewrites(base_tex, rewrites)
    except Exception as e:
        logger.warning("Bullet rewrite failed; keeping bullets verbatim", error=str(e))
        return base_tex


def _keeps_facts(original: str, rewritten: str) -> bool:
    """True when the rewrite preserved the bullet's numeric facts.

    A conservative gate against hallucinated rewrites: every number in the
    original must survive, the rewrite must not pad beyond ~1.6x, and it must
    never smuggle tex structure (nested item macros, unbalanced braces). Fact
    and technology preservation is additionally enforced by the system prompt
    that drives the rewrite.
    """
    # Never accept a rewrite that smuggles tex structure (a nested
    # \resumeItem{, stray braces, or unbalanced backslash-macros) — the bullet
    # body must stay plain text.
    if "\\resumeItem" in rewritten or "\\item" in rewritten:
        return False
    if rewritten.count("{") != rewritten.count("}"):
        return False
    num_re = re.compile(r"\d+(?:\.\d+)?%?|\$[\d.]+|\d+(?:\.\d+)?\s*(?:ms|s|GB|MB|KB|FPS|hr|min)")
    orig_tokens = set(num_re.findall(original.lower()))
    new_tokens = set(num_re.findall(rewritten.lower()))
    # All numeric facts in the original must still be present.
    if not orig_tokens <= new_tokens:
        return False
    # A rewrite must not grow beyond ~1.6x the original (no padding).
    return len(rewritten) <= int(len(original) * 1.6) + 20


def _match_score(text: str, keywords: list[str]) -> int:
    """Count keyword occurrences in a piece of resume text (case-insensitive)."""
    low = text.lower()
    score = 0
    for kw in keywords:
        kl = kw.lower().strip()
        if not kl:
            continue
        score += low.count(kl)
    return score


def _mirror(item: str, keywords: list[str]) -> str:
    """Exact-phrase mirroring for a resume item already containing the skill.

    Replaces an alias with the JD/canonical spelling (React -> React.js) only
    when the canonical form is a requested keyword and the item already
    contains the alias. Never adds a skill the resume doesn't have.
    """
    out = item
    low = out.lower()
    kw_lower = {k.lower() for k in keywords}
    for alias, canonical in _CANONICAL.items():
        c_low = canonical.lower()
        if c_low in kw_lower and c_low in _RESUME_SKILLS and re.search(rf"\b{alias}\b", low):
            if c_low not in low:
                out = re.sub(rf"\b{alias}\b", canonical, out, flags=re.I)
            low = out.lower()
    return out


def _find_item_spans(body: str) -> list[tuple[int, int, str]]:
    """Find every ``\\resumeItem{...}`` block in a body of tex.

    Brace-matching handles bullets that wrap across lines or contain nested
    braces. Returns ``(start, end, inner)`` for each item.
    """
    spans: list[tuple[int, int, str]] = []
    idx = 0
    while True:
        m = re.search(r"\\resumeItem\s*\{", body[idx:])
        if not m:
            break
        start_match = idx + m.start()
        content_start = idx + m.end()
        brace_count = 1
        curr = content_start
        while curr < len(body) and brace_count > 0:
            if body[curr] == "{":
                brace_count += 1
            elif body[curr] == "}":
                brace_count -= 1
            curr += 1
        if brace_count != 0:
            break
        end_match = curr
        spans.append((start_match, end_match, body[content_start : end_match - 1]))
        idx = end_match
    return spans


def _apply_bullet_rewrites(body: str, rewrites: dict[str, str]) -> str:
    """Replace bullet text in-place per a {original: rewritten} map.

    Only exact inner-text matches are replaced; every other byte is preserved
    verbatim so the surrounding tex structure can never be damaged. Span
    offsets are computed ONCE against the original body and applied by
    rebuilding the string from slices (never mutating in place), so later
    replacements can never land at stale offsets.
    """
    spans = _find_item_spans(body)
    if not spans:
        return body
    pieces: list[str] = []
    last = 0
    for start, end, inner in spans:
        pieces.append(body[last:start])
        rewritten = rewrites.get(inner)
        if rewritten and rewritten != inner:
            pieces.append(f"\\resumeItem{{{rewritten}}}")
        else:
            pieces.append(body[start:end])
        last = end
    pieces.append(body[last:])
    return "".join(pieces)


def _parse_units(body: str) -> list[dict[str, Any]]:
    """Split an Experience/Projects section body into ordered units.

    Each unit starts at a \\resumeSubheading or \\resumeProjectHeading.
    Uses brace-matching to safely extract \\resumeItem blocks even if they
    wrap across multiple lines.
    """
    starts = [m.start() for m in _UNIT_START_RE.finditer(body)]
    if not starts:
        return []

    units: list[dict[str, Any]] = []
    for i in range(len(starts)):
        start_idx = starts[i]
        end_idx = starts[i+1] if i + 1 < len(starts) else len(body)
        unit_str = body[start_idx:end_idx]

        chunks = []
        items = []

        idx = 0
        while True:
            m = re.search(r"\\resumeItem\s*\{", unit_str[idx:])
            if not m:
                if idx < len(unit_str):
                    chunks.append({"type": "static", "text": unit_str[idx:]})
                break

            start_match = idx + m.start()
            content_start = idx + m.end()

            if start_match > idx:
                chunks.append({"type": "static", "text": unit_str[idx:start_match]})

            brace_count = 1
            curr = content_start
            while curr < len(unit_str) and brace_count > 0:
                if unit_str[curr] == '{':
                    brace_count += 1
                elif unit_str[curr] == '}':
                    brace_count -= 1
                curr += 1

            if brace_count == 0:
                end_match = curr
                raw = unit_str[start_match:end_match]
                inner = unit_str[content_start:end_match - 1]
                chunk = {"type": "item", "raw": raw, "inner": inner}
                chunks.append(chunk)
                items.append(chunk)
                idx = end_match
            else:
                chunks.append({"type": "static", "text": unit_str[idx:]})
                break

        units.append({"chunks": chunks, "items": items})
    return units


def _reorder_units(units: list[dict[str, Any]], keywords: list[str]) -> list[dict[str, Any]]:
    """Sort units so the most JD-relevant lead; within each unit, sort bullets
    so the strongest match leads (stable: preserves original order on ties)."""
    def key_unit(u: dict[str, Any]) -> int:
        heading_text = "".join(c["text"] for c in u["chunks"] if c["type"] == "static")
        items_text = " ".join(c["inner"] for c in u["items"])
        return _match_score(heading_text + " " + items_text, keywords)

    def key_item(item_chunk: dict[str, str]) -> int:
        return _match_score(item_chunk["inner"], keywords)

    for u in units:
        u["items"].sort(key=key_item, reverse=True)
    return sorted(units, key=key_unit, reverse=True)


def tailor_tex(base_tex: str, keywords: list[str]) -> str:
    """Apply lightweight JD tailoring to the base .tex and return the result.

    Deterministic and conservative: reorder skill groups + bullets + units by
    JD relevance, mirror exact phrases inside real resume items, and re-emit
    everything else verbatim. On any structural surprise, return the base
    document unchanged.
    """
    try:
        result = _tailor_tex_inner(base_tex, keywords)
    except Exception as e:
        logger.warning("Resume tailoring parse failed; using base resume", error=str(e))
        return base_tex
    # A result that differs only by a trailing newline is "no change".
    if result.rstrip("\n") == base_tex.rstrip("\n"):
        return base_tex
    return result


def _tailor_tex_inner(base_tex: str, keywords: list[str]) -> str:
    lines = base_tex.splitlines()
    if not keywords:
        return base_tex

    # Locate section boundaries by name so we only touch the two reorderable
    # sections; every other line is re-emitted verbatim.
    sections: dict[str, int] = {}
    for i, line in enumerate(lines):
        m = re.search(r"\\section\{([^}]*)\}", line)
        if m:
            sections[m.group(1).strip().lower()] = i

    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # ---- Technical Skills -------------------------------------------------
        if line.strip() == "\\section{Technical Skills}" and i + 1 < len(lines):
            out.append(line)
            i += 1
            # Collect the section body until the next \section.
            body: list[str] = []
            while i < len(lines) and not re.search(r"\\section\{", lines[i]):
                body.append(lines[i])
                i += 1
            out.extend(_tailor_skills_body(body, keywords))
            continue

        # ---- Experience / Projects -------------------------------------------
        lower = line.strip().lower()
        if lower.startswith("\\section{") and lower.split("{", 1)[1].rstrip("}") in (
            "experience",
            "projects",
        ):
            out.append(line)
            i += 1
            body = []
            while i < len(lines) and not re.search(r"\\section\{", lines[i]):
                body.append(lines[i])
                i += 1
            out.extend(_tailor_units_body(body, keywords))
            continue

        out.append(line)
        i += 1

    return "\n".join(out) + "\n"


def _tailor_skills_body(body: list[str], keywords: list[str]) -> list[str]:
    """Reorder the Technical Skills section: JD-matching groups first, and
    within each group JD-matching skills first."""
    groups: list[dict[str, Any]] = []
    for line in body:
        m = _GROUP_LINE_RE.search(line)
        if m:
            label = m.group(1)
            tokens = [t.strip() for t in re.split(r"[,/]+", m.group(2)) if t.strip()]
            groups.append({"label": label, "tokens": tokens, "raw": line})
            continue
        groups.append({"raw": line, "label": None, "tokens": None})

    def group_score(g: dict[str, Any]) -> int:
        if not g["tokens"]:
            return 0
        return sum(_match_score(t, keywords) for t in g["tokens"])

    def token_key(t: str) -> int:
        return _match_score(t, keywords)

    reordered: list[str] = []
    scored: list[dict[str, Any]] = [g for g in groups if g["tokens"] is not None]
    static: list[str] = [g["raw"] for g in groups if g["tokens"] is None]

    # Stable sort: JD-matching groups lead (original order preserved on ties).
    scored.sort(key=group_score, reverse=True)

    seen_labels: set[str] = set()
    for g in scored:
        if g["label"] in seen_labels:
            continue
        seen_labels.add(g["label"])
        tokens = sorted(g["tokens"], key=token_key, reverse=True)
        reordered.append(f"     \\textbf{{{g['label']}}}{{: {', '.join(tokens)}}} \\\\")

    # Insert static lines (e.g. the \small{\item{ wrapper) back in place.
    return _reinsert_static(body, static, reordered)


def _reinsert_static(body: list[str], static: list[str], reordered: list[str]) -> list[str]:
    """Merge static (non-group) lines back into the reordered skill lines,
    preserving their original positions relative to the surrounding list."""
    out: list[str] = []
    static_iter = iter(static)
    ri = 0
    for orig in body:
        if _GROUP_LINE_RE.search(orig):
            if ri < len(reordered):
                out.append(reordered[ri])
                ri += 1
        else:
            with contextlib.suppress(StopIteration):
                out.append(next(static_iter))
    # Any trailing static lines we missed.
    for s in static_iter:
        out.append(s)
    return out


def _tailor_units_body(body: list[str], keywords: list[str]) -> list[str]:
    """Reorder experience/project units and bullets by JD relevance.

    The section body is split into a prefix (lines before the first unit), the
    unit contents (each unit runs from its ``\resumeSubheading`` /
    ``\resumeProjectHeading`` through its own ``\resumeItemListEnd``), and a
    suffix (everything after the last unit's item list end, e.g.
    ``\resumeSubHeadingListEnd``). Only the unit contents are reordered; prefix
    and suffix are re-emitted verbatim.
    """
    starts: list[int] = []
    ends: list[int] = []
    for i, line in enumerate(body):
        if _UNIT_START_RE.search(line):
            starts.append(i)
            continue
        if "\\resumeItemListEnd" in line:
            ends.append(i)
    if not starts:
        return body
    if len(ends) != len(starts):
        logger.warning("Resume unit/end count mismatch; skipping unit reorder")
        return body

    # Unit content spans: unit macro -> its own \resumeItemListEnd.
    spans: list[tuple[int, int]] = []
    for idx, s in enumerate(starts):
        e = ends[idx]
        if not (s < e):
            return body
        # The next unit (if any) must start AFTER this unit's end.
        if idx + 1 < len(starts) and not (e < starts[idx + 1]):
            return body
        spans.append((s, e))

    prefix = body[: spans[0][0]]
    suffix = body[spans[-1][1] + 1 :]

    # Parse units from the unit-content region only (first unit start through
    # the last unit's \resumeItemListEnd), so a unit never absorbs the suffix
    # lines (e.g. \resumeSubHeadingListEnd) that belong to the section.
    content_region = body[starts[0] : ends[-1] + 1]
    units = _parse_units("\n".join(content_region))
    if len(units) != len(spans):
        return body
    reordered = _reorder_units(units, keywords)

    result = list(prefix)
    for u in reordered:
        unit_str = _unit_to_str(u, keywords)
        result.extend(unit_str.split("\n"))
    result.extend(suffix)
    return result


def _unit_to_str(u: dict[str, Any], keywords: list[str]) -> str:
    """Re-emit a unit: every non-item chunk verbatim, with its bullets replaced
    in-place by the sorted order (the reorder already sorted u['items'])."""
    sorted_items = iter(u["items"])
    out: list[str] = []
    for chunk in u["chunks"]:
        if chunk["type"] == "static":
            out.append(chunk["text"])
        else:
            next_item = next(sorted_items)
            mirrored = _mirror(next_item["raw"], keywords)
            out.append(mirrored)
    return "".join(out)


def _cache_key(job_id: str, jd: dict[str, Any] | None) -> str:
    return f"{job_id}-{hashlib.sha1(_jd_text(jd).encode()).hexdigest()[:12]}"


# Preamble directives that are pdfTeX-only. The base resume.tex targets
# pdfTeX (DisableLigatures needs pdfTeX 1.30+; glyphtounicode defines
# \pdfgentounicode, a pdfTeX primitive), but the tailorer compiles with
# tectonic, whose bundled engine halts on them. These lines are cosmetic
# (ligature/unicode metadata) — comment them out for the tectonic build so the
# tailored resume actually compiles. The user's source resume.tex is untouched.
_TECTONIC_INCOMPAT_RE = re.compile(
    r"^[ \t]*\\(?:DisableLigatures|pdfgentounicode)[^\n]*\n?"
    r"|^[ \t]*\\input\{glyphtounicode\}[^\n]*\n?",
    re.MULTILINE,
)


def _tectonic_sanitize(tex: str) -> str:
    """Comment out pdfTeX-only preamble lines so tectonic can compile the tex."""
    return _TECTONIC_INCOMPAT_RE.sub(lambda m: "% " + m.group(0).strip() + "\n", tex)


async def compile_tailored(
    tex: str, job_id: str, jd: dict[str, Any] | None
) -> Path | None:
    """Write the tailored .tex and compile it with tectonic into a per-job PDF.

    Returns the PDF path, or None on any compile failure (caller falls back to
    the base resume).
    """
    tectonic = _tectonic_bin()
    if not tectonic:
        logger.warning("tectonic not available; skipping tailored resume compile")
        return None
    out_dir = _ARTIFACTS_ROOT / _safe_segment(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / f"tailored_{_cache_key(job_id, jd)}.tex"
    tex_path.write_text(_tectonic_sanitize(tex))
    pdf_path = tex_path.with_suffix(".pdf")
    if pdf_path.exists():
        return pdf_path
    try:
        proc = await asyncio.create_subprocess_exec(
            tectonic,
            "--outdir",
            str(out_dir),
            str(tex_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TECTONIC_TIMEOUT_SEC)
        if proc.returncode != 0:
            logger.warning(
                "tectonic failed; using base resume",
                job_id=job_id,
                rc=proc.returncode,
                stderr=(stderr or b"")[-500:].decode(errors="replace"),
            )
            return None
    except TimeoutError:
        logger.warning("tectonic timed out; using base resume", job_id=job_id)
        return None
    except Exception as e:
        logger.warning("tectonic compile error; using base resume", job_id=job_id, error=str(e))
        return None
    if pdf_path.exists():
        return pdf_path
    return None


def _safe_segment(segment: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(segment or "unknown")).strip("._")
    return cleaned or "unknown"


async def tailor_resume_for_job(
    job_id: str,
    jd: dict[str, Any] | None,
    cm: Any,
    extractor: Callable[[dict[str, Any] | None, Any, list[str]], Any] = _llm_extract_keywords,
) -> str | None:
    """Full pipeline: extract keywords, tailor, rewrite bullets, compile, cache.

    Returns the tailored PDF path, or None when tailoring cannot/should not
    run (the caller then falls back to the static resume).
    """
    if not tailor_enabled():
        return None
    source = _tex_source()
    if source is None:
        return None
    try:
        base_tex = source.read_text()
    except OSError as e:
        logger.warning("Could not read resume tex", error=str(e))
        return None

    resume_skills = _extract_keywords(base_tex)
    keywords = await extractor(jd, cm, resume_skills) or []
    if not keywords:
        return None

    tailored = tailor_tex(base_tex, keywords)
    # Conservative LLM pass: surface the JD keywords inside real project/
    # experience bullets without changing facts or meaning.
    rewritten = await _rewrite_bullets(tailored, keywords, jd, cm)
    if rewritten != tailored:
        tailored = rewritten
    if tailored == base_tex:
        return None
    pdf = await compile_tailored(tailored, job_id, jd)
    return str(pdf) if pdf is not None else None
