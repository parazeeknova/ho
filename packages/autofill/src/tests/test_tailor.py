"""Unit tests for the lightweight JD-tailored LaTeX resume tailorer."""

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import autofill.src.filling.tailor as tailor_mod
from autofill.src.filling.tailor import (
    _extract_keywords,
    _match_score,
    _mirror,
    _parse_units,
    tailor_enabled,
    tailor_tex,
)

_BASE_TEX = """\
%-------------------------
% Resume - Test Candidate
%------------------------

\\documentclass[letterpaper,11pt]{article}

\\begin{document}

%-----------HEADING----------
\\begin{center}
    \\textbf{\\Huge \\scshape Test Candidate} \\\\ \\vspace{1pt}
    \\href{mailto:test@example.com}{\\underline{test@example.com}}
\\end{center}

%-----------TECHNICAL SKILLS-----------
\\section{Technical Skills}
 \\begin{itemize}[leftmargin=0.15in, label={}, itemsep=0pt, topsep=0pt]
    \\small{\\item{
     \\textbf{Languages}{: JavaScript, Python, Rust} \\\\
     \\textbf{Frontend}{: React.js, Tailwind CSS, HTML} \\\\
     \\textbf{Backend \\& Cloud}{: Node.js, PostgreSQL, Redis, AWS} \\\\
    }}
 \\end{itemize}

%-----------EXPERIENCE-----------
\\section{Experience}
  \\resumeSubHeadingListStart
    \\resumeSubheading
      {Alpha Co}{Jan 2024 - Present}
      {Full Stack Developer}{Remote}

      \\resumeItemListStart
    \\resumeItem{Built payment APIs with Node.js and PostgreSQL, cutting latency by 40\\%}
    \\resumeItem{Wrote React.js dashboards used by 500 customers}
\\resumeItemListEnd
  \\resumeSubHeadingListEnd

%-----------PROJECTS-----------
\\section{Projects}
    \\resumeSubHeadingListStart
      \\resumeProjectHeading
          {\\textbf{AlphaDash} $|$ \\emph{React.js, PostgreSQL}}{\\href{https://example.com}{link}}
          \\resumeItemListStart
                \\resumeItem{Built a React.js dashboard for real-time analytics}
                \\resumeItem{Shipped a Redis cache layer}
            \\resumeItemListEnd

      \\resumeProjectHeading
          {\\textbf{PaySvc} $|$ \\emph{Node.js, PostgreSQL}}{\\href{https://example.com/pay}{link}}
          \\resumeItemListStart
                \\resumeItem{Built a Node.js payment service}
                \\resumeItem{Modeled the PostgreSQL schema for 1M rows}
            \\resumeItemListEnd
    \\resumeSubHeadingListEnd

%-----------ACHIEVEMENTS-----------
\\section{Achievements}
    \\begin{itemize}[leftmargin=0.15in, itemsep=0pt, topsep=0pt]
      \\small{
        \\item \\textbf{1st Place - Example Hackathon}
      }
    \\end{itemize}

\\end{document}
"""


def _write_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tex = tmp_path / "resume.tex"
    tex.write_text(_BASE_TEX)
    monkeypatch.setattr(tailor_mod, "_DEFAULT_TEX_PATH", tex)
    return tex


def _jd(desc: str, title: str = "Full Stack Developer") -> dict:
    return {"title": title, "company": "Acme", "description": desc}


def test_extract_keywords_from_skills_section() -> None:
    skills = _extract_keywords(_BASE_TEX)
    assert "JavaScript" in skills
    assert "PostgreSQL" in skills
    assert "React.js" in skills


def test_match_score_counts_keywords() -> None:
    text = "Built React.js dashboards with PostgreSQL"
    assert _match_score(text, ["react.js", "postgresql"]) == 2
    assert _match_score("built with rust", ["python"]) == 0


def test_mirror_react_to_react_js() -> None:
    assert _mirror("Built with React", ["React.js"]) == "Built with React.js"


def test_tailor_tex_unchanged_without_keywords() -> None:
    assert tailor_tex(_BASE_TEX, []) == _BASE_TEX


def test_tailor_tex_reorders_skills_group_order() -> None:
    # Backend & Cloud heavily emphasized -> its group should lead, and within
    # the group Node.js/PostgreSQL lead.
    tailored = tailor_tex(
        _BASE_TEX, ["Node.js", "PostgreSQL", "Redis", "AWS", "Rust", "Python"]
    )
    langs_idx = tailored.find("\\textbf{Languages}")
    backend_idx = tailored.find("\\textbf{Backend")
    frontend_idx = tailored.find("\\textbf{Frontend}")
    assert backend_idx < langs_idx < frontend_idx
    # Within backend group, Node.js and PostgreSQL appear before bun/other.
    backend_line = next(
        line for line in tailored.splitlines() if "\\textbf{Backend" in line
    )
    assert backend_line.index("Node.js") < backend_line.index("Redis")


def test_tailor_tex_reorders_projects_by_relevance() -> None:
    # A JD about Node.js/PostgreSQL payments should surface PaySvc before
    # AlphaDash.
    tailored = tailor_tex(
        _BASE_TEX,
        ["Node.js", "PostgreSQL", "payment", "api", "schema"],
    )
    pay_idx = tailored.find("PaySvc")
    dash_idx = tailored.find("AlphaDash")
    assert pay_idx != -1 and dash_idx != -1
    assert pay_idx < dash_idx


def test_tailor_tex_reorders_bullets_within_unit() -> None:
    tailored = tailor_tex(
        _BASE_TEX, ["React.js", "dashboard", "500", "customers"]
    )
    # Within the Experience unit, the React dashboard bullet should lead.
    exp_unit = tailored[tailored.find("\\section{Experience}") :]
    first_item = exp_unit.find("\\resumeItem{Built payment APIs")
    second_item = exp_unit.find("\\resumeItem{Wrote React.js dashboards")
    assert first_item != -1 and second_item != -1
    assert second_item < first_item


def test_tailor_tex_never_adds_unowned_skill() -> None:
    # A JD demanding Kubernetes must NOT inject Kubernetes into the resume.
    tailored = tailor_tex(_BASE_TEX, ["Kubernetes", "Docker", "Node.js"])
    assert "Kubernetes" not in tailored
    assert "Docker" not in tailored


def test_tailor_tex_preserves_structure() -> None:
    for keywords in (
        ["React.js", "Node.js"],
        ["Rust"],
        [],
    ):
        tailored = tailor_tex(_BASE_TEX, keywords)
        for token in (
            "resumeSubHeadingListStart",
            "resumeSubHeadingListEnd",
            "resumeItemListStart",
            "resumeItemListEnd",
            "begin{itemize}",
            "end{itemize}",
            "resumeSubheading",
            "resumeProjectHeading",
            "resumeItem{",
            "\\section{Technical Skills}",
            "\\section{Experience}",
            "\\section{Projects}",
            "\\section{Achievements}",
        ):
            assert _BASE_TEX.count(token) == tailored.count(token), token


def test_tailor_tex_falls_back_on_parse_error() -> None:
    broken = "\\section{Technical Skills}\n\\resumeSubheading\nunterminated"
    assert tailor_tex(broken, ["Node.js"]) == broken


def test_parse_units() -> None:
    units = _parse_units(_BASE_TEX)
    assert any(u["items"] for u in units)


def test_tex_source_uses_repo_root_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tex = tmp_path / "resume.tex"
    tex.write_text(_BASE_TEX)
    monkeypatch.setattr(tailor_mod, "_DEFAULT_TEX_PATH", tex)
    monkeypatch.delenv("RESUME_TEX_PATH", raising=False)
    monkeypatch.delenv("RESUME_TEX_URL", raising=False)
    assert tailor_mod._tex_source() == tex


def test_tex_source_uses_env_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tex = tmp_path / "custom.tex"
    tex.write_text(_BASE_TEX)
    monkeypatch.setenv("RESUME_TEX_PATH", str(tex))
    assert tailor_mod._tex_source() == tex


@pytest.mark.asyncio
async def test_tex_source_downloads_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import urllib.request

    monkeypatch.setenv("RESUME_TEX_URL", "https://example.com/resume.tex")
    monkeypatch.setattr(tailor_mod, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(tailor_mod, "_DEFAULT_TEX_PATH", tmp_path / "nope.tex")

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"\\section{X}"

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    result = tailor_mod._tex_source()
    assert result is not None
    assert result.read_text() == "\\section{X}"


def test_tex_source_url_download_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import urllib.request

    monkeypatch.setenv("RESUME_TEX_URL", "https://example.com/resume.tex")
    monkeypatch.setattr(tailor_mod, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(tailor_mod, "_DEFAULT_TEX_PATH", tmp_path / "nope.tex")
    cache_dir = tmp_path / "packages" / "node" / "artifacts" / "resume-tex"
    cache_dir.mkdir(parents=True)
    key = hashlib.sha1(b"https://example.com/resume.tex").hexdigest()[:12]
    (cache_dir / f"{key}.tex").write_text("cached")

    calls = []

    def _urlopen(*a, **k):
        calls.append(1)
        raise AssertionError("should not download")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    result = tailor_mod._tex_source()
    assert result is not None
    assert result.read_text() == "cached"
    assert calls == []


@pytest.mark.asyncio
async def test_tailor_enabled_false_when_tex_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tailor_mod, "_DEFAULT_TEX_PATH", Path("/nonexistent/resume.tex"))
    monkeypatch.delenv("RESUME_TEX_PATH", raising=False)
    monkeypatch.setattr(tailor_mod.shutil, "which", lambda _: "/usr/bin/tectonic")
    assert tailor_enabled() is False


@pytest.mark.asyncio
async def test_tailor_enabled_false_without_tectonic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_base(tmp_path, monkeypatch)
    monkeypatch.setattr(tailor_mod.shutil, "which", lambda _: None)
    assert tailor_enabled() is False


@pytest.mark.asyncio
async def test_tailor_resume_for_job_returns_none_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_base(tmp_path, monkeypatch)
    monkeypatch.setattr(tailor_mod.shutil, "which", lambda _: None)
    cm = AsyncMock()
    result = await tailor_mod.tailor_resume_for_job("job-1", _jd("react and node"), cm)
    assert result is None


@pytest.mark.asyncio
async def test_tailor_resume_for_job_compiles_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_base(tmp_path, monkeypatch)
    monkeypatch.setattr(tailor_mod.shutil, "which", lambda _: "/usr/bin/tectonic")
    monkeypatch.setattr(tailor_mod, "_ARTIFACTS_ROOT", tmp_path / "artifacts")

    async def fake_extract(jd, cm, skills):
        return ["React.js", "Node.js", "PostgreSQL"]

    cm = AsyncMock()
    cm.chat = AsyncMock(return_value="[\"React.js\", \"Node.js\", \"PostgreSQL\"]")

    # Stub the actual tectonic subprocess so the test never invokes a compiler;
    # the fake proc writes a PDF next to the .tex that tectonic would produce.
    with patch.object(tailor_mod.asyncio, "create_subprocess_exec") as exec_mock:
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        exec_mock.return_value = proc

        async def _fake_communicate():
            artifacts = tmp_path / "artifacts"
            for tex in artifacts.rglob("*.tex"):
                tex.with_suffix(".pdf").write_bytes(b"%PDF-fake")
            return b"", b""

        proc.communicate = _fake_communicate

        pdf = await tailor_mod.tailor_resume_for_job(
            "job-1", _jd("react and node"), cm, extractor=fake_extract
        )

    assert pdf is not None
    assert Path(pdf).exists()
    exec_mock.assert_awaited_once()
