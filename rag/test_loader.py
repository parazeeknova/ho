"""Tests for resume loader: PDF extraction, chunking."""

from rag.loader import chunk_resume


class TestChunkResume:
    def test_empty_text(self) -> None:
        result = chunk_resume("")
        # Empty text creates empty header section — that's fine
        assert "header" in result or result == {}

    def test_skills_section(self) -> None:
        text = "SKILLS\nPython\nDjango\nReact\nPostgreSQL"
        result = chunk_resume(text)
        assert "skills" in result
        assert "Python" in result["skills"]

    def test_experience_section(self) -> None:
        text = "EXPERIENCE\nSoftware Engineer at Google\nBuilt APIs with Python"
        result = chunk_resume(text)
        assert "experience" in result
        assert "Google" in result["experience"]

    def test_education_section(self) -> None:
        text = "EDUCATION\nBachelor of Science in CS\nUniversity of California"
        result = chunk_resume(text)
        assert "education" in result
        assert "Bachelor" in result["education"]

    def test_projects_section(self) -> None:
        text = "PROJECTS\nBuilt a RAG pipeline\nDeployed on AWS"
        result = chunk_resume(text)
        assert "projects" in result
        assert "RAG" in result["projects"]

    def test_mixed_sections(self) -> None:
        text = (
            "SKILLS\nPython, Go, Rust\n\n"
            "EXPERIENCE\nIntern at StartupCo\nBuilt CI/CD pipeline\n\n"
            "EDUCATION\nBS Computer Science\nGPA 3.8"
        )
        result = chunk_resume(text)
        assert "skills" in result
        assert "experience" in result
        assert "education" in result
        assert "Python" in result["skills"]
        assert "StartupCo" in result["experience"]

    def test_header_fallback(self) -> None:
        text = "Parazeek Nova\nparazeek@email.com\nGitHub: github.com/para"
        result = chunk_resume(text)
        assert "header" in result
        assert "parazeek" in result["header"].lower()

    def test_section_header_detection(self) -> None:
        text = "TECHNICAL SKILLS\npython\nAWS\n\nWORK EXPERIENCE\nSoftware Dev"
        result = chunk_resume(text)
        assert "skills" in result
        assert "experience" in result
        assert "python" in result["skills"].lower()
        assert "Software Dev" in result["experience"]
