"""Tests for output writer: markdown table generation."""

import os
import tempfile
from datetime import UTC

from src.output.writer import compute_days_ago, write_md


class TestComputeDaysAgo:
    def test_none(self) -> None:
        assert compute_days_ago(None) == "-"

    def test_empty_string(self) -> None:
        assert compute_days_ago("") == "-"

    def test_today(self) -> None:
        from datetime import datetime

        today = datetime.now(UTC).isoformat()
        assert compute_days_ago(today) == "Today"

    def test_one_day_ago(self) -> None:
        from datetime import datetime, timedelta

        yesterday = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        assert "1d ago" in compute_days_ago(yesterday)

    def test_seven_days_ago(self) -> None:
        from datetime import datetime, timedelta

        week_ago = (datetime.now(UTC) - timedelta(days=7)).isoformat()
        assert "7d ago" in compute_days_ago(week_ago)

    def test_invalid_date(self) -> None:
        assert compute_days_ago("not-a-date") == "not-a-date"


class TestWriteMd:
    def test_empty_jobs(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = f.name
        try:
            write_md([], path)
            with open(path) as f:
                content = f.read()
            assert "Job Matches" in content
            assert "0 positions" in content.lower()
        finally:
            os.unlink(path)

    def test_single_job(self) -> None:
        jobs = [
            {
                "role": "Full Stack Developer",
                "company": "Acme Corp",
                "match_percent": 85,
                "shortlist_probability": 70,
                "salary": "$80K",
                "posted_date": None,
                "location": "Remote",
                "apply_link": "https://apply.example.com",
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = f.name
        try:
            write_md(jobs, path)
            with open(path) as f:
                content = f.read()
            assert "Full Stack Developer" in content
            assert "Acme Corp" in content
            assert "85%" in content
            assert "70%" in content
            assert "Remote" in content
            assert "[Apply]" in content
            assert "1 positions" in content.lower()
        finally:
            os.unlink(path)

    def test_multiple_jobs(self) -> None:
        jobs = [
            {
                "role": f"Role {i}",
                "company": f"Company {i}",
                "match_percent": 90 - i,
                "shortlist_probability": 80 - i,
                "salary": None,
                "posted_date": None,
                "location": "Remote",
                "apply_link": None,
            }
            for i in range(5)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = f.name
        try:
            write_md(jobs, path)
            with open(path) as f:
                content = f.read()
            assert "Role 0" in content
            assert "Role 4" in content
            assert "5 positions" in content.lower()
        finally:
            os.unlink(path)

    def test_no_apply_link(self) -> None:
        jobs = [
            {
                "role": "Dev",
                "company": "Co",
                "match_percent": 50,
                "shortlist_probability": 30,
                "salary": None,
                "posted_date": None,
                "location": "Onsite",
                "apply_link": None,
                "source_url": "",
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = f.name
        try:
            write_md(jobs, path)
            with open(path) as f:
                content = f.read()
            assert "| - |" in content or "|-" in content  # empty link col
        finally:
            os.unlink(path)
