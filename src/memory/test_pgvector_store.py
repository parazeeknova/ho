"""Tests for pgvector_store domain discovery methods (mocked asyncpg)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.pgvector_store import CREATE_TABLES_SQL, MemoryStore


class TestDiscoveredDomainsSQL:
    """Verify the discovered_domains table DDL is present."""

    def test_table_in_create_sql(self) -> None:
        assert "discovered_domains" in CREATE_TABLES_SQL
        assert "domain" in CREATE_TABLES_SQL
        assert "crawled" in CREATE_TABLES_SQL
        assert "discovered_at" in CREATE_TABLES_SQL


class TestAddDiscoveredDomain:
    """Tests for add_discovered_domain with mocked pool."""

    @pytest.mark.asyncio
    async def test_inserts_new_domain(self) -> None:
        store = await _mock_store(execute_return="INSERT 0 1")
        added = await store.add_discovered_domain(
            "boards.greenhouse.io/acmecorp",
            "https://boards.greenhouse.io/acmecorp/jobs/1",
        )
        assert added is True

    @pytest.mark.asyncio
    async def test_on_conflict_do_nothing(self) -> None:
        store = await _mock_store(execute_return="INSERT 0 0")
        added = await store.add_discovered_domain("jobs.lever.co/stripe")
        assert added is False


class TestGetUncrawledDomains:
    """Tests for get_uncrawled_domains with mocked pool."""

    @pytest.mark.asyncio
    async def test_returns_domain_list(self) -> None:
        rows = [{"domain": "boards.greenhouse.io/acme"}, {"domain": "jobs.lever.co/xyz"}]
        store = await _mock_store(fetch_return=rows)
        domains = await store.get_uncrawled_domains(limit=10)
        assert domains == ["boards.greenhouse.io/acme", "jobs.lever.co/xyz"]

    @pytest.mark.asyncio
    async def test_empty_when_none_found(self) -> None:
        store = await _mock_store(fetch_return=[])
        domains = await store.get_uncrawled_domains()
        assert domains == []


class TestMarkDomainsCrawled:
    """Tests for mark_domains_crawled with mocked pool."""

    @pytest.mark.asyncio
    async def test_marks_domains(self) -> None:
        store = await _mock_store()
        await store.mark_domains_crawled(["boards.greenhouse.io/acme"])
        # No exception = success

    @pytest.mark.asyncio
    async def test_empty_list_noop(self) -> None:
        store = await _mock_store()
        await store.mark_domains_crawled([])
        # No exception = success


# ── helpers ────────────────────────────────────────────────────────────


async def _mock_store(
    *,
    execute_return: str = "INSERT 0 1",
    fetch_return: list[dict] | None = None,
) -> MemoryStore:
    """Create a MemoryStore backed by a fully mocked asyncpg pool."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=execute_return)
    if fetch_return is not None:
        mock_conn.fetch = AsyncMock(return_value=fetch_return)

    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)),
        patch("pgvector.asyncpg.register_vector", new=AsyncMock()),
    ):
        store = await MemoryStore.create()
        store._pool = mock_pool  # override with our controlled mock
    return store
