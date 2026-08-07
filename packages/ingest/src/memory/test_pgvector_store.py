"""Tests for pgvector_store domain discovery methods (mocked asyncpg)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pgvector import Vector

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


# Helpers


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
        mock_conn.fetchrow = AsyncMock(return_value=fetch_return[0] if fetch_return else None)

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


class TestEmbedCache:
    """Tests for the content-hash-keyed embedding cache."""

    def test_embed_cache_ddl_present(self) -> None:
        assert "embed_cache" in CREATE_TABLES_SQL
        assert "text_hash" in CREATE_TABLES_SQL
        assert "embedding" in CREATE_TABLES_SQL

    def test_resume_embeddings_has_content_hash(self) -> None:
        assert "content_hash" in CREATE_TABLES_SQL

    def test_obs_embeddings_has_content_hash(self) -> None:
        assert "obs_embeddings" in CREATE_TABLES_SQL

    async def test_get_cached_embedding_hit(self) -> None:
        store = await _mock_store()
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"embedding": Vector([0.1, 0.2, 0.3])})
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        store._pool = mock_pool

        emb = await store.get_cached_embedding("abc123")
        assert emb == pytest.approx([0.1, 0.2, 0.3])

    async def test_get_cached_embedding_miss(self) -> None:
        store = await _mock_store()
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        store._pool = mock_pool

        assert await store.get_cached_embedding("missing") is None

    async def test_put_cached_embedding_executes_upsert(self) -> None:
        store = await _mock_store()
        executed: list[tuple] = []

        async def _execute(sql: str, *args: object) -> str:
            executed.append((sql, args))
            return "INSERT 0 1"

        mock_conn = AsyncMock()
        mock_conn.execute = _execute
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        store._pool = mock_pool

        await store.put_cached_embedding("abc", [0.1, 0.2])
        assert executed, "expected an upsert against embed_cache"
        assert "embed_cache" in executed[0][0]
        assert executed[0][1][0] == "abc"

    async def test_index_resume_chunks_prunes_stale_and_upserts(self) -> None:
        store = await _mock_store()
        mock_conn = AsyncMock()
        mock_conn.transaction = MagicMock(return_value=AsyncMock())  # async ctx mgr
        mock_conn.fetch = AsyncMock(
            return_value=[
                {"content_hash": "stale1"},
                {"content_hash": "stale2"},
            ]
        )
        executed: list[tuple] = []

        async def _execute(sql: str, *args: object) -> str:
            executed.append((sql, args))
            return "INSERT 0 1"

        mock_conn.execute = _execute
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        store._pool = mock_pool

        chunks = [
            {
                "section": "skills",
                "content": "Python",
                "content_hash": "fresh1",
                "embedding": [0.1, 0.2],
            }
        ]
        await store.index_resume_chunks(chunks, current_hashes={"fresh1"})

        sql = "\n".join(sql for sql, _ in executed)
        delete_args = [args for sql, args in executed if "DELETE" in sql]
        insert_args = [args for sql, args in executed if "INSERT" in sql]
        assert delete_args and "stale1" in delete_args[0][0] and "stale2" in delete_args[0][0]
        assert "ON CONFLICT (content_hash)" in sql
        assert insert_args and "fresh1" in insert_args[0][2]

    async def test_existing_resume_hashes_returns_matching(self) -> None:
        store = await _mock_store(fetch_return=[{"content_hash": "a"}, {"content_hash": "c"}])
        found = await store.existing_resume_hashes(["a", "b", "c"])
        assert found == {"a", "c"}


class TestCompanyOsintCache:
    """Company OSINT cache keys are normalized to lowercase on read/write."""

    @pytest.mark.asyncio
    async def test_put_normalizes_key_to_lowercase(self) -> None:
        store = await _mock_store()
        await store.put_company_osint("Cloudflare", {"founders": [{"name": "A"}]})
        executed = store._pool.acquire.return_value.__aenter__.return_value.execute.await_args_list
        assert executed and executed[-1].args[1] == "cloudflare"

    @pytest.mark.asyncio
    async def test_get_looks_up_with_lowercase_key(self) -> None:
        store = await _mock_store(
            fetch_return=[{"data": {"founders": [{"name": "A"}]}, "expires_at": 2e12}]
        )
        data = await store.get_company_osint("Cloudflare")
        called = store._pool.acquire.return_value.__aenter__.return_value.fetchrow.await_args
        assert called.args[1] == "cloudflare"
        assert data == {"founders": [{"name": "A"}]}

    @pytest.mark.asyncio
    async def test_get_expired_returns_none(self) -> None:
        store = await _mock_store(fetch_return=[{"data": {"founders": []}, "expires_at": 1.0}])
        data = await store.get_company_osint("Cloudflare")
        assert data is None
