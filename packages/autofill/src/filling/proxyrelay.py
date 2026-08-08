"""Local per-job HTTP proxy that injects residential-provider credentials.

Chrome drops credentials embedded in ``--proxy-server=http://user:pass@host:port``
(the connection is silently discarded, so the page never loads), and Stagehand's
local launcher ignores the ``proxy.username/password`` fields. A residential
provider URL therefore cannot be handed to Chrome as-is.

This module runs a tiny localhost HTTP proxy per job. Chrome is pointed at
``http://127.0.0.1:<port>`` (no credentials), and the relay rewrites the upstream
request to the real residential gateway, injecting the per-job
``Proxy-Authorization`` header. Each job gets its own ephemeral port, so
concurrent jobs never share a credential context.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
from typing import Any

from src.logging import get_logger

logger = get_logger("autofill.src.filling.proxyrelay")

_CRLF = b"\r\n"


class ProxyRelay:
    """A localhost HTTP proxy that forwards to one upstream proxy with auth."""

    def __init__(
        self,
        username: str,
        password: str,
        upstream_host: str,
        upstream_port: int,
    ) -> None:
        self._auth = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
        self._upstream = (upstream_host, int(upstream_port))
        self._server: asyncio.AbstractServer | None = None
        self._sockets: set[asyncio.StreamWriter] = set()
        self.port = 0

    @property
    def local_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _track(self, w: asyncio.StreamWriter) -> None:
        self._sockets.add(w)

    def _untrack(self, w: asyncio.StreamWriter) -> None:
        self._sockets.discard(w)

    async def start(self) -> str:
        """Bind an ephemeral localhost port and return the client proxy URL."""
        self._server = await asyncio.start_server(self._on_client, "127.0.0.1", 0)
        sock = self._server.sockets[0]
        self.port = int(sock.getsockname()[1])
        logger.info("ProxyRelay listening", port=self.port, upstream=self._upstream)
        return self.local_url

    async def stop(self) -> None:
        """Abort every open socket and close the listener.

        ``stop()`` must never ``wait_closed()`` before aborting: a handler
        blocked on a slow upstream read would otherwise stall shutdown forever.
        Closing every tracked writer unblocks the blocking pumps (EOF), so the
        handler tasks finish promptly.
        """
        for w in list(self._sockets):
            with contextlib.suppress(Exception):
                w.close()
        self._sockets.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    # ── client handling ────────────────────────────────────────────

    async def _on_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._track(writer)
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=30)
            if not request_line:
                writer.close()
                return
            parts = request_line.decode("iso-8859-1").rstrip("\r\n").split(" ")
            if len(parts) < 2:
                writer.close()
                return
            method, target = parts[0], parts[1]
            headers = await self._read_headers(reader)
            if method == "CONNECT":
                await self._handle_connect(reader, writer, target)
            else:
                await self._handle_plain(reader, writer, method, target, headers)
        except asyncio.CancelledError, ConnectionError, OSError:
            pass
        except Exception:
            logger.warning("ProxyRelay client error", exc_info=True)
        finally:
            self._untrack(writer)
            with contextlib.suppress(Exception):
                writer.close()

    async def _read_headers(self, reader: asyncio.StreamReader) -> list[bytes]:
        headers: list[bytes] = []
        while True:
            line = await reader.readline()
            if not line or line in (_CRLF, b"\n"):
                break
            headers.append(line)
        return headers

    # ── CONNECT (HTTPS tunnel) ─────────────────────────────────────

    async def _handle_connect(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        target: str,
    ) -> None:
        upstream_reader, upstream_writer = await asyncio.open_connection(*self._upstream)
        self._track(upstream_writer)
        try:
            req = (
                f"CONNECT {target} HTTP/1.1\r\n"
                f"Host: {target}\r\n"
                f"Proxy-Authorization: {self._auth}\r\n\r\n"
            )
            upstream_writer.write(req.encode("ascii"))
            await upstream_writer.drain()
            resp_line = await asyncio.wait_for(upstream_reader.readline(), timeout=30)
            await self._read_headers(upstream_reader)
            if not resp_line.startswith(b"HTTP/1.1 200"):
                logger.warning("Upstream CONNECT rejected", target=target, resp=resp_line[:80])
                return
            client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await client_writer.drain()
            await self._relay(client_reader, client_writer, upstream_reader, upstream_writer)
        finally:
            self._untrack(upstream_writer)
            try:
                upstream_writer.close()
                await upstream_writer.wait_closed()
            except Exception:
                pass

    # ── Plain HTTP (absolute-form requests) ────────────────────────

    async def _handle_plain(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        method: str,
        target: str,
        headers: list[bytes],
    ) -> None:
        upstream_reader, upstream_writer = await asyncio.open_connection(*self._upstream)
        self._track(upstream_writer)
        try:
            lines = [f"{method} {target} HTTP/1.1\r\n".encode("iso-8859-1")]
            for h in headers:
                low = h.lower()
                if low.startswith(b"proxy-") or low.startswith(b"connection"):
                    continue
                lines.append(h)
            lines.append(f"Proxy-Authorization: {self._auth}\r\n".encode("ascii"))
            lines.append(b"Connection: close\r\n\r\n")
            upstream_writer.write(b"".join(lines))
            await upstream_writer.drain()
            await self._relay(client_reader, client_writer, upstream_reader, upstream_writer)
        finally:
            self._untrack(upstream_writer)
            try:
                upstream_writer.close()
                await upstream_writer.wait_closed()
            except Exception:
                pass

    # ── byte relay ─────────────────────────────────────────────────

    @staticmethod
    async def _relay(
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        upstream_reader: asyncio.StreamReader,
        upstream_writer: asyncio.StreamWriter,
    ) -> None:
        """Relay bytes both ways until either direction closes.

        Two pump tasks (client<->upstream) race; when the first finishes the
        other is cancelled and both sockets are closed, so a half-open tunnel
        can never leak a connection.
        """

        async def pump(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
            try:
                while True:
                    data = await src.read(65536)
                    if not data:
                        break
                    dst.write(data)
                    await dst.drain()
            except ConnectionError, OSError, asyncio.CancelledError:
                pass

        async def close_writer(dst: asyncio.StreamWriter) -> None:
            with contextlib.suppress(Exception):
                dst.close()

        task = asyncio.create_task(pump(client_reader, upstream_writer))
        try:
            await pump(upstream_reader, client_writer)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await close_writer(client_writer)
            await close_writer(upstream_writer)


def parse_template_url(template: str) -> dict[str, Any]:
    """Split a proxy URL template into auth + upstream parts.

    ``http://<user>:{password}@host:port`` -> ``{"username", "password",
    "host", "port"}``. The template may contain ``{SID}`` — callers substitute
    it BEFORE calling this.
    """
    from urllib.parse import urlsplit

    parts = urlsplit(template)
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported proxy scheme: {parts.scheme!r}")
    if not parts.hostname:
        raise ValueError("Proxy URL missing host")
    username = parts.username or ""
    password = parts.password or ""
    port = parts.port or 80
    return {
        "username": username,
        "password": password,
        "host": parts.hostname,
        "port": port,
    }
