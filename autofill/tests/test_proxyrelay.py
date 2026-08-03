"""End-to-end test: ProxyRelay injects creds so Chrome can reach a target
through an upstream HTTP proxy that requires Basic auth."""

import asyncio
import base64
import contextlib
import socket
import threading

import pytest

from autofill.proxyrelay import ProxyRelay, parse_template_url


class AuthProxyServer:
    """Minimal HTTP CONNECT + absolute-form proxy requiring Basic auth."""

    def __init__(self, user: str = "user", password: str = "pass") -> None:
        self.user = user
        self.password = password
        self.saw_auth_headers: list[str] = []
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(16)
        self.port = self.server.getsockname()[1]
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._clients: list[socket.socket] = []
        self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            self._clients.append(conn)
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn) -> None:
        try:
            request_line = b""
            while not request_line.endswith(b"\r\n"):
                chunk = conn.recv(1)
                if not chunk:
                    return
                request_line += chunk
            headers = b""
            while b"\r\n\r\n" not in headers:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                headers += chunk
            auth = b""
            for line in headers.split(b"\r\n"):
                if line.lower().startswith(b"proxy-authorization:"):
                    auth = line.split(b":", 1)[1].strip()
                    self.saw_auth_headers.append(auth.decode())
            expected = base64.b64encode(f"{self.user}:{self.password}".encode()).decode()
            ok = auth.decode() == f"Basic {expected}"
            parts = request_line.decode("iso-8859-1").rstrip("\r\n").split(" ")
            method, target = parts[0], parts[1]
            if not ok:
                conn.sendall(
                    b"HTTP/1.1 407 Proxy Authentication Required\r\nContent-Length: 0\r\n\r\n"
                )
                conn.close()
                return
            if method == "CONNECT":
                conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                self._tunnel(conn, target)
            else:
                host_port = target.split("/")[2]
                host, port = host_port.rsplit(":", 1)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((host, int(port)))
                sock.sendall(
                    request_line
                    + headers.split(b"\r\n\r\n")[0].replace(
                        b"Proxy-Authorization: " + b"Basic " + expected.encode(), b""
                    )
                    + b"\r\n\r\n"
                )
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                sock.close()
                conn.close()
        except Exception:
            conn.close()

    def _tunnel(self, conn, target: str) -> None:
        host, port = target.rsplit(":", 1)
        try:
            upstream = socket.create_connection((host, int(port)), timeout=10)
        except OSError:
            conn.close()
            return

        def forward(src, dst):
            try:
                while True:
                    data = src.recv(65536)
                    if not data:
                        break
                    dst.sendall(data)
            except OSError:
                pass
            finally:
                with contextlib.suppress(OSError):
                    dst.close()

        t = threading.Thread(target=forward, args=(conn, upstream), daemon=True)
        t.start()
        forward(upstream, conn)

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.server.close()
        for c in self._clients:
            with contextlib.suppress(OSError):
                c.close()


@pytest.mark.asyncio
async def test_relay_injects_auth_and_tunnels_https() -> None:
    # A local echo server stands in for the HTTPS origin so the tunnel test
    # needs no external network (both relay directions are still exercised).
    echo = await asyncio.start_server(lambda r, w: w.write(b"hello-from-origin"), "127.0.0.1", 0)
    echo_port = echo.sockets[0].getsockname()[1]
    upstream = AuthProxyServer()
    relay = ProxyRelay("user", "pass", "127.0.0.1", upstream.port)
    try:
        local_url = await relay.start()
        assert local_url.startswith("http://127.0.0.1:")
        reader, writer = await asyncio.open_connection("127.0.0.1", relay.port)
        writer.write(f"CONNECT 127.0.0.1:{echo_port} HTTP/1.1\r\n".encode())
        writer.write(f"Host: 127.0.0.1:{echo_port}\r\n\r\n".encode())
        await writer.drain()
        resp = await asyncio.wait_for(reader.readline(), timeout=5)
        assert resp.startswith(b"HTTP/1.1 200")
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=5)
            if line in (b"\r\n", b"\n", b""):
                break
        body = await asyncio.wait_for(reader.read(64), timeout=5)
        assert body == b"hello-from-origin"
        assert upstream.saw_auth_headers, "upstream must have seen Proxy-Authorization"
        assert upstream.saw_auth_headers[0] == "Basic " + base64.b64encode(b"user:pass").decode()
        writer.close()
        await writer.wait_closed()
    finally:
        await relay.stop()
        upstream.close()
        echo.close()
        await echo.wait_closed()


@pytest.mark.asyncio
async def test_relay_407_when_wrong_creds() -> None:
    upstream = AuthProxyServer(user="user", password="realpass")
    relay = ProxyRelay("user", "wrongpass", "127.0.0.1", upstream.port)
    try:
        await relay.start()
        reader, writer = await asyncio.open_connection("127.0.0.1", relay.port)
        writer.write(b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n")
        await writer.drain()
        resp = await asyncio.wait_for(reader.readline(), timeout=5)
        # Upstream rejected the CONNECT, so the relay never sends 200 to the client.
        assert not resp.startswith(b"HTTP/1.1 200")
        assert upstream.saw_auth_headers, "upstream saw the injected wrong creds"
        writer.close()
        await writer.wait_closed()
    finally:
        await relay.stop()
        upstream.close()


def test_parse_template_url() -> None:
    parts = parse_template_url("http://user-country-in-session-abc:secret@geo.iproyal.com:12321")
    assert parts["username"] == "user-country-in-session-abc"
    assert parts["password"] == "secret"
    assert parts["host"] == "geo.iproyal.com"
    assert parts["port"] == 12321


def test_parse_template_url_missing_host() -> None:
    with pytest.raises(ValueError):
        parse_template_url("http://user:pass@")
