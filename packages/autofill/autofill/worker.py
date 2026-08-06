"""Background worker queue processor for autofill service."""

from __future__ import annotations

import asyncio
import binascii
import contextlib
import datetime as _dt
import json
import os
import random
import re
import secrets
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from src.logging import get_logger
from src.memory.pgvector_store import MemoryStore

from autofill.db import AutofillDB
from autofill.discord import (
    DiscordNotConfiguredError,
    DiscordQuestionBridge,
    DiscordSendError,
)
from autofill.profile import build_profile
from autofill.rag import ScreenerRAG
from autofill.resolve import (
    DEFER_MARKER,
    DeferredError,
    resolve_cover_letter,
    resolve_question,
)
from autofill.resume import resolve_resume_path

if TYPE_CHECKING:
    from autofill.proxyrelay import ProxyRelay

logger = get_logger("autofill.worker")

# Repo root (worker.py lives in autofill/). Used to locate docker-compose.yaml
# for the torproxy lifecycle commands.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_overnight() -> bool:
    """True when running in overnight mode (OVERNIGHT_LOOP=true).

    Overnight there is no human present: unknown screener questions defer the
    job for the morning digest instead of blocking on a Telegram prompt, and
    fully-fillable jobs are submitted automatically.
    """
    return os.getenv("OVERNIGHT_LOOP", "").strip().lower() == "true"


def _next_digest_time(summary_time: str) -> _dt.datetime:
    """Next local-time occurrence of the daily digest hour (e.g. "08:00")."""
    try:
        hh, mm = summary_time.strip().split(":")
        target = _dt.time(int(hh), int(mm))
    except Exception:
        logger.warning("Invalid AUTOFILL_DAILY_SUMMARY, using 08:00", value=summary_time)
        target = _dt.time(8, 0)
    now = _dt.datetime.now()
    candidate = _dt.datetime.combine(now.date(), target)
    if candidate <= now:
        candidate += _dt.timedelta(days=1)
    return candidate


def _autofill_proxy() -> str:
    """SOCKS5 proxy for browser runs (AUTOFILL_PROXY), e.g. socks5://127.0.0.1:9050."""
    return os.getenv("AUTOFILL_PROXY", "").strip()


def _autofill_proxy_template() -> str:
    """Residential proxy template (AUTOFILL_PROXY_TEMPLATE) for browser runs.

    A per-job URL with a ``{SID}`` placeholder for the session id — each job
    substitutes its own random SID so every application egresses from a fresh
    residential IP (the provider's sticky-per-session rotation). Example:
    ``http://<user>-country-in-session-{SID}:<pass>@geo.iproyal.com:12321``
    """
    return os.getenv("AUTOFILL_PROXY_TEMPLATE", "").strip()


def _new_session_id() -> str:
    """Random opaque session id shared by a job's proxy session and fingerprint seed."""
    return secrets.token_hex(8)


def _per_job_proxy(session_id: str) -> str | None:
    """Resolve the proxy URL for one job.

    With a proxy template the ``{SID}`` placeholder is substituted with this
    job's session id (fresh residential IP per application). Without a
    template the legacy static AUTOFILL_PROXY (Tor) is used unchanged. Returns
    None when no proxy is configured at all.
    """
    template = _autofill_proxy_template()
    if template:
        return template.replace("{SID}", session_id)
    static = _autofill_proxy()
    return static or None


async def _start_proxy_relay(template_url: str) -> ProxyRelay | None:
    """Start the per-job localhost relay that injects residential credentials.

    Chrome drops credentials embedded in a proxy URL and Stagehand's local
    launcher ignores the ``proxy.username/password`` fields, so a residential
    template URL cannot be handed to Chrome as-is. This starts a localhost
    relay on an ephemeral port (credential-free), pointed at the residential
    gateway with Basic auth injected per request. Returns None when the relay
    cannot start (caller should fall back to a direct connection).
    """
    from autofill.proxyrelay import ProxyRelay, parse_template_url

    try:
        parts = parse_template_url(template_url)
    except ValueError as e:
        logger.warning("Invalid AUTOFILL_PROXY_TEMPLATE URL", error=str(e))
        return None
    relay = ProxyRelay(
        username=parts["username"],
        password=parts["password"],
        upstream_host=parts["host"],
        upstream_port=parts["port"],
    )
    try:
        await relay.start()
        logger.info("Per-job proxy relay started", port=relay.port)
        return relay
    except Exception as e:
        logger.warning("Could not start per-job proxy relay", error=str(e))
        await relay.stop()
        return None


# Per-job writing-tone seeds. Injected into the prompt grounding so answers and
# cover letters vary in phrasing from application to application (the LLM
# chat() endpoint has no temperature parameter, so variation is prompt-level).
_VOICE_SEEDS = (
    "concise and direct",
    "warm and personable",
    "measured and professional",
    "confident and energetic",
    "clear and matter-of-fact",
)


def _pick_voice() -> str:
    return random.choice(_VOICE_SEEDS)


def _safe_segment(segment: str) -> str:
    """Sanitize a path segment (job id) for use as a directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(segment or "unknown")).strip("._")
    return cleaned or "unknown"


def _per_job_resume(
    resume_path: str | None,
    first_name: str = "",
    last_name: str = "",
    job_id: str = "",
) -> str | None:
    """Point this job's resume at a name-based temp copy.

    Every application otherwise uploads the same generic basename (e.g.
    ``resume.pdf``), a small "identical resume across applications" signal. A
    per-job copy named ``<First>_<Last>_Resume.pdf`` (inside a per-job
    subdirectory so concurrent jobs for the same person never clobber each
    other) varies the uploaded filename without touching the resume content.
    Returns the new path, or the original when nothing needs copying.
    """
    if not resume_path:
        return None
    src = Path(resume_path)
    if not src.exists():
        return resume_path
    name_slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{first_name}_{last_name}").strip("_")
    if not name_slug:
        name_slug = src.stem
    dest_dir = (
        Path(__file__).resolve().parent.parent
        / "node"
        / "artifacts"
        / "resumes"
        / _safe_segment(job_id)
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{name_slug}_Resume{src.suffix}"
    if not dest.exists():
        # Atomic: write to a temp file first so a concurrent reader (or a
        # parallel worker process) never observes a half-written resume.
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)
    return str(dest)


async def _torproxy_ready() -> bool:
    """True when the Tor SOCKS5 port is accepting connections."""
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", 9050), timeout=1.0)
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


async def _ensure_torproxy() -> None:
    """Best-effort start torproxy (docker compose) when AUTOFILL_PROXY is set."""
    if not _autofill_proxy():
        return
    if await _torproxy_ready():
        return
    logger.info("AUTOFILL_PROXY set but torproxy not up; starting it...")
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "compose",
            "-f",
            str(_PROJECT_ROOT / "docker-compose.yaml"),
            "up",
            "-d",
            "torproxy",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
    except Exception as e:
        logger.warning("Could not start torproxy", error=str(e))
    for _ in range(45):
        if await _torproxy_ready():
            logger.info("torproxy SOCKS5 ready on :9050")
            return
        await asyncio.sleep(1)
    logger.warning(
        "torproxy not ready on :9050; browser runs will fail to connect "
        "(start it with: make tor-up)"
    )


async def _read_tor_cookie_hex() -> str:
    """Hex-encoded Tor control cookie from the torproxy container (for NEWNYM)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "compose",
            "-f",
            str(_PROJECT_ROOT / "docker-compose.yaml"),
            "ps",
            "-q",
            "torproxy",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        container = out.decode().strip()
        if not container:
            return ""
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            container,
            "cat",
            "/etc/tor/run/control.authcookie",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        return binascii.hexlify(out).decode()
    except Exception:
        return ""


async def _rotate_tor_circuit() -> None:
    """Request a fresh Tor circuit (NEWNYM) so the next run exits from a new IP.

    Speaks the Tor control protocol directly (the image's `torproxy.sh -n`
    mangles cookie auth), authenticating with the hex-encoded control cookie.
    """
    if not _autofill_proxy():
        return
    try:
        cookie_hex = await _read_tor_cookie_hex()
        if not cookie_hex:
            raise RuntimeError("could not read tor control cookie")
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", 9051), timeout=3.0
        )
        writer.write(f"AUTHENTICATE {cookie_hex}\r\n".encode())
        await writer.drain()
        await asyncio.sleep(0.3)
        auth = (await asyncio.wait_for(reader.read(256), timeout=3.0)).decode(errors="replace")
        if "250 OK" not in auth:
            raise RuntimeError(f"tor control AUTHENTICATE failed: {auth.strip()}")
        writer.write(b"SIGNAL NEWNYM\r\nQUIT\r\n")
        await writer.drain()
        await asyncio.sleep(0.3)
        await asyncio.wait_for(reader.read(256), timeout=3.0)
        writer.close()
        await writer.wait_closed()
        logger.info("Tor circuit rotated (NEWNYM)")
    except Exception as e:
        logger.warning("Tor circuit rotation failed (continuing)", error=str(e))


class AutofillWorker:
    """Async background worker polling PostgreSQL queue and driving Stagehand processes."""

    def __init__(self, db: AutofillDB, max_concurrent: int = 4) -> None:
        self.db = db
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._running = False
        self._running_tasks: set[asyncio.Task] = set()
        self._summary_task: asyncio.Task[None] | None = None
        # Count of jobs this worker has STARTED processing; used to skip the
        # inter-job spacing delay before the very first job of a batch.
        self._jobs_started = 0
        # Debug-run ledger (AUTOFILL_DEBUG_RUN=true): every question + answer
        # for every company attempted is persisted to a JSON file so the batch
        # can be reviewed afterwards. TEMP — only meaningful during the debug
        # batch; no-op when the env flag is unset.
        self._debug_enabled = os.getenv("AUTOFILL_DEBUG_RUN", "").strip().lower() == "true"
        self._debug_path = Path(os.getenv("AUTOFILL_DEBUG_PATH", "logs/debug-run.json"))
        self._debug_lock = asyncio.Lock()
        self._debug_started = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
        # Persistent Chrome profile pool (AUTOFILL_PROFILE_POOL_SIZE, default
        # 4). Each concurrent runner claims its own profile dir and releases it
        # when the job finishes, so profiles accumulate cookies/storage/history
        # like a lived-in browser while concurrent runs never share a dir (a
        # shared user-data-dir would trip Chrome's SingletonLock).
        self._profile_lock = asyncio.Lock()
        self._profile_pool = self._build_profile_pool(max_concurrent)
        self._profile_in_use: set[str] = set()

    @staticmethod
    def _build_profile_pool(min_size: int = 4) -> list[str]:
        """Create (if needed) and return the persistent profile directories.

        Each directory must already exist: chrome-launcher opens
        ``<userDataDir>/chrome-out.log`` during prepare() before Chrome starts,
        so a missing dir aborts the launch. The pool is sized to at least
        ``min_size`` (the worker's concurrency) so a concurrent runner never
        silently falls back to no persistent profile.
        """
        base = Path(__file__).resolve().parent / "node" / "artifacts" / "profiles"
        try:
            pool_size = int(os.getenv("AUTOFILL_PROFILE_POOL_SIZE", "4"))
        except TypeError, ValueError:
            pool_size = 4
        pool_size = max(min_size, pool_size)
        pool_size = max(1, min(pool_size, 16))
        base.mkdir(parents=True, exist_ok=True)
        dirs: list[str] = []
        for i in range(pool_size):
            d = base / f"profile-{i}"
            d.mkdir(parents=True, exist_ok=True)
            dirs.append(str(d))
        return dirs

    async def _acquire_profile(self) -> str | None:
        """Claim a free persistent profile dir, or None when the pool is busy.

        A profile dir killed mid-run keeps a stale ``SingletonLock``/``chrome.pid``
        that makes the next Chrome launch wait forever on a dead browser — clear
        those before handing the dir out.
        """
        async with self._profile_lock:
            for p in self._profile_pool:
                if p not in self._profile_in_use:
                    for name in (
                        "SingletonLock",
                        "SingletonSocket",
                        "SingletonCookie",
                        "chrome.pid",
                    ):
                        stale = Path(p) / name
                        with contextlib.suppress(OSError):
                            stale.unlink()
                    self._profile_in_use.add(p)
                    return p
            return None

    def _release_profile(self, profile: str | None) -> None:
        """Return a claimed profile dir to the pool."""
        if profile:
            self._profile_in_use.discard(profile)

    async def _debug_finalize(
        self, rec: dict[str, Any] | None, status: str, error: str | None = None
    ) -> None:
        """Record a job's terminal outcome in the debug ledger (if enabled)."""
        if not self._debug_enabled or rec is None:
            return
        rec["status"] = status
        rec["error"] = error
        rec["updated_at"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
        async with self._debug_lock:
            try:
                self._debug_path.parent.mkdir(parents=True, exist_ok=True)
                data: dict[str, Any] = {}
                if self._debug_path.exists():
                    try:
                        data = json.loads(self._debug_path.read_text())
                    except OSError, json.JSONDecodeError:
                        data = {}
                jobs = data.setdefault("jobs", {})
                jobs[rec["job_id"]] = rec
                run = data.setdefault("run", {})
                run["name"] = "debug"
                run["mode"] = "autosubmit" if is_overnight() else "review"
                run["started_at"] = self._debug_started
                run["updated_at"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
                counts: dict[str, int] = {}
                for j in jobs.values():
                    s = str(j.get("status") or "?")
                    counts[s] = counts.get(s, 0) + 1
                run["counts"] = counts
                tmp = self._debug_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(data, indent=2) + "\n")
                os.replace(tmp, self._debug_path)
            except Exception as e:
                logger.warning("Debug ledger flush failed", error=str(e))

    async def start(self) -> None:
        """Start the worker polling loop and the daily digest scheduler."""
        self._running = True
        logger.info("AutofillWorker started polling loop...")
        self._summary_task = asyncio.create_task(self._daily_summary_loop())
        try:
            while self._running:
                # The slot is acquired BEFORE claiming and only released when
                # the runner exits, so max_concurrent bounds the number of
                # simultaneous Stagehand processes — not just claims. Any
                # failure in acquire/claim releases the slot and backs off so
                # a single transient DB error cannot kill the whole worker.
                await self.semaphore.acquire()
                try:
                    job = await self.db.claim_next_job(lease_seconds=3600)
                except asyncio.CancelledError:
                    self.semaphore.release()
                    raise
                except Exception as claim_err:
                    self.semaphore.release()
                    logger.warning("Claim failed; backing off", error=str(claim_err))
                    await asyncio.sleep(5)
                    continue
                if not job:
                    self.semaphore.release()
                    await asyncio.sleep(2)
                    continue

                logger.info(
                    "Claimed job for processing", job_id=job["job_id"], link=job["apply_link"]
                )
                task = asyncio.create_task(self._process_job(job))
                self._running_tasks.add(task)
                task.add_done_callback(self._on_task_done)
        except asyncio.CancelledError:
            logger.info("AutofillWorker loop cancelled.")
        finally:
            self.stop()

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        """Release a processing slot once a job's runner has fully exited."""
        self._running_tasks.discard(task)
        self.semaphore.release()

    def stop(self) -> None:
        """Stop the worker loop and cancel active tasks."""
        self._running = False
        if self._summary_task:
            self._summary_task.cancel()
            self._summary_task = None
        for task in list(self._running_tasks):
            if not task.done():
                task.cancel()
        logger.info("AutofillWorker stopped and active tasks cancelled.")

    # ── morning digest ──────────────────────────────────────────────

    async def _daily_summary_loop(self) -> None:
        """Send the daily morning digest of deferred jobs at AUTOFILL_DAILY_SUMMARY."""
        summary_time = os.getenv("AUTOFILL_DAILY_SUMMARY", "08:00")
        bridge = DiscordQuestionBridge()
        while self._running:
            try:
                next_time = _next_digest_time(summary_time)
                delay = (next_time - _dt.datetime.now()).total_seconds()
                logger.info("Morning digest scheduled", at=str(next_time), in_seconds=round(delay))
                await asyncio.sleep(delay)
                if not self._running:
                    return
                await self._send_daily_digest(bridge)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("Morning digest loop error", error=str(e))
                await asyncio.sleep(60)

    async def _send_daily_digest(self, bridge: DiscordQuestionBridge) -> None:
        """Query deferred jobs and send one summary message, if any exist."""
        rows = await self.db.get_pending_summary_jobs()
        if not rows:
            logger.info("Morning digest: no deferred jobs to report")
            return
        if not bridge.is_configured:
            logger.warning("Morning digest skipped: Telegram not configured")
            return
        text = self._format_digest(rows)
        ok = await self._send_chunked(bridge, text)
        if ok:
            await self.db.mark_summary_sent([r["job_id"] for r in rows])
            logger.info("Morning digest sent", count=len(rows))
        else:
            logger.warning("Morning digest send failed; jobs kept for next digest")

    async def send_end_of_run_summary(self) -> None:
        """Send a summary of ALL deferred jobs (with links) when the overnight
        run ends, so the user can resume them the next morning.

        Called from ``run_worker`` after the polling loop stops. Lists every
        deferred job with its posting link regardless of ``summary_sent`` so the
        end-of-run report is always complete.
        """
        if not is_overnight():
            return
        rows = await self.db.get_deferred_jobs()
        if not rows:
            logger.info("End-of-run summary: no deferred jobs to report")
            return
        bridge = DiscordQuestionBridge()
        if not bridge.is_configured:
            logger.warning("End-of-run summary skipped: Telegram not configured")
            return
        text = self._format_digest(rows, title="🏁 Overnight run finished — deferred jobs")
        ok = await self._send_chunked(bridge, text)
        if ok:
            logger.info("End-of-run summary sent", count=len(rows))
        else:
            logger.warning("End-of-run summary send failed", count=len(rows))

    @staticmethod
    async def _send_chunked(bridge: DiscordQuestionBridge, text: str, max_len: int = 3900) -> bool:
        """Send ``text`` to Telegram, splitting it into <= ``max_len`` chunks on
        paragraph boundaries when it exceeds Telegram's 4096-char message cap.
        Returns True only when every chunk was delivered."""
        if not text:
            return True
        if len(text) <= max_len:
            return await bridge.send(text)
        pieces: list[str] = []
        current = ""
        for para in text.split("\n\n"):
            if not para.strip():
                continue
            candidate = (current + "\n\n" + para).strip()
            if current and len(candidate) > max_len:
                pieces.append(current)
                current = para
            else:
                current = candidate
        if current.strip():
            pieces.append(current)
        ok = True
        for piece in pieces:
            if not await bridge.send(piece):
                ok = False
        return ok

    @staticmethod
    def _format_digest(
        rows: list[dict[str, Any]], title: str = "⏰ Morning Digest — jobs deferred for your input"
    ) -> str:
        lines = [f"<b>{title}</b>", ""]
        for i, r in enumerate(rows, 1):
            role = r.get("role") or "Position"
            company = r.get("company") or "Company"
            link = r.get("apply_link") or ""
            questions = r.get("pending_questions") or []
            lines.append(f"<b>{i}. {company}</b> — {role}")
            if link:
                lines.append(f'    <a href="{link}">Open posting →</a>')
            if questions:
                lines.append(f"    Needs input ({len(questions)}):")
                for entry in questions[:6]:
                    if isinstance(entry, str):
                        lines.append(f"    • {entry}")
                    else:
                        lines.append(f"    • {AutofillWorker._format_pending(entry)}")
            lines.append("")
        lines.append("Answer them with <code>python -m autofill resume &lt;job_id&gt;</code>")
        return "\n".join(lines)

    async def _process_job(self, job: dict[str, Any]) -> None:
        job_id = job["job_id"]
        apply_link = job["apply_link"]
        apply_mode = job.get("apply_mode", "review")

        deferred_pending: list[dict[str, Any]] = []
        rag: ScreenerRAG | None = None
        debug_rec: dict[str, Any] | None = None
        proxy_relay: Any = None
        profile_dir: str | None = None
        if self._debug_enabled:
            debug_rec = {
                "job_id": job_id,
                "company": job.get("company"),
                "role": job.get("role"),
                "url": apply_link,
                "status": "in_progress",
                "error": None,
                "job_context": {},
                "questions": [],
                "filled_fields": None,
                "screenshot_path": None,
                "started_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        try:
            try:
                store = await MemoryStore.create()
            except Exception as e:
                logger.warning(
                    "Could not connect to memory store; persona retrieval disabled",
                    error=str(e),
                )
                store = None
            profile = await build_profile(store=store)
            profile.resumePath = _per_job_resume(
                await resolve_resume_path(),
                first_name=profile.firstName,
                last_name=profile.lastName,
                job_id=job_id,
            )
            rag = ScreenerRAG(profile=profile, store=store)
            bridge = DiscordQuestionBridge()
            question_timeout = float(os.getenv("AUTOFILL_QUESTION_TIMEOUT", "300"))
            overnight = is_overnight()
            run_mode = "auto" if overnight else apply_mode
            logger.info("Processing job", job_id=job_id, mode=run_mode, overnight=overnight)
            job_payload = {
                "jobId": job_id,
                "url": apply_link,
                "mode": run_mode,
                "profile": profile.model_dump(by_alias=False),
                # Overnight (no human): fully-fillable jobs are submitted
                # automatically. In day mode the form is filled and verified
                # but never submitted (no-apply phase).
                "submitAllowed": overnight,
            }
            payload_str = json.dumps(job_payload)

            node_dir = os.path.join(os.path.dirname(__file__), "node")

            process_env = {**os.environ}
            # Per-job identity: one session id drives BOTH the residential proxy
            # session (fresh Indian IP) and the browser fingerprint seed (fresh
            # device), so each application is a new device from a new IP — the
            # two strongest ATS fraud signals ("same device", "same IP") are
            # both defeated. A per-job writing-tone seed varies answer phrasing.
            session_id = _new_session_id()
            voice_seed = _pick_voice()
            profile_dir = await self._acquire_profile()
            if profile_dir:
                process_env["AUTOFILL_USER_DATA_DIR"] = profile_dir
                logger.info(
                    "Using persistent browser profile",
                    job_id=job_id,
                    profile=profile_dir,
                )
            if _autofill_proxy_template():
                # Residential pool path: substitute this job's {SID} into the
                # proxy template and seed the browser fingerprint from it. Chrome
                # cannot use creds-in-URL, so a per-job localhost relay injects
                # the Basic auth. Each job gets a fresh IP + device, so the
                # inter-job delay is only a small throughput-throttle.
                proxy = _per_job_proxy(session_id)
                if proxy:
                    proxy_relay = await _start_proxy_relay(proxy)
                    if proxy_relay is not None:
                        process_env["AUTOFILL_PROXY"] = proxy_relay.local_url
                    else:
                        logger.warning(
                            "Proxy relay unavailable; running this job without a proxy",
                            job_id=job_id,
                        )
                process_env["AUTOFILL_FINGERPRINT_SEED"] = session_id
                inter_delay_ms = int(os.getenv("AUTOFILL_INTER_JOB_DELAY_MS", "15000"))
                if inter_delay_ms > 0:
                    await asyncio.sleep(inter_delay_ms / 1000.0)
            elif _autofill_proxy():
                # Legacy Tor path: pause between jobs and request a fresh Tor
                # circuit so each run exits from a new IP (defeats the "many
                # applications from one IP" fraud signal).
                inter_delay_ms = int(os.getenv("AUTOFILL_INTER_JOB_DELAY_MS", "60000"))
                if inter_delay_ms > 0:
                    await asyncio.sleep(inter_delay_ms / 1000.0)
                await _rotate_tor_circuit()
                process_env["AUTOFILL_FINGERPRINT_SEED"] = session_id
            else:
                # Direct/free-IP path (e.g. the host's own residential egress):
                # no proxy rotation exists, so space submissions out so they
                # never arrive as a same-IP burst. Longer when auto-submitting
                # (a human does one application at a time), shorter for review.
                # The delay is skipped for the FIRST job of a worker so a fresh
                # batch starts immediately (it spaces SUBSEQUENT submissions).
                default_delay = "300000" if overnight else "20000"
                inter_delay_ms = int(os.getenv("AUTOFILL_INTER_JOB_DELAY_MS", default_delay))
                if inter_delay_ms > 0 and self._jobs_started > 0:
                    await asyncio.sleep(inter_delay_ms / 1000.0)
                process_env["AUTOFILL_FINGERPRINT_SEED"] = session_id
            # Mark this job as started AFTER the inter-job spacing decision, so
            # the first job of a batch skips the delay and subsequent jobs are
            # spaced out. (Incremented once per processed job.)
            self._jobs_started += 1
            if overnight:
                # No human inspects the browser in the overnight worker: hold the
                # review step only briefly so the queue never stalls. In day mode
                # a short review hold (AUTOFILL_REVIEW_HOLD_MS) applies so a
                # human has a moment to review the filled form without stalling
                # the queue on batch runs.
                process_env["AUTOFILL_REVIEW_HOLD_MS"] = "3000"
                # Overnight safety net: if the browser hangs with no observable
                # progress (stuck fill/submit), the runner aborts after
                # AUTOFILL_ACTIVITY_TIMEOUT_MS instead of holding a worker slot
                # until the hour-long DB lease. Any runner/status/RPC activity
                # resets the idle timer, so a healthy run is never cut off.
                # TEMP (batch-run 2026-08-02): 5-minute absolute life for the
                # 100-job trial. REVERT to "360000" after the batch run.
                process_env["AUTOFILL_ACTIVITY_TIMEOUT_MS"] = "300000"
            else:
                # Day mode (no auto-submit): the filled form stays open for
                # human review, bounded by AUTOFILL_REVIEW_HOLD_MS (default 2
                # min; a manual-submit session sets it much higher so the user
                # has time to click Submit themselves). The activity timeout is
                # an absolute-life safety net so a stuck runner never holds a
                # worker slot; it resets on any runner/status/RPC activity, so
                # a healthy run is never cut off.
                hold = os.getenv("AUTOFILL_REVIEW_HOLD_MS", "120000")
                process_env["AUTOFILL_REVIEW_HOLD_MS"] = hold
                process_env["AUTOFILL_ACTIVITY_TIMEOUT_MS"] = os.getenv(
                    "AUTOFILL_ACTIVITY_TIMEOUT_MS", "300000"
                )

            process = await asyncio.create_subprocess_exec(
                "npx",
                "tsx",
                "runner.ts",
                cwd=node_dir,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=process_env,
            )

            if process.stdin:
                process.stdin.write(f"{payload_str}\n".encode())
                await process.stdin.drain()

            stderr_task = asyncio.create_task(self._read_stderr(process.stderr, job_id))

            # True once a terminal status event (submitted/skipped/failed) was
            # handled for this run, so the exit-code fallback below never
            # clobbers a specific error (e.g. CAPTCHA_DETECTED) with a generic
            # "Runner exited with code N" message.
            terminal_seen = False

            # Job description context extracted by the Node adapter; arrives
            # via the job_context RPC before any question is resolved. The
            # per-job writing-tone seed rides along so answers and cover
            # letters vary in phrasing between applications.
            job_context: dict[str, Any] = {"voice": voice_seed}

            # Monitor stdout for status events and RPC requests
            while True:
                line = await process.stdout.readline() if process.stdout else b""
                if not line:
                    break

                line_str = line.decode("utf-8").strip()
                if line_str.startswith("RPC_REQUEST:"):
                    raw_rpc = line_str[len("RPC_REQUEST:") :]
                    try:
                        rpc_req = json.loads(raw_rpc)
                        req_id = rpc_req.get("id")
                        method = rpc_req.get("method")
                        args = rpc_req.get("args", {})

                        logger.info(
                            "Received RPC request from Node runner",
                            job_id=job_id,
                            method=method,
                            req_id=req_id,
                        )

                        if method == "job_context":
                            # JD extracted by the Node adapter (title, company,
                            # location, description). Stored for the rest of the
                            # run; open-ended answers and scoped questions use it.
                            # The voice seed is preserved across the merge.
                            job_context = {
                                **job_context,
                                **{str(k): v for k, v in (args or {}).items()},
                            }
                            logger.info(
                                "Received job context from runner",
                                job_id=job_id,
                                title=job_context.get("title"),
                                location=job_context.get("location"),
                            )
                            if debug_rec is not None:
                                debug_rec["job_context"] = job_context
                            if process.stdin and not process.stdin.is_closing():
                                rpc_resp = json.dumps(
                                    {"type": "RPC_RESPONSE", "id": req_id, "result": {"ok": True}}
                                )
                                process.stdin.write(f"{rpc_resp}\n".encode())
                                await process.stdin.drain()
                            continue

                        if method == "cover_letter":
                            answer, source = await resolve_cover_letter(
                                rag, job_context=job_context
                            )
                            await self._record_fill(
                                job_id, "cover letter", answer or None, source=source
                            )

                            pdf_path = None
                            if answer:
                                try:
                                    from autofill.pdf import create_cover_letter_pdf

                                    pdf_path = create_cover_letter_pdf(
                                        answer,
                                        first_name=profile.firstName,
                                        last_name=profile.lastName,
                                        job_id=job_id,
                                    )
                                except Exception as pdf_err:  # never fail the RPC over a PDF
                                    logger.warning(
                                        "Cover letter PDF generation failed; falling back to text",
                                        job_id=job_id,
                                        error=str(pdf_err),
                                    )
                                    pdf_path = None

                            if process.stdin and not process.stdin.is_closing():
                                rpc_resp = json.dumps(
                                    {
                                        "type": "RPC_RESPONSE",
                                        "id": req_id,
                                        "result": {
                                            "answer": answer,
                                            "source": source,
                                            "pdf_path": pdf_path,
                                        },
                                    }
                                )
                                process.stdin.write(f"{rpc_resp}\n".encode())
                                await process.stdin.drain()
                            continue

                        if method == "answer_question":
                            question = str(args.get("question", "")).strip()
                            kind = str(args.get("kind", "text"))
                            options = [str(o) for o in (args.get("options") or [])]

                            # Never fabricate personal facts: unknown questions
                            # are answered by the user via Telegram (day) or
                            # defer the job for the morning digest (overnight).
                            try:
                                answer, source = await resolve_question(
                                    rag,
                                    bridge,
                                    question,
                                    kind=kind,
                                    options=options,
                                    overnight=overnight,
                                    timeout=question_timeout,
                                    job_context=job_context,
                                    required=bool(args.get("required", True)),
                                )
                            except DiscordNotConfiguredError as tg_err:
                                logger.error(
                                    "Cannot answer question without Telegram",
                                    job_id=job_id,
                                    error=str(tg_err),
                                )
                                if process.stdin and not process.stdin.is_closing():
                                    rpc_resp = json.dumps(
                                        {
                                            "type": "RPC_RESPONSE",
                                            "id": req_id,
                                            "error": str(tg_err),
                                        }
                                    )
                                    process.stdin.write(f"{rpc_resp}\n".encode())
                                    await process.stdin.drain()
                                continue
                            except DiscordSendError as send_err:
                                # A question that could not be delivered is NOT a
                                # user decline: abort loudly via the RPC error
                                # (the screener rethrows real RPC failures)
                                # rather than filling a decline for a prompt the
                                # user never saw.
                                logger.error(
                                    "Telegram question not sent",
                                    job_id=job_id,
                                    error=str(send_err),
                                )
                                if process.stdin and not process.stdin.is_closing():
                                    rpc_resp = json.dumps(
                                        {
                                            "type": "RPC_RESPONSE",
                                            "id": req_id,
                                            "error": str(send_err),
                                        }
                                    )
                                    process.stdin.write(f"{rpc_resp}\n".encode())
                                    await process.stdin.drain()
                                continue
                            except DeferredError as deferred:
                                pending = {
                                    "question": deferred.question,
                                    "kind": deferred.kind,
                                    "options": deferred.options,
                                }
                                # Accumulate: a form can raise several unknown
                                # questions. They are recorded once, together,
                                # after the runner exits (see below), so the
                                # digest lists every question — never only the
                                # last one.
                                deferred_pending.append(pending)
                                if debug_rec is not None:
                                    debug_rec["questions"].append(
                                        {
                                            "question": deferred.question,
                                            "kind": deferred.kind,
                                            "options": list(deferred.options or []),
                                            "answer": None,
                                            "source": "deferred",
                                        }
                                    )
                                if process.stdin and not process.stdin.is_closing():
                                    rpc_resp = json.dumps(
                                        {
                                            "type": "RPC_RESPONSE",
                                            "id": req_id,
                                            "error": DEFER_MARKER,
                                        }
                                    )
                                    process.stdin.write(f"{rpc_resp}\n".encode())
                                    await process.stdin.drain()
                                # Persist after the RPC response is written so a
                                # slow store never stalls the runner's pending RPC.
                                await self._record_fill(
                                    job_id,
                                    deferred.question,
                                    None,
                                    source="deferred",
                                    options=deferred.options,
                                )
                                continue

                            if process.stdin and not process.stdin.is_closing():
                                rpc_resp = json.dumps(
                                    {
                                        "type": "RPC_RESPONSE",
                                        "id": req_id,
                                        "result": {"answer": answer, "source": source},
                                    }
                                )
                                process.stdin.write(f"{rpc_resp}\n".encode())
                                await process.stdin.drain()
                            await self._record_fill(
                                job_id, question, answer, source=source, options=options
                            )
                            if debug_rec is not None:
                                debug_rec["questions"].append(
                                    {
                                        "question": question,
                                        "kind": kind,
                                        "options": options,
                                        "answer": answer,
                                        "source": source,
                                    }
                                )
                            continue

                    except Exception as rpc_err:
                        # Never leave the Node side hanging on an unexpected
                        # error: without a response the runner's RPC promise
                        # only rejects after the 30-min timeout. Send the error
                        # response so the fill aborts promptly.
                        logger.error("Error handling RPC request", error=str(rpc_err))
                        if process.stdin and not process.stdin.is_closing():
                            try:
                                rpc_resp = json.dumps(
                                    {"type": "RPC_RESPONSE", "id": req_id, "error": str(rpc_err)}
                                )
                                process.stdin.write(f"{rpc_resp}\n".encode())
                                await process.stdin.drain()
                            except Exception:
                                logger.warning(
                                    "Could not send error response for RPC", job_id=job_id
                                )

                elif line_str.startswith("STATUS_EVENT:"):
                    event_raw = line_str[len("STATUS_EVENT:") :]
                    try:
                        event = json.loads(event_raw)
                        status = event.get("status")
                        screenshot_path = event.get("screenshotPath")
                        filled_fields = event.get("filledFields")

                        logger.info("Received IPC status event", job_id=job_id, status=status)

                        if status == "awaiting_review":
                            if deferred_pending:
                                # The runner keeps filling after a deferral and
                                # still reaches the review step. Never clobber
                                # the deferred status: it is terminal for the
                                # claim loop and drives the morning digest.
                                logger.info(
                                    "Job has deferred questions; keeping status deferred",
                                    job_id=job_id,
                                )
                            else:
                                await self.db.update_status(
                                    job_id,
                                    status="awaiting_review",
                                    filled_payload=filled_fields,
                                    screenshot_path=screenshot_path,
                                )
                                if job_payload.get("submitAllowed", True):
                                    # Submission path: wait for a human decision.
                                    decision = await self._wait_for_human_decision(job_id)
                                    if process.stdin and not process.stdin.is_closing():
                                        action_payload = json.dumps({"action": decision})
                                        process.stdin.write(f"{action_payload}\n".encode())
                                        await process.stdin.drain()

                        elif status == "submitted":
                            terminal_seen = True
                            await self.db.update_status(
                                job_id,
                                status="submitted",
                                filled_payload=filled_fields,
                                screenshot_path=screenshot_path,
                            )
                            await self._debug_finalize(debug_rec, "submitted")
                        elif status == "skipped":
                            terminal_seen = True
                            if deferred_pending:
                                logger.info(
                                    "Job skipped after deferral; keeping status deferred",
                                    job_id=job_id,
                                )
                            else:
                                await self.db.update_status(
                                    job_id,
                                    status="skipped",
                                    filled_payload=filled_fields,
                                    screenshot_path=screenshot_path,
                                )
                                await self._debug_finalize(debug_rec, "skipped")
                        elif status == "failed":
                            error_msg = event.get("error", "Runner failed")
                            if DEFER_MARKER in error_msg or deferred_pending:
                                # Abort of a deferred job: the row is already
                                # marked deferred; never overwrite with failed.
                                logger.info(
                                    "Runner aborted for deferred job",
                                    job_id=job_id,
                                )
                            else:
                                terminal_seen = True
                                await self.db.update_status(
                                    job_id, status="failed", error=error_msg
                                )
                                await self._debug_finalize(debug_rec, "failed", error=error_msg)
                                if "CAPTCHA_DETECTED" in error_msg:
                                    # A bot-detection challenge blocked the form:
                                    # the fill could not proceed. Alert the user
                                    # so they know the application needs manual
                                    # attention instead of silently failing.
                                    await self._notify_captcha(
                                        bridge,
                                        job_id,
                                        apply_link,
                                        error_msg,
                                        role=job.get("role"),
                                        company=job.get("company"),
                                    )

                    except json.JSONDecodeError:
                        logger.error("Failed to parse status event JSON", raw=event_raw)

            await process.wait()
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await stderr_task

            if deferred_pending:
                # Record every unknown question once, together, so the morning
                # digest and resume flow see the full list.
                await self._defer_job(
                    job_id,
                    apply_link,
                    deferred_pending,
                    bridge,
                    role=job.get("role"),
                    company=job.get("company"),
                )
                await self._debug_finalize(
                    debug_rec,
                    "deferred",
                    error="; ".join(q.get("question", "") for q in deferred_pending),
                )
            elif process.returncode != 0 and not terminal_seen:
                # Runner exited abnormally WITHOUT a terminal status event
                # (startup failure, schema error, browser crash). Fail the job
                # loudly instead of leaving it stuck in 'filling' until the
                # lease. When a status event already persisted the specific
                # reason (e.g. CAPTCHA_DETECTED), do not clobber it.
                await self.db.update_status(
                    job_id,
                    status="failed",
                    error=f"Runner exited with code {process.returncode}",
                )
                await self._debug_finalize(
                    debug_rec, "failed", error=f"Runner exited with code {process.returncode}"
                )
            elif not terminal_seen:
                # A normal runner exit WITHOUT a terminal status event and
                # without a non-zero return code leaves the job in 'filling'
                # (lease-bound). Flag it so the debug review sees it.
                await self._debug_finalize(debug_rec, "no_terminal_event")

        except Exception as e:
            logger.exception("Error processing job", job_id=job_id, error=str(e))
            if deferred_pending:
                # Questions were collected before the crash — record them so the
                # digest/resume still sees the full list.
                await self._defer_job(
                    job_id,
                    apply_link,
                    deferred_pending,
                    bridge,
                    role=job.get("role"),
                    company=job.get("company"),
                )
                await self._debug_finalize(debug_rec, "deferred", error=str(e))
            else:
                await self.db.update_status(job_id, status="failed", error=str(e))
                await self._debug_finalize(debug_rec, "failed", error=str(e))
        finally:
            if rag is not None:
                await rag.close()
            self._release_profile(profile_dir)
            if proxy_relay is not None:
                with contextlib.suppress(Exception):
                    await proxy_relay.stop()
                proxy_relay = None

    async def _defer_job(
        self,
        job_id: str,
        apply_link: str,
        questions: list[dict[str, Any]],
        bridge: DiscordQuestionBridge,
        role: Any = None,
        company: Any = None,
    ) -> None:
        """Mark a job deferred (needs user input) and alert via Telegram."""
        await self.db.mark_deferred(job_id, questions=questions, reason="needs user input")
        if bridge.is_configured:
            role_str = str(role or "Position")
            company_str = str(company or "Company")
            text = (
                f"⛔ <b>Deferred</b>: {company_str} — {role_str}\n"
                f'<a href="{apply_link}">Open posting →</a>\n'
                f"Needs your input ({len(questions)}):\n"
                + "\n".join(self._format_pending(q) for q in questions[:6])
            )
            await bridge.send(text)

    async def _notify_captcha(
        self,
        bridge: DiscordQuestionBridge,
        job_id: str,
        apply_link: str,
        error_msg: str,
        role: Any = None,
        company: Any = None,
    ) -> None:
        """Alert the user that a captcha/challenge blocked the application form.

        The fill was aborted with status failed (error CAPTCHA_DETECTED); unlike
        a deferred question this is terminal — the form cannot be completed by
        automation, so the user needs to know it failed and why.
        """
        if not bridge.is_configured:
            logger.warning(
                "Captcha blocked fill but Telegram not configured",
                job_id=job_id,
                error=error_msg,
            )
            return
        role_str = str(role or "Position")
        company_str = str(company or "Company")
        text = (
            f"🛡️ <b>Captcha blocked</b>: {company_str} — {role_str}\n"
            f'<a href="{apply_link}">Open posting →</a>\n'
            f"The application form is behind a bot-detection challenge "
            f"({error_msg.split(':', 1)[-1].strip()}). Automation could not "
            f"fill it — the job was marked <b>failed</b>.\n"
            f"Submit it manually or retry later."
        )
        ok = await bridge.send(text)
        if ok:
            logger.info("Captcha-blocked notification sent", job_id=job_id)
        else:
            logger.warning("Captcha-blocked notification send failed", job_id=job_id)

    @staticmethod
    def _format_pending(entry: dict[str, Any]) -> str:
        q = entry.get("question") or "?"
        options = entry.get("options") or []
        hint = f"  [{', '.join(str(o) for o in options[:6])}]" if options else ""
        return f"• {q}{hint}"

    async def _record_fill(
        self,
        job_id: str,
        question: str,
        answer: str | None,
        source: str | None = None,
        options: list[str] | None = None,
    ) -> None:
        """Persist one question + the answer autofill committed (best-effort)."""
        try:
            await self.db.record_fill(job_id, question, answer, source=source, options=options)
        except Exception as e:
            logger.warning(
                "Fill record not persisted", job_id=job_id, question=question, error=str(e)
            )

    async def _wait_for_human_decision(self, job_id: str, poll_interval: float = 2.0) -> str:
        """Poll the database until job status moves to 'approved' or 'skipped'."""
        logger.info("Waiting for human approval/skip decision...", job_id=job_id)
        while self._running:
            job = await self.db.get_job(job_id)
            if not job:
                return "skip"

            status = job.get("status")
            if status == "approved":
                logger.info("Human APPROVED job submission", job_id=job_id)
                return "submit"
            elif status == "skipped":
                logger.info("Human SKIPPED job submission", job_id=job_id)
                return "skip"

            await asyncio.sleep(poll_interval)
        return "skip"

    async def _read_stderr(self, stderr_stream: asyncio.StreamReader | None, job_id: str) -> None:
        if not stderr_stream:
            return
        while True:
            line = await stderr_stream.readline()
            if not line:
                break
            logger.debug(f"[NodeRunner Stderr-{job_id}] {line.decode('utf-8').strip()}")


async def run_worker() -> None:
    """CLI Entrypoint to run the background worker."""
    load_dotenv()
    db = await AutofillDB.create()
    # Number of simultaneous Stagehand runs (browser processes). Env-driven so
    # a batch can override the default of 4 concurrent job fills.
    max_concurrent = int(os.getenv("AUTOFILL_MAX_CONCURRENT", "4"))
    if _autofill_proxy() and not _autofill_proxy_template() and max_concurrent > 1:
        # Legacy static proxy (Tor): only one SOCKS5 circuit exists, so runs
        # must serialize or they would share the same exit IP.
        logger.warning(
            "AUTOFILL_PROXY set without a template; capping max_concurrent to 1 so "
            "jobs share the Tor proxy without overlapping circuits"
        )
        max_concurrent = 1
    if _autofill_proxy_template():
        # Residential pool path: every job gets its OWN session id -> fresh IP,
        # so concurrent runs are safe (and each still differs by IP).
        logger.info(
            "AUTOFILL_PROXY_TEMPLATE set; per-job residential IP rotation active "
            f"(max_concurrent={max_concurrent})"
        )
    else:
        await _ensure_torproxy()
    worker = AutofillWorker(db, max_concurrent=max_concurrent)
    try:
        await worker.start()
    finally:
        # When the overnight run ends, report every deferred job (with links)
        # on Telegram so the user can resume them next morning.
        try:
            await worker.send_end_of_run_summary()
        except Exception as summary_err:
            logger.warning("End-of-run summary failed", error=str(summary_err))
        await db.close()


if __name__ == "__main__":
    asyncio.run(run_worker())
