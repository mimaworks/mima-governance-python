"""mima_governance.daemon — Guard sidecar daemon.

A single long-running process that owns the guard log file, accepting entries
via a Unix domain socket.  Multiple app workers (gunicorn, uWSGI, Celery) each
connect to the socket, send a newline-delimited JSON line, and disconnect
immediately.  The daemon is the sole writer — no locking required.

Public API used by cli.py:
    start_daemon(mode, forward)
    stop_daemon()
    status_daemon()

Used by guard.py (import the constant only):
    _SOCK_PATH
"""

from __future__ import annotations

import json
import os
import queue
import select
import signal
import socket
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

_MIMA_DIR  = Path.home() / ".mima"
_SOCK_PATH = _MIMA_DIR / "guard.sock"
_PID_PATH  = _MIMA_DIR / "guard.pid"
_LOG_PATH  = _MIMA_DIR / "guard_log.jsonl"

# ── Log rotation ──────────────────────────────────────────────────────────────

_ROTATE_MAX_BYTES   = 10 * 1024 * 1024   # 10 MB per file
_ROTATE_BACKUP_COUNT = 3                  # keep .1 – .3


def _rotate_log(log_path: Path) -> None:
    """Rotate *log_path* when its size meets or exceeds _ROTATE_MAX_BYTES."""
    try:
        if not log_path.exists():
            return
        if log_path.stat().st_size < _ROTATE_MAX_BYTES:
            return
        for i in range(_ROTATE_BACKUP_COUNT - 1, 0, -1):
            src = log_path.parent / f"guard_log.{i}.jsonl"
            dst = log_path.parent / f"guard_log.{i + 1}.jsonl"
            if src.exists():
                src.rename(dst)
        log_path.rename(log_path.parent / "guard_log.1.jsonl")
    except OSError:
        pass


# ── PID management ────────────────────────────────────────────────────────────

def _write_pid(pid_path: Path, pid: int) -> None:
    pid_path.write_text(str(pid))
    pid_path.chmod(0o600)


def _read_pid(pid_path: Path) -> "int | None":
    try:
        return int(pid_path.read_text().strip())
    except (OSError, ValueError):
        return None


def _clear_pid(pid_path: Path) -> None:
    try:
        pid_path.unlink()
    except OSError:
        pass


def _is_running(pid: int) -> bool:
    """Return True if a process with *pid* exists."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ── Recent-event counter ──────────────────────────────────────────────────────

def _count_recent_events(log_path: Path, minutes: int = 60) -> int:
    """Count log entries within the last *minutes* minutes."""
    if not log_path.exists():
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    count = 0
    try:
        with log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ts = datetime.fromisoformat(json.loads(line)["ts"])
                    if ts >= cutoff:
                        count += 1
                except Exception:
                    pass
    except OSError:
        pass
    return count


# ── GuardDaemon ───────────────────────────────────────────────────────────────

class GuardDaemon:
    """Unix-socket log server.  One instance per daemon process.

    Usage (in-process, e.g. tests):
        d = GuardDaemon(sock_path=..., pid_path=..., log_path=...)
        t = threading.Thread(target=d._serve, daemon=True)
        t.start()
        # ... send entries via the socket ...
        d.stop()
        t.join()

    Usage (forked daemon via start_daemon()):
        The grandchild process calls d._serve() directly after writing its PID.
    """

    def __init__(
        self,
        sock_path: "Path | str" = _SOCK_PATH,
        pid_path:  "Path | str" = _PID_PATH,
        log_path:  "Path | str" = _LOG_PATH,
        *,
        api_key: "str | None" = None,
        workspace_id: "str | None" = None,
        base_url: "str | None" = None,
    ) -> None:
        self._sock_path = Path(sock_path)
        self._pid_path  = Path(pid_path)
        self._log_path  = Path(log_path)
        self._stop      = threading.Event()
        self._queue: "queue.Queue[dict | None]" = queue.Queue(maxsize=10_000)
        # Forwarding (optional).
        self._api_key = api_key
        self._workspace_id = workspace_id
        self._base_url = base_url or "https://api.mima.ai"
        self._forward_queue: "queue.Queue[dict] | None" = (
            queue.Queue(maxsize=10_000) if api_key and workspace_id else None
        )

    # ── Public control ────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the serve loop to exit cleanly."""
        self._stop.set()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _serve(self) -> None:
        """Blocking accept loop.  Exits when stop() is called."""
        self._sock_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove stale socket from a previous crash.
        if self._sock_path.exists():
            self._sock_path.unlink()

        drain_thread = threading.Thread(
            target=self._drain,
            daemon=True,
            name="mima-guard-drain",
        )
        drain_thread.start()

        # Start forwarder thread if credentials were provided.
        forwarder_thread = None
        if self._forward_queue is not None:
            forwarder_thread = threading.Thread(
                target=_start_forwarder,
                args=(
                    self._forward_queue,
                    self._api_key,
                    self._workspace_id,
                    self._base_url,
                    self._stop,
                ),
                daemon=True,
                name="mima-guard-forwarder",
            )
            forwarder_thread.start()

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            # AF_UNIX paths are limited to ~104 chars on macOS / 108 on Linux.
            # Bind using a relative path by chdir-ing to the socket's parent.
            old_cwd = os.getcwd()
            try:
                os.chdir(str(self._sock_path.parent))
                srv.bind(self._sock_path.name)
            finally:
                os.chdir(old_cwd)
            srv.listen(128)
            srv.setblocking(False)

            while not self._stop.is_set():
                ready, _, _ = select.select([srv], [], [], 0.1)
                if not ready:
                    continue
                try:
                    conn, _ = srv.accept()
                except OSError:
                    break
                threading.Thread(
                    target=self._handle_client,
                    args=(conn,),
                    daemon=True,
                ).start()
        finally:
            srv.close()
            # Drain remaining items, then send sentinel.
            self._queue.put(None)
            drain_thread.join(timeout=3.0)
            try:
                self._sock_path.unlink()
            except OSError:
                pass

    def _handle_client(self, conn: socket.socket) -> None:
        with conn:
            try:
                data = b""
                conn.settimeout(2.0)
                while b"\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                if not data:
                    return
                line = data.split(b"\n")[0].strip()
                if not line:
                    return
                entry = json.loads(line.decode("utf-8"))
                try:
                    self._queue.put_nowait(entry)
                except queue.Full:
                    pass
                if self._forward_queue is not None:
                    try:
                        self._forward_queue.put_nowait(entry)
                    except queue.Full:
                        pass
            except Exception:
                pass

    def _drain(self) -> None:
        """Background thread: queue → rotating JSONL file."""
        while True:
            try:
                item = self._queue.get(timeout=2.0)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                _rotate_log(self._log_path)
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(item) + "\n")
            except OSError:
                pass


# ── Forwarder ─────────────────────────────────────────────────────────────────

def _start_forwarder(
    forward_queue: "queue.Queue[dict]",
    api_key: str,
    workspace_id: str,
    base_url: str,
    stop_event: threading.Event,
    *,
    interval: float = 5.0,
) -> None:
    """Background thread: drain *forward_queue* → POST batch every *interval* seconds.

    On 401/403 the forwarder disables itself permanently (no retry storm).
    On network errors the events are requeued for the next cycle.
    """
    import httpx

    url = f"{base_url.rstrip('/')}/api/workspaces/{workspace_id}/guard/events"
    headers = {"Authorization": f"Bearer {api_key}"}

    while not stop_event.is_set():
        stop_event.wait(interval)
        if stop_event.is_set():
            break

        # Drain up to 100 events.
        events: list[dict] = []
        while len(events) < 100:
            try:
                events.append(forward_queue.get_nowait())
            except queue.Empty:
                break

        if not events:
            continue

        try:
            resp = httpx.post(url, json={"events": events}, headers=headers, timeout=10.0)
            if resp.status_code in (401, 403):
                # Credentials invalid — disable permanently.
                return
        except Exception:
            # Network error — requeue for next cycle.
            for e in events:
                try:
                    forward_queue.put_nowait(e)
                except queue.Full:
                    pass


# ── CLI-facing public API ─────────────────────────────────────────────────────

def start_daemon(mode: str = "warn", forward: bool = False) -> None:
    """Fork and start the guard daemon.  Prints confirmation to stdout."""
    existing_pid = _read_pid(_PID_PATH)
    if existing_pid and _is_running(existing_pid):
        print(f"Guard daemon already running (PID {existing_pid})")
        return

    # Resolve forwarding credentials in the parent process (before fork).
    api_key: "str | None" = None
    workspace_id: "str | None" = None
    base_url: str = "https://api.mima.ai"
    forward_warning: str = ""

    if forward:
        from mima_governance import config
        api_key = config.get_api_key()
        workspace_id = config.get_workspace_id()
        base_url = config.get_base_url()
        if not api_key or not workspace_id:
            forward_warning = "Warning: forwarding disabled (no credentials). Run 'mima login' first."
            api_key = None
            workspace_id = None

    first_fork = os.fork()
    if first_fork > 0:
        # Parent: wait up to 5 s for the daemon to write its PID file.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if _PID_PATH.exists():
                break
            time.sleep(0.05)
        daemon_pid = _read_pid(_PID_PATH)
        if forward_warning:
            print(forward_warning)
        print(f"Guard daemon started (PID {daemon_pid})")
        return

    # ── First child ───────────────────────────────────────────────────────────
    try:
        os.setsid()
    except OSError:
        pass

    second_fork = os.fork()
    if second_fork > 0:
        os._exit(0)

    # ── Grandchild (the actual daemon) ────────────────────────────────────────
    # Redirect stdin/stdout/stderr to /dev/null.
    try:
        devnull_r = open(os.devnull, "r")
        devnull_w = open(os.devnull, "w")
        os.dup2(devnull_r.fileno(), sys.stdin.fileno())
        os.dup2(devnull_w.fileno(), sys.stdout.fileno())
        os.dup2(devnull_w.fileno(), sys.stderr.fileno())
    except OSError:
        pass

    daemon = GuardDaemon(
        sock_path=_SOCK_PATH,
        pid_path=_PID_PATH,
        log_path=_LOG_PATH,
        api_key=api_key,
        workspace_id=workspace_id,
        base_url=base_url,
    )
    _write_pid(_PID_PATH, os.getpid())

    def _sigterm(sig: int, frame: object) -> None:
        daemon.stop()

    signal.signal(signal.SIGTERM, _sigterm)

    try:
        daemon._serve()
    finally:
        _clear_pid(_PID_PATH)
        os._exit(0)


def stop_daemon() -> None:
    """Send SIGTERM to the guard daemon.  Prints confirmation to stdout."""
    pid = _read_pid(_PID_PATH)
    if not pid or not _is_running(pid):
        print("Guard daemon not running")
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass

    # Wait up to 5 s for the process to exit.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _is_running(pid):
            break
        time.sleep(0.1)

    _clear_pid(_PID_PATH)
    print(f"Guard daemon stopped (was PID {pid})")


def status_daemon() -> None:
    """Print daemon status and exit 0 (running) or 1 (not running)."""
    pid = _read_pid(_PID_PATH)
    if not pid or not _is_running(pid):
        print("Guard daemon not running")
        sys.exit(1)

    log_entries = 0
    try:
        if _LOG_PATH.exists():
            with _LOG_PATH.open() as fh:
                log_entries = sum(1 for _ in fh)
    except OSError:
        pass

    events_hour = _count_recent_events(_LOG_PATH, minutes=60)

    print(f"Guard daemon   running (PID {pid})")
    print(f"Socket         {_SOCK_PATH}")
    print(f"Log            {_LOG_PATH}  ({log_entries} entries)")
    print(f"Forwarding     disabled  (run with --forward to stream to dashboard)")
    print(f"Events/hour    ~{events_hour}  (last 60 min)")
    sys.exit(0)
