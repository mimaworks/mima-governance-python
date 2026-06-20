"""Tests for mima_governance.daemon — GuardDaemon socket server, log writer, PID mgmt."""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _connect_unix(sock_path: Path, timeout: float = 2.0) -> socket.socket:
    """Return a connected AF_UNIX socket using a relative path to avoid the
    104-char macOS path-length limit for AF_UNIX addresses."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    old_cwd = os.getcwd()
    try:
        os.chdir(str(sock_path.parent))
        s.connect(sock_path.name)
    finally:
        os.chdir(old_cwd)
    return s


def _send_entry(sock_path: Path, entry: dict, timeout: float = 2.0) -> None:
    """Connect to the daemon socket and send one JSON entry."""
    with _connect_unix(sock_path, timeout) as client:
        client.sendall(json.dumps(entry).encode() + b"\n")


def _start_server_thread(daemon) -> threading.Thread:
    """Run daemon._serve() in a daemon thread. Returns the thread."""
    t = threading.Thread(target=daemon._serve, daemon=True)
    t.start()
    return t


def _wait_for_socket(sock_path: Path, timeout: float = 2.0) -> bool:
    """Block until sock_path exists or timeout. Returns True if found."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sock_path.exists():
            return True
        time.sleep(0.02)
    return False


def _wait_for_log(log_path: Path, min_bytes: int = 1, timeout: float = 2.0) -> bool:
    """Block until log_path has at least min_bytes or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log_path.exists() and log_path.stat().st_size >= min_bytes:
            return True
        time.sleep(0.02)
    return False


# ── TestRotateLog ─────────────────────────────────────────────────────────────

class TestRotateLog:
    def test_no_rotation_below_threshold(self, tmp_path):
        from mima_governance.daemon import _rotate_log, _ROTATE_MAX_BYTES
        log = tmp_path / "guard_log.jsonl"
        log.write_text("x" * 100)
        _rotate_log(log)
        assert log.exists()
        assert not (tmp_path / "guard_log.1.jsonl").exists()

    def test_rotates_at_threshold(self, tmp_path):
        from mima_governance.daemon import _ROTATE_MAX_BYTES
        from mima_governance.daemon import _rotate_log
        log = tmp_path / "guard_log.jsonl"
        log.write_bytes(b"x" * _ROTATE_MAX_BYTES)
        _rotate_log(log)
        assert not log.exists()
        assert (tmp_path / "guard_log.1.jsonl").exists()

    def test_rotation_chain(self, tmp_path):
        from mima_governance.daemon import _ROTATE_MAX_BYTES, _rotate_log
        log = tmp_path / "guard_log.jsonl"
        (tmp_path / "guard_log.1.jsonl").write_text("old1")
        (tmp_path / "guard_log.2.jsonl").write_text("old2")
        log.write_bytes(b"x" * _ROTATE_MAX_BYTES)
        _rotate_log(log)
        assert (tmp_path / "guard_log.2.jsonl").read_text() == "old1"
        assert (tmp_path / "guard_log.3.jsonl").read_text() == "old2"

    def test_missing_file_is_noop(self, tmp_path):
        from mima_governance.daemon import _rotate_log
        _rotate_log(tmp_path / "nonexistent.jsonl")  # must not raise


# ── TestPidManagement ─────────────────────────────────────────────────────────

class TestPidManagement:
    def test_write_and_read_pid(self, tmp_path):
        from mima_governance.daemon import _write_pid, _read_pid
        pid_file = tmp_path / "guard.pid"
        _write_pid(pid_file, 12345)
        assert pid_file.exists()
        assert _read_pid(pid_file) == 12345

    def test_pid_file_perms(self, tmp_path):
        from mima_governance.daemon import _write_pid
        pid_file = tmp_path / "guard.pid"
        _write_pid(pid_file, 99)
        mode = oct(pid_file.stat().st_mode & 0o777)
        assert mode == oct(0o600)

    def test_read_missing_pid(self, tmp_path):
        from mima_governance.daemon import _read_pid
        assert _read_pid(tmp_path / "missing.pid") is None

    def test_read_corrupt_pid(self, tmp_path):
        from mima_governance.daemon import _read_pid
        f = tmp_path / "bad.pid"
        f.write_text("not-a-number")
        assert _read_pid(f) is None

    def test_clear_pid(self, tmp_path):
        from mima_governance.daemon import _write_pid, _clear_pid
        pid_file = tmp_path / "guard.pid"
        _write_pid(pid_file, 1)
        _clear_pid(pid_file)
        assert not pid_file.exists()

    def test_clear_missing_is_noop(self, tmp_path):
        from mima_governance.daemon import _clear_pid
        _clear_pid(tmp_path / "missing.pid")  # must not raise


# ── TestIsRunning ─────────────────────────────────────────────────────────────

class TestIsRunning:
    def test_current_process_is_running(self):
        from mima_governance.daemon import _is_running
        assert _is_running(os.getpid()) is True

    def test_dead_pid_not_running(self):
        from mima_governance.daemon import _is_running
        # PID 1 is init/launchd — always alive on Unix.
        # PID 99999999 is almost certainly dead.
        assert _is_running(99999999) is False


# ── TestDaemonLifecycle ───────────────────────────────────────────────────────

@pytest.mark.skipif(sys.platform == "win32", reason="AF_UNIX sockets tested on Unix only")
class TestDaemonLifecycle:
    def test_serve_creates_socket_file(self, tmp_path):
        from mima_governance.daemon import GuardDaemon
        d = GuardDaemon(
            sock_path=tmp_path / "guard.sock",
            pid_path=tmp_path / "guard.pid",
            log_path=tmp_path / "guard_log.jsonl",
        )
        t = _start_server_thread(d)
        found = _wait_for_socket(tmp_path / "guard.sock")
        d.stop()
        t.join(timeout=3.0)
        assert found, "socket file did not appear"

    def test_stop_removes_socket_file(self, tmp_path):
        from mima_governance.daemon import GuardDaemon
        d = GuardDaemon(
            sock_path=tmp_path / "guard.sock",
            pid_path=tmp_path / "guard.pid",
            log_path=tmp_path / "guard_log.jsonl",
        )
        t = _start_server_thread(d)
        _wait_for_socket(tmp_path / "guard.sock")
        d.stop()
        t.join(timeout=3.0)
        assert not (tmp_path / "guard.sock").exists()

    def test_stop_without_start_is_noop(self, tmp_path):
        from mima_governance.daemon import GuardDaemon
        d = GuardDaemon(
            sock_path=tmp_path / "guard.sock",
            pid_path=tmp_path / "guard.pid",
            log_path=tmp_path / "guard_log.jsonl",
        )
        d.stop()  # must not raise


# ── TestSocketIO ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(sys.platform == "win32", reason="AF_UNIX sockets tested on Unix only")
class TestSocketIO:
    def _run_daemon(self, tmp_path):
        from mima_governance.daemon import GuardDaemon
        d = GuardDaemon(
            sock_path=tmp_path / "guard.sock",
            pid_path=tmp_path / "guard.pid",
            log_path=tmp_path / "guard_log.jsonl",
        )
        t = _start_server_thread(d)
        assert _wait_for_socket(tmp_path / "guard.sock"), "socket did not appear"
        return d, t

    def test_entry_written_to_log(self, tmp_path):
        d, t = self._run_daemon(tmp_path)
        entry = {"ts": "2026-06-20T00:00:00+00:00", "call": "openai.OpenAI.chat", "pid": 42}
        try:
            _send_entry(tmp_path / "guard.sock", entry)
            assert _wait_for_log(tmp_path / "guard_log.jsonl"), "log file did not grow"
            written = json.loads((tmp_path / "guard_log.jsonl").read_text().strip())
            assert written["call"] == "openai.OpenAI.chat"
            assert written["pid"] == 42
        finally:
            d.stop()
            t.join(timeout=3.0)

    def test_multiple_entries_appended(self, tmp_path):
        d, t = self._run_daemon(tmp_path)
        entries = [
            {"ts": "2026-06-20T00:00:01+00:00", "call": "openai.OpenAI.chat", "pid": 1},
            {"ts": "2026-06-20T00:00:02+00:00", "call": "anthropic.Anthropic.messages", "pid": 2},
            {"ts": "2026-06-20T00:00:03+00:00", "call": "litellm.completion", "pid": 3},
        ]
        try:
            for e in entries:
                _send_entry(tmp_path / "guard.sock", e)
            # Wait for all three entries
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                log = tmp_path / "guard_log.jsonl"
                if log.exists() and len(log.read_text().splitlines()) >= 3:
                    break
                time.sleep(0.05)
            lines = (tmp_path / "guard_log.jsonl").read_text().splitlines()
            assert len(lines) == 3
            calls = [json.loads(l)["call"] for l in lines]
            assert "openai.OpenAI.chat" in calls
            assert "anthropic.Anthropic.messages" in calls
        finally:
            d.stop()
            t.join(timeout=3.0)

    def test_malformed_json_does_not_crash_server(self, tmp_path):
        d, t = self._run_daemon(tmp_path)
        try:
            with _connect_unix(tmp_path / "guard.sock") as client:
                client.sendall(b"not json at all\n")
            time.sleep(0.2)
            # Server still alive — send a valid entry
            entry = {"ts": "2026-06-20T00:00:00+00:00", "call": "openai.chat", "pid": 1}
            _send_entry(tmp_path / "guard.sock", entry)
            assert _wait_for_log(tmp_path / "guard_log.jsonl")
        finally:
            d.stop()
            t.join(timeout=3.0)

    def test_client_disconnect_without_data_is_handled(self, tmp_path):
        d, t = self._run_daemon(tmp_path)
        try:
            with _connect_unix(tmp_path / "guard.sock") as client:
                pass  # Close immediately without sending anything
            time.sleep(0.2)
            # Server must still be running
            assert not d._stop.is_set()
        finally:
            d.stop()
            t.join(timeout=3.0)


# ── TestSocketFallback ────────────────────────────────────────────────────────

@pytest.mark.skipif(sys.platform == "win32", reason="AF_UNIX sockets tested on Unix only")
class TestSocketFallback:
    def test_send_via_socket_returns_true_when_daemon_running(self, tmp_path):
        from mima_governance.daemon import GuardDaemon
        from mima_governance.guard import _send_via_socket
        d = GuardDaemon(
            sock_path=tmp_path / "guard.sock",
            pid_path=tmp_path / "guard.pid",
            log_path=tmp_path / "guard_log.jsonl",
        )
        t = _start_server_thread(d)
        assert _wait_for_socket(tmp_path / "guard.sock")
        entry = {"ts": "2026-06-20T00:00:00+00:00", "call": "test", "pid": 1}
        try:
            result = _send_via_socket(entry, sock_path=tmp_path / "guard.sock")
            assert result is True
        finally:
            d.stop()
            t.join(timeout=3.0)

    def test_send_via_socket_returns_false_when_no_daemon(self, tmp_path):
        from mima_governance.guard import _send_via_socket
        entry = {"ts": "2026-06-20T00:00:00+00:00", "call": "test", "pid": 1}
        result = _send_via_socket(entry, sock_path=tmp_path / "guard.sock")
        assert result is False

    def test_append_report_falls_back_to_queue_when_no_daemon(self, tmp_path, monkeypatch):
        """When no socket exists, _append_report uses the in-process queue."""
        from mima_governance import guard
        monkeypatch.setattr(guard, "_guard_enabled", True)
        monkeypatch.setattr(guard, "_guard_mode", "report")

        queued = []

        def fake_put_nowait(entry):
            queued.append(entry)

        fake_q = MagicMock()
        fake_q.put_nowait = fake_put_nowait

        with patch("mima_governance.guard._ensure_report_thread", return_value=fake_q):
            with patch("mima_governance.guard._SOCK_PATH", tmp_path / "guard.sock"):
                guard._append_report("openai.OpenAI.chat")

        assert len(queued) == 1
        assert queued[0]["call"] == "openai.OpenAI.chat"


# ── TestCliCommands ───────────────────────────────────────────────────────────

class TestCliCommands:
    def test_guard_help(self, capsys):
        with patch("sys.argv", ["mima", "guard", "--help"]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "start" in out
        assert "stop" in out
        assert "status" in out

    def test_guard_status_not_running(self, capsys, tmp_path):
        from mima_governance import daemon as dm
        with patch.object(dm, "_PID_PATH", tmp_path / "guard.pid"):
            with patch.object(dm, "_SOCK_PATH", tmp_path / "guard.sock"):
                with patch.object(dm, "_LOG_PATH", tmp_path / "guard_log.jsonl"):
                    with patch("sys.argv", ["mima", "guard", "status"]):
                        with pytest.raises(SystemExit) as exc:
                            from mima_governance.cli import main
                            main()
                        assert exc.value.code == 1
        err_out = capsys.readouterr()
        assert "not running" in err_out.out

    def test_guard_stop_not_running(self, capsys, tmp_path):
        from mima_governance import daemon as dm
        with patch.object(dm, "_PID_PATH", tmp_path / "guard.pid"):
            with patch.object(dm, "_SOCK_PATH", tmp_path / "guard.sock"):
                with patch("sys.argv", ["mima", "guard", "stop"]):
                    from mima_governance.cli import main
                    main()  # must not raise
        out = capsys.readouterr().out
        assert "not running" in out

    @pytest.mark.skipif(sys.platform == "win32", reason="fork-based daemon Unix only")
    def test_guard_start_and_stop(self, tmp_path, capsys):
        from mima_governance import daemon as dm
        with patch.object(dm, "_PID_PATH", tmp_path / "guard.pid"):
            with patch.object(dm, "_SOCK_PATH", tmp_path / "guard.sock"):
                with patch.object(dm, "_LOG_PATH", tmp_path / "guard_log.jsonl"):
                    # start
                    with patch("sys.argv", ["mima", "guard", "start", "--mode", "report"]):
                        from mima_governance.cli import main
                        main()
                    out = capsys.readouterr().out
                    assert "started" in out

                    # status — should be running
                    with patch("sys.argv", ["mima", "guard", "status"]):
                        with pytest.raises(SystemExit) as exc:
                            main()
                        assert exc.value.code == 0
                    out = capsys.readouterr().out
                    assert "running" in out

                    # stop
                    with patch("sys.argv", ["mima", "guard", "stop"]):
                        main()
                    out = capsys.readouterr().out
                    assert "stopped" in out


# ── TestCountRecentEvents ─────────────────────────────────────────────────────

class TestCountRecentEvents:
    def test_counts_recent_entries(self, tmp_path):
        from mima_governance.daemon import _count_recent_events
        from datetime import datetime, timezone, timedelta
        log = tmp_path / "guard_log.jsonl"
        now = datetime.now(timezone.utc)
        entries = [
            {"ts": (now - timedelta(minutes=5)).isoformat(), "call": "a", "pid": 1},
            {"ts": (now - timedelta(minutes=30)).isoformat(), "call": "b", "pid": 1},
            {"ts": (now - timedelta(minutes=90)).isoformat(), "call": "c", "pid": 1},
        ]
        with log.open("w") as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")
        count = _count_recent_events(log, minutes=60)
        assert count == 2  # only last 5min and 30min entries

    def test_missing_log_returns_zero(self, tmp_path):
        from mima_governance.daemon import _count_recent_events
        assert _count_recent_events(tmp_path / "missing.jsonl") == 0
