# mima-guard sidecar daemon — Implementation Plan

## Problem statement

`mode="report"` in `enable_guard()` currently writes to `~/.mima/guard_log.jsonl` via a
background thread queue. This breaks under any multi-process runtime (gunicorn, uWSGI,
Celery): each worker spawns its own background thread and all write to the same file
simultaneously with no locking. Log corruption under load is guaranteed.

Beyond correctness, the design seals off real-time streaming to the Mima platform — the
feature that turns `mode="report"` from a log file into a live observability product.

## Architecture

```
App process 1 ──→  Unix socket (~/.mima/guard.sock)
App process 2 ──→  Unix socket                      →  mima-guard-daemon
App process N ──→  Unix socket                           │
                                                         ├── guard_log.jsonl (single writer)
                                                         └── POST /guard/events (optional --forward)
```

Wire protocol: newline-delimited JSON over `AF_UNIX SOCK_STREAM`. Client opens socket,
writes `json.dumps(entry) + "\n"`, closes immediately. No response. Zero blocking on the
caller side.

Fallback: if `~/.mima/guard.sock` does not exist, `_append_report` falls back to the
existing queue → background thread path. No breaking change for single-process apps or
users who haven't started the daemon.

## Dependency graph

```
daemon.py (new)
    ← guard.py (modify: socket-first _append_report)
    ← cli.py   (modify: _cmd_guard, _COMMANDS, main() help)
    ← tests/test_guard_daemon.py (new)
```

`daemon.py` has no imports from other mima_governance modules (avoids circular deps).
`guard.py` imports `daemon._SOCK_PATH` as a constant only.

## Files touched

| File | Action | Notes |
|---|---|---|
| `mima_governance/daemon.py` | CREATE | GuardDaemon: socket server, log writer, PID mgmt, forwarder |
| `mima_governance/guard.py` | MODIFY | `_send_via_socket`, socket-first `_append_report` |
| `mima_governance/cli.py` | MODIFY | `_cmd_guard`, register in `_COMMANDS`, update help |
| `tests/test_guard_daemon.py` | CREATE | lifecycle, socket I/O, fallback, forward |

---

## Task G1 — daemon.py: socket server + log writer

**What**: `mima_governance/daemon.py` — the entire daemon lives here.

**Scope**:
- Constants: `_SOCK_PATH`, `_PID_PATH`, `_LOG_PATH` — all under `~/.mima/`
- `_rotate_log(log_path)` — moved from guard.py (guard.py imports it back to avoid duplication)
- `_write_pid()`, `_read_pid()`, `_clear_pid()` — PID file management with 0o600 perms
- `_is_running(pid)` — `os.kill(pid, 0)` check
- `GuardDaemon` class:
  - `start(mode, forward)` — forks (Unix) or subprocesses (Windows), writes PID
  - `_daemonize()` — double-fork, setsid, redirect stdio to /dev/null
  - `_serve(mode, forward)` — blocking accept loop: `socket.AF_UNIX SOCK_STREAM`
  - `_handle_client(conn)` — reads bytes until newline, parses JSON, enqueues
  - `_drain(q)` — background thread: queue → `_rotate_log` + file write
  - `_start_forwarder(q, api_key, workspace_id, base_url)` — timer thread, POST every 5s
- `start_daemon(mode, forward)`, `stop_daemon()`, `status_daemon()` — public CLI-facing API

**Acceptance criteria**:
- `start_daemon()` writes `~/.mima/guard.pid` and `~/.mima/guard.sock` appears
- A client that connects and sends a JSON line sees the entry in `guard_log.jsonl`
- `stop_daemon()` removes both files and the process exits cleanly
- `start_daemon()` called twice prints "already running (PID N)" and exits 0
- Rotation: log file rolls at 10 MB, keeps 3 backups

**Verification**: `python3 -m pytest tests/test_guard_daemon.py::TestDaemonLifecycle -x`

---

## Task G2 — guard.py: socket-first _append_report

**What**: Update `_append_report` in `guard.py` to try the Unix socket before falling back
to the in-process queue.

**Scope**:
- `_send_via_socket(entry: dict) -> bool` — opens `AF_UNIX`, writes JSON+\n, closes, returns True on success
- Update `_append_report`: `if not _send_via_socket(entry): _ensure_report_thread().put_nowait(entry)`
- Move `_rotate_log` import to `daemon.py` (or keep copy in guard.py — decision: keep copy, no circular dep)
- Remove the "upgrade path" comment (it's now the current path)
- Update docstring on `enable_guard` mode="report" to mention daemon

**Acceptance criteria**:
- With daemon running: `_append_report` connects to socket, returns in < 1ms, entry lands in daemon log
- Without daemon: falls back to queue thread exactly as before — no behaviour change
- `_send_via_socket` swallows all exceptions (never raises to caller)
- Test with `guard_enabled=True, mode="report"` in both daemon-present and daemon-absent states

**Verification**: `python3 -m pytest tests/test_guard_daemon.py::TestSocketFallback -x`

---

## CHECKPOINT 1 — multi-process safety verified

After G1 + G2: start the daemon in one shell, run two Python processes both calling
`_append_report`, verify single `guard_log.jsonl` with interleaved entries, no corruption.

---

## Task G3 — cli.py: mima guard start|stop|status

**What**: Add `_cmd_guard` to `cli.py`.

**Scope**:
```
mima guard start [--mode warn|block|report] [--forward]
mima guard stop
mima guard status
mima guard --help
```
- `_cmd_guard(args)` — parse subcommand, dispatch to `daemon.start_daemon/stop_daemon/status_daemon`
- `start`: prints socket path and next-step hint
- `stop`: prints confirmation
- `status` output:
  ```
  Guard daemon   running (PID 12345)
  Socket         ~/.mima/guard.sock
  Log            ~/.mima/guard_log.jsonl  (1,204 entries)
  Forwarding     disabled  (run with --forward to stream to dashboard)
  Events/hour    ~340  (last 60 min)
  ```
- Register `"guard": _cmd_guard` in `_COMMANDS`
- Add `mima guard start` to `main()` help block

**Acceptance criteria**:
- `mima guard --help` exits 0 and documents all flags
- `mima guard start` with daemon already running exits 0 with "already running" message
- `mima guard status` when not running exits 1 with "not running" message
- `mima guard stop` when not running exits 0 with "nothing to stop" message

**Verification**: `python3 -m pytest tests/test_guard_daemon.py::TestCliCommands -x`

---

## CHECKPOINT 2 — end-to-end local path

After G3: full flow works:
```bash
mima guard start --mode report
# in app: enable_guard(mode="report") → calls _append_report → socket → daemon → log
mima guard status
mima guard stop
```

---

## Task G4 — --forward: batch POST to Mima platform

**What**: `--forward` flag on `mima guard start` enables real-time event forwarding.

**Scope**:
- `_start_forwarder(q, api_key, workspace_id, base_url)` in `daemon.py`
  - Background `threading.Timer` loop, 5s interval
  - Drains a secondary `forward_queue` (daemon writes to both log queue and forward queue)
  - Batches up to 100 entries per POST: `POST /api/workspaces/:id/guard/events`
  - Body: `{"events": [{"ts": ..., "call": ..., "pid": ...}, ...]}`
  - On 401/403: disables forwarding, logs warning to stderr
  - On network error: retries next cycle (events not dropped)
- `status_daemon()` shows "Forwarding: enabled (last flush Ns ago, N events/5s)"
- Reads credentials from `config.get_api_key()` / `config.get_workspace_id()` at start time
- If no credentials: `--forward` prints warning and starts without forwarding

**Acceptance criteria**:
- With mocked httpx: POST fires within 6s of first event arriving
- Batch contains all events since last flush
- 401 response disables further POST attempts (no retry storm)
- `--forward` without credentials starts daemon without forwarding, prints warning

**Verification**: `python3 -m pytest tests/test_guard_daemon.py::TestForwarder -x`

---

## Task G5 — tests/test_guard_daemon.py

Full test suite — written alongside each task.

**Test classes**:
- `TestDaemonLifecycle` — start/stop/idempotent start/already-running
- `TestSocketIO` — client sends entry, daemon writes to log
- `TestSocketFallback` — no sock file → queue path used
- `TestRotation` — oversized log triggers rename chain
- `TestCliCommands` — `mima guard start/stop/status` via patched sys.argv
- `TestForwarder` — mocked httpx, batch POST, 401 disables

**Target**: 30+ new assertions, all passing, no side effects on CI (temp dirs, no real sockets in CI)

---

## Constants and wire format

```python
# daemon.py
_MIMA_DIR  = Path.home() / ".mima"
_SOCK_PATH = _MIMA_DIR / "guard.sock"
_PID_PATH  = _MIMA_DIR / "guard.pid"
_LOG_PATH  = _MIMA_DIR / "guard_log.jsonl"

# Wire: newline-delimited JSON
# {"ts": "2026-06-20T12:00:00+00:00", "call": "openai.OpenAI.chat", "pid": 1234}
```

## What we are NOT doing (and why)

| Not doing | Why |
|---|---|
| WebSocket/SSE from daemon directly | Mima API handles browser fan-out; daemon is a writer not a server |
| launchd/systemd service registration | `mima guard start` is sufficient; OS-level daemonization is a later step |
| Windows named pipes | AF_UNIX available on Windows 10+ (Python 3.9+); fallback to queue handles older |
| Per-worker daemon | One daemon per machine; all workers share one socket |
| TLS on socket | `~/.mima/guard.sock` is owner-only; no network exposure |
