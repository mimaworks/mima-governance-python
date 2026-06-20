# mima-guard sidecar — Task List

## Phase 1: Core daemon

- [ ] **G1** — `mima_governance/daemon.py`: GuardDaemon socket server, log writer, PID management
  - Constants: `_SOCK_PATH`, `_PID_PATH`, `_LOG_PATH`
  - `_rotate_log`, `_write_pid`, `_read_pid`, `_clear_pid`, `_is_running`
  - `GuardDaemon._daemonize()`, `_serve()`, `_handle_client()`, `_drain()`
  - `start_daemon(mode, forward)`, `stop_daemon()`, `status_daemon()`
  - Verify: `test_guard_daemon.py::TestDaemonLifecycle` passes

- [ ] **G2** — `mima_governance/guard.py`: socket-first `_append_report`
  - `_send_via_socket(entry) -> bool`
  - Update `_append_report`: socket → queue fallback
  - Update `enable_guard` docstring
  - Verify: `test_guard_daemon.py::TestSocketFallback` passes

## CHECKPOINT 1 — multi-process safety
> Two processes → same daemon → single log file, no corruption

## Phase 2: CLI

- [ ] **G3** — `mima_governance/cli.py`: `mima guard start|stop|status`
  - `_cmd_guard(args)` with subcommand dispatch
  - Register in `_COMMANDS`, update `main()` help
  - `status` output: PID, socket path, log path, events/hour
  - Verify: `test_guard_daemon.py::TestCliCommands` passes

## CHECKPOINT 2 — end-to-end local
> `mima guard start` → `enable_guard(mode="report")` → entry in log → `mima guard stop`

## Phase 3: Real-time forwarding

- [ ] **G4** — `daemon.py`: `--forward` batch POST to Mima API
  - `_start_forwarder(q, api_key, workspace_id, base_url)`
  - 5s interval, up to 100 events/batch
  - `POST /api/workspaces/:id/guard/events`
  - 401 disables; network error retries next cycle
  - `status_daemon()` shows forwarding state
  - Verify: `test_guard_daemon.py::TestForwarder` passes (mocked httpx)

## Phase 4: Tests

- [ ] **G5** — `tests/test_guard_daemon.py`: full suite
  - `TestDaemonLifecycle` (start/stop/idempotent)
  - `TestSocketIO` (entry written to log)
  - `TestSocketFallback` (no sock → queue path)
  - `TestRotation` (10 MB rollover)
  - `TestCliCommands` (mima guard start/stop/status)
  - `TestForwarder` (mocked POST, 401 disables)
  - Target: 30+ assertions, 0 failures

## Final verification

- [ ] `python3 -m pytest tests/ -q` — 200+ passing, 0 failures
- [ ] `mima guard start && mima guard status && mima guard stop` — clean lifecycle
- [ ] Two gunicorn-style workers → daemon → single `guard_log.jsonl` — no corruption
