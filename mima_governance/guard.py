"""mima_governance.guard — opt-in runtime enforcement for AI attestation.

Enable at your application entry point:

    from mima_governance.guard import enable_guard
    enable_guard(mode="warn")   # "warn" | "block" | "report"

Modes:
  warn   — emit UserWarning when an AI client method is called outside @mima.attest()
  block  — raise MimaAttestationError (fails fast; use in strict testing environments)
  report — silently log to ~/.mima/guard_log.jsonl for later review

Integration with @mima.attest():
    The decorator calls _set_attested(True) before the decorated function runs and
    _set_attested(False) afterwards. The guard checks this flag on every wrapped call.
    Works correctly across threads (threading.local) and async tasks (contextvars).
"""

from __future__ import annotations

import contextvars
import importlib
import json
import queue
import socket
import threading
import warnings
from pathlib import Path
from typing import Any, Callable

from mima_governance.daemon import _SOCK_PATH

# ── OpenTelemetry detection (cached, fail-safe) ─────────────────────────────

_otel_checked: bool = False
_otel_available: bool = False


def _has_configured_otel() -> bool:
    """Return True if opentelemetry-api is installed AND a real TracerProvider
    (not the default proxy/no-op) is configured.

    Result is cached after first call.  Any exception falls safe to False so the
    daemon/queue path is used instead.
    """
    global _otel_checked, _otel_available
    if _otel_checked:
        return _otel_available
    try:
        from opentelemetry import trace
        from opentelemetry.trace import NoOpTracerProvider, ProxyTracerProvider

        provider = trace.get_tracer_provider()
        # ProxyTracerProvider is the default when no SDK is installed/configured.
        # NoOpTracerProvider is what you get when explicitly set to no-op.
        # Both mean "OTEL is not meaningfully configured."
        _otel_available = not isinstance(
            provider, (NoOpTracerProvider, ProxyTracerProvider)
        )
    except Exception:
        _otel_available = False
    _otel_checked = True
    return _otel_available


def _emit_otel_span(name: str) -> bool:
    """Emit an OTEL span for an unattested AI call.

    Returns True on success, False on any failure.  Never raises.
    """
    try:
        from opentelemetry import trace

        from mima_governance.config import get_workspace_id

        tracer = trace.get_tracer("mima.guard", "0.3.0")
        with tracer.start_as_current_span("mima.ai_call") as span:
            span.set_attribute("mima.call_site", name)
            span.set_attribute("mima.attested", False)
            ws_id = get_workspace_id()
            if ws_id:
                span.set_attribute("mima.workspace_id", ws_id)
        return True
    except Exception:
        return False

# ── Attestation context ──────────────────────────────────────────────────────

# Async context (asyncio tasks inherit a copy, so concurrent tasks are isolated).
_async_attested: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_mima_attested", default=False
)

# Sync/thread context (each thread has its own value).
_thread_local = threading.local()


def _is_attested() -> bool:
    # Prefer thread-local (set by sync @mima.attest); fall back to ContextVar (async).
    return getattr(_thread_local, "attested", False) or _async_attested.get(False)


def _set_attested(value: bool) -> None:
    """Set the attested flag in both contexts. Called by @mima.attest() wrappers."""
    _thread_local.attested = value
    # We don't set the ContextVar here by default — async callers use the token pattern.


def _set_attested_async(value: bool) -> "contextvars.Token | None":
    """Set the async ContextVar; returns a token so the caller can reset it."""
    return _async_attested.set(value)


# ── Guard state ───────────────────────────────────────────────────────────────

_guard_enabled: bool = False
_guard_mode: str = "warn"

# Known AI client classes and the methods to wrap.
# Each entry: (module_path, class_name, method_name)
_WRAP_TARGETS: list[tuple[str, str, str]] = [
    ("openai",     "OpenAI",          "chat"),
    ("openai",     "AsyncOpenAI",     "chat"),
    ("anthropic",  "Anthropic",       "messages"),
    ("anthropic",  "AsyncAnthropic",  "messages"),
]

# Simpler: wrap the top-level completion/create entry points by wrapping __init__
# and monkey-patching the call path. The lightweight approach below wraps the
# module-level functions that are the most common call paths.
_FUNCTION_TARGETS: list[tuple[str, str]] = [
    ("litellm",    "completion"),
    ("litellm",    "acompletion"),
]


class MimaAttestationError(Exception):
    """Raised in block mode when an AI call is made outside @mima.attest()."""


def _make_wrapper(original: Callable, name: str, mode: str) -> Callable:
    """Return a wrapper that checks the attestation flag before delegating."""
    import functools

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _is_attested():
            msg = (
                f"[mima guard] Unattested AI call: {name}. "
                "Wrap with @mima.attest() to suppress this."
            )
            if mode == "warn":
                warnings.warn(msg, UserWarning, stacklevel=2)
            elif mode == "block":
                raise MimaAttestationError(msg)
            elif mode == "report":
                _append_report(name)
        return original(*args, **kwargs)

    @functools.wraps(original)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _is_attested():
            msg = (
                f"[mima guard] Unattested AI call: {name}. "
                "Wrap with @mima.attest() to suppress this."
            )
            if mode == "warn":
                warnings.warn(msg, UserWarning, stacklevel=2)
            elif mode == "block":
                raise MimaAttestationError(msg)
            elif mode == "report":
                _append_report(name)
        return await original(*args, **kwargs)

    import asyncio
    return async_wrapper if asyncio.iscoroutinefunction(original) else wrapper


# ── Report-mode: async queue + background writer with log rotation ────────────
#
# Design: a single daemon thread drains a bounded queue to a JSONL file.
# Rotation: 10 MB per file, 3 backup files (guard_log.1.jsonl … .3.jsonl).
#
# This is the "sidecar-lite" pattern — the background thread plays the same
# role as a log-aggregation sidecar in a container stack, but within the process.
# Upgrade path to a true sidecar: replace _drain_report_queue with a Unix-socket
# client that forwards entries to a mima-guard-daemon process. Appropriate when
# you have multiple gunicorn workers writing the same file, or when you want to
# forward guard events to the Mima platform in real time.

_REPORT_MAX_BYTES = 10 * 1024 * 1024   # 10 MB per file
_REPORT_BACKUP_COUNT = 3                # keep guard_log.1–.3.jsonl
_report_queue: "queue.Queue[dict | None] | None" = None
_report_thread: "threading.Thread | None" = None
_report_lock = threading.Lock()


def _rotate_log(log_path: Path) -> None:
    """Rotate log_path when it exceeds _REPORT_MAX_BYTES."""
    try:
        if not log_path.exists() or log_path.stat().st_size < _REPORT_MAX_BYTES:
            return
        for i in range(_REPORT_BACKUP_COUNT - 1, 0, -1):
            src = log_path.parent / f"guard_log.{i}.jsonl"
            dst = log_path.parent / f"guard_log.{i + 1}.jsonl"
            if src.exists():
                src.rename(dst)
        log_path.rename(log_path.parent / "guard_log.1.jsonl")
    except OSError:
        pass


def _drain_report_queue(q: "queue.Queue[dict | None]") -> None:
    """Background daemon thread: drain the report queue to the rotating log."""
    import json
    log_path = Path.home() / ".mima" / "guard_log.jsonl"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    while True:
        try:
            item = q.get(timeout=2.0)
        except queue.Empty:
            continue
        if item is None:        # Sentinel — shut down cleanly.
            break
        try:
            _rotate_log(log_path)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(item) + "\n")
        except OSError:
            pass


def _ensure_report_thread() -> "queue.Queue[dict | None]":
    """Start the background drain thread on first use (idempotent)."""
    global _report_queue, _report_thread
    with _report_lock:
        if _report_queue is None:
            _report_queue = queue.Queue(maxsize=10_000)
            _report_thread = threading.Thread(
                target=_drain_report_queue,
                args=(_report_queue,),
                daemon=True,
                name="mima-guard-reporter",
            )
            _report_thread.start()
        return _report_queue


def _send_via_socket(entry: dict, sock_path: "Path | None" = None) -> bool:
    """Send *entry* to the guard daemon via the Unix socket.

    Returns True on success, False if the socket is unavailable or any error
    occurs.  Never raises.
    """
    import os as _os
    path = Path(sock_path) if sock_path is not None else _SOCK_PATH
    try:
        if not path.exists():
            return False
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(0.5)
        # Use relative path to avoid the ~104-char AF_UNIX limit on macOS.
        old_cwd = _os.getcwd()
        try:
            _os.chdir(str(path.parent))
            s.connect(path.name)
        finally:
            _os.chdir(old_cwd)
        with s:
            s.sendall(json.dumps(entry).encode() + b"\n")
        return True
    except Exception:
        return False


def _append_report(name: str) -> None:
    """Send an unattested call record via the best available channel.

    Priority:
      1. OpenTelemetry span (if a configured TracerProvider is present)
      2. Daemon Unix socket (~/.mima/guard.sock)
      3. In-process queue + background thread

    Non-blocking: queue.put_nowait drops silently when full (> 10 000 items).
    Never raises to the caller.
    """
    try:
        if _has_configured_otel():
            if _emit_otel_span(name):
                return
    except Exception:
        pass  # Fall through to daemon/queue path.

    import os
    from datetime import datetime, timezone

    entry = {
        "ts":   datetime.now(timezone.utc).isoformat(),
        "call": name,
        "pid":  os.getpid(),
    }
    try:
        if not _send_via_socket(entry):
            _ensure_report_thread().put_nowait(entry)
    except Exception:
        pass  # Never let reporting block the guarded call.


def enable_guard(mode: str = "warn") -> None:
    """Activate the runtime attestation guard.

    Args:
        mode: "warn"   — UserWarning on unattested calls (default)
              "block"  — raise MimaAttestationError on unattested calls
              "report" — silently log to ~/.mima/guard_log.jsonl
    """
    global _guard_enabled, _guard_mode

    valid_modes = ("warn", "block", "report")
    if mode not in valid_modes:
        raise ValueError(f"enable_guard: mode must be one of {valid_modes}, got {mode!r}")

    if _guard_enabled:
        return  # Idempotent — calling twice is a no-op.

    _guard_mode = mode
    _guard_enabled = True

    patched_count = 0

    # Patch module-level functions (litellm etc.)
    for mod_name, fn_name in _FUNCTION_TARGETS:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        original = getattr(mod, fn_name, None)
        if original is None or getattr(original, "_mima_guarded", False):
            continue
        wrapper = _make_wrapper(original, f"{mod_name}.{fn_name}", mode)
        wrapper._mima_guarded = True  # type: ignore[attr-defined]
        setattr(mod, fn_name, wrapper)
        patched_count += 1

    # Patch class methods (openai, anthropic)
    # We wrap the __init__ of each class so that instances created after
    # enable_guard() have their key method wrapped. This avoids wrapping
    # every call path and keeps overhead to a single thread-local read.
    for mod_name, class_name, method_name in _WRAP_TARGETS:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        cls = getattr(mod, class_name, None)
        if cls is None:
            continue
        _patch_class_init(cls, class_name, method_name, mod_name, mode)
        patched_count += 1


def _patch_class_init(cls: type, class_name: str, method_name: str,
                      mod_name: str, mode: str) -> None:
    """Wrap cls.__init__ so every new instance gets its method_name proxy wrapped."""
    original_init = cls.__init__

    if getattr(original_init, "_mima_guarded", False):
        return

    import functools

    @functools.wraps(original_init)
    def guarded_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        # Wrap the attribute only if it's a real object (not a NamedAttribute proxy)
        obj = getattr(self, method_name, None)
        if obj is not None and not getattr(obj, "_mima_guarded", False):
            wrapped = _make_wrapper(
                obj.__call__ if callable(obj) else obj,
                f"{mod_name}.{class_name}.{method_name}",
                mode,
            )
            wrapped._mima_guarded = True  # type: ignore[attr-defined]
            # Replace the attribute with a simple callable wrapper
            try:
                object.__setattr__(self, method_name, wrapped)
            except (AttributeError, TypeError):
                pass  # Some clients use __slots__ or property descriptors; skip.

    guarded_init._mima_guarded = True  # type: ignore[attr-defined]
    cls.__init__ = guarded_init  # type: ignore[method-assign]


def disable_guard() -> None:
    """Disable the guard (useful in tests).  Does NOT undo monkey-patches."""
    global _guard_enabled
    _guard_enabled = False
    _thread_local.attested = False
