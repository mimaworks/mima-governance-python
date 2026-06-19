"""Mima Governance SDK — sync client with decorator, batch, and signing."""

import atexit
import hashlib
import json
import functools
import time
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import httpx

from ._base import MimaAttestationError, _MimaGrcMixin
from .types import AttestationRecord, AttestationResult, AuthorisedBy, GrcRecord, GrcResult

# Re-export so callers that do `from mima_governance.client import MimaAttestationError` keep working.
__all__ = ["MimaGovernance", "MimaAttestationError"]


class MimaGovernance(_MimaGrcMixin):
    """
    Mima AI Governance SDK client (sync).

    Usage:
        mima = MimaGovernance(
            workspace_id="uuid",
            api_key="mima_ext_...",
            system_name="my-ai-pipeline",
        )

        @mima.attest(tool_name="generate_report")
        def generate_report(data):
            return call_llm(data)
    """

    def __init__(
        self,
        workspace_id: str,
        api_key: str,
        system_name: str,
        *,
        base_url: str = "https://api.mima.ai",
        agent_name: Optional[str] = None,
        signing_key: Optional[bytes] = None,
        authorised_by: Optional[AuthorisedBy] = None,
        on_error: str = "warn",  # "warn" | "raise" | "silent"
        batch_flush_interval: float = 30.0,
        batch_max_size: int = 100,
    ):
        self.workspace_id = workspace_id
        self.api_key = api_key
        self.system_name = system_name
        self.agent_name = agent_name or system_name
        self.base_url = base_url.rstrip("/")
        self.signing_key = signing_key
        self.authorised_by = authorised_by
        self.on_error = on_error
        self._batch_flush_interval = batch_flush_interval
        self._batch_max_size = batch_max_size
        self._closed = False

        self._http = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "mima-governance-python/0.1.0",
            },
            timeout=15.0,
        )

        # Batch queue
        self._batch_queue: List[AttestationRecord] = []
        self._batch_lock = threading.Lock()
        self._batch_timer: Optional[threading.Timer] = None

        # Register flush on interpreter shutdown. atexit is reliable; __del__
        # is not — Python does not guarantee destructor call order or timing,
        # so records queued near shutdown can silently vanish with __del__ alone.
        atexit.register(self._flush_batch)

    # ── Decorator: zero-effort attestation ───────────────────────────────────

    def attest(
        self,
        tool_name: str,
        *,
        mode: str = "sync",
        model_id: Optional[str] = None,
        authorised_by: Optional[AuthorisedBy] = None,
    ) -> Callable:
        """
        Decorator that automatically attests a function's execution.

        @mima.attest(tool_name="classify_document")
        def classify(doc):
            return model.predict(doc)

        Args:
            tool_name: Name of the tool/action being executed.
            mode: "sync" (default, immediate push) or "batch" (buffered).
            model_id: LLM model identifier (e.g. "gpt-4o").
            authorised_by: Override the client-level authorised_by.
        """

        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                input_data = _serialize_for_hash(args, kwargs)
                input_hash = _sha256(input_data)

                result = fn(*args, **kwargs)

                output_hash = _sha256(_serialize_for_hash(result))

                record = AttestationRecord(
                    tool_name=tool_name,
                    input_hash=input_hash,
                    output_hash=output_hash,
                    model_id=model_id,
                    executed_at=datetime.now(timezone.utc).isoformat(),
                    authorised_by=authorised_by or self.authorised_by,
                )

                if mode == "batch":
                    self._enqueue(record)
                else:
                    self._push_sync(record)

                return result

            return wrapper

        return decorator

    # ── Explicit push ────────────────────────────────────────────────────────

    def push(
        self,
        tool_name: str,
        input_hash: str,
        output_hash: str,
        *,
        model_id: Optional[str] = None,
        authorised_by: Optional[AuthorisedBy] = None,
    ) -> AttestationResult:
        """Push a single attestation immediately (sync mode)."""
        record = AttestationRecord(
            tool_name=tool_name,
            input_hash=input_hash,
            output_hash=output_hash,
            model_id=model_id,
            executed_at=datetime.now(timezone.utc).isoformat(),
            authorised_by=authorised_by or self.authorised_by,
        )
        return self._push_sync(record)

    # ── Context manager for tracing ──────────────────────────────────────────

    @contextmanager
    def trace(self, tool_name: str, *, model_id: Optional[str] = None):
        """
        Context manager for explicit tracing.

        with mima.trace("classify_document") as t:
            t.set_input(document)
            result = model.predict(document)
            t.set_output(result)
            t.set_model_id("gpt-4o")
        """
        ctx = _TraceContext(tool_name, model_id)
        yield ctx
        record = AttestationRecord(
            tool_name=tool_name,
            input_hash=ctx.input_hash or "",
            output_hash=ctx.output_hash or "",
            model_id=ctx.model_id or model_id,
            executed_at=datetime.now(timezone.utc).isoformat(),
            authorised_by=self.authorised_by,
        )
        self._push_sync(record)

    # ── Batch context manager ────────────────────────────────────────────────

    @contextmanager
    def batch(self):
        """
        Batch context manager — flushes on exit.

        with mima.batch() as b:
            for item in items:
                b.add("process", input=item, output=result)
        """
        ctx = _BatchContext(self)
        yield ctx
        self._flush_batch()

    # ── Internal: sync push ──────────────────────────────────────────────────

    def _push_sync(self, record: AttestationRecord) -> AttestationResult:
        payload = self._build_payload(record)

        try:
            resp = self._http.post(
                f"/api/workspaces/{self.workspace_id}/governance/attestations/external",
                json=payload,
            )

            if resp.status_code == 429:
                # Rate limited — retry once after delay
                retry_after = int(resp.headers.get("retry-after", "5"))
                time.sleep(min(retry_after, 60))
                resp = self._http.post(
                    f"/api/workspaces/{self.workspace_id}/governance/attestations/external",
                    json=payload,
                )

            if resp.status_code >= 400:
                return self._handle_error(
                    f"Attestation push failed: HTTP {resp.status_code} — {resp.text}"
                )

            data = resp.json()
            return AttestationResult(
                attestation_id=data["attestation_id"],
                external_verified=data["external_verified"],
                trust_tier=data["trust_tier"],
                detail=data["detail"],
            )

        except httpx.TimeoutException:
            return self._handle_error("Attestation push timed out (15s)")
        except Exception as e:
            return self._handle_error(f"Attestation push error: {e}")

    # ── Internal: GRC push ───────────────────────────────────────────────────

    def _push_grc(self, record: GrcRecord) -> GrcResult:
        """Push a single GRC evidence record synchronously."""
        payload = self._build_grc_payload(record)
        try:
            resp = self._http.post(
                f"/api/workspaces/{self.workspace_id}/governance/grc/evidence",
                json=payload,
            )
            if resp.status_code >= 400:
                return self._handle_grc_error(
                    f"GRC push failed: HTTP {resp.status_code} — {resp.text}",
                    record,
                )
            data = resp.json()
            return GrcResult(
                record_id=data["record_id"],
                record_type=data["record_type"],
                mapped_controls=data.get("mapped_controls", []),
                detail="ok",
            )
        except httpx.TimeoutException:
            return self._handle_grc_error("GRC push timed out (15s)", record)
        except Exception as e:
            return self._handle_grc_error(f"GRC push error: {e}", record)

    # ── Internal: batch queue ────────────────────────────────────────────────

    def _enqueue(self, record: AttestationRecord) -> None:
        # Drain inline when at capacity — avoids re-entering _flush_batch
        # (which would deadlock on _batch_lock since Lock is not reentrant).
        records_to_flush: List[AttestationRecord] = []
        with self._batch_lock:
            self._batch_queue.append(record)
            if len(self._batch_queue) >= self._batch_max_size:
                if self._batch_timer:
                    self._batch_timer.cancel()
                    self._batch_timer = None
                records_to_flush = self._batch_queue[:]
                self._batch_queue.clear()
            elif self._batch_timer is None:
                self._batch_timer = threading.Timer(
                    self._batch_flush_interval, self._flush_batch
                )
                self._batch_timer.daemon = True
                self._batch_timer.start()
        # Push outside the lock.
        for r in records_to_flush:
            self._push_sync(r)

    def _flush_batch(self) -> None:
        with self._batch_lock:
            if self._batch_timer:
                self._batch_timer.cancel()
                self._batch_timer = None
            records = self._batch_queue[:]
            self._batch_queue.clear()

        # Guard: HTTP client may already be closed if close() was called explicitly
        # before atexit fires. Queue is empty in that case, but be explicit.
        if self._closed:
            return

        if not records:
            return

        # Try batch endpoint first (server >= current). Fall back to per-record
        # push on 404 (older server without batch route).
        payloads = [self._build_payload(r) for r in records]
        try:
            resp = self._http.post(
                f"/api/workspaces/{self.workspace_id}/governance/attestations/batch",
                json={"records": payloads},
            )
            if resp.status_code == 404 and "/batch" in str(resp.url):
                # Server doesn't support batch yet — fall back to per-record.
                for record in records:
                    self._push_sync(record)
                return

            if resp.status_code >= 400:
                self._handle_error(
                    f"Batch push failed: HTTP {resp.status_code} — {resp.text}"
                )
                return

            data = resp.json()
            if data.get("rejected", 0) > 0:
                rejected = [r for r in data["results"] if r["status"] == "rejected"]
                import warnings
                warnings.warn(
                    f"[mima-governance] Batch: {len(rejected)} records rejected: "
                    f"{rejected[0]['reason']}{'...' if len(rejected) > 1 else ''}",
                    stacklevel=2,
                )

        except httpx.TimeoutException:
            # Batch timed out — fall back to per-record (may partially succeed).
            for record in records:
                self._push_sync(record)
        except Exception as e:
            self._handle_error(f"Batch push error: {e}")

    # ── Error handling ───────────────────────────────────────────────────────

    def _handle_error(self, message: str) -> AttestationResult:
        if self.on_error == "raise":
            raise MimaAttestationError(message)
        elif self.on_error == "warn":
            import warnings
            warnings.warn(f"[mima-governance] {message}", stacklevel=3)

        return AttestationResult(
            attestation_id="",
            external_verified=False,
            trust_tier="declared",
            detail=message,
        )

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush any pending batch items and close the HTTP client."""
        self._flush_batch()
        self._closed = True
        self._http.close()

    # __del__ intentionally omitted. atexit.register (in __init__) is the
    # correct mechanism for interpreter-shutdown flushing — __del__ call order
    # and timing are undefined, and accessing other objects from __del__ is
    # unsafe during garbage collection.


# ── Trace context ────────────────────────────────────────────────────────────


class _TraceContext:
    def __init__(self, tool_name: str, model_id: Optional[str] = None):
        self.tool_name = tool_name
        self.model_id = model_id
        self.input_hash: Optional[str] = None
        self.output_hash: Optional[str] = None

    def set_input(self, data: Any) -> None:
        self.input_hash = _sha256(_serialize_for_hash(data))

    def set_output(self, data: Any) -> None:
        self.output_hash = _sha256(_serialize_for_hash(data))

    def set_model_id(self, model_id: str) -> None:
        self.model_id = model_id


# ── Batch context ────────────────────────────────────────────────────────────


class _BatchContext:
    def __init__(self, client: MimaGovernance):
        self._client = client

    def add(
        self,
        tool_name: str,
        *,
        input: Any = None,
        output: Any = None,
        input_hash: Optional[str] = None,
        output_hash: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> None:
        ih = input_hash or _sha256(_serialize_for_hash(input))
        oh = output_hash or _sha256(_serialize_for_hash(output))
        record = AttestationRecord(
            tool_name=tool_name,
            input_hash=ih,
            output_hash=oh,
            model_id=model_id,
            executed_at=datetime.now(timezone.utc).isoformat(),
            authorised_by=self._client.authorised_by,
        )
        self._client._enqueue(record)


# ── Utilities ────────────────────────────────────────────────────────────────


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _serialize_for_hash(*args: Any, **kwargs: Any) -> str:
    """Deterministic serialization for hashing."""
    try:
        return json.dumps(args if len(args) != 1 else args[0], sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(args)
