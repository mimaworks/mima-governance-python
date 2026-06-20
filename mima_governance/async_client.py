"""Mima Governance SDK — async client for asyncio-based frameworks."""

from __future__ import annotations

import asyncio
import functools
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

import httpx

from ._base import MimaAttestationError, _MimaGrcMixin
from .client import _sha256, _serialize_for_hash
from .types import AttestationRecord, AttestationResult, AuthorisedBy, GrcRecord, GrcResult


class AsyncMimaGovernance(_MimaGrcMixin):
    """
    Async variant of MimaGovernance for use in asyncio event loops
    (FastAPI, Starlette, asyncio pipelines, AutoGen 0.4+).

    Use as a context manager to guarantee flush on exit:

        async with AsyncMimaGovernance(workspace_id=..., api_key=..., system_name=...) as mima:
            await mima.access_review(...)

    Or manage the lifecycle explicitly:

        mima = AsyncMimaGovernance(...)
        try:
            await mima.access_review(...)
        finally:
            await mima.close()  # flushes batch queue and closes transport
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

        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "mima-governance-python/0.1.0",
            },
            timeout=15.0,
        )

        self._batch_queue: List[AttestationRecord] = []
        # asyncio.Lock must be created inside a running loop;
        # create lazily on first use via _get_lock().
        self._batch_lock: Optional[asyncio.Lock] = None

    # ── Async context manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> "AsyncMimaGovernance":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ── Async trace context manager ───────────────────────────────────────────

    @asynccontextmanager
    async def trace(self, tool_name: str, *, model_id: Optional[str] = None):
        """
        Async context manager for explicit tracing.

        async with mima.trace("classify_document") as t:
            t.set_input(document)
            result = await model.predict(document)
            t.set_output(result)

        After the block, t.record_id holds the attestation ID from the ledger.
        """
        ctx = _AsyncTraceContext(tool_name, model_id)
        yield ctx
        record = AttestationRecord(
            tool_name=tool_name,
            input_hash=ctx.input_hash or "",
            output_hash=ctx.output_hash or "",
            model_id=ctx.model_id or model_id,
            executed_at=datetime.now(timezone.utc).isoformat(),
            authorised_by=self.authorised_by,
        )
        result = await self._push_async(record)
        ctx.record_id = result.attestation_id

    # ── Async decorator ───────────────────────────────────────────────────────

    def attest(
        self,
        tool_name: str,
        *,
        mode: str = "sync",
        model_id: Optional[str] = None,
        authorised_by: Optional[AuthorisedBy] = None,
    ) -> Callable:
        """
        Decorator for async functions only.

        @mima.attest(tool_name="classify_document")
        async def classify(doc):
            return await model.predict(doc)

        Raises TypeError if applied to a sync function — use MimaGovernance for those.
        """

        def decorator(fn: Callable) -> Callable:
            if not asyncio.iscoroutinefunction(fn):
                raise TypeError(
                    f"AsyncMimaGovernance.attest requires an async function — "
                    f"use MimaGovernance for sync functions. Got: {fn!r}"
                )

            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                import sys
                _guard = sys.modules.get("mima_governance.guard")
                _token = None
                if _guard and _guard._guard_enabled:
                    _token = _guard._set_attested_async(True)
                try:
                    input_hash = _sha256(_serialize_for_hash(args, kwargs))
                    result = await fn(*args, **kwargs)
                    output_hash = _sha256(_serialize_for_hash(result))
                finally:
                    if _guard and _guard._guard_enabled and _token is not None:
                        _guard._async_attested.reset(_token)

                record = AttestationRecord(
                    tool_name=tool_name,
                    input_hash=input_hash,
                    output_hash=output_hash,
                    model_id=model_id,
                    executed_at=datetime.now(timezone.utc).isoformat(),
                    authorised_by=authorised_by or self.authorised_by,
                )

                if mode == "batch":
                    await self._enqueue(record)
                else:
                    await self._push_async(record)

                return result

            return wrapper

        return decorator

    # ── Explicit push ─────────────────────────────────────────────────────────

    async def push(
        self,
        tool_name: str,
        input_hash: str,
        output_hash: str,
        *,
        model_id: Optional[str] = None,
        authorised_by: Optional[AuthorisedBy] = None,
    ) -> AttestationResult:
        """Push a single attestation immediately (async mode)."""
        record = AttestationRecord(
            tool_name=tool_name,
            input_hash=input_hash,
            output_hash=output_hash,
            model_id=model_id,
            executed_at=datetime.now(timezone.utc).isoformat(),
            authorised_by=authorised_by or self.authorised_by,
        )
        return await self._push_async(record)

    # ── Internal: async attestation push ──────────────────────────────────────

    async def _push_async(self, record: AttestationRecord) -> AttestationResult:
        payload = self._build_payload(record)
        try:
            resp = await self._http.post(
                f"/api/workspaces/{self.workspace_id}/governance/attestations/external",
                json=payload,
            )
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("retry-after", "5"))
                await asyncio.sleep(min(retry_after, 60))
                resp = await self._http.post(
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

    # ── Internal: async GRC push ──────────────────────────────────────────────

    async def _push_grc(self, record: GrcRecord) -> GrcResult:
        """Push a GRC evidence record asynchronously."""
        payload = self._build_grc_payload(record)
        try:
            resp = await self._http.post(
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

    # ── Internal: batch queue ─────────────────────────────────────────────────

    def _get_lock(self) -> asyncio.Lock:
        """Lazily create the asyncio.Lock inside a running event loop."""
        if self._batch_lock is None:
            self._batch_lock = asyncio.Lock()
        return self._batch_lock

    async def _enqueue(self, record: AttestationRecord) -> None:
        """Enqueue a record; flush immediately if the queue hits batch_max_size."""
        records_to_flush: List[AttestationRecord] = []
        async with self._get_lock():
            self._batch_queue.append(record)
            if len(self._batch_queue) >= self._batch_max_size:
                records_to_flush = self._batch_queue[:]
                self._batch_queue.clear()
        if records_to_flush:
            await self._flush_records(records_to_flush)

    async def _flush_batch(self) -> None:
        """Drain the batch queue and push all pending records."""
        async with self._get_lock():
            records = self._batch_queue[:]
            self._batch_queue.clear()
        await self._flush_records(records)

    async def _flush_records(self, records: List[AttestationRecord]) -> None:
        if not records:
            return
        payloads = [self._build_payload(r) for r in records]
        try:
            resp = await self._http.post(
                f"/api/workspaces/{self.workspace_id}/governance/attestations/batch",
                json={"records": payloads},
            )
            if resp.status_code == 404 and "/batch" in str(resp.url):
                for record in records:
                    await self._push_async(record)
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
            for record in records:
                await self._push_async(record)
        except Exception as e:
            self._handle_error(f"Batch push error: {e}")

    # ── Error handling ────────────────────────────────────────────────────────

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

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Flush any pending batch records and close the HTTP transport."""
        await self._flush_batch()
        await self._http.aclose()


# ── Async trace context ───────────────────────────────────────────────────────


class _AsyncTraceContext:
    """Context object yielded by AsyncMimaGovernance.trace()."""

    def __init__(self, tool_name: str, model_id: Optional[str] = None):
        self.tool_name = tool_name
        self.model_id = model_id
        self.input_hash: Optional[str] = None
        self.output_hash: Optional[str] = None
        self.record_id: Optional[str] = None  # populated by trace() on exit

    def set_input(self, data: Any) -> None:
        self.input_hash = _sha256(_serialize_for_hash(data))

    def set_output(self, data: Any) -> None:
        self.output_hash = _sha256(_serialize_for_hash(data))

    def set_model_id(self, model_id: str) -> None:
        self.model_id = model_id
