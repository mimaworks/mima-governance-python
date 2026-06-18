"""Mima Governance SDK — core client with decorator, batch, and signing."""

import hashlib
import json
import functools
import time
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import httpx

from .types import AttestationRecord, AttestationResult, AuthorisedBy, GrcRecord, GrcResult


class MimaAttestationError(Exception):
    """Raised when attestation push fails in sync mode."""

    pass


class MimaGovernance:
    """
    Mima AI Governance SDK client.

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
                # Hash inputs
                input_data = _serialize_for_hash(args, kwargs)
                input_hash = _sha256(input_data)

                # Execute the function
                result = fn(*args, **kwargs)

                # Hash output
                output_hash = _sha256(_serialize_for_hash(result))

                # Build record
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

    # ── Internal: batch queue ────────────────────────────────────────────────

    def _enqueue(self, record: AttestationRecord) -> None:
        with self._batch_lock:
            self._batch_queue.append(record)
            if len(self._batch_queue) >= self._batch_max_size:
                self._flush_batch()
            elif self._batch_timer is None:
                self._batch_timer = threading.Timer(
                    self._batch_flush_interval, self._flush_batch
                )
                self._batch_timer.daemon = True
                self._batch_timer.start()

    def _flush_batch(self) -> None:
        with self._batch_lock:
            if self._batch_timer:
                self._batch_timer.cancel()
                self._batch_timer = None
            records = self._batch_queue[:]
            self._batch_queue.clear()

        for record in records:
            self._push_sync(record)

    # ── Internal: payload construction ───────────────────────────────────────

    def _build_payload(self, record: AttestationRecord) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "system_name": self.system_name,
            "agent_name": self.agent_name,
            "tool_name": record.tool_name,
            "input_hash": record.input_hash,
            "output_hash": record.output_hash,
            "schema_version": 2,
        }

        if record.model_id:
            payload["model_id"] = record.model_id
        if record.executed_at:
            payload["executed_at"] = record.executed_at
        if record.authorised_by:
            payload["authorised_by"] = record.authorised_by.to_dict()

        # Ed25519 signing if key is configured
        if self.signing_key:
            sig, vk_hex = self._sign(record)
            payload["witness_sig"] = sig
            payload["verifying_key_hex"] = vk_hex

        return payload

    def _sign(self, record: AttestationRecord) -> tuple:
        """Sign the attestation with Ed25519."""
        from nacl.signing import SigningKey

        signing_key = SigningKey(self.signing_key)
        message = f"{record.input_hash}:{record.output_hash}:{record.executed_at}"
        signed = signing_key.sign(message.encode())
        sig_hex = signed.signature.hex()
        vk_hex = signing_key.verify_key.encode().hex()
        return sig_hex, vk_hex

    # ── GRC: internal helpers ────────────────────────────────────────────────

    def _build_grc_payload(self, record: GrcRecord) -> dict:
        """Serialize a GrcRecord into the wire format, dropping None fields."""
        base = {
            "record_type": record.record_type,
            "payload":     record.payload,
            "system_name": record.system_name,
        }
        optional = {
            "identity":    record.identity,
            "resource":    record.resource,
            "environment": record.environment,
            "occurred_at": record.occurred_at,
        }
        return {**base, **{k: v for k, v in optional.items() if v is not None}}

    def _push_grc(self, record: GrcRecord) -> GrcResult:
        """Push a single GRC evidence record synchronously.

        GRC evidence is always synchronous — missed control evidence is a
        compliance gap, not a retryable metric.

        Check ``result.record_id == ''`` to detect a failed push.
        """
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

    def _handle_grc_error(self, message: str, record: GrcRecord) -> GrcResult:
        """Warn on stderr and return an empty GrcResult. Never raises for GRC errors."""
        import warnings
        warnings.warn(f"[mima-governance] {message}", stacklevel=4)
        return GrcResult(
            record_id="",
            record_type=record.record_type,
            mapped_controls=[],
            detail=message,
        )

    # ── GRC: public methods ──────────────────────────────────────────────────

    def access_review(
        self,
        user: str,
        resource: str,
        granted: bool,
        *,
        reviewed_by: str,
        review_type: str = "periodic",
        reason: Optional[str] = None,
    ) -> GrcResult:
        """Record an access review decision.

        Evidences SOC2 CC6.1 / CC6.2 / CC6.3 (Logical Access Controls) and
        ISO 27001:2022 5.16 / 5.18, plus ISO 42001 A.9.2.

        Args:
            user:        The identity whose access was reviewed.
            resource:    The system or resource the access applies to.
            granted:     Whether access was granted (True) or revoked (False).
            reviewed_by: Identity of the reviewer (keyword-only).
            review_type: "periodic" (default), "triggered", or "offboarding".
            reason:      Optional justification for the decision.

        Returns:
            GrcResult. Check ``record_id == ''`` to detect a failed push.
        """
        payload: Dict[str, Any] = {
            "user":        user,
            "resource":    resource,
            "granted":     granted,
            "reviewed_by": reviewed_by,
            "review_type": review_type,
        }
        if reason is not None:
            payload["reason"] = reason

        record = GrcRecord(
            record_type="access_review",
            payload=payload,
            system_name=self.system_name,
            identity=user,
            resource=resource,
            occurred_at=datetime.now(timezone.utc).isoformat(),
        )
        return self._push_grc(record)

    def change_event(
        self,
        type: str,
        by: str,
        description: str,
        *,
        environment: str,
        system: str,
        change_id: Optional[str] = None,
    ) -> GrcResult:
        """Record a system change event.

        Evidences SOC2 CC8.1 (Change Management), ISO 27001:2022 8.32, and
        ISO 42001 A.6.2 (AI system lifecycle change control).

        Args:
            type:        Change type (e.g. "deployment", "config", "schema").
            by:          Identity who made the change.
            description: Human-readable description of what changed.
            environment: Target environment (keyword-only, e.g. "production").
            system:      Name of the system that was changed (keyword-only).
            change_id:   Optional ticket / change-request ID.

        Returns:
            GrcResult. Check ``record_id == ''`` to detect a failed push.
        """
        payload: Dict[str, Any] = {
            "type":        type,
            "by":          by,
            "description": description,
            "environment": environment,
            "system":      system,
        }
        if change_id is not None:
            payload["change_id"] = change_id

        record = GrcRecord(
            record_type="change_event",
            payload=payload,
            system_name=self.system_name,
            identity=by,
            resource=system,
            environment=environment,
            occurred_at=datetime.now(timezone.utc).isoformat(),
        )
        return self._push_grc(record)

    def vendor_risk(
        self,
        vendor: str,
        tier: str,
        *,
        last_reviewed: str,
        findings: int = 0,
        contacts: Optional[list] = None,
    ) -> GrcResult:
        """Record a vendor risk assessment.

        Evidences SOC2 CC9.2 (Vendor/Partner Risk Assessment), ISO 27001:2022
        5.19 / 5.22 (supplier relationships and monitoring), and ISO 42001 A.10.3.

        Args:
            vendor:       Name of the vendor.
            tier:         Risk tier — must be "critical", "high", "medium", or "low".
            last_reviewed: ISO 8601 date when the assessment was completed (keyword-only).
            findings:     Number of open findings from the review.
            contacts:     Optional list of vendor contact identities.

        Returns:
            GrcResult. Check ``record_id == ''`` to detect a failed push.

        Raises:
            ValueError: If ``tier`` is not one of the four valid values.
        """
        valid_tiers = ("critical", "high", "medium", "low")
        if tier not in valid_tiers:
            raise ValueError(
                f"vendor_risk tier must be one of {valid_tiers}, got '{tier}'"
            )

        payload: Dict[str, Any] = {
            "vendor":        vendor,
            "tier":          tier,
            "last_reviewed": last_reviewed,
            "findings":      findings,
        }
        if contacts is not None:
            payload["contacts"] = contacts

        record = GrcRecord(
            record_type="vendor_risk",
            payload=payload,
            system_name=self.system_name,
            resource=vendor,
            occurred_at=datetime.now(timezone.utc).isoformat(),
        )
        return self._push_grc(record)

    def policy_acknowledged(
        self,
        policy: str,
        user: str,
        *,
        version: str,
        channel: str = "in-app",
        session_id: Optional[str] = None,
    ) -> GrcResult:
        """Record a policy acknowledgment by a user.

        Evidences SOC2 CC1.4 / CC5.3 (Commitment to Competence; deploying controls
        through policy), ISO 27001:2022 6.3 (awareness and training), and
        ISO 42001 A.2.2 (responsibility for AI objectives).

        Args:
            policy:     Name or identifier of the policy acknowledged.
            user:       Identity of the user who acknowledged.
            version:    Policy version acknowledged (keyword-only).
            channel:    How acknowledgment was captured (keyword-only, default "in-app").
            session_id: Optional session ID for full audit trail reconstruction.

        Returns:
            GrcResult. Check ``record_id == ''`` to detect a failed push.
        """
        payload: Dict[str, Any] = {
            "policy":  policy,
            "user":    user,
            "version": version,
            "channel": channel,
        }
        if session_id is not None:
            payload["session_id"] = session_id

        record = GrcRecord(
            record_type="policy_acknowledged",
            payload=payload,
            system_name=self.system_name,
            identity=user,
            resource=policy,
            occurred_at=datetime.now(timezone.utc).isoformat(),
        )
        return self._push_grc(record)

    def incident_report(
        self,
        title: str,
        severity: str,
        *,
        description: str,
        affected_systems: list,
        detected_at: Optional[str] = None,
    ) -> GrcResult:
        """Record a security or AI incident.

        Evidences SOC2 CC7.3 / CC7.4 (incident detection and response),
        ISO 27001:2022 5.25 / 5.26 (event assessment and incident response),
        and ISO 42001 A.3.2 (reporting of AI incidents).

        Args:
            title:            Short title for the incident.
            severity:         Must be "critical", "high", "medium", or "low".
            description:      Full description of what occurred (keyword-only).
            affected_systems: List of system names affected (keyword-only).
            detected_at:      ISO 8601 timestamp of detection; defaults to now().

        Returns:
            GrcResult. Check ``record_id == ''`` to detect a failed push.

        Raises:
            ValueError: If ``severity`` is not one of the four valid values.
        """
        valid_severities = ("critical", "high", "medium", "low")
        if severity not in valid_severities:
            raise ValueError(
                f"incident_report severity must be one of {valid_severities}, got '{severity}'"
            )

        occurred = detected_at or datetime.now(timezone.utc).isoformat()

        record = GrcRecord(
            record_type="incident_report",
            payload={
                "title":            title,
                "severity":         severity,
                "description":      description,
                "affected_systems": affected_systems,
            },
            system_name=self.system_name,
            occurred_at=occurred,
        )
        return self._push_grc(record)

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
        self._http.close()

    def __del__(self) -> None:
        try:
            self._flush_batch()
        except Exception:
            pass


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
