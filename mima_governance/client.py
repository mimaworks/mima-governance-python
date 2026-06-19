"""Mima Governance SDK — core client with decorator, batch, and signing."""

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
        """Handle a GRC push failure according to self.on_error.

        Consistent with attestation error handling:
          "warn"   — emit warnings.warn to stderr, return empty GrcResult
          "raise"  — raise MimaAttestationError
          "silent" — return empty GrcResult with no output

        Callers should check ``result.record_id == ''`` to detect failures
        regardless of on_error mode (except "raise", which never returns).
        """
        if self.on_error == "raise":
            raise MimaAttestationError(f"[mima-governance] {message}")
        if self.on_error == "warn":
            import warnings
            warnings.warn(f"[mima-governance] {message}", stacklevel=4)
        # "silent": no output
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
        authority_notified_at: Optional[str] = None,
    ) -> GrcResult:
        """Record a security or AI incident.

        Evidences SOC2 CC7.3 / CC7.4 (incident detection and response),
        ISO 27001:2022 5.25 / 5.26 (event assessment and incident response),
        ISO 42001 A.3.2 (reporting of AI incidents), EU AI Act Art. 73
        (serious incident reporting), and SOC2 CC3.3 / CC4.2.

        Note on Art. 73: ``authority_notified_at`` records when a national
        authority was notified, but the SDK cannot verify notification occurred
        or enforce the 15-day deadline. Full Art. 73 workflow compliance
        requires an external notification system.

        Args:
            title:                 Short title for the incident.
            severity:              Must be "critical", "high", "medium", or "low".
            description:           Full description of what occurred (keyword-only).
            affected_systems:      List of system names affected (keyword-only).
            detected_at:           ISO 8601 timestamp of detection; defaults to now().
            authority_notified_at: ISO 8601 timestamp when national authority was
                                   notified under EU AI Act Art. 73. Optional.

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

        payload: Dict[str, Any] = {
            "title":            title,
            "severity":         severity,
            "description":      description,
            "affected_systems": affected_systems,
        }
        if authority_notified_at is not None:
            payload["authority_notified_at"] = authority_notified_at

        record = GrcRecord(
            record_type="incident_report",
            payload=payload,
            system_name=self.system_name,
            occurred_at=occurred,
        )
        return self._push_grc(record)

    def ai_risk_assessment(
        self,
        system_name: str,
        risk_tier: str,
        use_case: str,
        *,
        impact_domains: list,
        art5_self_assessment: bool,
        assessor: str,
        assessment_date: Optional[str] = None,
        technical_doc_url: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> GrcResult:
        """Record an AI system risk classification and assessment.

        Evidences EU AI Act Art. 9 (risk management system), Art. 11
        (technical documentation), NIST AI RMF GOVERN / MAP, ISO 42001
        A.6.1 / A.9.1, and SOC2 CC3.1 / CC3.2 / CC5.1.

        Call once per AI system, or whenever the risk classification changes.
        Not intended for per-inference calls.

        All records are retained as an audit trail of classification changes.
        The readiness scorer uses the latest record by ``occurred_at`` per system.

        Args:
            system_name:          Name of the AI system being assessed.
            risk_tier:            EU AI Act tier — must be "unacceptable",
                                  "high", "limited", or "minimal".
            use_case:             Intended use case (e.g. "credit_scoring").
            impact_domains:       Domains affected — e.g. ["employment",
                                  "credit", "housing"] (keyword-only).
            art5_self_assessment: Art. 5 — provider self-declares that no
                                  prohibited practices apply to this system.
                                  The SDK cannot detect violations; this is a
                                  self-assertion (keyword-only).
            assessor:             Identity who performed the assessment
                                  (keyword-only).
            assessment_date:      ISO 8601 date; defaults to now().
            technical_doc_url:    URL to Art. 11 technical documentation.
            notes:                Optional free-text notes.

        Returns:
            GrcResult. Check ``record_id == ''`` to detect a failed push.

        Raises:
            ValueError: If ``risk_tier`` is not one of the four valid values.
        """
        valid_tiers = ("unacceptable", "high", "limited", "minimal")
        if risk_tier not in valid_tiers:
            raise ValueError(
                f"ai_risk_assessment risk_tier must be one of {valid_tiers}, got '{risk_tier}'"
            )

        payload: Dict[str, Any] = {
            "system_name":         system_name,
            "risk_tier":           risk_tier,
            "use_case":            use_case,
            "impact_domains":      impact_domains,
            "art5_self_assessment": art5_self_assessment,
            "assessor":            assessor,
        }
        if technical_doc_url is not None:
            payload["technical_doc_url"] = technical_doc_url
        if notes is not None:
            payload["notes"] = notes

        occurred = assessment_date or datetime.now(timezone.utc).isoformat()

        record = GrcRecord(
            record_type="ai_risk_assessment",
            payload=payload,
            system_name=self.system_name,
            resource=system_name,
            occurred_at=occurred,
        )
        return self._push_grc(record)

    def training_data_governance(
        self,
        model_id: str,
        dataset_id: str,
        record_count: int,
        *,
        bias_checks_performed: bool,
        approved_by: str,
        data_sources: list,
        data_categories: list,
        approval_date: Optional[str] = None,
        known_limitations: Optional[str] = None,
    ) -> GrcResult:
        """Record governance approval for a training dataset.

        Evidences EU AI Act Art. 10 (data and data governance), NIST AI RMF
        MAP, and ISO 42001 A.5.4 / A.6.5.

        Call when a training dataset is approved for use in a model, not at
        inference time.

        Args:
            model_id:              Identifier of the model the dataset trains.
            dataset_id:            Identifier of the training dataset.
            record_count:          Number of records in the dataset.
            bias_checks_performed: Whether bias checks were run (keyword-only).
            approved_by:           Identity who approved the dataset (keyword-only).
            data_sources:          Source system identifiers — e.g.
                                   ["internal-crm", "credit-bureau"] (keyword-only).
            data_categories:       Data categories present — e.g.
                                   ["financial_history", "demographic"] (keyword-only).
            approval_date:         ISO 8601 date; defaults to now().
            known_limitations:     Optional description of known dataset limitations.

        Returns:
            GrcResult. Check ``record_id == ''`` to detect a failed push.
        """
        payload: Dict[str, Any] = {
            "model_id":              model_id,
            "dataset_id":            dataset_id,
            "record_count":          record_count,
            "bias_checks_performed": bias_checks_performed,
            "approved_by":           approved_by,
            "data_sources":          data_sources,
            "data_categories":       data_categories,
        }
        if known_limitations is not None:
            payload["known_limitations"] = known_limitations

        occurred = approval_date or datetime.now(timezone.utc).isoformat()

        record = GrcRecord(
            record_type="training_data_governance",
            payload=payload,
            system_name=self.system_name,
            resource=dataset_id,
            identity=approved_by,
            occurred_at=occurred,
        )
        return self._push_grc(record)

    def model_evaluation(
        self,
        model_id: str,
        dataset: str,
        accuracy: float,
        *,
        evaluated_by: str,
        evaluation_type: str = "quarterly",
        bias_metrics: Optional[Dict[str, Any]] = None,
        robustness_score: Optional[float] = None,
        passed_threshold: Optional[bool] = None,
        evaluation_date: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> GrcResult:
        """Record a model evaluation run.

        Evidences EU AI Act Art. 15 (accuracy / robustness), Art. 9 (ongoing
        risk management), NIST AI RMF MEASURE, ISO 42001 A.6.3 / A.9.2, and
        SOC2 CC3.2 / CC3.3 / CC4.1 / CC5.1.

        Call after each evaluation run: pre-deployment, periodic, or triggered
        by drift detection.

        Args:
            model_id:         Identifier of the model evaluated.
            dataset:          Identifier of the evaluation dataset used.
            accuracy:         Accuracy score in range 0.0–1.0.
            evaluated_by:     Identity who ran the evaluation (keyword-only).
            evaluation_type:  Must be "initial", "quarterly", or "triggered".
            bias_metrics:     Dict of bias metric name to value — e.g.
                              {"demographic_parity": 0.02}.
            robustness_score: Robustness score in range 0.0–1.0.
            passed_threshold: Whether the model passed the acceptance threshold.
            evaluation_date:  ISO 8601 date; defaults to now().
            notes:            Optional free-text notes.

        Returns:
            GrcResult. Check ``record_id == ''`` to detect a failed push.

        Raises:
            ValueError: If ``evaluation_type`` is not one of the three valid values.
        """
        valid_types = ("initial", "quarterly", "triggered")
        if evaluation_type not in valid_types:
            raise ValueError(
                f"model_evaluation evaluation_type must be one of {valid_types}, "
                f"got '{evaluation_type}'"
            )

        payload: Dict[str, Any] = {
            "model_id":        model_id,
            "dataset":         dataset,
            "accuracy":        accuracy,
            "evaluated_by":    evaluated_by,
            "evaluation_type": evaluation_type,
        }
        if bias_metrics is not None:
            payload["bias_metrics"] = bias_metrics
        if robustness_score is not None:
            payload["robustness_score"] = robustness_score
        if passed_threshold is not None:
            payload["passed_threshold"] = passed_threshold
        if notes is not None:
            payload["notes"] = notes

        occurred = evaluation_date or datetime.now(timezone.utc).isoformat()

        record = GrcRecord(
            record_type="model_evaluation",
            payload=payload,
            system_name=self.system_name,
            resource=model_id,
            identity=evaluated_by,
            occurred_at=occurred,
        )
        return self._push_grc(record)

    def human_oversight(
        self,
        decision_id: str,
        ai_recommendation: str,
        human_decision: str,
        *,
        reviewer: str,
        rationale: Optional[str] = None,
        model_id: Optional[str] = None,
        override: Optional[bool] = None,
    ) -> GrcResult:
        """Record a human review of an AI decision.

        Evidences EU AI Act Art. 14 (human oversight) and Art. 13
        (transparency), NIST AI RMF GOVERN, and ISO 42001 A.6.6.

        Call when a human reviews and potentially overrides an AI decision.

        Args:
            decision_id:       External identifier for the AI decision reviewed.
            ai_recommendation: The decision or output the AI produced.
            human_decision:    The decision the human made after review.
            reviewer:          Identity of the human reviewer (keyword-only).
            rationale:         Reason for the human decision, if different from AI.
            model_id:          Identifier of the model that produced the decision.
            override:          Whether the human overrode the AI. Auto-computed
                               from differing decisions if omitted.

        Returns:
            GrcResult. Check ``record_id == ''`` to detect a failed push.
        """
        did_override = override if override is not None else (
            ai_recommendation != human_decision
        )

        payload: Dict[str, Any] = {
            "decision_id":        decision_id,
            "ai_recommendation":  ai_recommendation,
            "human_decision":     human_decision,
            "reviewer":           reviewer,
            "override":           did_override,
        }
        if rationale is not None:
            payload["rationale"] = rationale
        if model_id is not None:
            payload["model_id"] = model_id

        record = GrcRecord(
            record_type="human_oversight",
            payload=payload,
            system_name=self.system_name,
            resource=decision_id,
            identity=reviewer,
            occurred_at=datetime.now(timezone.utc).isoformat(),
        )
        return self._push_grc(record)

    def model_drift_event(
        self,
        model_id: str,
        metric: str,
        baseline: float,
        current: float,
        threshold: float,
        *,
        drift_type: str = "performance",
        detected_by: str,
        action_taken: Optional[str] = None,
        detection_date: Optional[str] = None,
    ) -> GrcResult:
        """Record a model drift detection event.

        Evidences EU AI Act Art. 72 (post-market monitoring) and Art. 9,
        NIST AI RMF MEASURE / MANAGE, ISO 42001 A.6.4, and SOC2 CC4.1 / CC4.2.

        Call when monitoring detects drift beyond a defined threshold.

        Args:
            model_id:       Identifier of the model that drifted.
            metric:         Metric that drifted — e.g. "f1_score", "accuracy",
                            "psi", or a custom metric name.
            baseline:       Baseline value of the metric at last evaluation.
            current:        Current observed value of the metric.
            threshold:      Threshold value that triggered this event.
            drift_type:     Must be "performance", "data", or "concept".
            detected_by:    Identity or system that detected the drift
                            (keyword-only).
            action_taken:   Action taken in response — e.g.
                            "retraining_scheduled", "model_paused".
            detection_date: ISO 8601 date; defaults to now().

        Returns:
            GrcResult. Check ``record_id == ''`` to detect a failed push.

        Raises:
            ValueError: If ``drift_type`` is not one of the three valid values.
        """
        valid_drift_types = ("performance", "data", "concept")
        if drift_type not in valid_drift_types:
            raise ValueError(
                f"model_drift_event drift_type must be one of {valid_drift_types}, "
                f"got '{drift_type}'"
            )

        payload: Dict[str, Any] = {
            "model_id":    model_id,
            "metric":      metric,
            "baseline":    baseline,
            "current":     current,
            "threshold":   threshold,
            "drift_type":  drift_type,
            "detected_by": detected_by,
        }
        if action_taken is not None:
            payload["action_taken"] = action_taken

        occurred = detection_date or datetime.now(timezone.utc).isoformat()

        record = GrcRecord(
            record_type="model_drift_event",
            payload=payload,
            system_name=self.system_name,
            resource=model_id,
            identity=detected_by,
            occurred_at=occurred,
        )
        return self._push_grc(record)

    def governance_review(
        self,
        reviewed_by: str,
        report_type: str,
        *,
        frameworks_reviewed: list,
        overall_readiness: int,
        action_items: int = 0,
        review_date: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> GrcResult:
        """Record a governance readiness review by a named identity.

        Evidences SOC2 CC2.1 (quality information supporting internal control)
        and CC5.2 (board oversight of internal control). Captures when a board
        member, audit committee, or named reviewer accesses and reviews the
        Mima governance readiness report.

        Args:
            reviewed_by:         Identity or role of the reviewer — e.g.
                                 "board-audit-committee" (keyword-only via
                                 positional first arg).
            report_type:         Type of review — e.g. "ai_governance_quarterly",
                                 "annual_review".
            frameworks_reviewed: List of framework slugs reviewed — e.g.
                                 ["soc2_type2", "iso_42001"] (keyword-only).
            overall_readiness:   Readiness score snapshot at time of review,
                                 0–100 (keyword-only).
            action_items:        Number of open action items noted during review.
            review_date:         ISO 8601 date; defaults to now().
            notes:               Optional free-text notes from the review.

        Returns:
            GrcResult. Check ``record_id == ''`` to detect a failed push.

        Raises:
            ValueError: If ``overall_readiness`` is not in range 0–100.
        """
        if not (0 <= overall_readiness <= 100):
            raise ValueError(
                f"governance_review overall_readiness must be 0–100, "
                f"got '{overall_readiness}'"
            )

        payload: Dict[str, Any] = {
            "reviewed_by":        reviewed_by,
            "report_type":        report_type,
            "frameworks_reviewed": frameworks_reviewed,
            "overall_readiness":  overall_readiness,
            "action_items":       action_items,
        }
        if notes is not None:
            payload["notes"] = notes

        occurred = review_date or datetime.now(timezone.utc).isoformat()

        record = GrcRecord(
            record_type="governance_review",
            payload=payload,
            system_name=self.system_name,
            identity=reviewed_by,
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

    # __del__ intentionally removed. atexit.register (in __init__) is the
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
