"""Shared validation, payload-building, and GRC methods for sync and async clients."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from nacl.signing import SigningKey as _NaclSigningKey
except ImportError:  # pragma: no cover
    _NaclSigningKey = None  # type: ignore[assignment,misc]

from .types import AttestationRecord, GrcRecord, GrcResult


class MimaAttestationError(Exception):
    """Raised when an attestation or GRC push fails with on_error='raise'."""

    pass


class _MimaGrcMixin:
    """
    Pure-logic mixin — no I/O.

    Subclasses must provide as instance attributes:
        workspace_id: str
        system_name: str
        agent_name: str
        signing_key: Optional[bytes]
        on_error: str
        authorised_by: Optional[AuthorisedBy]

    And implement:
        _push_grc(record: GrcRecord) -> GrcResult  (sync or coroutine)
    """

    # ── Payload builders ──────────────────────────────────────────────────────

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

    def _build_payload(self, record: AttestationRecord) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "system_name":    self.system_name,
            "agent_name":     self.agent_name,
            "tool_name":      record.tool_name,
            "input_hash":     record.input_hash,
            "output_hash":    record.output_hash,
            "schema_version": 2,
        }
        if record.model_id:
            payload["model_id"] = record.model_id
        if record.executed_at:
            payload["executed_at"] = record.executed_at
        if record.authorised_by:
            payload["authorised_by"] = record.authorised_by.to_dict()
        if self.signing_key:
            sig, vk_hex = self._sign(record)
            payload["witness_sig"] = sig
            payload["verifying_key_hex"] = vk_hex
        return payload

    def _sign(self, record: AttestationRecord) -> tuple:
        """Sign the attestation with Ed25519."""
        if _NaclSigningKey is None:  # pragma: no cover
            raise RuntimeError("pynacl is required for Ed25519 signing — pip install pynacl")
        signing_key = _NaclSigningKey(self.signing_key)
        message = f"{record.input_hash}:{record.output_hash}:{record.executed_at}"
        signed = signing_key.sign(message.encode())
        return signed.signature.hex(), signing_key.verify_key.encode().hex()

    # ── GRC error handling ────────────────────────────────────────────────────

    def _handle_grc_error(self, message: str, record: GrcRecord) -> GrcResult:
        """Handle a GRC push failure according to self.on_error."""
        if self.on_error == "raise":
            raise MimaAttestationError(f"[mima-governance] {message}")
        if self.on_error == "warn":
            warnings.warn(f"[mima-governance] {message}", stacklevel=4)
        return GrcResult(
            record_id="",
            record_type=record.record_type,
            mapped_controls=[],
            detail=message,
        )

    # ── GRC public methods ────────────────────────────────────────────────────

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
        """Record an access review decision."""
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
        """Record a system change event."""
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
        """Record a vendor risk assessment."""
        valid_tiers = ("critical", "high", "medium", "low")
        if tier not in valid_tiers:
            raise ValueError(f"vendor_risk tier must be one of {valid_tiers}, got '{tier}'")
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
        """Record a policy acknowledgment by a user."""
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
        """Record a security or AI incident."""
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
        """Record an AI system risk classification and assessment."""
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
        """Record governance approval for a training dataset."""
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
        """Record a model evaluation run."""
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
        """Record a human review of an AI decision."""
        did_override = override if override is not None else (
            ai_recommendation != human_decision
        )
        payload: Dict[str, Any] = {
            "decision_id":       decision_id,
            "ai_recommendation": ai_recommendation,
            "human_decision":    human_decision,
            "reviewer":          reviewer,
            "override":          did_override,
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
        """Record a model drift detection event."""
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
        """Record a governance readiness review by a named identity."""
        if not (0 <= overall_readiness <= 100):
            raise ValueError(
                f"governance_review overall_readiness must be 0–100, "
                f"got '{overall_readiness}'"
            )
        payload: Dict[str, Any] = {
            "reviewed_by":         reviewed_by,
            "report_type":         report_type,
            "frameworks_reviewed": frameworks_reviewed,
            "overall_readiness":   overall_readiness,
            "action_items":        action_items,
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
