"""Tests for the GRC Evidence SDK methods (T5, T6, T7, T8, T9)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from mima_governance import GrcRecord, GrcResult, MimaGovernance
from mima_governance.types import GrcRecord, GrcResult


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def client():
    return MimaGovernance(
        workspace_id="test-workspace-id",
        api_key="mima_ext_test_key",
        system_name="test-system",
        base_url="https://test.mima.ai",
        on_error="warn",
    )


def _mock_grc_ok(record_type: str = "access_review") -> MagicMock:
    return MagicMock(
        status_code=200,
        json=lambda: {
            "record_id":       "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "record_type":     record_type,
            "mapped_controls": ["SOC2_CC6.1", "ISO27001_2022_5.16"],
        },
    )


# ── T5: Types ─────────────────────────────────────────────────────────────────


class TestTypes:
    def test_grc_record_constructs(self):
        r = GrcRecord("access_review", {"user": "alice"}, "sys")
        assert r.record_type == "access_review"
        assert r.payload == {"user": "alice"}
        assert r.system_name == "sys"
        assert r.identity is None

    def test_grc_record_with_optionals(self):
        r = GrcRecord(
            "vendor_risk",
            {"vendor": "acme"},
            "sys",
            identity="alice@co.com",
            resource="acme-corp",
            environment="production",
            occurred_at="2026-01-01T00:00:00Z",
        )
        assert r.identity == "alice@co.com"
        assert r.occurred_at == "2026-01-01T00:00:00Z"

    def test_grc_result_constructs(self):
        r = GrcResult(
            record_id="abc",
            record_type="incident_report",
            mapped_controls=["SOC2_CC7.3"],
            detail="ok",
        )
        assert r.record_id == "abc"
        assert "SOC2_CC7.3" in r.mapped_controls

    def test_grc_result_empty_record_id_signals_failure(self):
        r = GrcResult(record_id="", record_type="change_event", mapped_controls=[], detail="timeout")
        assert r.record_id == ""


# ── T9: _push_grc internals ───────────────────────────────────────────────────


class TestPushGrc:
    def test_push_grc_calls_correct_url(self, client):
        with patch.object(client._http, "post") as mock_post:
            mock_post.return_value = _mock_grc_ok("access_review")
            record = GrcRecord("access_review", {"user": "alice"}, "sys")
            result = client._push_grc(record)

        call_url = mock_post.call_args[0][0]
        assert "/governance/grc/evidence" in call_url
        assert "test-workspace-id" in call_url
        assert result.record_id == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    def test_push_grc_drops_none_fields(self, client):
        with patch.object(client._http, "post") as mock_post:
            mock_post.return_value = _mock_grc_ok("vendor_risk")
            record = GrcRecord("vendor_risk", {"vendor": "acme"}, "sys")
            client._push_grc(record)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert "identity" not in payload
        assert "environment" not in payload
        assert "resource" not in payload
        assert payload["record_type"] == "vendor_risk"
        assert payload["system_name"] == "sys"

    def test_push_grc_4xx_returns_empty_record_id(self, client):
        with patch.object(client._http, "post") as mock_post:
            mock_post.return_value = MagicMock(status_code=422, text="unknown record_type")
            record = GrcRecord("unknown_type", {}, "sys")
            import warnings
            with warnings.catch_warnings(record=True):
                result = client._push_grc(record)

        assert result.record_id == ""
        assert "422" in result.detail

    def test_push_grc_5xx_returns_empty_record_id(self, client):
        with patch.object(client._http, "post") as mock_post:
            mock_post.return_value = MagicMock(status_code=500, text="server error")
            record = GrcRecord("incident_report", {}, "sys")
            import warnings
            with warnings.catch_warnings(record=True):
                result = client._push_grc(record)

        assert result.record_id == ""
        assert result.detail != ""

    def test_push_grc_timeout_returns_empty_record_id(self, client):
        import httpx
        with patch.object(client._http, "post", side_effect=httpx.TimeoutException("timeout")):
            record = GrcRecord("change_event", {}, "sys")
            import warnings
            with warnings.catch_warnings(record=True):
                result = client._push_grc(record)

        assert result.record_id == ""
        assert "timed" in result.detail.lower()


# ── T6: access_review + change_event ─────────────────────────────────────────


class TestGrcMethods:
    def test_access_review_calls_push_grc(self, client):
        with patch.object(client._http, "post") as mock_post:
            mock_post.return_value = _mock_grc_ok("access_review")
            result = client.access_review(
                "alice@co.com",
                "prod-db",
                True,
                reviewed_by="bob@co.com",
            )

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["record_type"] == "access_review"
        assert payload["identity"] == "alice@co.com"
        assert payload["resource"] == "prod-db"
        assert payload["payload"]["granted"] is True
        assert payload["payload"]["reviewed_by"] == "bob@co.com"
        assert result.record_id != ""

    def test_access_review_reviewed_by_is_keyword_only(self, client):
        with pytest.raises(TypeError):
            # reviewed_by must be keyword-only
            client.access_review("alice@co.com", "prod-db", True, "bob@co.com")  # type: ignore[call-arg]

    def test_change_event_calls_push_grc(self, client):
        with patch.object(client._http, "post") as mock_post:
            mock_post.return_value = _mock_grc_ok("change_event")
            client.change_event(
                "deployment",
                "alice@co.com",
                "Deployed v2.1.0",
                environment="production",
                system="api-server",
            )

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["record_type"] == "change_event"
        assert payload["payload"]["environment"] == "production"
        assert payload["payload"]["system"] == "api-server"

    # ── T7: vendor_risk + policy_acknowledged ─────────────────────────────────

    def test_vendor_risk_invalid_tier_raises(self, client):
        with pytest.raises(ValueError, match="tier"):
            client.vendor_risk("acme", "extreme", last_reviewed="2026-01-01")

    def test_vendor_risk_valid_tiers(self, client):
        for tier in ("critical", "high", "medium", "low"):
            with patch.object(client._http, "post") as mock_post:
                mock_post.return_value = _mock_grc_ok("vendor_risk")
                client.vendor_risk("acme", tier, last_reviewed="2026-01-01")
                payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
                assert payload["payload"]["tier"] == tier

    def test_vendor_risk_does_not_call_push_on_invalid_tier(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            with pytest.raises(ValueError):
                client.vendor_risk("acme", "invalid", last_reviewed="2026-01-01")
            mock_push.assert_not_called()

    def test_policy_acknowledged_includes_session_id(self, client):
        with patch.object(client._http, "post") as mock_post:
            mock_post.return_value = _mock_grc_ok("policy_acknowledged")
            client.policy_acknowledged(
                "ai-acceptable-use-v3",
                "alice@co.com",
                version="3.0",
                session_id="sess_abc123",
            )

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["payload"]["session_id"] == "sess_abc123"
        assert payload["record_type"] == "policy_acknowledged"

    def test_policy_acknowledged_omits_session_id_when_not_supplied(self, client):
        with patch.object(client._http, "post") as mock_post:
            mock_post.return_value = _mock_grc_ok("policy_acknowledged")
            client.policy_acknowledged("policy-v1", "bob@co.com", version="1.0")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert "session_id" not in payload["payload"]

    # ── T8: incident_report ───────────────────────────────────────────────────

    def test_incident_report_invalid_severity_raises(self, client):
        with pytest.raises(ValueError, match="severity"):
            client.incident_report(
                "Auth bypass",
                "urgent",
                description="...",
                affected_systems=["api"],
            )

    def test_incident_report_does_not_call_push_on_invalid_severity(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            with pytest.raises(ValueError):
                client.incident_report("T", "urgent", description="d", affected_systems=[])
            mock_push.assert_not_called()

    def test_incident_report_detected_at_flows_to_occurred_at(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            mock_push.return_value = GrcResult("id", "incident_report", [], "ok")
            client.incident_report(
                "Test incident",
                "high",
                description="Something broke",
                affected_systems=["svc-a"],
                detected_at="2026-01-15T10:00:00Z",
            )

        record: GrcRecord = mock_push.call_args[0][0]
        assert record.occurred_at == "2026-01-15T10:00:00Z"

    def test_incident_report_defaults_occurred_at_to_now(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            mock_push.return_value = GrcResult("id", "incident_report", [], "ok")
            before = datetime.now(timezone.utc).isoformat()
            client.incident_report(
                "Test incident",
                "low",
                description="Minor issue",
                affected_systems=[],
            )

        record: GrcRecord = mock_push.call_args[0][0]
        assert record.occurred_at is not None
        assert record.occurred_at >= before

    def test_incident_report_authority_notified_at_in_payload(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            mock_push.return_value = GrcResult("id", "incident_report", [], "ok")
            client.incident_report(
                "Serious AI incident",
                "critical",
                description="Model caused harm",
                affected_systems=["loan-api"],
                authority_notified_at="2026-06-19T10:00:00Z",
            )
        record: GrcRecord = mock_push.call_args[0][0]
        assert record.payload["authority_notified_at"] == "2026-06-19T10:00:00Z"

    def test_incident_report_authority_notified_at_omitted_when_not_supplied(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            mock_push.return_value = GrcResult("id", "incident_report", [], "ok")
            client.incident_report("T", "low", description="d", affected_systems=[])
        record: GrcRecord = mock_push.call_args[0][0]
        assert "authority_notified_at" not in record.payload


# ── Phase 2: AI-specific record types ─────────────────────────────────────────


class TestAiRiskAssessment:
    def test_valid_call_sends_correct_record_type(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            mock_push.return_value = GrcResult("id", "ai_risk_assessment", [], "ok")
            client.ai_risk_assessment(
                "loan-scoring-v2",
                "high",
                "credit_scoring",
                impact_domains=["employment", "credit"],
                art5_self_assessment=True,
                assessor="j.smith@co.com",
            )
        record: GrcRecord = mock_push.call_args[0][0]
        assert record.record_type == "ai_risk_assessment"
        assert record.payload["risk_tier"] == "high"
        assert record.payload["art5_self_assessment"] is True
        assert record.payload["impact_domains"] == ["employment", "credit"]
        assert record.payload["assessor"] == "j.smith@co.com"
        assert record.resource == "loan-scoring-v2"

    def test_invalid_risk_tier_raises_value_error(self, client):
        with pytest.raises(ValueError, match="risk_tier"):
            client.ai_risk_assessment(
                "sys", "dangerous", "use",
                impact_domains=[], art5_self_assessment=True, assessor="a",
            )

    def test_invalid_risk_tier_does_not_call_push(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            with pytest.raises(ValueError):
                client.ai_risk_assessment(
                    "sys", "bad", "use",
                    impact_domains=[], art5_self_assessment=False, assessor="a",
                )
            mock_push.assert_not_called()

    def test_all_valid_risk_tiers_accepted(self, client):
        for tier in ("unacceptable", "high", "limited", "minimal"):
            with patch.object(client, "_push_grc") as mock_push:
                mock_push.return_value = GrcResult("id", "ai_risk_assessment", [], "ok")
                client.ai_risk_assessment(
                    "sys", tier, "use",
                    impact_domains=[], art5_self_assessment=True, assessor="a",
                )
                assert mock_push.call_args[0][0].payload["risk_tier"] == tier

    def test_optional_fields_included_when_supplied(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            mock_push.return_value = GrcResult("id", "ai_risk_assessment", [], "ok")
            client.ai_risk_assessment(
                "sys", "high", "use",
                impact_domains=[],
                art5_self_assessment=True,
                assessor="a",
                technical_doc_url="https://docs.example.com",
                notes="Initial classification",
            )
        record: GrcRecord = mock_push.call_args[0][0]
        assert record.payload["technical_doc_url"] == "https://docs.example.com"
        assert record.payload["notes"] == "Initial classification"

    def test_field_name_is_art5_self_assessment_not_old_name(self, client):
        import inspect
        sig = inspect.signature(client.ai_risk_assessment)
        assert "art5_self_assessment" in sig.parameters
        assert "prohibited_practices_checked" not in sig.parameters


class TestTrainingDataGovernance:
    def test_valid_call_sends_correct_record_type(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            mock_push.return_value = GrcResult("id", "training_data_governance", [], "ok")
            client.training_data_governance(
                "fraud-detector-v2",
                "credit-bureau-2026-q1",
                1_500_000,
                bias_checks_performed=True,
                approved_by="data-governance@co.com",
                data_sources=["internal-crm", "credit-bureau"],
                data_categories=["financial_history"],
            )
        record: GrcRecord = mock_push.call_args[0][0]
        assert record.record_type == "training_data_governance"
        assert record.payload["bias_checks_performed"] is True
        assert record.payload["record_count"] == 1_500_000
        assert record.identity == "data-governance@co.com"

    def test_known_limitations_omitted_when_not_supplied(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            mock_push.return_value = GrcResult("id", "training_data_governance", [], "ok")
            client.training_data_governance(
                "m", "d", 100,
                bias_checks_performed=False,
                approved_by="a",
                data_sources=[],
                data_categories=[],
            )
        assert "known_limitations" not in mock_push.call_args[0][0].payload


class TestModelEvaluation:
    def test_valid_call_sends_correct_record_type(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            mock_push.return_value = GrcResult("id", "model_evaluation", [], "ok")
            client.model_evaluation(
                "fraud-detector-v2",
                "holdout-2026-q2",
                0.94,
                evaluated_by="ml-ops@co.com",
                evaluation_type="quarterly",
                bias_metrics={"demographic_parity": 0.02},
                passed_threshold=True,
            )
        record: GrcRecord = mock_push.call_args[0][0]
        assert record.record_type == "model_evaluation"
        assert record.payload["accuracy"] == 0.94
        assert record.payload["passed_threshold"] is True
        assert record.payload["bias_metrics"] == {"demographic_parity": 0.02}

    def test_invalid_evaluation_type_raises_value_error(self, client):
        with pytest.raises(ValueError, match="evaluation_type"):
            client.model_evaluation("m", "d", 0.9, evaluated_by="a", evaluation_type="annual")

    def test_all_valid_evaluation_types_accepted(self, client):
        for et in ("initial", "quarterly", "triggered"):
            with patch.object(client, "_push_grc") as mock_push:
                mock_push.return_value = GrcResult("id", "model_evaluation", [], "ok")
                client.model_evaluation("m", "d", 0.9, evaluated_by="a", evaluation_type=et)
                assert mock_push.call_args[0][0].payload["evaluation_type"] == et


class TestHumanOversight:
    def test_valid_call_sends_correct_record_type(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            mock_push.return_value = GrcResult("id", "human_oversight", [], "ok")
            client.human_oversight(
                "loan-decision-9821",
                "deny",
                "approve",
                reviewer="j.smith@co.com",
                rationale="Income verified independently",
            )
        record: GrcRecord = mock_push.call_args[0][0]
        assert record.record_type == "human_oversight"
        assert record.payload["override"] is True
        assert record.payload["rationale"] == "Income verified independently"
        assert record.identity == "j.smith@co.com"

    def test_override_auto_computed_when_decisions_differ(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            mock_push.return_value = GrcResult("id", "human_oversight", [], "ok")
            client.human_oversight("d1", "deny", "approve", reviewer="a")
        assert mock_push.call_args[0][0].payload["override"] is True

    def test_override_false_when_decisions_match(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            mock_push.return_value = GrcResult("id", "human_oversight", [], "ok")
            client.human_oversight("d2", "approve", "approve", reviewer="a")
        assert mock_push.call_args[0][0].payload["override"] is False

    def test_explicit_override_takes_precedence(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            mock_push.return_value = GrcResult("id", "human_oversight", [], "ok")
            client.human_oversight("d3", "approve", "approve", reviewer="a", override=True)
        assert mock_push.call_args[0][0].payload["override"] is True


class TestModelDriftEvent:
    def test_valid_call_sends_correct_record_type(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            mock_push.return_value = GrcResult("id", "model_drift_event", [], "ok")
            client.model_drift_event(
                "fraud-detector-v2",
                "f1_score",
                0.92,
                0.78,
                0.85,
                drift_type="performance",
                detected_by="monitoring-bot",
                action_taken="retraining_scheduled",
            )
        record: GrcRecord = mock_push.call_args[0][0]
        assert record.record_type == "model_drift_event"
        assert record.payload["baseline"] == 0.92
        assert record.payload["current"] == 0.78
        assert record.payload["action_taken"] == "retraining_scheduled"

    def test_invalid_drift_type_raises_value_error(self, client):
        with pytest.raises(ValueError, match="drift_type"):
            client.model_drift_event("m", "f1", 0.9, 0.7, 0.8, drift_type="unknown", detected_by="a")

    def test_all_valid_drift_types_accepted(self, client):
        for dt in ("performance", "data", "concept"):
            with patch.object(client, "_push_grc") as mock_push:
                mock_push.return_value = GrcResult("id", "model_drift_event", [], "ok")
                client.model_drift_event("m", "f1", 0.9, 0.7, 0.8, drift_type=dt, detected_by="a")
                assert mock_push.call_args[0][0].payload["drift_type"] == dt

    def test_action_taken_omitted_when_not_supplied(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            mock_push.return_value = GrcResult("id", "model_drift_event", [], "ok")
            client.model_drift_event("m", "f1", 0.9, 0.7, 0.8, detected_by="a")
        assert "action_taken" not in mock_push.call_args[0][0].payload


class TestGovernanceReview:
    def test_valid_call_sends_correct_record_type(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            mock_push.return_value = GrcResult("id", "governance_review", [], "ok")
            client.governance_review(
                "board-audit-committee",
                "ai_governance_quarterly",
                frameworks_reviewed=["soc2_type2", "iso_42001"],
                overall_readiness=72,
                action_items=2,
            )
        record: GrcRecord = mock_push.call_args[0][0]
        assert record.record_type == "governance_review"
        assert record.payload["overall_readiness"] == 72
        assert record.payload["frameworks_reviewed"] == ["soc2_type2", "iso_42001"]
        assert record.payload["action_items"] == 2
        assert record.identity == "board-audit-committee"

    def test_readiness_above_100_raises_value_error(self, client):
        with pytest.raises(ValueError, match="overall_readiness"):
            client.governance_review(
                "reviewer", "quarterly",
                frameworks_reviewed=[], overall_readiness=101,
            )

    def test_readiness_below_0_raises_value_error(self, client):
        with pytest.raises(ValueError, match="overall_readiness"):
            client.governance_review(
                "reviewer", "quarterly",
                frameworks_reviewed=[], overall_readiness=-1,
            )

    def test_boundary_values_0_and_100_accepted(self, client):
        for score in (0, 100):
            with patch.object(client, "_push_grc") as mock_push:
                mock_push.return_value = GrcResult("id", "governance_review", [], "ok")
                client.governance_review(
                    "r", "q", frameworks_reviewed=[], overall_readiness=score,
                )
                assert mock_push.call_args[0][0].payload["overall_readiness"] == score

    def test_does_not_call_push_on_invalid_readiness(self, client):
        with patch.object(client, "_push_grc") as mock_push:
            with pytest.raises(ValueError):
                client.governance_review("r", "q", frameworks_reviewed=[], overall_readiness=101)
            mock_push.assert_not_called()
