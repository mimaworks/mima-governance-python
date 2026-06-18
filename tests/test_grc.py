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
