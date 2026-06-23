"""Unit tests for the pre-approval gates SDK.

Covers all 15 success criteria from docs/specs/pre-approval-gates.md:
  1.  require_approval() creates a pending record (POST to /approvals)
  2.  Code blocks until approved (poll interval 2s→30s cap)
  3.  ApprovalDenied raised on rejection
  4.  ApprovalTimeout raised when expires_at passes (on_timeout='raise')
  5.  ApprovalToken single-use — second @mima.attest() raises ValueError
  6.  human_oversight record with oversight_status='approved', approved_by, approval_id
  7-8. Approve/Reject visible within next poll (tested via mock status transitions)
  9.  Expired approvals: server returns expired → SDK raises ApprovalTimeout
  10. on_timeout='warn' returns TimeoutToken without raising
  11. TimeoutToken produces oversight_status='timeout_unblocked', not 'approved'
  12. EUAIA_ART14 NOT earned by timeout_unblocked (payload check)
  13. Dashboard indicator is out-of-scope here; payload has oversight_status
  14. AsyncMimaGovernance.require_approval passes same suite (async variants below)
  15. /decide rejects expired — not tested here (backend responsibility)
"""

from __future__ import annotations

import asyncio
import warnings
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import respx
import httpx

from mima_governance.approvals import (
    ApprovalCancelled,
    ApprovalDenied,
    ApprovalTimeout,
    ApprovalToken,
    MimaGovernanceError,
    TimeoutToken,
    _handle_timeout,
    _next_interval,
    poll_approval_sync,
    poll_approval_async,
)
from mima_governance.client import MimaGovernance
from mima_governance.async_client import AsyncMimaGovernance


# ── fixtures ───────────────────────────────────────────────────────────────────

BASE = "https://api.mima.ai"
WS   = "ws-test-123"
APPR = "appr-abc-456"

PENDING_RESP  = {"approval_id": APPR, "status": "pending",  "action_type": "high_risk_ai_decision", "expires_at": "2026-12-31T00:00:00Z"}
APPROVED_RESP = {"approval_id": APPR, "status": "approved", "action_type": "high_risk_ai_decision", "approved_by": "grc@example.com", "decided_at": "2026-06-20T10:00:00Z", "expires_at": "2026-12-31T00:00:00Z"}
REJECTED_RESP = {"approval_id": APPR, "status": "rejected", "approved_by": "grc@example.com", "rejection_reason": "Too risky", "expires_at": "2026-12-31T00:00:00Z"}
EXPIRED_RESP  = {"approval_id": APPR, "status": "expired",  "action_type": "high_risk_ai_decision", "expires_at": "2026-06-20T09:00:00Z"}
CANCELLED_RESP = {"approval_id": APPR, "status": "cancelled", "expires_at": "2026-12-31T00:00:00Z"}

CREATE_RESP = {"approval_id": APPR, "status": "pending", "expires_at": "2026-12-31T00:00:00Z"}
GRC_RESP    = {"record_id": "rec-001", "record_type": "human_oversight", "mapped_controls": ["EUAIA_ART14", "EUAIA_ART13"]}
ATTEST_RESP = {"attestation_id": "att-001", "external_verified": True, "trust_tier": "attested", "detail": "ok"}


def _approval_token(used: bool = False) -> ApprovalToken:
    return ApprovalToken(
        approval_id=APPR,
        action_type="high_risk_ai_decision",
        approved_by="grc@example.com",
        approved_at=datetime(2026, 6, 20, 10, 0, tzinfo=timezone.utc),
        expires_at=datetime(2026, 12, 31, tzinfo=timezone.utc),
        _used=used,
    )


def _timeout_token() -> TimeoutToken:
    return TimeoutToken(
        approval_id=APPR,
        action_type="high_risk_ai_decision",
        timed_out_at=datetime.now(timezone.utc),
        timeout_seconds=300,
    )


# ── _next_interval ─────────────────────────────────────────────────────────────


class TestPollIntervals:
    def test_first_five_intervals(self) -> None:
        assert _next_interval(0) == 2
        assert _next_interval(1) == 4
        assert _next_interval(2) == 8
        assert _next_interval(3) == 16
        assert _next_interval(4) == 30

    def test_caps_at_30(self) -> None:
        assert _next_interval(5) == 30
        assert _next_interval(99) == 30


# ── ApprovalToken single-use ───────────────────────────────────────────────────


class TestApprovalTokenSingleUse:
    def test_first_use_succeeds(self) -> None:
        token = _approval_token()
        token._mark_used()
        assert token._used is True

    def test_second_use_raises_value_error(self) -> None:
        token = _approval_token()
        token._mark_used()
        with pytest.raises(ValueError, match="single-use"):
            token._mark_used()

    def test_already_used_token_raises_immediately(self) -> None:
        token = _approval_token(used=True)
        with pytest.raises(ValueError):
            token._mark_used()


# ── _handle_timeout ────────────────────────────────────────────────────────────


class TestHandleTimeout:
    def test_raise_mode_raises_approval_timeout(self) -> None:
        with pytest.raises(ApprovalTimeout) as exc_info:
            _handle_timeout(APPR, "high_risk", 300, "raise")
        assert exc_info.value.approval_id == APPR
        assert exc_info.value.timeout_seconds == 300

    def test_warn_mode_returns_timeout_token(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            token = _handle_timeout(APPR, "high_risk", 300, "warn")
        assert isinstance(token, TimeoutToken)
        assert token.approval_id == APPR
        assert token.timeout_seconds == 300
        assert len(w) == 1
        assert "timeout_unblocked" in str(w[0].message)

    def test_warn_mode_does_not_raise(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = _handle_timeout(APPR, "high_risk", 300, "warn")
        assert result is not None


# ── poll_approval_sync ─────────────────────────────────────────────────────────


class TestPollApprovalSync:
    def _make_http(self, responses: list) -> MagicMock:
        """Stub httpx.Client.get() returning successive responses."""
        mock_http = MagicMock()
        mock_http.get.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value=r))
            for r in responses
        ]
        return mock_http

    def test_approved_on_first_poll(self) -> None:
        http = self._make_http([APPROVED_RESP])
        with patch("time.sleep"):
            token = poll_approval_sync(http, WS, APPR, 300, "raise")
        assert isinstance(token, ApprovalToken)
        assert token.approved_by == "grc@example.com"
        assert token.approval_id == APPR

    def test_pending_then_approved(self) -> None:
        http = self._make_http([PENDING_RESP, PENDING_RESP, APPROVED_RESP])
        with patch("time.sleep"):
            token = poll_approval_sync(http, WS, APPR, 300, "raise")
        assert isinstance(token, ApprovalToken)
        assert http.get.call_count == 3

    def test_rejected_raises_approval_denied(self) -> None:
        http = self._make_http([PENDING_RESP, REJECTED_RESP])
        with patch("time.sleep"), pytest.raises(ApprovalDenied) as exc_info:
            poll_approval_sync(http, WS, APPR, 300, "raise")
        assert exc_info.value.approval_id == APPR
        assert exc_info.value.rejected_by == "grc@example.com"
        assert exc_info.value.reason == "Too risky"

    def test_expired_raises_approval_timeout(self) -> None:
        http = self._make_http([EXPIRED_RESP])
        with patch("time.sleep"), pytest.raises(ApprovalTimeout) as exc_info:
            poll_approval_sync(http, WS, APPR, 300, "raise")
        assert exc_info.value.approval_id == APPR

    def test_expired_warn_returns_timeout_token(self) -> None:
        http = self._make_http([EXPIRED_RESP])
        with patch("time.sleep"), warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            token = poll_approval_sync(http, WS, APPR, 300, "warn")
        assert isinstance(token, TimeoutToken)

    def test_cancelled_raises_approval_cancelled(self) -> None:
        http = self._make_http([CANCELLED_RESP])
        with patch("time.sleep"), pytest.raises(ApprovalCancelled) as exc_info:
            poll_approval_sync(http, WS, APPR, 300, "raise")
        assert exc_info.value.approval_id == APPR

    def test_http_error_raises_governance_error(self) -> None:
        mock_http = MagicMock()
        mock_http.get.return_value = MagicMock(status_code=500, text="server error")
        with pytest.raises(MimaGovernanceError, match="500"):
            poll_approval_sync(mock_http, WS, APPR, 300, "raise")

    def test_client_side_deadline_returns_timeout_token(self) -> None:
        """If monotonic clock passes deadline before server expires, client handles it."""
        http = self._make_http([PENDING_RESP])
        # timeout_seconds=1 but we exhaust time on first pending response
        with patch("time.sleep"), patch("time.monotonic", side_effect=[0.0, 2.0, 2.0, 2.0]):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                token = poll_approval_sync(http, WS, APPR, 1, "warn")
        assert isinstance(token, TimeoutToken)

    def test_polls_correct_url(self) -> None:
        http = self._make_http([APPROVED_RESP])
        with patch("time.sleep"):
            poll_approval_sync(http, WS, APPR, 300, "raise")
        http.get.assert_called_once_with(
            f"/api/workspaces/{WS}/governance/approvals/{APPR}"
        )


# ── poll_approval_async ────────────────────────────────────────────────────────


class TestPollApprovalAsync:
    def _make_http(self, responses: list) -> MagicMock:
        mock_http = MagicMock()

        async def async_get(url):
            resp_data = responses.pop(0)
            return MagicMock(status_code=200, json=MagicMock(return_value=resp_data))

        mock_http.get = async_get
        return mock_http

    @pytest.mark.asyncio
    async def test_approved_async(self) -> None:
        http = self._make_http([APPROVED_RESP])
        with patch("asyncio.sleep"):
            token = await poll_approval_async(http, WS, APPR, 300, "raise")
        assert isinstance(token, ApprovalToken)
        assert token.approved_by == "grc@example.com"

    @pytest.mark.asyncio
    async def test_rejected_async(self) -> None:
        http = self._make_http([REJECTED_RESP])
        with patch("asyncio.sleep"), pytest.raises(ApprovalDenied):
            await poll_approval_async(http, WS, APPR, 300, "raise")

    @pytest.mark.asyncio
    async def test_expired_async_raises(self) -> None:
        http = self._make_http([EXPIRED_RESP])
        with patch("asyncio.sleep"), pytest.raises(ApprovalTimeout):
            await poll_approval_async(http, WS, APPR, 300, "raise")

    @pytest.mark.asyncio
    async def test_expired_async_warn_returns_timeout_token(self) -> None:
        http = self._make_http([EXPIRED_RESP])
        with patch("asyncio.sleep"), warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            token = await poll_approval_async(http, WS, APPR, 300, "warn")
        assert isinstance(token, TimeoutToken)

    @pytest.mark.asyncio
    async def test_cancelled_async(self) -> None:
        http = self._make_http([CANCELLED_RESP])
        with patch("asyncio.sleep"), pytest.raises(ApprovalCancelled):
            await poll_approval_async(http, WS, APPR, 300, "raise")


# ── MimaGovernance.require_approval ───────────────────────────────────────────


class TestRequireApprovalSync:
    @respx.mock
    def test_creates_approval_and_returns_token(self) -> None:
        respx.post(f"{BASE}/api/workspaces/{WS}/governance/approvals").mock(
            return_value=httpx.Response(200, json=CREATE_RESP)
        )
        respx.get(f"{BASE}/api/workspaces/{WS}/governance/approvals/{APPR}").mock(
            return_value=httpx.Response(200, json=APPROVED_RESP)
        )
        mima = MimaGovernance(workspace_id=WS, api_key="key", system_name="sys")
        with patch("time.sleep"):
            token = mima.require_approval("high_risk_ai_decision", {"x": 1})
        assert isinstance(token, ApprovalToken)
        assert token.approved_by == "grc@example.com"

    @respx.mock
    def test_passes_approver_hint(self) -> None:
        post_route = respx.post(f"{BASE}/api/workspaces/{WS}/governance/approvals").mock(
            return_value=httpx.Response(200, json=CREATE_RESP)
        )
        respx.get(f"{BASE}/api/workspaces/{WS}/governance/approvals/{APPR}").mock(
            return_value=httpx.Response(200, json=APPROVED_RESP)
        )
        mima = MimaGovernance(workspace_id=WS, api_key="key", system_name="sys")
        with patch("time.sleep"):
            mima.require_approval("act", {}, approver="grc@example.com")
        body = post_route.calls[0].request.content
        import json
        assert json.loads(body)["approver_hint"] == "grc@example.com"

    @respx.mock
    def test_create_failure_raises_governance_error(self) -> None:
        respx.post(f"{BASE}/api/workspaces/{WS}/governance/approvals").mock(
            return_value=httpx.Response(500, text="error")
        )
        mima = MimaGovernance(workspace_id=WS, api_key="key", system_name="sys")
        with pytest.raises(MimaGovernanceError, match="500"):
            mima.require_approval("act", {})

    @respx.mock
    def test_rejected_propagates_approval_denied(self) -> None:
        respx.post(f"{BASE}/api/workspaces/{WS}/governance/approvals").mock(
            return_value=httpx.Response(200, json=CREATE_RESP)
        )
        respx.get(f"{BASE}/api/workspaces/{WS}/governance/approvals/{APPR}").mock(
            return_value=httpx.Response(200, json=REJECTED_RESP)
        )
        mima = MimaGovernance(workspace_id=WS, api_key="key", system_name="sys")
        with patch("time.sleep"), pytest.raises(ApprovalDenied):
            mima.require_approval("act", {})


# ── AsyncMimaGovernance.require_approval ──────────────────────────────────────


class TestRequireApprovalAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_creates_approval_and_returns_token(self) -> None:
        respx.post(f"{BASE}/api/workspaces/{WS}/governance/approvals").mock(
            return_value=httpx.Response(200, json=CREATE_RESP)
        )
        respx.get(f"{BASE}/api/workspaces/{WS}/governance/approvals/{APPR}").mock(
            return_value=httpx.Response(200, json=APPROVED_RESP)
        )
        async with AsyncMimaGovernance(workspace_id=WS, api_key="key", system_name="sys") as mima:
            with patch("asyncio.sleep"):
                token = await mima.require_approval("high_risk_ai_decision", {"x": 1})
        assert isinstance(token, ApprovalToken)

    @respx.mock
    @pytest.mark.asyncio
    async def test_timeout_warn_returns_timeout_token(self) -> None:
        respx.post(f"{BASE}/api/workspaces/{WS}/governance/approvals").mock(
            return_value=httpx.Response(200, json=CREATE_RESP)
        )
        respx.get(f"{BASE}/api/workspaces/{WS}/governance/approvals/{APPR}").mock(
            return_value=httpx.Response(200, json=EXPIRED_RESP)
        )
        async with AsyncMimaGovernance(workspace_id=WS, api_key="key", system_name="sys") as mima:
            with patch("asyncio.sleep"), warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                token = await mima.require_approval("act", {}, on_timeout="warn")
        assert isinstance(token, TimeoutToken)


# ── @mima.attest(approval_token=...) ──────────────────────────────────────────


class TestAttestWithApprovalToken:
    @respx.mock
    def test_approval_token_pushes_human_oversight_record(self) -> None:
        respx.post(
            f"{BASE}/api/workspaces/{WS}/governance/attestations/external"
        ).mock(return_value=httpx.Response(200, json=ATTEST_RESP))
        grc_route = respx.post(
            f"{BASE}/api/workspaces/{WS}/governance/grc/evidence"
        ).mock(return_value=httpx.Response(200, json=GRC_RESP))

        token = _approval_token()
        mima = MimaGovernance(workspace_id=WS, api_key="key", system_name="sys")

        @mima.attest(tool_name="loan_decision", approval_token=token)
        def make_decision():
            return "approved"

        make_decision()

        # Confirm the GRC push happened
        assert grc_route.called
        import json
        body = json.loads(grc_route.calls[0].request.content)
        assert body["record_type"] == "human_oversight"
        assert body["payload"]["oversight_status"] == "approved"
        assert body["payload"]["approved_by"] == "grc@example.com"
        assert body["payload"]["approval_id"] == APPR

    @respx.mock
    def test_approval_token_marks_used_after_attest(self) -> None:
        respx.post(f"{BASE}/api/workspaces/{WS}/governance/attestations/external").mock(
            return_value=httpx.Response(200, json=ATTEST_RESP)
        )
        respx.post(f"{BASE}/api/workspaces/{WS}/governance/grc/evidence").mock(
            return_value=httpx.Response(200, json=GRC_RESP)
        )
        token = _approval_token()
        mima = MimaGovernance(workspace_id=WS, api_key="key", system_name="sys")

        @mima.attest(tool_name="loan_decision", approval_token=token)
        def make_decision():
            return "approved"

        make_decision()
        assert token._used is True

    def test_second_attest_call_raises_value_error(self) -> None:
        """Criterion 5: single-use token enforcement."""
        token = _approval_token(used=True)
        mima = MimaGovernance(workspace_id=WS, api_key="key", system_name="sys")

        @mima.attest(tool_name="loan_decision", approval_token=token)
        def make_decision():
            return "approved"

        with pytest.raises(ValueError, match="single-use"):
            make_decision()

    @respx.mock
    def test_timeout_token_produces_timeout_unblocked_record(self) -> None:
        """Criterion 11: TimeoutToken → oversight_status='timeout_unblocked'."""
        respx.post(f"{BASE}/api/workspaces/{WS}/governance/attestations/external").mock(
            return_value=httpx.Response(200, json=ATTEST_RESP)
        )
        grc_route = respx.post(f"{BASE}/api/workspaces/{WS}/governance/grc/evidence").mock(
            return_value=httpx.Response(200, json={
                "record_id": "rec-002",
                "record_type": "human_oversight",
                "mapped_controls": [],  # no EUAIA_ART14
            })
        )
        token = _timeout_token()
        mima = MimaGovernance(workspace_id=WS, api_key="key", system_name="sys")

        @mima.attest(tool_name="loan_decision", approval_token=token)
        def make_decision():
            return "ok"

        make_decision()

        import json
        body = json.loads(grc_route.calls[0].request.content)
        assert body["payload"]["oversight_status"] == "timeout_unblocked"
        assert "approved_by" not in body["payload"]

    @respx.mock
    def test_timeout_unblocked_does_not_earn_euaia_art14(self) -> None:
        """Criterion 12: EUAIA_ART14 absent from timeout_unblocked response."""
        respx.post(f"{BASE}/api/workspaces/{WS}/governance/attestations/external").mock(
            return_value=httpx.Response(200, json=ATTEST_RESP)
        )
        grc_route = respx.post(f"{BASE}/api/workspaces/{WS}/governance/grc/evidence").mock(
            return_value=httpx.Response(200, json={
                "record_id": "rec-003",
                "record_type": "human_oversight",
                "mapped_controls": [],
            })
        )
        token = _timeout_token()
        mima = MimaGovernance(workspace_id=WS, api_key="key", system_name="sys")

        @mima.attest(tool_name="loan_decision", approval_token=token)
        def make_decision():
            return "ok"

        make_decision()
        import json
        body = json.loads(grc_route.calls[0].request.content)
        # The payload does not include approved_by, so backend cannot award EUAIA_ART14
        assert body["payload"].get("approved_by") is None
        assert body["payload"]["oversight_status"] == "timeout_unblocked"

    def test_wrong_token_type_raises_type_error(self) -> None:
        mima = MimaGovernance(workspace_id=WS, api_key="key", system_name="sys")
        with pytest.raises(TypeError, match="ApprovalToken or TimeoutToken"):
            @mima.attest(tool_name="t", approval_token="not-a-token")
            def fn():
                return "x"

    @respx.mock
    def test_attest_without_approval_token_still_works(self) -> None:
        """Backward compatibility: approval_token=None is the default."""
        respx.post(f"{BASE}/api/workspaces/{WS}/governance/attestations/external").mock(
            return_value=httpx.Response(200, json=ATTEST_RESP)
        )
        mima = MimaGovernance(workspace_id=WS, api_key="key", system_name="sys")

        @mima.attest(tool_name="t")
        def fn():
            return "x"

        result = fn()
        assert result == "x"


# ── AsyncMimaGovernance attest with approval_token ────────────────────────────


class TestAsyncAttestWithApprovalToken:
    @respx.mock
    @pytest.mark.asyncio
    async def test_approval_token_async(self) -> None:
        respx.post(f"{BASE}/api/workspaces/{WS}/governance/attestations/external").mock(
            return_value=httpx.Response(200, json=ATTEST_RESP)
        )
        grc_route = respx.post(f"{BASE}/api/workspaces/{WS}/governance/grc/evidence").mock(
            return_value=httpx.Response(200, json=GRC_RESP)
        )
        token = _approval_token()
        async with AsyncMimaGovernance(workspace_id=WS, api_key="key", system_name="sys") as mima:
            @mima.attest(tool_name="loan_decision", approval_token=token)
            async def make_decision():
                return "ok"

            await make_decision()

        assert grc_route.called
        import json
        body = json.loads(grc_route.calls[0].request.content)
        assert body["payload"]["oversight_status"] == "approved"
        assert token._used is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_timeout_token_async_produces_timeout_unblocked(self) -> None:
        respx.post(f"{BASE}/api/workspaces/{WS}/governance/attestations/external").mock(
            return_value=httpx.Response(200, json=ATTEST_RESP)
        )
        grc_route = respx.post(f"{BASE}/api/workspaces/{WS}/governance/grc/evidence").mock(
            return_value=httpx.Response(200, json={"record_id": "r", "record_type": "human_oversight", "mapped_controls": []})
        )
        token = _timeout_token()
        async with AsyncMimaGovernance(workspace_id=WS, api_key="key", system_name="sys") as mima:
            @mima.attest(tool_name="loan_decision", approval_token=token)
            async def make_decision():
                return "ok"

            await make_decision()

        import json
        body = json.loads(grc_route.calls[0].request.content)
        assert body["payload"]["oversight_status"] == "timeout_unblocked"
        assert "approved_by" not in body["payload"]

    @pytest.mark.asyncio
    async def test_async_single_use_enforcement(self) -> None:
        token = _approval_token(used=True)
        async with AsyncMimaGovernance(workspace_id=WS, api_key="key", system_name="sys") as mima:
            @mima.attest(tool_name="t", approval_token=token)
            async def fn():
                return "x"

            with pytest.raises(ValueError, match="single-use"):
                await fn()
