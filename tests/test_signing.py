"""Tests for GRC evidence record signing (build item #6).

HMAC-SHA256 client-side signing on `_build_grc_payload` — stdlib only, no new dep.

Acceptance criteria:
  S1  With signing_key set → payload includes client_sig (hex) + client_sig_algo="hmac-sha256"
  S2  Without signing_key → neither field present
  S3  Signature is verifiable with stdlib hmac.compare_digest
  S4  Canonical message is deterministic: same inputs → same sig across calls
  S5  Different signing_key → different signature
  S6  Different record content → different signature (sig covers payload)
  S7  Async client also signs when signing_key is set
  S8  GRC method (access_review) signs at the SDK level when signing_key configured
"""

import hashlib
import hmac
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import respx
import httpx

from mima_governance._base import _MimaGrcMixin
from mima_governance.types import GrcRecord

_KEY = b"test-signing-key-32bytes-padded!!"


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_client(key=_KEY, **kwargs):
    """Build a sync MimaGovernance client with a mock HTTP transport."""
    from mima_governance.client import MimaGovernance
    m = MimaGovernance(
        workspace_id="ws-uuid-1234",
        api_key="mima_ext_test",
        system_name="test-system",
        signing_key=key,
        **kwargs,
    )
    return m


def _make_async_client(key=_KEY):
    """Build an async AsyncMimaGovernance client."""
    from mima_governance.async_client import AsyncMimaGovernance  # noqa: F401
    return AsyncMimaGovernance(
        workspace_id="ws-uuid-1234",
        api_key="mima_ext_test",
        system_name="test-system",
        signing_key=key,
    )


def _record(record_type="access_review", **overrides) -> GrcRecord:
    defaults = dict(
        record_type=record_type,
        payload={"user": "alice@example.com", "resource": "prod-db", "granted": True, "reviewed_by": "bob@example.com"},
        system_name="test-system",
        occurred_at="2026-06-20T12:00:00+00:00",
    )
    defaults.update(overrides)
    return GrcRecord(**defaults)


def _canonical(record: GrcRecord, wire: dict, workspace_id: str) -> bytes:
    """Reproduce the canonical message the SDK uses to sign."""
    msg = json.dumps({
        "occurred_at":  wire.get("occurred_at", ""),
        "payload":      record.payload,
        "record_type":  record.record_type,
        "system_name":  record.system_name,
        "workspace_id": workspace_id,
    }, sort_keys=True, separators=(",", ":"))
    return msg.encode()


# ── S1: signing_key set → fields present ─────────────────────────────────────

class TestSigningPresent:
    def test_client_sig_in_payload(self):
        c = _make_client(_KEY)
        rec = _record()
        wire = c._build_grc_payload(rec)
        assert "client_sig" in wire
        assert "client_sig_algo" in wire

    def test_algo_is_hmac_sha256(self):
        c = _make_client(_KEY)
        wire = c._build_grc_payload(_record())
        assert wire["client_sig_algo"] == "hmac-sha256"

    def test_client_sig_is_64char_hex(self):
        c = _make_client(_KEY)
        wire = c._build_grc_payload(_record())
        sig = wire["client_sig"]
        assert isinstance(sig, str)
        assert len(sig) == 64  # HMAC-SHA256 = 32 bytes = 64 hex chars
        int(sig, 16)  # must be valid hex


# ── S2: no signing_key → fields absent ────────────────────────────────────────

class TestSigningAbsent:
    def test_no_client_sig_without_key(self):
        c = _make_client(None)
        wire = c._build_grc_payload(_record())
        assert "client_sig" not in wire
        assert "client_sig_algo" not in wire


# ── S3: verifiable with stdlib ────────────────────────────────────────────────

class TestSigningVerifiable:
    def test_hmac_verifiable(self):
        c = _make_client(_KEY)
        rec = _record()
        wire = c._build_grc_payload(rec)

        canonical = _canonical(rec, wire, c.workspace_id)
        expected = hmac.new(_KEY, canonical, hashlib.sha256).hexdigest()
        assert hmac.compare_digest(expected, wire["client_sig"])

    def test_wrong_key_fails_verification(self):
        c = _make_client(_KEY)
        rec = _record()
        wire = c._build_grc_payload(rec)

        wrong_key = b"wrong-key-for-verification-test!!"
        canonical = _canonical(rec, wire, c.workspace_id)
        wrong_sig = hmac.new(wrong_key, canonical, hashlib.sha256).hexdigest()
        # Different key → different sig → compare_digest returns False
        assert not hmac.compare_digest(wrong_sig, wire["client_sig"])


# ── S4: deterministic ─────────────────────────────────────────────────────────

class TestSigningDeterministic:
    def test_same_inputs_same_sig(self):
        c = _make_client(_KEY)
        rec = _record()
        sig1 = c._build_grc_payload(rec)["client_sig"]
        sig2 = c._build_grc_payload(rec)["client_sig"]
        assert sig1 == sig2


# ── S5: different key → different sig ────────────────────────────────────────

class TestSigningKeyIsolation:
    def test_different_keys_produce_different_sigs(self):
        key_a = b"key-a-signing-test-32bytes-pad!!"
        key_b = b"key-b-signing-test-32bytes-pad!!"
        rec = _record()

        sig_a = _make_client(key_a)._build_grc_payload(rec)["client_sig"]
        sig_b = _make_client(key_b)._build_grc_payload(rec)["client_sig"]
        assert sig_a != sig_b


# ── S6: different content → different sig ────────────────────────────────────

class TestSigningContentBinding:
    def test_payload_change_changes_sig(self):
        c = _make_client(_KEY)
        rec_a = _record(payload={"user": "alice@example.com", "resource": "db-a", "granted": True, "reviewed_by": "bob@example.com"})
        rec_b = _record(payload={"user": "alice@example.com", "resource": "db-b", "granted": True, "reviewed_by": "bob@example.com"})
        sig_a = c._build_grc_payload(rec_a)["client_sig"]
        sig_b = c._build_grc_payload(rec_b)["client_sig"]
        assert sig_a != sig_b

    def test_record_type_change_changes_sig(self):
        c = _make_client(_KEY)
        rec_a = _record("access_review")
        rec_b = _record("change_event", payload={"by": "ci-bot", "description": "Deploy", "environment": "prod", "system": "api"})
        sig_a = c._build_grc_payload(rec_a)["client_sig"]
        sig_b = c._build_grc_payload(rec_b)["client_sig"]
        assert sig_a != sig_b

    def test_workspace_id_in_canonical(self):
        """Different workspace → different sig even with identical record."""
        from mima_governance.client import MimaGovernance
        rec = _record()

        c_ws1 = MimaGovernance(workspace_id="ws-1111", api_key="k", system_name="s", signing_key=_KEY)
        c_ws2 = MimaGovernance(workspace_id="ws-2222", api_key="k", system_name="s", signing_key=_KEY)
        sig1 = c_ws1._build_grc_payload(rec)["client_sig"]
        sig2 = c_ws2._build_grc_payload(rec)["client_sig"]
        assert sig1 != sig2


# ── S7: async client also signs ───────────────────────────────────────────────

class TestSigningAsync:
    def test_async_client_signs(self):
        c = _make_async_client(_KEY)
        wire = c._build_grc_payload(_record())
        assert "client_sig" in wire
        assert wire["client_sig_algo"] == "hmac-sha256"

    def test_async_client_no_key_no_sig(self):
        c = _make_async_client(None)
        wire = c._build_grc_payload(_record())
        assert "client_sig" not in wire


# ── S8: SDK-level method (access_review) signs ───────────────────────────────

class TestSigningEndToEnd:
    @respx.mock
    def test_access_review_sends_signed_payload(self):
        from mima_governance.client import MimaGovernance

        route = respx.post("https://api.mima.ai/api/workspaces/ws-uuid-1234/governance/grc/evidence").mock(
            return_value=httpx.Response(200, json={
                "record_id": "rec-001",
                "record_type": "access_review",
                "mapped_controls": ["SOC2_CC6.1"],
            })
        )

        c = MimaGovernance(
            workspace_id="ws-uuid-1234",
            api_key="mima_ext_test",
            system_name="test-system",
            signing_key=_KEY,
            on_error="raise",
        )
        result = c.access_review("alice@example.com", "prod-db", granted=True, reviewed_by="bob@example.com")

        assert route.called
        body = route.calls[0].request.content
        sent = json.loads(body)
        assert "client_sig" in sent
        assert sent["client_sig_algo"] == "hmac-sha256"
        assert result.record_id == "rec-001"
