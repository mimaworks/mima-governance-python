"""Live-server integration tests — require a running server at MIMA_TEST_BASE_URL.

Run with:
    pytest tests/test_live_server.py --live

Skipped automatically in CI unless --live flag is passed.
"""

from __future__ import annotations

import hashlib
import os
import time
import pytest
import httpx
from mima_governance import MimaGovernance

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL     = os.getenv("MIMA_TEST_BASE_URL",  "http://127.0.0.1:8081")
WORKSPACE_ID = os.getenv("MIMA_TEST_WORKSPACE", "a92d5798-9e09-4166-a9a6-9dea300aae0e")
API_KEY      = os.getenv("MIMA_TEST_API_KEY",   "mima_live_e2etest2026lovable")

_WS = WORKSPACE_ID


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture(scope="module")
def live(request):
    if not request.config.getoption("--live"):
        pytest.skip("pass --live to run live-server tests")


@pytest.fixture(scope="module")
def client():
    return MimaGovernance(
        workspace_id=WORKSPACE_ID,
        api_key=API_KEY,
        system_name="sdk-live-test",
        base_url=BASE_URL,
        on_error="raise",
    )


@pytest.fixture(scope="module")
def http():
    return httpx.Client(
        base_url=BASE_URL,
        headers={"X-API-Key": API_KEY},
        timeout=10,
    )


# ── Health ────────────────────────────────────────────────────────────────────

class TestServerHealth:
    def test_server_is_reachable(self, live, http):
        r = http.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ── Posture / Readiness ───────────────────────────────────────────────────────

class TestPosture:
    def test_readiness_returns_score(self, live, http):
        r = http.get(f"/api/workspaces/{_WS}/governance/grc/readiness")
        assert r.status_code == 200
        data = r.json()
        # overall_pct or frameworks.score_pct
        assert "frameworks" in data or "overall_pct" in data

    def test_readiness_has_frameworks(self, live, http):
        r = http.get(f"/api/workspaces/{_WS}/governance/grc/readiness")
        assert r.status_code == 200
        data = r.json()
        frameworks = data.get("frameworks", [])
        assert len(frameworks) > 0
        fw = frameworks[0]
        assert "framework" in fw or "name" in fw
        assert "score_pct" in fw


# ── Attestation round-trip ─────────────────────────────────────────────────────

class TestAttestationRoundTrip:
    def test_push_returns_attestation_id(self, live, client):
        """push() returns an AttestationResult with a non-empty attestation_id."""
        result = client.push(
            tool_name="sdk_live_test",
            input_hash=_sha256(f"live-test-{time.time()}"),
            output_hash=_sha256("ok"),
            model_id="test-model",
        )
        assert result.attestation_id, "push() must return a non-empty attestation_id"
        assert result.trust_tier in ("full", "partial", "external", "unverified", "limited", "declared")

    def test_grc_push_returns_mapped_controls(self, live, client):
        """GRC push returns a GrcResult with mapped_controls list."""
        result = client.push_grc_raw(
            record_type="human_oversight_check",
            payload={"reviewer": "live-test", "decision": "approved"},
        ) if hasattr(client, "push_grc_raw") else client.human_oversight(
            decision_id="live-test-controls-001",
            ai_recommendation="approve",
            human_decision="approve",
            reviewer="live-test",
        )
        assert isinstance(result.mapped_controls, list)


# ── GRC evidence ──────────────────────────────────────────────────────────────

class TestGrcEvidence:
    def test_model_evaluation_push(self, live, client):
        result = client.model_evaluation(
            model_id="test-model-live",
            dataset="live-test-suite",
            accuracy=0.92,
            evaluated_by="sdk-live-test",
        )
        assert result.record_id

    def test_human_oversight_push(self, live, client):
        result = client.human_oversight(
            decision_id="live-test-decision-002",
            ai_recommendation="approve",
            human_decision="approve",
            reviewer="live-test-runner",
        )
        assert result.record_id

    def test_dry_run_returns_controls_without_writing(self, live, http):
        r = http.post(
            f"/api/workspaces/{_WS}/governance/grc/evidence",
            params={"dry_run": "true"},
            json={
                "record_type": "ai_risk_assessment",
                "payload": {"source": "live-test"},
                "system_name": "sdk-live-test",
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "mapped_controls" in data or "would_map_controls" in data
        # dry run: record_id is either empty or a nil UUID (all zeros)
        record_id = data.get("record_id", "")
        nil_uuid = "00000000-0000-0000-0000-000000000000"
        assert record_id in ("", nil_uuid) or record_id is None


# ── Gates ─────────────────────────────────────────────────────────────────────

class TestGates:
    def test_list_gates_returns_policies(self, live, http):
        r = http.get(f"/api/workspaces/{_WS}/governance/grc/gates")
        assert r.status_code == 200
        data = r.json()
        policies = data.get("policies", data) if isinstance(data, dict) else data
        assert isinstance(policies, list)

    def test_check_gates_returns_pass_status(self, live, http):
        r = http.get(f"/api/workspaces/{_WS}/governance/grc/gates/check")
        assert r.status_code == 200
        data = r.json()
        assert "passed" in data
        assert isinstance(data["passed"], bool)

    def test_gate_policies_have_required_fields(self, live, http):
        r = http.get(f"/api/workspaces/{_WS}/governance/grc/gates")
        assert r.status_code == 200
        data = r.json()
        policies = data.get("policies", []) if isinstance(data, dict) else data
        if not policies:
            pytest.skip("no gate policies configured in test workspace")
        p = policies[0]
        assert "id" in p or "policy_name" in p or "framework" in p


# ── Systems ───────────────────────────────────────────────────────────────────

class TestSystems:
    def test_list_systems_returns_list(self, live, http):
        r = http.get(f"/api/workspaces/{_WS}/governance/grc/systems")
        assert r.status_code == 200
        data = r.json()
        systems = data.get("systems", data) if isinstance(data, dict) else data
        assert isinstance(systems, list)

    def test_registered_system_has_name(self, live, http):
        r = http.get(f"/api/workspaces/{_WS}/governance/grc/systems")
        assert r.status_code == 200
        data = r.json()
        systems = data.get("systems", []) if isinstance(data, dict) else data
        if not systems:
            pytest.skip("no systems in test workspace")
        s = systems[0]
        assert "system_name" in s or "name" in s
