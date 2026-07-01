"""Live-server integration tests — require a running server at MIMA_TEST_BASE_URL.

Run with:
    pytest tests/test_live_server.py --live

Skipped automatically in CI unless --live flag is passed.
"""

from __future__ import annotations

import os
import time
import pytest
from mima_governance import MimaGovernance

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL     = os.getenv("MIMA_TEST_BASE_URL",  "http://127.0.0.1:8081")
WORKSPACE_ID = os.getenv("MIMA_TEST_WORKSPACE", "a92d5798-9e09-4166-a9a6-9dea300aae0e")
API_KEY      = os.getenv("MIMA_TEST_API_KEY",   "mima_live_e2etest2026lovable")


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False,
                     help="Run live-server integration tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires a running Mima server")


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


# ── Health ────────────────────────────────────────────────────────────────────

class TestServerHealth:
    def test_server_is_reachable(self, live, client):
        import httpx
        r = httpx.get(f"{BASE_URL}/api/health", timeout=5)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ── Posture ───────────────────────────────────────────────────────────────────

class TestPosture:
    def test_get_posture_returns_score(self, live, client):
        p = client.get_posture()
        assert isinstance(p.overall_pct, (int, float))
        assert 0 <= p.overall_pct <= 100

    def test_posture_has_frameworks(self, live, client):
        p = client.get_posture()
        assert len(p.frameworks) > 0
        fw = p.frameworks[0]
        assert hasattr(fw, "name")
        assert hasattr(fw, "score_pct")


# ── Attestation round-trip ─────────────────────────────────────────────────────

class TestAttestationRoundTrip:
    def test_attest_and_verify_in_evidence(self, live, client):
        """Push a record and confirm it appears in list_evidence within 3s."""
        tag = f"live-test-{int(time.time())}"

        result = client.attest(
            tool_name="sdk_live_test",
            input_data={"tag": tag, "test": "round_trip"},
            output_data={"status": "ok"},
            model_id="test-model",
        )

        assert result.record_id, "attest() must return a non-empty record_id"

        # Poll up to 3s for the record to appear
        deadline = time.time() + 3
        found = False
        while time.time() < deadline:
            evidence = client.list_evidence(system_name="sdk-live-test", days=1)
            if any(e.get("record_id") == result.record_id for e in evidence):
                found = True
                break
            time.sleep(0.4)

        assert found, f"record {result.record_id} not found in evidence after 3s"

    def test_attest_returns_mapped_controls(self, live, client):
        result = client.attest(
            tool_name="human_oversight_check",
            input_data={"reviewer": "live-test"},
            output_data={"decision": "approved"},
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
            decision="approved",
            reviewed_by="live-test-runner",
            context="SDK live test",
        )
        assert result.record_id

    def test_dry_run_returns_controls_without_writing(self, live, client):
        result = client.dry_run_attest(
            tool_name="ai_risk_assessment",
            input_data={"source": "live-test"},
            output_data={"risk": "low"},
        )
        assert isinstance(result.would_map_controls, list)
        assert result.record_id == ""  # dry run — nothing written


# ── Gates ─────────────────────────────────────────────────────────────────────

class TestGates:
    def test_check_gates_returns_list(self, live, client):
        gates = client.check_gates()
        assert isinstance(gates, list)

    def test_gate_objects_have_required_fields(self, live, client):
        gates = client.check_gates()
        if not gates:
            pytest.skip("no gates configured in test workspace")
        g = gates[0]
        assert "name" in g or hasattr(g, "name")


# ── Systems ───────────────────────────────────────────────────────────────────

class TestSystems:
    def test_list_systems_returns_list(self, live, client):
        systems = client.list_systems()
        assert isinstance(systems, list)

    def test_register_system_dry_run(self, live, client):
        result = client.register_system(
            name="sdk-live-dry-run",
            description="Automated live test — dry run only",
            risk_tier="limited",
            dry_run=True,
        )
        assert result.system_id == "" or result.dry_run is True
