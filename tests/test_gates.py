"""Tests for `mima gates` CLI commands (build item #7).

Uses respx to mock the HTTP layer — no live API needed.

Test coverage:
  G1  gates_list_empty           — no policies → clean output, exit 0
  G2  gates_list_with_policies   — workspace-wide + per-system policies rendered
  G3  gates_check_passes         — all advisory → exit 0
  G4  gates_check_fails          — one required gate below threshold → exit 1
  G5  gates_check_json_passes    — --json output is valid JSON, passed=true, exit 0
  G6  gates_check_json_fails     — --json output, passed=false, exit 1
  G7  gates_check_system         — --system param forwarded in URL
  G8  gates_set_required         — PUT called with correct body
  G9  gates_set_advisory         — mode=advisory, no --threshold defaults to 80
  G10 gates_set_with_system      — scope = system name in body
  G11 gates_unset_found          — DELETE called for matching policy id
  G12 gates_unset_not_found      — no matching policy → exit 1, no DELETE call
  G13 gates_set_unknown_mode     — exit 2 before API call
  G14 gates_no_credentials       — missing MIMA_API_KEY → exit 1 before API call
"""
from __future__ import annotations

import json
import sys
from unittest.mock import patch

import httpx
import pytest
import respx


# ── Fixtures ──────────────────────────────────────────────────────────────────

BASE = "https://api.mima.ai/api"
WS   = "test-workspace-uuid"

EMPTY_RESPONSE = {"policies": [], "unconfigured_frameworks": ["eu_ai_act", "soc2_type2"]}

POLICIES_RESPONSE = {
    "policies": [
        {
            "id": "pol-ws-1",
            "framework": "eu_ai_act",
            "mode": "required",
            "threshold_pct": 60,
            "scope": "workspace",
            "current_pct": 48,
            "status": "failing",
            "created_by": "nora@mima.ai",
            "created_at": "2026-06-20T10:00:00Z",
        },
        {
            "id": "pol-ws-2",
            "framework": "soc2_type2",
            "mode": "advisory",
            "threshold_pct": 80,
            "scope": "workspace",
            "current_pct": 82,
            "status": "passing",
            "created_by": "nora@mima.ai",
            "created_at": "2026-06-20T10:01:00Z",
        },
        {
            "id": "pol-sys-1",
            "framework": "eu_ai_act",
            "mode": "required",
            "threshold_pct": 80,
            "scope": "inference-service",
            "current_pct": 31,
            "status": "failing",
            "created_by": "nora@mima.ai",
            "created_at": "2026-06-20T10:02:00Z",
        },
    ],
    "unconfigured_frameworks": ["iso_27001", "iso_42001", "nist_airf"],
}

CHECK_PASS_RESPONSE = {
    "passed": True,
    "system_name": None,
    "results": [
        {"framework": "soc2_type2", "mode": "advisory",  "threshold_pct": 80, "current_pct": 82, "status": "passing"},
    ],
}

CHECK_FAIL_RESPONSE = {
    "passed": False,
    "system_name": None,
    "results": [
        {"framework": "eu_ai_act",  "mode": "required", "threshold_pct": 60, "current_pct": 48, "status": "failing"},
        {"framework": "soc2_type2", "mode": "advisory",  "threshold_pct": 80, "current_pct": 82, "status": "passing"},
    ],
}

CHECK_SYSTEM_RESPONSE = {
    "passed": False,
    "system_name": "inference-service",
    "results": [
        {"framework": "eu_ai_act", "mode": "required", "threshold_pct": 80, "current_pct": 31, "status": "failing"},
    ],
}

UPSERT_RESPONSE = {
    "id": "pol-new-1",
    "framework": "eu_ai_act",
    "mode": "required",
    "threshold_pct": 60,
    "scope": "workspace",
    "current_pct": 48,
    "status": "failing",
    "created_by": "nora@mima.ai",
    "created_at": "2026-06-20T10:00:00Z",
}


def env_creds():
    """Patch env vars providing credentials (BASE includes /api path prefix)."""
    return patch.dict("os.environ", {
        "MIMA_API_KEY":      "mima_ext_test",
        "MIMA_WORKSPACE_ID": WS,
        "MIMA_BASE_URL":     BASE,
    })


# ── G1: empty list ────────────────────────────────────────────────────────────

class TestGatesList:
    @respx.mock
    def test_gates_list_empty(self, capsys):
        respx.get(f"{BASE}/workspaces/{WS}/governance/grc/gates").mock(
            return_value=httpx.Response(200, json=EMPTY_RESPONSE)
        )
        from mima_governance.gates import cmd_list
        with env_creds():
            cmd_list([])  # returns normally on success
        out = capsys.readouterr().out
        assert "Advisory by default" in out

    @respx.mock
    def test_gates_list_with_policies(self, capsys):
        respx.get(f"{BASE}/workspaces/{WS}/governance/grc/gates").mock(
            return_value=httpx.Response(200, json=POLICIES_RESPONSE)
        )
        from mima_governance.gates import cmd_list
        with env_creds():
            cmd_list([])  # returns normally on success
        out = capsys.readouterr().out
        assert "WORKSPACE-WIDE" in out
        assert "PER-SYSTEM OVERRIDES" in out
        assert "inference-service" in out
        assert "FAILING" in out
        assert "passing (advisory)" in out


# ── G3–G7: gates check ────────────────────────────────────────────────────────

class TestGatesCheck:
    @respx.mock
    def test_gates_check_passes(self, capsys):
        respx.get(f"{BASE}/workspaces/{WS}/governance/grc/gates/check").mock(
            return_value=httpx.Response(200, json=CHECK_PASS_RESPONSE)
        )
        from mima_governance.gates import cmd_check
        with env_creds():
            with pytest.raises(SystemExit) as exc:
                cmd_check([])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "PASS" in out

    @respx.mock
    def test_gates_check_fails(self, capsys):
        respx.get(f"{BASE}/workspaces/{WS}/governance/grc/gates/check").mock(
            return_value=httpx.Response(200, json=CHECK_FAIL_RESPONSE)
        )
        from mima_governance.gates import cmd_check
        with env_creds():
            with pytest.raises(SystemExit) as exc:
                cmd_check([])
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "FAIL" in out

    @respx.mock
    def test_gates_check_json_passes(self, capsys):
        respx.get(f"{BASE}/workspaces/{WS}/governance/grc/gates/check").mock(
            return_value=httpx.Response(200, json=CHECK_PASS_RESPONSE)
        )
        from mima_governance.gates import cmd_check
        with env_creds():
            with pytest.raises(SystemExit) as exc:
                cmd_check(["--json"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["passed"] is True

    @respx.mock
    def test_gates_check_json_fails(self, capsys):
        respx.get(f"{BASE}/workspaces/{WS}/governance/grc/gates/check").mock(
            return_value=httpx.Response(200, json=CHECK_FAIL_RESPONSE)
        )
        from mima_governance.gates import cmd_check
        with env_creds():
            with pytest.raises(SystemExit) as exc:
                cmd_check(["--json"])
        assert exc.value.code == 1
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["passed"] is False

    @respx.mock
    def test_gates_check_system_forwarded(self):
        route = respx.get(
            f"{BASE}/workspaces/{WS}/governance/grc/gates/check?system_name=inference-service"
        ).mock(return_value=httpx.Response(200, json=CHECK_SYSTEM_RESPONSE))
        from mima_governance.gates import cmd_check
        with env_creds():
            with pytest.raises(SystemExit):
                cmd_check(["--system", "inference-service", "--json"])
        assert route.called


# ── G8–G10: gates set ─────────────────────────────────────────────────────────

class TestGatesSet:
    @respx.mock
    def test_gates_set_required(self, capsys):
        route = respx.put(f"{BASE}/workspaces/{WS}/governance/grc/gates").mock(
            return_value=httpx.Response(200, json=UPSERT_RESPONSE)
        )
        from mima_governance.gates import cmd_set
        with env_creds():
            cmd_set(["eu_ai_act", "required", "--threshold", "60"])
        assert route.called
        body = json.loads(route.calls[0].request.content)
        assert body["framework"] == "eu_ai_act"
        assert body["mode"] == "required"
        assert body["threshold_pct"] == 60
        assert body["scope"] == "workspace"

    @respx.mock
    def test_gates_set_advisory_default_threshold(self, capsys):
        route = respx.put(f"{BASE}/workspaces/{WS}/governance/grc/gates").mock(
            return_value=httpx.Response(200, json={**UPSERT_RESPONSE, "mode": "advisory"})
        )
        from mima_governance.gates import cmd_set
        with env_creds():
            cmd_set(["soc2_type2", "advisory"])
        body = json.loads(route.calls[0].request.content)
        assert body["threshold_pct"] == 80  # default

    @respx.mock
    def test_gates_set_with_system(self):
        route = respx.put(f"{BASE}/workspaces/{WS}/governance/grc/gates").mock(
            return_value=httpx.Response(200, json=UPSERT_RESPONSE)
        )
        from mima_governance.gates import cmd_set
        with env_creds():
            cmd_set(["eu_ai_act", "required", "--threshold", "80", "--system", "inference-service"])
        body = json.loads(route.calls[0].request.content)
        assert body["scope"] == "inference-service"


# ── G11–G12: gates unset ──────────────────────────────────────────────────────

class TestGatesUnset:
    @respx.mock
    def test_gates_unset_found(self, capsys):
        respx.get(f"{BASE}/workspaces/{WS}/governance/grc/gates").mock(
            return_value=httpx.Response(200, json=POLICIES_RESPONSE)
        )
        delete_route = respx.delete(f"{BASE}/workspaces/{WS}/governance/grc/gates/pol-ws-1").mock(
            return_value=httpx.Response(204)
        )
        from mima_governance.gates import cmd_unset
        with env_creds():
            cmd_unset(["eu_ai_act"])
        assert delete_route.called
        out = capsys.readouterr().out
        assert "Gate removed" in out

    @respx.mock
    def test_gates_unset_not_found(self, capsys):
        respx.get(f"{BASE}/workspaces/{WS}/governance/grc/gates").mock(
            return_value=httpx.Response(200, json=EMPTY_RESPONSE)
        )
        from mima_governance.gates import cmd_unset
        with env_creds():
            with pytest.raises(SystemExit) as exc:
                cmd_unset(["eu_ai_act"])
        assert exc.value.code == 1
        assert "no policy found" in capsys.readouterr().err


# ── G13–G14: error cases ──────────────────────────────────────────────────────

class TestGatesErrors:
    def test_unknown_mode_exits_before_api(self):
        from mima_governance.gates import cmd_set
        with env_creds():
            with pytest.raises(SystemExit) as exc:
                cmd_set(["eu_ai_act", "blah"])
        assert exc.value.code == 2

    def test_no_credentials_exits_1(self, capsys):
        from mima_governance.gates import cmd_check
        with patch.dict("os.environ", {}, clear=True):
            with patch("mima_governance.config.get_api_key",      return_value=None):
                with patch("mima_governance.config.get_workspace_id", return_value=None):
                    with pytest.raises(SystemExit) as exc:
                        cmd_check([])
        assert exc.value.code == 1
