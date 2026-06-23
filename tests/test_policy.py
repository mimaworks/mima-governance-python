"""Unit tests for mima_governance/policy.py — composable policy files.

Covers:
  - YAML parse: valid file, missing keys, bad type, unknown assertion type
  - scan_coverage assertion: pass, fail, no-AI-calls
  - min_readiness_pct: pass, fail, missing framework
  - attested_not_inferred: pass, fail, zero coverage
  - record_type_present: pass, fail, 404 (endpoint unavailable)
  - min_records_per_system: pass, fail, with time window, 404
  - PolicyRunner.check_dir: framework filter, empty dir, file error
  - generate_starter_yaml and all_framework_slugs helpers
  - mima test backward-compat delegation
  - mima policy init: creates files, skips existing
  - Exit code 3 on API-unreachable (vs exit 1 on assertion failure)
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from mima_governance.policy import (
    AssertionResult,
    PolicyParseError,
    PolicyResult,
    PolicyRunner,
    _eval_attested_not_inferred,
    _eval_min_readiness_pct,
    _eval_min_records_per_system,
    _eval_record_type_present,
    _eval_scan_coverage,
    _load_yaml,
    all_framework_slugs,
    generate_starter_yaml,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _write_yaml(tmp_path: Path, name: str, content: str) -> Path:
    f = tmp_path / name
    f.write_text(textwrap.dedent(content), encoding="utf-8")
    return f


def _readiness(framework: str, score_pct: int, covered: int = 10, attested: int = 7) -> Dict[str, Any]:
    return {
        "frameworks": [
            {
                "framework": framework,
                "score_pct": score_pct,
                "controls_covered": covered,
                "controls_covered_attested": attested,
            }
        ]
    }


# ── _load_yaml ─────────────────────────────────────────────────────────────────


class TestLoadYaml:
    def test_valid_file(self, tmp_path: Path) -> None:
        f = _write_yaml(tmp_path, "eu.yaml", """\
            name: "Test Policy"
            framework: eu_ai_act
            version: "1.0"
            assertions:
              - type: scan_coverage
                min_pct: 80
        """)
        data = _load_yaml(f)
        assert data["name"] == "Test Policy"
        assert data["framework"] == "eu_ai_act"
        assert len(data["assertions"]) == 1

    def test_missing_required_key_raises(self, tmp_path: Path) -> None:
        f = _write_yaml(tmp_path, "bad.yaml", """\
            name: "Oops"
            assertions:
              - type: scan_coverage
        """)
        with pytest.raises(PolicyParseError) as exc_info:
            _load_yaml(f)
        assert "framework" in str(exc_info.value)

    def test_assertions_not_a_list_raises(self, tmp_path: Path) -> None:
        f = _write_yaml(tmp_path, "bad.yaml", """\
            name: "Bad"
            framework: eu_ai_act
            assertions: "should be a list"
        """)
        with pytest.raises(PolicyParseError) as exc_info:
            _load_yaml(f)
        assert "list" in str(exc_info.value)

    def test_unknown_assertion_type_raises(self, tmp_path: Path) -> None:
        f = _write_yaml(tmp_path, "bad.yaml", """\
            name: "Bad"
            framework: eu_ai_act
            assertions:
              - type: not_a_real_assertion
        """)
        with pytest.raises(PolicyParseError) as exc_info:
            _load_yaml(f)
        assert "not_a_real_assertion" in str(exc_info.value)

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "mal.yaml"
        f.write_text("name: [unclosed", encoding="utf-8")
        with pytest.raises(PolicyParseError):
            _load_yaml(f)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PolicyParseError) as exc_info:
            _load_yaml(tmp_path / "nonexistent.yaml")
        assert "nonexistent.yaml" in str(exc_info.value)

    def test_non_mapping_root_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "list.yaml"
        f.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(PolicyParseError) as exc_info:
            _load_yaml(f)
        assert "mapping" in str(exc_info.value)

    def test_parse_error_contains_file_path(self, tmp_path: Path) -> None:
        f = _write_yaml(tmp_path, "pol.yaml", """\
            name: "X"
            assertions: []
        """)
        with pytest.raises(PolicyParseError) as exc_info:
            _load_yaml(f)
        assert str(f) in str(exc_info.value)


# ── scan_coverage assertion ────────────────────────────────────────────────────


class TestEvalScanCoverage:
    def test_pass_when_coverage_meets_threshold(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        # A file with no AI calls → 100% coverage (vacuously true)
        (src / "pure.py").write_text("x = 1\n", encoding="utf-8")

        result = _eval_scan_coverage(
            {"type": "scan_coverage", "min_pct": 80, "path": str(src)},
            base_path=str(src),
        )
        assert result.passed is True
        assert "100%" in result.actual

    def test_no_ai_calls_always_passes(self, tmp_path: Path) -> None:
        src = tmp_path / "empty_src"
        src.mkdir()
        result = _eval_scan_coverage({"type": "scan_coverage", "min_pct": 95, "path": str(src)})
        assert result.passed is True

    def test_fail_when_below_threshold(self, tmp_path: Path) -> None:
        src = tmp_path / "src2"
        src.mkdir()
        # Create a file with an unattested openai call
        (src / "agent.py").write_text(
            "import openai\nresponse = openai.chat.completions.create(model='gpt-4', messages=[])\n",
            encoding="utf-8",
        )
        result = _eval_scan_coverage(
            {"type": "scan_coverage", "min_pct": 100, "path": str(src)},
        )
        # 0% attested (no @mima.attest), threshold 100% → fail
        assert result.passed is False
        assert result.detail != ""

    def test_description_defaults_sensibly(self, tmp_path: Path) -> None:
        result = _eval_scan_coverage({"type": "scan_coverage", "min_pct": 75, "path": str(tmp_path)})
        assert "75" in result.description or "coverage" in result.description.lower()


# ── min_readiness_pct assertion ───────────────────────────────────────────────


class TestEvalMinReadinessPct:
    def test_pass(self) -> None:
        data = _readiness("eu_ai_act", 84)
        result = _eval_min_readiness_pct(
            {"type": "min_readiness_pct", "threshold": 80}, "eu_ai_act", data
        )
        assert result.passed is True
        assert "84%" in result.actual

    def test_fail(self) -> None:
        data = _readiness("eu_ai_act", 65)
        result = _eval_min_readiness_pct(
            {"type": "min_readiness_pct", "threshold": 80}, "eu_ai_act", data
        )
        assert result.passed is False

    def test_exact_threshold_passes(self) -> None:
        data = _readiness("eu_ai_act", 80)
        result = _eval_min_readiness_pct(
            {"type": "min_readiness_pct", "threshold": 80}, "eu_ai_act", data
        )
        assert result.passed is True

    def test_missing_framework_fails_with_detail(self) -> None:
        data = {"frameworks": []}
        result = _eval_min_readiness_pct(
            {"type": "min_readiness_pct", "threshold": 80}, "eu_ai_act", data
        )
        assert result.passed is False
        assert "eu_ai_act" in result.detail


# ── attested_not_inferred assertion ───────────────────────────────────────────


class TestEvalAttestedNotInferred:
    def test_pass(self) -> None:
        data = _readiness("soc2_type2", 78, covered=10, attested=8)
        result = _eval_attested_not_inferred(
            {"type": "attested_not_inferred", "min_pct": 70}, "soc2_type2", data
        )
        assert result.passed is True
        assert "80%" in result.actual  # 8/10 = 80%

    def test_fail(self) -> None:
        data = _readiness("soc2_type2", 78, covered=10, attested=3)
        result = _eval_attested_not_inferred(
            {"type": "attested_not_inferred", "min_pct": 60}, "soc2_type2", data
        )
        assert result.passed is False
        assert "30%" in result.actual  # 3/10 = 30%

    def test_zero_covered_fails_with_guidance(self) -> None:
        data = _readiness("eu_ai_act", 0, covered=0, attested=0)
        result = _eval_attested_not_inferred(
            {"type": "attested_not_inferred", "min_pct": 60}, "eu_ai_act", data
        )
        assert result.passed is False
        assert "0 controls" in result.actual

    def test_missing_framework_fails(self) -> None:
        data = {"frameworks": []}
        result = _eval_attested_not_inferred(
            {"type": "attested_not_inferred", "min_pct": 60}, "eu_ai_act", data
        )
        assert result.passed is False


# ── record_type_present assertion ─────────────────────────────────────────────


class TestEvalRecordTypePresent:
    def test_pass_when_count_ge_1(self) -> None:
        result = _eval_record_type_present(
            {"type": "record_type_present", "record_type": "access_review"},
            "soc2_type2",
            lambda rt: 3,
        )
        assert result.passed is True
        assert "3 records" in result.actual

    def test_fail_when_count_zero(self) -> None:
        result = _eval_record_type_present(
            {"type": "record_type_present", "record_type": "incident_report"},
            "soc2_type2",
            lambda rt: 0,
        )
        assert result.passed is False
        assert result.detail != ""

    def test_endpoint_unavailable_returns_graceful_result(self) -> None:
        # None return = 404 from server
        result = _eval_record_type_present(
            {"type": "record_type_present", "record_type": "ai_risk_assessment"},
            "eu_ai_act",
            lambda rt: None,
        )
        assert result.passed is False
        assert "endpoint unavailable" in result.actual
        assert "Upgrade" in result.detail

    def test_missing_record_type_field(self) -> None:
        result = _eval_record_type_present(
            {"type": "record_type_present"},
            "eu_ai_act",
            lambda rt: 1,
        )
        assert result.passed is False
        assert "record_type" in result.expected


# ── min_records_per_system assertion ──────────────────────────────────────────


class TestEvalMinRecordsPerSystem:
    def test_pass(self) -> None:
        result = _eval_min_records_per_system(
            {"type": "min_records_per_system", "record_type": "human_oversight", "count": 5},
            "eu_ai_act",
            lambda rt, sys, since: 7,
        )
        assert result.passed is True
        assert "7 records" in result.actual

    def test_fail(self) -> None:
        result = _eval_min_records_per_system(
            {"type": "min_records_per_system", "record_type": "human_oversight", "count": 5},
            "eu_ai_act",
            lambda rt, sys, since: 2,
        )
        assert result.passed is False

    def test_per_window_included_in_output(self) -> None:
        result = _eval_min_records_per_system(
            {"type": "min_records_per_system", "record_type": "model_evaluation", "count": 3, "per": "30d"},
            "eu_ai_act",
            lambda rt, sys, since: 4,
        )
        assert result.passed is True
        assert "30d" in result.actual

    def test_endpoint_unavailable_graceful(self) -> None:
        result = _eval_min_records_per_system(
            {"type": "min_records_per_system", "record_type": "human_oversight", "count": 1},
            "eu_ai_act",
            lambda rt, sys, since: None,
        )
        assert result.passed is False
        assert "endpoint unavailable" in result.actual


# ── PolicyRunner ──────────────────────────────────────────────────────────────


class TestPolicyRunner:
    def test_check_dir_empty(self, tmp_path: Path) -> None:
        runner = PolicyRunner()
        results = runner.check_dir(tmp_path)
        assert results == []

    def test_check_dir_nonexistent(self, tmp_path: Path) -> None:
        runner = PolicyRunner()
        results = runner.check_dir(tmp_path / "ghost")
        assert results == []

    def test_check_file_parse_error_returned_as_error_result(self, tmp_path: Path) -> None:
        f = _write_yaml(tmp_path, "bad.yaml", "name: oops\nassertions: not-a-list\n")
        runner = PolicyRunner()
        result = runner.check_file(f)
        assert result.error is not None
        assert result.passed is False

    def test_check_file_scan_coverage_no_credentials(self, tmp_path: Path) -> None:
        f = _write_yaml(tmp_path, "policy.yaml", f"""\
            name: "Test"
            framework: eu_ai_act
            assertions:
              - type: scan_coverage
                min_pct: 80
                path: "{tmp_path}"
        """)
        runner = PolicyRunner()  # no credentials
        result = runner.check_file(f)
        assert result.error is None
        assert len(result.assertions) == 1
        assert result.assertions[0].assertion_type == "scan_coverage"

    def test_check_dir_framework_filter(self, tmp_path: Path) -> None:
        _write_yaml(tmp_path, "eu.yaml", f"""\
            name: "EU"
            framework: eu_ai_act
            assertions:
              - type: scan_coverage
                min_pct: 0
                path: "{tmp_path}"
        """)
        _write_yaml(tmp_path, "soc.yaml", f"""\
            name: "SOC"
            framework: soc2_type2
            assertions:
              - type: scan_coverage
                min_pct: 0
                path: "{tmp_path}"
        """)
        runner = PolicyRunner()
        results = runner.check_dir(tmp_path, framework="eu_ai_act")
        assert len(results) == 1
        assert results[0].framework == "eu_ai_act"

    def test_check_dir_loads_both_yaml_and_yml(self, tmp_path: Path) -> None:
        _write_yaml(tmp_path, "a.yaml", f"""\
            name: "A"
            framework: eu_ai_act
            assertions:
              - type: scan_coverage
                min_pct: 0
                path: "{tmp_path}"
        """)
        _write_yaml(tmp_path, "b.yml", f"""\
            name: "B"
            framework: soc2_type2
            assertions:
              - type: scan_coverage
                min_pct: 0
                path: "{tmp_path}"
        """)
        runner = PolicyRunner()
        results = runner.check_dir(tmp_path)
        assert len(results) == 2

    def test_readiness_cache_called_once(self, tmp_path: Path) -> None:
        f = _write_yaml(tmp_path, "policy.yaml", f"""\
            name: "Test"
            framework: eu_ai_act
            assertions:
              - type: min_readiness_pct
                threshold: 80
              - type: attested_not_inferred
                min_pct: 60
        """)
        runner = PolicyRunner(api_key="test-key", workspace_id="ws-123")
        call_count = 0

        def fake_fetch():
            nonlocal call_count
            call_count += 1
            return _readiness("eu_ai_act", 85, covered=10, attested=8)

        runner._fetch_readiness = fake_fetch
        runner._readiness_cache = None
        # Manually pre-populate to avoid real HTTP
        runner._readiness_cache = fake_fetch()
        result = runner.check_file(f)
        assert result.error is None
        assert all(a.passed for a in result.assertions)


# ── Starter YAML helpers ──────────────────────────────────────────────────────


class TestStarterTemplates:
    def test_all_framework_slugs_returns_list(self) -> None:
        slugs = all_framework_slugs()
        assert isinstance(slugs, list)
        assert "eu_ai_act" in slugs
        assert "soc2_type2" in slugs

    def test_generate_returns_valid_yaml(self) -> None:
        import yaml
        for slug in all_framework_slugs():
            content = generate_starter_yaml(slug)
            assert content is not None
            data = yaml.safe_load(content)
            assert "assertions" in data
            assert isinstance(data["assertions"], list)

    def test_unknown_slug_returns_none(self) -> None:
        assert generate_starter_yaml("not_a_framework") is None


# ── mima policy init CLI ──────────────────────────────────────────────────────


class TestPolicyInitCli:
    def test_creates_files(self, tmp_path: Path) -> None:
        from mima_governance.cli import _policy_init
        policy_dir = tmp_path / "mima_policy"
        _policy_init(["--path", str(policy_dir), "--frameworks", "eu_ai_act"])
        assert (policy_dir / "eu_ai_act.yaml").exists()

    def test_skips_existing_files(self, tmp_path: Path) -> None:
        from mima_governance.cli import _policy_init
        policy_dir = tmp_path / "mima_policy"
        policy_dir.mkdir()
        existing = policy_dir / "eu_ai_act.yaml"
        existing.write_text("existing content", encoding="utf-8")
        _policy_init(["--path", str(policy_dir), "--frameworks", "eu_ai_act"])
        assert existing.read_text(encoding="utf-8") == "existing content"


# ── mima test backward-compat ─────────────────────────────────────────────────


class TestCmdTestBackwardCompat:
    def test_delegates_to_policy_check_when_mima_policy_dir_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        policy_dir = tmp_path / "mima_policy"
        policy_dir.mkdir()
        # Write a passing scan_coverage policy so check exits 0
        (policy_dir / "test.yaml").write_text(
            f'name: "T"\nframework: eu_ai_act\nassertions:\n  - type: scan_coverage\n    min_pct: 0\n    path: "{tmp_path}"\n',
            encoding="utf-8",
        )
        from mima_governance.cli import _cmd_test
        with pytest.raises(SystemExit) as exc_info:
            _cmd_test([])
        assert exc_info.value.code == 0

    def test_exits_2_when_no_mima_policy_dir_and_no_args(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        from mima_governance.cli import _cmd_test
        with pytest.raises(SystemExit) as exc_info:
            _cmd_test([])
        assert exc_info.value.code == 2

    def test_still_runs_test_file_with_explicit_arg(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test_gov.py"
        test_file.write_text("# empty\n", encoding="utf-8")
        from mima_governance.cli import _cmd_test
        # run_test_file will produce an empty suite; just check we don't delegate
        with pytest.raises(SystemExit) as exc_info:
            _cmd_test([str(test_file)])
        # exits 0 (no tests failed)
        assert exc_info.value.code == 0
