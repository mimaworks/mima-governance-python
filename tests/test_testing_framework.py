"""Tests for the mima_governance.testing module."""

from pathlib import Path

import pytest

from mima_governance.testing import (
    GovernanceTest,
    ScanResult,
    TestResult,
    TestSuiteResult,
    assert_attested,
    assert_no_unattested,
    assert_max_unattested,
)
from mima_governance.cli import Detection


class TestScanResult:
    def test_empty_detections_full_coverage(self):
        r = ScanResult(detections=[], path=".", duration_ms=1.0)
        assert r.coverage == 1.0
        assert r.total == 0
        assert r.attested == 0
        assert r.unattested == 0

    def test_mixed_detections(self):
        detections = [
            Detection(file="a.py", line=1, library="openai", attested=True, confidence="high"),
            Detection(file="a.py", line=5, library="openai", attested=False, confidence="high"),
            Detection(file="a.py", line=10, library="openai", attested=False, confidence="low"),
        ]
        r = ScanResult(detections=detections, path=".", duration_ms=5.0)
        assert r.total == 3
        assert r.attested == 1
        assert r.unattested == 1  # only high-confidence unattested
        assert r.coverage == 0.5  # 1 attested out of 2 high-confidence

    def test_all_attested(self):
        detections = [
            Detection(file="a.py", line=1, library="anthropic", attested=True, confidence="high"),
            Detection(file="a.py", line=5, library="openai", attested=True, confidence="high"),
        ]
        r = ScanResult(detections=detections, path="src/", duration_ms=2.0)
        assert r.coverage == 1.0
        assert r.unattested == 0


class TestAssertions:
    def test_assert_attested_passes(self):
        r = ScanResult(
            detections=[Detection("a.py", 1, "openai", True, "high")],
            path=".", duration_ms=1.0,
        )
        result = assert_attested(r, min_coverage=1.0)
        assert result.passed is True

    def test_assert_attested_fails(self):
        r = ScanResult(
            detections=[
                Detection("a.py", 1, "openai", True, "high"),
                Detection("a.py", 5, "openai", False, "high"),
            ],
            path=".", duration_ms=1.0,
        )
        result = assert_attested(r, min_coverage=1.0)
        assert result.passed is False
        assert "50%" in result.message

    def test_assert_attested_threshold(self):
        r = ScanResult(
            detections=[
                Detection("a.py", 1, "openai", True, "high"),
                Detection("a.py", 5, "openai", False, "high"),
            ],
            path=".", duration_ms=1.0,
        )
        # 50% coverage meets 50% threshold
        result = assert_attested(r, min_coverage=0.5)
        assert result.passed is True

    def test_assert_no_unattested(self):
        r = ScanResult(
            detections=[Detection("a.py", 1, "openai", False, "high")],
            path=".", duration_ms=1.0,
        )
        result = assert_no_unattested(r)
        assert result.passed is False

    def test_assert_max_unattested_passes(self):
        r = ScanResult(
            detections=[
                Detection("a.py", 1, "openai", False, "high"),
                Detection("a.py", 5, "openai", False, "high"),
            ],
            path=".", duration_ms=1.0,
        )
        result = assert_max_unattested(r, max_count=5)
        assert result.passed is True

    def test_assert_max_unattested_fails(self):
        r = ScanResult(
            detections=[
                Detection("a.py", 1, "openai", False, "high"),
                Detection("a.py", 5, "openai", False, "high"),
                Detection("a.py", 9, "openai", False, "high"),
            ],
            path=".", duration_ms=1.0,
        )
        result = assert_max_unattested(r, max_count=2)
        assert result.passed is False


class TestGovernanceTestClass:
    def test_scan_returns_scan_result(self, tmp_path):
        (tmp_path / "empty.py").write_text("x = 1\n")
        gt = GovernanceTest()
        result = gt.scan(str(tmp_path))
        assert isinstance(result, ScanResult)
        assert result.total == 0

    def test_run_all_discovers_methods(self, tmp_path):
        class MyTest(GovernanceTest):
            def test_passes(self):
                return TestResult(name="test_passes", passed=True)

            def test_also_passes(self):
                return TestResult(name="test_also_passes", passed=True)

        suite = MyTest().run_all()
        assert suite.passed == 2
        assert suite.failed == 0

    def test_run_all_catches_assertion_error(self):
        class FailingTest(GovernanceTest):
            def test_boom(self):
                raise AssertionError("something wrong")

        suite = FailingTest().run_all()
        assert suite.failed == 1
        assert "something wrong" in suite.results[0].message

    def test_run_all_catches_exceptions(self):
        class ErrorTest(GovernanceTest):
            def test_error(self):
                raise RuntimeError("oops")

        suite = ErrorTest().run_all()
        assert suite.failed == 1
        assert "RuntimeError" in suite.results[0].message


class TestTestSuiteResult:
    def test_all_passed(self):
        suite = TestSuiteResult(results=[
            TestResult(name="a", passed=True),
            TestResult(name="b", passed=True),
        ])
        assert suite.all_passed is True
        assert suite.passed == 2
        assert suite.failed == 0

    def test_mixed(self):
        suite = TestSuiteResult(results=[
            TestResult(name="a", passed=True),
            TestResult(name="b", passed=False, message="nope"),
        ])
        assert suite.all_passed is False
        assert suite.passed == 1
        assert suite.failed == 1
