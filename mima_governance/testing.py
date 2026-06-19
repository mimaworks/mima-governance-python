"""mima testing — governance assertion framework for CI/CD pipelines.

Run governance policy tests like DeepEval runs LLM evals:

    from mima_governance.testing import GovernanceTest, assert_attested

    class TestMyAgent(GovernanceTest):
        def test_all_calls_attested(self):
            results = self.scan("src/")
            assert_attested(results, min_coverage=1.0)

Or from the CLI:

    mima test tests/test_governance.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .cli import Detection, _scan_path


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass
class ScanResult:
    """Result of scanning a path for AI call sites."""
    detections: List[Detection]
    path: str
    duration_ms: float

    @property
    def total(self) -> int:
        return len(self.detections)

    @property
    def attested(self) -> int:
        return sum(1 for d in self.detections if d.attested)

    @property
    def unattested(self) -> int:
        return sum(1 for d in self.detections if not d.attested and d.confidence == "high")

    @property
    def coverage(self) -> float:
        """Fraction of high-confidence detections that are attested (0.0–1.0)."""
        high = [d for d in self.detections if d.confidence == "high"]
        if not high:
            return 1.0  # No AI calls found = fully covered
        return sum(1 for d in high if d.attested) / len(high)


@dataclass
class TestResult:
    """Result of a single governance test assertion."""
    # Tell pytest not to collect this as a test class.
    __test__ = False

    name: str
    passed: bool
    message: str = ""
    duration_ms: float = 0.0
    class_name: str = ""  # set by run_test_file; empty for direct use


@dataclass
class TestSuiteResult:
    """Aggregated results of running a governance test suite."""
    # Tell pytest not to collect this as a test class.
    __test__ = False

    results: List[TestResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


# ── Assertion helpers ────────────────────────────────────────────────────────


def assert_attested(result: ScanResult, *, min_coverage: float = 1.0) -> TestResult:
    """Assert that at least `min_coverage` fraction of AI calls are attested."""
    passed = result.coverage >= min_coverage
    if passed:
        msg = f"Coverage {result.coverage:.0%} >= {min_coverage:.0%}"
    else:
        msg = (
            f"Coverage {result.coverage:.0%} < {min_coverage:.0%} — "
            f"{result.unattested} unattested call site(s)"
        )
    return TestResult(name="assert_attested", passed=passed, message=msg)


def assert_no_unattested(result: ScanResult) -> TestResult:
    """Assert zero unattested high-confidence AI call sites."""
    return assert_attested(result, min_coverage=1.0)


def assert_max_unattested(result: ScanResult, *, max_count: int) -> TestResult:
    """Assert at most `max_count` unattested call sites (for gradual adoption)."""
    passed = result.unattested <= max_count
    if passed:
        msg = f"{result.unattested} unattested <= {max_count} allowed"
    else:
        msg = f"{result.unattested} unattested > {max_count} allowed"
    return TestResult(name="assert_max_unattested", passed=passed, message=msg)


# ── GovernanceTest base class ────────────────────────────────────────────────


class GovernanceTest:
    """Base class for governance test suites.

    Subclass this and define test_* methods. Each method should use
    self.scan() and the assert_* helpers.

    Usage:
        class TestMyProject(GovernanceTest):
            def test_full_coverage(self):
                result = self.scan("src/")
                return assert_attested(result, min_coverage=0.95)
    """

    def scan(self, path: str, *, include: str = "**/*.py") -> ScanResult:
        """Scan a path for AI call sites. Returns a ScanResult."""
        root = Path(path)
        start = time.perf_counter()
        detections, _files_scanned = _scan_path(root, include)
        duration_ms = (time.perf_counter() - start) * 1000
        return ScanResult(detections=detections, path=path, duration_ms=duration_ms)

    def run_all(self) -> TestSuiteResult:
        """Discover and run all test_* methods on this instance."""
        suite = TestSuiteResult()
        methods = sorted(m for m in dir(self) if m.startswith("test_") and callable(getattr(self, m)))

        for method_name in methods:
            method = getattr(self, method_name)
            start = time.perf_counter()
            try:
                result = method()
                duration_ms = (time.perf_counter() - start) * 1000
                if isinstance(result, TestResult):
                    result.duration_ms = duration_ms
                    result.name = method_name
                    suite.results.append(result)
                else:
                    # Method didn't return a TestResult — treat no exception as pass
                    suite.results.append(TestResult(
                        name=method_name, passed=True, duration_ms=duration_ms
                    ))
            except AssertionError as e:
                duration_ms = (time.perf_counter() - start) * 1000
                suite.results.append(TestResult(
                    name=method_name, passed=False, message=str(e), duration_ms=duration_ms
                ))
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                suite.results.append(TestResult(
                    name=method_name, passed=False,
                    message=f"Error: {type(e).__name__}: {e}", duration_ms=duration_ms
                ))

        return suite


# ── CLI runner ───────────────────────────────────────────────────────────────


def run_test_file(path: str) -> TestSuiteResult:
    """Load a Python file, find GovernanceTest subclasses, run them all."""
    import importlib.util

    file_path = Path(path).resolve()
    if not file_path.exists():
        print(f"mima test: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("_mima_test_module", str(file_path))
    if spec is None or spec.loader is None:
        print(f"mima test: cannot load: {path}", file=sys.stderr)
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    suite = TestSuiteResult()
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, GovernanceTest)
            and obj is not GovernanceTest
        ):
            instance = obj()
            class_suite = instance.run_all()
            for result in class_suite.results:
                result.class_name = name
            suite.results.extend(class_suite.results)

    return suite


def print_suite_result(suite: TestSuiteResult) -> None:
    """Pretty-print test suite results in DeepEval-style grouped layout."""
    from collections import OrderedDict

    # Group results by class_name, preserving insertion order.
    groups: "OrderedDict[str, list]" = OrderedDict()
    for r in suite.results:
        key = r.class_name or ""
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    for class_name, results in groups.items():
        if class_name:
            print(f"\n  {class_name}")

        for r in results:
            symbol   = "\u2713" if r.passed else "\u2717"
            timing   = f"{r.duration_ms:.0f}ms" if r.duration_ms else ""
            # Column layout: name (32), timing (8), inline note
            name_col = r.name.ljust(32)
            time_col = timing.ljust(8)
            note     = r.message if r.passed else "FAILED"
            print(f"    {symbol} {name_col}  {time_col}  {note}")

            if not r.passed and r.message:
                print()
                for line in r.message.split("\n"):
                    print(f"        {line}")
                print()

    total_ms = sum(r.duration_ms for r in suite.results)
    ms_str   = f"{total_ms:.0f}ms"
    print(f"\n  {suite.passed} passed  {suite.failed} failed  in {ms_str}")
