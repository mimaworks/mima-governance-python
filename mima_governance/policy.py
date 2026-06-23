"""Composable governance policy files — mima_policy/*.yaml runner.

Policy files declare compliance assertions in YAML that GRC managers own
and engineers execute in CI. `mima policy check` loads all YAML files in
the mima_policy/ directory and runs each assertion against the live API
or a local scan.

Policy file format:

    # mima_policy/eu_ai_act.yaml
    name: "EU AI Act Art.9 Readiness"
    framework: eu_ai_act
    version: "1.0"
    assertions:
      - type: min_readiness_pct
        threshold: 80
        description: "Must reach 80% before December 2027 (Annex III enforcement)"
      - type: record_type_present
        record_type: ai_risk_assessment
        description: "Every system must have a risk assessment on record"
      - type: attested_not_inferred
        min_pct: 60
        description: "60%+ of coverage must be SDK-attested"
      - type: scan_coverage
        min_pct: 80
        path: "src/"
        description: "80% of AI call sites must be attested"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class AssertionResult:
    assertion_type: str
    description: str
    passed: bool
    actual: str
    expected: str
    detail: str = ""


@dataclass
class PolicyResult:
    policy_name: str
    framework: str
    file_path: str
    assertions: List[AssertionResult] = field(default_factory=list)
    error: Optional[str] = None          # set when the file failed to parse/run

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        return all(a.passed for a in self.assertions)


# ── YAML loader ───────────────────────────────────────────────────────────────

_REQUIRED_TOP_KEYS = {"name", "framework", "assertions"}
_VALID_ASSERTION_TYPES = frozenset([
    "min_readiness_pct",
    "record_type_present",
    "min_records_per_system",
    "attested_not_inferred",
    "scan_coverage",
])

_FRAMEWORK_SLUGS = frozenset([
    "soc2_type2", "iso_27001", "iso_42001", "eu_ai_act", "nist_airf",
])


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Parse a YAML policy file. Raises PolicyParseError on failure."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        raise PolicyParseError(
            str(path),
            0,
            "PyYAML is not installed. Run: pip install 'mima-governance[policy]'",
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise PolicyParseError(str(path), 0, str(e))

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        # Extract line number from YAMLError if available
        line = 0
        if hasattr(e, "problem_mark") and e.problem_mark is not None:
            line = e.problem_mark.line + 1
        raise PolicyParseError(str(path), line, str(e))

    if not isinstance(data, dict):
        raise PolicyParseError(str(path), 0, "Policy file must be a YAML mapping at the top level")

    missing = _REQUIRED_TOP_KEYS - data.keys()
    if missing:
        raise PolicyParseError(str(path), 0, f"Missing required keys: {', '.join(sorted(missing))}")

    if not isinstance(data.get("assertions"), list):
        raise PolicyParseError(str(path), 0, "'assertions' must be a list")

    for i, a in enumerate(data["assertions"]):
        if not isinstance(a, dict):
            raise PolicyParseError(str(path), 0, f"assertions[{i}] must be a mapping")
        if "type" not in a:
            raise PolicyParseError(str(path), 0, f"assertions[{i}] is missing 'type'")
        if a["type"] not in _VALID_ASSERTION_TYPES:
            raise PolicyParseError(
                str(path), 0,
                f"assertions[{i}].type '{a['type']}' is not valid. "
                f"Valid types: {', '.join(sorted(_VALID_ASSERTION_TYPES))}",
            )

    return data


class PolicyParseError(Exception):
    def __init__(self, file_path: str, line: int, message: str):
        self.file_path = file_path
        self.line = line
        self.message = message
        loc = f":{line}" if line else ""
        super().__init__(f"{file_path}{loc}: {message}")


# ── Assertion evaluators ──────────────────────────────────────────────────────


def _eval_scan_coverage(
    assertion: Dict[str, Any],
    *,
    base_path: str = ".",
) -> AssertionResult:
    """Local scan — no credentials required."""
    from .cli import _scan_path

    path = assertion.get("path", base_path)
    min_pct = int(assertion.get("min_pct", 80))
    description = assertion.get("description", f"Scan coverage ≥ {min_pct}%")

    detections, _ = _scan_path(Path(path))
    high = [d for d in detections if d.confidence == "high"]
    if not high:
        return AssertionResult(
            assertion_type="scan_coverage",
            description=description,
            passed=True,
            actual="100% (no AI calls found)",
            expected=f"≥ {min_pct}%",
        )

    attested_count = sum(1 for d in high if d.attested)
    actual_pct = int(attested_count / len(high) * 100)
    passed = actual_pct >= min_pct

    return AssertionResult(
        assertion_type="scan_coverage",
        description=description,
        passed=passed,
        actual=f"{actual_pct}%  ({attested_count}/{len(high)} attested)",
        expected=f"≥ {min_pct}%",
        detail=(
            "" if passed
            else f"Run `mima scan {path}` to see unattested call sites."
        ),
    )


def _eval_min_readiness_pct(
    assertion: Dict[str, Any],
    framework: str,
    readiness_data: Dict[str, Any],
) -> AssertionResult:
    threshold = int(assertion.get("threshold", 80))
    description = assertion.get("description", f"Readiness ≥ {threshold}%")

    fw_row = next(
        (f for f in readiness_data.get("frameworks", []) if f["framework"] == framework),
        None,
    )
    if fw_row is None:
        return AssertionResult(
            assertion_type="min_readiness_pct",
            description=description,
            passed=False,
            actual="no data",
            expected=f"≥ {threshold}%",
            detail=f"Framework '{framework}' not found in readiness response.",
        )

    actual_pct = fw_row["score_pct"]
    passed = actual_pct >= threshold

    return AssertionResult(
        assertion_type="min_readiness_pct",
        description=description,
        passed=passed,
        actual=f"{actual_pct}%",
        expected=f"≥ {threshold}%",
    )


def _eval_attested_not_inferred(
    assertion: Dict[str, Any],
    framework: str,
    readiness_data: Dict[str, Any],
) -> AssertionResult:
    min_pct = int(assertion.get("min_pct", 60))
    description = assertion.get("description", f"SDK-attested coverage ≥ {min_pct}%")

    fw_row = next(
        (f for f in readiness_data.get("frameworks", []) if f["framework"] == framework),
        None,
    )
    if fw_row is None:
        return AssertionResult(
            assertion_type="attested_not_inferred",
            description=description,
            passed=False,
            actual="no data",
            expected=f"attested ≥ {min_pct}% of covered",
            detail=f"Framework '{framework}' not found.",
        )

    covered = fw_row.get("controls_covered", 0)
    attested = fw_row.get("controls_covered_attested", covered)

    if covered == 0:
        return AssertionResult(
            assertion_type="attested_not_inferred",
            description=description,
            passed=False,
            actual="0 controls covered",
            expected=f"attested ≥ {min_pct}% of covered",
            detail="Push evidence records to build coverage first.",
        )

    actual_pct = int(attested / covered * 100)
    passed = actual_pct >= min_pct

    return AssertionResult(
        assertion_type="attested_not_inferred",
        description=description,
        passed=passed,
        actual=f"{actual_pct}%  ({attested}/{covered} controls SDK-attested)",
        expected=f"≥ {min_pct}%",
        detail=(
            "" if passed
            else "Replace inferred controls with explicit SDK calls (@mima.attest / mima push)."
        ),
    )


def _eval_record_type_present(
    assertion: Dict[str, Any],
    framework: str,
    summary_fn,       # callable(record_type) -> int | None (None = endpoint unavailable)
) -> AssertionResult:
    record_type = assertion.get("record_type", "")
    description = assertion.get("description", f"At least one {record_type} record exists")

    if not record_type:
        return AssertionResult(
            assertion_type="record_type_present",
            description=description,
            passed=False,
            actual="configuration error",
            expected="record_type field required",
        )

    count = summary_fn(record_type)

    if count is None:
        return AssertionResult(
            assertion_type="record_type_present",
            description=description,
            passed=False,
            actual="endpoint unavailable",
            expected="≥ 1 record",
            detail=(
                "The /grc/evidence/summary endpoint is not available on this server version. "
                "Upgrade the Mima server to use record_type_present assertions."
            ),
        )

    passed = count >= 1
    return AssertionResult(
        assertion_type="record_type_present",
        description=description,
        passed=passed,
        actual=f"{count} record{'s' if count != 1 else ''}",
        expected="≥ 1 record",
        detail=(
            "" if passed
            else f"Push a {record_type} record: `mima push {record_type} --help`"
        ),
    )


def _eval_min_records_per_system(
    assertion: Dict[str, Any],
    framework: str,
    summary_fn,       # callable(record_type, system_name, since) -> int | None
) -> AssertionResult:
    record_type = assertion.get("record_type", "")
    count_required = int(assertion.get("count", 1))
    per = assertion.get("per")                 # e.g. "30d"
    description = assertion.get("description",
                                f"≥ {count_required} {record_type} records")

    since_iso: Optional[str] = None
    if per:
        import datetime
        days = int(per.rstrip("d"))
        since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
        since_iso = since.date().isoformat()

    count = summary_fn(record_type, None, since_iso)

    if count is None:
        return AssertionResult(
            assertion_type="min_records_per_system",
            description=description,
            passed=False,
            actual="endpoint unavailable",
            expected=f"≥ {count_required}",
            detail=(
                "The /grc/evidence/summary endpoint is not available. "
                "Upgrade the Mima server to use min_records_per_system assertions."
            ),
        )

    passed = count >= count_required
    window = f" in last {per}" if per else ""
    return AssertionResult(
        assertion_type="min_records_per_system",
        description=description,
        passed=passed,
        actual=f"{count} record{'s' if count != 1 else ''}{window}",
        expected=f"≥ {count_required}{window}",
    )


# ── PolicyRunner ──────────────────────────────────────────────────────────────


class PolicyRunner:
    """Load and evaluate governance policy files against the Mima API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace_id: Optional[str] = None,
        base_url: str = "https://api.mima.ai",
    ):
        self._api_key = api_key
        self._workspace_id = workspace_id
        self._base_url = base_url.rstrip("/")
        self._readiness_cache: Optional[Dict[str, Any]] = None

    # ── API helpers ──────────────────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _fetch_readiness(self) -> Optional[Dict[str, Any]]:
        """Fetch readiness once and cache for the duration of a check run."""
        if self._readiness_cache is not None:
            return self._readiness_cache
        if not self._api_key or not self._workspace_id:
            return None
        try:
            import httpx
            resp = httpx.get(
                f"{self._base_url}/api/workspaces/{self._workspace_id}/governance/grc/readiness",
                headers=self._headers(),
                timeout=10.0,
            )
            if resp.status_code == 200:
                self._readiness_cache = resp.json()
                return self._readiness_cache
        except Exception:
            pass
        return None

    def _fetch_evidence_count(
        self,
        record_type: str,
        system_name: Optional[str] = None,
        since: Optional[str] = None,
    ) -> Optional[int]:
        """Call /grc/evidence/summary. Returns None if endpoint unavailable."""
        if not self._api_key or not self._workspace_id:
            return None
        try:
            import httpx
            params: Dict[str, str] = {"record_type": record_type}
            if system_name:
                params["system_name"] = system_name
            if since:
                params["since"] = since
            resp = httpx.get(
                f"{self._base_url}/api/workspaces/{self._workspace_id}/governance/grc/evidence/summary",
                headers=self._headers(),
                params=params,
                timeout=10.0,
            )
            if resp.status_code == 404:
                return None   # endpoint not yet deployed
            if resp.status_code == 200:
                return int(resp.json().get("count", 0))
        except Exception:
            pass
        return None

    # ── Core evaluation ──────────────────────────────────────────────────────

    def check_file(self, path: Path) -> PolicyResult:
        """Load and evaluate a single policy file."""
        try:
            data = _load_yaml(path)
        except PolicyParseError as e:
            return PolicyResult(
                policy_name=str(path.name),
                framework="",
                file_path=str(path),
                error=str(e),
            )

        policy_name = data["name"]
        framework = data["framework"]
        assertions_raw = data["assertions"]

        # Fetch shared readiness data once for all server-side assertions in this file.
        readiness = self._fetch_readiness() if self._api_key else None
        needs_api = any(
            a.get("type") not in ("scan_coverage",)
            for a in assertions_raw
        )
        if needs_api and readiness is None and self._api_key:
            return PolicyResult(
                policy_name=policy_name,
                framework=framework,
                file_path=str(path),
                error=(
                    "Could not reach the Mima API. "
                    "Check MIMA_API_KEY, MIMA_WORKSPACE_ID, and network connectivity."
                ),
            )

        results: List[AssertionResult] = []
        for a in assertions_raw:
            atype = a["type"]

            if atype == "scan_coverage":
                results.append(_eval_scan_coverage(a, base_path="."))

            elif atype == "min_readiness_pct":
                if readiness is None:
                    results.append(AssertionResult(
                        assertion_type=atype,
                        description=a.get("description", ""),
                        passed=False,
                        actual="no API",
                        expected=f"≥ {a.get('threshold', 80)}%",
                        detail="Run `mima login` to connect to the API.",
                    ))
                else:
                    results.append(_eval_min_readiness_pct(a, framework, readiness))

            elif atype == "attested_not_inferred":
                if readiness is None:
                    results.append(AssertionResult(
                        assertion_type=atype,
                        description=a.get("description", ""),
                        passed=False,
                        actual="no API",
                        expected=f"≥ {a.get('min_pct', 60)}% attested",
                        detail="Run `mima login` to connect to the API.",
                    ))
                else:
                    results.append(_eval_attested_not_inferred(a, framework, readiness))

            elif atype == "record_type_present":
                results.append(_eval_record_type_present(
                    a, framework,
                    lambda rt: self._fetch_evidence_count(rt),
                ))

            elif atype == "min_records_per_system":
                results.append(_eval_min_records_per_system(
                    a, framework,
                    lambda rt, sys, since: self._fetch_evidence_count(rt, sys, since),
                ))

        return PolicyResult(
            policy_name=policy_name,
            framework=framework,
            file_path=str(path),
            assertions=results,
        )

    def check_dir(
        self,
        path: Path,
        framework: Optional[str] = None,
    ) -> List[PolicyResult]:
        """Load and evaluate all *.yaml files in a directory."""
        if not path.exists():
            return []

        files = sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml"))
        results = []
        for f in files:
            result = self.check_file(f)
            if framework and result.framework and result.framework != framework:
                continue
            results.append(result)
        return results

    def invalidate_cache(self) -> None:
        self._readiness_cache = None


# ── Starter YAML templates ────────────────────────────────────────────────────


_STARTER_TEMPLATES: Dict[str, str] = {
    "eu_ai_act": """\
name: "EU AI Act Art.9 Readiness"
framework: eu_ai_act
version: "1.0"

assertions:
  - type: min_readiness_pct
    threshold: 80
    description: "EU AI Act readiness must reach 80%"

  - type: record_type_present
    record_type: ai_risk_assessment
    description: "At least one AI risk assessment must be on record"

  - type: record_type_present
    record_type: human_oversight
    description: "At least one human oversight decision must be on record"

  - type: attested_not_inferred
    min_pct: 60
    description: "At least 60% of EU AI Act coverage must be SDK-attested"

  - type: scan_coverage
    min_pct: 80
    path: "."
    description: "80% of AI call sites must be attested"
""",
    "soc2_type2": """\
name: "SOC 2 Type II Gate"
framework: soc2_type2
version: "1.0"

assertions:
  - type: min_readiness_pct
    threshold: 70
    description: "SOC 2 readiness must reach 70%"

  - type: record_type_present
    record_type: access_review
    description: "At least one access review must be on record"

  - type: record_type_present
    record_type: incident_report
    description: "At least one incident report must be on record"

  - type: attested_not_inferred
    min_pct: 50
    description: "At least 50% of SOC 2 coverage must be SDK-attested"
""",
    "iso_42001": """\
name: "ISO 42001 AI Management System"
framework: iso_42001
version: "1.0"

assertions:
  - type: min_readiness_pct
    threshold: 70
    description: "ISO 42001 readiness must reach 70%"

  - type: record_type_present
    record_type: ai_risk_assessment
    description: "AI risk assessment required for ISO 42001 clause 6.1"

  - type: record_type_present
    record_type: model_evaluation
    description: "Model evaluation required for ISO 42001 clause 6.3"
""",
    "nist_airf": """\
name: "NIST AI RMF Gate"
framework: nist_airf
version: "1.0"

assertions:
  - type: min_readiness_pct
    threshold: 60
    description: "NIST AI RMF readiness must reach 60%"

  - type: record_type_present
    record_type: ai_risk_assessment
    description: "GOVERN/MAP: risk assessment required"

  - type: record_type_present
    record_type: model_evaluation
    description: "MEASURE: model evaluation required"
""",
    "iso_27001": """\
name: "ISO 27001:2022 Gate"
framework: iso_27001
version: "1.0"

assertions:
  - type: min_readiness_pct
    threshold: 70
    description: "ISO 27001 readiness must reach 70%"

  - type: record_type_present
    record_type: access_review
    description: "Access review required for clause 5.16/5.18"
""",
}


def generate_starter_yaml(framework: str) -> Optional[str]:
    """Return starter YAML content for a framework slug, or None if unknown."""
    return _STARTER_TEMPLATES.get(framework)


def all_framework_slugs() -> List[str]:
    return list(_STARTER_TEMPLATES.keys())
