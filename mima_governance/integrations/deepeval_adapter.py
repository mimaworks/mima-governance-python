"""DeepEval → Mima Governance adapter.

Translates a DeepEval ``EvaluationResult`` (or a flat list of ``TestResult``
objects) into a ``mima.model_evaluation()`` GRC evidence record, with control
mappings for EU AI Act Art.10/15, NIST AI RMF MEA-1/MEA-2.5, and SOC 2 CC4.1.

Usage::

    from mima_governance.integrations.deepeval import report_to_mima

    results = deepeval.evaluate(test_cases, metrics)

    grc_result = report_to_mima(
        mima_client=mima,
        model_id="gpt-4o",
        test_results=results,
    )

The adapter uses duck-typing throughout — it never imports ``deepeval`` directly,
so ``pip install mima-governance`` works without DeepEval installed. Install
DeepEval independently (``pip install deepeval``) and pass its result objects in.

Control mappings
----------------
DeepEval metric name → compliance controls that this metric provides evidence for:

  Answer Relevancy / Faithfulness / Contextual*  →  EUAIA_ART15, NIST_AIRF_MEA_1
  Bias                                            →  EUAIA_ART10, NIST_AIRF_MEA_2_5
  Toxicity                                        →  EUAIA_ART9,  SOC2_CC4_1
  Hallucination                                   →  EUAIA_ART15, NIST_AIRF_MEA_1
  Task Completion                                 →  EUAIA_ART15, EUAIA_ART14
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

# ── Control mapping ───────────────────────────────────────────────────────────

# Keys are lowercase DeepEval metric names (exact or prefix match).
METRIC_CONTROL_MAP: Dict[str, List[str]] = {
    "answer relevancy":     ["EUAIA_ART15", "NIST_AIRF_MEA_1"],
    "faithfulness":         ["EUAIA_ART15", "NIST_AIRF_MEA_1"],
    "contextual precision": ["EUAIA_ART15"],
    "contextual recall":    ["EUAIA_ART15"],
    "contextual relevancy": ["EUAIA_ART15"],
    "summarization":        ["EUAIA_ART15"],
    "task completion":      ["EUAIA_ART15", "EUAIA_ART14"],
    "g-eval":               ["EUAIA_ART15"],
    "hallucination":        ["EUAIA_ART15", "NIST_AIRF_MEA_1"],
    "bias":                 ["EUAIA_ART10", "NIST_AIRF_MEA_2_5"],
    "toxicity":             ["EUAIA_ART9",  "SOC2_CC4_1"],
}

# Metrics where a *lower* score is worse (i.e. a high bias score = bad).
# These are excluded from the accuracy average and mapped to bias_metrics /
# robustness_score instead.
_NEGATIVE_METRICS = {"hallucination", "bias", "toxicity"}

# Positive quality metrics that contribute to the accuracy average.
_QUALITY_METRICS = {
    "answer relevancy", "faithfulness",
    "contextual precision", "contextual recall", "contextual relevancy",
    "summarization", "task completion", "g-eval",
}


# ── Extraction helpers ────────────────────────────────────────────────────────

def _extract_metrics(test_results: Any) -> List[Dict[str, Any]]:
    """Extract a flat list of ``{name, score, success}`` dicts from any
    DeepEval result shape.

    Handles:
    - ``EvaluationResult`` (has ``.test_results``)
    - ``list[TestResult]`` (each has ``.metrics_data``)
    - Older API: ``.metrics`` dict on a TestResult
    """
    # Unwrap EvaluationResult
    if hasattr(test_results, "test_results"):
        test_results = test_results.test_results

    metrics: List[Dict[str, Any]] = []
    for result in test_results:
        # Preferred: .metrics_data (list of MetricData objects, deepeval >= 0.21)
        data = getattr(result, "metrics_data", None)
        if data is not None:
            for m in data:
                raw_score = getattr(m, "score", None)
                metrics.append({
                    "name":    str(getattr(m, "name", "")).lower(),
                    "score":   float(raw_score) if raw_score is not None else 0.0,
                    "success": bool(getattr(m, "success", True)),
                })
        # Fallback: .metrics dict (older versions)
        elif hasattr(result, "metrics") and isinstance(result.metrics, dict):
            for name, score in result.metrics.items():
                try:
                    metrics.append({
                        "name":    str(name).lower(),
                        "score":   float(score),
                        "success": True,
                    })
                except (TypeError, ValueError):
                    pass

    return metrics


def _ci_identity() -> str:
    """Build a traceable identity from CI environment variables.

    Returns a string an auditor can trace back to a specific pipeline run.
    Falls back to empty string — callers must provide their own default when
    this returns empty.

    Checks in priority order:
      1. GitHub Actions  — ``github:<actor>@<sha[:8]>``
      2. GitLab CI       — ``gitlab:<user>@<short_sha>``
      3. Generic CI      — ``ci:<CI_ACTOR|CI_USER>[@<sha[:8]>]``
    """
    actor = os.environ.get("GITHUB_ACTOR")
    sha   = os.environ.get("GITHUB_SHA")
    if actor and sha:
        return f"github:{actor}@{sha[:8]}"

    user      = os.environ.get("GITLAB_USER_LOGIN")
    short_sha = os.environ.get("CI_COMMIT_SHORT_SHA")
    if user and short_sha:
        return f"gitlab:{user}@{short_sha}"

    ci_user = os.environ.get("CI_ACTOR") or os.environ.get("CI_USER")
    ci_sha  = os.environ.get("CI_COMMIT_SHA") or os.environ.get("CI_COMMIT")
    if ci_user:
        suffix = f"@{ci_sha[:8]}" if ci_sha else ""
        return f"ci:{ci_user}{suffix}"

    return ""


def _build_model_eval_args(
    model_id: str,
    metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Translate a flat metrics list into kwargs for ``model_evaluation()``."""

    quality_scores: List[float] = []
    quality_names:  List[str]   = []
    negative: Dict[str, float] = {}
    all_passed = True

    for m in metrics:
        name = m["name"]
        score = m["score"]
        if not m["success"]:
            all_passed = False

        if name in _QUALITY_METRICS:
            quality_scores.append(score)
            quality_names.append(name)
        elif name in _NEGATIVE_METRICS:
            negative[name] = score

    accuracy = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    # bias_metrics — only populated when bias/toxicity scores are present.
    # Server-side `map_controls_model_evaluation` conditions SOC2_CC3.3 and
    # SOC2_CC5.1 on this field being a non-empty object — leave as None when
    # no bias data was produced so those controls are not claimed.
    bias_metrics: Optional[Dict[str, Any]] = (
        {k: v for k, v in negative.items() if k in ("bias", "toxicity")} or None
    )

    # robustness_score — derived from inverse hallucination when present.
    # Server-side `map_controls_model_evaluation` conditions EUAIA_ART15_R on
    # this field being numeric — leave as None when hallucination was not
    # measured so that control is not claimed.
    # NOTE: hallucination rate (factual grounding) is a proxy for EUAIA ART.15
    # robustness (adversarial resilience) — correlated, not identical.  The
    # notes field states this derivation explicitly for auditor review.
    hallucination = negative.get("hallucination")
    robustness_score: Optional[float] = (
        round(1.0 - hallucination, 4) if hallucination is not None else None
    )

    # notes — explicit composite description + full metric snapshot.
    # An auditor seeing "accuracy: 0.87" must know it is a composite of
    # multiple DeepEval dimensions, not a single coherent measurement.
    notes_payload: Dict[str, Any] = {}
    if quality_names:
        notes_payload["_accuracy_composite"] = quality_names
    if hallucination is not None:
        notes_payload["_robustness_derivation"] = (
            "1 - hallucination_score (proxy for EUAIA ART.15 robustness — "
            "factual grounding, not adversarial resilience)"
        )
    notes_payload["metrics"] = {m["name"]: round(m["score"], 4) for m in metrics}
    notes = json.dumps(notes_payload, sort_keys=True)

    return {
        "accuracy":         round(accuracy, 4),
        "bias_metrics":     bias_metrics,
        "robustness_score": robustness_score,
        "passed_threshold": all_passed if metrics else None,
        "notes":            notes,
    }


# ── Public adapter ────────────────────────────────────────────────────────────

def report_to_mima(
    mima_client: Any,
    *,
    model_id: str,
    test_results: Any,
    dataset: str = "deepeval",
    evaluated_by: Optional[str] = None,
    evaluation_type: str = "triggered",
) -> Any:
    """Push DeepEval evaluation results to the Mima GRC ledger.

    Args:
        mima_client:     A ``MimaGovernance`` (or ``AsyncMimaGovernance``) instance.
        model_id:        Model identifier, e.g. ``"gpt-4o"``, ``"claude-3-5-sonnet"``.
        test_results:    DeepEval ``EvaluationResult`` or ``list[TestResult]``.
        dataset:         Dataset name to record in the evidence entry.
        evaluated_by:    Identity for the eval run.  If omitted, auto-detected from
                         CI environment variables (``GITHUB_ACTOR``, ``GITLAB_USER_LOGIN``,
                         etc.).  Pass an explicit value such as a service account
                         email or pipeline URL to override.
        evaluation_type: One of ``"initial"``, ``"quarterly"``, ``"triggered"``
                         (default: ``"triggered"``).

    Returns:
        ``GrcResult`` from ``model_evaluation()``.
    """
    resolved_by = evaluated_by or _ci_identity() or "deepeval-ci"
    metrics = _extract_metrics(test_results)
    args = _build_model_eval_args(model_id, metrics)

    return mima_client.model_evaluation(
        model_id,
        dataset,
        args["accuracy"],
        evaluated_by=resolved_by,
        evaluation_type=evaluation_type,
        bias_metrics=args["bias_metrics"],
        robustness_score=args["robustness_score"],
        passed_threshold=args["passed_threshold"],
        notes=args["notes"],
    )
