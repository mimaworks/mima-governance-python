"""Langfuse → Mima Governance adapter.

Translates Langfuse score objects (from ``langfuse.get_scores()`` or a dataset
run) into a ``mima.model_evaluation()`` GRC evidence record.

Usage::

    from langfuse import Langfuse
    from mima_governance.integrations.langfuse import report_to_mima

    langfuse = Langfuse(public_key=..., secret_key=..., host=...)
    scores = langfuse.get_scores(name="faithfulness", limit=100).data

    grc_result = report_to_mima(
        mima_client=mima,
        model_id="gpt-4o",
        scores=scores,
        dataset="prod-traces-2026-06",
    )

Like the DeepEval adapter this module never imports ``langfuse`` directly.
Pass Langfuse score objects or plain dicts with ``name`` and ``value`` keys.

Score shape accepted
--------------------
Both Langfuse SDK objects and plain dicts are accepted::

    # SDK object
    score.name   # str
    score.value  # float (0–1)
    score.comment  # Optional[str]

    # Plain dict
    {"name": "faithfulness", "value": 0.87, "comment": "..."}

Non-numeric values (e.g. categoricals like ``"good"`` / ``"bad"``) are
silently skipped — they cannot be mapped to a numeric accuracy field.

Control mappings
----------------
The same ``METRIC_CONTROL_MAP`` from the DeepEval adapter is reused, keyed on
lowercase score names. Langfuse score names don't need to exactly match DeepEval
metric names — only lowercase equality matters.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from mima_governance.integrations.deepeval_adapter import (
    _NEGATIVE_METRICS,
    _QUALITY_METRICS,
    _build_model_eval_args,
    _ci_identity,
)


# ── Extraction helper ─────────────────────────────────────────────────────────

def _parse_langfuse_scores(scores: List[Any]) -> List[Dict[str, Any]]:
    """Normalise Langfuse score objects or dicts into the shared metric format.

    Returns a list of ``{name, score, success}`` dicts, skipping any entry
    whose value is not numeric.
    """
    result: List[Dict[str, Any]] = []
    for s in scores:
        # Accept both object attributes and dict keys
        if isinstance(s, dict):
            name  = s.get("name", "")
            value = s.get("value")
        else:
            name  = getattr(s, "name", "")
            value = getattr(s, "value", None)

        # Skip non-numeric values (categoricals, labels)
        try:
            score = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue

        result.append({
            "name":    str(name).lower(),
            "score":   score,
            "success": True,  # Langfuse scores don't have pass/fail thresholds
        })

    return result


# ── Public adapter ────────────────────────────────────────────────────────────

def report_to_mima(
    mima_client: Any,
    *,
    model_id: str,
    scores: List[Any],
    dataset: str = "langfuse",
    evaluated_by: Optional[str] = None,
    evaluation_type: str = "triggered",
) -> Any:
    """Push Langfuse evaluation scores to the Mima GRC ledger.

    Args:
        mima_client:     A ``MimaGovernance`` (or ``AsyncMimaGovernance``) instance.
        model_id:        Model identifier, e.g. ``"gpt-4o"``.
        scores:          Langfuse score objects or plain dicts with ``name``/``value``.
        dataset:         Dataset or trace window label for the evidence record.
        evaluated_by:    Identity for the scores.  Auto-detected from CI environment
                         variables if omitted (same logic as ``deepeval_adapter``).
        evaluation_type: One of ``"initial"``, ``"quarterly"``, ``"triggered"``.

    Returns:
        ``GrcResult`` from ``model_evaluation()``.
    """
    resolved_by = evaluated_by or _ci_identity() or "langfuse"
    metrics = _parse_langfuse_scores(scores)
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
