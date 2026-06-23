"""Tests for DeepEval and Langfuse → Mima adapter integrations.

Uses stub objects — no real deepeval or langfuse import required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from mima_governance.integrations.deepeval_adapter import (
    report_to_mima as deepeval_report,
    _extract_metrics,
    _build_model_eval_args,
    _ci_identity,
    METRIC_CONTROL_MAP,
)
from mima_governance.integrations.langfuse_adapter import (
    report_to_mima as langfuse_report,
    _parse_langfuse_scores,
)

# ── Stub objects (no real SDK imports) ───────────────────────────────────────


class _MetricData:
    def __init__(self, name: str, score: float, success: bool = True):
        self.name = name
        self.score = score
        self.success = success


class _TestResult:
    def __init__(self, *metrics: _MetricData):
        self.metrics_data = list(metrics)


class _EvaluationResult:
    def __init__(self, *results: _TestResult):
        self.test_results = list(results)


def _make_mima():
    """Return a MimaGovernance mock with .model_evaluation tracked."""
    m = MagicMock()
    m.model_evaluation.return_value = MagicMock(record_id="test-grc-id", mapped_controls=[])
    return m


# ── _extract_metrics ──────────────────────────────────────────────────────────


class TestExtractMetrics:
    def test_extracts_from_evaluation_result(self):
        result = _EvaluationResult(
            _TestResult(_MetricData("Answer Relevancy", 0.9), _MetricData("Bias", 0.1)),
        )
        metrics = _extract_metrics(result)
        assert len(metrics) == 2
        assert metrics[0] == {"name": "answer relevancy", "score": 0.9, "success": True}
        assert metrics[1] == {"name": "bias", "score": 0.1, "success": True}

    def test_extracts_from_list_of_test_results(self):
        results = [
            _TestResult(_MetricData("Faithfulness", 0.85)),
            _TestResult(_MetricData("Hallucination", 0.15, success=False)),
        ]
        metrics = _extract_metrics(results)
        assert len(metrics) == 2
        assert metrics[1]["success"] is False

    def test_handles_missing_score_gracefully(self):
        m = _MetricData("Custom Metric", None)  # type: ignore[arg-type]
        metrics = _extract_metrics([_TestResult(m)])
        assert metrics[0]["score"] == 0.0

    def test_empty_results(self):
        assert _extract_metrics([]) == []
        assert _extract_metrics(_EvaluationResult()) == []

    def test_normalises_name_to_lowercase(self):
        metrics = _extract_metrics([_TestResult(_MetricData("G-Eval", 0.7))])
        assert metrics[0]["name"] == "g-eval"


# ── _build_model_eval_args ────────────────────────────────────────────────────


class TestBuildModelEvalArgs:
    def _metrics(self, *pairs):
        return [{"name": n, "score": s, "success": True} for n, s in pairs]

    def test_accuracy_is_mean_of_quality_metrics(self):
        metrics = self._metrics(
            ("answer relevancy", 0.8),
            ("faithfulness", 0.6),
        )
        args = _build_model_eval_args("gpt-4o", metrics)
        assert args["accuracy"] == pytest.approx(0.7)

    def test_bias_metrics_populated_when_present(self):
        metrics = self._metrics(("bias", 0.12), ("toxicity", 0.05))
        args = _build_model_eval_args("gpt-4o", metrics)
        assert args["bias_metrics"] == {"bias": 0.12, "toxicity": 0.05}

    def test_bias_metrics_none_when_absent(self):
        metrics = self._metrics(("answer relevancy", 0.9))
        args = _build_model_eval_args("gpt-4o", metrics)
        assert args["bias_metrics"] is None

    def test_robustness_derived_from_hallucination(self):
        metrics = self._metrics(("hallucination", 0.2))
        args = _build_model_eval_args("gpt-4o", metrics)
        assert args["robustness_score"] == pytest.approx(0.8)

    def test_passed_threshold_false_when_any_metric_failed(self):
        metrics = [
            {"name": "faithfulness", "score": 0.9, "success": True},
            {"name": "bias", "score": 0.4, "success": False},
        ]
        args = _build_model_eval_args("gpt-4o", metrics)
        assert args["passed_threshold"] is False

    def test_passed_threshold_true_when_all_pass(self):
        metrics = self._metrics(("faithfulness", 0.9))
        args = _build_model_eval_args("gpt-4o", metrics)
        assert args["passed_threshold"] is True

    def test_accuracy_falls_back_to_zero_when_no_quality_metrics(self):
        # Only bias/toxicity — no quality metrics
        metrics = self._metrics(("bias", 0.05))
        args = _build_model_eval_args("gpt-4o", metrics)
        assert args["accuracy"] == 0.0

    def test_notes_contains_all_metric_names(self):
        metrics = self._metrics(("faithfulness", 0.9), ("bias", 0.1))
        args = _build_model_eval_args("gpt-4o", metrics)
        assert "faithfulness" in args["notes"]
        assert "bias" in args["notes"]

    def test_notes_names_accuracy_composite_sources(self):
        metrics = self._metrics(("faithfulness", 0.9), ("answer relevancy", 0.8))
        args = _build_model_eval_args("gpt-4o", metrics)
        parsed = json.loads(args["notes"])
        assert "_accuracy_composite" in parsed
        composite = parsed["_accuracy_composite"]
        assert "faithfulness" in composite
        assert "answer relevancy" in composite

    def test_notes_states_robustness_derivation_when_hallucination_present(self):
        metrics = self._metrics(("hallucination", 0.15))
        args = _build_model_eval_args("gpt-4o", metrics)
        parsed = json.loads(args["notes"])
        assert "_robustness_derivation" in parsed
        assert "hallucination" in parsed["_robustness_derivation"].lower()

    def test_notes_no_robustness_derivation_when_no_hallucination(self):
        metrics = self._metrics(("faithfulness", 0.9))
        args = _build_model_eval_args("gpt-4o", metrics)
        parsed = json.loads(args["notes"])
        assert "_robustness_derivation" not in parsed

    def test_notes_no_accuracy_composite_key_when_no_quality_metrics(self):
        metrics = self._metrics(("bias", 0.05))
        args = _build_model_eval_args("gpt-4o", metrics)
        parsed = json.loads(args["notes"])
        assert "_accuracy_composite" not in parsed


import json

# ── _ci_identity ─────────────────────────────────────────────────────────────


class TestCiIdentity:
    def test_github_actions(self, monkeypatch):
        monkeypatch.setenv("GITHUB_ACTOR", "alice")
        monkeypatch.setenv("GITHUB_SHA",   "abc12345def")
        assert _ci_identity() == "github:alice@abc12345"

    def test_gitlab_ci(self, monkeypatch):
        monkeypatch.delenv("GITHUB_ACTOR", raising=False)
        monkeypatch.delenv("GITHUB_SHA", raising=False)
        monkeypatch.setenv("GITLAB_USER_LOGIN",    "bob")
        monkeypatch.setenv("CI_COMMIT_SHORT_SHA",  "abcd1234")
        assert _ci_identity() == "gitlab:bob@abcd1234"

    def test_generic_ci_user_with_sha(self, monkeypatch):
        monkeypatch.delenv("GITHUB_ACTOR", raising=False)
        monkeypatch.delenv("GITHUB_SHA", raising=False)
        monkeypatch.delenv("GITLAB_USER_LOGIN", raising=False)
        monkeypatch.setenv("CI_ACTOR",      "deploy-bot")
        monkeypatch.setenv("CI_COMMIT_SHA", "ffffffff00000000")
        assert _ci_identity() == "ci:deploy-bot@ffffffff"

    def test_no_ci_env_returns_empty_string(self, monkeypatch):
        for key in ("GITHUB_ACTOR", "GITHUB_SHA", "GITLAB_USER_LOGIN",
                    "CI_COMMIT_SHORT_SHA", "CI_ACTOR", "CI_USER", "CI_COMMIT_SHA"):
            monkeypatch.delenv(key, raising=False)
        assert _ci_identity() == ""


# ── METRIC_CONTROL_MAP completeness ───────────────────────────────────────────


class TestMetricControlMap:
    def test_bias_maps_to_euaia_art10(self):
        assert "EUAIA_ART10" in METRIC_CONTROL_MAP["bias"]

    def test_hallucination_maps_to_euaia_art15(self):
        assert "EUAIA_ART15" in METRIC_CONTROL_MAP["hallucination"]

    def test_toxicity_maps_to_soc2(self):
        assert "SOC2_CC4_1" in METRIC_CONTROL_MAP["toxicity"]


# ── report_to_mima (DeepEval) ─────────────────────────────────────────────────


class TestDeepEvalReportToMima:
    def test_calls_model_evaluation_with_correct_model_id(self):
        mima = _make_mima()
        result = _EvaluationResult(
            _TestResult(_MetricData("Answer Relevancy", 0.88))
        )
        deepeval_report(mima, model_id="gpt-4o", test_results=result)
        call_kwargs = mima.model_evaluation.call_args
        assert call_kwargs.args[0] == "gpt-4o"

    def test_returns_grc_result(self):
        mima = _make_mima()
        result = _EvaluationResult(_TestResult(_MetricData("Faithfulness", 0.9)))
        grc = deepeval_report(mima, model_id="claude-3-5", test_results=result)
        assert grc.record_id == "test-grc-id"

    def test_evaluation_type_defaults_to_triggered(self):
        mima = _make_mima()
        deepeval_report(
            mima,
            model_id="gpt-4o",
            test_results=_EvaluationResult(_TestResult(_MetricData("Faithfulness", 0.9))),
        )
        kwargs = mima.model_evaluation.call_args.kwargs
        assert kwargs["evaluation_type"] == "triggered"

    def test_custom_dataset_and_evaluated_by(self):
        mima = _make_mima()
        deepeval_report(
            mima,
            model_id="gpt-4o",
            test_results=_EvaluationResult(_TestResult(_MetricData("Faithfulness", 0.9))),
            dataset="staging-eval-2026",
            evaluated_by="ci-bot",
        )
        kwargs = mima.model_evaluation.call_args.kwargs
        assert kwargs["evaluated_by"] == "ci-bot"
        assert mima.model_evaluation.call_args.args[1] == "staging-eval-2026"

    def test_evaluated_by_auto_detected_from_github_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_ACTOR", "workflow-bot")
        monkeypatch.setenv("GITHUB_SHA",   "deadbeef1234")
        mima = _make_mima()
        deepeval_report(
            mima,
            model_id="gpt-4o",
            test_results=_EvaluationResult(_TestResult(_MetricData("Faithfulness", 0.9))),
        )
        kwargs = mima.model_evaluation.call_args.kwargs
        assert kwargs["evaluated_by"] == "github:workflow-bot@deadbeef"

    def test_evaluated_by_falls_back_to_deepeval_ci_with_no_env(self, monkeypatch):
        for key in ("GITHUB_ACTOR", "GITHUB_SHA", "GITLAB_USER_LOGIN",
                    "CI_COMMIT_SHORT_SHA", "CI_ACTOR", "CI_USER", "CI_COMMIT_SHA"):
            monkeypatch.delenv(key, raising=False)
        mima = _make_mima()
        deepeval_report(
            mima,
            model_id="gpt-4o",
            test_results=_EvaluationResult(_TestResult(_MetricData("Faithfulness", 0.9))),
        )
        kwargs = mima.model_evaluation.call_args.kwargs
        assert kwargs["evaluated_by"] == "deepeval-ci"

    def test_empty_results_uses_zero_accuracy(self):
        mima = _make_mima()
        deepeval_report(mima, model_id="gpt-4o", test_results=_EvaluationResult())
        args = mima.model_evaluation.call_args.args
        assert args[2] == 0.0  # accuracy

    def test_list_of_test_results_accepted(self):
        mima = _make_mima()
        deepeval_report(
            mima,
            model_id="gpt-4o",
            test_results=[_TestResult(_MetricData("Faithfulness", 0.75))],
        )
        mima.model_evaluation.assert_called_once()


# ── _parse_langfuse_scores ────────────────────────────────────────────────────


class TestParseLangfuseScores:
    def test_accepts_dicts(self):
        scores = [
            {"name": "accuracy", "value": 0.9, "comment": "good"},
            {"name": "faithfulness", "value": 0.85},
        ]
        parsed = _parse_langfuse_scores(scores)
        assert len(parsed) == 2
        assert parsed[0] == {"name": "accuracy", "score": 0.9, "success": True}

    def test_accepts_objects_with_attributes(self):
        class LFScore:
            def __init__(self, name, value):
                self.name = name
                self.value = value
                self.comment = None

        parsed = _parse_langfuse_scores([LFScore("bias", 0.05)])
        assert parsed[0] == {"name": "bias", "score": 0.05, "success": True}

    def test_normalises_names(self):
        parsed = _parse_langfuse_scores([{"name": "Answer Relevancy", "value": 0.8}])
        assert parsed[0]["name"] == "answer relevancy"

    def test_filters_out_non_numeric_values(self):
        scores = [
            {"name": "label", "value": "good"},   # string value — skip
            {"name": "accuracy", "value": 0.9},
        ]
        parsed = _parse_langfuse_scores(scores)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "accuracy"

    def test_empty_list(self):
        assert _parse_langfuse_scores([]) == []


# ── report_to_mima (Langfuse) ─────────────────────────────────────────────────


class TestLangfuseReportToMima:
    def test_calls_model_evaluation(self):
        mima = _make_mima()
        scores = [{"name": "faithfulness", "value": 0.87}]
        langfuse_report(mima, model_id="gpt-4o", scores=scores)
        mima.model_evaluation.assert_called_once()

    def test_accuracy_extracted_from_faithfulness(self):
        mima = _make_mima()
        scores = [{"name": "faithfulness", "value": 0.75}]
        langfuse_report(mima, model_id="gpt-4o", scores=scores)
        args = mima.model_evaluation.call_args.args
        assert args[2] == pytest.approx(0.75)

    def test_bias_populated_from_bias_score(self):
        mima = _make_mima()
        scores = [{"name": "bias", "value": 0.03}]
        langfuse_report(mima, model_id="gpt-4o", scores=scores)
        kwargs = mima.model_evaluation.call_args.kwargs
        assert kwargs["bias_metrics"] == {"bias": 0.03}

    def test_dataset_defaults_to_langfuse(self):
        mima = _make_mima()
        langfuse_report(mima, model_id="gpt-4o", scores=[{"name": "accuracy", "value": 0.9}])
        args = mima.model_evaluation.call_args.args
        assert args[1] == "langfuse"

    def test_custom_dataset(self):
        mima = _make_mima()
        langfuse_report(
            mima,
            model_id="gpt-4o",
            scores=[{"name": "accuracy", "value": 0.9}],
            dataset="prod-traces-2026-06",
        )
        args = mima.model_evaluation.call_args.args
        assert args[1] == "prod-traces-2026-06"

    def test_evaluated_by_falls_back_to_langfuse_with_no_env(self, monkeypatch):
        for key in ("GITHUB_ACTOR", "GITHUB_SHA", "GITLAB_USER_LOGIN",
                    "CI_COMMIT_SHORT_SHA", "CI_ACTOR", "CI_USER", "CI_COMMIT_SHA"):
            monkeypatch.delenv(key, raising=False)
        mima = _make_mima()
        langfuse_report(mima, model_id="gpt-4o", scores=[{"name": "accuracy", "value": 0.9}])
        kwargs = mima.model_evaluation.call_args.kwargs
        assert kwargs["evaluated_by"] == "langfuse"

    def test_empty_scores(self):
        mima = _make_mima()
        langfuse_report(mima, model_id="gpt-4o", scores=[])
        args = mima.model_evaluation.call_args.args
        assert args[2] == 0.0  # accuracy = 0 when no scores
