import pandas as pd
import pytest

from pyhmsc.neural.predictive_selection import (
    PredictiveNoDegradationThresholds,
    evaluate_cross_dataset_predictive_gate,
)


def _metrics(
    scale_brier,
    scale_log,
    candidate_brier,
    candidate_log,
    *,
    scale_richness=1.0,
    candidate_richness=0.99,
):
    return pd.DataFrame(
        [
            {
                "model": "neural_predictive_only_calibrated",
                "brier_score": scale_brier,
                "log_loss": scale_log,
                "predictive_rmse": scale_brier**0.5,
                "richness_mae": scale_richness,
            },
            {
                "model": "neural_predictive_mean_calibrated",
                "brier_score": candidate_brier,
                "log_loss": candidate_log,
                "predictive_rmse": candidate_brier**0.5,
                "richness_mae": candidate_richness,
            },
        ]
    )


def test_cross_dataset_gate_rejects_any_realdata_degradation():
    result = evaluate_cross_dataset_predictive_gate(
        [
            {
                "label": "whittaker",
                "metrics": _metrics(0.10, 0.20, 0.101, 0.199),
            },
            {
                "label": "big_spatial",
                "metrics": _metrics(0.12, 0.22, 0.119, 0.219),
            },
        ]
    )

    assert not result["promotion_gate_passed"]
    assert result["datasets"][0]["brier_score_ratio"] > 1.0
    assert "whittaker" in result["failure_reasons"][0]


def test_cross_dataset_gate_passes_when_every_dataset_improves():
    result = evaluate_cross_dataset_predictive_gate(
        [
            {"label": "whittaker", "metrics": _metrics(0.10, 0.20, 0.099, 0.199)},
            {"label": "big_spatial", "metrics": _metrics(0.12, 0.22, 0.119, 0.219)},
        ]
    )

    assert result["promotion_gate_passed"]
    assert all(row["passed"] for row in result["datasets"])


def test_cross_dataset_gate_can_require_simulated_gain():
    result = evaluate_cross_dataset_predictive_gate(
        [{"label": "whittaker", "metrics": _metrics(0.10, 0.20, 0.099, 0.199)}],
        thresholds=PredictiveNoDegradationThresholds(
            min_simulated_brier_gain=0.002,
            min_simulated_log_loss_gain=0.002,
        ),
        simulated_summary=[
            {"run": "external_monotone", "brier_score": 0.20, "log_loss": 0.40},
            {
                "run": "external_monotone_response",
                "brier_score": 0.199,
                "log_loss": 0.399,
            },
        ],
    )

    assert not result["promotion_gate_passed"]
    assert not result["simulated_gate"]["passed"]
    assert any("simulated" in reason for reason in result["failure_reasons"])


def test_cross_dataset_gate_reads_aggregate_simulated_columns():
    result = evaluate_cross_dataset_predictive_gate(
        [{"label": "whittaker", "metrics": _metrics(0.10, 0.20, 0.099, 0.199)}],
        thresholds=PredictiveNoDegradationThresholds(
            min_simulated_brier_gain=0.0001,
            min_simulated_log_loss_gain=0.0001,
        ),
        simulated_summary=[
            {
                "run": "external_monotone",
                "brier_score_mean": 0.20,
                "log_loss_mean": 0.40,
            },
            {
                "run": "external_monotone_response",
                "brier_score_mean": 0.199,
                "log_loss_mean": 0.399,
            },
        ],
    )

    assert result["promotion_gate_passed"]
    assert result["simulated_gate"]["passed"]


def test_cross_dataset_gate_rejects_rmse_or_richness_degradation():
    result = evaluate_cross_dataset_predictive_gate(
        [
            {
                "label": "whittaker",
                "metrics": _metrics(
                    0.10,
                    0.20,
                    0.099,
                    0.199,
                    candidate_richness=1.01,
                ),
            }
        ]
    )

    assert not result["promotion_gate_passed"]
    assert result["datasets"][0]["richness_mae_ratio"] > 1.0


def test_cross_dataset_gate_requires_positive_mean_realdata_gain():
    metrics = _metrics(0.10, 0.20, 0.10, 0.20, candidate_richness=1.0)
    result = evaluate_cross_dataset_predictive_gate(
        [{"label": "whittaker", "metrics": metrics}],
        thresholds=PredictiveNoDegradationThresholds(
            min_mean_brier_gain=1.0e-6,
            min_mean_log_loss_gain=1.0e-6,
        ),
    )

    assert not result["promotion_gate_passed"]
    assert any("mean Brier gain" in reason for reason in result["failure_reasons"])


def test_thresholds_reject_invalid_ratios():
    with pytest.raises(ValueError, match="max_brier_ratio"):
        evaluate_cross_dataset_predictive_gate(
            [{"label": "whittaker", "metrics": _metrics(0.10, 0.20, 0.099, 0.199)}],
            thresholds=PredictiveNoDegradationThresholds(max_brier_ratio=0.99),
        )
