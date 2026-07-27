import json
import subprocess
from pathlib import Path

import pandas as pd

from examples.aggregate_neural_hmsc_realdata_sensitivity import (
    aggregate_sensitivity,
    summarize_sensitivity,
)


def test_realdata_sensitivity_aggregates_paired_seed_runs(tmp_path):
    seeds = [101, 102, 103]
    for seed in seeds:
        seed_root = tmp_path / f"seed_{seed}"
        _write_whittaker(seed_root / "whittaker")
        _write_big_spatial(seed_root / "big_spatial")
        _write_promotion_gate(seed_root / "promotion_gate")

    rows = aggregate_sensitivity(tmp_path, seeds, strict=True)
    summary = summarize_sensitivity(pd.DataFrame(rows))

    assert len(rows) == 6
    assert summary["decision"] == "stable_guarded_selector_promotion_candidate"
    assert summary["paired_pass_count"] == 3
    assert summary["paired_promotion_gate_pass_count"] == 3
    assert summary["paired_genuine_transfer_improvement_count"] == 3
    assert summary["paired_safe_identity_fallback_count"] == 0
    assert summary["paired_mcmc_advantage_count"] == 3
    assert summary["whittaker"]["acceptance_pass_count"] == 3
    assert summary["whittaker"]["promotion_gate_pass_count"] == 3
    assert summary["big_spatial"]["acceptance_pass_count"] == 3
    assert summary["big_spatial"]["promotion_gate_pass_count"] == 3
    assert summary["big_spatial"]["genuine_transfer_improvement_count"] == 3
    assert summary["whittaker"]["predictive_vs_reference_brier_ratio_mean"] > 1.0


def test_realdata_sensitivity_does_not_promote_identity_only_fallback(tmp_path):
    seeds = [101, 102, 103]
    for seed in seeds:
        seed_root = tmp_path / f"seed_{seed}"
        _write_whittaker(seed_root / "whittaker")
        _write_big_spatial(seed_root / "big_spatial", selector_action="identity")
        _write_promotion_gate(
            seed_root / "promotion_gate",
            big_spatial_brier_ratio=1.0,
            big_spatial_log_loss_ratio=1.0,
        )

    rows = aggregate_sensitivity(tmp_path, seeds, strict=True)
    summary = summarize_sensitivity(pd.DataFrame(rows))

    assert summary["paired_pass_count"] == 3
    assert summary["paired_promotion_gate_pass_count"] == 3
    assert summary["paired_genuine_transfer_improvement_count"] == 0
    assert summary["paired_safe_identity_fallback_count"] == 3
    assert summary["decision"] == "safe_identity_fallback_not_promotable"


def test_realdata_sensitivity_lumi_script_syntax():
    script = Path("docs/lumi_neural_hmsc_realdata_sensitivity_sbatch.sh")
    text = script.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(script)], check=True)

    assert "SEEDS" in text
    assert "run_neural_hmsc_whittaker.py" in text
    assert "run_neural_hmsc_big_spatial_transfer.py" in text
    assert "aggregate_neural_hmsc_realdata_sensitivity.py" in text
    assert "evaluate_neural_hmsc_predictive_promotion.py" in text
    assert "PREDICTIVE_MEAN_SELECTION_POLICY" in text
    assert "WHITTAKER_REFERENCE_PARITY_METRICS" in text
    assert "BIG_SPATIAL_REFERENCE_PARITY_METRICS" in text


def _write_whittaker(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "whittaker_acceptance.json").write_text(
        json.dumps(
            {
                "qualification_acceptance_passed": True,
                "reference_parity_qualified": True,
                "predictive_mean_selector_decision": {
                    "action": "identity",
                    "selected": False,
                    "context_family": "source_like",
                    "reason": "source_like_context_uses_identity",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_heldout(root / "whittaker_heldout_metrics.csv")
    (root / "whittaker_neural_sbc_diagnostics.csv").write_text(
        "\n".join(
            [
                "posterior_variant,sbc_stratum_kind,sbc_beta_interval_coverage_95,sbc_rank_mean,sbc_rank_variance,sbc_beta_mean_rmse",
                "coefficient_calibrated,overall,0.956,0.495,0.070,0.46",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_metadata(root / "run_metadata.json")


def _write_big_spatial(root: Path, *, selector_action: str = "apply_candidate") -> None:
    root.mkdir(parents=True)
    selector_selected = selector_action == "apply_candidate"
    selector_reason = (
        "active_transfer_like_context"
        if selector_selected
        else "candidate_failed_transfer_stability_guard"
    )
    (root / "big_spatial_transfer_acceptance.json").write_text(
        json.dumps(
            {
                "predictive_transfer_acceptance_passed": True,
                "reference_parity_qualified": True,
                "predictive_mean_selector_decision": {
                    "action": selector_action,
                    "selected": selector_selected,
                    "context_family": "transfer_like",
                    "reason": selector_reason,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_heldout(
        root / "big_spatial_transfer_heldout_metrics.csv",
        predictive_mean_is_identity=not selector_selected,
    )
    _write_metadata(root / "run_metadata.json")


def _write_heldout(path: Path, *, predictive_mean_is_identity: bool = False) -> None:
    predictive_brier = 0.055
    predictive_log_loss = 0.22
    predictive_mean_brier = 0.054
    predictive_mean_log_loss = 0.218
    if predictive_mean_is_identity:
        predictive_mean_brier = predictive_brier
        predictive_mean_log_loss = predictive_log_loss
    pd.DataFrame(
        [
            {
                "model": "neural_uncalibrated",
                "brier_score": 0.06,
                "log_loss": 0.24,
                "macro_auc": 0.61,
                "prevalence_mae": 0.06,
                "richness_mae": 5.0,
            },
            {
                "model": "neural_predictive_only_calibrated",
                "brier_score": predictive_brier,
                "log_loss": predictive_log_loss,
                "macro_auc": 0.62,
                "prevalence_mae": 0.055,
                "richness_mae": 4.7,
            },
            {
                "model": "neural_predictive_mean_calibrated",
                "brier_score": predictive_mean_brier,
                "log_loss": predictive_mean_log_loss,
                "macro_auc": 0.621,
                "prevalence_mae": 0.054,
                "richness_mae": 4.6,
            },
            {
                "model": "qualified_python_mcmc_fixed",
                "brier_score": 0.05,
                "log_loss": 0.20,
                "macro_auc": 0.64,
                "prevalence_mae": 0.04,
                "richness_mae": 3.6,
            },
        ]
    ).to_csv(path, index=False)


def _write_promotion_gate(
    root: Path,
    *,
    big_spatial_brier_ratio: float = 0.99,
    big_spatial_log_loss_ratio: float = 0.98,
) -> None:
    root.mkdir(parents=True)
    (root / "predictive_mean_promotion_gate.json").write_text(
        json.dumps(
            {
                "promotion_gate_passed": True,
                "datasets": [
                    {
                        "dataset": "whittaker",
                        "passed": True,
                        "brier_score_ratio": 1.0,
                        "log_loss_ratio": 1.0,
                        "failure_reasons": [],
                    },
                    {
                        "dataset": "big_spatial",
                        "passed": True,
                        "brier_score_ratio": big_spatial_brier_ratio,
                        "log_loss_ratio": big_spatial_log_loss_ratio,
                        "failure_reasons": [],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_metadata(path: Path) -> None:
    path.write_text(
        json.dumps({"neural_inference_seconds": 0.1, "mcmc_seconds": 30.0}) + "\n",
        encoding="utf-8",
    )
