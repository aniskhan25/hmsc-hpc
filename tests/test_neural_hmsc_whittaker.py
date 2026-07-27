import subprocess

import pandas as pd
import pytest

from examples.run_neural_hmsc_whittaker import (
    _heldout_metrics,
    _load_reference_qualification,
    _metric_row,
)


class _FixedPrediction:
    def __init__(self, prediction: pd.DataFrame):
        self.prediction = prediction

    def predict_mean(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.prediction.loc[X.index]


def test_whittaker_metrics_report_prevalence_strata():
    index = pd.Index(["s1", "s2", "s3", "s4"])
    columns = pd.Index(["rare", "intermediate", "common"])
    X = pd.DataFrame({"TMG": [-1.0, -0.25, 0.25, 1.0]}, index=index)
    Y = pd.DataFrame(
        [[0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0]],
        index=index,
        columns=columns,
    )
    prediction = pd.DataFrame(0.5, index=index, columns=columns)

    row = _heldout_metrics(
        model="neural_predictive_only_calibrated",
        fit=_FixedPrediction(prediction),
        X=X,
        Y=Y,
        training_prevalence=pd.Series([0.05, 0.20, 0.50], index=columns),
    )

    assert row["rare_species"] == 1
    assert row["intermediate_species"] == 1
    assert row["common_species"] == 1
    assert row["rare_brier_score"] == 0.25
    assert row["predictive_rmse"] == 0.5
    assert row["intermediate_prevalence_mae"] == 0.0
    assert row["common_prevalence_mae"] == 0.25


def test_metric_row_requires_one_named_variant():
    metrics = pd.DataFrame([{"model": "uncalibrated", "brier_score": 0.1}])

    assert _metric_row(metrics, "uncalibrated")["brier_score"] == 0.1


def test_reference_qualification_requires_passed_parity(tmp_path):
    metrics = tmp_path / "parity.json"
    metrics.write_text(
        '{"parity_passed": false, "boundary_checks": {}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="did not pass"):
        _load_reference_qualification(metrics)


def test_reference_qualification_summarizes_direct_parity_metrics(tmp_path):
    metrics = tmp_path / "parity.json"
    metrics.write_text(
        """
{
  "parity_passed": true,
  "config": "examples/projects/big_spatial_plants_validation/model_spatial_full.yaml",
  "posterior_gates": "diagnostic",
  "boundary_checks": {
    "Y": {"passed": true},
    "X": {"passed": true}
  },
  "acceptance_gates": {
    "boundary_arrays": {"passed": true},
    "prediction_mae_delta": {"passed": true}
  },
  "beta_compare": {"mean_correlation": 0.98},
  "gamma_compare": {"mean_correlation": 0.99},
  "random_level_compare": {
    "levels": [
      {"association_compare": {"mean_correlation": 0.75}}
    ]
  },
  "metric_deltas_python_native_minus_r_bridge": {
    "prediction_mae": -0.01
  }
}
""",
        encoding="utf-8",
    )

    summary = _load_reference_qualification(metrics)

    assert summary["parity_passed"]
    assert summary["boundary_arrays_passed"]
    assert summary["acceptance_gates_passed"]
    assert summary["source"].endswith("model_spatial_full.yaml")
    assert summary["beta_mean_correlation"] == 0.98
    assert summary["random_level_association_correlation"] == 0.75
    assert summary["metric_deltas_python_native_minus_r_bridge"]["prediction_mae"] == -0.01


def test_neural_whittaker_lumi_script_syntax():
    text = open("docs/lumi_neural_hmsc_whittaker_sbatch.sh", encoding="utf-8").read()

    subprocess.run(
        ["bash", "-n", "docs/lumi_neural_hmsc_whittaker_sbatch.sh"],
        check=True,
    )

    assert 'COEFFICIENT_CALIBRATION="${COEFFICIENT_CALIBRATION:-external_monotone}"' in text
    assert "--coefficient-calibration" in text
    assert "--external-monotone-datasets" in text
    assert "REFERENCE_PARITY_METRICS" in text
    assert "--reference-parity-metrics" in text


def test_whittaker_runner_supports_transfer_response_affine():
    text = open("examples/run_neural_hmsc_whittaker.py", encoding="utf-8").read()

    assert '"probit_transfer_response_affine"' in text
    assert "fit_beta_transfer_response_mean_calibration" in text
    assert "transfer_validation:covariate_shift" not in text
    assert 'label=f"transfer_validation:{regime}"' in text
    assert '"probit_source_transfer_response_affine"' in text
    assert "fit_beta_transfer_response_branch_calibration" in text
    assert "independent_source_transfer_predictive_mean_selector_metadata" in text
