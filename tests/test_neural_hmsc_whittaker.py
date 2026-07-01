import subprocess

import pandas as pd

from examples.run_neural_hmsc_whittaker import _heldout_metrics, _metric_row


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
    assert row["intermediate_prevalence_mae"] == 0.0
    assert row["common_prevalence_mae"] == 0.25


def test_metric_row_requires_one_named_variant():
    metrics = pd.DataFrame([{"model": "uncalibrated", "brier_score": 0.1}])

    assert _metric_row(metrics, "uncalibrated")["brier_score"] == 0.1


def test_neural_whittaker_lumi_script_syntax():
    subprocess.run(
        ["bash", "-n", "docs/lumi_neural_hmsc_whittaker_sbatch.sh"],
        check=True,
    )
