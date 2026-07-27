import numpy as np
import pandas as pd

from examples.compare_neural_hmsc_predictive_scores import (
    compare_predictive_runs,
    write_predictive_score_report,
)
from pyhmsc.neural.posterior_heads import BetaPosterior
from pyhmsc.neural.storage import write_beta_posterior_hdf5


def test_compare_predictive_runs_reports_score_ratios(tmp_path):
    X = pd.DataFrame(index=["s1", "s2", "s3"])
    Y = pd.DataFrame({"sp1": [0.0, 1.0, 1.0]}, index=X.index)
    base = tmp_path / "base" / "probit"
    candidate = tmp_path / "candidate" / "probit"
    for root in [base, candidate]:
        (root / "data").mkdir(parents=True)
        X.to_csv(root / "data" / "X.csv")
        Y.to_csv(root / "data" / "Y.csv")

    base_posterior = BetaPosterior(
        mean=np.array([[[-0.5]]], dtype=np.float32),
        scale=np.ones((1, 1, 1), dtype=np.float32) * 0.01,
    )
    candidate_posterior = BetaPosterior(
        mean=np.array([[[0.5]]], dtype=np.float32),
        scale=np.ones((1, 1, 1), dtype=np.float32) * 0.01,
    )
    write_beta_posterior_hdf5(
        base_posterior,
        base / "neural_predictive_distribution.h5",
        covariate_names=["Intercept"],
        species_names=["sp1"],
        distribution="probit",
        formula="~ 1",
        chains=1,
        draws=8,
        seed=1,
    )
    write_beta_posterior_hdf5(
        candidate_posterior,
        candidate / "neural_predictive_distribution.h5",
        covariate_names=["Intercept"],
        species_names=["sp1"],
        distribution="probit",
        formula="~ 1",
        chains=1,
        draws=8,
        seed=2,
        metadata={
            "predictive_mean_calibration": {
                "method": "probit_response_affine",
                "selected": True,
            }
        },
    )

    result = compare_predictive_runs(
        [
            {"label": "base", "path": str(base.parent), "predictive": base / "neural_predictive_distribution.h5", "data_dir": base / "data"},
            {
                "label": "candidate",
                "path": str(candidate.parent),
                "predictive": candidate / "neural_predictive_distribution.h5",
                "data_dir": candidate / "data",
            },
        ],
        baseline_label="base",
    )
    rows = {row["run"]: row for row in result["summary"]}

    assert rows["candidate"]["brier_score_ratio_vs_baseline"] < 1.0
    assert rows["candidate"]["log_loss_ratio_vs_baseline"] < 1.0
    assert (
        rows["candidate"]["predictive_mean_calibration"]["method"]
        == "probit_response_affine"
    )

    paths = write_predictive_score_report(result, tmp_path / "report")
    assert paths["json"].exists()
    assert paths["markdown"].exists()
