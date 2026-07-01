import json

import numpy as np
import pandas as pd
import pytest

from pyhmsc.neural.benchmark import write_sbc_report
from pyhmsc.neural.diagnostics import (
    beta_expected_design_information,
    beta_sbc_rank_diagnostics,
    beta_sbc_stratified_diagnostics,
)


def test_beta_sbc_rank_diagnostics_recovers_constructed_uniform_ranks():
    n_draws = 9
    samples = np.ones((n_draws + 1, n_draws, 1, 1), dtype=float)
    truth = np.zeros((n_draws + 1, 1, 1), dtype=float)
    for rank in range(n_draws + 1):
        samples[rank, :rank, 0, 0] = -1.0

    diagnostics = beta_sbc_rank_diagnostics(samples, truth, n_bins=5, seed=7)

    assert diagnostics.histogram_counts == (2, 2, 2, 2, 2)
    assert diagnostics.rank_mean == pytest.approx(0.5)
    assert diagnostics.rank_variance == pytest.approx(
        diagnostics.expected_rank_variance
    )
    assert diagnostics.chi_square_statistic == pytest.approx(0.0)
    assert diagnostics.chi_square_pvalue == pytest.approx(1.0)


def test_beta_sbc_rank_diagnostics_exposes_upper_rank_bias():
    samples = np.zeros((20, 10, 1, 1), dtype=float)
    truth = np.ones((20, 1, 1), dtype=float)

    diagnostics = beta_sbc_rank_diagnostics(samples, truth, n_bins=5, seed=3)

    assert diagnostics.rank_mean > 0.9
    assert diagnostics.upper_tail_fraction == 1.0
    assert diagnostics.beta_interval_coverage_95 == 0.0
    assert diagnostics.chi_square_pvalue < 0.01


def test_beta_sbc_rank_diagnostics_validates_shapes():
    with pytest.raises(ValueError, match="beta_true shape"):
        beta_sbc_rank_diagnostics(
            np.zeros((4, 10, 2, 1)),
            np.zeros((4, 1, 1)),
        )


def test_beta_sbc_rank_diagnostics_applies_instance_mask():
    samples = np.zeros((4, 5, 2, 1), dtype=float)
    truth = np.zeros((4, 2, 1), dtype=float)
    samples[:, :, 1, 0] = -1.0
    mask = np.zeros(truth.shape, dtype=bool)
    mask[:, 1, 0] = True

    diagnostics = beta_sbc_rank_diagnostics(
        samples,
        truth,
        n_bins=3,
        seed=3,
        coefficient_mask=mask,
    )

    assert diagnostics.n_coefficients == 1
    assert diagnostics.n_ranks == 4
    assert diagnostics.rank_mean > 0.9
    assert diagnostics.beta_interval_coverage_95 == 0.0


def test_beta_sbc_stratified_diagnostics_reports_expected_groups():
    rng = np.random.default_rng(11)
    replicates, draws, sites, covariates, species = 6, 9, 10, 2, 3
    truth = rng.normal(size=(replicates, covariates, species))
    samples = truth[:, None, :, :] + rng.normal(
        scale=0.4,
        size=(replicates, draws, covariates, species),
    )
    X = np.empty((replicates, sites, covariates), dtype=float)
    for replicate in range(replicates):
        X[replicate, :, 0] = 1.0
        X[replicate, :, 1] = np.linspace(-1.0, 1.0, sites) * (replicate + 1)
    Y = np.zeros((replicates, sites, species), dtype=float)
    Y[:, :1, 0] = 1.0
    Y[:, :2, 1] = 1.0
    Y[:, :7, 2] = 1.0

    rows = beta_sbc_stratified_diagnostics(
        samples,
        truth,
        X=X,
        Y=Y,
        distribution="probit",
        covariate_names=["Intercept", "x1"],
        n_bins=5,
        seed=5,
    )

    labels = {(row.kind, row.label) for row in rows}
    assert ("overall", "overall") in labels
    assert {label for kind, label in labels if kind == "prevalence"} == {
        "rare",
        "intermediate",
        "common",
    }
    assert {label for kind, label in labels if kind == "coefficient"} == {
        "Intercept",
        "x1",
    }
    assert {label for kind, label in labels if kind == "design_information"} == {
        "low",
        "intermediate",
        "high",
    }
    overall = next(row for row in rows if row.kind == "overall")
    rare = next(row for row in rows if (row.kind, row.label) == ("prevalence", "rare"))
    intercept = next(
        row for row in rows if (row.kind, row.label) == ("coefficient", "Intercept")
    )
    assert overall.diagnostics.n_ranks == replicates * covariates * species
    assert rare.diagnostics.n_ranks == replicates * covariates
    assert intercept.diagnostics.n_ranks == replicates * species


def test_beta_expected_design_information_matches_gaussian_curvature():
    samples = np.zeros((2, 3, 2, 1), dtype=float)
    X = np.array(
        [
            [[1.0, -1.0], [1.0, 0.0], [1.0, 2.0]],
            [[1.0, -2.0], [1.0, 1.0], [1.0, 3.0]],
        ]
    )

    information = beta_expected_design_information(samples, X, distribution="normal")

    np.testing.assert_allclose(information[:, 0, 0], [3.0, 3.0])
    np.testing.assert_allclose(information[:, 1, 0], [5.0, 14.0])


def test_write_sbc_report_serializes_histograms(tmp_path):
    row = {
        "distribution": "normal",
        "simulation_domain": "in_distribution",
        "ood_regime": None,
        "posterior_variant": "calibrated",
        "sbc_n_replicates": 8,
        "sbc_n_draws": 16,
        "sbc_histogram_counts": [2, 2, 2, 2],
        "sbc_rank_mean": 0.5,
        "sbc_rank_variance": 0.08,
        "sbc_expected_rank_variance": 0.08,
        "sbc_chi_square_pvalue": 1.0,
        "sbc_beta_mean_rmse": 0.1,
        "sbc_beta_interval_coverage_95": 0.95,
    }

    paths = write_sbc_report([row], tmp_path)

    assert json.loads(paths.json.read_text(encoding="utf-8"))[0][
        "sbc_histogram_counts"
    ] == [2, 2, 2, 2]
    assert json.loads(pd.read_csv(paths.csv).loc[0, "sbc_histogram_counts"]) == [
        2,
        2,
        2,
        2,
    ]
    assert "Simulation-Based Calibration" in paths.markdown.read_text(encoding="utf-8")
