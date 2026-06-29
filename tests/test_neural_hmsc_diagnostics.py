import json

import numpy as np
import pandas as pd
import pytest

from pyhmsc.neural.benchmark import write_sbc_report
from pyhmsc.neural.diagnostics import beta_sbc_rank_diagnostics


def test_beta_sbc_rank_diagnostics_recovers_constructed_uniform_ranks():
    n_draws = 9
    samples = np.ones((n_draws + 1, n_draws, 1, 1), dtype=float)
    truth = np.zeros((n_draws + 1, 1, 1), dtype=float)
    for rank in range(n_draws + 1):
        samples[rank, :rank, 0, 0] = -1.0

    diagnostics = beta_sbc_rank_diagnostics(samples, truth, n_bins=5, seed=7)

    assert diagnostics.histogram_counts == (2, 2, 2, 2, 2)
    assert diagnostics.rank_mean == pytest.approx(0.5)
    assert diagnostics.rank_variance == pytest.approx(diagnostics.expected_rank_variance)
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

    assert json.loads(paths.json.read_text(encoding="utf-8"))[0]["sbc_histogram_counts"] == [2, 2, 2, 2]
    assert json.loads(pd.read_csv(paths.csv).loc[0, "sbc_histogram_counts"]) == [2, 2, 2, 2]
    assert "Simulation-Based Calibration" in paths.markdown.read_text(encoding="utf-8")
