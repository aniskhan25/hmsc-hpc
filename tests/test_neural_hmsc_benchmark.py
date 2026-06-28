import json

import h5py
import numpy as np
import pandas as pd
import pytest

from pyhmsc.neural.benchmark import compare_beta_posterior_files, write_benchmark_report


def test_compare_beta_posterior_files_reports_reference_metrics(tmp_path):
    truth = pd.DataFrame(
        [[0.5, -0.25], [1.0, 0.25]],
        index=["Intercept", "x1"],
        columns=["sp1", "sp2"],
    )
    neural_samples = np.array(
        [
            [
                [[0.45, -0.20], [0.90, 0.30]],
                [[0.55, -0.30], [1.10, 0.20]],
                [[0.50, -0.25], [1.05, 0.25]],
            ]
        ],
        dtype=float,
    )
    mcmc_samples = np.array(
        [
            [
                [[0.40, -0.30], [0.95, 0.20]],
                [[0.60, -0.20], [1.05, 0.30]],
                [[0.50, -0.25], [1.00, 0.25]],
            ],
            [
                [[0.52, -0.24], [1.02, 0.28]],
                [[0.48, -0.26], [0.98, 0.22]],
                [[0.51, -0.23], [1.01, 0.24]],
            ],
        ],
        dtype=float,
    )
    neural_path = _write_posterior(tmp_path / "neural.h5", neural_samples)
    mcmc_path = _write_posterior(tmp_path / "mcmc.h5", mcmc_samples)
    truth_path = tmp_path / "truth_beta.csv"
    truth.to_csv(truth_path)

    row = compare_beta_posterior_files(
        neural_posterior=neural_path,
        mcmc_posterior=mcmc_path,
        truth_beta=truth_path,
        dataset="unit",
        distribution="normal",
    )

    assert row["dataset"] == "unit"
    assert row["distribution"] == "normal"
    assert row["n_covariates"] == 2
    assert row["n_species"] == 2
    assert row["beta_mean_rmse_mcmc"] >= 0.0
    assert row["beta_sd_rmse_mcmc"] >= 0.0
    assert 0.0 <= row["beta_ci_overlap_95"] <= 1.0
    assert row["neural_beta_mean_rmse_truth"] < 0.1
    assert row["mcmc_beta_mean_rmse_truth"] < 0.1
    assert 0.0 <= row["neural_beta_interval_coverage_truth_95"] <= 1.0


def test_compare_beta_posterior_files_adds_predictive_and_runtime_metrics(tmp_path):
    truth = pd.DataFrame(
        [[0.0], [1.0]],
        index=["Intercept", "x1"],
        columns=["sp1"],
    )
    samples = np.array([[[[0.0], [1.0]], [[0.1], [0.9]], [[-0.1], [1.1]]]], dtype=float)
    neural_path = _write_posterior(tmp_path / "neural.h5", samples)
    mcmc_path = _write_posterior(tmp_path / "mcmc.h5", samples + 0.01)
    truth_path = tmp_path / "truth_beta.csv"
    truth.to_csv(truth_path)
    X = pd.DataFrame({"x1": [-1.0, 0.0, 1.0]}, index=["s1", "s2", "s3"])
    Y = pd.DataFrame({"sp1": [-1.0, 0.0, 1.0]}, index=X.index)
    x_path = tmp_path / "X.csv"
    y_path = tmp_path / "Y.csv"
    X.to_csv(x_path)
    Y.to_csv(y_path)

    row = compare_beta_posterior_files(
        neural_posterior=neural_path,
        mcmc_posterior=mcmc_path,
        truth_beta=truth_path,
        dataset="predictive",
        distribution="normal",
        neural_seconds=0.5,
        mcmc_seconds=5.0,
        X=x_path,
        Y=y_path,
        formula="~ x1",
    )

    assert row["speedup_factor"] == 10.0
    assert row["neural_posterior_predictive_mean_rmse"] < 0.1
    assert row["mcmc_posterior_predictive_mean_rmse"] < 0.1
    assert 0.0 <= row["neural_species_mean_coverage_95"] <= 1.0


def test_poisson_predictive_metrics_respect_declared_eta_clip(tmp_path):
    samples = np.array([[[[10.0]], [[12.0]], [[14.0]]]], dtype=float)
    neural_path = _write_posterior(tmp_path / "neural.h5", samples, distribution="poisson")
    mcmc_path = _write_posterior(tmp_path / "mcmc.h5", samples, distribution="poisson")
    X = pd.DataFrame(index=["s1"])
    Y = pd.DataFrame({"sp1": [np.exp(6.0)]}, index=X.index)

    row = compare_beta_posterior_files(
        neural_posterior=neural_path,
        mcmc_posterior=mcmc_path,
        distribution="poisson",
        X=X,
        Y=Y,
        formula="~ 1",
        poisson_eta_clip=(-6.0, 6.0),
    )

    assert row["predictive_poisson_eta_clip_lower"] == -6.0
    assert row["predictive_poisson_eta_clip_upper"] == 6.0
    assert row["neural_posterior_predictive_mean_rmse"] == pytest.approx(0.0)
    assert row["mcmc_posterior_predictive_mean_rmse"] == pytest.approx(0.0)
    assert row["neural_poisson_eta_clipped_fraction"] == pytest.approx(1.0)
    assert row["mcmc_poisson_eta_clipped_fraction"] == pytest.approx(1.0)


def test_poisson_predictive_metrics_reject_invalid_eta_clip(tmp_path):
    samples = np.zeros((1, 2, 1, 1), dtype=float)
    neural_path = _write_posterior(tmp_path / "neural.h5", samples, distribution="poisson")
    mcmc_path = _write_posterior(tmp_path / "mcmc.h5", samples, distribution="poisson")

    with pytest.raises(ValueError, match="finite, ordered bounds"):
        compare_beta_posterior_files(
            neural_posterior=neural_path,
            mcmc_posterior=mcmc_path,
            distribution="poisson",
            X=pd.DataFrame(index=["s1"]),
            Y=pd.DataFrame({"sp1": [1.0]}, index=["s1"]),
            formula="~ 1",
            poisson_eta_clip=(6.0, -6.0),
        )


def test_poisson_predictive_metrics_fail_loudly_on_unbounded_overflow(tmp_path):
    samples = np.full((1, 2, 1, 1), 1000.0, dtype=float)
    neural_path = _write_posterior(tmp_path / "neural.h5", samples, distribution="poisson")
    mcmc_path = _write_posterior(tmp_path / "mcmc.h5", samples, distribution="poisson")

    with pytest.raises(ValueError, match="Poisson response predictions overflowed"):
        compare_beta_posterior_files(
            neural_posterior=neural_path,
            mcmc_posterior=mcmc_path,
            distribution="poisson",
            X=pd.DataFrame(index=["s1"]),
            Y=pd.DataFrame({"sp1": [1.0]}, index=["s1"]),
            formula="~ 1",
        )


def test_write_benchmark_report_writes_csv_and_markdown(tmp_path):
    paths = write_benchmark_report(
        [
            {
                "dataset": "normal",
                "distribution": "normal",
                "beta_mean_rmse_mcmc": 0.1,
                "beta_sd_rmse_mcmc": 0.2,
                "beta_ci_overlap_95": 0.8,
            }
        ],
        tmp_path,
        stem="report",
    )

    assert paths.csv.exists()
    assert paths.markdown.exists()
    assert "Neural-HMSC MCMC Reference Benchmark" in paths.markdown.read_text(encoding="utf-8")
    assert pd.read_csv(paths.csv).loc[0, "dataset"] == "normal"


def test_benchmark_report_exposes_neural_calibration_metadata(tmp_path):
    samples = np.array([[[[0.0], [1.0]], [[0.1], [0.9]], [[-0.1], [1.1]]]], dtype=float)
    calibration = {
        "method": "temperature_scale",
        "scale_multiplier": 1.5,
        "nominal_level": 0.95,
        "uncalibrated_coverage": 0.5,
        "calibrated_coverage": 0.95,
        "n_observations": 4,
        "domain": {"distribution": "normal", "n_covariates": 2, "n_species": 1},
    }
    neural_path = _write_posterior(tmp_path / "neural.h5", samples, calibration=calibration)
    mcmc_path = _write_posterior(tmp_path / "mcmc.h5", samples)

    row = compare_beta_posterior_files(
        neural_posterior=neural_path,
        mcmc_posterior=mcmc_path,
        dataset="calibrated",
    )

    assert row["neural_calibration_method"] == "temperature_scale"
    assert row["neural_calibration_scale_multiplier"] == 1.5
    assert row["neural_calibration_calibrated_coverage"] == 0.95
    assert row["neural_calibration_domain_distribution"] == "normal"


def _write_posterior(path, beta_samples, calibration=None, distribution="normal"):
    metadata = {
        "names": {
            "covariates": ["Intercept", "x1"][: beta_samples.shape[2]],
            "species": ["sp1", "sp2"][: beta_samples.shape[3]],
            "traits": ["Intercept"],
        },
        "formula": {"X": "~ x1"},
        "distribution": distribution,
    }
    if calibration is not None:
        metadata["calibration"] = calibration
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Beta", data=beta_samples)
        handle.attrs["nChains"] = int(beta_samples.shape[0])
        handle.attrs["nDraws"] = int(beta_samples.shape[1])
        handle.attrs["pyhmsc_metadata"] = json.dumps(metadata)
    return path
