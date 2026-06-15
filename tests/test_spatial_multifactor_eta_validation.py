import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from examples.analyze_spatial_multifactor_eta_validation import build_metrics, _match_estimated_factors_to_truth
from pyhmsc.config import load_model_config, model_from_config
from pyhmsc.simulate import simulate_spatial_multifactor_eta_effect_data
from pyhmsc.validation import validate_compiled_native_model


PROJECT = Path("examples/projects/simulated_spatial_multifactor_eta_validation")


def test_spatial_multifactor_eta_project_structure():
    expected = [
        "README.md",
        "model_spatial_nngp.yaml",
        "data/Y.csv",
        "data/X.csv",
        "data/study_design.csv",
        "data/truth_beta.csv",
        "data/truth_eta.csv",
        "data/truth_lambda.csv",
        "data/truth_linear_predictor.csv",
    ]
    for relative in expected:
        assert (PROJECT / relative).exists(), relative


def test_spatial_multifactor_eta_project_files_match_simulator():
    Y, X, study_design, truth = simulate_spatial_multifactor_eta_effect_data(
        n_sites=64,
        n_species=8,
        n_factors=2,
        spatial_ranges=(0.20, 0.45),
        spatial_sd=1.2,
        lambda_scale=1.1,
        noise_sd=0.08,
        distr="normal",
        seed=211,
    )
    base = PROJECT / "data"
    pd.testing.assert_frame_equal(pd.read_csv(base / "Y.csv", index_col=0), Y)
    pd.testing.assert_frame_equal(pd.read_csv(base / "X.csv", index_col=0), X)
    pd.testing.assert_frame_equal(pd.read_csv(base / "study_design.csv", index_col=0), study_design)
    pd.testing.assert_frame_equal(pd.read_csv(base / "truth_beta.csv", index_col=0), truth["beta"])
    pd.testing.assert_frame_equal(pd.read_csv(base / "truth_eta.csv", index_col=0), truth["site_effect"])
    pd.testing.assert_frame_equal(pd.read_csv(base / "truth_lambda.csv", index_col=0), truth["lambda"])
    pd.testing.assert_frame_equal(
        pd.read_csv(base / "truth_linear_predictor.csv", index_col=0),
        truth["linear_predictor"],
    )


def test_spatial_multifactor_eta_config_compiles_and_validates(tmp_path):
    config = load_model_config(PROJECT / "model_spatial_nngp.yaml")
    assert config["random_levels"]["plot"]["type"] == "spatial_nngp"
    assert config["random_levels"]["plot"]["nf"] == 2
    assert config["random_levels"]["plot"]["n_neighbors"] == 10
    model, loaded = model_from_config(PROJECT / "model_spatial_nngp.yaml")
    compiled = model.compile(tmp_path / "compiled", chains=loaded["chains"])
    results = validate_compiled_native_model(compiled.init_json)
    assert all(result.passed for result in results)


def test_spatial_multifactor_eta_analyzer_smoke(tmp_path):
    posterior = tmp_path / "posterior.h5"
    _write_multifactor_posterior(PROJECT, posterior)
    result = subprocess.run(
        [
            sys.executable,
            "examples/analyze_spatial_multifactor_eta_validation.py",
            "--project",
            str(PROJECT),
            "--posterior",
            str(posterior),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "Simulated Spatial Multi-Factor Eta Validation Report" in result.stdout
    assert "eta_aligned_mean_corr" in result.stdout
    assert "association_truth_corr" in result.stdout


def test_spatial_multifactor_eta_analyzer_metrics(tmp_path):
    posterior = tmp_path / "posterior.h5"
    _write_multifactor_posterior(PROJECT, posterior)
    metrics = build_metrics(PROJECT, posterior)

    assert metrics["n_factors"] == 2
    assert metrics["n_neighbors"] == 10
    assert metrics["beta_sign_recovered"] == "8 / 8"
    assert float(metrics["eta_aligned_mean_corr"]) == pytest.approx(1.0)
    assert float(metrics["lambda_aligned_mean_corr"]) == pytest.approx(1.0)
    assert float(metrics["association_truth_corr"]) > 0.999


def test_multifactor_analyzer_matches_truth_factors_when_sampler_has_extras():
    truth = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    estimated = np.array(
        [
            [0.01, 0.0, 0.01],
            [0.0, -2.0, 0.0],
            [3.0, 0.0, -3.0],
            [0.0, 0.02, 0.0],
        ]
    )

    order, signs = _match_estimated_factors_to_truth(estimated, truth)

    assert order == [2, 1]
    assert signs.tolist() == [1.0, -1.0]


def _write_multifactor_posterior(project, path):
    truth_beta = pd.read_csv(project / "data" / "truth_beta.csv", index_col=0)
    truth_eta = pd.read_csv(project / "data" / "truth_eta.csv", index_col=0)
    truth_lambda = pd.read_csv(project / "data" / "truth_lambda.csv", index_col=0)
    _write_hdf5(
        path,
        beta=truth_beta.to_numpy(dtype=float),
        species=list(truth_beta.columns),
        eta=truth_eta.to_numpy(dtype=float),
        lam=truth_lambda.to_numpy(dtype=float),
    )


def _write_hdf5(path, beta, species, eta, lam):
    h5py = pytest.importorskip("h5py")
    covariates = ["Intercept", "env"]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Beta", data=np.stack([beta, beta + 0.01], axis=0)[None, ...])
        handle.create_dataset("sigma", data=np.ones((1, 2, len(species))) * 0.1)
        level = handle.create_group("random_levels").create_group("0")
        level.create_dataset("Eta", data=np.stack([eta, eta + 0.01], axis=0)[None, ...])
        level.create_dataset("Lambda", data=np.stack([lam, lam + 0.01], axis=0)[None, ...])
        handle.attrs["pyhmsc_metadata"] = (
            '{"names":{"covariates":'
            + repr(covariates).replace("'", '"')
            + ',"species":'
            + repr(species).replace("'", '"')
            + '},"formula":{"X":"~ env"},"distribution":"normal"}'
        )
