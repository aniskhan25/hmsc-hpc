import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from examples.analyze_spatial_random_slope_validation import build_metrics_table
from pyhmsc.config import load_model_config, model_from_config
from pyhmsc.simulate import simulate_spatial_random_slope_effect_data
from pyhmsc.validation import validate_compiled_native_model


PROJECT = Path("examples/projects/simulated_spatial_random_slope_validation")
STRONG_PROJECT = Path("examples/projects/simulated_spatial_random_slope_strong_validation")


def test_spatial_random_slope_project_structure():
    expected = [
        "README.md",
        "model_spatial_full.yaml",
        "model_spatial_gpp.yaml",
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


def test_strong_spatial_random_slope_project_structure():
    expected = [
        "README.md",
        "model_spatial_full.yaml",
        "model_spatial_gpp.yaml",
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
        assert (STRONG_PROJECT / relative).exists(), relative


def test_spatial_random_slope_project_files_match_simulator():
    Y, X, study_design, truth = simulate_spatial_random_slope_effect_data(
        n_sites=49,
        n_species=5,
        seed=41,
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


def test_strong_spatial_random_slope_project_files_match_simulator():
    Y, X, study_design, truth = simulate_spatial_random_slope_effect_data(
        n_sites=81,
        n_species=6,
        spatial_range=0.32,
        spatial_sd=1.4,
        lambda_intercept_scale=1.1,
        lambda_slope_scale=1.8,
        noise_sd=0.05,
        distr="normal",
        seed=91,
    )
    base = STRONG_PROJECT / "data"
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


def test_spatial_random_slope_configs_compile_and_validate(tmp_path):
    configs = _model_configs(PROJECT) + _model_configs(STRONG_PROJECT)
    for config_path in configs:
        model, config = model_from_config(config_path)
        compiled = model.compile(tmp_path / config_path.stem, chains=config["chains"])
        results = validate_compiled_native_model(compiled.init_json)
        assert all(result.passed for result in results), config_path


def test_spatial_random_slope_configs_define_expected_features():
    full = load_model_config(PROJECT / "model_spatial_full.yaml")
    gpp = load_model_config(PROJECT / "model_spatial_gpp.yaml")
    nngp = load_model_config(PROJECT / "model_spatial_nngp.yaml")
    assert full["random_levels"]["plot"]["type"] == "spatial_full"
    assert gpp["random_levels"]["plot"]["type"] == "spatial_gpp"
    assert gpp["random_levels"]["plot"]["n_knots"] == 9
    assert nngp["random_levels"]["plot"]["type"] == "spatial_nngp"
    assert nngp["random_levels"]["plot"]["n_neighbors"] == 8
    for config in [full, gpp, nngp]:
        assert config["random_levels"]["plot"]["x_formula"] == "~ slope_env"


def test_strong_spatial_random_slope_configs_define_expected_features():
    full = load_model_config(STRONG_PROJECT / "model_spatial_full.yaml")
    gpp = load_model_config(STRONG_PROJECT / "model_spatial_gpp.yaml")
    nngp = load_model_config(STRONG_PROJECT / "model_spatial_nngp.yaml")
    assert full["distribution"] == "normal"
    assert full["samples"] == 2000
    assert full["transient"] == 1000
    assert gpp["random_levels"]["plot"]["n_knots"] == 16
    assert nngp["random_levels"]["plot"]["n_neighbors"] == 10
    for config in [full, gpp, nngp]:
        assert config["random_levels"]["plot"]["x_formula"] == "~ slope_env"


def test_spatial_random_slope_analyzer_smoke(tmp_path):
    posteriors = _make_posteriors(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "examples/analyze_spatial_random_slope_validation.py",
            "--project",
            str(PROJECT),
            "--spatial-full-posterior",
            str(posteriors["spatial_full"]),
            "--spatial-gpp-posterior",
            str(posteriors["spatial_gpp"]),
            "--spatial-nngp-posterior",
            str(posteriors["spatial_nngp"]),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "Simulated Spatial Random-Slope Validation Report" in result.stdout
    assert "spatial_full" in result.stdout
    assert "spatial_gpp" in result.stdout
    assert "spatial_nngp" in result.stdout
    assert "lambda_slope_truth_corr" in result.stdout


def test_spatial_random_slope_analyzer_metrics(tmp_path):
    posteriors = _make_posteriors(tmp_path)
    metrics = build_metrics_table(PROJECT, posteriors)

    assert list(metrics["model"]) == ["spatial_full", "spatial_gpp", "spatial_nngp"]
    assert metrics["beta_sign_recovered"].tolist() == ["4 / 4", "4 / 4", "4 / 4"]
    assert metrics["eta_truth_corr"].astype(float).to_numpy() == pytest.approx([1.0, 1.0, 1.0])
    assert metrics["lambda_intercept_truth_corr"].astype(float).to_numpy() == pytest.approx([1.0, 1.0, 1.0])
    assert metrics["lambda_slope_truth_corr"].astype(float).to_numpy() == pytest.approx([1.0, 1.0, 1.0])


def _make_posteriors(tmp_path):
    posteriors = {
        "spatial_full": tmp_path / "spatial_full.h5",
        "spatial_gpp": tmp_path / "spatial_gpp.h5",
        "spatial_nngp": tmp_path / "spatial_nngp.h5",
    }
    for path in posteriors.values():
        _write_spatial_random_slope_posterior(PROJECT, path)
    return posteriors


def _model_configs(project):
    return [
        project / "model_spatial_full.yaml",
        project / "model_spatial_gpp.yaml",
        project / "model_spatial_nngp.yaml",
    ]


def _write_spatial_random_slope_posterior(project, path):
    truth_beta = pd.read_csv(project / "data" / "truth_beta.csv", index_col=0)
    truth_eta = pd.read_csv(project / "data" / "truth_eta.csv", index_col=0)
    truth_lambda = pd.read_csv(project / "data" / "truth_lambda.csv", index_col=0)
    lam = np.stack(
        [
            truth_lambda.loc["Intercept"].to_numpy(dtype=float),
            truth_lambda.loc["slope_env"].to_numpy(dtype=float),
        ],
        axis=-1,
    )[None, ...]
    _write_hdf5(
        path,
        beta=truth_beta.to_numpy(dtype=float),
        species=list(truth_beta.columns),
        eta=truth_eta.to_numpy(dtype=float),
        lam=lam,
    )


def _write_hdf5(path, beta, species, eta, lam):
    h5py = pytest.importorskip("h5py")
    covariates = ["Intercept", "env"]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Beta", data=np.stack([beta, beta + 0.01], axis=0)[None, ...])
        level = handle.create_group("random_levels").create_group("0")
        level.create_dataset("Eta", data=np.stack([eta, eta + 0.01], axis=0)[None, ...])
        level.create_dataset("Lambda", data=np.stack([lam, lam + 0.01], axis=0)[None, ...])
        handle.attrs["pyhmsc_metadata"] = (
            '{"names":{"covariates":'
            + repr(covariates).replace("'", '"')
            + ',"species":'
            + repr(species).replace("'", '"')
            + '},"formula":{"X":"~ env"},"distribution":"probit"}'
        )
