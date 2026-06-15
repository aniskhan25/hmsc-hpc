import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from examples.analyze_spatial_eta_validation import build_metrics_table
from pyhmsc.config import load_model_config, model_from_config
from pyhmsc.simulate import simulate_spatial_eta_effect_data
from pyhmsc.validation import validate_compiled_native_model


PROJECT = Path("examples/projects/simulated_spatial_eta_validation")
MODEL_NAMES = ["spatial_full", "spatial_gpp", "spatial_nngp_5", "spatial_nngp_10", "spatial_nngp_20"]


def test_spatial_eta_project_structure():
    expected = [
        "README.md",
        "model_spatial_full.yaml",
        "model_spatial_gpp.yaml",
        "model_spatial_nngp_5.yaml",
        "model_spatial_nngp_10.yaml",
        "model_spatial_nngp_20.yaml",
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


def test_spatial_eta_project_files_match_simulator():
    Y, X, study_design, truth = simulate_spatial_eta_effect_data(
        n_sites=100,
        n_species=6,
        spatial_range=0.24,
        spatial_sd=1.6,
        lambda_scale=1.2,
        noise_sd=0.06,
        distr="normal",
        seed=121,
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


def test_spatial_eta_configs_compile_and_validate(tmp_path):
    for config_path in _model_configs():
        model, config = model_from_config(config_path)
        compiled = model.compile(tmp_path / config_path.stem, chains=config["chains"])
        results = validate_compiled_native_model(compiled.init_json)
        assert all(result.passed for result in results), config_path


def test_spatial_eta_configs_define_expected_features():
    full = load_model_config(PROJECT / "model_spatial_full.yaml")
    gpp = load_model_config(PROJECT / "model_spatial_gpp.yaml")
    nngp_5 = load_model_config(PROJECT / "model_spatial_nngp_5.yaml")
    nngp_10 = load_model_config(PROJECT / "model_spatial_nngp_10.yaml")
    nngp_20 = load_model_config(PROJECT / "model_spatial_nngp_20.yaml")

    assert full["distribution"] == "normal"
    assert full["samples"] == 1500
    assert full["transient"] == 750
    assert full["random_levels"]["plot"]["type"] == "spatial_full"
    assert gpp["random_levels"]["plot"]["type"] == "spatial_gpp"
    assert gpp["random_levels"]["plot"]["n_knots"] == 25
    assert nngp_5["random_levels"]["plot"]["n_neighbors"] == 5
    assert nngp_10["random_levels"]["plot"]["n_neighbors"] == 10
    assert nngp_20["random_levels"]["plot"]["n_neighbors"] == 20


def test_spatial_eta_analyzer_smoke(tmp_path):
    posteriors = _make_posteriors(tmp_path)
    args = [
        sys.executable,
        "examples/analyze_spatial_eta_validation.py",
        "--project",
        str(PROJECT),
    ]
    for name, path in posteriors.items():
        args.extend(["--posterior", f"{name}={path}"])
    result = subprocess.run(args, check=True, text=True, capture_output=True)

    assert "Simulated Spatial Eta Validation Report" in result.stdout
    assert "spatial_full" in result.stdout
    assert "spatial_nngp_20" in result.stdout
    assert "eta_truth_corr" in result.stdout
    assert "eta_rmse_scaled" in result.stdout


def test_spatial_eta_analyzer_metrics(tmp_path):
    posteriors = _make_posteriors(tmp_path)
    metrics = build_metrics_table(PROJECT, posteriors)

    assert list(metrics["model"]) == MODEL_NAMES
    assert metrics["beta_sign_recovered"].tolist() == ["6 / 6"] * len(MODEL_NAMES)
    assert metrics["eta_truth_corr"].astype(float).to_numpy() == pytest.approx([1.0] * len(MODEL_NAMES))
    assert metrics["lambda_truth_corr"].astype(float).to_numpy() == pytest.approx([1.0] * len(MODEL_NAMES))
    assert metrics.loc[metrics["model"] == "spatial_nngp_5", "n_neighbors"].iloc[0] == 5
    assert metrics.loc[metrics["model"] == "spatial_nngp_20", "n_neighbors"].iloc[0] == 20


def _model_configs():
    return [PROJECT / f"model_{name}.yaml" for name in MODEL_NAMES]


def _make_posteriors(tmp_path):
    posteriors = {name: tmp_path / f"{name}.h5" for name in MODEL_NAMES}
    for path in posteriors.values():
        _write_spatial_eta_posterior(PROJECT, path)
    return posteriors


def _write_spatial_eta_posterior(project, path):
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
