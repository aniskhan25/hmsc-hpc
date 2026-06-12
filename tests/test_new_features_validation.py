import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from examples.analyze_new_features_validation import build_random_slope_metrics, build_spatial_gpp_metrics
from pyhmsc.config import load_model_config, model_from_config
from pyhmsc.simulate import simulate_random_slope_effect_data
from pyhmsc.validation import validate_compiled_native_model


PROJECT = Path("examples/projects/simulated_new_features_validation")


def test_simulated_new_features_project_structure():
    expected = [
        "README.md",
        "random_slope/model_fixed.yaml",
        "random_slope/model_random_slope.yaml",
        "random_slope/data/Y.csv",
        "random_slope/data/X.csv",
        "random_slope/data/study_design.csv",
        "random_slope/data/truth_lambda.csv",
        "spatial_gpp/model_spatial_full.yaml",
        "spatial_gpp/model_spatial_gpp.yaml",
        "spatial_gpp/data/Y.csv",
        "spatial_gpp/data/X.csv",
        "spatial_gpp/data/study_design.csv",
        "spatial_gpp/data/truth_lambda.csv",
    ]
    for relative in expected:
        assert (PROJECT / relative).exists(), relative


def test_random_slope_project_files_match_simulator():
    Y, X, study_design, truth = simulate_random_slope_effect_data(
        n_groups=12,
        sites_per_group=4,
        n_species=5,
        seed=31,
    )
    base = PROJECT / "random_slope" / "data"
    pd.testing.assert_frame_equal(pd.read_csv(base / "Y.csv", index_col=0), Y)
    pd.testing.assert_frame_equal(pd.read_csv(base / "X.csv", index_col=0), X)
    pd.testing.assert_frame_equal(pd.read_csv(base / "study_design.csv", index_col=0), study_design)
    pd.testing.assert_frame_equal(pd.read_csv(base / "truth_beta.csv", index_col=0), truth["beta"])
    pd.testing.assert_frame_equal(pd.read_csv(base / "truth_eta.csv", index_col=0), truth["site_effect"])
    pd.testing.assert_frame_equal(pd.read_csv(base / "truth_lambda.csv", index_col=0), truth["lambda"])


def test_new_feature_model_configs_compile_and_validate(tmp_path):
    configs = [
        PROJECT / "random_slope" / "model_fixed.yaml",
        PROJECT / "random_slope" / "model_random_slope.yaml",
        PROJECT / "spatial_gpp" / "model_spatial_full.yaml",
        PROJECT / "spatial_gpp" / "model_spatial_gpp.yaml",
    ]
    for config_path in configs:
        model, config = model_from_config(config_path)
        compiled = model.compile(tmp_path / config_path.parent.name / config_path.stem, chains=config["chains"])
        results = validate_compiled_native_model(compiled.init_json)
        assert all(result.passed for result in results), config_path


def test_new_feature_configs_define_expected_features():
    random_slope = load_model_config(PROJECT / "random_slope" / "model_random_slope.yaml")
    spatial_gpp = load_model_config(PROJECT / "spatial_gpp" / "model_spatial_gpp.yaml")
    assert random_slope["random_levels"]["plot"]["x_formula"] == "~ slope_env"
    assert random_slope["random_levels"]["plot"]["type"] == "iid"
    assert spatial_gpp["random_levels"]["plot"]["type"] == "spatial_gpp"
    assert spatial_gpp["random_levels"]["plot"]["n_knots"] == 9


def test_new_features_analyzer_smoke(tmp_path):
    posteriors = _make_posteriors(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "examples/analyze_new_features_validation.py",
            "--project",
            str(PROJECT),
            "--random-fixed-posterior",
            str(posteriors["random_fixed"]),
            "--random-slope-posterior",
            str(posteriors["random_slope"]),
            "--spatial-full-posterior",
            str(posteriors["spatial_full"]),
            "--spatial-gpp-posterior",
            str(posteriors["spatial_gpp"]),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "Simulated New-Feature Validation Report" in result.stdout
    assert "Random Slope" in result.stdout
    assert "Spatial GPP" in result.stdout
    assert "lambda_slope_truth_corr" in result.stdout


def test_new_features_analyzer_metrics(tmp_path):
    posteriors = _make_posteriors(tmp_path)
    random_metrics = build_random_slope_metrics(PROJECT / "random_slope", posteriors)
    spatial_metrics = build_spatial_gpp_metrics(PROJECT / "spatial_gpp", posteriors)

    assert list(random_metrics["model"]) == ["fixed", "random_slope"]
    assert list(random_metrics["random_effects"]) == ["none", "known"]
    assert random_metrics["beta_sign_recovered"].tolist() == ["4 / 4", "4 / 4"]
    assert random_metrics.loc[0, "eta_truth_corr"] == "n/a"
    assert random_metrics.loc[1, "eta_truth_corr"] == pytest.approx(1.0)
    assert random_metrics.loc[1, "lambda_slope_truth_corr"] == pytest.approx(1.0)

    assert list(spatial_metrics["model"]) == ["spatial_full", "spatial_gpp"]
    assert spatial_metrics["beta_sign_recovered"].tolist() == ["4 / 4", "4 / 4"]
    assert spatial_metrics["eta_truth_corr"].astype(float).to_numpy() == pytest.approx([1.0, 1.0])
    assert spatial_metrics["lambda_truth_corr"].astype(float).to_numpy() == pytest.approx([1.0, 1.0])


def _make_posteriors(tmp_path):
    posteriors = {
        "random_fixed": tmp_path / "random_fixed.h5",
        "random_slope": tmp_path / "random_slope.h5",
        "spatial_full": tmp_path / "spatial_full.h5",
        "spatial_gpp": tmp_path / "spatial_gpp.h5",
    }
    _write_fixed_posterior(PROJECT / "random_slope", posteriors["random_fixed"])
    _write_random_slope_posterior(PROJECT / "random_slope", posteriors["random_slope"])
    _write_spatial_posterior(PROJECT / "spatial_gpp", posteriors["spatial_full"])
    _write_spatial_posterior(PROJECT / "spatial_gpp", posteriors["spatial_gpp"])
    return posteriors


def _write_fixed_posterior(project, path):
    truth_beta = pd.read_csv(project / "data" / "truth_beta.csv", index_col=0)
    _write_hdf5(path, beta=truth_beta.to_numpy(dtype=float), species=list(truth_beta.columns))


def _write_random_slope_posterior(project, path):
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


def _write_spatial_posterior(project, path):
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


def _write_hdf5(path, beta, species, eta=None, lam=None):
    h5py = pytest.importorskip("h5py")
    covariates = ["Intercept", "env"]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Beta", data=np.stack([beta, beta + 0.01], axis=0)[None, ...])
        if eta is not None and lam is not None:
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

