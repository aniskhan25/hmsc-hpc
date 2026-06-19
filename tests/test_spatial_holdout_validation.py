import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from examples.analyze_spatial_holdout_validation import _predict_interval_compat, build_metrics_table
from pyhmsc.config import load_model_config, model_from_config
from pyhmsc.simulate import simulate_spatial_holdout_data
from pyhmsc.validation import validate_compiled_native_model


PROJECT = Path("examples/projects/simulated_spatial_holdout_validation")
MODEL_NAMES = ["fixed", "spatial_full", "spatial_gpp", "spatial_nngp"]


def test_spatial_holdout_simulator_is_deterministic_and_disjoint():
    first = simulate_spatial_holdout_data(seed=321)
    second = simulate_spatial_holdout_data(seed=321)

    for key in first:
        pd.testing.assert_frame_equal(first[key], second[key])
    assert len(first["train_Y"]) == 80
    assert len(first["test_Y"]) == 20
    assert set(first["train_Y"].index).isdisjoint(first["test_Y"].index)
    train_coords = first["train_study_design"][["xcoord", "ycoord"]]
    test_coords = first["test_coords"]
    assert not any(tuple(row) in set(map(tuple, train_coords.to_numpy())) for row in test_coords.to_numpy())


def test_spatial_holdout_project_matches_generator():
    generated = simulate_spatial_holdout_data(seed=321)
    paths = {
        "train_Y": "data/train/Y.csv",
        "train_X": "data/train/X.csv",
        "train_study_design": "data/train/study_design.csv",
        "test_Y": "data/test/Y.csv",
        "test_X": "data/test/X.csv",
        "test_study_design": "data/test/study_design.csv",
        "test_coords": "data/test/coords.csv",
        "truth_linear_predictor": "data/test/truth_linear_predictor.csv",
        "truth_beta": "data/truth_beta.csv",
        "truth_lambda": "data/truth_lambda.csv",
        "split": "data/split.csv",
    }
    for key, relative in paths.items():
        actual = pd.read_csv(PROJECT / relative, index_col=0)
        pd.testing.assert_frame_equal(actual, generated[key])


def test_spatial_holdout_configs_compile_and_validate(tmp_path):
    for name in MODEL_NAMES:
        config_path = PROJECT / f"model_{name}.yaml"
        model, config = model_from_config(config_path)
        compiled = model.compile(tmp_path / name, chains=config["chains"])
        results = validate_compiled_native_model(compiled.init_json)
        assert all(result.passed for result in results), name


def test_spatial_holdout_configs_define_prediction_models():
    fixed = load_model_config(PROJECT / "model_fixed.yaml")
    full = load_model_config(PROJECT / "model_spatial_full.yaml")
    gpp = load_model_config(PROJECT / "model_spatial_gpp.yaml")
    nngp = load_model_config(PROJECT / "model_spatial_nngp.yaml")

    assert "random_levels" not in fixed
    assert full["random_levels"]["plot"]["type"] == "spatial_full"
    assert gpp["random_levels"]["plot"]["type"] == "spatial_gpp"
    assert gpp["random_levels"]["plot"]["n_knots"] == 20
    assert nngp["random_levels"]["plot"]["type"] == "spatial_nngp"
    assert nngp["random_levels"]["plot"]["n_neighbors"] == 15


def test_spatial_holdout_analyzer_metrics_and_report(tmp_path):
    truth = pd.read_csv(PROJECT / "data/test/truth_linear_predictor.csv", index_col=0)
    offsets = {"fixed": 0.5, "spatial_full": 0.1, "spatial_gpp": 0.2, "spatial_nngp": 0.3}
    predictions = {}
    for name, offset in offsets.items():
        path = tmp_path / f"{name}.csv"
        (truth + offset).to_csv(path)
        predictions[name] = path

    metrics = build_metrics_table(PROJECT, predictions)

    assert list(metrics["model"]) == MODEL_NAMES
    np.testing.assert_allclose(metrics["rmse"], [0.5, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(metrics["mae"], [0.5, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(metrics["rmse_improvement_vs_fixed"], [0.0, 0.4, 0.3, 0.2])

    output = tmp_path / "report.txt"
    args = [
        sys.executable,
        "examples/analyze_spatial_holdout_validation.py",
        "--project",
        str(PROJECT),
        "--output",
        str(output),
    ]
    for name, path in predictions.items():
        args.extend(["--prediction", f"{name}={path}"])
    result = subprocess.run(args, check=True, text=True, capture_output=True)

    assert "Simulated Spatial Hold-Out Prediction Validation Report" in result.stdout
    assert "spatial_full" in result.stdout
    assert "Lowest held-out RMSE: spatial_full" in result.stdout
    assert output.exists()
    assert output.with_suffix(".csv").exists()


def test_interval_compat_merges_prediction_metadata_for_legacy_signature():
    class LegacyFit:
        def __init__(self):
            self.columns = []

        def predict_ci(self, X, level, response, random_effects, unseen_groups):
            self.columns = list(X.columns)
            values = pd.DataFrame(0.0, index=X.index, columns=["sp1"])
            return {"lower": values, "upper": values}

    fit = LegacyFit()
    X = pd.DataFrame({"env": [0.0]}, index=["site_1"])
    study = pd.DataFrame({"plot": ["new"]}, index=["site_1"])
    coords = pd.DataFrame({"xcoord": [0.5], "ycoord": [0.5]}, index=["site_1"])

    _predict_interval_compat(
        fit,
        X,
        level=0.95,
        study_design=study,
        coords=coords,
        random_effects="known",
        unseen_groups="nearest",
    )

    assert fit.columns == ["env", "plot", "xcoord", "ycoord"]
