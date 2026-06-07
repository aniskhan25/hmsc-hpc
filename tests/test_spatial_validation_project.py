from pathlib import Path

import pandas as pd

from pyhmsc.config import model_from_config
from pyhmsc.config import load_model_config
from pyhmsc.simulate import simulate_spatial_effect_data
from pyhmsc.validation import validate_compiled_native_model


PROJECT = Path("examples/projects/simulated_spatial_validation")


def test_simulated_spatial_validation_project_structure():
    expected_files = {
        "README.md",
        "model_fixed.yaml",
        "model_iid.yaml",
        "model_spatial_full.yaml",
        "data/Y.csv",
        "data/X.csv",
        "data/study_design.csv",
        "data/truth_beta.csv",
        "data/truth_site_effect.csv",
        "data/truth_lambda.csv",
    }

    assert {str(path.relative_to(PROJECT)) for path in PROJECT.rglob("*") if path.is_file()} == expected_files

    Y = pd.read_csv(PROJECT / "data" / "Y.csv", index_col=0)
    X = pd.read_csv(PROJECT / "data" / "X.csv", index_col=0)
    study_design = pd.read_csv(PROJECT / "data" / "study_design.csv", index_col=0)
    truth_beta = pd.read_csv(PROJECT / "data" / "truth_beta.csv", index_col=0)
    truth_site_effect = pd.read_csv(PROJECT / "data" / "truth_site_effect.csv", index_col=0)
    truth_lambda = pd.read_csv(PROJECT / "data" / "truth_lambda.csv", index_col=0)

    assert Y.shape == (36, 5)
    assert X.shape == (36, 1)
    assert study_design.shape == (36, 3)
    assert truth_beta.shape == (2, 5)
    assert truth_site_effect.shape == (36, 1)
    assert truth_lambda.shape == (1, 5)

    assert list(Y.index) == list(X.index) == list(study_design.index) == list(truth_site_effect.index)
    assert list(truth_beta.columns) == list(Y.columns)
    assert list(truth_lambda.columns) == list(Y.columns)
    assert list(X.columns) == ["env"]
    assert list(study_design.columns) == ["plot", "xcoord", "ycoord"]
    assert list(truth_beta.index) == ["Intercept", "env"]
    assert list(truth_site_effect.columns) == ["eta"]
    assert list(truth_lambda.index) == ["factor_0"]


def test_simulated_spatial_validation_model_configs_define_expected_models():
    fixed = load_model_config(PROJECT / "model_fixed.yaml")
    iid = load_model_config(PROJECT / "model_iid.yaml")
    spatial = load_model_config(PROJECT / "model_spatial_full.yaml")

    for config in [fixed, iid, spatial]:
        assert config["response"] == "data/Y.csv"
        assert config["covariates"] == "data/X.csv"
        assert config["formula"] == {"X": "~ env"}
        assert config["distribution"] == "probit"
        assert config["chains"] == 2
        assert config["samples"] == 1000
        assert config["transient"] == 500
        assert config["thin"] == 10

    assert "random_levels" not in fixed

    assert iid["study_design"] == "data/study_design.csv"
    assert iid["random_levels"] == {
        "plot": {
            "column": "plot",
            "type": "iid",
            "nf": 1,
        }
    }

    assert spatial["study_design"] == "data/study_design.csv"
    assert spatial["random_levels"] == {
        "plot": {
            "column": "plot",
            "type": "spatial_full",
            "coords": ["xcoord", "ycoord"],
            "nf": 1,
        }
    }


def test_simulated_spatial_validation_files_match_simulator():
    Y, X, study_design, truth = simulate_spatial_effect_data(n_sites=36, n_species=5, seed=21)

    pd.testing.assert_frame_equal(pd.read_csv(PROJECT / "data" / "Y.csv", index_col=0), Y)
    pd.testing.assert_frame_equal(pd.read_csv(PROJECT / "data" / "X.csv", index_col=0), X)
    pd.testing.assert_frame_equal(
        pd.read_csv(PROJECT / "data" / "study_design.csv", index_col=0),
        study_design,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(PROJECT / "data" / "truth_beta.csv", index_col=0),
        truth["beta"],
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(PROJECT / "data" / "truth_site_effect.csv", index_col=0),
        truth["site_effect"],
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(PROJECT / "data" / "truth_lambda.csv", index_col=0),
        truth["lambda"],
    )


def test_simulated_spatial_validation_configs_compile_and_validate(tmp_path):
    for config_name in ["model_fixed.yaml", "model_iid.yaml", "model_spatial_full.yaml"]:
        model, config = model_from_config(PROJECT / config_name)
        compiled = model.compile(tmp_path / config_name.removesuffix(".yaml"), chains=config["chains"])
        results = validate_compiled_native_model(compiled.init_json)

        assert all(result.passed for result in results), config_name
