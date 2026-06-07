from pathlib import Path

import pandas as pd

from pyhmsc.config import load_model_config
from pyhmsc.config import model_from_config
from pyhmsc.validation import validate_compiled_native_model


PROJECT = Path("examples/projects/big_spatial_plants_validation")


def test_big_spatial_project_structure():
    expected_files = {
        "README.md",
        "model_fixed.yaml",
        "model_iid.yaml",
        "model_spatial_full.yaml",
        "data/Y_presence.csv",
        "data/X.csv",
        "data/study_design.csv",
        "data/taxonomy.csv",
    }

    assert {str(path.relative_to(PROJECT)) for path in PROJECT.rglob("*") if path.is_file()} == expected_files

    Y = pd.read_csv(PROJECT / "data" / "Y_presence.csv", index_col=0)
    X = pd.read_csv(PROJECT / "data" / "X.csv", index_col=0)
    study_design = pd.read_csv(PROJECT / "data" / "study_design.csv", index_col=0)
    taxonomy = pd.read_csv(PROJECT / "data" / "taxonomy.csv", index_col=0)

    assert Y.shape == (400, 40)
    assert X.shape == (400, 4)
    assert study_design.shape == (400, 3)
    assert taxonomy.shape[0] == 40
    assert list(Y.index) == list(X.index) == list(study_design.index)
    assert list(taxonomy.index) == list(Y.columns)
    assert list(study_design.columns) == ["site", "xcoord", "ycoord"]
    assert Y.mean(axis=0).min() > 0
    assert Y.mean(axis=0).max() < 1


def test_big_spatial_model_configs_define_expected_models():
    fixed = load_model_config(PROJECT / "model_fixed.yaml")
    iid = load_model_config(PROJECT / "model_iid.yaml")
    spatial = load_model_config(PROJECT / "model_spatial_full.yaml")

    for config in [fixed, iid, spatial]:
        assert config["response"] == "data/Y_presence.csv"
        assert config["covariates"] == "data/X.csv"
        assert config["formula"] == {
            "X": "~ Hillshading270_40 + HA_All_rivers_normalised + Thorium_mosaic_GWR2 + Max_temp_smooth"
        }
        assert config["distribution"] == "probit"
        assert config["chains"] == 2
        assert config["samples"] == 1000
        assert config["transient"] == 500
        assert config["thin"] == 10

    assert "random_levels" not in fixed
    assert iid["random_levels"] == {
        "site": {
            "column": "site",
            "type": "iid",
            "nf": 2,
        }
    }
    assert spatial["random_levels"] == {
        "site": {
            "column": "site",
            "type": "spatial_full",
            "coords": ["xcoord", "ycoord"],
            "nf": 2,
        }
    }


def test_big_spatial_configs_compile_and_validate(tmp_path):
    for config_name in ["model_fixed.yaml", "model_iid.yaml", "model_spatial_full.yaml"]:
        model, config = model_from_config(PROJECT / config_name)
        compiled = model.compile(tmp_path / config_name.removesuffix(".yaml"), chains=config["chains"])
        results = validate_compiled_native_model(compiled.init_json)

        assert all(result.passed for result in results), config_name
