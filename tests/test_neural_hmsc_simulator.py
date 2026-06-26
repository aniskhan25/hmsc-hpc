import json
from pathlib import Path

import numpy as np
import pandas as pd

from pyhmsc.config import model_from_config
from pyhmsc.neural.datasets import load_benchmark_config
from pyhmsc.neural.simulator import generate_fixed_effect_corpus, simulate_fixed_effect_dataset
from pyhmsc.serialization import read_compiled_model


def test_simulate_fixed_effect_dataset_shapes_and_truth():
    dataset = simulate_fixed_effect_dataset(
        n_sites=10,
        n_species=3,
        distribution="normal",
        seed=123,
    )

    assert dataset.Y.shape == (10, 3)
    assert dataset.X.shape == (10, 2)
    assert list(dataset.X.columns) == ["x1", "x2"]
    assert dataset.truth_beta.shape == (3, 3)
    assert list(dataset.truth_beta.index) == ["Intercept", "x1", "x2"]
    assert dataset.linear_predictor.shape == (10, 3)
    assert dataset.metadata["n_covariates"] == 3


def test_generate_fixed_effect_corpus_writes_compiled_artifacts(tmp_path):
    config = load_benchmark_config("examples/projects/neural_hmsc_fixed_gaussian/benchmark.yaml")
    config["simulation"]["corpus_sizes"]["tiny"] = {"train": 2, "validation": 1, "test": 1}

    manifest = generate_fixed_effect_corpus(config, tmp_path / "corpus", profile="tiny", chains=1)

    assert manifest["benchmark"] == "neural_hmsc_fixed_gaussian"
    assert manifest["splits"]["train"]["count"] == 2
    metadata = json.loads((tmp_path / "corpus" / "corpus_metadata.json").read_text(encoding="utf-8"))
    assert metadata["profile"] == "tiny"

    dataset_dir = tmp_path / "corpus" / manifest["splits"]["train"]["datasets"][0]["path"]
    assert (dataset_dir / "data" / "Y.csv").exists()
    assert (dataset_dir / "data" / "X.csv").exists()
    assert (dataset_dir / "data" / "truth_beta.csv").exists()
    assert (dataset_dir / "compiled" / "init.json").exists()

    model, loaded_config = model_from_config(dataset_dir / "model.yaml")
    assert model.distr == "normal"
    assert loaded_config["formula"]["X"] == "~ x1 + x2"

    compiled_metadata, arrays = read_compiled_model(dataset_dir / "compiled" / "init.json")
    truth_beta = pd.read_csv(dataset_dir / "data" / "truth_beta.csv", index_col=0)
    assert compiled_metadata["capabilities"]["fixed_effects"] is True
    assert compiled_metadata["capabilities"]["random_levels"] is False
    assert compiled_metadata["dimensions"]["n_covariates"] == 3
    assert arrays["Beta_init"].shape == (1, 3, truth_beta.shape[1])
    np.testing.assert_allclose(arrays["X"][:, 0], np.ones(arrays["X"].shape[0]))


def test_generate_fixed_effect_corpus_rejects_nonempty_dataset_dir(tmp_path):
    config = load_benchmark_config("examples/projects/neural_hmsc_fixed_gaussian/benchmark.yaml")
    config["simulation"]["corpus_sizes"]["tiny"] = {"train": 1}
    occupied = tmp_path / "corpus" / "train" / "dataset_000000"
    occupied.mkdir(parents=True)
    (occupied / "placeholder.txt").write_text("existing", encoding="utf-8")

    try:
        generate_fixed_effect_corpus(config, tmp_path / "corpus", profile="tiny", chains=1)
    except FileExistsError as exc:
        assert "dataset_000000" in str(exc)
    else:
        raise AssertionError("Expected FileExistsError for non-empty dataset directory")


def test_generate_fixed_effect_corpus_is_reproducible(tmp_path):
    left_config = load_benchmark_config("examples/projects/neural_hmsc_fixed_gaussian/benchmark.yaml")
    right_config = load_benchmark_config("examples/projects/neural_hmsc_fixed_gaussian/benchmark.yaml")
    left_config["simulation"]["corpus_sizes"]["tiny"] = {"train": 1}
    right_config["simulation"]["corpus_sizes"]["tiny"] = {"train": 1}

    left = generate_fixed_effect_corpus(left_config, tmp_path / "left", profile="tiny", chains=1)
    right = generate_fixed_effect_corpus(right_config, tmp_path / "right", profile="tiny", chains=1)

    left_dir = tmp_path / "left" / left["splits"]["train"]["datasets"][0]["path"]
    right_dir = tmp_path / "right" / right["splits"]["train"]["datasets"][0]["path"]
    pd.testing.assert_frame_equal(
        pd.read_csv(left_dir / "data" / "Y.csv", index_col=0),
        pd.read_csv(right_dir / "data" / "Y.csv", index_col=0),
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(left_dir / "data" / "truth_beta.csv", index_col=0),
        pd.read_csv(right_dir / "data" / "truth_beta.csv", index_col=0),
    )


def test_benchmark_configs_generate_tiny_corpora(tmp_path):
    for name in ["gaussian", "probit", "poisson"]:
        config_path = Path(f"examples/projects/neural_hmsc_fixed_{name}/benchmark.yaml")
        config = load_benchmark_config(config_path)
        config["simulation"]["corpus_sizes"]["tiny"] = {"train": 1}

        manifest = generate_fixed_effect_corpus(
            config,
            tmp_path / name,
            profile="tiny",
            chains=1,
        )

        dataset_dir = tmp_path / name / manifest["splits"]["train"]["datasets"][0]["path"]
        compiled_metadata, arrays = read_compiled_model(dataset_dir / "compiled" / "init.json")
        assert compiled_metadata["distribution"] in {"normal", "probit", "poisson"}
        assert arrays["Y"].shape[1] in {2, 4, 8}


def test_benchmark_configs_define_calibration_split():
    for name in ["gaussian", "probit", "poisson"]:
        config_path = Path(f"examples/projects/neural_hmsc_fixed_{name}/benchmark.yaml")
        config = load_benchmark_config(config_path)

        assert config["simulation"]["corpus_sizes"]["smoke"]["calibration"] > 0
        assert config["simulation"]["corpus_sizes"]["default"]["calibration"] > 0
