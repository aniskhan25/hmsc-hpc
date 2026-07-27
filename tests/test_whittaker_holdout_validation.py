import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from examples.analyze_whittaker_holdout_validation import build_metrics_table
from examples.generate_whittaker_holdout_validation import generate_project
from pyhmsc.config import model_from_config
from pyhmsc.validation import validate_compiled_native_model


SOURCE = Path("examples/projects/whittaker_plants_hmsc_book")


def test_whittaker_holdout_generator_preserves_species_and_tmg_range(tmp_path):
    project = tmp_path / "project"
    generate_project(SOURCE, project)

    train_Y = pd.read_csv(project / "data/train/Y.csv", index_col=0)
    train_X = pd.read_csv(project / "data/train/X.csv", index_col=0)
    train_study = pd.read_csv(project / "data/train/study_design.csv", index_col=0)
    test_Y = pd.read_csv(project / "data/test/Y.csv", index_col=0)
    test_X = pd.read_csv(project / "data/test/X.csv", index_col=0)
    test_study = pd.read_csv(project / "data/test/study_design.csv", index_col=0)
    traits = pd.read_csv(project / "data/traits.csv", index_col=0)
    phylo = pd.read_csv(project / "data/phylo_cov.csv", index_col=0)

    assert train_Y.shape == (40, 75)
    assert test_Y.shape == (12, 75)
    assert train_Y.index.equals(train_X.index) and train_Y.index.equals(train_study.index)
    assert test_Y.index.equals(test_X.index) and test_Y.index.equals(test_study.index)
    assert (train_Y.sum(axis=0) > 0).all()
    assert train_Y.columns.equals(traits.index)
    assert train_Y.columns.equals(phylo.index) and phylo.index.equals(phylo.columns)
    assert test_X["TMG"].min() == pd.read_csv(SOURCE / "data/X.csv", index_col=0)["TMG"].min()
    assert test_X["TMG"].max() == pd.read_csv(SOURCE / "data/X.csv", index_col=0)["TMG"].max()


def test_whittaker_holdout_models_compile(tmp_path):
    project = tmp_path / "project"
    generate_project(SOURCE, project)
    for name in ["fixed", "iid"]:
        model, config = model_from_config(project / f"model_{name}.yaml")
        compiled = model.compile(tmp_path / f"compiled_{name}", chains=config["chains"])
        assert all(result.passed for result in validate_compiled_native_model(compiled.init_json))


def test_whittaker_holdout_analyzer_smoke(tmp_path):
    project = tmp_path / "project"
    run_root = tmp_path / "run"
    generate_project(SOURCE, project)
    species = pd.read_csv(project / "data/test/Y.csv", index_col=0).columns
    for name in ["fixed", "iid"]:
        model, config = model_from_config(project / f"model_{name}.yaml")
        compiled = model.compile(tmp_path / f"metadata_{name}", chains=config["chains"])
        metadata = json.loads(compiled.init_json.read_text(encoding="utf-8"))
        model_root = run_root / name
        model_root.mkdir(parents=True)
        with h5py.File(model_root / "posterior.h5", "w") as handle:
            handle.create_dataset("Beta", data=np.zeros((2, 4, 2, len(species))))
            if name == "fixed":
                handle.create_dataset(
                    "Gamma",
                    data=np.zeros((2, 4, 2, metadata["dimensions"]["n_traits"])),
                )
            if name == "iid":
                level = handle.create_group("random_levels").create_group("0")
                level.create_dataset("Eta", data=np.zeros((2, 4, 40, 1)))
                level.create_dataset("Lambda", data=np.zeros((2, 4, 1, len(species))))
            handle.attrs["pyhmsc_metadata"] = json.dumps(metadata)
        (model_root / "resource_metrics.txt").write_text(
            "elapsed_seconds=12.5\nmax_rss_kb=2048\nsamples=4\ntransient=2\nthin=1\n",
            encoding="utf-8",
        )

    metrics = build_metrics_table(project, run_root)

    assert metrics["model"].tolist() == ["fixed", "iid"]
    assert metrics["random_effects"].tolist() == ["none", "marginal"]
    np.testing.assert_allclose(metrics["brier_score"], 0.25)
    np.testing.assert_allclose(metrics["macro_auc"], 0.5)
    assert metrics["auc_species"].tolist() == [32, 32]
    assert (metrics["observed_richness_slope"] < 0).all()
    assert (metrics["observed_weighted_cn_slope"] > 0).all()
    assert metrics.loc[metrics["model"] == "fixed", "gamma_tmg_cn_mean"].iloc[0] == 0.0
    assert np.isnan(metrics.loc[metrics["model"] == "iid", "gamma_tmg_cn_mean"].iloc[0])
    np.testing.assert_allclose(metrics["samples"], 4)


def test_whittaker_holdout_lumi_script_syntax():
    subprocess.run(["bash", "-n", "docs/lumi_whittaker_holdout_validation_sbatch.sh"], check=True)
