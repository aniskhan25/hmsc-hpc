import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from examples.analyze_big_spatial_holdout_validation import build_metrics_table
from examples.generate_big_spatial_holdout_validation import generate_project
from pyhmsc.config import model_from_config
from pyhmsc.validation import validate_compiled_native_model


SOURCE = Path("examples/projects/big_spatial_plants_validation")


def test_big_spatial_holdout_generator_builds_blocked_aligned_project(tmp_path):
    project = tmp_path / "project"
    generate_project(SOURCE, project)

    train_Y = pd.read_csv(project / "data/train/Y.csv", index_col=0)
    train_X = pd.read_csv(project / "data/train/X.csv", index_col=0)
    train_design = pd.read_csv(project / "data/train/study_design.csv", index_col=0)
    test_Y = pd.read_csv(project / "data/test/Y.csv", index_col=0)
    test_X = pd.read_csv(project / "data/test/X.csv", index_col=0)
    test_design = pd.read_csv(project / "data/test/study_design.csv", index_col=0)
    test_coords = pd.read_csv(project / "data/test/coords.csv", index_col=0)
    split = pd.read_csv(project / "data/split.csv", index_col=0)

    assert train_Y.shape == (319, 40)
    assert test_Y.shape == (81, 40)
    assert train_Y.index.equals(train_X.index) and train_Y.index.equals(train_design.index)
    assert test_Y.index.equals(test_X.index) and test_Y.index.equals(test_design.index)
    assert test_Y.index.equals(test_coords.index)
    assert set(train_Y.index).isdisjoint(test_Y.index)
    assert (train_Y.sum() > 0).all() and (train_Y.sum() < len(train_Y)).all()
    assert split["split"].value_counts().to_dict() == {"train": 319, "test": 81}
    assert (split.loc[test_Y.index, "split"] == "test").all()


def test_big_spatial_holdout_configs_compile(tmp_path):
    project = tmp_path / "project"
    generate_project(SOURCE, project)
    for name in ["fixed", "spatial_full", "spatial_gpp", "spatial_nngp"]:
        model, config = model_from_config(project / f"model_{name}.yaml")
        compiled = model.compile(tmp_path / f"compiled_{name}", chains=config["chains"])
        assert all(result.passed for result in validate_compiled_native_model(compiled.init_json))


def test_big_spatial_holdout_analyzer_smoke(tmp_path):
    project = tmp_path / "project"
    run_root = tmp_path / "run"
    generate_project(SOURCE, project)
    species = pd.read_csv(project / "data/test/Y.csv", index_col=0).columns
    for name in ["fixed", "spatial_full", "spatial_gpp", "spatial_nngp"]:
        model, config = model_from_config(project / f"model_{name}.yaml")
        compiled = model.compile(tmp_path / f"metadata_{name}", chains=config["chains"])
        metadata = json.loads(compiled.init_json.read_text(encoding="utf-8"))
        model_root = run_root / name
        model_root.mkdir(parents=True)
        with h5py.File(model_root / "posterior.h5", "w") as handle:
            handle.create_dataset("Beta", data=np.zeros((1, 2, 5, len(species))))
            if name != "fixed":
                level = handle.create_group("random_levels").create_group("0")
                level.create_dataset("Eta", data=np.zeros((1, 2, 319, 1)))
                level.create_dataset("Lambda", data=np.zeros((1, 2, 1, len(species))))
                level.create_dataset("Alpha", data=np.ones((1, 2, 1), dtype=int))
            handle.attrs["pyhmsc_metadata"] = json.dumps(metadata)
        (model_root / "resource_metrics.txt").write_text(
            "elapsed_seconds=12.5\nmax_rss_kb=2048\ncompiled_bytes=100\nposterior_bytes=200\n"
            "samples=250\ntransient=250\nthin=5\n",
            encoding="utf-8",
        )

    metrics = build_metrics_table(project, run_root)

    assert metrics["model"].tolist() == ["fixed", "spatial_full", "spatial_gpp", "spatial_nngp"]
    np.testing.assert_allclose(metrics["brier_score"], 0.25)
    np.testing.assert_allclose(metrics["macro_auc"], 0.5)
    assert metrics["auc_species"].tolist() == [38, 38, 38, 38]
    np.testing.assert_allclose(metrics["elapsed_seconds"], 12.5)
    np.testing.assert_allclose(metrics["max_rss_kb"], 2048)
    np.testing.assert_allclose(metrics["samples"], 250)
    np.testing.assert_allclose(metrics["transient"], 250)
    np.testing.assert_allclose(metrics["thin"], 5)


def test_big_spatial_holdout_lumi_script_syntax():
    scripts = [
        "docs/lumi_big_spatial_holdout_validation_sbatch.sh",
        "docs/lumi_big_spatial_holdout_nngp_array_sbatch.sh",
        "docs/lumi_big_spatial_holdout_finalize_sbatch.sh",
    ]
    for script in scripts:
        subprocess.run(["bash", "-n", script], check=True)
