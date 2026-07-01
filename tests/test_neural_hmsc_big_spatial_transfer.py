import ast
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from examples.generate_neural_hmsc_big_spatial_transfer import generate_project
from examples.run_neural_hmsc_big_spatial_transfer import _directory_sha256


SOURCE_MATRIX = Path("examples/big_spatial/data")
SOURCE_PROJECT = Path("examples/projects/big_spatial_plants_validation")
RUNNER = Path("examples/run_neural_hmsc_big_spatial_transfer.py")


def test_big_spatial_transfer_projection_matches_frozen_shape(tmp_path):
    project = tmp_path / "transfer"

    generate_project(SOURCE_MATRIX, SOURCE_PROJECT, project)

    train_Y = pd.read_csv(project / "data/train/Y.csv", index_col=0)
    train_X = pd.read_csv(project / "data/train/X.csv", index_col=0)
    test_Y = pd.read_csv(project / "data/test/Y.csv", index_col=0)
    test_X = pd.read_csv(project / "data/test/X.csv", index_col=0)
    split = pd.read_csv(project / "data/split.csv", index_col=0)
    metadata = json.loads(
        (project / "projection_metadata.json").read_text(encoding="utf-8")
    )

    assert train_Y.shape == (40, 75)
    assert train_X.shape == (40, 1)
    assert test_Y.shape == (360, 75)
    assert test_X.shape == (360, 1)
    assert train_Y.index.equals(train_X.index)
    assert test_Y.index.equals(test_X.index)
    assert train_Y.columns.equals(test_Y.columns)
    assert set(train_Y.index).isdisjoint(test_Y.index)
    assert (train_Y.sum(axis=0) > 0).all()
    assert (train_Y.sum(axis=0) < len(train_Y)).all()
    assert split["split"].value_counts().to_dict() == {"test": 360, "train": 40}
    assert metadata["holdout_used_for_selection"] is False
    assert metadata["source_covariate"] == "Max_temp_smooth"
    source_X = pd.read_csv(SOURCE_PROJECT / "data/X.csv", index_col=0)
    np.testing.assert_allclose(
        train_X["TMG"],
        source_X.loc[train_X.index, "Max_temp_smooth"],
    )


def test_frozen_artifact_directory_fingerprint_is_content_sensitive(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text("one\n", encoding="utf-8")
    first = _directory_sha256(checkpoint)

    (checkpoint / "weights.h5").write_bytes(b"weights")
    second = _directory_sha256(checkpoint)

    assert first != second
    assert second == _directory_sha256(checkpoint)


def test_transfer_runner_has_no_training_or_calibration_fit_path():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "fit" not in called_attributes
    assert "fit_beta_scale_calibration" not in called_names


def test_big_spatial_transfer_lumi_script_syntax():
    subprocess.run(
        ["bash", "-n", "docs/lumi_neural_hmsc_big_spatial_transfer_sbatch.sh"],
        check=True,
    )
