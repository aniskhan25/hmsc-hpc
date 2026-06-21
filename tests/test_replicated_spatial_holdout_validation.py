import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from examples.analyze_replicated_spatial_holdout_validation import (
    _aggregate_metrics,
    _nngp_ordering_deltas,
)
from examples.generate_replicated_spatial_holdout_validation import generate_projects
from pyhmsc.simulate import apply_spatial_holdout_group_order, simulate_spatial_holdout_data


def test_spatial_group_ordering_changes_only_group_labels():
    base = simulate_spatial_holdout_data(n_sites=25, n_species=3, seed=41)
    canonical = apply_spatial_holdout_group_order(base, "canonical", seed=5)
    reverse = apply_spatial_holdout_group_order(base, "reverse", seed=5)
    random_first = apply_spatial_holdout_group_order(base, "random", seed=5)
    random_second = apply_spatial_holdout_group_order(base, "random", seed=5)

    for key in base:
        if key in {"train_study_design", "test_study_design"}:
            unchanged = [column for column in base[key] if column != "plot"]
            pd.testing.assert_frame_equal(base[key][unchanged], reverse[key][unchanged])
        else:
            pd.testing.assert_frame_equal(base[key], reverse[key])
    pd.testing.assert_frame_equal(random_first["train_study_design"], random_second["train_study_design"])
    assert not canonical["train_study_design"]["plot"].equals(reverse["train_study_design"]["plot"])
    assert not canonical["train_study_design"]["plot"].equals(random_first["train_study_design"]["plot"])

    canonical_order = canonical["train_study_design"].sort_values("plot").index.tolist()
    reverse_order = reverse["train_study_design"].sort_values("plot").index.tolist()
    assert reverse_order == list(reversed(canonical_order))


def test_replicated_project_generator_writes_manifest_and_order_variants(tmp_path):
    output = tmp_path / "projects"
    manifest = generate_projects(output, [321])

    assert manifest["task_id"].tolist() == list(range(6))
    assert manifest.groupby("ordering").size().to_dict() == {
        "canonical": 4,
        "random": 1,
        "reverse": 1,
    }
    assert manifest.loc[manifest["ordering"] != "canonical", "model"].tolist() == [
        "spatial_nngp",
        "spatial_nngp",
    ]
    persisted = pd.read_csv(output / "tasks.csv")
    pd.testing.assert_frame_equal(persisted, manifest)

    canonical = output / "seed_321/canonical"
    reverse = output / "seed_321/reverse"
    for model in ["fixed", "spatial_full", "spatial_gpp", "spatial_nngp"]:
        assert (canonical / f"model_{model}.yaml").exists()
    pd.testing.assert_frame_equal(
        pd.read_csv(canonical / "data/train/Y.csv", index_col=0),
        pd.read_csv(reverse / "data/train/Y.csv", index_col=0),
    )
    canonical_design = pd.read_csv(canonical / "data/train/study_design.csv", index_col=0)
    reverse_design = pd.read_csv(reverse / "data/train/study_design.csv", index_col=0)
    pd.testing.assert_frame_equal(
        canonical_design[["xcoord", "ycoord"]],
        reverse_design[["xcoord", "ycoord"]],
    )
    assert not canonical_design["plot"].equals(reverse_design["plot"])


def test_replicated_summary_and_nngp_ordering_deltas():
    rows = []
    for seed, canonical_rmse in [(1, 0.5), (2, 0.7)]:
        for ordering, delta in [("canonical", 0.0), ("reverse", 0.1), ("random", -0.05)]:
            rows.append(
                {
                    "task_id": len(rows),
                    "seed": seed,
                    "ordering": ordering,
                    "model": "spatial_nngp",
                    "correlation": 0.9 - delta,
                    "rmse": canonical_rmse + delta,
                    "mae": 0.4 + delta,
                    "interval_coverage": 0.95 - delta,
                    "mean_interval_width": 2.0 + delta,
                }
            )
    raw = pd.DataFrame(rows)

    summary = _aggregate_metrics(raw)
    ordering = _nngp_ordering_deltas(raw)

    canonical = summary.loc[summary["ordering"] == "canonical"].iloc[0]
    assert canonical["replicates"] == 2
    assert canonical["rmse_mean"] == 0.6
    assert canonical["coverage_bias"] == 0.0
    assert canonical["coverage_min"] == 0.95
    assert canonical["coverage_max"] == 0.95
    reverse = ordering.loc[ordering["ordering"] == "reverse"]
    np.testing.assert_allclose(reverse["rmse_delta"], [0.1, 0.1])
    np.testing.assert_allclose(reverse["correlation_delta"], [-0.1, -0.1])
    random = ordering.loc[ordering["ordering"] == "random"]
    np.testing.assert_allclose(random["rmse_delta"], [-0.05, -0.05])


def test_replicated_lumi_scripts_have_valid_shell_syntax():
    scripts = [
        Path("docs/lumi_replicated_spatial_holdout_array_sbatch.sh"),
        Path("docs/lumi_replicated_spatial_holdout_analyze_sbatch.sh"),
        Path("docs/lumi_replicated_spatial_holdout_submit.sh"),
    ]
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
    array_script = scripts[0].read_text(encoding="utf-8")
    submit_script = scripts[2].read_text(encoding="utf-8")
    assert "TASKS_PER_SEED=6" in array_script
    assert 'seed_array="${SEED_ARRAY:-' in submit_script
    assert '"${SUBMIT_ANALYSIS:-0}" == "1"' in submit_script
