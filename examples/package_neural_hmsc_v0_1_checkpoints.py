#!/usr/bin/env python3
"""Package retained Neural-HMSC v0.1 calibrations without refitting them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.neural import (  # noqa: E402
    NeuralHmscInference,
    package_neural_hmsc_coefficient_calibration,
)


DEFAULT_SEEDS = (20260721, 20260722, 20260723)
PREDICTIVE_ARTIFACTS = (
    {
        "dataset": "whittaker",
        "policy": "affine_branch",
        "name": "neural_predictive_distribution.h5",
        "source": "member",
    },
    {
        "dataset": "big_spatial",
        "policy": "affine_branch",
        "name": "big_spatial_neural_predictive_distribution.h5",
        "source": "member",
    },
    {
        "dataset": "whittaker",
        "policy": "scale_only",
        "name": "neural_predictive_distribution_scale_only.h5",
        "source": "sensitivity",
    },
    {
        "dataset": "big_spatial",
        "policy": "scale_only",
        "name": "big_spatial_neural_predictive_distribution_scale_only.h5",
        "source": "sensitivity",
        "source_name": "neural_predictive_distribution_scale_only.h5",
    },
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_seed(
    *,
    seed: int,
    source_members_root: Path,
    sensitivity_root: Path,
    output_root: Path,
) -> dict[str, object]:
    source_member = source_members_root / str(seed)
    source_checkpoint = source_member / "neural_checkpoint"
    run_metadata_path = (
        sensitivity_root / f"seed_{seed}" / "whittaker" / "run_metadata.json"
    )
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    if run_metadata.get("coefficient_calibration") != "external_monotone":
        raise ValueError(f"seed {seed} did not use external_monotone calibration")
    calibration_metadata = run_metadata.get("coefficient_calibration_metadata")
    if not isinstance(calibration_metadata, dict):
        raise ValueError(f"seed {seed} lacks coefficient calibration metadata")
    if calibration_metadata.get("method") != "external_context_monotone_scale":
        raise ValueError(f"seed {seed} calibration method differs")
    source_engine = NeuralHmscInference.load(source_checkpoint)
    if source_engine.distribution != "probit":
        raise ValueError(f"seed {seed} checkpoint is not probit")
    destination_member = output_root / str(seed)
    destination_member.mkdir()
    provenance = {
        "kind": "independent_simulation_calibration_provenance",
        "training_corpus_version": source_engine.training_corpus_version,
        "calibration_training_role": "independent_simulation",
        "target_response_used_for_calibration": False,
        "packaging_refit_performed": False,
        "packaging_reselection_performed": False,
        "source_run_metadata_path": str(run_metadata_path),
        "source_run_metadata_sha256": file_sha256(run_metadata_path),
        "source_seed": int(seed),
    }
    packaged_checkpoint = package_neural_hmsc_coefficient_calibration(
        source_checkpoint,
        destination_member / "neural_checkpoint",
        calibration_metadata=calibration_metadata,
        provenance=provenance,
    )
    copied_predictive = []
    for artifact in PREDICTIVE_ARTIFACTS:
        name = str(artifact["name"])
        if artifact["source"] == "member":
            source_path = source_member / name
        else:
            source_path = (
                sensitivity_root
                / f"seed_{seed}"
                / str(artifact["dataset"])
                / str(artifact.get("source_name", name))
            )
        destination_path = destination_member / name
        shutil.copy2(source_path, destination_path)
        source_hash = file_sha256(source_path)
        destination_hash = file_sha256(destination_path)
        if source_hash != destination_hash:
            raise ValueError(f"seed {seed} predictive artifact copy hash mismatch")
        copied_predictive.append(
            {
                "dataset": artifact["dataset"],
                "policy": artifact["policy"],
                "name": name,
                "source_sha256": source_hash,
                "packaged_sha256": destination_hash,
            }
        )
    source_weights_hash = file_sha256(source_checkpoint / "weights.weights.h5")
    packaged_weights_hash = file_sha256(packaged_checkpoint / "weights.weights.h5")
    if source_weights_hash != packaged_weights_hash:
        raise ValueError(f"seed {seed} checkpoint weights changed during packaging")
    loaded = NeuralHmscInference.load(packaged_checkpoint)
    if loaded.coefficient_calibration is None:
        raise ValueError(f"seed {seed} packaged checkpoint lacks calibration")
    return {
        "seed": int(seed),
        "source_checkpoint": str(source_checkpoint),
        "packaged_checkpoint": str(Path(str(seed)) / "neural_checkpoint"),
        "checkpoint_version": loaded.checkpoint_version,
        "training_corpus_version": loaded.training_corpus_version,
        "distribution": loaded.distribution,
        "dimensions": loaded.dimensions,
        "calibration_method": loaded.coefficient_calibration.method,
        "source_weights_sha256": source_weights_hash,
        "packaged_weights_sha256": packaged_weights_hash,
        "checkpoint_manifest_sha256": file_sha256(
            packaged_checkpoint / "neural_checkpoint.json"
        ),
        "calibration_artifact_sha256": file_sha256(
            packaged_checkpoint / "coefficient_calibration.json"
        ),
        "run_metadata_sha256": provenance["source_run_metadata_sha256"],
        "predictive_artifacts": copied_predictive,
        "weights_unchanged": True,
        "model_fitting_performed": False,
        "calibration_selection_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-members-root", type=Path, required=True)
    parser.add_argument("--sensitivity-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_members_root = args.source_members_root.expanduser().resolve()
    sensitivity_root = args.sensitivity_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        rows = [
            package_seed(
                seed=int(seed),
                source_members_root=source_members_root,
                sensitivity_root=sensitivity_root,
                output_root=staging,
            )
            for seed in args.seeds
        ]
        manifest = {
            "schema_version": 1,
            "kind": "neural_hmsc_v0_1_packaged_checkpoint_bundle",
            "release_scope": "fixed_shape_fixed_effect_probit_beta",
            "seeds": [int(seed) for seed in args.seeds],
            "model_fitting_performed": False,
            "calibration_selection_performed": False,
            "all_weights_unchanged": all(row["weights_unchanged"] for row in rows),
            "members": rows,
        }
        manifest_path = staging / "package_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    final_manifest = output_root / "package_manifest.json"
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "manifest": str(final_manifest),
                "manifest_sha256": file_sha256(final_manifest),
                "members": len(args.seeds),
                "weights_unchanged": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
