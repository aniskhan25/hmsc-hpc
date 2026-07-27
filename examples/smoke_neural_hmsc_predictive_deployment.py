"""Smoke the manifest-backed neural predictive deployment policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.neural.ensemble import (
    DEFAULT_PREDICTIVE_MEAN_POLICY,
    PREDICTIVE_MEAN_POLICIES,
    file_sha256,
    load_predictive_mean_ensemble,
)
from pyhmsc.neural.deployment import (
    PROMOTED_PREDICTIVE_BASELINE_ID,
    load_predictive_deployment_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest-dir", type=Path)
    source.add_argument("--baseline-root", type=Path)
    parser.add_argument(
        "--baseline-id", default=PROMOTED_PREDICTIVE_BASELINE_ID
    )
    parser.add_argument("--frozen-run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["whittaker", "big_spatial"],
        choices=["whittaker", "big_spatial"],
    )
    parser.add_argument(
        "--policy",
        choices=PREDICTIVE_MEAN_POLICIES,
        default=DEFAULT_PREDICTIVE_MEAN_POLICY,
    )
    parser.add_argument(
        "--fallback-policy",
        choices=PREDICTIVE_MEAN_POLICIES,
        default="scale_only",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.policy == args.fallback_policy:
        parser.error("--policy and --fallback-policy must differ")

    result = run_deployment_smoke(
        manifest_dir=args.manifest_dir,
        baseline_root=args.baseline_root,
        baseline_id=args.baseline_id,
        frozen_run_root=args.frozen_run_root,
        seed=args.seed,
        datasets=args.datasets,
        policy=args.policy,
        fallback_policy=args.fallback_policy,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predictive_deployment_smoke.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = render_deployment_smoke(result)
    (args.output / "predictive_deployment_smoke.md").write_text(
        report, encoding="utf-8"
    )
    print(report)


def run_deployment_smoke(
    *,
    manifest_dir: Path | None = None,
    baseline_root: Path | None = None,
    baseline_id: str = PROMOTED_PREDICTIVE_BASELINE_ID,
    frozen_run_root: Path,
    seed: int,
    datasets: Sequence[str],
    policy: str = DEFAULT_PREDICTIVE_MEAN_POLICY,
    fallback_policy: str = "scale_only",
) -> dict[str, Any]:
    """Exercise default/fallback prediction without opening target outcomes."""
    rows = []
    for dataset in datasets:
        if (manifest_dir is None) == (baseline_root is None):
            raise ValueError("exactly one deployment source must be configured")
        if baseline_root is not None:
            default_ensemble = load_predictive_deployment_baseline(
                baseline_root,
                baseline_id=baseline_id,
                dataset=dataset,
                policy=policy,
            )
            fallback_ensemble = load_predictive_deployment_baseline(
                baseline_root,
                baseline_id=baseline_id,
                dataset=dataset,
                policy=fallback_policy,
            )
            active_manifest_dir = (
                baseline_root / baseline_id / "manifests"
            )
        else:
            default_ensemble = load_predictive_mean_ensemble(
                manifest_dir,
                dataset=dataset,
                policy=policy,
            )
            fallback_ensemble = load_predictive_mean_ensemble(
                manifest_dir,
                dataset=dataset,
                policy=fallback_policy,
            )
            active_manifest_dir = manifest_dir
        if default_ensemble.seeds != fallback_ensemble.seeds:
            raise ValueError(f"{dataset} default/fallback seed order differs")
        if default_ensemble.compatibility != fallback_ensemble.compatibility:
            raise ValueError(f"{dataset} default/fallback compatibility differs")
        X_path = _dataset_X_path(frozen_run_root, seed=seed, dataset=dataset)
        X = pd.read_csv(X_path, index_col=0)
        default_prediction = default_ensemble.predict_mean(X)
        fallback_prediction = fallback_ensemble.predict_mean(X)
        if not default_prediction.index.equals(fallback_prediction.index):
            raise ValueError(f"{dataset} default/fallback prediction index differs")
        if not default_prediction.columns.equals(fallback_prediction.columns):
            raise ValueError(f"{dataset} default/fallback species order differs")
        default_values = default_prediction.to_numpy(dtype=float)
        fallback_values = fallback_prediction.to_numpy(dtype=float)
        if not np.isfinite(default_values).all() or not np.isfinite(
            fallback_values
        ).all():
            raise ValueError(f"{dataset} deployment prediction is non-finite")
        reference = default_ensemble.qualified_mcmc_reference
        default_manifest = (
            active_manifest_dir / f"{dataset}_{policy}_ensemble.json"
        ).resolve()
        fallback_manifest = (
            active_manifest_dir / f"{dataset}_{fallback_policy}_ensemble.json"
        ).resolve()
        rows.append(
            {
                "dataset": dataset,
                "default_policy": policy,
                "fallback_policy": fallback_policy,
                "default_manifest": str(default_manifest),
                "default_manifest_sha256": file_sha256(default_manifest),
                "fallback_manifest": str(fallback_manifest),
                "fallback_manifest_sha256": file_sha256(fallback_manifest),
                "ordered_seeds": list(default_ensemble.seeds),
                "default_member_hashes": [
                    member.sha256 for member in default_ensemble.members
                ],
                "fallback_member_hashes": [
                    member.sha256 for member in fallback_ensemble.members
                ],
                "prediction_shape": list(default_values.shape),
                "default_probability_min": float(default_values.min()),
                "default_probability_max": float(default_values.max()),
                "default_fallback_max_abs_delta": float(
                    np.max(np.abs(default_values - fallback_values))
                ),
                "parity_provenance_qualified": bool(
                    default_ensemble.parity_provenance_qualified
                ),
                "mcmc_reference_kind": reference["kind"],
                "mcmc_reference_member_count": len(reference["ordered_members"]),
                "mcmc_used_for_neural_prediction": False,
                "target_response_opened": False,
                "passed": True,
            }
        )
    return {
        "kind": "neural_predictive_deployment_default_wiring_smoke",
        "decision": "predictive_deployment_smoke_passed",
        "default_policy": policy,
        "fallback_policy": fallback_policy,
        "baseline_id": baseline_id if baseline_root is not None else None,
        "response_semantics": "predictive_only",
        "qualified_python_mcmc_role": "statistical_reference_only",
        "all_datasets_passed": all(row["passed"] for row in rows),
        "datasets": rows,
    }


def render_deployment_smoke(result: dict[str, Any]) -> str:
    rows = pd.DataFrame(result["datasets"])
    display = rows[
        [
            "dataset",
            "default_policy",
            "fallback_policy",
            "prediction_shape",
            "default_fallback_max_abs_delta",
            "parity_provenance_qualified",
            "mcmc_used_for_neural_prediction",
            "passed",
        ]
    ]
    return "\n".join(
        [
            "# Neural Predictive Deployment Smoke",
            "",
            f"Decision: `{result['decision']}`",
            f"Default policy: `{result['default_policy']}`",
            f"Fallback policy: `{result['fallback_policy']}`",
            "Qualified Python MCMC role: `statistical_reference_only`",
            "",
            "```text",
            display.to_string(index=False),
            "```",
            "",
        ]
    )


def _dataset_X_path(root: Path, *, seed: int, dataset: str) -> Path:
    dataset_root = root / f"seed_{seed}" / dataset
    if dataset == "whittaker":
        return dataset_root / "whittaker_holdout/data/test/X.csv"
    if dataset == "big_spatial":
        return dataset_root / "big_spatial_transfer_project/data/test/X.csv"
    raise ValueError(f"unsupported deployment dataset: {dataset!r}")


if __name__ == "__main__":
    main()
