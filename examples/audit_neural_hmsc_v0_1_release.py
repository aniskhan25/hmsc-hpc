#!/usr/bin/env python3
"""Audit the frozen Neural-HMSC v0.1 candidate without fitting a new model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.compiler import compile_hmsc_model
from pyhmsc.neural import NeuralHmscCompatibilityError, NeuralHmscInference
from pyhmsc.neural.deployment import validate_predictive_deployment_baseline


SEEDS = (20260721, 20260722, 20260723)
DATASETS = ("whittaker", "big_spatial")
POLICIES = ("affine_branch", "scale_only")
SBC_COVERAGE_RANGE = (0.925, 0.975)
MAX_RANK_ERROR = 0.025
MAX_PROPER_SCORE_RATIO = 1.10
MIN_INFERENCE_SPEEDUP = 100.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    observed = file_sha256(path)
    return {
        "path": str(path),
        "sha256": observed,
        "expected_sha256": expected_sha256,
        "hash_matches": expected_sha256 is None or observed == expected_sha256,
    }


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_baseline(baseline_root: Path, members_root: Path) -> dict[str, Any]:
    baseline_path = baseline_root / "baseline.json"
    baseline = validate_predictive_deployment_baseline(
        baseline_path,
        expected_baseline_id="neural_predictive_affine_v1",
    )
    records: list[dict[str, Any]] = [file_record(baseline_path)]
    local_members: list[dict[str, Any]] = []
    checkpoint_artifacts: list[dict[str, Any]] = []
    parity_hashes: set[str] = set()
    for seed in SEEDS:
        checkpoint = members_root / str(seed) / "neural_checkpoint"
        checkpoint_artifacts.extend(
            [
                {
                    "seed": seed,
                    "role": "manifest",
                    **file_record(checkpoint / "neural_checkpoint.json"),
                },
                {
                    "seed": seed,
                    "role": "weights",
                    **file_record(checkpoint / "weights.weights.h5"),
                },
            ]
        )
    for dataset in DATASETS:
        for policy in POLICIES:
            manifest_record = baseline["datasets"][dataset][policy]
            manifest_path = baseline_root / manifest_record["path"]
            records.append(
                file_record(
                    manifest_path,
                    expected_sha256=str(manifest_record["sha256"]),
                )
            )
            manifest = _json(manifest_path)
            if policy != "affine_branch":
                continue
            local_name = (
                "neural_predictive_distribution.h5"
                if dataset == "whittaker"
                else "big_spatial_neural_predictive_distribution.h5"
            )
            for member in manifest["members"]:
                seed = int(member["seed"])
                local_path = members_root / str(seed) / local_name
                local_members.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        **file_record(
                            local_path,
                            expected_sha256=str(member["sha256"]),
                        ),
                    }
                )
                parity_hashes.add(str(member["provenance"]["parity_metrics_sha256"]))
    for evidence in baseline["evidence"].values():
        records.append(
            file_record(
                baseline_root / evidence["path"],
                expected_sha256=str(evidence["sha256"]),
            )
        )
    return {
        "baseline_id": baseline["baseline_id"],
        "default_policy": baseline["default_policy"],
        "fallback_policy": baseline["fallback_policy"],
        "bundle_validation_passed": True,
        "bundle_files": records,
        "checkpoint_artifacts": checkpoint_artifacts,
        "local_affine_members": local_members,
        "local_member_hashes_passed": all(row["hash_matches"] for row in local_members),
        "recorded_parity_sha256": sorted(parity_hashes),
    }


def audit_public_api(
    members_root: Path,
    sensitivity_root: Path,
    output: Path,
) -> dict[str, Any]:
    seed = SEEDS[0]
    checkpoint = members_root / str(seed) / "neural_checkpoint"
    engine = NeuralHmscInference.load(checkpoint)
    source = (
        sensitivity_root
        / f"seed_{seed}"
        / "whittaker"
        / "whittaker_holdout"
        / "data"
        / "train"
    )
    X = pd.read_csv(source / "X.csv", index_col=0)
    Y = pd.read_csv(source / "Y.csv", index_col=0)
    compiled = compile_hmsc_model(
        Y=Y,
        X=X,
        formula="~ TMG",
        distr="probit",
        chains=1,
        output=output / "api_smoke" / "compiled_fixed",
    )
    compatibility = engine.check_compatibility(compiled.init_json)
    raw_posterior = engine.predict_beta_posterior(compiled.init_json, calibrated=False)
    calibrated_posterior = engine.predict_beta_posterior(compiled.init_json)
    fit = engine.infer(
        compiled.init_json,
        draws=8,
        chains=1,
        seed=20260721,
        output=output / "api_smoke" / "posterior.h5",
    )
    import h5py

    with h5py.File(fit.output_file, "r") as handle:
        emitted_metadata = json.loads(handle.attrs["pyhmsc_metadata"])
    summary_shapes = {
        "beta_samples": list(fit.beta_samples().shape),
        "beta_mean": list(fit.beta_mean().shape),
        "beta_ci_lower": list(fit.beta_ci()["lower"].shape),
        "predict_mean": list(fit.predict_mean(X).shape),
    }

    traits = pd.DataFrame(
        {"audit_trait": [float(index) for index in range(Y.shape[1])]},
        index=Y.columns,
    )
    unsupported = compile_hmsc_model(
        Y=Y,
        X=X,
        formula="~ TMG",
        distr="probit",
        traits=traits,
        trait_formula="~ audit_trait",
        chains=1,
        output=output / "api_smoke" / "compiled_traits",
    )
    rejection = None
    try:
        engine.check_compatibility(unsupported.init_json)
    except NeuralHmscCompatibilityError as exc:
        rejection = str(exc)

    manifest_path = checkpoint / "neural_checkpoint.json"
    manifest = _json(manifest_path)
    calibration_keys = {
        key: manifest.get(key)
        for key in (
            "coefficient_calibration",
            "coefficient_calibration_metadata",
            "calibration",
        )
        if key in manifest
    }
    calibration_record = manifest.get("coefficient_calibration")
    calibration_artifact = None
    if isinstance(calibration_record, dict):
        calibration_artifact = file_record(
            checkpoint / str(calibration_record["path"]),
            expected_sha256=str(calibration_record["sha256"]),
        )
    raw_scale = np.asarray(raw_posterior.scale)
    calibrated_scale = np.asarray(calibrated_posterior.scale)
    calibration_applied = not np.allclose(raw_scale, calibrated_scale)
    qualified_calibration_packaged = (
        engine.coefficient_calibration is not None
        and engine.coefficient_calibration.method == "external_context_monotone_scale"
        and calibration_artifact is not None
        and calibration_artifact["hash_matches"]
        and calibration_applied
    )
    emitted_calibration_method = emitted_metadata.get("calibration", {}).get("method")
    emitted_calibration_hash = (
        emitted_metadata.get("neural_api", {})
        .get("coefficient_calibration", {})
        .get("sha256")
    )
    emitted_metadata_verified = (
        emitted_calibration_method == "external_context_monotone_scale"
        and isinstance(calibration_record, dict)
        and emitted_calibration_hash == calibration_record.get("sha256")
    )
    return {
        "checkpoint": file_record(manifest_path),
        "weights": file_record(checkpoint / "weights.weights.h5"),
        "checkpoint_loaded": True,
        "compiled_artifact_compatible": bool(compatibility["compatible"]),
        "posterior_emitted": Path(fit.output_file).exists(),
        "posterior_metadata_calibration_method": emitted_calibration_method,
        "posterior_metadata_calibration_sha256": emitted_calibration_hash,
        "posterior_metadata_verified": emitted_metadata_verified,
        "hmscfit_summary_and_prediction_shapes": summary_shapes,
        "unsupported_traits_rejected": rejection is not None,
        "unsupported_traits_error": rejection,
        "public_limitations": list(manifest.get("limitations", [])),
        "checkpoint_calibration_keys": calibration_keys,
        "coefficient_calibration_artifact": calibration_artifact,
        "coefficient_calibration_method": (
            None
            if engine.coefficient_calibration is None
            else engine.coefficient_calibration.method
        ),
        "coefficient_calibration_applied": calibration_applied,
        "mean_calibrated_to_raw_scale_ratio": float(
            np.mean(calibrated_scale / raw_scale)
        ),
        "qualified_calibration_packaged_with_checkpoint": (
            qualified_calibration_packaged
        ),
    }


def audit_simulation(sensitivity_root: Path) -> dict[str, Any]:
    distributions: dict[str, Any] = {}
    probit_rows = []
    source_files = []
    for seed in SEEDS:
        path = (
            sensitivity_root
            / f"seed_{seed}"
            / "whittaker"
            / "whittaker_neural_sbc_diagnostics.json"
        )
        source_files.append(file_record(path))
        matches = [
            row
            for row in _json(path)
            if row["posterior_variant"] == "coefficient_calibrated"
            and row["sbc_stratum_kind"] == "overall"
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one calibrated overall SBC row in {path}")
        row = matches[0]
        expected_variance = float(row["sbc_expected_rank_variance"])
        values = {
            "seed": seed,
            "coverage_95": float(row["sbc_beta_interval_coverage_95"]),
            "rank_mean": float(row["sbc_rank_mean"]),
            "rank_mean_error": abs(float(row["sbc_rank_mean"]) - 0.5),
            "rank_variance": float(row["sbc_rank_variance"]),
            "rank_variance_error": abs(
                float(row["sbc_rank_variance"]) - expected_variance
            ),
            "beta_mean_rmse": float(row["sbc_beta_mean_rmse"]),
            "n_replicates": int(row["sbc_n_replicates"]),
            "n_draws": int(row["sbc_n_draws"]),
        }
        values["gate_passed"] = (
            SBC_COVERAGE_RANGE[0] <= values["coverage_95"] <= SBC_COVERAGE_RANGE[1]
            and values["rank_mean_error"] <= MAX_RANK_ERROR
            and values["rank_variance_error"] <= MAX_RANK_ERROR
        )
        probit_rows.append(values)
    distributions["probit"] = {
        "api_implemented": True,
        "release_qualified": all(row["gate_passed"] for row in probit_rows),
        "rows": probit_rows,
        "source_files": source_files,
    }
    for distribution in ("normal", "poisson"):
        distributions[distribution] = {
            "api_implemented": True,
            "release_qualified": False,
            "rows": [],
            "reason": "no retained production-shape fixed-evaluation SBC evidence was supplied to the audit",
        }
    return {
        "thresholds": {
            "coverage_95": list(SBC_COVERAGE_RANGE),
            "max_rank_mean_error": MAX_RANK_ERROR,
            "max_rank_variance_error": MAX_RANK_ERROR,
        },
        "distributions": distributions,
        "all_advertised_distributions_qualified": all(
            value["release_qualified"] for value in distributions.values()
        ),
        "narrowed_probit_scope_qualified": distributions["probit"]["release_qualified"],
    }


def audit_ecological(requalification_path: Path) -> dict[str, Any]:
    payload = _json(requalification_path)
    rows = []
    for row in payload["mcmc_comparison"]["full_rows"]:
        record = {
            "dataset": str(row["dataset"]),
            "brier_ratio_vs_mcmc": float(row["affine_vs_mcmc_brier_score_ratio"]),
            "log_loss_ratio_vs_mcmc": float(row["affine_vs_mcmc_log_loss_ratio"]),
            "brier_score": float(row["affine_brier_score"]),
            "log_loss": float(row["affine_log_loss"]),
            "mcmc_brier_score": float(row["mcmc_brier_score"]),
            "mcmc_log_loss": float(row["mcmc_log_loss"]),
        }
        record["gate_passed"] = (
            record["brier_ratio_vs_mcmc"] <= MAX_PROPER_SCORE_RATIO
            and record["log_loss_ratio_vs_mcmc"] <= MAX_PROPER_SCORE_RATIO
        )
        rows.append(record)
    return {
        "source": file_record(requalification_path),
        "threshold": {"max_brier_and_log_loss_ratio_vs_mcmc": MAX_PROPER_SCORE_RATIO},
        "rows": rows,
        "gate_passed": {row["dataset"] for row in rows} == set(DATASETS)
        and all(row["gate_passed"] for row in rows),
        "mcmc_role": "qualified statistical reference",
        "neural_equivalence_claimed": False,
    }


def audit_runtime(sensitivity_root: Path) -> dict[str, Any]:
    rows = []
    sources = []
    for seed in SEEDS:
        whittaker_metadata = _json(
            sensitivity_root / f"seed_{seed}" / "whittaker" / "run_metadata.json"
        )
        training_seconds = float(whittaker_metadata["training_seconds"])
        for dataset in DATASETS:
            path = sensitivity_root / f"seed_{seed}" / dataset / "run_metadata.json"
            sources.append(file_record(path))
            metadata = _json(path)
            inference_seconds = float(metadata["neural_inference_seconds"])
            mcmc_seconds = float(metadata["mcmc_seconds"])
            rows.append(
                {
                    "seed": seed,
                    "dataset": dataset,
                    "checkpoint_training_seconds": training_seconds,
                    "neural_inference_seconds": inference_seconds,
                    "mcmc_seconds": mcmc_seconds,
                    "inference_speedup": mcmc_seconds / inference_seconds,
                    "amortization_break_even_datasets": training_seconds
                    / (mcmc_seconds - inference_seconds),
                }
            )
    summaries = {}
    for dataset in DATASETS:
        selected = [row for row in rows if row["dataset"] == dataset]
        summaries[dataset] = {
            key: mean(float(row[key]) for row in selected)
            for key in (
                "checkpoint_training_seconds",
                "neural_inference_seconds",
                "mcmc_seconds",
                "inference_speedup",
                "amortization_break_even_datasets",
            )
        }
        summaries[dataset]["gate_passed"] = (
            summaries[dataset]["inference_speedup"] >= MIN_INFERENCE_SPEEDUP
        )
    return {
        "threshold": {"min_inference_only_speedup": MIN_INFERENCE_SPEEDUP},
        "training_time_semantics": "one Whittaker-shape checkpoint training cost reused for both ecological evaluations",
        "rows": rows,
        "summary": summaries,
        "source_files": sources,
        "gate_passed": all(summary["gate_passed"] for summary in summaries.values()),
    }


def support_matrix(
    simulation: dict[str, Any], *, calibration_packaged: bool
) -> list[dict[str, Any]]:
    return [
        {
            "capability": "fixed-effect Beta posterior, fixed checkpoint shape, probit",
            "implementation": "public",
            "qualification": (
                "release-qualified"
                if calibration_packaged
                else "qualified evidence; packaging blocker remains"
            ),
        },
        {
            "capability": "fixed-effect Beta posterior, fixed checkpoint shape, Gaussian/Normal",
            "implementation": "public API",
            "qualification": "experimental; no retained release SBC evidence",
        },
        {
            "capability": "fixed-effect Beta posterior, fixed checkpoint shape, Poisson",
            "implementation": "public API",
            "qualification": "experimental; no retained release SBC evidence",
        },
        {
            "capability": "manifest-backed Whittaker/Big Spatial predictive ensemble",
            "implementation": "public predictive-only deployment API",
            "qualification": "qualified for the two named datasets and exact manifests",
        },
        {
            "capability": "variable site/species shape",
            "implementation": "prototype",
            "qualification": "unsupported by public checkpoint API",
        },
        {
            "capability": "traits or phylogeny",
            "implementation": "prototype",
            "qualification": "rejected by public compatibility boundary",
        },
        {
            "capability": "iid/spatial latent effects or random slopes",
            "implementation": "prototype",
            "qualification": "rejected by public compatibility boundary",
        },
        {
            "capability": "detection submodel, GPP, or NNGP",
            "implementation": "not public",
            "qualification": "unsupported",
        },
        {
            "capability": "joint-posterior or full-HMSC MCMC equivalence",
            "implementation": "not provided",
            "qualification": "explicitly not claimed",
        },
    ]


def render_markdown(audit: dict[str, Any]) -> str:
    decision = audit["decision"]
    lines = [
        "# Neural-HMSC v0.1 release-readiness audit",
        "",
        f"Decision: **{decision}**.",
        "",
        "No model was fitted or selected by this audit. It validates retained artifacts, runs a public-API smoke, and applies the frozen gates.",
        "",
        "## Gate summary",
        "",
        "| Gate | Result |",
        "| --- | --- |",
    ]
    for name, gate in audit["gates"].items():
        lines.append(f"| {name.replace('_', ' ')} | {'pass' if gate else 'fail'} |")
    lines.extend(
        [
            "",
            "Only gates listed in `release_required_gates` determine the narrowed probit v0.1 decision. The three-distribution row records the broader implementation gap.",
        ]
    )
    lines.extend(["", "## Simulation evidence", ""])
    lines.append("| Distribution | Public API | Release qualified | Evidence |")
    lines.append("| --- | --- | --- | --- |")
    for name, values in audit["simulation"]["distributions"].items():
        evidence = (
            f"{len(values['rows'])} production-shape seed(s)"
            if values["rows"]
            else values["reason"]
        )
        lines.append(
            f"| {name} | {str(values['api_implemented']).lower()} | "
            f"{str(values['release_qualified']).lower()} | {evidence} |"
        )
    probit = audit["simulation"]["distributions"]["probit"]["rows"]
    lines.extend(
        [
            "",
            "Probit calibrated SBC across the three frozen seeds:",
            "",
            "| Seed | Coverage 95 | Rank mean error | Rank variance error | Beta RMSE | Gate |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in probit:
        lines.append(
            f"| {row['seed']} | {row['coverage_95']:.6f} | {row['rank_mean_error']:.6f} | "
            f"{row['rank_variance_error']:.6f} | {row['beta_mean_rmse']:.6f} | "
            f"{'pass' if row['gate_passed'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Ecological predictive evidence",
            "",
            "| Dataset | Brier ratio vs MCMC | Log-loss ratio vs MCMC | Gate |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in audit["ecological"]["rows"]:
        lines.append(
            f"| {row['dataset']} | {row['brier_ratio_vs_mcmc']:.6f} | "
            f"{row['log_loss_ratio_vs_mcmc']:.6f} | {'pass' if row['gate_passed'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "Qualified Python MCMC remains the statistical reference. These scores establish a bounded predictive approximation, not posterior equivalence.",
            "",
            "## Runtime",
            "",
            "| Dataset | Training seconds | Inference seconds | MCMC seconds | Speedup | Break-even datasets |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset, row in audit["runtime"]["summary"].items():
        lines.append(
            f"| {dataset} | {row['checkpoint_training_seconds']:.3f} | "
            f"{row['neural_inference_seconds']:.6f} | {row['mcmc_seconds']:.3f} | "
            f"{row['inference_speedup']:.1f}x | {row['amortization_break_even_datasets']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Release blockers",
            "",
        ]
    )
    for blocker in audit["release_blockers"]:
        lines.append(f"- {blocker}")
    if not audit["release_blockers"]:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Support matrix",
            "",
            "| Capability | Implementation | Qualification |",
            "| --- | --- | --- |",
        ]
    )
    for row in audit["support_matrix"]:
        lines.append(
            f"| {row['capability']} | {row['implementation']} | {row['qualification']} |"
        )
    interpretation = (
        "The narrowed probit v0.1 release envelope passes. The public checkpoint now carries and applies the qualified external_monotone coefficient calibration, while Normal and Poisson remain explicitly experimental."
        if decision == "release_ready"
        else "The numerical probit envelope passes, but one or more required release gates remain unresolved. Normal and Poisson remain explicitly experimental."
    )
    next_step = (
        "The next roadmap step is to freeze the packaged checkpoints, calibration artifacts, predictive manifests, evidence, and support matrix under one immutable v0.1 release identifier, then publish the end-to-end inference example."
        if decision == "release_ready"
        else "The next roadmap step is to resolve only the listed release blockers and rerun this unchanged audit."
    )
    lines.extend(["", "## Interpretation", "", interpretation, "", next_step, ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--members-root", type=Path, required=True)
    parser.add_argument("--sensitivity-root", type=Path, required=True)
    parser.add_argument("--requalification-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline = audit_baseline(args.baseline_root.resolve(), args.members_root.resolve())
    public_api = audit_public_api(
        args.members_root.resolve(), args.sensitivity_root.resolve(), output
    )
    simulation = audit_simulation(args.sensitivity_root.resolve())
    ecological = audit_ecological(args.requalification_json.resolve())
    runtime = audit_runtime(args.sensitivity_root.resolve())
    gates = {
        "frozen_artifact_and_provenance": baseline["bundle_validation_passed"]
        and baseline["local_member_hashes_passed"],
        "public_api_mechanics": all(
            public_api[key]
            for key in (
                "checkpoint_loaded",
                "compiled_artifact_compatible",
                "posterior_emitted",
                "posterior_metadata_verified",
                "unsupported_traits_rejected",
            )
        ),
        "qualified_calibration_packaged_with_public_checkpoint": public_api[
            "qualified_calibration_packaged_with_checkpoint"
        ],
        "all_three_distributions_qualified": simulation[
            "all_advertised_distributions_qualified"
        ],
        "narrowed_probit_simulation_envelope": simulation[
            "narrowed_probit_scope_qualified"
        ],
        "ecological_predictive_envelope": ecological["gate_passed"],
        "runtime_envelope": runtime["gate_passed"],
    }
    release_required_gates = (
        "frozen_artifact_and_provenance",
        "public_api_mechanics",
        "qualified_calibration_packaged_with_public_checkpoint",
        "narrowed_probit_simulation_envelope",
        "ecological_predictive_envelope",
        "runtime_envelope",
    )
    blockers = []
    if not gates["qualified_calibration_packaged_with_public_checkpoint"]:
        blockers.append(
            "The public checkpoint manifest contains no external_monotone coefficient calibration, so NeuralHmscInference.load() emits the uncalibrated posterior rather than the qualified one."
        )
    release_ready = all(gates[name] for name in release_required_gates)
    audit = {
        "schema_version": 1,
        "kind": "neural_hmsc_v0_1_release_readiness_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "release_ready" if release_ready else "not_release_ready",
        "model_fitting_performed": False,
        "calibration_selection_performed": False,
        "gates": gates,
        "release_required_gates": list(release_required_gates),
        "release_blockers": blockers,
        "release_scope_exclusions": {
            "normal": "implemented but not release-qualified",
            "poisson": "implemented but not release-qualified",
        },
        "baseline": baseline,
        "public_api": public_api,
        "simulation": simulation,
        "ecological": ecological,
        "runtime": runtime,
        "support_matrix": support_matrix(
            simulation,
            calibration_packaged=public_api[
                "qualified_calibration_packaged_with_checkpoint"
            ],
        ),
        "claim_boundary": {
            "qualified_claim": "bounded accelerated fixed-shape probit Beta approximation after packaging is fixed",
            "not_claimed": [
                "joint-posterior equivalence",
                "full-HMSC structural equivalence",
                "predictive superiority over MCMC",
                "Normal or Poisson release qualification",
            ],
        },
    }
    json_path = output / "neural_hmsc_v0_1_release_audit.json"
    markdown_path = output / "neural_hmsc_v0_1_release_audit.md"
    json_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(audit), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": audit["decision"],
                "json": str(json_path),
                "json_sha256": file_sha256(json_path),
                "markdown": str(markdown_path),
                "markdown_sha256": file_sha256(markdown_path),
                "release_blockers": blockers,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
