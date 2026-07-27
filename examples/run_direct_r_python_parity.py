"""Run direct R/Python HMSC-HPC parity for a model config.

This is a non-neural boundary workflow for compact fixtures. It compares:

1. Python-native JSON/HDF5 compilation and sampling.
2. R-created Hmsc-HPC initialization imported through the legacy RDS boundary.

Use this for controlled fixed-effect and iid-random-effect parity before moving
back to real-data requalification workflows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.run_whittaker_r_python_parity import _compare_array, _posterior_compare
from hmsc.utils.export_rds_utils import load_model_from_rds
from pyhmsc.config import model_from_config
from pyhmsc.posterior import HmscFit
from pyhmsc.r_bridge import make_init_with_r, write_init_script_with_r
from pyhmsc.runner import run_gibbs_sampler
from pyhmsc.serialization import read_compiled_model
from pyhmsc.validation import validate_compiled_native_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--transient", type=int, default=100)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--chains", type=int)
    parser.add_argument("--verbose", type=int, default=100)
    parser.add_argument("--rng-seed", type=int, default=20260718)
    parser.add_argument("--fp", type=int, choices=[32, 64], default=64)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--prepare-r-init-script", action="store_true")
    parser.add_argument("--r-init-file", type=Path)
    parser.add_argument(
        "--reuse-existing-posteriors",
        action="store_true",
        help=(
            "skip sampling and regenerate comparison reports from existing "
            "python_native/posterior.h5 and r_bridge/posterior.h5 outputs"
        ),
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--min-beta-corr", type=float, default=0.9)
    parser.add_argument("--min-gamma-corr", type=float, default=0.9)
    parser.add_argument("--min-association-corr", type=float, default=0.75)
    parser.add_argument("--max-prediction-mae-delta", type=float, default=0.25)
    parser.add_argument(
        "--posterior-gates",
        choices=["strict", "diagnostic"],
        default="strict",
        help="whether posterior-summary correlations are enforced acceptance gates or reported diagnostics",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    model, config = model_from_config(args.config)
    settings = {
        "chains": int(args.chains or config.get("chains", 2)),
        "samples": int(args.samples),
        "transient": int(args.transient),
        "thin": int(args.thin),
        "verbose": int(args.verbose),
        "rng_seed": int(args.rng_seed),
        "fp": int(args.fp),
    }

    native_root = args.output / "python_native"
    r_root = args.output / "r_bridge"
    native_root.mkdir(exist_ok=True)
    r_root.mkdir(exist_ok=True)

    if args.prepare_r_init_script:
        script_path = write_init_script_with_r(
            model,
            init_file=r_root / "init_file.rds",
            workdir=r_root,
            samples=settings["samples"],
            transient=settings["transient"],
            thin=settings["thin"],
            chains=settings["chains"],
            verbose=settings["verbose"],
        )
        manifest = {
            "config": str(args.config),
            "r_script": str(script_path),
            "r_init_file": str(r_root / "init_file.rds"),
            "settings": settings,
        }
        (args.output / "r_init_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(script_path)
        return

    if args.reuse_existing_posteriors:
        native_fit, native_init, native_elapsed = _load_existing_python_native(model, native_root)
        r_fit, r_init, r_elapsed = _load_existing_r_bridge(model, r_root, r_init_file=args.r_init_file)
    else:
        native_fit, native_init, native_elapsed = _run_python_native(
            model,
            native_root,
            settings=settings,
            python=args.python,
        )
        r_fit, r_init, r_elapsed = _run_r_bridge(
            model,
            r_root,
            settings=settings,
            python=args.python,
            rscript=args.rscript,
            r_init_file=args.r_init_file,
        )

    metadata, native_arrays = read_compiled_model(native_init)
    r_import, r_model = load_model_from_rds(r_init)
    boundary = _boundary_checks(native_arrays, r_model)
    beta_compare = _posterior_compare(native_fit.beta_samples(), r_fit.beta_samples())
    gamma_compare = _posterior_compare(native_fit.gamma_samples(), r_fit.gamma_samples())
    random_compare = _random_level_compare(native_fit, r_fit)
    metrics = pd.DataFrame(
        [
            {"model": "python_native", **_predictive_metrics(model, native_fit)},
            {"model": "r_bridge", **_predictive_metrics(model, r_fit)},
        ]
    )
    deltas = _metric_deltas(metrics, baseline="r_bridge", candidate="python_native")
    gates = _acceptance_gates(
        boundary=boundary,
        beta_compare=beta_compare,
        gamma_compare=gamma_compare,
        random_compare=random_compare,
        deltas=deltas,
        min_beta_corr=args.min_beta_corr,
        min_gamma_corr=args.min_gamma_corr,
        min_association_corr=args.min_association_corr,
        max_prediction_mae_delta=args.max_prediction_mae_delta,
        enforce_posterior_gates=args.posterior_gates == "strict",
    )

    result = {
        "config": str(args.config),
        "settings": settings,
        "native_elapsed_seconds": native_elapsed,
        "r_bridge_elapsed_seconds": r_elapsed,
        "native_metadata_dimensions": metadata.get("dimensions", {}),
        "r_bridge_n_chains": int(r_import.get("nChains", [settings["chains"]])[0]),
        "boundary_checks": boundary,
        "beta_compare": beta_compare,
        "gamma_compare": gamma_compare,
        "random_level_compare": random_compare,
        "predictive_metrics": metrics.to_dict(orient="records"),
        "metric_deltas_python_native_minus_r_bridge": deltas,
        "posterior_gates": args.posterior_gates,
        "acceptance_gates": gates,
        "parity_passed": bool(all(gate["passed"] for gate in gates.values())),
    }

    (args.output / "direct_r_python_parity_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    metrics.to_csv(args.output / "direct_r_python_parity_predictive_metrics.csv", index=False)
    _write_report(args.output / "direct_r_python_parity_report.md", result, metrics)
    print((args.output / "direct_r_python_parity_report.md").read_text(encoding="utf-8"))

    if args.strict and not result["parity_passed"]:
        raise SystemExit(1)


def _run_python_native(model: Any, root: Path, *, settings: dict[str, int], python: str) -> tuple[HmscFit, Path, float]:
    compiled = model.compile(root / "compiled", chains=settings["chains"])
    failed = [result for result in validate_compiled_native_model(compiled.init_json) if not result.passed]
    if failed:
        details = "; ".join(f"{result.name}: {result.details}" for result in failed)
        raise RuntimeError(f"Python-native compiled model is not sampler-ready: {details}")
    posterior = root / "posterior.h5"
    start = time.perf_counter()
    run_gibbs_sampler(
        init_file=compiled.init_json,
        output_file=posterior,
        samples=settings["samples"],
        transient=settings["transient"],
        thin=settings["thin"],
        verbose=settings["verbose"],
        python=python,
        rng_seed=settings["rng_seed"],
        fp=settings["fp"],
    )
    elapsed = time.perf_counter() - start
    return HmscFit.from_file(posterior, model=model), compiled.init_json, elapsed


def _load_existing_python_native(model: Any, root: Path) -> tuple[HmscFit, Path, None]:
    init_file = root / "compiled" / "init.json"
    posterior = root / "posterior.h5"
    _require_existing(init_file)
    _require_existing(posterior)
    return HmscFit.from_file(posterior, model=model), init_file, None


def _run_r_bridge(
    model: Any,
    root: Path,
    *,
    settings: dict[str, int],
    python: str,
    rscript: str,
    r_init_file: Path | None = None,
) -> tuple[HmscFit, Path, float]:
    init_file = Path(r_init_file) if r_init_file is not None else root / "init_file.rds"
    posterior = root / "posterior.h5"
    start = time.perf_counter()
    if r_init_file is None:
        make_init_with_r(
            model,
            init_file=init_file,
            workdir=root,
            samples=settings["samples"],
            transient=settings["transient"],
            thin=settings["thin"],
            chains=settings["chains"],
            verbose=settings["verbose"],
            rscript=rscript,
        )
    elif not init_file.exists():
        raise FileNotFoundError(f"prebuilt R init file not found: {init_file}")
    run_gibbs_sampler(
        init_file=init_file,
        output_file=posterior,
        samples=settings["samples"],
        transient=settings["transient"],
        thin=settings["thin"],
        verbose=settings["verbose"],
        python=python,
        rng_seed=settings["rng_seed"],
        fp=settings["fp"],
    )
    elapsed = time.perf_counter() - start
    return HmscFit.from_file(posterior, model=model), init_file, elapsed


def _load_existing_r_bridge(model: Any, root: Path, *, r_init_file: Path | None = None) -> tuple[HmscFit, Path, None]:
    init_file = Path(r_init_file) if r_init_file is not None else root / "init_file.rds"
    posterior = root / "posterior.h5"
    _require_existing(init_file)
    _require_existing(posterior)
    return HmscFit.from_file(posterior, model=model), init_file, None


def _require_existing(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"reuse requested but required file does not exist: {path}")


def _boundary_checks(native_arrays: dict[str, np.ndarray], r_model: dict[str, Any]) -> dict[str, dict[str, Any]]:
    checks = {
        "Y": _compare_array(native_arrays["Y"], r_model.get("YScaled")),
        "X": _compare_array(native_arrays["X"], r_model.get("XScaled")),
        "T": _compare_array(native_arrays["T"], r_model.get("TrScaled")),
    }
    if "C" in native_arrays:
        checks["C"] = _compare_array(native_arrays["C"], r_model.get("C"))
    if "Pi" in native_arrays:
        checks["Pi"] = _compare_array(native_arrays["Pi"], _r_pi_zero_based(r_model.get("Pi")), atol=0.0)
    return checks


def _r_pi_zero_based(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=int) - 1


def _random_level_compare(native_fit: HmscFit, r_fit: HmscFit) -> dict[str, Any]:
    comparisons: dict[str, Any] = {"n_levels": 0, "levels": []}
    level = 0
    while True:
        try:
            native_eta = native_fit.eta_samples(level)
            r_eta = r_fit.eta_samples(level)
            native_lambda = native_fit.lambda_samples(level)
            r_lambda = r_fit.lambda_samples(level)
        except ValueError:
            break
        assoc_compare = _posterior_compare(
            native_fit.species_association_samples(level=level, correlation=False),
            r_fit.species_association_samples(level=level, correlation=False),
        )
        comparisons["levels"].append(
            {
                "level": level,
                "eta_shape_match": native_eta.shape == r_eta.shape,
                "lambda_shape_match": native_lambda.shape == r_lambda.shape,
                "native_eta_shape": [int(value) for value in native_eta.shape[2:]],
                "r_bridge_eta_shape": [int(value) for value in r_eta.shape[2:]],
                "native_lambda_shape": [int(value) for value in native_lambda.shape[2:]],
                "r_bridge_lambda_shape": [int(value) for value in r_lambda.shape[2:]],
                "association_compare": assoc_compare,
            }
        )
        level += 1
    comparisons["n_levels"] = level
    return comparisons


def _predictive_metrics(model: Any, fit: HmscFit) -> dict[str, float]:
    random_effects = "known" if getattr(model, "random_levels", None) else "none"
    prediction = fit.predict_mean(
        model.X,
        random_effects=random_effects,
        study_design=model.study_design,
    ).loc[model.Y.index, model.Y.columns]
    observed = model.Y.to_numpy(dtype=float)
    predicted = prediction.to_numpy(dtype=float)
    distribution = str(model.distr).lower()
    if distribution == "poisson":
        predicted = np.clip(predicted, 1e-12, 1e12)
        return {
            "prediction_mae": float(np.mean(np.abs(predicted - observed))),
            "prediction_rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
            "poisson_deviance": float(2.0 * np.mean(_poisson_deviance_terms(observed, predicted))),
        }
    if distribution in {"probit", "bernoulli", "binomial"}:
        predicted = np.clip(predicted, 1e-9, 1 - 1e-9)
        return {
            "prediction_mae": float(np.mean(np.abs(predicted - observed))),
            "brier_score": float(np.mean((predicted - observed) ** 2)),
            "log_loss": float(-np.mean(observed * np.log(predicted) + (1 - observed) * np.log(1 - predicted))),
        }
    return {
        "prediction_mae": float(np.mean(np.abs(predicted - observed))),
        "prediction_rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
    }


def _poisson_deviance_terms(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    ratio = np.ones_like(observed, dtype=float)
    positive = observed > 0
    ratio[positive] = observed[positive] / predicted[positive]
    terms = np.where(positive, observed * np.log(ratio) - (observed - predicted), predicted)
    return np.maximum(terms, 0.0)


def _metric_deltas(metrics: pd.DataFrame, *, baseline: str, candidate: str) -> dict[str, float]:
    indexed = metrics.set_index("model")
    rows = {}
    for column in metrics.columns:
        if column == "model":
            continue
        rows[column] = float(indexed.loc[candidate, column] - indexed.loc[baseline, column])
    return rows


def _acceptance_gates(
    *,
    boundary: dict[str, dict[str, Any]],
    beta_compare: dict[str, float],
    gamma_compare: dict[str, float],
    random_compare: dict[str, Any],
    deltas: dict[str, float],
    min_beta_corr: float,
    min_gamma_corr: float,
    min_association_corr: float,
    max_prediction_mae_delta: float,
    enforce_posterior_gates: bool = True,
) -> dict[str, dict[str, Any]]:
    beta_condition = bool(
        beta_compare.get("shape_match", False)
        and beta_compare["mean_correlation"] >= min_beta_corr
    )
    gamma_condition = bool(
        gamma_compare.get("shape_match", False)
        and gamma_compare["mean_correlation"] >= min_gamma_corr
    )
    gates = {
        "boundary_arrays": {
            "passed": bool(all(check["passed"] for check in boundary.values())),
            "threshold": "exact shape and max_abs_diff <= 1e-10",
        },
        "beta_mean_correlation": {
            "passed": bool(beta_condition or not enforce_posterior_gates),
            "observed": beta_compare["mean_correlation"],
            "threshold": min_beta_corr,
            "enforced": enforce_posterior_gates,
        },
        "gamma_mean_correlation": {
            "passed": bool(gamma_condition or not enforce_posterior_gates),
            "observed": gamma_compare["mean_correlation"],
            "threshold": min_gamma_corr,
            "enforced": enforce_posterior_gates,
        },
        "prediction_mae_delta": {
            "passed": bool(deltas["prediction_mae"] <= max_prediction_mae_delta),
            "observed": deltas["prediction_mae"],
            "threshold": f"<= {max_prediction_mae_delta}",
        },
    }
    if random_compare["n_levels"]:
        level_passed = []
        observed = []
        for level in random_compare["levels"]:
            assoc = level["association_compare"]
            observed.append(assoc["mean_correlation"])
            level_passed.append(
                level["eta_shape_match"]
                and level["lambda_shape_match"]
                and assoc.get("shape_match", False)
                and assoc["mean_correlation"] >= min_association_corr
            )
        gates["random_level_association_correlation"] = {
            "passed": bool(all(level_passed) or not enforce_posterior_gates),
            "observed": min(observed) if observed else float("nan"),
            "threshold": min_association_corr,
            "enforced": enforce_posterior_gates,
        }
    return gates


def _write_report(path: Path, result: dict[str, Any], metrics: pd.DataFrame) -> None:
    boundary_lines = [
        f"| {name} | {check['passed']} | {check['native_shape']} | {check['r_bridge_shape']} | {check['max_abs_diff']} |"
        for name, check in result["boundary_checks"].items()
    ]
    gate_lines = [
        f"| {name} | {gate['passed']} | {gate.get('observed', '')} | {gate.get('threshold', gate.get('threshold_abs', ''))} |"
        for name, gate in result["acceptance_gates"].items()
    ]
    random_lines = []
    for level in result["random_level_compare"]["levels"]:
        assoc = level["association_compare"]
        random_lines.append(
            f"| {level['level']} | {level['eta_shape_match']} | {level['lambda_shape_match']} | "
            f"{assoc['shape_match']} | {assoc['mean_correlation']:.6f} | {assoc['mean_mae']:.6f} |"
        )
    if not random_lines:
        random_lines = ["| none |  |  |  |  |  |"]
    report = [
        "# Direct R/Python HMSC-HPC Parity Report",
        "",
        f"Config: `{result['config']}`",
        f"Parity passed: `{result['parity_passed']}`",
        "",
        "This report compares Python-native HMSC compile/sample against an R-created Hmsc object imported through the original RDS Hmsc-HPC compatibility boundary. It does not involve neural calibration.",
        "",
        "## Settings",
        "",
        "```json",
        json.dumps(result["settings"], indent=2, sort_keys=True),
        "```",
        "",
        "## Boundary Arrays",
        "",
        "| array | passed | python-native shape | R boundary shape | max abs diff |",
        "| --- | --- | --- | --- | --- |",
        *boundary_lines,
        "",
        "## Posterior Summary Agreement",
        "",
        f"Beta compared shape: `{result['beta_compare']['left_shape']}` vs `{result['beta_compare']['right_shape']}`",
        f"Beta mean correlation: `{result['beta_compare']['mean_correlation']:.6f}`",
        f"Beta mean MAE: `{result['beta_compare']['mean_mae']:.6f}`",
        f"Gamma compared shape: `{result['gamma_compare']['left_shape']}` vs `{result['gamma_compare']['right_shape']}`",
        f"Gamma mean correlation: `{result['gamma_compare']['mean_correlation']:.6f}`",
        f"Gamma mean MAE: `{result['gamma_compare']['mean_mae']:.6f}`",
        "",
        "## Random-Level Agreement",
        "",
        "| level | Eta shape | Lambda shape | association shape | association corr | association MAE |",
        "| --- | --- | --- | --- | --- | --- |",
        *random_lines,
        "",
        "## Predictive Metrics",
        "",
        _frame_to_markdown(metrics),
        "",
        "## Acceptance Gates",
        "",
        "| gate | passed | observed | threshold |",
        "| --- | --- | --- | --- |",
        *gate_lines,
        "",
        "## Interpretation",
        "",
        "Passing this workflow supports Python-only HMSC parity with the original R+Python HMSC-HPC boundary for the selected compact fixture. Fixed-effect failures usually indicate preprocessing/import drift; random-level failures usually indicate study-design, factor-code, or latent-factor export drift.",
    ]
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```\n" + frame.to_string(index=False) + "\n```"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
