"""Run direct Whittaker parity between Python-native and R-created Hmsc inputs.

This workflow tests the non-neural boundary:

1. Generate the deterministic Whittaker held-out-site project.
2. Compile and sample the fixed-effect trait/phylogeny model with Python-native
   pyhmsc artifacts.
3. Ask R/Hmsc to create the official Hmsc-HPC initialization RDS, then sample
   the same model through the legacy RDS compatibility boundary.
4. Compare boundary arrays, posterior summaries, and held-out predictions.
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

from examples.generate_whittaker_holdout_validation import generate_project
from pyhmsc.config import model_from_config
from pyhmsc.posterior import HmscFit
from pyhmsc.r_bridge import make_init_with_r, write_init_script_with_r
from pyhmsc.runner import run_gibbs_sampler
from pyhmsc.serialization import read_compiled_model
from pyhmsc.validation import validate_compiled_native_model
from hmsc.utils.export_rds_utils import load_model_from_rds


SOURCE = Path("examples/projects/whittaker_plants_hmsc_book")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-sites", type=int, default=12)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--transient", type=int)
    parser.add_argument("--thin", type=int)
    parser.add_argument("--chains", type=int)
    parser.add_argument("--verbose", type=int, default=500)
    parser.add_argument("--rng-seed", type=int, default=20260716)
    parser.add_argument("--fp", type=int, choices=[32, 64], default=64)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--prepare-r-init-script",
        action="store_true",
        help="write R init inputs/script and exit before running samplers",
    )
    parser.add_argument(
        "--r-init-file",
        type=Path,
        help="prebuilt R/Hmsc init RDS to use for the R-bridge sampler phase",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--min-beta-corr", type=float, default=0.95)
    parser.add_argument("--min-gamma-corr", type=float, default=0.95)
    parser.add_argument("--max-brier-delta", type=float, default=0.02)
    parser.add_argument("--max-log-loss-delta", type=float, default=0.05)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    project = args.output / "project"
    generate_project(args.source, project, n_test=args.test_sites)
    model, config = model_from_config(project / "model_fixed.yaml")

    settings = {
        "chains": int(args.chains or config.get("chains", 2)),
        "samples": int(args.samples or config.get("samples", 1000)),
        "transient": int(args.transient or config.get("transient", 500)),
        "thin": int(args.thin or config.get("thin", 10)),
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
            "project": str(project),
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
    heldout = pd.DataFrame(
        [
            {"model": "python_native", **_heldout_metrics(project, native_fit)},
            {"model": "r_bridge", **_heldout_metrics(project, r_fit)},
        ]
    )
    deltas = _metric_deltas(heldout, baseline="r_bridge", candidate="python_native")
    gates = _acceptance_gates(
        boundary=boundary,
        beta_compare=beta_compare,
        gamma_compare=gamma_compare,
        deltas=deltas,
        min_beta_corr=args.min_beta_corr,
        min_gamma_corr=args.min_gamma_corr,
        max_brier_delta=args.max_brier_delta,
        max_log_loss_delta=args.max_log_loss_delta,
    )

    result = {
        "project": str(project),
        "source": str(args.source),
        "settings": settings,
        "native_elapsed_seconds": native_elapsed,
        "r_bridge_elapsed_seconds": r_elapsed,
        "native_metadata_dimensions": metadata.get("dimensions", {}),
        "r_bridge_n_chains": int(r_import.get("nChains", [settings["chains"]])[0]),
        "boundary_checks": boundary,
        "beta_compare": beta_compare,
        "gamma_compare": gamma_compare,
        "heldout_metrics": heldout.to_dict(orient="records"),
        "metric_deltas_python_native_minus_r_bridge": deltas,
        "acceptance_gates": gates,
        "parity_passed": bool(all(gate["passed"] for gate in gates.values())),
    }

    (args.output / "whittaker_r_python_parity_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    heldout.to_csv(args.output / "whittaker_r_python_parity_heldout_metrics.csv", index=False)
    _write_report(args.output / "whittaker_r_python_parity_report.md", result, heldout)
    print((args.output / "whittaker_r_python_parity_report.md").read_text(encoding="utf-8"))

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


def _boundary_checks(native_arrays: dict[str, np.ndarray], r_model: dict[str, Any]) -> dict[str, dict[str, Any]]:
    checks = {
        "Y": _compare_array(native_arrays["Y"], r_model.get("YScaled")),
        "X": _compare_array(native_arrays["X"], r_model.get("XScaled")),
        "T": _compare_array(native_arrays["T"], r_model.get("TrScaled")),
    }
    if "C" in native_arrays:
        checks["C"] = _compare_array(native_arrays["C"], r_model.get("C"))
    return checks


def _compare_array(native: np.ndarray, r_value: Any, *, atol: float = 1e-10) -> dict[str, Any]:
    native = np.asarray(native, dtype=float)
    r_array = np.asarray(r_value, dtype=float)
    shape_ok = native.shape == r_array.shape
    max_abs_diff = None
    values_ok = False
    if shape_ok:
        max_abs_diff = float(np.max(np.abs(native - r_array))) if native.size else 0.0
        values_ok = bool(np.allclose(native, r_array, atol=atol, rtol=0.0))
    return {
        "passed": bool(shape_ok and values_ok),
        "native_shape": [int(value) for value in native.shape],
        "r_bridge_shape": [int(value) for value in r_array.shape],
        "max_abs_diff": max_abs_diff,
    }


def _posterior_compare(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_flat = np.asarray(left, dtype=float).reshape((-1,) + left.shape[2:])
    right_flat = np.asarray(right, dtype=float).reshape((-1,) + right.shape[2:])
    if left_flat.shape[1:] != right_flat.shape[1:]:
        return {
            "shape_match": False,
            "left_shape": [int(value) for value in left_flat.shape[1:]],
            "right_shape": [int(value) for value in right_flat.shape[1:]],
            "mean_correlation": float("nan"),
            "mean_mae": float("nan"),
            "mean_rmse": float("nan"),
            "sd_correlation": float("nan"),
            "sd_mae": float("nan"),
        }
    left_mean = left_flat.mean(axis=0).ravel()
    right_mean = right_flat.mean(axis=0).ravel()
    left_sd = left_flat.std(axis=0, ddof=1).ravel()
    right_sd = right_flat.std(axis=0, ddof=1).ravel()
    return {
        "shape_match": True,
        "left_shape": [int(value) for value in left_flat.shape[1:]],
        "right_shape": [int(value) for value in right_flat.shape[1:]],
        "mean_correlation": _correlation(left_mean, right_mean),
        "mean_mae": float(np.mean(np.abs(left_mean - right_mean))),
        "mean_rmse": float(np.sqrt(np.mean((left_mean - right_mean) ** 2))),
        "sd_correlation": _correlation(left_sd, right_sd),
        "sd_mae": float(np.mean(np.abs(left_sd - right_sd))),
    }


def _heldout_metrics(project: Path, fit: HmscFit) -> dict[str, float]:
    Y = pd.read_csv(project / "data/test/Y.csv", index_col=0)
    X = pd.read_csv(project / "data/test/X.csv", index_col=0)
    traits = pd.read_csv(project / "data/traits.csv", index_col=0)
    observed = Y.to_numpy(dtype=float)
    prediction = fit.predict_mean(X).loc[Y.index, Y.columns].clip(1e-9, 1 - 1e-9)
    probability = prediction.to_numpy(dtype=float)
    observed_richness = observed.sum(axis=1)
    predicted_richness = probability.sum(axis=1)
    cn = traits.loc[Y.columns, "CN"].to_numpy(dtype=float)
    observed_weighted_cn = (observed @ cn) / np.maximum(observed_richness, 1.0)
    predicted_weighted_cn = (probability @ cn) / np.maximum(predicted_richness, 1e-12)
    tmg = X["TMG"].to_numpy(dtype=float)
    return {
        "brier_score": float(np.mean((probability - observed) ** 2)),
        "log_loss": float(-np.mean(observed * np.log(probability) + (1 - observed) * np.log(1 - probability))),
        "macro_auc": _macro_auc(Y, prediction),
        "prevalence_mae": float(np.mean(np.abs(probability.mean(axis=0) - observed.mean(axis=0)))),
        "richness_mae": float(np.mean(np.abs(predicted_richness - observed_richness))),
        "observed_richness_slope": _slope(tmg, observed_richness),
        "predicted_richness_slope": _slope(tmg, predicted_richness),
        "observed_weighted_cn_slope": _slope(tmg, observed_weighted_cn),
        "predicted_weighted_cn_slope": _slope(tmg, predicted_weighted_cn),
    }


def _metric_deltas(metrics: pd.DataFrame, *, baseline: str, candidate: str) -> dict[str, float]:
    indexed = metrics.set_index("model")
    rows = {}
    for column in ["brier_score", "log_loss", "macro_auc", "prevalence_mae", "richness_mae"]:
        rows[column] = float(indexed.loc[candidate, column] - indexed.loc[baseline, column])
    return rows


def _acceptance_gates(
    *,
    boundary: dict[str, dict[str, Any]],
    beta_compare: dict[str, float],
    gamma_compare: dict[str, float],
    deltas: dict[str, float],
    min_beta_corr: float,
    min_gamma_corr: float,
    max_brier_delta: float,
    max_log_loss_delta: float,
) -> dict[str, dict[str, Any]]:
    return {
        "boundary_arrays": {
            "passed": bool(all(check["passed"] for check in boundary.values())),
            "threshold": "exact shape and max_abs_diff <= 1e-10",
        },
        "beta_mean_correlation": {
            "passed": bool(
                beta_compare.get("shape_match", False)
                and beta_compare["mean_correlation"] >= min_beta_corr
            ),
            "observed": beta_compare["mean_correlation"],
            "threshold": min_beta_corr,
        },
        "gamma_mean_correlation": {
            "passed": bool(
                gamma_compare.get("shape_match", False)
                and gamma_compare["mean_correlation"] >= min_gamma_corr
            ),
            "observed": gamma_compare["mean_correlation"],
            "threshold": min_gamma_corr,
        },
        "heldout_brier_delta": {
            "passed": bool(abs(deltas["brier_score"]) <= max_brier_delta),
            "observed": deltas["brier_score"],
            "threshold_abs": max_brier_delta,
        },
        "heldout_log_loss_delta": {
            "passed": bool(abs(deltas["log_loss"]) <= max_log_loss_delta),
            "observed": deltas["log_loss"],
            "threshold_abs": max_log_loss_delta,
        },
    }


def _write_report(path: Path, result: dict[str, Any], heldout: pd.DataFrame) -> None:
    boundary_lines = [
        f"| {name} | {check['passed']} | {check['native_shape']} | {check['r_bridge_shape']} | {check['max_abs_diff']} |"
        for name, check in result["boundary_checks"].items()
    ]
    gate_lines = [
        f"| {name} | {gate['passed']} | {gate.get('observed', '')} | {gate.get('threshold', gate.get('threshold_abs', ''))} |"
        for name, gate in result["acceptance_gates"].items()
    ]
    report = [
        "# Whittaker Python-Only HMSC Parity Report",
        "",
        f"Project: `{result['project']}`",
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
        "## Held-Out Whittaker Metrics",
        "",
        _frame_to_markdown(heldout),
        "",
        "## Acceptance Gates",
        "",
        "| gate | passed | observed | threshold |",
        "| --- | --- | --- | --- |",
        *gate_lines,
        "",
        "## Interpretation",
        "",
        "Passing this workflow supports Python-only HMSC parity with the original R+Python HMSC-HPC boundary for the Whittaker fixed-effect trait/phylogeny model. Failing boundary arrays indicate a compile/import mismatch; failing posterior or held-out gates indicate sampler or initialization behavior that needs separate investigation.",
    ]
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def _macro_auc(Y: pd.DataFrame, prediction: pd.DataFrame) -> float:
    values = []
    for species in Y.columns:
        observed = Y[species].to_numpy(dtype=int)
        positive = int(observed.sum())
        negative = len(observed) - positive
        if positive == 0 or negative == 0:
            continue
        ranks = pd.Series(prediction[species].to_numpy(dtype=float)).rank(method="average").to_numpy()
        rank_sum = float(ranks[observed == 1].sum())
        values.append((rank_sum - positive * (positive + 1) / 2) / (positive * negative))
    return float(np.mean(values)) if values else float("nan")


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0])


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size or left.size == 0:
        return float("nan")
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


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
