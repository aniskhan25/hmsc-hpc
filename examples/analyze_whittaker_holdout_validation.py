"""Analyze held-out Whittaker predictions with traits and phylogeny."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.config import model_from_config
from pyhmsc.posterior import HmscFit


MODELS = ["fixed", "iid"]


def build_metrics_table(project: Path, run_root: Path, prediction_seed: int = 17) -> pd.DataFrame:
    Y = pd.read_csv(project / "data/test/Y.csv", index_col=0)
    X = pd.read_csv(project / "data/test/X.csv", index_col=0)
    study = pd.read_csv(project / "data/test/study_design.csv", index_col=0)
    traits = pd.read_csv(project / "data/traits.csv", index_col=0)
    observed = Y.to_numpy(dtype=float)
    observed_richness = observed.sum(axis=1)
    cn = traits.loc[Y.columns, "CN"].to_numpy(dtype=float)
    observed_weighted_cn = (observed @ cn) / np.maximum(observed_richness, 1.0)
    tmg = X["TMG"].to_numpy(dtype=float)
    rows = []
    for name in MODELS:
        model, _config = model_from_config(project / f"model_{name}.yaml")
        model_root = run_root / name
        fit = HmscFit.from_file(model_root / "posterior.h5", model=model)
        if name == "fixed":
            prediction = fit.predict_mean(X)
            random_effects = "none"
        else:
            prediction = fit.predict_mean(
                X,
                study_design=study,
                random_effects="marginal",
                rng_seed=prediction_seed,
            )
            random_effects = "marginal"
        prediction = prediction.loc[Y.index, Y.columns].clip(1e-9, 1 - 1e-9)
        probability = prediction.to_numpy(dtype=float)
        if not np.isfinite(probability).all():
            raise ValueError(f"{name} produced non-finite held-out predictions")
        predicted_richness = probability.sum(axis=1)
        predicted_weighted_cn = (probability @ cn) / np.maximum(predicted_richness, 1e-12)
        gamma = _gamma_tmg_cn(fit)
        diagnostics = fit.diagnostics_overview("Beta", ess_threshold=200)
        resource = _read_resource_metrics(model_root / "resource_metrics.txt")
        rows.append(
            {
                "model": name,
                "random_effects": random_effects,
                "brier_score": float(np.mean((probability - observed) ** 2)),
                "log_loss": float(
                    -np.mean(observed * np.log(probability) + (1 - observed) * np.log(1 - probability))
                ),
                "macro_auc": _macro_auc(Y, prediction),
                "auc_species": _auc_species_count(Y),
                "prevalence_mae": float(np.mean(np.abs(probability.mean(axis=0) - observed.mean(axis=0)))),
                "richness_mae": float(np.mean(np.abs(predicted_richness - observed_richness))),
                "observed_richness_slope": _slope(tmg, observed_richness),
                "predicted_richness_slope": _slope(tmg, predicted_richness),
                "observed_weighted_cn_slope": _slope(tmg, observed_weighted_cn),
                "predicted_weighted_cn_slope": _slope(tmg, predicted_weighted_cn),
                "gamma_tmg_cn_mean": gamma[0],
                "gamma_tmg_cn_lower": gamma[1],
                "gamma_tmg_cn_upper": gamma[2],
                "beta_rhat_max": diagnostics["rhat_max"],
                "beta_ess_min": diagnostics["ess_min"],
                "elapsed_seconds": resource.get("elapsed_seconds", float("nan")),
                "max_rss_kb": resource.get("max_rss_kb", float("nan")),
                "samples": resource.get("samples", float("nan")),
                "transient": resource.get("transient", float("nan")),
                "thin": resource.get("thin", float("nan")),
            }
        )
    return pd.DataFrame(rows)


def _gamma_tmg_cn(fit: HmscFit) -> tuple[float, float, float]:
    try:
        summary = fit.gamma_summary()
    except (KeyError, ValueError):
        return float("nan"), float("nan"), float("nan")
    row = summary.loc[(summary["covariate"] == "TMG") & (summary["trait"] == "CN")]
    if len(row) != 1:
        return float("nan"), float("nan"), float("nan")
    value = row.iloc[0]
    return float(value["mean"]), float(value["lower"]), float(value["upper"])


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0])


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


def _auc_species_count(Y: pd.DataFrame) -> int:
    present = Y.sum(axis=0)
    return int(((present > 0) & (present < len(Y))).sum())


def _read_resource_metrics(path: Path) -> dict[str, float]:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            values[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return values


def _build_report(project: Path, metrics: pd.DataFrame) -> str:
    best = metrics.loc[metrics["brier_score"].idxmin()]
    return "\n".join(
        [
            "# Whittaker Trait/Phylogeny Hold-Out Validation Report",
            "",
            f"Project: {project}",
            "Training sites: 40",
            "Held-out sites: 12",
            "Species: 75",
            "",
            metrics.to_string(index=False),
            "",
            f"Lowest Brier score: {best['model']} ({float(best['brier_score']):.6f})",
            "",
            "Expected ecological signs: richness slope < 0, weighted-CN slope > 0, fixed-model Gamma TMG x CN > 0.",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prediction-seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = build_metrics_table(args.project, args.run_root, args.prediction_seed)
    report = _build_report(args.project, metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    metrics.to_csv(args.output.with_suffix(".csv"), index=False)
    print(report)


if __name__ == "__main__":
    main()
