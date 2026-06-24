"""Analyze real-data spatial hold-out prediction and resource use."""

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


MODELS = ["fixed", "spatial_full", "spatial_gpp", "spatial_nngp"]


def build_metrics_table(project: Path, run_root: Path, prediction_seed: int = 17) -> pd.DataFrame:
    Y = pd.read_csv(project / "data/test/Y.csv", index_col=0)
    X = pd.read_csv(project / "data/test/X.csv", index_col=0)
    study = pd.read_csv(project / "data/test/study_design.csv", index_col=0)
    coords = pd.read_csv(project / "data/test/coords.csv", index_col=0)
    rows = []
    for name in MODELS:
        model, _config = model_from_config(project / f"model_{name}.yaml")
        model_root = run_root / name
        posterior = model_root / "posterior.h5"
        fit = HmscFit.from_file(posterior, model=model)
        if name == "fixed":
            prediction = fit.predict_mean(X)
        else:
            prediction = fit.predict_mean(
                X,
                study_design=study,
                coords=coords,
                random_effects="known",
                spatial_prediction="conditional",
                rng_seed=prediction_seed,
            )
        prediction = prediction.loc[Y.index, Y.columns].clip(1e-9, 1 - 1e-9)
        observed = Y.to_numpy(dtype=float)
        probability = prediction.to_numpy(dtype=float)
        if not np.isfinite(probability).all():
            raise ValueError(f"{name} produced non-finite held-out predictions")
        resource = _read_resource_metrics(model_root / "resource_metrics.txt")
        rows.append(
            {
                "model": name,
                "brier_score": float(np.mean((probability - observed) ** 2)),
                "log_loss": float(
                    -np.mean(observed * np.log(probability) + (1 - observed) * np.log(1 - probability))
                ),
                "macro_auc": _macro_auc(Y, prediction),
                "auc_species": _auc_species_count(Y),
                "prevalence_mae": float(np.mean(np.abs(probability.mean(axis=0) - observed.mean(axis=0)))),
                "richness_mae": float(np.mean(np.abs(probability.sum(axis=1) - observed.sum(axis=1)))),
                "elapsed_seconds": resource.get("elapsed_seconds", float("nan")),
                "max_rss_kb": resource.get("max_rss_kb", float("nan")),
                "compiled_bytes": resource.get("compiled_bytes", float("nan")),
                "posterior_bytes": resource.get("posterior_bytes", float("nan")),
                "samples": resource.get("samples", float("nan")),
                "transient": resource.get("transient", float("nan")),
                "thin": resource.get("thin", float("nan")),
            }
        )
    metrics = pd.DataFrame(rows)
    fixed_brier = float(metrics.loc[metrics["model"] == "fixed", "brier_score"].iloc[0])
    metrics["brier_improvement_vs_fixed"] = fixed_brier - metrics["brier_score"]
    return metrics


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
    values: dict[str, float] = {}
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
            "# Big Spatial Plant Hold-Out Validation Report",
            "",
            f"Project: {project}",
            "Training sites: 319",
            "Held-out sites: 81",
            "Species: 40",
            "",
            metrics.to_string(index=False),
            "",
            f"Lowest Brier score: {best['model']} ({float(best['brier_score']):.6f})",
            "",
            "Lower Brier score, log loss, prevalence MAE, and richness MAE are better; higher AUC is better.",
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
