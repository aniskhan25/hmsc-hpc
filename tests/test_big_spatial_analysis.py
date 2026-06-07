from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from examples.analyze_big_spatial_plants import build_metrics_table


PROJECT = Path("examples/projects/big_spatial_plants_validation")


def test_big_spatial_analyzer_smoke(tmp_path):
    fixed, iid, spatial = _make_posteriors(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "examples/analyze_big_spatial_plants.py",
            "--project",
            str(PROJECT),
            "--fixed-posterior",
            str(fixed),
            "--iid-posterior",
            str(iid),
            "--spatial-posterior",
            str(spatial),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Big Spatial Plant Validation Report" in result.stdout
    assert "Model Comparison" in result.stdout
    assert "neighbor_residual_corr" in result.stdout
    assert "Interpretation Target" in result.stdout
    assert "Lambda:" in result.stdout


def test_big_spatial_analyzer_metrics(tmp_path):
    fixed, iid, spatial = _make_posteriors(tmp_path)

    metrics = build_metrics_table(
        PROJECT,
        {
            "fixed": fixed,
            "iid": iid,
            "spatial": spatial,
        },
        level=0.95,
        ppc_seed=3,
    )

    assert list(metrics["model"]) == ["fixed", "iid", "spatial"]
    assert list(metrics["random_effects"]) == ["none", "known", "known"]
    assert metrics["species_covered"].str.endswith(" / 40").all()
    assert metrics["site_richness_covered"].str.endswith(" / 400").all()
    assert np.isfinite(metrics["species_mae"].to_numpy(dtype=float)).all()
    assert np.isfinite(metrics["site_richness_mae"].to_numpy(dtype=float)).all()
    assert np.isfinite(metrics["neighbor_residual_corr"].to_numpy(dtype=float)).all()


def _make_posteriors(tmp_path):
    y_data = pd.read_csv(PROJECT / "data" / "Y_presence.csv", index_col=0)
    x_data = pd.read_csv(PROJECT / "data" / "X.csv", index_col=0)
    species = list(y_data.columns)
    covariates = ["Intercept", *x_data.columns]
    beta = np.zeros((len(covariates), len(species)), dtype=float)
    eta = np.zeros((len(y_data), 2), dtype=float)
    lam = np.zeros((2, len(species)), dtype=float)
    lam[0, :] = np.linspace(-0.2, 0.2, len(species))
    lam[1, :] = np.linspace(0.2, -0.2, len(species))

    fixed = tmp_path / "fixed.h5"
    iid = tmp_path / "iid.h5"
    spatial = tmp_path / "spatial.h5"
    _write_posterior(fixed, beta, species, covariates)
    _write_posterior(iid, beta, species, covariates, eta=eta, lam=lam)
    _write_posterior(spatial, beta, species, covariates, eta=eta, lam=lam)
    return fixed, iid, spatial


def _write_posterior(path, beta, species, covariates, eta=None, lam=None):
    import h5py

    draws = np.stack([beta, beta + 0.01], axis=0)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Beta", data=draws[None, ...])
        if eta is not None and lam is not None:
            level = handle.create_group("random_levels").create_group("0")
            level.create_dataset("Eta", data=np.stack([eta, eta + 0.01], axis=0)[None, ...])
            level.create_dataset("Lambda", data=np.stack([lam, lam + 0.01], axis=0)[None, ...])
        handle.attrs["pyhmsc_metadata"] = json.dumps(
            {
                "names": {
                    "covariates": covariates,
                    "species": species,
                },
                "formula": {"X": "~ " + " + ".join(covariates[1:])},
                "distribution": "probit",
            }
        )
