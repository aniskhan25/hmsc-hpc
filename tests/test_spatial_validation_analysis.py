import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from examples.analyze_spatial_validation import build_metrics_table


PROJECT = Path("examples/projects/simulated_spatial_validation")


def test_spatial_validation_analyzer_smoke(tmp_path):
    fixed, iid, spatial = _make_posteriors(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "examples/analyze_spatial_validation.py",
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

    assert "Simulated Spatial Validation Report" in result.stdout
    assert "Model Comparison" in result.stdout
    assert "fixed" in result.stdout
    assert "iid" in result.stdout
    assert "spatial" in result.stdout
    assert "eta_truth_corr" in result.stdout
    assert "neighbor_residual_corr" in result.stdout
    assert "Lambda:" in result.stdout


def test_spatial_validation_analyzer_metrics(tmp_path):
    fixed, iid, spatial = _make_posteriors(tmp_path)

    metrics = build_metrics_table(
        PROJECT,
        {
            "fixed": fixed,
            "iid": iid,
            "spatial": spatial,
        },
        level=0.95,
        ppc_seed=4,
    )

    assert list(metrics["model"]) == ["fixed", "iid", "spatial"]
    assert list(metrics["random_effects"]) == ["none", "known", "known"]
    assert list(metrics["beta_sign_recovered"]) == ["4 / 4", "4 / 4", "4 / 4"]
    assert metrics["species_covered"].str.endswith(" / 5").all()
    assert metrics["site_richness_covered"].str.endswith(" / 36").all()
    assert np.isfinite(metrics["species_mae"].to_numpy(dtype=float)).all()
    assert np.isfinite(metrics["site_richness_mae"].to_numpy(dtype=float)).all()
    assert np.isfinite(metrics["neighbor_residual_corr"].to_numpy(dtype=float)).all()

    eta_corr = dict(zip(metrics["model"], metrics["eta_truth_corr"], strict=True))
    assert eta_corr["fixed"] == "n/a"
    assert eta_corr["iid"] == pytest.approx(1.0)
    assert eta_corr["spatial"] == pytest.approx(1.0)


def _make_posteriors(tmp_path):
    project = PROJECT
    x_data = pd.read_csv(project / "data" / "X.csv", index_col=0)
    y_data = pd.read_csv(project / "data" / "Y.csv", index_col=0)
    truth_beta = pd.read_csv(project / "data" / "truth_beta.csv", index_col=0)
    truth_eta = pd.read_csv(project / "data" / "truth_site_effect.csv", index_col=0)
    truth_lambda = pd.read_csv(project / "data" / "truth_lambda.csv", index_col=0)
    species = list(y_data.columns)
    covariates = ["Intercept", "env"]
    beta = truth_beta.loc[covariates, species].to_numpy(dtype=float)
    eta = truth_eta.loc[x_data.index, ["eta"]].to_numpy(dtype=float)
    lam = truth_lambda.loc[["factor_0"], species].to_numpy(dtype=float)

    fixed = tmp_path / "fixed.h5"
    iid = tmp_path / "iid.h5"
    spatial = tmp_path / "spatial.h5"
    _write_posterior(fixed, beta, species, covariates)
    _write_posterior(iid, beta, species, covariates, eta=eta, lam=lam)
    _write_posterior(spatial, beta, species, covariates, eta=eta, lam=lam)
    return fixed, iid, spatial


def _write_posterior(path, beta, species, covariates, eta=None, lam=None):
    h5py = pytest.importorskip("h5py")

    draws = np.stack([beta, beta + 0.01], axis=0)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Beta", data=draws[None, ...])
        if eta is not None and lam is not None:
            level = handle.create_group("random_levels").create_group("0")
            level.create_dataset("Eta", data=np.stack([eta, eta + 0.01], axis=0)[None, ...])
            level.create_dataset("Lambda", data=np.stack([lam, lam + 0.01], axis=0)[None, ...])
        handle.attrs["pyhmsc_metadata"] = (
            '{"names":{"covariates":'
            + repr(covariates).replace("'", '"')
            + ',"species":'
            + repr(species).replace("'", '"')
            + '},"formula":{"X":"~ env"},"distribution":"probit"}'
        )
