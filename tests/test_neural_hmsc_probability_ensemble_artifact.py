import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from pyhmsc import (
    PredictiveProbabilityEnsemble as PublicPredictiveEnsemble,
    load_predictive_mean_ensemble as public_load_predictive_mean_ensemble,
)
from pyhmsc.neural.ensemble import (
    DEFAULT_PREDICTIVE_MEAN_POLICY,
    PredictiveProbabilityEnsemble,
    file_sha256,
    load_predictive_mean_ensemble,
)
from pyhmsc.neural.posterior_heads import BetaPosterior
from pyhmsc.neural.storage import write_beta_posterior_hdf5
from pyhmsc.posterior import HmscFit


def test_probability_ensemble_manifest_round_trip_and_predict_mean(tmp_path):
    first = _write_member(tmp_path / "first.h5", mean=0.0, seed=11)
    second = _write_member(tmp_path / "second.h5", mean=0.8, seed=12)
    provenance = [
        _qualified_provenance(tmp_path, label="first"),
        _qualified_provenance(tmp_path, label="second"),
    ]
    ensemble = PredictiveProbabilityEnsemble.create(
        [first, second],
        seeds=[11, 12],
        calibration_role="affine_branch",
        member_provenance=provenance,
        provenance={"dataset": "fixture"},
    )
    manifest = ensemble.save(tmp_path / "ensemble.json", relative_paths=True)
    loaded = PredictiveProbabilityEnsemble.from_manifest(manifest)
    X = pd.DataFrame({"TMG": [-1.0, 0.0, 1.0]})

    prediction = loaded.predict_mean(X)
    expected = (
        HmscFit.from_file(first).predict_mean(X)
        + HmscFit.from_file(second).predict_mean(X)
    ) / 2.0
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    pd.testing.assert_frame_equal(prediction, expected)
    assert loaded.seeds == (11, 12)
    assert loaded.calibration_role == "affine_branch"
    assert loaded.parity_provenance_qualified
    assert payload["ordered_members"]
    assert payload["members"][0]["path"] == "first.h5"
    assert len(payload["members"][0]["sha256"]) == 64
    assert PublicPredictiveEnsemble is PredictiveProbabilityEnsemble

    reversed_subset = loaded.subset([12, 11])
    assert reversed_subset.seeds == (12, 11)
    pd.testing.assert_frame_equal(reversed_subset.predict_mean(X), expected)


def test_probability_ensemble_rejects_member_hash_change(tmp_path):
    first = _write_member(tmp_path / "first.h5", mean=0.0, seed=21)
    second = _write_member(tmp_path / "second.h5", mean=0.5, seed=22)
    ensemble = PredictiveProbabilityEnsemble.create(
        [first, second],
        seeds=[21, 22],
        calibration_role="scale_only",
    )
    manifest = ensemble.save(tmp_path / "ensemble.json")
    with second.open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(ValueError, match="hash mismatch"):
        PredictiveProbabilityEnsemble.from_manifest(manifest)


def test_probability_ensemble_rejects_parity_provenance_hash_change(tmp_path):
    first = _write_member(tmp_path / "first.h5", mean=0.0, seed=23)
    second = _write_member(tmp_path / "second.h5", mean=0.5, seed=24)
    first_provenance = _qualified_provenance(tmp_path, label="first")
    second_provenance = _qualified_provenance(tmp_path, label="second")
    ensemble = PredictiveProbabilityEnsemble.create(
        [first, second],
        seeds=[23, 24],
        calibration_role="scale_only",
        member_provenance=[first_provenance, second_provenance],
    )
    manifest = ensemble.save(tmp_path / "ensemble.json")
    Path(first_provenance["parity_metrics_path"]).write_text(
        "changed", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="parity provenance hash mismatch"):
        PredictiveProbabilityEnsemble.from_manifest(manifest)


def test_probability_ensemble_rejects_species_mismatch(tmp_path):
    first = _write_member(tmp_path / "first.h5", mean=0.0, seed=31)
    second = _write_member(
        tmp_path / "second.h5",
        mean=0.5,
        seed=32,
        species_name="other_species",
    )

    with pytest.raises(ValueError, match="incompatible"):
        PredictiveProbabilityEnsemble.create(
            [first, second],
            seeds=[31, 32],
            calibration_role="scale_only",
        )


def test_probability_ensemble_subset_rejects_unknown_seed(tmp_path):
    first = _write_member(tmp_path / "first.h5", mean=0.0, seed=41)
    second = _write_member(tmp_path / "second.h5", mean=0.5, seed=42)
    ensemble = PredictiveProbabilityEnsemble.create(
        [first, second],
        seeds=[41, 42],
        calibration_role="scale_only",
    )

    with pytest.raises(ValueError, match="unknown seeds"):
        ensemble.subset([41, 43])


def test_deployment_loader_defaults_to_affine_and_keeps_scale_fallback(tmp_path):
    provenance = [
        _qualified_provenance(tmp_path, label="first"),
        _qualified_provenance(tmp_path, label="second"),
    ]
    seeds = [51, 52]
    affine_paths = [
        _write_member(tmp_path / "affine_first.h5", mean=0.2, seed=51),
        _write_member(tmp_path / "affine_second.h5", mean=0.4, seed=52),
    ]
    scale_paths = [
        _write_member(tmp_path / "scale_first.h5", mean=0.0, seed=51),
        _write_member(tmp_path / "scale_second.h5", mean=0.1, seed=52),
    ]
    deployment_provenance = _deployment_provenance(
        provenance, seeds=seeds, dataset="whittaker"
    )
    PredictiveProbabilityEnsemble.create(
        affine_paths,
        seeds=seeds,
        calibration_role="affine_branch",
        member_provenance=provenance,
        provenance=deployment_provenance,
    ).save(tmp_path / "whittaker_affine_branch_ensemble.json")
    PredictiveProbabilityEnsemble.create(
        scale_paths,
        seeds=seeds,
        calibration_role="scale_only",
        member_provenance=provenance,
        provenance=deployment_provenance,
    ).save(tmp_path / "whittaker_scale_only_ensemble.json")

    promoted = load_predictive_mean_ensemble(tmp_path, dataset="whittaker")
    fallback = load_predictive_mean_ensemble(
        tmp_path, dataset="whittaker", policy="scale_only"
    )

    assert DEFAULT_PREDICTIVE_MEAN_POLICY == "affine_branch"
    assert public_load_predictive_mean_ensemble is load_predictive_mean_ensemble
    assert promoted.calibration_role == "affine_branch"
    assert fallback.calibration_role == "scale_only"
    assert promoted.qualified_mcmc_reference["ordered_members"][0]["seed"] == 51


def _write_member(
    path,
    *,
    mean: float,
    seed: int,
    species_name: str = "species_a",
):
    posterior = BetaPosterior(
        mean=tf.constant([[[mean], [0.25]]], dtype=tf.float32),
        scale=tf.ones((1, 2, 1), dtype=tf.float32) * 1.0e-6,
    )
    return write_beta_posterior_hdf5(
        posterior,
        path,
        covariate_names=["Intercept", "TMG"],
        species_names=[species_name],
        distribution="probit",
        formula="~ TMG",
        chains=1,
        draws=8,
        seed=seed,
        metadata={"artifact_role": "predictive_only"},
    )


def _qualified_provenance(tmp_path, *, label: str):
    values = {
        "reference_parity_qualified": True,
        "dataset_acceptance_passed": True,
    }
    for name in (
        "acceptance",
        "run_metadata",
        "parity_metrics",
        "mcmc_reference",
    ):
        path = tmp_path / f"{label}_{name}.json"
        path.write_text(f"{label}:{name}", encoding="utf-8")
        values[f"{name}_path"] = str(path)
        values[f"{name}_sha256"] = file_sha256(path)
    return values


def _deployment_provenance(member_provenance, *, seeds, dataset):
    return {
        "dataset": dataset,
        "response_semantics": "predictive_only",
        "selection_outcomes_used": False,
        "qualified_python_mcmc_reference": {
            "kind": "qualified_python_mcmc_probability_reference",
            "aggregation": "arithmetic_mean_response_probability",
            "ordered_members": [
                {
                    "seed": seed,
                    "path": provenance["mcmc_reference_path"],
                    "sha256": provenance["mcmc_reference_sha256"],
                }
                for seed, provenance in zip(seeds, member_provenance)
            ],
        },
    }
