import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyhmsc.compiler import compile_hmsc_model
from pyhmsc.neural import (
    NEURAL_CHECKPOINT_VERSION,
    NeuralHmscCompatibilityError,
    NeuralHmscInference,
    package_neural_hmsc_coefficient_calibration,
    simulate_fixed_effect_dataset,
)
from pyhmsc.neural.conditional_calibration import (
    ConditionalBetaScaleCalibration,
    apply_conditional_beta_scale_calibration,
)
from pyhmsc.neural.train import fixed_shape_training_data


def test_public_neural_hmsc_api_saves_loads_and_infers(tmp_path):
    train = [
        simulate_fixed_effect_dataset(
            n_sites=12,
            n_species=2,
            distribution="normal",
            seed=1100 + idx,
        )
        for idx in range(2)
    ]
    test = simulate_fixed_effect_dataset(
        n_sites=12,
        n_species=2,
        distribution="normal",
        seed=1200,
    )
    engine = NeuralHmscInference.for_fixed_effects(
        n_sites=12,
        n_species=2,
        distribution="normal",
        formula="~ x1 + x2",
    )

    history = engine.fit(train, epochs=1, batch_size=1)
    checkpoint = engine.save(tmp_path / "checkpoint")
    loaded = NeuralHmscInference.load(checkpoint)
    fit = loaded.infer(
        test,
        draws=5,
        chains=1,
        seed=123,
        output=tmp_path / "posterior.h5",
    )

    assert history.loss
    assert (checkpoint / "neural_checkpoint.json").exists()
    assert fit.output_file == tmp_path / "posterior.h5"
    assert fit.beta_samples().shape == (1, 5, 3, 2)
    assert list(fit.beta_mean().index) == ["Intercept", "x1", "x2"]
    assert list(fit.beta_mean().columns) == ["sp1", "sp2"]
    assert fit.beta_ci()["lower"].shape == (3, 2)
    assert fit.predict_mean(test.X).shape == test.Y.shape
    assert loaded.predict_beta_posterior(test).mean.shape == (1, 3, 2)
    assert loaded.predict_beta_posterior(test).scale_tril is None


def test_public_neural_hmsc_api_infers_from_compiled_artifact(tmp_path):
    dataset = simulate_fixed_effect_dataset(
        n_sites=10,
        n_species=2,
        distribution="normal",
        seed=1300,
    )
    compiled = compile_hmsc_model(
        Y=dataset.Y,
        X=dataset.X,
        formula="~ x1 + x2",
        distr="normal",
        chains=1,
        output=tmp_path / "compiled",
    )
    engine = NeuralHmscInference.for_fixed_effects(
        n_sites=10,
        n_species=2,
        distribution="normal",
        formula="~ x1 + x2",
    )

    report = engine.check_compatibility(compiled.init_json)
    fit = engine.infer(compiled.init_json, draws=4, chains=1, seed=1301)

    assert report["compatible"] is True
    assert report["dimensions"] == {"n_sites": 10, "n_covariates": 3, "n_species": 2}
    assert fit.beta_samples().shape == (1, 4, 3, 2)
    assert fit.predict_mean(dataset.X).shape == dataset.Y.shape


def test_public_neural_hmsc_api_rejects_unsupported_compiled_structures(tmp_path):
    Y = pd.DataFrame(np.ones((6, 2)), columns=["sp1", "sp2"])
    X = pd.DataFrame({"x1": np.linspace(0, 1, 6), "x2": np.linspace(1, 0, 6)})
    traits = pd.DataFrame({"body": [1.0, 2.0]}, index=["sp1", "sp2"])
    compiled = compile_hmsc_model(
        Y=Y,
        X=X,
        formula="~ x1 + x2",
        distr="normal",
        traits=traits,
        trait_formula="~ body",
        chains=1,
        output=tmp_path / "compiled_traits",
    )
    engine = NeuralHmscInference.for_fixed_effects(
        n_sites=6,
        n_species=2,
        distribution="normal",
        formula="~ x1 + x2",
    )

    with pytest.raises(
        NeuralHmscCompatibilityError, match="unsupported compiled features"
    ):
        engine.infer(compiled.init_json, draws=3)


def test_public_neural_hmsc_checkpoint_manifest_is_versioned(tmp_path):
    engine = NeuralHmscInference.for_fixed_effects(n_sites=4, n_species=1)

    checkpoint = engine.save(tmp_path / "checkpoint")
    manifest = json.loads(
        (checkpoint / "neural_checkpoint.json").read_text(encoding="utf-8")
    )

    assert manifest["checkpoint_version"] == NEURAL_CHECKPOINT_VERSION
    assert manifest["training_corpus_version"] == "0.1"
    assert manifest["model_family"] == "fixed_effect_beta"
    assert manifest["posterior_family"] == "diagonal_normal"
    assert manifest["probit_anchor"] == "ridge"
    assert "limitations" in manifest


def test_public_neural_hmsc_loads_legacy_diagonal_checkpoint(tmp_path):
    engine = NeuralHmscInference.for_fixed_effects(
        n_sites=4,
        n_species=1,
        posterior_family="diagonal_normal",
    )
    checkpoint = engine.save(tmp_path / "checkpoint")
    manifest_path = checkpoint / "neural_checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["posterior_family"]
    manifest["checkpoint_version"] = "0.2"
    for key in (
        "probit_anchor",
        "probit_anchor_iterations",
        "probit_anchor_prior_precision",
        "probit_anchor_eta_clip",
    ):
        manifest.pop(key)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = NeuralHmscInference.load(checkpoint)

    assert loaded.model.posterior_family == "diagonal_normal"
    assert loaded.model.probit_anchor == "ridge"


def test_public_neural_hmsc_round_trips_probit_anchor_configuration(tmp_path):
    engine = NeuralHmscInference.for_fixed_effects(
        n_sites=8,
        n_species=2,
        distribution="probit",
        probit_anchor="irls_laplace",
        probit_anchor_iterations=6,
        probit_anchor_prior_precision=1.5,
        probit_anchor_eta_clip=5.0,
    )

    loaded = NeuralHmscInference.load(engine.save(tmp_path / "anchor_checkpoint"))

    assert loaded.model.probit_anchor == "irls_laplace"
    assert loaded.model.probit_anchor_iterations == 6
    assert loaded.model.probit_anchor_prior_precision == pytest.approx(1.5)
    assert loaded.model.probit_anchor_eta_clip == pytest.approx(5.0)


def test_public_neural_hmsc_packages_and_applies_frozen_external_calibration(
    tmp_path,
):
    dataset = simulate_fixed_effect_dataset(
        n_sites=8,
        n_species=2,
        distribution="probit",
        seed=1350,
    )
    source_engine = NeuralHmscInference.for_fixed_effects(
        n_sites=8,
        n_species=2,
        distribution="probit",
    )
    source = source_engine.save(tmp_path / "source")
    source_weights_hash = _sha256(source / "weights.weights.h5")
    calibration = _external_monotone_calibration(n_covariates=3, n_species=2)
    packaged = package_neural_hmsc_coefficient_calibration(
        source,
        tmp_path / "packaged",
        calibration_metadata=calibration.to_metadata(),
        provenance=_calibration_provenance(seed=1350),
    )

    loaded = NeuralHmscInference.load(packaged)
    raw = loaded.predict_beta_posterior(dataset, calibrated=False)
    calibrated = loaded.predict_beta_posterior(dataset)
    data = fixed_shape_training_data([dataset])
    expected = apply_conditional_beta_scale_calibration(
        raw,
        calibration,
        X=data.X,
        Y=data.Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1", "x2"),
    )
    fit = loaded.infer(
        dataset,
        draws=8,
        seed=1351,
        output=tmp_path / "packaged_posterior.h5",
    )

    np.testing.assert_allclose(calibrated.mean.numpy(), expected.mean.numpy())
    np.testing.assert_allclose(calibrated.scale.numpy(), expected.scale.numpy())
    assert not np.allclose(calibrated.scale.numpy(), raw.scale.numpy())
    assert loaded.coefficient_calibration is not None
    assert loaded.coefficient_calibration.method == "external_context_monotone_scale"
    assert loaded.coefficient_calibration_record["parameter"] == "Beta"
    assert fit.beta_samples().shape == (1, 8, 3, 2)
    assert _sha256(packaged / "weights.weights.h5") == source_weights_hash


def test_public_neural_hmsc_rejects_tampered_calibration_artifact(tmp_path):
    engine = NeuralHmscInference.for_fixed_effects(
        n_sites=8,
        n_species=2,
        distribution="probit",
    )
    packaged = package_neural_hmsc_coefficient_calibration(
        engine.save(tmp_path / "source"),
        tmp_path / "packaged",
        calibration_metadata=_external_monotone_calibration(
            n_covariates=3, n_species=2
        ).to_metadata(),
        provenance=_calibration_provenance(seed=1360),
    )
    artifact = packaged / "coefficient_calibration.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["calibration"]["scale_multiplier"] = 99.0
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        NeuralHmscCompatibilityError, match="calibration artifact hash mismatch"
    ):
        NeuralHmscInference.load(packaged)


def test_public_neural_hmsc_rejects_calibration_domain_mismatch(tmp_path):
    engine = NeuralHmscInference.for_fixed_effects(
        n_sites=8,
        n_species=2,
        distribution="probit",
    )

    with pytest.raises(ValueError, match="species dimension mismatch"):
        package_neural_hmsc_coefficient_calibration(
            engine.save(tmp_path / "source"),
            tmp_path / "packaged",
            calibration_metadata=_external_monotone_calibration(
                n_covariates=3, n_species=3
            ).to_metadata(),
            provenance=_calibration_provenance(seed=1370),
        )


def test_public_neural_hmsc_rejects_unqualified_calibration_provenance(tmp_path):
    engine = NeuralHmscInference.for_fixed_effects(
        n_sites=8,
        n_species=2,
        distribution="probit",
    )
    provenance = _calibration_provenance(seed=1380)
    provenance["target_response_used_for_calibration"] = True

    with pytest.raises(
        ValueError,
        match="target_response_used_for_calibration mismatch",
    ):
        package_neural_hmsc_coefficient_calibration(
            engine.save(tmp_path / "source"),
            tmp_path / "packaged",
            calibration_metadata=_external_monotone_calibration(
                n_covariates=3, n_species=2
            ).to_metadata(),
            provenance=provenance,
        )


@pytest.mark.parametrize(
    ("distribution", "expected_family"),
    [
        ("normal", "diagonal_normal"),
        ("probit", "diagonal_normal"),
        ("poisson", "full_covariance_normal"),
    ],
)
def test_public_neural_hmsc_auto_family_is_distribution_aware(
    distribution, expected_family
):
    engine = NeuralHmscInference.for_fixed_effects(
        n_sites=4,
        n_species=1,
        distribution=distribution,
    )

    assert engine.model.posterior_family == expected_family


def test_public_neural_hmsc_rejects_unknown_posterior_family():
    with pytest.raises(ValueError, match="posterior_family must be"):
        NeuralHmscInference.for_fixed_effects(
            n_sites=4,
            n_species=1,
            posterior_family="unknown",
        )


def test_public_neural_hmsc_load_rejects_unknown_checkpoint_version(tmp_path):
    engine = NeuralHmscInference.for_fixed_effects(n_sites=4, n_species=1)
    checkpoint = engine.save(tmp_path / "checkpoint")
    manifest_path = checkpoint / "neural_checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_version"] = "99.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        NeuralHmscCompatibilityError, match="unsupported Neural-HMSC checkpoint version"
    ):
        NeuralHmscInference.load(checkpoint)


def test_public_neural_hmsc_rejects_negative_mse_weight():
    dataset = simulate_fixed_effect_dataset(
        n_sites=4,
        n_species=1,
        distribution="normal",
        seed=1400,
    )
    engine = NeuralHmscInference.for_fixed_effects(n_sites=4, n_species=1)

    with pytest.raises(ValueError, match="mse_weight must be non-negative"):
        engine.fit([dataset], epochs=1, batch_size=1, mse_weight=-1.0)


def _external_monotone_calibration(
    *, n_covariates: int, n_species: int
) -> ConditionalBetaScaleCalibration:
    raw_names = (
        "prevalence_logit",
        "log_design_information",
        "log_raw_scale",
    )
    feature_names = (
        tuple(
            name
            for raw_name in raw_names
            for name in (raw_name, f"{raw_name}_positive_hinge")
        )
        + tuple(f"coefficient_{index}" for index in range(n_covariates))
        + tuple(f"prevalence_by_coefficient_{index}" for index in range(n_covariates))
    )
    zero_matrix = tuple(tuple(0.0 for _ in range(n_covariates)) for _ in range(3))
    return ConditionalBetaScaleCalibration(
        global_scale_multiplier=2.0,
        normalization_multiplier=2.0,
        feature_location=(0.0, 0.0, 0.0),
        feature_scale=(1.0, 1.0, 1.0),
        weights=tuple(0.0 for _ in feature_names),
        feature_names=feature_names,
        coefficient_names=tuple(
            ["Intercept"] + [f"x{index}" for index in range(1, n_covariates)]
        ),
        nominal_level=0.95,
        uncalibrated_coverage=0.5,
        calibrated_coverage=0.95,
        n_observations=100,
        distribution="probit",
        n_covariates=n_covariates,
        n_species=n_species,
        regularization=0.0,
        epochs=1,
        learning_rate=0.1,
        scalar_nll=1.0,
        conditional_nll=0.9,
        mean_bias_correction=zero_matrix,
        rank_centering_offsets=zero_matrix,
        base_scale_stratum_offsets=tuple(0.0 for _ in range(6 + n_covariates)),
        fallback_strength=0.0,
        method="external_context_monotone_scale",
    )


def _calibration_provenance(*, seed: int) -> dict[str, object]:
    return {
        "kind": "independent_simulation_calibration_provenance",
        "training_corpus_version": "0.1",
        "calibration_training_role": "independent_simulation",
        "target_response_used_for_calibration": False,
        "packaging_refit_performed": False,
        "packaging_reselection_performed": False,
        "source_run_metadata_path": f"seed_{seed}/run_metadata.json",
        "source_run_metadata_sha256": "a" * 64,
        "source_seed": seed,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_neural_hmsc_rank_mean_penalty_records_history():
    datasets = [
        simulate_fixed_effect_dataset(
            n_sites=12,
            n_species=8,
            distribution="probit",
            seed=1500 + idx,
        )
        for idx in range(4)
    ]
    engine = NeuralHmscInference.for_fixed_effects(
        n_sites=12,
        n_species=8,
        distribution="probit",
    )

    history = engine.fit(
        datasets,
        epochs=2,
        batch_size=1,
        rank_mean_penalty_weight=0.01,
        rank_mean_penalty_holdout_fraction=0.25,
        rank_mean_penalty_start_fraction=0.5,
        rank_mean_penalty_design_guard_weight=0.05,
        rank_mean_penalty_design_guard_floor=0.9,
        rank_mean_penalty_signed_mean_weight=0.5,
        rank_mean_penalty_design_mean_guard_weight=0.25,
        rank_mean_penalty_design_mean_guard_tolerance=0.03,
    )

    assert history.rank_mean_penalty is not None
    assert len(history.rank_mean_penalty) == 2
    assert history.rank_mean_penalty[0] >= 0.0
    assert history.rank_mean_penalty[1] >= 0.0


def test_public_neural_hmsc_crossfit_rank_penalty_records_history():
    datasets = [
        simulate_fixed_effect_dataset(
            n_sites=12,
            n_species=8,
            distribution="probit",
            seed=1550 + idx,
        )
        for idx in range(8)
    ]
    engine = NeuralHmscInference.for_fixed_effects(
        n_sites=12,
        n_species=8,
        distribution="probit",
    )

    history = engine.fit(
        datasets,
        epochs=1,
        batch_size=1,
        rank_mean_penalty_weight=0.01,
        rank_mean_penalty_holdout_fraction=0.5,
        rank_mean_penalty_holdout_folds=2,
        rank_mean_penalty_crossfit_min_agreement=0.5,
        rank_mean_penalty_signed_mean_weight=0.1,
        rank_mean_penalty_design_mean_guard_weight=0.25,
    )

    assert history.rank_mean_penalty is not None
    assert len(history.rank_mean_penalty) == 1
    assert history.rank_mean_penalty[0] >= 0.0


def test_public_neural_hmsc_rejects_negative_rank_mean_penalty():
    dataset = simulate_fixed_effect_dataset(
        n_sites=4,
        n_species=1,
        distribution="normal",
        seed=1600,
    )
    engine = NeuralHmscInference.for_fixed_effects(n_sites=4, n_species=1)

    with pytest.raises(
        ValueError, match="rank_mean_penalty_weight must be non-negative"
    ):
        engine.fit([dataset], epochs=1, batch_size=1, rank_mean_penalty_weight=-1.0)


def test_public_neural_hmsc_rejects_invalid_rank_mean_penalty_schedule():
    dataset = simulate_fixed_effect_dataset(
        n_sites=4,
        n_species=1,
        distribution="normal",
        seed=1601,
    )
    engine = NeuralHmscInference.for_fixed_effects(n_sites=4, n_species=1)

    with pytest.raises(
        ValueError, match=r"rank_mean_penalty_start_fraction must be in \[0, 1\)"
    ):
        engine.fit(
            [dataset],
            epochs=1,
            batch_size=1,
            rank_mean_penalty_weight=0.01,
            rank_mean_penalty_start_fraction=1.0,
        )


def test_public_neural_hmsc_rejects_negative_rank_design_guard():
    dataset = simulate_fixed_effect_dataset(
        n_sites=4,
        n_species=1,
        distribution="normal",
        seed=1602,
    )
    engine = NeuralHmscInference.for_fixed_effects(n_sites=4, n_species=1)

    with pytest.raises(
        ValueError, match="rank_mean_penalty_design_guard_weight must be non-negative"
    ):
        engine.fit(
            [dataset],
            epochs=1,
            batch_size=1,
            rank_mean_penalty_weight=0.01,
            rank_mean_penalty_design_guard_weight=-0.1,
        )


def test_public_neural_hmsc_rejects_negative_signed_mean_penalty():
    dataset = simulate_fixed_effect_dataset(
        n_sites=4,
        n_species=1,
        distribution="normal",
        seed=1603,
    )
    engine = NeuralHmscInference.for_fixed_effects(n_sites=4, n_species=1)

    with pytest.raises(
        ValueError, match="rank_mean_penalty_signed_mean_weight must be non-negative"
    ):
        engine.fit(
            [dataset],
            epochs=1,
            batch_size=1,
            rank_mean_penalty_weight=0.01,
            rank_mean_penalty_signed_mean_weight=-0.1,
        )


def test_public_neural_hmsc_rejects_invalid_rank_holdout_folds():
    dataset = simulate_fixed_effect_dataset(
        n_sites=4,
        n_species=1,
        distribution="normal",
        seed=1604,
    )
    engine = NeuralHmscInference.for_fixed_effects(n_sites=4, n_species=1)

    with pytest.raises(
        ValueError, match="rank_mean_penalty_holdout_folds must be at least one"
    ):
        engine.fit(
            [dataset],
            epochs=1,
            batch_size=1,
            rank_mean_penalty_weight=0.01,
            rank_mean_penalty_holdout_folds=0,
        )
