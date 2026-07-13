import numpy as np
import pytest
import tensorflow as tf

from pyhmsc.neural.conditional_calibration import (
    ConditionalBetaOODCalibrationBatch,
    ConditionalBetaScaleCalibration,
    apply_conditional_beta_scale_calibration,
    conditional_beta_mean_support_diagnostics,
    conditional_beta_ood_uncertainty_inflation,
    conditional_beta_scale_multipliers,
    conditional_beta_support_trust,
    fit_conditional_beta_scale_calibration,
)
from pyhmsc.neural.posterior_heads import BetaPosterior


@pytest.fixture(scope="module")
def conditional_case():
    rng = np.random.default_rng(14)
    batch, sites, covariates, species = 48, 20, 2, 4
    X = np.ones((batch, sites, covariates), dtype=np.float32)
    X[:, :, 1] = rng.normal(size=(batch, sites))
    Y = np.zeros((batch, sites, species), dtype=np.float32)
    for species_index, prevalence in enumerate((0.05, 0.2, 0.6, 0.8)):
        Y[:, :, species_index] = rng.binomial(
            1, prevalence, size=(batch, sites)
        )
    mean = np.zeros((batch, covariates, species), dtype=np.float32)
    scale = np.full(mean.shape, 0.2, dtype=np.float32)
    error_scale = np.asarray((4.0, 3.0, 1.0, 1.0), dtype=np.float32)
    truth = scale * rng.normal(size=mean.shape) * error_scale[None, None, :]
    posterior = BetaPosterior(mean=tf.constant(mean), scale=tf.constant(scale))
    calibration = fit_conditional_beta_scale_calibration(
        posterior,
        truth,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
        epochs=80,
        learning_rate=0.03,
    )
    return posterior, truth, X, Y, calibration


def test_conditional_calibration_learns_prevalence_dependent_scale(conditional_case):
    posterior, _, X, Y, calibration = conditional_case

    multipliers = conditional_beta_scale_multipliers(
        posterior,
        calibration,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )
    species_multipliers = np.mean(multipliers, axis=(0, 1))

    assert calibration.calibrated_coverage == pytest.approx(0.95, abs=0.01)
    assert calibration.conditional_nll < calibration.scalar_nll
    assert calibration.conditional_rank_loss < calibration.scalar_rank_loss
    assert species_multipliers[0] > species_multipliers[-1]
    assert np.ptp(multipliers) > 0.5


def test_conditional_calibration_preserves_mean_and_round_trips_metadata(
    conditional_case,
):
    posterior, _, X, Y, calibration = conditional_case
    restored = ConditionalBetaScaleCalibration.from_metadata(
        calibration.to_metadata()
    )

    calibrated = apply_conditional_beta_scale_calibration(
        posterior,
        restored,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )
    original_multipliers = conditional_beta_scale_multipliers(
        posterior,
        calibration,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )
    restored_multipliers = conditional_beta_scale_multipliers(
        posterior,
        restored,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )

    np.testing.assert_allclose(calibrated.mean.numpy(), posterior.mean.numpy())
    np.testing.assert_allclose(original_multipliers, restored_multipliers)
    assert restored.method == "conditional_rank_aware_anchor_scale"
    assert restored.to_metadata()["semantics_version"] == 6
    assert restored.ood_uncertainty_strength > 0.0
    assert restored.ood_uncertainty_max_multiplier > 1.0


def test_conditional_calibration_can_fit_learned_ood_objective(conditional_case):
    posterior, truth, X, Y, _ = conditional_case
    shifted = BetaPosterior(
        mean=posterior.mean + 20.0,
        scale=posterior.scale,
    )

    calibration = fit_conditional_beta_scale_calibration(
        posterior,
        truth,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
        epochs=20,
        learning_rate=0.03,
        ood_uncertainty_max_multiplier=8.0,
        ood_calibration_batches=[
            ConditionalBetaOODCalibrationBatch(
                posterior=shifted,
                beta_true=truth,
                X=X,
                Y=Y,
                label="synthetic_shift",
            )
        ],
        ood_objective="support_excess_rank_coverage",
        ood_objective_epochs=20,
    )
    metadata = calibration.to_metadata()
    restored = ConditionalBetaScaleCalibration.from_metadata(metadata)
    inflation = conditional_beta_ood_uncertainty_inflation(
        shifted,
        restored,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )

    assert metadata["semantics_version"] == 7
    assert metadata["ood_objective"]["name"] == "support_excess_rank_coverage"
    assert metadata["ood_objective"]["domains"] == ["synthetic_shift"]
    assert metadata["support"]["ood_uncertainty"]["transform"] == (
        "support_effect_learned_softplus"
    )
    assert "support_linear" in metadata["support"]["ood_uncertainty"]["curve"]
    assert "effect_linear" in metadata["support"]["ood_uncertainty"]["curve"]
    assert restored.ood_inflation_parameters is not None
    assert len(restored.ood_inflation_parameters) == 5
    assert np.max(inflation) <= calibration.ood_uncertainty_max_multiplier
    assert np.mean(inflation) > 1.0


def test_conditional_calibration_loads_legacy_version_seven_curve(
    conditional_case,
):
    posterior, _, X, Y, calibration = conditional_case
    metadata = calibration.to_metadata()
    metadata["semantics_version"] = 7
    metadata["support"]["ood_uncertainty"]["transform"] = (
        "support_excess_learned_softplus"
    )
    metadata["support"]["ood_uncertainty"]["curve"] = {
        "offset": -3.0,
        "linear": 0.1,
        "quadratic": 0.5,
    }

    restored = ConditionalBetaScaleCalibration.from_metadata(metadata)
    shifted = BetaPosterior(mean=posterior.mean + 50.0, scale=posterior.scale)
    inflation = conditional_beta_ood_uncertainty_inflation(
        shifted,
        restored,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )

    assert restored.ood_inflation_parameters == pytest.approx((-3.0, 0.1, 0.5))
    assert restored.to_metadata()["support"]["ood_uncertainty"]["transform"] == (
        "support_excess_learned_softplus"
    )
    assert np.max(inflation) <= restored.ood_uncertainty_max_multiplier
    assert np.mean(inflation) > 1.0


def test_conditional_calibration_falls_back_to_scalar_outside_support(
    conditional_case,
):
    posterior, _, X, Y, calibration = conditional_case
    shifted = BetaPosterior(
        mean=posterior.mean,
        scale=posterior.scale * 100.0,
    )

    trust = conditional_beta_support_trust(
        shifted,
        calibration,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )
    multipliers = conditional_beta_scale_multipliers(
        shifted,
        calibration,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )
    inflation = conditional_beta_ood_uncertainty_inflation(
        shifted,
        calibration,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )
    assert np.max(trust) < 1e-6
    assert np.min(inflation) > 1.0
    assert np.max(inflation) <= calibration.ood_uncertainty_max_multiplier
    np.testing.assert_allclose(
        multipliers,
        calibration.global_scale_multiplier * inflation,
        rtol=1e-6,
        atol=1e-6,
    )


def test_conditional_calibration_detects_posterior_mean_shift(conditional_case):
    posterior, _, X, Y, calibration = conditional_case
    shifted = BetaPosterior(
        mean=posterior.mean + 50.0,
        scale=posterior.scale,
    )

    trust = conditional_beta_support_trust(
        shifted,
        calibration,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )
    multipliers = conditional_beta_scale_multipliers(
        shifted,
        calibration,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )
    inflation = conditional_beta_ood_uncertainty_inflation(
        shifted,
        calibration,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )
    diagnostics = conditional_beta_mean_support_diagnostics(shifted, calibration)

    assert np.max(trust) < 1e-6
    assert diagnostics["conditional_mean_magnitude_support_outside_fraction"] == 1.0
    assert diagnostics["conditional_mean_magnitude_support_max_abs_z"] > 1.0
    assert np.min(inflation) > 1.0
    np.testing.assert_allclose(
        multipliers,
        calibration.global_scale_multiplier * inflation,
        rtol=1e-6,
        atol=1e-6,
    )


def test_conditional_calibration_loads_version_three_metadata(conditional_case):
    _, _, _, _, calibration = conditional_case
    metadata = calibration.to_metadata()
    metadata["semantics_version"] = 3
    metadata["method"] = "conditional_structured_scale"
    metadata.pop("rank_aware")
    metadata.pop("support")
    metadata["training"].pop("scalar_rank_loss")
    metadata["training"].pop("conditional_rank_loss")

    restored = ConditionalBetaScaleCalibration.from_metadata(metadata)

    assert restored.method == "conditional_structured_scale"
    assert restored.rank_penalty_weight == 0.0
    assert restored.fallback_strength == 0.0


def test_conditional_calibration_loads_version_four_metadata(conditional_case):
    _, _, _, _, calibration = conditional_case
    metadata = calibration.to_metadata()
    metadata["semantics_version"] = 4
    metadata["method"] = "conditional_rank_aware_scale"
    metadata["support"].pop("mean_magnitude")
    metadata["support"].pop("ood_uncertainty")

    restored = ConditionalBetaScaleCalibration.from_metadata(metadata)

    assert restored.method == "conditional_rank_aware_scale"
    assert restored.mean_magnitude_lower == -1e9
    assert restored.mean_magnitude_upper == 1e9
    assert restored.ood_uncertainty_strength == 0.0


def test_conditional_calibration_loads_legacy_version_five_without_ood_inflation(
    conditional_case,
):
    posterior, _, X, Y, calibration = conditional_case
    metadata = calibration.to_metadata()
    metadata["semantics_version"] = 5
    metadata["support"].pop("ood_uncertainty")
    restored = ConditionalBetaScaleCalibration.from_metadata(metadata)
    shifted = BetaPosterior(mean=posterior.mean + 50.0, scale=posterior.scale)

    multipliers = conditional_beta_scale_multipliers(
        shifted,
        restored,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )
    inflation = conditional_beta_ood_uncertainty_inflation(
        shifted,
        restored,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )

    assert restored.to_metadata()["semantics_version"] == 5
    np.testing.assert_allclose(inflation, 1.0)
    np.testing.assert_allclose(
        multipliers,
        restored.global_scale_multiplier,
        rtol=1e-6,
        atol=1e-6,
    )


def test_conditional_calibration_applies_d_sigma_d_to_full_covariance(
    conditional_case,
):
    posterior, _, X, Y, calibration = conditional_case
    batch, _, species = posterior.mean.shape
    base_tril = np.asarray([[0.3, 0.0], [0.1, 0.2]], dtype=np.float32)
    scale_tril = np.broadcast_to(
        base_tril, (batch, species, 2, 2)
    ).copy()
    marginal = np.transpose(
        np.sqrt(np.sum(np.square(scale_tril), axis=-1)), (0, 2, 1)
    )
    full_posterior = BetaPosterior(
        mean=posterior.mean,
        scale=tf.constant(marginal),
        scale_tril=tf.constant(scale_tril),
    )
    multipliers = conditional_beta_scale_multipliers(
        full_posterior,
        calibration,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )

    calibrated = apply_conditional_beta_scale_calibration(
        full_posterior,
        calibration,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )

    original_covariance = scale_tril[0, 0] @ scale_tril[0, 0].T
    diagonal = np.diag(multipliers[0, :, 0])
    expected_covariance = diagonal @ original_covariance @ diagonal
    actual_tril = calibrated.scale_tril.numpy()[0, 0]
    np.testing.assert_allclose(
        actual_tril @ actual_tril.T, expected_covariance, rtol=1e-6, atol=1e-6
    )


def test_conditional_calibration_rejects_domain_mismatch(conditional_case):
    posterior, _, X, Y, calibration = conditional_case

    with pytest.raises(ValueError, match="distribution mismatch"):
        conditional_beta_scale_multipliers(
            posterior,
            calibration,
            X=X,
            Y=Y,
            distribution="poisson",
            coefficient_names=("Intercept", "x1"),
        )
    with pytest.raises(ValueError, match="coefficient names mismatch"):
        conditional_beta_scale_multipliers(
            posterior,
            calibration,
            X=X,
            Y=Y,
            distribution="probit",
            coefficient_names=("Intercept", "wrong"),
        )
