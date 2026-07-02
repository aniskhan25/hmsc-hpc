import numpy as np
import pytest
import tensorflow as tf

from pyhmsc.neural.conditional_calibration import (
    ConditionalBetaScaleCalibration,
    apply_conditional_beta_scale_calibration,
    conditional_beta_scale_multipliers,
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
    assert restored.method == "conditional_structured_scale"
    assert restored.to_metadata()["semantics_version"] == 3


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
