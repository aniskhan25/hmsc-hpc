import numpy as np
import tensorflow as tf

from pyhmsc.neural.calibration import (
    apply_beta_scale_calibration,
    fit_beta_scale_calibration,
)
from pyhmsc.neural.posterior_heads import BetaPosterior
from pyhmsc.neural.storage import write_beta_posterior_hdf5
from pyhmsc.posterior import HmscFit


def test_beta_scale_calibration_rescales_uncertainty_without_changing_mean():
    posterior = BetaPosterior(
        mean=tf.constant([[[0.0, 0.0], [0.0, 0.0]]], dtype=tf.float32),
        scale=tf.constant([[[0.1, 0.1], [0.1, 0.1]]], dtype=tf.float32),
    )
    truth = np.array([[[0.05, 0.25], [0.4, 0.6]]], dtype=np.float32)

    calibration = fit_beta_scale_calibration(
        posterior,
        truth,
        nominal_level=0.95,
        distribution="normal",
    )
    calibrated = apply_beta_scale_calibration(
        posterior,
        calibration,
        distribution="normal",
    )

    assert calibration.method == "temperature_scale"
    assert calibration.scale_multiplier > 1.0
    assert calibration.calibrated_coverage >= calibration.uncalibrated_coverage
    np.testing.assert_allclose(calibrated.mean.numpy(), posterior.mean.numpy())
    np.testing.assert_allclose(
        calibrated.scale.numpy(),
        posterior.scale.numpy() * calibration.scale_multiplier,
    )


def test_beta_scale_calibration_rescales_full_covariance_cholesky():
    scale_tril = tf.constant([[[[0.2, 0.0], [0.1, 0.3]]]], dtype=tf.float32)
    marginal = tf.transpose(tf.sqrt(tf.reduce_sum(tf.square(scale_tril), axis=-1)), [0, 2, 1])
    posterior = BetaPosterior(
        mean=tf.zeros((1, 2, 1), dtype=tf.float32),
        scale=marginal,
        scale_tril=scale_tril,
    )
    calibration = fit_beta_scale_calibration(
        posterior,
        np.ones((1, 2, 1), dtype=np.float32),
        distribution="normal",
    )

    calibrated = apply_beta_scale_calibration(posterior, calibration, distribution="normal")

    np.testing.assert_allclose(
        calibrated.scale_tril.numpy(),
        scale_tril.numpy() * calibration.scale_multiplier,
    )


def test_beta_scale_calibration_rejects_out_of_domain_application():
    posterior = BetaPosterior(
        mean=tf.zeros((1, 2, 1), dtype=tf.float32),
        scale=tf.ones((1, 2, 1), dtype=tf.float32),
    )
    truth = np.zeros((1, 2, 1), dtype=np.float32)
    calibration = fit_beta_scale_calibration(
        posterior,
        truth,
        distribution="normal",
    )

    try:
        apply_beta_scale_calibration(posterior, calibration, distribution="poisson")
    except ValueError as exc:
        assert "distribution mismatch" in str(exc)
    else:
        raise AssertionError("Expected calibration domain mismatch")


def test_write_beta_posterior_hdf5_records_calibration_metadata(tmp_path):
    posterior = BetaPosterior(
        mean=tf.zeros((1, 2, 1), dtype=tf.float32),
        scale=tf.ones((1, 2, 1), dtype=tf.float32),
    )
    calibration = fit_beta_scale_calibration(
        posterior,
        np.zeros((1, 2, 1), dtype=np.float32),
        distribution="normal",
    )

    path = write_beta_posterior_hdf5(
        posterior,
        tmp_path / "calibrated.h5",
        covariate_names=["Intercept", "x1"],
        species_names=["sp1"],
        distribution="normal",
        chains=1,
        draws=3,
        seed=1,
        calibration=calibration,
    )
    fit = HmscFit.from_file(path)

    assert fit.metadata["calibration"]["method"] == "temperature_scale"
    assert fit.metadata["calibration"]["domain"]["distribution"] == "normal"
    assert fit.metadata["calibration"]["domain"]["n_covariates"] == 2
    assert fit.metadata["calibration"]["domain"]["n_species"] == 1


def test_poisson_calibration_selects_scale_by_predictive_log_score():
    batch = 20
    posterior = BetaPosterior(
        mean=tf.zeros((batch, 1, 1), dtype=tf.float32),
        scale=tf.ones((batch, 1, 1), dtype=tf.float32) * 0.5,
    )
    truth = np.zeros((batch, 1, 1), dtype=np.float32)
    truth[-2:, 0, 0] = 3.0
    X = np.ones((batch, 32, 1), dtype=np.float32)

    calibration = fit_beta_scale_calibration(
        posterior,
        truth,
        distribution="poisson",
        predictive_X=X,
        poisson_eta_clip=(-6.0, 6.0),
        predictive_draws=256,
        predictive_seed=7,
    )

    assert calibration.method == "poisson_balanced_score_scale"
    assert calibration.coverage_scale_multiplier > 1.0
    assert calibration.scale_multiplier < calibration.coverage_scale_multiplier
    assert calibration.predictive_score_calibrated <= 1.10 * calibration.predictive_score_uncalibrated
    assert calibration.predictive_rate_rmse_calibrated <= 1.25 * calibration.predictive_rate_rmse_uncalibrated
    assert calibration.calibrated_coverage < calibration.nominal_level
