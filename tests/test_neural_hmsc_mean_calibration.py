import numpy as np
import tensorflow as tf

from pyhmsc.neural.mean_calibration import (
    BetaPredictiveMeanCalibration,
    BetaResponseCalibrationBatch,
    apply_beta_predictive_mean_calibration,
    domain_conditional_predictive_mean_selector_metadata,
    evaluate_beta_target_context_gate,
    fit_beta_predictive_mean_calibration,
    fit_beta_response_mean_calibration,
    fit_beta_transfer_response_branch_calibration,
    fit_beta_transfer_response_mean_calibration,
    independent_source_transfer_predictive_mean_selector_metadata,
    select_predictive_mean_calibration_for_context,
    target_context_conditioned_source_transfer_selector_metadata,
)
from pyhmsc.neural.posterior_heads import BetaPosterior


def test_predictive_mean_calibration_selects_validated_affine_correction():
    calibration_mean = np.array(
        [[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]],
        dtype=np.float32,
    )
    calibration_truth = 0.5 + 1.2 * calibration_mean
    validation_mean = np.array(
        [[[0.5, 1.5], [2.5, 3.5]], [[1.5, 2.5], [3.5, 4.5]]],
        dtype=np.float32,
    )
    validation_truth = 0.5 + 1.2 * validation_mean
    posterior = BetaPosterior(
        mean=tf.constant(calibration_mean),
        scale=tf.ones(calibration_mean.shape, dtype=tf.float32),
    )
    validation_posterior = BetaPosterior(
        mean=tf.constant(validation_mean),
        scale=tf.ones(validation_mean.shape, dtype=tf.float32),
    )

    calibration = fit_beta_predictive_mean_calibration(
        posterior,
        calibration_truth,
        validation_posterior=validation_posterior,
        validation_beta_true=validation_truth,
        distribution="probit",
        min_validation_rmse_improvement=0.01,
    )
    calibrated = apply_beta_predictive_mean_calibration(
        validation_posterior, calibration, distribution="probit"
    )

    assert calibration.method == "affine_shrinkage"
    assert calibration.selected
    assert calibration.validation_rmse_ratio < 1.0
    np.testing.assert_allclose(calibration.slope, 1.2, rtol=1e-6)
    np.testing.assert_allclose(calibration.intercept, 0.5, rtol=1e-6)
    np.testing.assert_allclose(calibrated.mean.numpy(), validation_truth, rtol=1e-6)
    np.testing.assert_allclose(
        calibrated.scale.numpy(), validation_posterior.scale.numpy()
    )


def test_predictive_mean_calibration_falls_back_when_validation_fails():
    calibration_mean = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    calibration_truth = 2.0 * calibration_mean
    validation_truth = np.array(calibration_mean, copy=True)
    posterior = BetaPosterior(
        mean=tf.constant(calibration_mean),
        scale=tf.ones(calibration_mean.shape, dtype=tf.float32),
    )

    calibration = fit_beta_predictive_mean_calibration(
        posterior,
        calibration_truth,
        validation_posterior=posterior,
        validation_beta_true=validation_truth,
        distribution="normal",
        min_validation_rmse_improvement=0.01,
    )
    calibrated = apply_beta_predictive_mean_calibration(
        posterior, calibration, distribution="normal"
    )

    assert not calibration.selected
    assert calibration.slope == 1.0
    assert calibration.intercept == 0.0
    assert calibration.validation_rmse_ratio == 1.0
    np.testing.assert_allclose(calibrated.mean.numpy(), calibration_mean)


def test_predictive_mean_calibration_round_trips_metadata_and_checks_domain():
    calibration = BetaPredictiveMeanCalibration(
        slope=0.9,
        intercept=0.1,
        method="affine_shrinkage",
        distribution="poisson",
        n_covariates=2,
        n_species=3,
        calibration_rmse_uncalibrated=1.0,
        calibration_rmse_calibrated=0.8,
        validation_rmse_uncalibrated=1.2,
        validation_rmse_calibrated=1.0,
        validation_rmse_ratio=0.833333,
        selected=True,
        max_validation_rmse_ratio=1.0,
        min_validation_rmse_improvement=0.01,
        n_calibration_observations=24,
        n_validation_observations=24,
    )

    restored = BetaPredictiveMeanCalibration.from_metadata(
        calibration.to_metadata()
    )

    assert restored == calibration
    posterior = BetaPosterior(
        mean=tf.zeros((1, 2, 3), dtype=tf.float32),
        scale=tf.ones((1, 2, 3), dtype=tf.float32),
    )
    apply_beta_predictive_mean_calibration(
        posterior, restored, distribution="poisson"
    )
    try:
        apply_beta_predictive_mean_calibration(
            posterior, restored, distribution="probit"
        )
    except ValueError as exc:
        assert "distribution mismatch" in str(exc)
    else:
        raise AssertionError("Expected predictive mean calibration domain mismatch")


def test_probit_response_mean_calibration_selects_on_heldout_scores():
    mean = np.full((2, 1, 1), 0.4, dtype=np.float32)
    scale = np.full_like(mean, 0.01)
    posterior = BetaPosterior(
        mean=tf.constant(mean),
        scale=tf.constant(scale),
    )
    X = np.ones((2, 20, 1), dtype=np.float32)
    Y = np.zeros((2, 20, 1), dtype=np.float32)

    calibration = fit_beta_response_mean_calibration(
        posterior,
        calibration_X=X,
        calibration_Y=Y,
        validation_posterior=posterior,
        validation_X=X,
        validation_Y=Y,
        distribution="probit",
        slope_grid=np.array([1.0]),
        intercept_grid=np.array([-0.4, 0.0]),
        min_validation_score_improvement=0.001,
    )
    calibrated = apply_beta_predictive_mean_calibration(
        posterior, calibration, distribution="probit"
    )

    assert calibration.method == "probit_response_affine"
    assert calibration.selected
    assert calibration.intercept == -0.4
    assert calibration.validation_brier_ratio < 1.0
    assert calibration.validation_log_loss_ratio < 1.0
    np.testing.assert_allclose(calibrated.mean.numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(calibrated.scale.numpy(), scale)
    metadata = calibration.to_metadata()
    assert metadata["response_validation"]["brier_ratio"] < 1.0
    assert (
        BetaPredictiveMeanCalibration.from_metadata(metadata)
        == calibration
    )


def test_probit_response_mean_calibration_falls_back_on_validation_degradation():
    mean = np.full((2, 1, 1), 0.4, dtype=np.float32)
    posterior = BetaPosterior(
        mean=tf.constant(mean),
        scale=tf.ones(mean.shape, dtype=tf.float32) * 0.01,
    )
    X = np.ones((2, 20, 1), dtype=np.float32)
    calibration_Y = np.zeros((2, 20, 1), dtype=np.float32)
    validation_Y = np.ones((2, 20, 1), dtype=np.float32)

    calibration = fit_beta_response_mean_calibration(
        posterior,
        calibration_X=X,
        calibration_Y=calibration_Y,
        validation_posterior=posterior,
        validation_X=X,
        validation_Y=validation_Y,
        distribution="probit",
        slope_grid=np.array([1.0]),
        intercept_grid=np.array([-0.4, 0.0]),
        min_validation_score_improvement=0.001,
    )

    assert not calibration.selected
    assert calibration.slope == 1.0
    assert calibration.intercept == 0.0
    assert calibration.validation_brier_ratio == 1.0
    assert calibration.validation_log_loss_ratio == 1.0


def test_transfer_response_mean_calibration_selects_transfer_gain():
    mean = np.zeros((1, 1, 1), dtype=np.float32)
    posterior = BetaPosterior(
        mean=tf.constant(mean),
        scale=tf.ones(mean.shape, dtype=tf.float32) * 0.01,
    )
    X = np.ones((1, 60, 1), dtype=np.float32)
    source_Y = np.ones((1, 60, 1), dtype=np.float32)
    transfer_Y = np.ones((1, 60, 1), dtype=np.float32)

    calibration = fit_beta_transfer_response_mean_calibration(
        posterior,
        calibration_X=X,
        calibration_Y=source_Y,
        source_validation_posterior=posterior,
        source_validation_X=X,
        source_validation_Y=source_Y,
        transfer_validation_batches=[
            BetaResponseCalibrationBatch(
                posterior=posterior,
                X=X,
                Y=transfer_Y,
                label="transfer_validation:effect_size_shift",
            )
        ],
        distribution="probit",
        slope_grid=np.array([1.0]),
        intercept_grid=np.array([0.0, 0.2]),
        min_transfer_validation_score_improvement=0.001,
    )

    assert calibration.selected
    assert calibration.method == "probit_transfer_response_affine"
    assert calibration.intercept == 0.2
    assert calibration.validation_brier_ratio < 1.0
    assert calibration.transfer_validation_brier_ratio < 1.0
    metadata = calibration.to_metadata()
    assert metadata["transfer_response_validation"]["labels"] == [
        "transfer_validation:effect_size_shift"
    ]
    assert (
        BetaPredictiveMeanCalibration.from_metadata(metadata)
        == calibration
    )


def test_transfer_response_mean_calibration_blocks_source_degradation():
    mean = np.zeros((1, 1, 1), dtype=np.float32)
    posterior = BetaPosterior(
        mean=tf.constant(mean),
        scale=tf.ones(mean.shape, dtype=tf.float32) * 0.01,
    )
    X = np.ones((1, 60, 1), dtype=np.float32)
    calibration_Y = np.ones((1, 60, 1), dtype=np.float32)
    source_validation_Y = np.zeros((1, 60, 1), dtype=np.float32)
    transfer_Y = np.ones((1, 60, 1), dtype=np.float32)

    calibration = fit_beta_transfer_response_mean_calibration(
        posterior,
        calibration_X=X,
        calibration_Y=calibration_Y,
        source_validation_posterior=posterior,
        source_validation_X=X,
        source_validation_Y=source_validation_Y,
        transfer_validation_batches=[
            BetaResponseCalibrationBatch(
                posterior=posterior,
                X=X,
                Y=transfer_Y,
                label="transfer_validation:combined_shift",
            )
        ],
        distribution="probit",
        slope_grid=np.array([1.0]),
        intercept_grid=np.array([0.0, 0.2]),
        min_transfer_validation_score_improvement=0.001,
    )

    assert not calibration.selected
    assert calibration.slope == 1.0
    assert calibration.intercept == 0.0
    assert calibration.transfer_validation_brier_ratio == 1.0
    assert calibration.validation_brier_ratio == 1.0


def test_transfer_branch_uses_independent_validation_pool():
    mean = np.zeros((1, 1, 1), dtype=np.float32)
    posterior = BetaPosterior(
        mean=tf.constant(mean),
        scale=tf.ones(mean.shape, dtype=tf.float32) * 0.01,
    )
    X = np.ones((1, 60, 1), dtype=np.float32)
    calibration_batch = BetaResponseCalibrationBatch(
        posterior=posterior,
        X=X,
        Y=np.ones((1, 60, 1), dtype=np.float32),
        label="transfer_calibration:effect_size_shift",
    )
    validation_batch = BetaResponseCalibrationBatch(
        posterior=posterior,
        X=X,
        Y=np.zeros((1, 60, 1), dtype=np.float32),
        label="transfer_validation:effect_size_shift",
    )

    calibration = fit_beta_transfer_response_branch_calibration(
        [calibration_batch],
        validation_batches=[validation_batch],
        distribution="probit",
        slope_grid=np.array([1.0]),
        intercept_grid=np.array([0.0, 0.2]),
        min_validation_score_improvement=0.001,
    )

    assert not calibration.selected
    assert calibration.slope == 1.0
    assert calibration.intercept == 0.0
    assert calibration.transfer_validation_labels == (
        "transfer_validation:effect_size_shift",
    )


def test_independent_branch_selector_chooses_predeclared_context_branch():
    source = BetaPredictiveMeanCalibration.identity(
        distribution="probit",
        n_covariates=1,
        n_species=1,
        method="probit_response_affine",
    )
    transfer = _response_affine_calibration(
        slope=1.025,
        intercept=0.02,
        brier_gain=0.002,
        log_loss_gain=0.003,
    )
    selector = independent_source_transfer_predictive_mean_selector_metadata(
        source,
        transfer,
    )

    whittaker, source_decision = select_predictive_mean_calibration_for_context(
        selector,
        context="whittaker",
        distribution="probit",
        n_covariates=1,
        n_species=1,
    )
    big_spatial, transfer_decision = (
        select_predictive_mean_calibration_for_context(
            selector,
            context="big_spatial_transfer",
            distribution="probit",
            n_covariates=1,
            n_species=1,
        )
    )

    assert whittaker is None
    assert source_decision["branch"] == "source_branch"
    assert source_decision["reason"] == "branch_not_selected_on_independent_simulation"
    assert big_spatial == transfer
    assert transfer_decision["branch"] == "transfer_branch"
    assert transfer_decision["action"] == "apply_candidate"


def test_target_context_gate_requires_both_independent_pools():
    posterior = BetaPosterior(
        mean=tf.zeros((1, 1, 1), dtype=tf.float32),
        scale=tf.ones((1, 1, 1), dtype=tf.float32) * 0.01,
    )
    X = np.ones((1, 80, 1), dtype=np.float32)
    positive = BetaResponseCalibrationBatch(
        posterior=posterior,
        X=X,
        Y=np.ones((1, 80, 1), dtype=np.float32),
        label="target_context:positive",
    )
    negative = BetaResponseCalibrationBatch(
        posterior=posterior,
        X=X,
        Y=np.zeros((1, 80, 1), dtype=np.float32),
        label="target_context:negative",
    )
    transfer = _response_affine_calibration(
        slope=1.0,
        intercept=0.2,
        brier_gain=0.002,
        log_loss_gain=0.003,
    )

    passed = evaluate_beta_target_context_gate(
        transfer,
        [positive],
        validation_batches=[positive],
        min_score_improvement=0.001,
        context_metadata={"target_responses_used": False},
    )
    failed = evaluate_beta_target_context_gate(
        transfer,
        [positive],
        validation_batches=[negative],
        min_score_improvement=0.001,
    )

    assert passed["passed"]
    assert passed["target_calibration"]["passed"]
    assert passed["target_validation"]["passed"]
    assert not passed["target_responses_used"]
    assert not failed["passed"]
    assert any(
        reason.startswith("target_validation:")
        for reason in failed["failure_reasons"]
    )


def test_target_context_selector_requires_generic_and_target_gates():
    source = BetaPredictiveMeanCalibration.identity(
        distribution="probit",
        n_covariates=1,
        n_species=1,
        method="probit_response_affine",
    )
    transfer = _response_affine_calibration(
        slope=1.025,
        intercept=0.02,
        brier_gain=0.002,
        log_loss_gain=0.003,
    )
    base = independent_source_transfer_predictive_mean_selector_metadata(
        source,
        transfer,
    )
    gate = {
        "kind": "target_context_independent_simulation_gate",
        "passed": False,
        "failure_reasons": ["target_validation:brier_ratio_above_limit"],
    }
    selector = target_context_conditioned_source_transfer_selector_metadata(
        base,
        gate,
    )

    rejected, rejected_decision = select_predictive_mean_calibration_for_context(
        selector,
        context="big_spatial_transfer",
        distribution="probit",
        n_covariates=1,
        n_species=1,
    )
    selector["target_context_gate"] = {**gate, "passed": True}
    accepted, accepted_decision = select_predictive_mean_calibration_for_context(
        selector,
        context="big_spatial_transfer",
        distribution="probit",
        n_covariates=1,
        n_species=1,
    )

    assert rejected is None
    assert rejected_decision["reason"] == "target_context_simulation_gate_failed"
    assert accepted == transfer
    assert accepted_decision["reason"] == (
        "generic_and_target_context_simulation_gates_passed"
    )


def test_domain_conditional_selector_falls_back_for_source_context():
    calibration = _response_affine_calibration(
        slope=1.025,
        intercept=0.02,
        brier_gain=0.002,
        log_loss_gain=0.003,
    )
    selector = domain_conditional_predictive_mean_selector_metadata(calibration)

    selected, decision = select_predictive_mean_calibration_for_context(
        selector,
        context="whittaker",
        distribution="probit",
        n_covariates=1,
        n_species=1,
    )

    assert selected is None
    assert not decision["selected"]
    assert decision["context_family"] == "source_like"
    assert decision["action"] == "identity"


def test_domain_conditional_selector_applies_for_transfer_context():
    calibration = _response_affine_calibration(
        slope=1.025,
        intercept=0.02,
        brier_gain=0.002,
        log_loss_gain=0.003,
    )
    selector = domain_conditional_predictive_mean_selector_metadata(calibration)

    selected, decision = select_predictive_mean_calibration_for_context(
        selector,
        context="big_spatial_transfer",
        distribution="probit",
        n_covariates=1,
        n_species=1,
    )

    assert selected == calibration
    assert decision["selected"]
    assert decision["context_family"] == "transfer_like"
    assert decision["action"] == "apply_candidate"
    assert decision["transfer_stability_guard"]["passed"]


def test_domain_conditional_selector_blocks_weak_transfer_gain():
    calibration = _response_affine_calibration(
        slope=1.025,
        intercept=0.02,
        brier_gain=0.00002,
        log_loss_gain=0.0002,
    )
    selector = domain_conditional_predictive_mean_selector_metadata(calibration)

    selected, decision = select_predictive_mean_calibration_for_context(
        selector,
        context="big_spatial_transfer",
        distribution="probit",
        n_covariates=1,
        n_species=1,
    )

    assert selected is None
    assert not decision["selected"]
    assert decision["action"] == "identity"
    assert decision["reason"] == "candidate_failed_transfer_stability_guard"
    failures = decision["transfer_stability_guard"]["failure_reasons"]
    assert any("validation_brier_gain_below_margin" in item for item in failures)
    assert any("validation_log_loss_gain_below_margin" in item for item in failures)


def test_domain_conditional_selector_blocks_large_transfer_movement():
    calibration = _response_affine_calibration(
        slope=1.025,
        intercept=0.05,
        brier_gain=0.002,
        log_loss_gain=0.003,
    )
    selector = domain_conditional_predictive_mean_selector_metadata(calibration)

    selected, decision = select_predictive_mean_calibration_for_context(
        selector,
        context="big_spatial_transfer",
        distribution="probit",
        n_covariates=1,
        n_species=1,
    )

    assert selected is None
    assert not decision["selected"]
    failures = decision["transfer_stability_guard"]["failure_reasons"]
    assert any("abs_intercept_above_cap" in item for item in failures)


def _response_affine_calibration(
    *,
    slope: float,
    intercept: float,
    brier_gain: float,
    log_loss_gain: float,
) -> BetaPredictiveMeanCalibration:
    calibration = BetaPredictiveMeanCalibration.identity(
        distribution="probit",
        n_covariates=1,
        n_species=1,
        method="probit_response_affine",
    )
    return BetaPredictiveMeanCalibration(
        **{
            **calibration.__dict__,
            "slope": slope,
            "intercept": intercept,
            "selected": True,
            "validation_brier_uncalibrated": 0.1,
            "validation_brier_calibrated": 0.1 - brier_gain,
            "validation_log_loss_uncalibrated": 0.3,
            "validation_log_loss_calibrated": 0.3 - log_loss_gain,
        }
    )
