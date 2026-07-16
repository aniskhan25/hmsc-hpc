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
    fit_external_context_monotone_calibration,
    _in_domain_gate_group_masks,
    _learned_ood_log_inflation_numpy,
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
        Y[:, :, species_index] = rng.binomial(1, prevalence, size=(batch, sites))
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


def test_in_domain_gate_groups_include_design_and_coefficient_strata():
    prevalence = np.asarray([0.05, 0.2, 0.6, 0.05, 0.2, 0.6, 0.05, 0.2, 0.6])
    design_information = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    coefficient_index = np.asarray([0, 1, 2, 1, 2, 0, 2, 0, 1])

    groups = _in_domain_gate_group_masks(
        prevalence=prevalence,
        log_design_information=design_information,
        coefficient_index=coefficient_index,
        prevalence_edges=(0.1, 0.3),
    )

    assert len(groups) >= 10
    assert any(np.array_equal(group, coefficient_index == 0) for group in groups)
    assert any(np.array_equal(group, coefficient_index == 1) for group in groups)
    assert any(np.array_equal(group, coefficient_index == 2) for group in groups)


def test_conditional_calibration_preserves_mean_and_round_trips_metadata(
    conditional_case,
):
    posterior, _, X, Y, calibration = conditional_case
    restored = ConditionalBetaScaleCalibration.from_metadata(calibration.to_metadata())

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


def test_conditional_calibration_rare_balanced_head_adjusts_rare_mean():
    rng = np.random.default_rng(141)
    batch, sites, covariates, species = 24, 20, 2, 3
    X = np.ones((batch, sites, covariates), dtype=np.float32)
    X[:, :, 1] = rng.normal(size=(batch, sites))
    Y = np.zeros((batch, sites, species), dtype=np.float32)
    for species_index, prevalence in enumerate((0.05, 0.25, 0.7)):
        Y[:, :, species_index] = rng.binomial(1, prevalence, size=(batch, sites))
    mean = np.zeros((batch, covariates, species), dtype=np.float32)
    scale = np.full(mean.shape, 0.2, dtype=np.float32)
    truth = np.zeros_like(mean)
    truth[:, :, 0] = 0.15
    posterior = BetaPosterior(mean=tf.constant(mean), scale=tf.constant(scale))
    rare_batch = ConditionalBetaOODCalibrationBatch(
        posterior=posterior,
        beta_true=truth,
        X=X,
        Y=Y,
        label="rare_balanced",
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
        rare_calibration_batches=[rare_batch],
        rare_validation_batches=[rare_batch],
    )
    metadata = calibration.to_metadata()["mean_bias_correction"]

    assert metadata["rare_balanced_n_observations"] > 0
    assert metadata["rare_balanced_selected_shrinkage"] > 0.0
    assert np.any(np.asarray(metadata["values"])[0] > 0.0)
    diagnostics = metadata["rare_balanced_diagnostics"]
    assert diagnostics["rare_pool"]["n_observations"] > 0
    assert diagnostics["validation"]["n_rare_observations"] > 0
    assert len(diagnostics["shrinkage_grid"]) >= 2
    assert len(diagnostics["candidate_offsets"]) == 3
    assert diagnostics["rare_pool"]["rare_observations_by_design_stratum"]
    assert diagnostics["rare_pool_by_cell"]
    assert diagnostics["validation"]["independent"]["n_rare_observations"] > 0

    calibrated = apply_conditional_beta_scale_calibration(
        posterior,
        calibration,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )

    assert float(tf.reduce_mean(calibrated.mean[:, :, 0]).numpy()) > 0.0

    restored = ConditionalBetaScaleCalibration.from_metadata(calibration.to_metadata())
    restored_diagnostics = restored.to_metadata()["mean_bias_correction"][
        "rare_balanced_diagnostics"
    ]
    assert restored_diagnostics["rare_pool"]["n_observations"] > 0


def test_conditional_calibration_rare_validation_scale_inflates_undercoverage():
    rng = np.random.default_rng(242)
    batch, sites, covariates, species = 24, 18, 2, 3
    X = np.ones((batch, sites, covariates), dtype=np.float32)
    X[:, :, 1] = rng.normal(size=(batch, sites))
    Y = np.zeros((batch, sites, species), dtype=np.float32)
    for species_index, prevalence in enumerate((0.05, 0.25, 0.7)):
        Y[:, :, species_index] = rng.binomial(1, prevalence, size=(batch, sites))
    mean = np.zeros((batch, covariates, species), dtype=np.float32)
    scale = np.full(mean.shape, 0.25, dtype=np.float32)
    truth = np.zeros_like(mean)
    posterior = BetaPosterior(mean=tf.constant(mean), scale=tf.constant(scale))

    validation_scale = np.full(mean.shape, 0.08, dtype=np.float32)
    validation_truth = np.full(mean.shape, 0.04, dtype=np.float32)
    validation_posterior = BetaPosterior(
        mean=tf.constant(mean),
        scale=tf.constant(validation_scale),
    )
    rare_validation_batch = ConditionalBetaOODCalibrationBatch(
        posterior=validation_posterior,
        beta_true=validation_truth,
        X=X,
        Y=Y,
        label="rare_validation:low_detection",
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
        rare_validation_batches=[rare_validation_batch],
    )
    metadata = calibration.to_metadata()["rare_validation_scale"]

    assert metadata["selected_shrinkage"] > 0.0
    assert max(metadata["multipliers"]) > 1.0
    assert (
        metadata["diagnostics"]["best_metrics"]["overall_coverage"]
        >= metadata["diagnostics"]["coverage_floor"]
    )

    base_calibration = ConditionalBetaScaleCalibration.from_metadata(
        {
            **calibration.to_metadata(),
            "rare_validation_scale": {
                **metadata,
                "selected_shrinkage": 0.0,
                "log_offsets": [0.0, 0.0, 0.0],
            },
        }
    )
    inflated = conditional_beta_scale_multipliers(
        validation_posterior,
        calibration,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )
    baseline = conditional_beta_scale_multipliers(
        validation_posterior,
        base_calibration,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )

    assert float(np.mean(inflated)) > float(np.mean(baseline))


def test_rare_validation_scale_small_pool_falls_back_to_identity():
    rng = np.random.default_rng(243)
    batch, sites, covariates, species = 1, 6, 2, 2
    X = np.ones((batch, sites, covariates), dtype=np.float32)
    X[:, :, 1] = rng.normal(size=(batch, sites))
    Y = rng.binomial(1, 0.2, size=(batch, sites, species)).astype(np.float32)
    mean = np.zeros((batch, covariates, species), dtype=np.float32)
    scale = np.full(mean.shape, 0.25, dtype=np.float32)
    truth = np.zeros_like(mean)
    posterior = BetaPosterior(mean=tf.constant(mean), scale=tf.constant(scale))
    rare_validation_batch = ConditionalBetaOODCalibrationBatch(
        posterior=posterior,
        beta_true=truth,
        X=X,
        Y=Y,
        label="rare_validation:small_sample",
    )

    calibration = fit_conditional_beta_scale_calibration(
        posterior,
        truth,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
        epochs=2,
        learning_rate=0.03,
        rare_validation_batches=[rare_validation_batch],
    )
    metadata = calibration.to_metadata()["rare_validation_scale"]
    multipliers = conditional_beta_scale_multipliers(
        posterior,
        calibration,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )

    assert metadata["selected_shrinkage"] == 0.0
    assert metadata["activation"]["threshold"] == 0.0
    assert metadata["activation"]["width"] == 1.0
    assert metadata["activation"]["community_occupancy_threshold"] == 0.0
    assert metadata["activation"]["community_occupancy_width"] == 1.0
    assert metadata["diagnostics"]["reason"] == "insufficient_validation_observations"
    assert np.all(np.isfinite(multipliers))


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


def test_external_context_monotone_calibration_selects_guarded_offsets(
    conditional_case,
):
    posterior, truth, X, Y, base_calibration = conditional_case
    shifted = BetaPosterior(
        mean=posterior.mean + 20.0,
        scale=posterior.scale,
    )

    calibration = fit_external_context_monotone_calibration(
        base_calibration,
        posterior,
        truth,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
        ood_validation_batches=[
            ConditionalBetaOODCalibrationBatch(
                posterior=shifted,
                beta_true=truth,
                X=X,
                Y=Y,
                label="combined_shift",
            )
        ],
        max_external_multiplier=10.0,
        min_mean_ood_gain=0.001,
        min_combined_shift_gain=0.001,
    )
    metadata = calibration.to_metadata()
    restored = ConditionalBetaScaleCalibration.from_metadata(metadata)
    baseline = conditional_beta_scale_multipliers(
        shifted,
        base_calibration,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )
    inflated = conditional_beta_scale_multipliers(
        shifted,
        restored,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )

    assert metadata["semantics_version"] == 9
    assert restored.method == "external_context_monotone_scale"
    assert restored.external_monotone_selected_shrinkage > 0.0
    assert metadata["external_context_monotone"]["kind"] == (
        "heldout_context_stratified_monotone_scale"
    )
    assert metadata["external_context_monotone"]["diagnostics"]["selected"] == (
        "external_monotone"
    )
    assert list(restored.external_monotone_log_offsets) == sorted(
        restored.external_monotone_log_offsets
    )
    assert float(np.mean(inflated)) > float(np.mean(baseline))


def test_conditional_calibration_can_fit_gated_effect_ood_objective(
    conditional_case,
):
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
                label="effect_size_shift",
            ),
            ConditionalBetaOODCalibrationBatch(
                posterior=shifted,
                beta_true=truth,
                X=X,
                Y=Y,
                label="combined_shift",
            ),
        ],
        ood_objective="support_effect_gated_rank_coverage",
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

    assert metadata["semantics_version"] == 8
    assert metadata["ood_objective"]["name"] == "support_effect_gated_rank_coverage"
    assert metadata["support"]["ood_uncertainty"]["transform"] == (
        "support_effect_gated_learned_softplus"
    )
    curve = metadata["support"]["ood_uncertainty"]["curve"]
    assert "effect_gate_intercept" in curve
    assert "effect_gate_support_linear" in curve
    assert "effect_gate_effect_linear" in curve
    assert "effect_high_design_suppression" in curve
    assert len(curve["effect_prevalence_gate_offsets"]) == 3
    assert len(curve["effect_design_gate_offsets"]) == 3
    assert len(curve["effect_coefficient_gate_offsets"]) == posterior.mean.shape[1]
    assert curve["effect_shift_head"]["kind"] == (
        "constrained_context_gated_effect_quantile_scale"
    )
    assert curve["effect_shift_head"]["pure_log_cap"] > 0.0
    assert curve["effect_shift_head"]["combined_log_cap"] > 0.0
    assert curve["effect_shift_head"]["parameter_count"] == 14
    assert len(curve["effect_shift_head"]["effect_bin_centers"]) == 3
    assert len(curve["effect_shift_head"]["pure_effect_bin_log_amplitudes"]) == 3
    assert len(curve["effect_shift_head"]["combined_effect_bin_log_amplitudes"]) == 3
    base_strata = metadata["base_scale_strata"]
    assert len(base_strata["prevalence_offsets"]) == 3
    assert len(base_strata["design_offsets"]) == 3
    assert len(base_strata["coefficient_offsets"]) == posterior.mean.shape[1]
    assert len(metadata["mean_bias_correction"]["values"]) == 3
    assert len(metadata["rank_centering"]["values"]) == 3
    assert len(metadata["rank_centering"]["values"][0]) == posterior.mean.shape[1]
    diagnostics = metadata["ood_objective"]["final_multiplier_diagnostics"]
    assert diagnostics["kind"] == "post_scale_final_multiplier_ood_diagnostics"
    assert diagnostics["domains"][0]["label"] == "effect_size_shift"
    assert "effect_gate_activation" in diagnostics["domains"][0]
    assert "learned_ood_inflation" in diagnostics["domains"][0]
    assert "rare_post_scale_multiplier" in diagnostics["domains"][0]
    assert "combined_shift_scale_multiplier" in diagnostics["domains"][0]
    assert "final_multiplier" in diagnostics["domains"][0]
    assert diagnostics["domains"][0]["effect_size_quantile_coverage"]
    assert "in_domain_gate" in diagnostics["domains"][0]
    assert "learned_combined_shift_context" in diagnostics["domains"][0]
    combined_training = metadata["ood_objective"]["combined_shift_training_objective"]
    assert combined_training["kind"] == "final_multiplier_aware_combined_shift_coverage"
    assert combined_training["coverage_weight"] > 0.0
    assert combined_training["schedule"]["kind"] == "coverage_warmup_then_overlap_ramp"
    selection = diagnostics["effect_shift_head_selection"]
    assert selection["kind"] == "post_fit_independent_effect_shift_head_selection"
    assert selection["selected"]["pure_shrinkage"] >= 0.0
    assert selection["selected"]["combined_shrinkage"] >= 0.0
    assert "pure_effect_accepted" in selection["selected"]
    assert "combined_shift_accepted" in selection["selected"]
    assert "effect_size_shift" in selection["baseline"]["domain_coverages"]
    assert "combined_shift" in selection["baseline"]["domain_coverages"]
    assert {candidate["branch"] for candidate in selection["candidates"]} == {
        "pure_effect",
        "combined_shift",
    }
    expert_selection = diagnostics["domain_expert_selection"]
    assert expert_selection["kind"] == "heldout_domain_expert_ood_selection"
    assert expert_selection["expert_overlap_penalty"]["kind"] == (
        "domain_localized_overlap_penalty"
    )
    assert expert_selection["expert_overlap_penalty"]["weight"] > 0.0
    assert expert_selection["expert_overlap_penalty"]["target_coverage_weight"] > 0.0
    assert expert_selection["expert_overlap_penalty"]["effect_quantile_weight"] > 0.0
    assert len(expert_selection["expert_overlap_penalty_grid"]) >= 2
    assert any(
        profile["kind"] == "two_stage_target_then_projection"
        and profile["fit_mode"] == "target_then_projection"
        and 0.0625 in profile["projection_cap_grid"]
        and 0.25 in profile["projection_cap_grid"]
        and profile["margin_weight"] > 0.0
        for profile in expert_selection["expert_overlap_penalty_grid"]
    )
    assert any(
        profile["name"] == "combined_target_w14_tol102_projection"
        and profile["target_domains"] == ("combined_shift",)
        and 0.0625 in profile["projection_cap_grid"]
        and profile["target_coverage_weight"] > 10.0
        for profile in expert_selection["expert_overlap_penalty_grid"]
    )
    assert 0.0 in expert_selection["shrinkage_grid"]
    assert 0.03125 in expert_selection["shrinkage_grid"]
    assert 1.0 in expert_selection["shrinkage_grid"]
    assert expert_selection["selected"]["expert"] in {
        "baseline",
        "pure_effect",
        "combined_shift",
    }
    assert set(expert_selection["split_modes"]) == {"pure_effect", "combined_shift"}
    assert {candidate["expert"] for candidate in expert_selection["candidates"]} == {
        "pure_effect",
        "combined_shift",
    }
    assert all(
        "selected_shrinkage" in candidate
        and "selected_projection_cap" in candidate
        and "selected_gate_compatible" in candidate
        and candidate["selection_rule"]
        and "shrinkage_grid" in candidate
        and candidate["shrinkage_grid"]
        and all(
            "projection_cap" in grid and "gate_compatible" in grid
            for grid in candidate["shrinkage_grid"]
        )
        and "overlap_penalty" in candidate
        for candidate in expert_selection["candidates"]
    )
    assert "effect_size_shift" in expert_selection["baseline"]["domain_coverages"]
    assert "combined_shift" in expert_selection["baseline"]["domain_coverages"]
    combined_scale = metadata["ood_objective"]["combined_shift_scale"]
    assert combined_scale["kind"] == "domain_specific_combined_shift_log_multiplier"
    assert combined_scale["log_amplitude"] >= 0.0
    assert len(combined_scale["effect_bin_edges"]) == 2
    assert len(combined_scale["effect_bin_log_amplitudes"]) == 3
    assert combined_scale["context_gate"]["kind"].endswith("_classifier")
    assert combined_scale["context_gate"]["strength"] >= 0.0
    assert combined_scale["activation"]["support_center"] >= 0.0
    assert combined_scale["activation"]["low_design_center"] > 0.0
    assert combined_scale["activation"]["low_community_center"] > 0.0
    combined_selection = diagnostics["combined_shift_scale_selection"]
    assert combined_selection["kind"] == "context_gated_combined_shift_scale_selection"
    assert len(combined_selection["selected"]["effect_bin_edges"]) == 2
    assert len(combined_selection["selected"]["effect_bin_log_amplitudes"]) == 3
    assert "context_gate_strength" in combined_selection["selected"]
    assert "in_domain_context_gate" in combined_selection["selected"]
    assert "max_in_domain_context_gate_mean" in combined_selection["thresholds"]
    assert "combined_shift_coverage_floor" in combined_selection["thresholds"]
    assert "combined_shift" in combined_selection["baseline"]["domain_coverages"]
    restored_diagnostics = restored.to_metadata()["ood_objective"][
        "final_multiplier_diagnostics"
    ]
    assert restored_diagnostics["domains"][0]["label"] == "effect_size_shift"
    assert (
        restored_diagnostics["effect_shift_head_selection"]["kind"]
        == "post_fit_independent_effect_shift_head_selection"
    )
    assert (
        restored.to_metadata()["ood_objective"]["combined_shift_scale"]["kind"]
        == "domain_specific_combined_shift_log_multiplier"
    )
    assert restored.ood_inflation_parameters is not None
    assert len(restored.ood_inflation_parameters) == (15 + posterior.mean.shape[1] + 14)
    assert np.max(inflation) <= calibration.ood_uncertainty_max_multiplier
    assert np.mean(inflation) > 1.0


def test_combined_shift_scale_metadata_inflates_applicable_context(
    conditional_case,
):
    posterior, _, X, Y, calibration = conditional_case
    metadata = calibration.to_metadata()
    base = ConditionalBetaScaleCalibration.from_metadata(metadata)
    metadata["ood_objective"]["combined_shift_scale"]["log_amplitude"] = float(
        np.log(2.0)
    )
    inflated = ConditionalBetaScaleCalibration.from_metadata(metadata)
    shifted = BetaPosterior(mean=posterior.mean + 25.0, scale=posterior.scale)

    baseline_multiplier = conditional_beta_scale_multipliers(
        shifted,
        base,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )
    inflated_multiplier = conditional_beta_scale_multipliers(
        shifted,
        inflated,
        X=X,
        Y=Y,
        distribution="probit",
        coefficient_names=("Intercept", "x1"),
    )

    assert float(np.mean(inflated_multiplier)) > float(np.mean(baseline_multiplier))
    assert np.max(inflated_multiplier) <= inflated.max_multiplier


def test_gated_effect_ood_curve_suppresses_high_design_close_support():
    support_excess = np.asarray([0.0, 0.0, 3.0], dtype=float)
    effect_signal = np.asarray([2.0, 2.0, 2.0], dtype=float)
    low_design = np.zeros_like(support_excess)
    high_design = np.asarray([3.0, 3.0, 3.0], dtype=float)
    parameters = (
        -4.0,
        0.01,
        0.1,
        0.4,
        0.2,
        -1.0,
        2.0,
        3.0,
        4.0,
    )

    low_design_inflation = _learned_ood_log_inflation_numpy(
        support_excess,
        effect_signal=effect_signal,
        design_signal=low_design,
        parameters=parameters,
        max_multiplier=8.0,
    )
    high_design_inflation = _learned_ood_log_inflation_numpy(
        support_excess,
        effect_signal=effect_signal,
        design_signal=high_design,
        parameters=parameters,
        max_multiplier=8.0,
    )

    assert high_design_inflation[0] < low_design_inflation[0]
    assert high_design_inflation[1] < low_design_inflation[1]
    assert high_design_inflation[2] > high_design_inflation[0]


def test_gated_effect_ood_curve_uses_stratum_offsets():
    support_excess = np.zeros(3, dtype=float)
    effect_signal = np.full(3, 2.0, dtype=float)
    design_signal = np.zeros(3, dtype=float)
    parameters = (
        -4.0,
        0.01,
        0.1,
        0.4,
        0.2,
        -1.0,
        2.0,
        3.0,
        0.0,
        -2.0,
        0.0,
        2.0,
        -1.0,
        0.0,
        1.0,
        -0.5,
        0.5,
    )

    inflation = _learned_ood_log_inflation_numpy(
        support_excess,
        effect_signal=effect_signal,
        design_signal=design_signal,
        prevalence_stratum=np.asarray([0, 1, 2], dtype=np.int32),
        design_stratum=np.asarray([0, 1, 2], dtype=np.int32),
        coefficient_stratum=np.asarray([0, 0, 1], dtype=np.int32),
        parameters=parameters,
        max_multiplier=8.0,
    )

    assert inflation[0] < inflation[1] < inflation[2]


def test_conditional_calibration_loads_legacy_version_seven_curve(
    conditional_case,
):
    posterior, _, X, Y, calibration = conditional_case
    metadata = calibration.to_metadata()
    metadata["semantics_version"] = 7
    metadata["support"]["ood_uncertainty"][
        "transform"
    ] = "support_excess_learned_softplus"
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
    scale_tril = np.broadcast_to(base_tril, (batch, species, 2, 2)).copy()
    marginal = np.transpose(np.sqrt(np.sum(np.square(scale_tril), axis=-1)), (0, 2, 1))
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
