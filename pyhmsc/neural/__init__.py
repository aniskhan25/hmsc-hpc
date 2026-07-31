"""Experimental neural HMSC utilities.

This namespace is intentionally separate from the stable HMSC sampler API. The
initial utilities support benchmark corpus generation for amortized Neural-HMSC
posterior inference.
"""

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pyhmsc-mpl"))
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pyhmsc-cache")
)

from pyhmsc.neural.simulator import (
    FIXED_EFFECT_OOD_REGIMES,
    FixedEffectDataset,
    IidLatentEffectDataset,
    SpatialLatentEffectDataset,
    TraitEffectDataset,
    generate_fixed_effect_corpus,
    simulate_iid_latent_effect_dataset,
    simulate_spatial_latent_effect_dataset,
    simulate_fixed_effect_dataset,
    simulate_fixed_effect_ood_dataset,
    simulate_trait_effect_dataset,
    simulate_trait_gamma_boundary_dataset,
)

__all__ = [
    "FIXED_EFFECT_OOD_REGIMES",
    "BetaSbcDiagnostics",
    "BetaSbcStratumDiagnostics",
    "FixedEffectDataset",
    "FixedShapeBetaPosteriorModel",
    "IidLatentEffectDataset",
    "IidLatentFactorPosteriorModel",
    "NEURAL_CHECKPOINT_VERSION",
    "NEURAL_TRAINING_CORPUS_VERSION",
    "NeuralHmscCompatibilityError",
    "NeuralHmscInference",
    "package_neural_hmsc_coefficient_calibration",
    "SpatialLatentEffectDataset",
    "SpatialLatentFactorPosteriorModel",
    "TraitEffectDataset",
    "TraitGammaPosteriorModel",
    "TraitGammaCalibration",
    "TraitGammaNeuralHmscInference",
    "TRAIT_GAMMA_BASELINE_ID",
    "TRAIT_GAMMA_CHECKPOINT_VERSION",
    "freeze_trait_gamma_baseline",
    "finite_sample_conformal_quantile",
    "load_trait_gamma_baseline",
    "package_trait_gamma_calibration",
    "validate_trait_gamma_baseline",
    "VariableShapeBetaPosteriorModel",
    "VariableDesignBetaPosteriorModel",
    "GatedVariableDesignBetaPosteriorModel",
    "VariableDesignBetaCalibration",
    "VariableDesignNeuralHmscInference",
    "GatedVariableDesignNeuralHmscInference",
    "VariableDesignTrainingData",
    "VariableDesignPredictiveAuxiliaryData",
    "VARIABLE_DESIGN_CHECKPOINT_VERSION",
    "VARIABLE_DESIGN_MODEL_FAMILY",
    "GATED_VARIABLE_DESIGN_CHECKPOINT_VERSION",
    "GATED_VARIABLE_DESIGN_MODEL_FAMILY",
    "VariableShapeBetaCalibration",
    "VariableShapeNeuralHmscInference",
    "VARIABLE_CHECKPOINT_VERSION",
    "VARIABLE_SHAPE_BASELINE_ID",
    "freeze_variable_shape_baseline",
    "load_variable_shape_baseline",
    "validate_variable_shape_baseline",
    "BetaScaleCalibration",
    "BetaPredictiveMeanCalibration",
    "PredictiveNoDegradationThresholds",
    "ConditionalBetaOODCalibrationBatch",
    "ConditionalBetaScaleCalibration",
    "apply_conditional_beta_scale_calibration",
    "apply_beta_predictive_calibration",
    "apply_beta_predictive_mean_calibration",
    "apply_beta_scale_calibration",
    "beta_sbc_rank_diagnostics",
    "beta_sbc_stratified_diagnostics",
    "beta_expected_design_information",
    "calibration_metadata",
    "compiled_trait_effect_training_data",
    "compare_beta_posterior_files",
    "compare_beta_posteriors",
    "compare_gamma_posterior_files",
    "compare_gamma_posteriors",
    "compare_iid_association_posterior_files",
    "compare_iid_association_posteriors",
    "evaluate_beta_posterior",
    "evaluate_gamma_posterior",
    "evaluate_iid_latent_posterior",
    "evaluate_masked_beta_posterior",
    "evaluate_spatial_latent_posterior",
    "fixed_shape_training_data",
    "fit_beta_scale_calibration",
    "BetaResponseCalibrationBatch",
    "PredictiveEnsembleMember",
    "PredictiveProbabilityEnsemble",
    "DEFAULT_PREDICTIVE_MEAN_POLICY",
    "PREDICTIVE_MEAN_POLICIES",
    "load_predictive_mean_ensemble",
    "PROMOTED_PREDICTIVE_BASELINE_ID",
    "freeze_predictive_deployment_baseline",
    "load_predictive_deployment_baseline",
    "validate_predictive_deployment_baseline",
    "NEURAL_HMSC_RELEASE_ID",
    "NeuralHmscRelease",
    "freeze_neural_hmsc_release",
    "load_neural_hmsc_release",
    "validate_neural_hmsc_release",
    "CORRELATION_OVERLAY_ID",
    "CorrelationFeatureNormalizer",
    "FixedProbitCorrelationHead",
    "FixedProbitCovarianceInference",
    "fit_fixed_probit_covariance_overlay",
    "STUDENT_T_OVERLAY_ID",
    "StudentTBetaPosterior",
    "StudentTFeatureNormalizer",
    "FixedProbitStudentTHead",
    "FixedProbitStudentTInference",
    "fit_fixed_probit_student_t_overlay",
    "sample_student_t_beta_posterior",
    "write_student_t_beta_posterior_hdf5",
    "McmcTeacherResponseBatch",
    "McmcTeacherResidualHead",
    "ContextIdentityGate",
    "fit_mcmc_teacher_residual_head",
    "fit_cross_fitted_mcmc_teacher_residual_head",
    "fit_context_identity_gate",
    "evaluate_mcmc_teacher_residual_head",
    "fit_beta_predictive_mean_calibration",
    "fit_beta_response_mean_calibration",
    "fit_beta_transfer_response_mean_calibration",
    "fit_beta_transfer_response_branch_calibration",
    "independent_source_transfer_predictive_mean_selector_metadata",
    "evaluate_beta_target_context_gate",
    "target_context_conditioned_source_transfer_selector_metadata",
    "evaluate_cross_dataset_predictive_gate",
    "fit_conditional_beta_scale_calibration",
    "conditional_beta_scale_multipliers",
    "conditional_beta_effect_size_signal",
    "conditional_beta_mean_support_diagnostics",
    "conditional_beta_ood_uncertainty_inflation",
    "conditional_beta_support_trust",
    "domain_conditional_predictive_mean_selector_metadata",
    "generate_fixed_effect_corpus",
    "iid_latent_training_data",
    "predict_beta_posterior",
    "predict_gamma_posterior",
    "predict_iid_latent_posterior",
    "predict_spatial_latent_posterior",
    "predict_variable_beta_posterior",
    "probit_irls_laplace_anchor",
    "probit_irls_laplace_full_anchor",
    "render_cross_dataset_predictive_gate_markdown",
    "select_predictive_mean_calibration_for_context",
    "simulate_fixed_effect_dataset",
    "simulate_fixed_effect_ood_dataset",
    "simulate_iid_latent_effect_dataset",
    "simulate_spatial_latent_effect_dataset",
    "simulate_trait_effect_dataset",
    "simulate_trait_gamma_boundary_dataset",
    "spatial_holdout_random_effect_rmse",
    "spatial_latent_training_data",
    "train_fixed_shape_beta_model",
    "train_trait_gamma_model",
    "trait_effect_training_data",
    "variable_shape_training_data",
    "variable_design_training_data",
    "variable_design_predictive_auxiliary_data",
    "variable_design_probit_score_loss",
    "write_benchmark_report",
    "write_sbc_report",
    "write_beta_posterior_hdf5",
    "write_gamma_posterior_hdf5",
    "write_trait_gamma_posterior_hdf5",
    "write_iid_latent_posterior_hdf5",
    "write_spatial_latent_posterior_hdf5",
    "GENERATIVE_IID_PROTOCOL",
    "GenerativeIidDataset",
    "GenerativeIidBatch",
    "GenerativeIidPosteriorModel",
    "GenerativeIidInference",
    "JointLowRankPosterior",
    "JointStateLayout",
    "batch_generative_iid_datasets",
    "simulate_generative_iid_dataset",
    "make_stratified_response_mask",
    "importance_weighted_variational_loss",
    "train_generative_iid_model",
    "run_exact_model_mcmc",
    "validate_generative_iid_checkpoint",
    "GENERATIVE_IID_V2_PROTOCOL",
    "GenerativeIidOrbitPosteriorModel",
    "GenerativeIidOrbitInference",
    "JointOrbitPosterior",
    "MaskedLowRankStudentT",
    "OrbitMatrixNormal",
    "generative_iid_v2_log_joint",
    "importance_weighted_orbit_loss",
    "train_generative_iid_orbit_model",
    "validate_generative_iid_v2_checkpoint",
]


def __getattr__(name: str) -> object:
    if name in {
        "GENERATIVE_IID_PROTOCOL",
        "GenerativeIidDataset",
        "GenerativeIidBatch",
        "GenerativeIidPosteriorModel",
        "JointLowRankPosterior",
        "JointStateLayout",
        "batch_generative_iid_datasets",
        "simulate_generative_iid_dataset",
        "make_stratified_response_mask",
        "importance_weighted_variational_loss",
        "train_generative_iid_model",
    }:
        from pyhmsc.neural.generative_iid import (
            GENERATIVE_IID_PROTOCOL,
            GenerativeIidBatch,
            GenerativeIidDataset,
            GenerativeIidPosteriorModel,
            JointLowRankPosterior,
            JointStateLayout,
            batch_generative_iid_datasets,
            importance_weighted_variational_loss,
            make_stratified_response_mask,
            simulate_generative_iid_dataset,
            train_generative_iid_model,
        )

        return {
            "GENERATIVE_IID_PROTOCOL": GENERATIVE_IID_PROTOCOL,
            "GenerativeIidDataset": GenerativeIidDataset,
            "GenerativeIidBatch": GenerativeIidBatch,
            "GenerativeIidPosteriorModel": GenerativeIidPosteriorModel,
            "JointLowRankPosterior": JointLowRankPosterior,
            "JointStateLayout": JointStateLayout,
            "batch_generative_iid_datasets": batch_generative_iid_datasets,
            "simulate_generative_iid_dataset": simulate_generative_iid_dataset,
            "make_stratified_response_mask": make_stratified_response_mask,
            "importance_weighted_variational_loss": (
                importance_weighted_variational_loss
            ),
            "train_generative_iid_model": train_generative_iid_model,
        }[name]
    if name in {
        "GenerativeIidInference",
        "validate_generative_iid_checkpoint",
    }:
        from pyhmsc.neural.generative_iid_artifact import (
            GenerativeIidInference,
            validate_generative_iid_checkpoint,
        )

        return {
            "GenerativeIidInference": GenerativeIidInference,
            "validate_generative_iid_checkpoint": (
                validate_generative_iid_checkpoint
            ),
        }[name]
    if name == "run_exact_model_mcmc":
        from pyhmsc.neural.generative_iid_mcmc import run_exact_model_mcmc

        return run_exact_model_mcmc
    if name in {
        "GENERATIVE_IID_V2_PROTOCOL",
        "GenerativeIidOrbitPosteriorModel",
        "JointOrbitPosterior",
        "MaskedLowRankStudentT",
        "OrbitMatrixNormal",
        "generative_iid_v2_log_joint",
        "importance_weighted_orbit_loss",
        "train_generative_iid_orbit_model",
    }:
        from pyhmsc.neural.generative_iid_v2 import (
            GENERATIVE_IID_V2_PROTOCOL,
            GenerativeIidOrbitPosteriorModel,
            JointOrbitPosterior,
            MaskedLowRankStudentT,
            OrbitMatrixNormal,
            generative_iid_v2_log_joint,
            importance_weighted_orbit_loss,
            train_generative_iid_orbit_model,
        )

        return {
            "GENERATIVE_IID_V2_PROTOCOL": GENERATIVE_IID_V2_PROTOCOL,
            "GenerativeIidOrbitPosteriorModel": GenerativeIidOrbitPosteriorModel,
            "JointOrbitPosterior": JointOrbitPosterior,
            "MaskedLowRankStudentT": MaskedLowRankStudentT,
            "OrbitMatrixNormal": OrbitMatrixNormal,
            "generative_iid_v2_log_joint": generative_iid_v2_log_joint,
            "importance_weighted_orbit_loss": importance_weighted_orbit_loss,
            "train_generative_iid_orbit_model": train_generative_iid_orbit_model,
        }[name]
    if name in {
        "GenerativeIidOrbitInference",
        "validate_generative_iid_v2_checkpoint",
    }:
        from pyhmsc.neural.generative_iid_v2_artifact import (
            GenerativeIidOrbitInference,
            validate_generative_iid_v2_checkpoint,
        )

        return {
            "GenerativeIidOrbitInference": GenerativeIidOrbitInference,
            "validate_generative_iid_v2_checkpoint": (
                validate_generative_iid_v2_checkpoint
            ),
        }[name]
    if name in {
        "TRAIT_GAMMA_BASELINE_ID",
        "TRAIT_GAMMA_CHECKPOINT_VERSION",
        "TraitGammaCalibration",
        "TraitGammaNeuralHmscInference",
        "freeze_trait_gamma_baseline",
        "finite_sample_conformal_quantile",
        "load_trait_gamma_baseline",
        "package_trait_gamma_calibration",
        "validate_trait_gamma_baseline",
    }:
        from pyhmsc.neural.trait_inference import (
            TRAIT_GAMMA_BASELINE_ID,
            TRAIT_GAMMA_CHECKPOINT_VERSION,
            TraitGammaCalibration,
            TraitGammaNeuralHmscInference,
            freeze_trait_gamma_baseline,
            finite_sample_conformal_quantile,
            load_trait_gamma_baseline,
            package_trait_gamma_calibration,
            validate_trait_gamma_baseline,
        )

        return {
            "TRAIT_GAMMA_BASELINE_ID": TRAIT_GAMMA_BASELINE_ID,
            "TRAIT_GAMMA_CHECKPOINT_VERSION": TRAIT_GAMMA_CHECKPOINT_VERSION,
            "TraitGammaCalibration": TraitGammaCalibration,
            "TraitGammaNeuralHmscInference": TraitGammaNeuralHmscInference,
            "freeze_trait_gamma_baseline": freeze_trait_gamma_baseline,
            "finite_sample_conformal_quantile": finite_sample_conformal_quantile,
            "load_trait_gamma_baseline": load_trait_gamma_baseline,
            "package_trait_gamma_calibration": package_trait_gamma_calibration,
            "validate_trait_gamma_baseline": validate_trait_gamma_baseline,
        }[name]
    if name in {
        "VARIABLE_CHECKPOINT_VERSION",
        "VARIABLE_SHAPE_BASELINE_ID",
        "VariableShapeBetaCalibration",
        "VariableShapeNeuralHmscInference",
        "freeze_variable_shape_baseline",
        "load_variable_shape_baseline",
        "validate_variable_shape_baseline",
    }:
        from pyhmsc.neural.variable_inference import (
            VARIABLE_CHECKPOINT_VERSION,
            VARIABLE_SHAPE_BASELINE_ID,
            VariableShapeBetaCalibration,
            VariableShapeNeuralHmscInference,
            freeze_variable_shape_baseline,
            load_variable_shape_baseline,
            validate_variable_shape_baseline,
        )

        return {
            "VARIABLE_CHECKPOINT_VERSION": VARIABLE_CHECKPOINT_VERSION,
            "VARIABLE_SHAPE_BASELINE_ID": VARIABLE_SHAPE_BASELINE_ID,
            "VariableShapeBetaCalibration": VariableShapeBetaCalibration,
            "VariableShapeNeuralHmscInference": VariableShapeNeuralHmscInference,
            "freeze_variable_shape_baseline": freeze_variable_shape_baseline,
            "load_variable_shape_baseline": load_variable_shape_baseline,
            "validate_variable_shape_baseline": validate_variable_shape_baseline,
        }[name]
    if name in {
        "NEURAL_HMSC_RELEASE_ID",
        "NeuralHmscRelease",
        "freeze_neural_hmsc_release",
        "load_neural_hmsc_release",
        "validate_neural_hmsc_release",
    }:
        from pyhmsc.neural.release import (
            NEURAL_HMSC_RELEASE_ID,
            NeuralHmscRelease,
            freeze_neural_hmsc_release,
            load_neural_hmsc_release,
            validate_neural_hmsc_release,
        )

        return {
            "NEURAL_HMSC_RELEASE_ID": NEURAL_HMSC_RELEASE_ID,
            "NeuralHmscRelease": NeuralHmscRelease,
            "freeze_neural_hmsc_release": freeze_neural_hmsc_release,
            "load_neural_hmsc_release": load_neural_hmsc_release,
            "validate_neural_hmsc_release": validate_neural_hmsc_release,
        }[name]
    if name in {
        "CORRELATION_OVERLAY_ID",
        "CorrelationFeatureNormalizer",
        "FixedProbitCorrelationHead",
        "FixedProbitCovarianceInference",
        "fit_fixed_probit_covariance_overlay",
    }:
        from pyhmsc.neural.covariance_inference import (
            CORRELATION_OVERLAY_ID,
            CorrelationFeatureNormalizer,
            FixedProbitCorrelationHead,
            FixedProbitCovarianceInference,
            fit_fixed_probit_covariance_overlay,
        )

        return {
            "CORRELATION_OVERLAY_ID": CORRELATION_OVERLAY_ID,
            "CorrelationFeatureNormalizer": CorrelationFeatureNormalizer,
            "FixedProbitCorrelationHead": FixedProbitCorrelationHead,
            "FixedProbitCovarianceInference": FixedProbitCovarianceInference,
            "fit_fixed_probit_covariance_overlay": (
                fit_fixed_probit_covariance_overlay
            ),
        }[name]
    if name in {
        "STUDENT_T_OVERLAY_ID",
        "StudentTBetaPosterior",
        "StudentTFeatureNormalizer",
        "FixedProbitStudentTHead",
        "FixedProbitStudentTInference",
        "fit_fixed_probit_student_t_overlay",
        "sample_student_t_beta_posterior",
        "write_student_t_beta_posterior_hdf5",
    }:
        from pyhmsc.neural.student_t_inference import (
            STUDENT_T_OVERLAY_ID,
            FixedProbitStudentTHead,
            FixedProbitStudentTInference,
            StudentTBetaPosterior,
            StudentTFeatureNormalizer,
            fit_fixed_probit_student_t_overlay,
            sample_student_t_beta_posterior,
            write_student_t_beta_posterior_hdf5,
        )

        return {
            "STUDENT_T_OVERLAY_ID": STUDENT_T_OVERLAY_ID,
            "StudentTBetaPosterior": StudentTBetaPosterior,
            "StudentTFeatureNormalizer": StudentTFeatureNormalizer,
            "FixedProbitStudentTHead": FixedProbitStudentTHead,
            "FixedProbitStudentTInference": FixedProbitStudentTInference,
            "fit_fixed_probit_student_t_overlay": fit_fixed_probit_student_t_overlay,
            "sample_student_t_beta_posterior": sample_student_t_beta_posterior,
            "write_student_t_beta_posterior_hdf5": (
                write_student_t_beta_posterior_hdf5
            ),
        }[name]
    if name in {
        "McmcTeacherResponseBatch",
        "McmcTeacherResidualHead",
        "ContextIdentityGate",
        "fit_mcmc_teacher_residual_head",
        "fit_cross_fitted_mcmc_teacher_residual_head",
        "fit_context_identity_gate",
        "evaluate_mcmc_teacher_residual_head",
    }:
        from pyhmsc.neural.teacher_residual import (
            ContextIdentityGate,
            McmcTeacherResponseBatch,
            McmcTeacherResidualHead,
            evaluate_mcmc_teacher_residual_head,
            fit_context_identity_gate,
            fit_cross_fitted_mcmc_teacher_residual_head,
            fit_mcmc_teacher_residual_head,
        )

        return {
            "ContextIdentityGate": ContextIdentityGate,
            "McmcTeacherResponseBatch": McmcTeacherResponseBatch,
            "McmcTeacherResidualHead": McmcTeacherResidualHead,
            "fit_mcmc_teacher_residual_head": fit_mcmc_teacher_residual_head,
            "fit_cross_fitted_mcmc_teacher_residual_head": (
                fit_cross_fitted_mcmc_teacher_residual_head
            ),
            "fit_context_identity_gate": fit_context_identity_gate,
            "evaluate_mcmc_teacher_residual_head": (
                evaluate_mcmc_teacher_residual_head
            ),
        }[name]
    if name in {
        "PROMOTED_PREDICTIVE_BASELINE_ID",
        "freeze_predictive_deployment_baseline",
        "load_predictive_deployment_baseline",
        "validate_predictive_deployment_baseline",
    }:
        from pyhmsc.neural.deployment import (
            PROMOTED_PREDICTIVE_BASELINE_ID,
            freeze_predictive_deployment_baseline,
            load_predictive_deployment_baseline,
            validate_predictive_deployment_baseline,
        )

        return {
            "PROMOTED_PREDICTIVE_BASELINE_ID": PROMOTED_PREDICTIVE_BASELINE_ID,
            "freeze_predictive_deployment_baseline": (
                freeze_predictive_deployment_baseline
            ),
            "load_predictive_deployment_baseline": (
                load_predictive_deployment_baseline
            ),
            "validate_predictive_deployment_baseline": (
                validate_predictive_deployment_baseline
            ),
        }[name]
    if name in {
        "DEFAULT_PREDICTIVE_MEAN_POLICY",
        "PREDICTIVE_MEAN_POLICIES",
        "PredictiveEnsembleMember",
        "PredictiveProbabilityEnsemble",
        "load_predictive_mean_ensemble",
    }:
        from pyhmsc.neural.ensemble import (
            DEFAULT_PREDICTIVE_MEAN_POLICY,
            PREDICTIVE_MEAN_POLICIES,
            PredictiveEnsembleMember,
            PredictiveProbabilityEnsemble,
            load_predictive_mean_ensemble,
        )

        return {
            "DEFAULT_PREDICTIVE_MEAN_POLICY": DEFAULT_PREDICTIVE_MEAN_POLICY,
            "PREDICTIVE_MEAN_POLICIES": PREDICTIVE_MEAN_POLICIES,
            "PredictiveEnsembleMember": PredictiveEnsembleMember,
            "PredictiveProbabilityEnsemble": PredictiveProbabilityEnsemble,
            "load_predictive_mean_ensemble": load_predictive_mean_ensemble,
        }[name]
    if name == "probit_irls_laplace_anchor":
        from pyhmsc.neural.models import probit_irls_laplace_anchor

        return probit_irls_laplace_anchor
    if name == "probit_irls_laplace_full_anchor":
        from pyhmsc.neural.models import probit_irls_laplace_full_anchor

        return probit_irls_laplace_full_anchor
    if name in {
        "BetaPredictiveMeanCalibration",
        "BetaResponseCalibrationBatch",
        "apply_beta_predictive_mean_calibration",
        "domain_conditional_predictive_mean_selector_metadata",
        "fit_beta_predictive_mean_calibration",
        "fit_beta_response_mean_calibration",
        "fit_beta_transfer_response_mean_calibration",
        "fit_beta_transfer_response_branch_calibration",
        "independent_source_transfer_predictive_mean_selector_metadata",
        "evaluate_beta_target_context_gate",
        "target_context_conditioned_source_transfer_selector_metadata",
        "select_predictive_mean_calibration_for_context",
    }:
        from pyhmsc.neural.mean_calibration import (
            BetaPredictiveMeanCalibration,
            BetaResponseCalibrationBatch,
            apply_beta_predictive_mean_calibration,
            domain_conditional_predictive_mean_selector_metadata,
            fit_beta_predictive_mean_calibration,
            fit_beta_response_mean_calibration,
            fit_beta_transfer_response_mean_calibration,
            fit_beta_transfer_response_branch_calibration,
            independent_source_transfer_predictive_mean_selector_metadata,
            evaluate_beta_target_context_gate,
            target_context_conditioned_source_transfer_selector_metadata,
            select_predictive_mean_calibration_for_context,
        )

        return {
            "BetaPredictiveMeanCalibration": BetaPredictiveMeanCalibration,
            "BetaResponseCalibrationBatch": BetaResponseCalibrationBatch,
            "apply_beta_predictive_mean_calibration": apply_beta_predictive_mean_calibration,
            "domain_conditional_predictive_mean_selector_metadata": (
                domain_conditional_predictive_mean_selector_metadata
            ),
            "fit_beta_predictive_mean_calibration": fit_beta_predictive_mean_calibration,
            "fit_beta_response_mean_calibration": fit_beta_response_mean_calibration,
            "fit_beta_transfer_response_mean_calibration": (
                fit_beta_transfer_response_mean_calibration
            ),
            "fit_beta_transfer_response_branch_calibration": (
                fit_beta_transfer_response_branch_calibration
            ),
            "independent_source_transfer_predictive_mean_selector_metadata": (
                independent_source_transfer_predictive_mean_selector_metadata
            ),
            "evaluate_beta_target_context_gate": evaluate_beta_target_context_gate,
            "target_context_conditioned_source_transfer_selector_metadata": (
                target_context_conditioned_source_transfer_selector_metadata
            ),
            "select_predictive_mean_calibration_for_context": (
                select_predictive_mean_calibration_for_context
            ),
        }[name]
    if name in {
        "PredictiveNoDegradationThresholds",
        "evaluate_cross_dataset_predictive_gate",
        "render_cross_dataset_predictive_gate_markdown",
    }:
        from pyhmsc.neural.predictive_selection import (
            PredictiveNoDegradationThresholds,
            evaluate_cross_dataset_predictive_gate,
            render_cross_dataset_predictive_gate_markdown,
        )

        return {
            "PredictiveNoDegradationThresholds": PredictiveNoDegradationThresholds,
            "evaluate_cross_dataset_predictive_gate": evaluate_cross_dataset_predictive_gate,
            "render_cross_dataset_predictive_gate_markdown": (
                render_cross_dataset_predictive_gate_markdown
            ),
        }[name]
    if name in {
        "ConditionalBetaScaleCalibration",
        "ConditionalBetaOODCalibrationBatch",
        "apply_conditional_beta_scale_calibration",
        "conditional_beta_scale_multipliers",
        "conditional_beta_effect_size_signal",
        "conditional_beta_mean_support_diagnostics",
        "conditional_beta_ood_uncertainty_inflation",
        "conditional_beta_support_trust",
        "fit_conditional_beta_scale_calibration",
    }:
        from pyhmsc.neural.conditional_calibration import (
            ConditionalBetaOODCalibrationBatch,
            ConditionalBetaScaleCalibration,
            apply_conditional_beta_scale_calibration,
            conditional_beta_scale_multipliers,
            conditional_beta_effect_size_signal,
            conditional_beta_mean_support_diagnostics,
            conditional_beta_ood_uncertainty_inflation,
            conditional_beta_support_trust,
            fit_conditional_beta_scale_calibration,
        )

        return {
            "ConditionalBetaScaleCalibration": ConditionalBetaScaleCalibration,
            "ConditionalBetaOODCalibrationBatch": ConditionalBetaOODCalibrationBatch,
            "apply_conditional_beta_scale_calibration": apply_conditional_beta_scale_calibration,
            "conditional_beta_scale_multipliers": conditional_beta_scale_multipliers,
            "conditional_beta_effect_size_signal": conditional_beta_effect_size_signal,
            "conditional_beta_mean_support_diagnostics": conditional_beta_mean_support_diagnostics,
            "conditional_beta_ood_uncertainty_inflation": conditional_beta_ood_uncertainty_inflation,
            "conditional_beta_support_trust": conditional_beta_support_trust,
            "fit_conditional_beta_scale_calibration": fit_conditional_beta_scale_calibration,
        }[name]
    if name == "FixedShapeBetaPosteriorModel":
        from pyhmsc.neural.models import FixedShapeBetaPosteriorModel

        return FixedShapeBetaPosteriorModel
    if name == "VariableShapeBetaPosteriorModel":
        from pyhmsc.neural.models import VariableShapeBetaPosteriorModel

        return VariableShapeBetaPosteriorModel
    if name == "VariableDesignBetaPosteriorModel":
        from pyhmsc.neural.models import VariableDesignBetaPosteriorModel

        return VariableDesignBetaPosteriorModel
    if name == "GatedVariableDesignBetaPosteriorModel":
        from pyhmsc.neural.models import GatedVariableDesignBetaPosteriorModel

        return GatedVariableDesignBetaPosteriorModel
    if name in {
        "VARIABLE_DESIGN_CHECKPOINT_VERSION",
        "VARIABLE_DESIGN_MODEL_FAMILY",
        "VariableDesignBetaCalibration",
        "VariableDesignNeuralHmscInference",
    }:
        from pyhmsc.neural.variable_design_inference import (
            VARIABLE_DESIGN_CHECKPOINT_VERSION,
            VARIABLE_DESIGN_MODEL_FAMILY,
            VariableDesignBetaCalibration,
            VariableDesignNeuralHmscInference,
        )

        return {
            "VARIABLE_DESIGN_CHECKPOINT_VERSION": VARIABLE_DESIGN_CHECKPOINT_VERSION,
            "VARIABLE_DESIGN_MODEL_FAMILY": VARIABLE_DESIGN_MODEL_FAMILY,
            "VariableDesignBetaCalibration": VariableDesignBetaCalibration,
            "VariableDesignNeuralHmscInference": (VariableDesignNeuralHmscInference),
        }[name]
    if name in {
        "GATED_VARIABLE_DESIGN_CHECKPOINT_VERSION",
        "GATED_VARIABLE_DESIGN_MODEL_FAMILY",
        "GatedVariableDesignNeuralHmscInference",
        "variable_design_probit_score_loss",
    }:
        from pyhmsc.neural.variable_design_gated_inference import (
            GATED_VARIABLE_DESIGN_CHECKPOINT_VERSION,
            GATED_VARIABLE_DESIGN_MODEL_FAMILY,
            GatedVariableDesignNeuralHmscInference,
            variable_design_probit_score_loss,
        )

        return {
            "GATED_VARIABLE_DESIGN_CHECKPOINT_VERSION": (
                GATED_VARIABLE_DESIGN_CHECKPOINT_VERSION
            ),
            "GATED_VARIABLE_DESIGN_MODEL_FAMILY": (
                GATED_VARIABLE_DESIGN_MODEL_FAMILY
            ),
            "GatedVariableDesignNeuralHmscInference": (
                GatedVariableDesignNeuralHmscInference
            ),
            "variable_design_probit_score_loss": variable_design_probit_score_loss,
        }[name]
    if name == "TraitGammaPosteriorModel":
        from pyhmsc.neural.models import TraitGammaPosteriorModel

        return TraitGammaPosteriorModel
    if name == "IidLatentFactorPosteriorModel":
        from pyhmsc.neural.models import IidLatentFactorPosteriorModel

        return IidLatentFactorPosteriorModel
    if name == "SpatialLatentFactorPosteriorModel":
        from pyhmsc.neural.models import SpatialLatentFactorPosteriorModel

        return SpatialLatentFactorPosteriorModel
    if name in {
        "NEURAL_CHECKPOINT_VERSION",
        "NEURAL_TRAINING_CORPUS_VERSION",
        "NeuralHmscCompatibilityError",
        "NeuralHmscInference",
        "package_neural_hmsc_coefficient_calibration",
    }:
        from pyhmsc.neural.inference import (
            NEURAL_CHECKPOINT_VERSION,
            NEURAL_TRAINING_CORPUS_VERSION,
            NeuralHmscCompatibilityError,
            NeuralHmscInference,
            package_neural_hmsc_coefficient_calibration,
        )

        return {
            "NEURAL_CHECKPOINT_VERSION": NEURAL_CHECKPOINT_VERSION,
            "NEURAL_TRAINING_CORPUS_VERSION": NEURAL_TRAINING_CORPUS_VERSION,
            "NeuralHmscCompatibilityError": NeuralHmscCompatibilityError,
            "NeuralHmscInference": NeuralHmscInference,
            "package_neural_hmsc_coefficient_calibration": (
                package_neural_hmsc_coefficient_calibration
            ),
        }[name]
    if name in {
        "BetaSbcDiagnostics",
        "BetaSbcStratumDiagnostics",
        "beta_expected_design_information",
        "beta_sbc_rank_diagnostics",
        "beta_sbc_stratified_diagnostics",
    }:
        from pyhmsc.neural.diagnostics import (
            BetaSbcDiagnostics,
            BetaSbcStratumDiagnostics,
            beta_expected_design_information,
            beta_sbc_rank_diagnostics,
            beta_sbc_stratified_diagnostics,
        )

        return {
            "BetaSbcDiagnostics": BetaSbcDiagnostics,
            "BetaSbcStratumDiagnostics": BetaSbcStratumDiagnostics,
            "beta_expected_design_information": beta_expected_design_information,
            "beta_sbc_rank_diagnostics": beta_sbc_rank_diagnostics,
            "beta_sbc_stratified_diagnostics": beta_sbc_stratified_diagnostics,
        }[name]
    if name in {
        "fixed_shape_training_data",
        "compiled_trait_effect_training_data",
        "iid_latent_training_data",
        "spatial_latent_training_data",
        "train_fixed_shape_beta_model",
        "train_trait_gamma_model",
        "trait_effect_training_data",
        "VariableDesignTrainingData",
        "VariableDesignPredictiveAuxiliaryData",
        "variable_design_training_data",
        "variable_design_predictive_auxiliary_data",
        "variable_shape_training_data",
    }:
        from pyhmsc.neural.train import (
            compiled_trait_effect_training_data,
            fixed_shape_training_data,
            iid_latent_training_data,
            spatial_latent_training_data,
            train_fixed_shape_beta_model,
            train_trait_gamma_model,
            trait_effect_training_data,
            VariableDesignPredictiveAuxiliaryData,
            VariableDesignTrainingData,
            variable_design_predictive_auxiliary_data,
            variable_design_training_data,
            variable_shape_training_data,
        )

        return {
            "fixed_shape_training_data": fixed_shape_training_data,
            "compiled_trait_effect_training_data": compiled_trait_effect_training_data,
            "iid_latent_training_data": iid_latent_training_data,
            "spatial_latent_training_data": spatial_latent_training_data,
            "train_fixed_shape_beta_model": train_fixed_shape_beta_model,
            "train_trait_gamma_model": train_trait_gamma_model,
            "trait_effect_training_data": trait_effect_training_data,
            "VariableDesignTrainingData": VariableDesignTrainingData,
            "VariableDesignPredictiveAuxiliaryData": (
                VariableDesignPredictiveAuxiliaryData
            ),
            "variable_design_training_data": variable_design_training_data,
            "variable_design_predictive_auxiliary_data": (
                variable_design_predictive_auxiliary_data
            ),
            "variable_shape_training_data": variable_shape_training_data,
        }[name]
    if name in {
        "evaluate_beta_posterior",
        "evaluate_gamma_posterior",
        "evaluate_iid_latent_posterior",
        "evaluate_masked_beta_posterior",
        "evaluate_spatial_latent_posterior",
        "predict_beta_posterior",
        "predict_gamma_posterior",
        "predict_iid_latent_posterior",
        "predict_spatial_latent_posterior",
        "predict_variable_beta_posterior",
        "spatial_holdout_random_effect_rmse",
    }:
        from pyhmsc.neural.evaluation import (
            evaluate_beta_posterior,
            evaluate_gamma_posterior,
            evaluate_iid_latent_posterior,
            evaluate_masked_beta_posterior,
            evaluate_spatial_latent_posterior,
            predict_beta_posterior,
            predict_gamma_posterior,
            predict_iid_latent_posterior,
            predict_spatial_latent_posterior,
            predict_variable_beta_posterior,
            spatial_holdout_random_effect_rmse,
        )

        return {
            "evaluate_beta_posterior": evaluate_beta_posterior,
            "evaluate_gamma_posterior": evaluate_gamma_posterior,
            "evaluate_iid_latent_posterior": evaluate_iid_latent_posterior,
            "evaluate_masked_beta_posterior": evaluate_masked_beta_posterior,
            "evaluate_spatial_latent_posterior": evaluate_spatial_latent_posterior,
            "predict_beta_posterior": predict_beta_posterior,
            "predict_gamma_posterior": predict_gamma_posterior,
            "predict_iid_latent_posterior": predict_iid_latent_posterior,
            "predict_spatial_latent_posterior": predict_spatial_latent_posterior,
            "predict_variable_beta_posterior": predict_variable_beta_posterior,
            "spatial_holdout_random_effect_rmse": spatial_holdout_random_effect_rmse,
        }[name]
    if name == "write_beta_posterior_hdf5":
        from pyhmsc.neural.storage import write_beta_posterior_hdf5

        return write_beta_posterior_hdf5
    if name == "write_gamma_posterior_hdf5":
        from pyhmsc.neural.storage import write_gamma_posterior_hdf5

        return write_gamma_posterior_hdf5
    if name == "write_trait_gamma_posterior_hdf5":
        from pyhmsc.neural.storage import write_trait_gamma_posterior_hdf5

        return write_trait_gamma_posterior_hdf5
    if name == "write_iid_latent_posterior_hdf5":
        from pyhmsc.neural.storage import write_iid_latent_posterior_hdf5

        return write_iid_latent_posterior_hdf5
    if name == "write_spatial_latent_posterior_hdf5":
        from pyhmsc.neural.storage import write_spatial_latent_posterior_hdf5

        return write_spatial_latent_posterior_hdf5
    if name in {
        "BetaScaleCalibration",
        "apply_beta_predictive_calibration",
        "apply_beta_scale_calibration",
        "calibration_metadata",
        "fit_beta_scale_calibration",
    }:
        from pyhmsc.neural.calibration import (
            BetaScaleCalibration,
            apply_beta_predictive_calibration,
            apply_beta_scale_calibration,
            calibration_metadata,
            fit_beta_scale_calibration,
        )

        return {
            "BetaScaleCalibration": BetaScaleCalibration,
            "apply_beta_predictive_calibration": apply_beta_predictive_calibration,
            "apply_beta_scale_calibration": apply_beta_scale_calibration,
            "calibration_metadata": calibration_metadata,
            "fit_beta_scale_calibration": fit_beta_scale_calibration,
        }[name]
    if name in {
        "compare_beta_posterior_files",
        "compare_beta_posteriors",
        "compare_gamma_posterior_files",
        "compare_gamma_posteriors",
        "compare_iid_association_posterior_files",
        "compare_iid_association_posteriors",
        "write_benchmark_report",
        "write_sbc_report",
    }:
        from pyhmsc.neural.benchmark import (
            compare_beta_posterior_files,
            compare_beta_posteriors,
            compare_gamma_posterior_files,
            compare_gamma_posteriors,
            compare_iid_association_posterior_files,
            compare_iid_association_posteriors,
            write_benchmark_report,
            write_sbc_report,
        )

        return {
            "compare_beta_posterior_files": compare_beta_posterior_files,
            "compare_beta_posteriors": compare_beta_posteriors,
            "compare_gamma_posterior_files": compare_gamma_posterior_files,
            "compare_gamma_posteriors": compare_gamma_posteriors,
            "compare_iid_association_posterior_files": compare_iid_association_posterior_files,
            "compare_iid_association_posteriors": compare_iid_association_posteriors,
            "write_benchmark_report": write_benchmark_report,
            "write_sbc_report": write_sbc_report,
        }[name]
    raise AttributeError(f"module 'pyhmsc.neural' has no attribute {name!r}")
