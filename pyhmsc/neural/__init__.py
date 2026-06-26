"""Experimental neural HMSC utilities.

This namespace is intentionally separate from the stable HMSC sampler API. The
initial utilities support benchmark corpus generation for amortized Neural-HMSC
posterior inference.
"""

from pyhmsc.neural.simulator import (
    FixedEffectDataset,
    IidLatentEffectDataset,
    SpatialLatentEffectDataset,
    TraitEffectDataset,
    generate_fixed_effect_corpus,
    simulate_iid_latent_effect_dataset,
    simulate_spatial_latent_effect_dataset,
    simulate_fixed_effect_dataset,
    simulate_trait_effect_dataset,
)

__all__ = [
    "FixedEffectDataset",
    "FixedShapeBetaPosteriorModel",
    "IidLatentEffectDataset",
    "IidLatentFactorPosteriorModel",
    "SpatialLatentEffectDataset",
    "SpatialLatentFactorPosteriorModel",
    "TraitEffectDataset",
    "TraitGammaPosteriorModel",
    "VariableShapeBetaPosteriorModel",
    "BetaScaleCalibration",
    "apply_beta_scale_calibration",
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
    "generate_fixed_effect_corpus",
    "iid_latent_training_data",
    "predict_beta_posterior",
    "predict_gamma_posterior",
    "predict_iid_latent_posterior",
    "predict_spatial_latent_posterior",
    "predict_variable_beta_posterior",
    "simulate_fixed_effect_dataset",
    "simulate_iid_latent_effect_dataset",
    "simulate_spatial_latent_effect_dataset",
    "simulate_trait_effect_dataset",
    "spatial_holdout_random_effect_rmse",
    "spatial_latent_training_data",
    "train_fixed_shape_beta_model",
    "train_trait_gamma_model",
    "trait_effect_training_data",
    "variable_shape_training_data",
    "write_benchmark_report",
    "write_beta_posterior_hdf5",
    "write_gamma_posterior_hdf5",
    "write_iid_latent_posterior_hdf5",
    "write_spatial_latent_posterior_hdf5",
]


def __getattr__(name: str) -> object:
    if name == "FixedShapeBetaPosteriorModel":
        from pyhmsc.neural.models import FixedShapeBetaPosteriorModel

        return FixedShapeBetaPosteriorModel
    if name == "VariableShapeBetaPosteriorModel":
        from pyhmsc.neural.models import VariableShapeBetaPosteriorModel

        return VariableShapeBetaPosteriorModel
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
        "fixed_shape_training_data",
        "compiled_trait_effect_training_data",
        "iid_latent_training_data",
        "spatial_latent_training_data",
        "train_fixed_shape_beta_model",
        "train_trait_gamma_model",
        "trait_effect_training_data",
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
    if name == "write_iid_latent_posterior_hdf5":
        from pyhmsc.neural.storage import write_iid_latent_posterior_hdf5

        return write_iid_latent_posterior_hdf5
    if name == "write_spatial_latent_posterior_hdf5":
        from pyhmsc.neural.storage import write_spatial_latent_posterior_hdf5

        return write_spatial_latent_posterior_hdf5
    if name in {
        "BetaScaleCalibration",
        "apply_beta_scale_calibration",
        "calibration_metadata",
        "fit_beta_scale_calibration",
    }:
        from pyhmsc.neural.calibration import (
            BetaScaleCalibration,
            apply_beta_scale_calibration,
            calibration_metadata,
            fit_beta_scale_calibration,
        )

        return {
            "BetaScaleCalibration": BetaScaleCalibration,
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
    }:
        from pyhmsc.neural.benchmark import (
            compare_beta_posterior_files,
            compare_beta_posteriors,
            compare_gamma_posterior_files,
            compare_gamma_posteriors,
            compare_iid_association_posterior_files,
            compare_iid_association_posteriors,
            write_benchmark_report,
        )

        return {
            "compare_beta_posterior_files": compare_beta_posterior_files,
            "compare_beta_posteriors": compare_beta_posteriors,
            "compare_gamma_posterior_files": compare_gamma_posterior_files,
            "compare_gamma_posteriors": compare_gamma_posteriors,
            "compare_iid_association_posterior_files": compare_iid_association_posterior_files,
            "compare_iid_association_posteriors": compare_iid_association_posteriors,
            "write_benchmark_report": write_benchmark_report,
        }[name]
    raise AttributeError(f"module 'pyhmsc.neural' has no attribute {name!r}")
