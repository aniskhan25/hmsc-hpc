"""Experimental neural HMSC utilities.

This namespace is intentionally separate from the stable HMSC sampler API. The
initial utilities support benchmark corpus generation for amortized Neural-HMSC
posterior inference.
"""

from pyhmsc.neural.simulator import (
    FixedEffectDataset,
    generate_fixed_effect_corpus,
    simulate_fixed_effect_dataset,
)

__all__ = [
    "FixedEffectDataset",
    "FixedShapeBetaPosteriorModel",
    "evaluate_beta_posterior",
    "fixed_shape_training_data",
    "generate_fixed_effect_corpus",
    "predict_beta_posterior",
    "simulate_fixed_effect_dataset",
    "train_fixed_shape_beta_model",
]


def __getattr__(name: str) -> object:
    if name == "FixedShapeBetaPosteriorModel":
        from pyhmsc.neural.models import FixedShapeBetaPosteriorModel

        return FixedShapeBetaPosteriorModel
    if name in {"fixed_shape_training_data", "train_fixed_shape_beta_model"}:
        from pyhmsc.neural.train import fixed_shape_training_data, train_fixed_shape_beta_model

        return {
            "fixed_shape_training_data": fixed_shape_training_data,
            "train_fixed_shape_beta_model": train_fixed_shape_beta_model,
        }[name]
    if name in {"evaluate_beta_posterior", "predict_beta_posterior"}:
        from pyhmsc.neural.evaluation import evaluate_beta_posterior, predict_beta_posterior

        return {
            "evaluate_beta_posterior": evaluate_beta_posterior,
            "predict_beta_posterior": predict_beta_posterior,
        }[name]
    raise AttributeError(f"module 'pyhmsc.neural' has no attribute {name!r}")
