"""Python user API for running Hmsc-HPC without hand-written R scripts."""

from pyhmsc.model import HmscModel
from pyhmsc.posterior import HmscFit
from pyhmsc.neural.ensemble import (
    DEFAULT_PREDICTIVE_MEAN_POLICY,
    PredictiveProbabilityEnsemble,
    load_predictive_mean_ensemble,
)
from pyhmsc.neural.deployment import (
    PROMOTED_PREDICTIVE_BASELINE_ID,
    load_predictive_deployment_baseline,
)
from pyhmsc.neural.release import (
    NEURAL_HMSC_RELEASE_ID,
    NeuralHmscRelease,
    load_neural_hmsc_release,
)
from pyhmsc.neural.variable_inference import (
    VARIABLE_SHAPE_BASELINE_ID,
    VariableShapeNeuralHmscInference,
    load_variable_shape_baseline,
)
from pyhmsc.compiler import CompiledModel, compile_hmsc_model
from pyhmsc.simulate import (
    apply_spatial_holdout_group_order,
    simulate_fixed_effect_data,
    simulate_random_slope_effect_data,
    simulate_spatial_multifactor_eta_effect_data,
    simulate_spatial_eta_effect_data,
    simulate_spatial_effect_data,
    simulate_spatial_holdout_data,
    simulate_spatial_random_slope_effect_data,
)

__all__ = [
    "HmscModel",
    "HmscFit",
    "DEFAULT_PREDICTIVE_MEAN_POLICY",
    "PredictiveProbabilityEnsemble",
    "load_predictive_mean_ensemble",
    "PROMOTED_PREDICTIVE_BASELINE_ID",
    "load_predictive_deployment_baseline",
    "NEURAL_HMSC_RELEASE_ID",
    "NeuralHmscRelease",
    "load_neural_hmsc_release",
    "VariableShapeNeuralHmscInference",
    "VARIABLE_SHAPE_BASELINE_ID",
    "load_variable_shape_baseline",
    "CompiledModel",
    "compile_hmsc_model",
    "apply_spatial_holdout_group_order",
    "simulate_fixed_effect_data",
    "simulate_random_slope_effect_data",
    "simulate_spatial_multifactor_eta_effect_data",
    "simulate_spatial_eta_effect_data",
    "simulate_spatial_effect_data",
    "simulate_spatial_holdout_data",
    "simulate_spatial_random_slope_effect_data",
]
