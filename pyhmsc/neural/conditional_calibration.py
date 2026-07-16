"""Conditional coefficient-scale calibration for Neural-HMSC posteriors."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
import tensorflow as tf
from scipy.special import ndtr
from scipy.stats import norm

from pyhmsc.neural.calibration import BetaScaleCalibration, fit_beta_scale_calibration
from pyhmsc.neural.posterior_heads import BetaPosterior


_RAW_FEATURE_NAMES = ("prevalence_logit", "log_design_information", "log_raw_scale")
_CONDITIONAL_METHODS = {
    "conditional_structured_scale",
    "conditional_rank_aware_scale",
    "conditional_rank_aware_anchor_scale",
    "external_context_monotone_scale",
}
_OOD_EFFECT_SHIFT_HEAD_BASE_PARAMETER_COUNT = 8
_OOD_EFFECT_SHIFT_HEAD_PARAMETER_COUNT = 14
_OOD_EFFECT_SHIFT_PURE_LOG_CAP = float(np.log(1.16))
_OOD_EFFECT_SHIFT_COMBINED_LOG_CAP = float(np.log(1.18))
_OOD_EFFECT_SHIFT_PURE_HIGH_EFFECT_CENTER = 1.55
_OOD_EFFECT_SHIFT_PURE_HIGH_EFFECT_WIDTH = 0.30
_OOD_EFFECT_SHIFT_COMBINED_SUPPORT_CENTER = 0.25
_OOD_EFFECT_SHIFT_COMBINED_SUPPORT_WIDTH = 0.25
_OOD_EFFECT_SHIFT_BIN_CENTERS = (0.35, 0.85, 1.35)
_OOD_EFFECT_SHIFT_BIN_WIDTH = 0.28
_OOD_EFFECT_SHIFT_PURE_CONTEXT_SUPPORT_CENTER = 0.18
_OOD_EFFECT_SHIFT_PURE_CONTEXT_SUPPORT_WIDTH = 0.30
_OOD_EFFECT_SHIFT_PURE_CONTEXT_LOW_DESIGN_CENTER = 0.95
_OOD_EFFECT_SHIFT_PURE_CONTEXT_LOW_DESIGN_WIDTH = 0.35
_OOD_EFFECT_SHIFT_PURE_CONTEXT_FLOOR = 0.20
_OOD_EFFECT_SHIFT_COMBINED_CONTEXT_SUPPORT_CENTER = 0.20
_OOD_EFFECT_SHIFT_COMBINED_CONTEXT_SUPPORT_WIDTH = 0.35
_OOD_EFFECT_SHIFT_COMBINED_CONTEXT_LOW_DESIGN_CENTER = 0.75
_OOD_EFFECT_SHIFT_COMBINED_CONTEXT_LOW_DESIGN_WIDTH = 0.35
_OOD_EFFECT_SHIFT_COMBINED_CONTEXT_LOW_COMMUNITY_CENTER = 0.45
_OOD_EFFECT_SHIFT_COMBINED_CONTEXT_LOW_COMMUNITY_WIDTH = 0.06
_OOD_EFFECT_SHIFT_COMBINED_CONTEXT_FLOOR = 0.35
_OOD_EFFECT_SHIFT_BIN_IN_DOMAIN_CAP = float(np.log(1.025))
_COMBINED_SHIFT_SCALE_SUPPORT_CENTER = 0.20
_COMBINED_SHIFT_SCALE_SUPPORT_WIDTH = 0.35
_COMBINED_SHIFT_SCALE_EFFECT_CENTER = 0.25
_COMBINED_SHIFT_SCALE_EFFECT_WIDTH = 0.50
_COMBINED_SHIFT_SCALE_LOW_DESIGN_CENTER = 0.75
_COMBINED_SHIFT_SCALE_LOW_DESIGN_WIDTH = 0.35
_COMBINED_SHIFT_SCALE_LOW_COMMUNITY_CENTER = 0.45
_COMBINED_SHIFT_SCALE_LOW_COMMUNITY_WIDTH = 0.06
_COMBINED_SHIFT_CONTEXT_GATE_SUPPORT_WEIGHT = 0.95
_COMBINED_SHIFT_CONTEXT_GATE_EFFECT_WEIGHT = 0.45
_COMBINED_SHIFT_CONTEXT_GATE_LOW_DESIGN_WEIGHT = 0.35
_COMBINED_SHIFT_CONTEXT_GATE_LOW_COMMUNITY_WEIGHT = 1.15
_COMBINED_SHIFT_CONTEXT_GATE_INTERACTION_WEIGHT = 0.55
_COMBINED_SHIFT_CONTEXT_GATE_WIDTH = 1.0
_COMBINED_SHIFT_OBJECTIVE_COVERAGE_WEIGHT = 4.0
_COMBINED_SHIFT_OBJECTIVE_QUANTILE_WEIGHT = 2.5
_COMBINED_SHIFT_OBJECTIVE_CONTEXT_WEIGHT = 2.0
_COMBINED_SHIFT_OBJECTIVE_OVERLAP_WEIGHT = 0.20
_COMBINED_SHIFT_OBJECTIVE_OVERLAP_MEAN_CAP = 0.16
_COMBINED_SHIFT_OBJECTIVE_OVERLAP_ACTIVE_CAP = 0.08
_COMBINED_SHIFT_OBJECTIVE_WARMUP_FRACTION = 0.55
_COMBINED_SHIFT_OBJECTIVE_WARMUP_COVERAGE_BOOST = 1.75
_COMBINED_SHIFT_OBJECTIVE_WARMUP_GATE_FRACTION = 0.35
_COMBINED_SHIFT_SCALE_MAX_LOG = float(np.log(6.0))
_OOD_OBJECTIVES = {
    "none",
    "support_excess_rank_coverage",
    "support_effect_gated_rank_coverage",
}


@dataclass(frozen=True)
class ConditionalBetaOODCalibrationBatch:
    """Held-out OOD calibration data for coefficient-scale objectives."""

    posterior: BetaPosterior
    beta_true: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    label: str = "ood"
    weight: float = 1.0


@dataclass(frozen=True)
class ConditionalBetaScaleCalibration:
    """Serializable structured scale head fitted from simulation truth."""

    global_scale_multiplier: float
    normalization_multiplier: float
    feature_location: tuple[float, float, float]
    feature_scale: tuple[float, float, float]
    weights: tuple[float, ...]
    feature_names: tuple[str, ...]
    coefficient_names: tuple[str, ...]
    nominal_level: float
    uncalibrated_coverage: float
    calibrated_coverage: float
    n_observations: int
    distribution: str
    n_covariates: int
    n_species: int
    regularization: float
    epochs: int
    learning_rate: float
    scalar_nll: float
    conditional_nll: float
    scalar_rank_loss: float = 0.0
    conditional_rank_loss: float = 0.0
    prevalence_weights: tuple[float, float, float] = (4.0, 2.0, 1.0)
    prevalence_edges: tuple[float, float] = (0.1, 0.3)
    rank_penalty_weight: float = 0.02
    rank_mean_tolerance: float = 0.025
    rank_variance_tolerance: float = 0.015
    support_lower: tuple[float, float, float] = (-1e9, -1e9, -1e9)
    support_upper: tuple[float, float, float] = (1e9, 1e9, 1e9)
    support_precision: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    support_radius: float = 1e9
    support_quantile: float = 0.99
    fallback_strength: float = 2.0
    mean_magnitude_location: float = 0.0
    mean_magnitude_scale: float = 1.0
    mean_magnitude_lower: float = -1e9
    mean_magnitude_upper: float = 1e9
    mean_bias_correction: tuple[tuple[float, ...], ...] = ()
    mean_bias_shrinkage: float = 32.0
    rare_mean_head_n_observations: int = 0
    rare_mean_head_selected_shrinkage: float = 0.0
    rare_mean_head_validation_rank_error: float = 0.0
    rare_mean_head_diagnostics: dict[str, Any] = field(default_factory=dict)
    rank_centering_offsets: tuple[tuple[float, ...], ...] = ()
    rank_centering_shrinkage: float = 0.0
    base_scale_stratum_offsets: tuple[float, ...] = ()
    rare_validation_scale_log_offsets: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rare_validation_scale_selected_shrinkage: float = 0.0
    rare_validation_scale_support_threshold: float = 0.0
    rare_validation_scale_support_width: float = 1.0
    rare_validation_scale_community_threshold: float = 0.0
    rare_validation_scale_community_width: float = 1.0
    rare_validation_scale_diagnostics: dict[str, Any] = field(default_factory=dict)
    ood_uncertainty_strength: float = 0.0
    ood_uncertainty_max_multiplier: float = 1.0
    ood_objective: str = "none"
    ood_objective_weight: float = 0.0
    ood_in_domain_gate_weight: float = 0.0
    ood_inflation_parameters: tuple[float, ...] | None = None
    ood_objective_domains: tuple[str, ...] = ()
    ood_objective_n_observations: int = 0
    ood_objective_loss: float = 0.0
    ood_objective_rank_loss: float = 0.0
    ood_in_domain_gate_loss: float = 0.0
    ood_final_multiplier_diagnostics: dict[str, Any] = field(default_factory=dict)
    combined_shift_scale_log_amplitude: float = 0.0
    combined_shift_scale_effect_bin_edges: tuple[float, float] = (0.25, 1.0)
    combined_shift_scale_effect_bin_log_amplitudes: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    combined_shift_scale_context_gate_strength: float = 0.0
    combined_shift_scale_context_gate_intercept: float = 0.0
    combined_shift_scale_diagnostics: dict[str, Any] = field(default_factory=dict)
    external_monotone_log_offsets: tuple[float, float, float] = (0.0, 0.0, 0.0)
    external_monotone_effect_bin_edges: tuple[float, float] = (0.5, 1.25)
    external_monotone_support_threshold: float = 0.0
    external_monotone_support_width: float = 1.0
    external_monotone_effect_threshold: float = 0.75
    external_monotone_effect_width: float = 0.50
    external_monotone_selected_shrinkage: float = 0.0
    external_monotone_diagnostics: dict[str, Any] = field(default_factory=dict)
    min_multiplier: float = 0.1
    max_multiplier: float = 20.0
    method: str = "conditional_rank_aware_anchor_scale"

    @property
    def scale_multiplier(self) -> float:
        """Return the normalization term used around the conditional head."""
        return self.normalization_multiplier

    def validate_domain(
        self,
        *,
        distribution: str | None = None,
        n_covariates: int | None = None,
        n_species: int | None = None,
        coefficient_names: Sequence[str] | None = None,
    ) -> None:
        """Raise when application data do not match the fitted domain."""
        if (
            distribution is not None
            and str(distribution).lower() != self.distribution.lower()
        ):
            raise ValueError(
                "conditional calibration distribution mismatch: "
                f"expected {self.distribution!r}, got {distribution!r}"
            )
        if n_covariates is not None and int(n_covariates) != self.n_covariates:
            raise ValueError(
                "conditional calibration covariate dimension mismatch: "
                f"expected {self.n_covariates}, got {n_covariates}"
            )
        if n_species is not None and int(n_species) != self.n_species:
            raise ValueError(
                "conditional calibration species dimension mismatch: "
                f"expected {self.n_species}, got {n_species}"
            )
        if coefficient_names is not None:
            names = tuple(str(name) for name in coefficient_names)
            if names != self.coefficient_names:
                raise ValueError(
                    "conditional calibration coefficient names mismatch: "
                    f"expected {self.coefficient_names!r}, got {names!r}"
                )

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable conditional-calibration metadata."""
        semantics_version = {
            "conditional_structured_scale": 3,
            "conditional_rank_aware_scale": 4,
            "conditional_rank_aware_anchor_scale": 5,
            "external_context_monotone_scale": 9,
        }[self.method]
        if (
            self.method == "conditional_rank_aware_anchor_scale"
            and self.ood_uncertainty_strength > 0.0
            and self.ood_uncertainty_max_multiplier > 1.0
        ):
            semantics_version = 6
        if self.ood_inflation_parameters is not None:
            semantics_version = 7
            if len(self.ood_inflation_parameters) >= 7:
                semantics_version = 8
        if self.method == "external_context_monotone_scale":
            semantics_version = max(semantics_version, 9)
        n_covariates = int(self.n_covariates)
        ood_uncertainty = {
            "transform": "support_excess_exp",
            "strength": float(self.ood_uncertainty_strength),
            "max_multiplier": float(self.ood_uncertainty_max_multiplier),
        }
        if self.ood_inflation_parameters is not None:
            curve: dict[str, Any]
            if len(self.ood_inflation_parameters) >= 7:
                prevalence_gate_offsets = (0.0, 0.0, 0.0)
                design_gate_offsets = (0.0, 0.0, 0.0)
                coefficient_gate_offsets = (0.0,) * n_covariates
                effect_shift_head = ()
                if len(self.ood_inflation_parameters) >= 9:
                    (
                        offset,
                        support_linear,
                        support_quadratic,
                        effect_linear,
                        effect_quadratic,
                        effect_gate_intercept,
                        effect_gate_support_linear,
                        effect_gate_effect_linear,
                        effect_high_design_suppression,
                    ) = self.ood_inflation_parameters[:9]
                    extra_parameters = self.ood_inflation_parameters[9:]
                    if len(extra_parameters) >= 6:
                        prevalence_gate_offsets = tuple(extra_parameters[:3])
                        design_gate_offsets = tuple(extra_parameters[3:6])
                        coefficient_gate_offsets = tuple(
                            extra_parameters[6 : 6 + n_covariates]
                        )
                        if len(coefficient_gate_offsets) < n_covariates:
                            coefficient_gate_offsets = coefficient_gate_offsets + (
                                (0.0,) * (n_covariates - len(coefficient_gate_offsets))
                            )
                        effect_shift_head = tuple(
                            extra_parameters[
                                6
                                + n_covariates : 6
                                + n_covariates
                                + _OOD_EFFECT_SHIFT_HEAD_PARAMETER_COUNT
                            ]
                        )
                elif len(self.ood_inflation_parameters) >= 8:
                    (
                        offset,
                        support_linear,
                        support_quadratic,
                        effect_linear,
                        effect_quadratic,
                        effect_gate_intercept,
                        effect_gate_support_linear,
                        effect_gate_effect_linear,
                    ) = self.ood_inflation_parameters[:8]
                    effect_high_design_suppression = 0.0
                else:
                    (
                        offset,
                        support_linear,
                        support_quadratic,
                        effect_linear,
                        effect_quadratic,
                        effect_gate_intercept,
                        effect_gate_support_linear,
                    ) = self.ood_inflation_parameters[:7]
                    effect_gate_effect_linear = 0.0
                    effect_high_design_suppression = 0.0
                transform = "support_effect_gated_learned_softplus"
                curve = {
                    "offset": float(offset),
                    "support_linear": float(support_linear),
                    "support_quadratic": float(support_quadratic),
                    "effect_linear": float(effect_linear),
                    "effect_quadratic": float(effect_quadratic),
                    "effect_gate_intercept": float(effect_gate_intercept),
                    "effect_gate_support_linear": float(effect_gate_support_linear),
                    "effect_gate_effect_linear": float(effect_gate_effect_linear),
                    "effect_high_design_suppression": float(
                        effect_high_design_suppression
                    ),
                    "effect_prevalence_gate_offsets": [
                        float(value) for value in prevalence_gate_offsets
                    ],
                    "effect_design_gate_offsets": [
                        float(value) for value in design_gate_offsets
                    ],
                    "effect_coefficient_gate_offsets": [
                        float(value) for value in coefficient_gate_offsets
                    ],
                }
                if len(effect_shift_head) in (
                    _OOD_EFFECT_SHIFT_HEAD_BASE_PARAMETER_COUNT,
                    _OOD_EFFECT_SHIFT_HEAD_PARAMETER_COUNT,
                ):
                    (
                        pure_intercept,
                        pure_effect_linear,
                        pure_support_suppression,
                        pure_amplitude,
                        combined_intercept,
                        combined_effect_linear,
                        combined_support_linear,
                        combined_amplitude,
                    ) = effect_shift_head[:_OOD_EFFECT_SHIFT_HEAD_BASE_PARAMETER_COUNT]
                    pure_bin_amplitudes = (0.0, 0.0, 0.0)
                    combined_bin_amplitudes = (0.0, 0.0, 0.0)
                    if len(effect_shift_head) >= _OOD_EFFECT_SHIFT_HEAD_PARAMETER_COUNT:
                        pure_bin_amplitudes = tuple(effect_shift_head[8:11])
                        combined_bin_amplitudes = tuple(effect_shift_head[11:14])
                    curve["effect_shift_head"] = {
                        "kind": "constrained_context_gated_effect_quantile_scale",
                        "parameter_count": len(effect_shift_head),
                        "pure_effect_intercept": float(pure_intercept),
                        "pure_effect_linear": float(pure_effect_linear),
                        "pure_support_suppression": float(pure_support_suppression),
                        "pure_log_amplitude": float(pure_amplitude),
                        "pure_effect_bin_log_amplitudes": [
                            float(value) for value in pure_bin_amplitudes
                        ],
                        "pure_log_cap": _OOD_EFFECT_SHIFT_PURE_LOG_CAP,
                        "pure_high_effect_taper_center": (
                            _OOD_EFFECT_SHIFT_PURE_HIGH_EFFECT_CENTER
                        ),
                        "pure_high_effect_taper_width": (
                            _OOD_EFFECT_SHIFT_PURE_HIGH_EFFECT_WIDTH
                        ),
                        "combined_intercept": float(combined_intercept),
                        "combined_effect_linear": float(combined_effect_linear),
                        "combined_support_linear": float(combined_support_linear),
                        "combined_log_amplitude": float(combined_amplitude),
                        "combined_effect_bin_log_amplitudes": [
                            float(value) for value in combined_bin_amplitudes
                        ],
                        "combined_log_cap": _OOD_EFFECT_SHIFT_COMBINED_LOG_CAP,
                        "combined_support_gate_center": (
                            _OOD_EFFECT_SHIFT_COMBINED_SUPPORT_CENTER
                        ),
                        "combined_support_gate_width": (
                            _OOD_EFFECT_SHIFT_COMBINED_SUPPORT_WIDTH
                        ),
                        "effect_bin_centers": [
                            float(value) for value in _OOD_EFFECT_SHIFT_BIN_CENTERS
                        ],
                        "effect_bin_width": _OOD_EFFECT_SHIFT_BIN_WIDTH,
                    }
            elif len(self.ood_inflation_parameters) >= 5:
                (
                    offset,
                    support_linear,
                    support_quadratic,
                    effect_linear,
                    effect_quadratic,
                ) = self.ood_inflation_parameters[:5]
                transform = "support_effect_learned_softplus"
                curve = {
                    "offset": float(offset),
                    "support_linear": float(support_linear),
                    "support_quadratic": float(support_quadratic),
                    "effect_linear": float(effect_linear),
                    "effect_quadratic": float(effect_quadratic),
                }
            else:
                offset, support_linear, support_quadratic = (
                    self.ood_inflation_parameters
                )
                transform = "support_excess_learned_softplus"
                curve = {
                    "offset": float(offset),
                    "linear": float(support_linear),
                    "quadratic": float(support_quadratic),
                }
            ood_uncertainty.update(
                {
                    "transform": transform,
                    "curve": curve,
                }
            )
        return {
            "semantics_version": semantics_version,
            "method": self.method,
            "parameter": "Beta",
            "scale_multiplier": float(self.normalization_multiplier),
            "scale_multiplier_kind": "conditional_normalization",
            "global_scale_multiplier": float(self.global_scale_multiplier),
            "nominal_level": float(self.nominal_level),
            "uncalibrated_coverage": float(self.uncalibrated_coverage),
            "calibrated_coverage": float(self.calibrated_coverage),
            "n_observations": int(self.n_observations),
            "domain": {
                "distribution": self.distribution,
                "n_covariates": int(self.n_covariates),
                "n_species": int(self.n_species),
                "coefficient_names": list(self.coefficient_names),
            },
            "features": {
                "raw_names": list(_RAW_FEATURE_NAMES),
                "design_names": list(self.feature_names),
                "location": list(self.feature_location),
                "scale": list(self.feature_scale),
            },
            "weights": list(self.weights),
            "training": {
                "regularization": float(self.regularization),
                "epochs": int(self.epochs),
                "learning_rate": float(self.learning_rate),
                "scalar_nll": float(self.scalar_nll),
                "conditional_nll": float(self.conditional_nll),
                "scalar_rank_loss": float(self.scalar_rank_loss),
                "conditional_rank_loss": float(self.conditional_rank_loss),
            },
            "ood_objective": {
                "name": self.ood_objective,
                "weight": float(self.ood_objective_weight),
                "in_domain_gate_weight": float(self.ood_in_domain_gate_weight),
                "domains": list(self.ood_objective_domains),
                "n_observations": int(self.ood_objective_n_observations),
                "loss": float(self.ood_objective_loss),
                "rank_loss": float(self.ood_objective_rank_loss),
                "in_domain_gate_loss": float(self.ood_in_domain_gate_loss),
                "combined_shift_training_objective": {
                    "kind": "final_multiplier_aware_combined_shift_coverage",
                    "coverage_weight": _COMBINED_SHIFT_OBJECTIVE_COVERAGE_WEIGHT,
                    "effect_quantile_weight": (
                        _COMBINED_SHIFT_OBJECTIVE_QUANTILE_WEIGHT
                    ),
                    "context_weight": _COMBINED_SHIFT_OBJECTIVE_CONTEXT_WEIGHT,
                    "in_domain_overlap_weight": (
                        _COMBINED_SHIFT_OBJECTIVE_OVERLAP_WEIGHT
                    ),
                    "in_domain_overlap_mean_cap": (
                        _COMBINED_SHIFT_OBJECTIVE_OVERLAP_MEAN_CAP
                    ),
                    "in_domain_overlap_active_fraction_cap": (
                        _COMBINED_SHIFT_OBJECTIVE_OVERLAP_ACTIVE_CAP
                    ),
                    "schedule": {
                        "kind": "coverage_warmup_then_overlap_ramp",
                        "warmup_fraction": (_COMBINED_SHIFT_OBJECTIVE_WARMUP_FRACTION),
                        "warmup_coverage_boost": (
                            _COMBINED_SHIFT_OBJECTIVE_WARMUP_COVERAGE_BOOST
                        ),
                        "warmup_gate_fraction": (
                            _COMBINED_SHIFT_OBJECTIVE_WARMUP_GATE_FRACTION
                        ),
                    },
                },
                "final_multiplier_diagnostics": self.ood_final_multiplier_diagnostics,
                "combined_shift_scale": {
                    "kind": "domain_specific_combined_shift_log_multiplier",
                    "log_amplitude": float(self.combined_shift_scale_log_amplitude),
                    "multiplier": float(
                        np.exp(float(self.combined_shift_scale_log_amplitude))
                    ),
                    "effect_bin_edges": [
                        float(value)
                        for value in self.combined_shift_scale_effect_bin_edges
                    ],
                    "effect_bin_log_amplitudes": [
                        float(value)
                        for value in self.combined_shift_scale_effect_bin_log_amplitudes
                    ],
                    "effect_bin_multipliers": [
                        float(np.exp(float(value)))
                        for value in self.combined_shift_scale_effect_bin_log_amplitudes
                    ],
                    "context_gate": {
                        "kind": "support_effect_low_design_low_community_classifier",
                        "strength": float(
                            self.combined_shift_scale_context_gate_strength
                        ),
                        "intercept": float(
                            self.combined_shift_scale_context_gate_intercept
                        ),
                        "width": _COMBINED_SHIFT_CONTEXT_GATE_WIDTH,
                        "support_weight": (_COMBINED_SHIFT_CONTEXT_GATE_SUPPORT_WEIGHT),
                        "effect_weight": _COMBINED_SHIFT_CONTEXT_GATE_EFFECT_WEIGHT,
                        "low_design_weight": (
                            _COMBINED_SHIFT_CONTEXT_GATE_LOW_DESIGN_WEIGHT
                        ),
                        "low_community_weight": (
                            _COMBINED_SHIFT_CONTEXT_GATE_LOW_COMMUNITY_WEIGHT
                        ),
                        "support_low_community_interaction_weight": (
                            _COMBINED_SHIFT_CONTEXT_GATE_INTERACTION_WEIGHT
                        ),
                    },
                    "max_log_amplitude": _COMBINED_SHIFT_SCALE_MAX_LOG,
                    "activation": {
                        "support_center": _COMBINED_SHIFT_SCALE_SUPPORT_CENTER,
                        "support_width": _COMBINED_SHIFT_SCALE_SUPPORT_WIDTH,
                        "effect_center": _COMBINED_SHIFT_SCALE_EFFECT_CENTER,
                        "effect_width": _COMBINED_SHIFT_SCALE_EFFECT_WIDTH,
                        "low_design_center": (_COMBINED_SHIFT_SCALE_LOW_DESIGN_CENTER),
                        "low_design_width": _COMBINED_SHIFT_SCALE_LOW_DESIGN_WIDTH,
                        "low_community_center": (
                            _COMBINED_SHIFT_SCALE_LOW_COMMUNITY_CENTER
                        ),
                        "low_community_width": (
                            _COMBINED_SHIFT_SCALE_LOW_COMMUNITY_WIDTH
                        ),
                    },
                    "diagnostics": self.combined_shift_scale_diagnostics,
                },
            },
            "rank_aware": {
                "prevalence_weights": list(self.prevalence_weights),
                "prevalence_edges": list(self.prevalence_edges),
                "penalty_weight": float(self.rank_penalty_weight),
                "mean_tolerance": float(self.rank_mean_tolerance),
                "variance_tolerance": float(self.rank_variance_tolerance),
            },
            "mean_bias_correction": {
                "kind": "prevalence_coefficient_residual_mean",
                "prevalence_edges": list(self.prevalence_edges),
                "shrinkage": float(self.mean_bias_shrinkage),
                "rare_balanced_n_observations": int(self.rare_mean_head_n_observations),
                "rare_balanced_selected_shrinkage": float(
                    self.rare_mean_head_selected_shrinkage
                ),
                "rare_balanced_validation_rank_error": float(
                    self.rare_mean_head_validation_rank_error
                ),
                "rare_balanced_diagnostics": self.rare_mean_head_diagnostics,
                "values": [list(row) for row in self.mean_bias_correction],
            },
            "rank_centering": {
                "kind": "heldout_prevalence_coefficient_standardized_shift",
                "prevalence_edges": list(self.prevalence_edges),
                "selected_shrinkage": float(self.rank_centering_shrinkage),
                "values": [list(row) for row in self.rank_centering_offsets],
            },
            "base_scale_strata": {
                "kind": "prevalence_design_coefficient_log_offsets",
                "prevalence_offsets": [
                    float(value) for value in self.base_scale_stratum_offsets[:3]
                ],
                "design_offsets": [
                    float(value) for value in self.base_scale_stratum_offsets[3:6]
                ],
                "coefficient_offsets": [
                    float(value)
                    for value in self.base_scale_stratum_offsets[
                        6 : 6 + int(self.n_covariates)
                    ]
                ],
            },
            "rare_validation_scale": {
                "kind": "rare_validation_design_log_multiplier",
                "prevalence_edges": list(self.prevalence_edges),
                "design_strata": ["low", "intermediate", "high"],
                "selected_shrinkage": float(
                    self.rare_validation_scale_selected_shrinkage
                ),
                "log_offsets": [
                    float(value) for value in self.rare_validation_scale_log_offsets
                ],
                "multipliers": [
                    float(np.exp(value))
                    for value in self.rare_validation_scale_log_offsets
                ],
                "activation": {
                    "kind": "thresholded_low_community_or_support_excess",
                    "threshold": float(self.rare_validation_scale_support_threshold),
                    "width": float(self.rare_validation_scale_support_width),
                    "community_occupancy_threshold": float(
                        self.rare_validation_scale_community_threshold
                    ),
                    "community_occupancy_width": float(
                        self.rare_validation_scale_community_width
                    ),
                },
                "diagnostics": self.rare_validation_scale_diagnostics,
            },
            "external_context_monotone": {
                "kind": "heldout_context_stratified_monotone_scale",
                "effect_bin_edges": [
                    float(value) for value in self.external_monotone_effect_bin_edges
                ],
                "effect_bin_log_offsets": [
                    float(value) for value in self.external_monotone_log_offsets
                ],
                "effect_bin_multipliers": [
                    float(np.exp(float(value)))
                    for value in self.external_monotone_log_offsets
                ],
                "activation": {
                    "kind": "support_or_effect_ramp",
                    "support_threshold": float(
                        self.external_monotone_support_threshold
                    ),
                    "support_width": float(self.external_monotone_support_width),
                    "effect_threshold": float(
                        self.external_monotone_effect_threshold
                    ),
                    "effect_width": float(self.external_monotone_effect_width),
                },
                "selected_shrinkage": float(
                    self.external_monotone_selected_shrinkage
                ),
                "diagnostics": self.external_monotone_diagnostics,
            },
            "support": {
                "lower": list(self.support_lower),
                "upper": list(self.support_upper),
                "precision": [list(row) for row in self.support_precision],
                "radius": float(self.support_radius),
                "quantile": float(self.support_quantile),
                "fallback_strength": float(self.fallback_strength),
                "fallback_multiplier": float(self.global_scale_multiplier),
                "blend_space": "log_scale",
                "mean_magnitude": {
                    "transform": "log1p_abs",
                    "location": float(self.mean_magnitude_location),
                    "scale": float(self.mean_magnitude_scale),
                    "lower": float(self.mean_magnitude_lower),
                    "upper": float(self.mean_magnitude_upper),
                },
                "ood_uncertainty": {
                    **ood_uncertainty,
                },
            },
            "multiplier_bounds": [
                float(self.min_multiplier),
                float(self.max_multiplier),
            ],
        }

    @classmethod
    def from_metadata(
        cls, metadata: dict[str, Any]
    ) -> "ConditionalBetaScaleCalibration":
        """Reconstruct a conditional calibrator from stored metadata."""
        method = str(metadata.get("method"))
        if method not in _CONDITIONAL_METHODS:
            raise ValueError(
                "metadata does not describe conditional structured scaling"
            )
        domain = metadata["domain"]
        features = metadata["features"]
        training = metadata["training"]
        bounds = metadata.get("multiplier_bounds", (0.1, 20.0))
        raw_names = tuple(str(value) for value in features["raw_names"])
        if raw_names != _RAW_FEATURE_NAMES:
            raise ValueError(
                "conditional calibration raw feature specification mismatch"
            )
        feature_names = tuple(str(value) for value in features["design_names"])
        weights = tuple(float(value) for value in metadata["weights"])
        if len(weights) != len(feature_names):
            raise ValueError("conditional calibration weights do not match features")
        location = tuple(float(value) for value in features["location"])
        feature_scale = tuple(float(value) for value in features["scale"])
        if len(location) != 3 or len(feature_scale) != 3:
            raise ValueError("conditional calibration requires three raw features")
        rank_aware = metadata.get("rank_aware", {})
        support = metadata.get("support", {})
        support_lower = tuple(
            float(value) for value in support.get("lower", (-1e9,) * 3)
        )
        support_upper = tuple(
            float(value) for value in support.get("upper", (1e9,) * 3)
        )
        support_precision = tuple(
            tuple(float(value) for value in row)
            for row in support.get("precision", np.eye(3).tolist())
        )
        if (
            len(support_lower) != 3
            or len(support_upper) != 3
            or len(support_precision) != 3
            or any(len(row) != 3 for row in support_precision)
        ):
            raise ValueError(
                "conditional calibration support must have three dimensions"
            )
        mean_support = support.get("mean_magnitude", {})
        ood_uncertainty = support.get("ood_uncertainty", {})
        ood_objective = metadata.get("ood_objective", {})
        ood_curve = ood_uncertainty.get("curve")
        ood_inflation_parameters = None
        if isinstance(ood_curve, dict):
            if (
                "effect_gate_intercept" in ood_curve
                or "effect_gate_support_linear" in ood_curve
                or "effect_gate_effect_linear" in ood_curve
            ):
                ood_inflation_parameters = (
                    float(ood_curve["offset"]),
                    float(ood_curve["support_linear"]),
                    float(ood_curve["support_quadratic"]),
                    float(ood_curve["effect_linear"]),
                    float(ood_curve["effect_quadratic"]),
                    float(ood_curve["effect_gate_intercept"]),
                    float(ood_curve["effect_gate_support_linear"]),
                    float(ood_curve.get("effect_gate_effect_linear", 0.0)),
                    float(ood_curve.get("effect_high_design_suppression", 0.0)),
                )
                prevalence_gate_offsets = tuple(
                    float(value)
                    for value in ood_curve.get(
                        "effect_prevalence_gate_offsets", (0.0, 0.0, 0.0)
                    )
                )
                design_gate_offsets = tuple(
                    float(value)
                    for value in ood_curve.get(
                        "effect_design_gate_offsets", (0.0, 0.0, 0.0)
                    )
                )
                coefficient_gate_offsets = tuple(
                    float(value)
                    for value in ood_curve.get(
                        "effect_coefficient_gate_offsets",
                        (0.0,) * int(domain["n_covariates"]),
                    )
                )
                if len(prevalence_gate_offsets) != 3:
                    raise ValueError(
                        "version 8 prevalence gate offsets must have length three"
                    )
                if len(design_gate_offsets) != 3:
                    raise ValueError(
                        "version 8 design gate offsets must have length three"
                    )
                if len(coefficient_gate_offsets) != int(domain["n_covariates"]):
                    raise ValueError(
                        "version 8 coefficient gate offsets must match n_covariates"
                    )
                effect_shift_head = ()
                effect_shift_head_metadata = ood_curve.get("effect_shift_head")
                if isinstance(effect_shift_head_metadata, dict):
                    pure_bin_amplitudes = tuple(
                        float(value)
                        for value in effect_shift_head_metadata.get(
                            "pure_effect_bin_log_amplitudes", ()
                        )
                    )
                    combined_bin_amplitudes = tuple(
                        float(value)
                        for value in effect_shift_head_metadata.get(
                            "combined_effect_bin_log_amplitudes", ()
                        )
                    )
                    if pure_bin_amplitudes or combined_bin_amplitudes:
                        if len(pure_bin_amplitudes) != 3:
                            raise ValueError(
                                "pure effect-bin log amplitudes must have length three"
                            )
                        if len(combined_bin_amplitudes) != 3:
                            raise ValueError(
                                "combined effect-bin log amplitudes must have length three"
                            )
                    effect_shift_head = (
                        float(effect_shift_head_metadata["pure_effect_intercept"]),
                        float(effect_shift_head_metadata["pure_effect_linear"]),
                        float(effect_shift_head_metadata["pure_support_suppression"]),
                        float(effect_shift_head_metadata["pure_log_amplitude"]),
                        float(effect_shift_head_metadata["combined_intercept"]),
                        float(effect_shift_head_metadata["combined_effect_linear"]),
                        float(effect_shift_head_metadata["combined_support_linear"]),
                        float(effect_shift_head_metadata["combined_log_amplitude"]),
                    )
                    if pure_bin_amplitudes or combined_bin_amplitudes:
                        effect_shift_head = (
                            effect_shift_head
                            + pure_bin_amplitudes
                            + combined_bin_amplitudes
                        )
                ood_inflation_parameters = (
                    ood_inflation_parameters
                    + prevalence_gate_offsets
                    + design_gate_offsets
                    + coefficient_gate_offsets
                    + effect_shift_head
                )
            elif "effect_linear" in ood_curve or "effect_quadratic" in ood_curve:
                ood_inflation_parameters = (
                    float(ood_curve["offset"]),
                    float(ood_curve["support_linear"]),
                    float(ood_curve["support_quadratic"]),
                    float(ood_curve["effect_linear"]),
                    float(ood_curve["effect_quadratic"]),
                )
            else:
                ood_inflation_parameters = (
                    float(ood_curve["offset"]),
                    float(ood_curve["linear"]),
                    float(ood_curve["quadratic"]),
                )
        bias_metadata = metadata.get("mean_bias_correction", {})
        bias_values = bias_metadata.get(
            "values", [[0.0] * int(domain["n_covariates"]) for _ in range(3)]
        )
        mean_bias_correction = tuple(
            tuple(float(value) for value in row) for row in bias_values
        )
        if len(mean_bias_correction) != 3 or any(
            len(row) != int(domain["n_covariates"]) for row in mean_bias_correction
        ):
            raise ValueError(
                "mean bias correction must have shape prevalence strata x covariates"
            )
        rank_centering_metadata = metadata.get("rank_centering", {})
        rank_centering_values = rank_centering_metadata.get(
            "values", [[0.0] * int(domain["n_covariates"]) for _ in range(3)]
        )
        rank_centering_offsets = tuple(
            tuple(float(value) for value in row) for row in rank_centering_values
        )
        if len(rank_centering_offsets) != 3 or any(
            len(row) != int(domain["n_covariates"]) for row in rank_centering_offsets
        ):
            raise ValueError(
                "rank centering offsets must have shape prevalence strata x covariates"
            )
        base_scale_strata = metadata.get("base_scale_strata", {})
        base_prevalence_offsets = tuple(
            float(value)
            for value in base_scale_strata.get("prevalence_offsets", (0.0, 0.0, 0.0))
        )
        base_design_offsets = tuple(
            float(value)
            for value in base_scale_strata.get("design_offsets", (0.0, 0.0, 0.0))
        )
        base_coefficient_offsets = tuple(
            float(value)
            for value in base_scale_strata.get(
                "coefficient_offsets", (0.0,) * int(domain["n_covariates"])
            )
        )
        if len(base_prevalence_offsets) != 3 or len(base_design_offsets) != 3:
            raise ValueError(
                "base scale prevalence/design offsets must have length three"
            )
        if len(base_coefficient_offsets) != int(domain["n_covariates"]):
            raise ValueError("base scale coefficient offsets must match n_covariates")
        base_scale_stratum_offsets = (
            base_prevalence_offsets + base_design_offsets + base_coefficient_offsets
        )
        rare_validation_scale = metadata.get("rare_validation_scale", {})
        rare_validation_scale_log_offsets = tuple(
            float(value)
            for value in rare_validation_scale.get("log_offsets", (0.0, 0.0, 0.0))
        )
        if len(rare_validation_scale_log_offsets) != 3:
            raise ValueError("rare validation scale log offsets must have length three")
        rare_validation_activation = rare_validation_scale.get("activation", {})
        combined_shift_scale = ood_objective.get("combined_shift_scale", {})
        combined_shift_effect_bin_edges = tuple(
            float(value)
            for value in combined_shift_scale.get("effect_bin_edges", (0.25, 1.0))
        )
        if len(combined_shift_effect_bin_edges) != 2:
            raise ValueError("combined-shift effect bin edges must have length two")
        combined_shift_effect_bin_log_amplitudes = tuple(
            float(value)
            for value in combined_shift_scale.get(
                "effect_bin_log_amplitudes", (0.0, 0.0, 0.0)
            )
        )
        if len(combined_shift_effect_bin_log_amplitudes) != 3:
            raise ValueError(
                "combined-shift effect-bin amplitudes must have length three"
            )
        combined_shift_context_gate = combined_shift_scale.get("context_gate", {})
        external_monotone = metadata.get("external_context_monotone", {})
        external_monotone_effect_bin_edges = tuple(
            float(value)
            for value in external_monotone.get("effect_bin_edges", (0.5, 1.25))
        )
        if len(external_monotone_effect_bin_edges) != 2:
            raise ValueError("external monotone effect bin edges must have length two")
        external_monotone_log_offsets = tuple(
            float(value)
            for value in external_monotone.get(
                "effect_bin_log_offsets", (0.0, 0.0, 0.0)
            )
        )
        if len(external_monotone_log_offsets) != 3:
            raise ValueError("external monotone log offsets must have length three")
        external_monotone_activation = external_monotone.get("activation", {})
        return cls(
            global_scale_multiplier=float(metadata["global_scale_multiplier"]),
            normalization_multiplier=float(metadata["scale_multiplier"]),
            feature_location=location,
            feature_scale=feature_scale,
            weights=weights,
            feature_names=feature_names,
            coefficient_names=tuple(
                str(value) for value in domain["coefficient_names"]
            ),
            nominal_level=float(metadata["nominal_level"]),
            uncalibrated_coverage=float(metadata["uncalibrated_coverage"]),
            calibrated_coverage=float(metadata["calibrated_coverage"]),
            n_observations=int(metadata["n_observations"]),
            distribution=str(domain["distribution"]),
            n_covariates=int(domain["n_covariates"]),
            n_species=int(domain["n_species"]),
            regularization=float(training["regularization"]),
            epochs=int(training["epochs"]),
            learning_rate=float(training["learning_rate"]),
            scalar_nll=float(training["scalar_nll"]),
            conditional_nll=float(training["conditional_nll"]),
            scalar_rank_loss=float(training.get("scalar_rank_loss", 0.0)),
            conditional_rank_loss=float(training.get("conditional_rank_loss", 0.0)),
            prevalence_weights=tuple(
                float(value)
                for value in rank_aware.get("prevalence_weights", (1.0, 1.0, 1.0))
            ),
            prevalence_edges=tuple(
                float(value) for value in rank_aware.get("prevalence_edges", (0.1, 0.3))
            ),
            rank_penalty_weight=float(rank_aware.get("penalty_weight", 0.0)),
            rank_mean_tolerance=float(rank_aware.get("mean_tolerance", 0.025)),
            rank_variance_tolerance=float(rank_aware.get("variance_tolerance", 0.015)),
            support_lower=support_lower,
            support_upper=support_upper,
            support_precision=support_precision,
            support_radius=float(support.get("radius", 1e9)),
            support_quantile=float(support.get("quantile", 1.0)),
            fallback_strength=float(support.get("fallback_strength", 0.0)),
            mean_magnitude_location=float(mean_support.get("location", 0.0)),
            mean_magnitude_scale=float(mean_support.get("scale", 1.0)),
            mean_magnitude_lower=float(mean_support.get("lower", -1e9)),
            mean_magnitude_upper=float(mean_support.get("upper", 1e9)),
            mean_bias_correction=mean_bias_correction,
            mean_bias_shrinkage=float(bias_metadata.get("shrinkage", 32.0)),
            rare_mean_head_n_observations=int(
                bias_metadata.get("rare_balanced_n_observations", 0)
            ),
            rare_mean_head_selected_shrinkage=float(
                bias_metadata.get("rare_balanced_selected_shrinkage", 0.0)
            ),
            rare_mean_head_validation_rank_error=float(
                bias_metadata.get("rare_balanced_validation_rank_error", 0.0)
            ),
            rare_mean_head_diagnostics=dict(
                bias_metadata.get("rare_balanced_diagnostics", {})
            ),
            rank_centering_offsets=rank_centering_offsets,
            rank_centering_shrinkage=float(
                rank_centering_metadata.get("selected_shrinkage", 0.0)
            ),
            base_scale_stratum_offsets=base_scale_stratum_offsets,
            rare_validation_scale_log_offsets=rare_validation_scale_log_offsets,
            rare_validation_scale_selected_shrinkage=float(
                rare_validation_scale.get("selected_shrinkage", 0.0)
            ),
            rare_validation_scale_support_threshold=float(
                rare_validation_activation.get("threshold", 0.0)
            ),
            rare_validation_scale_support_width=float(
                rare_validation_activation.get("width", 1.0)
            ),
            rare_validation_scale_community_threshold=float(
                rare_validation_activation.get("community_occupancy_threshold", 0.0)
            ),
            rare_validation_scale_community_width=float(
                rare_validation_activation.get("community_occupancy_width", 1.0)
            ),
            rare_validation_scale_diagnostics=dict(
                rare_validation_scale.get("diagnostics", {})
            ),
            ood_uncertainty_strength=float(ood_uncertainty.get("strength", 0.0)),
            ood_uncertainty_max_multiplier=float(
                ood_uncertainty.get("max_multiplier", 1.0)
            ),
            ood_objective=str(ood_objective.get("name", "none")),
            ood_objective_weight=float(ood_objective.get("weight", 0.0)),
            ood_in_domain_gate_weight=float(
                ood_objective.get("in_domain_gate_weight", 0.0)
            ),
            ood_inflation_parameters=ood_inflation_parameters,
            ood_objective_domains=tuple(
                str(value) for value in ood_objective.get("domains", ())
            ),
            ood_objective_n_observations=int(ood_objective.get("n_observations", 0)),
            ood_objective_loss=float(ood_objective.get("loss", 0.0)),
            ood_objective_rank_loss=float(ood_objective.get("rank_loss", 0.0)),
            ood_in_domain_gate_loss=float(
                ood_objective.get("in_domain_gate_loss", 0.0)
            ),
            ood_final_multiplier_diagnostics=dict(
                ood_objective.get("final_multiplier_diagnostics", {})
            ),
            combined_shift_scale_log_amplitude=float(
                combined_shift_scale.get("log_amplitude", 0.0)
            ),
            combined_shift_scale_effect_bin_edges=combined_shift_effect_bin_edges,
            combined_shift_scale_effect_bin_log_amplitudes=(
                combined_shift_effect_bin_log_amplitudes
            ),
            combined_shift_scale_context_gate_strength=float(
                combined_shift_context_gate.get("strength", 0.0)
            ),
            combined_shift_scale_context_gate_intercept=float(
                combined_shift_context_gate.get("intercept", 0.0)
            ),
            combined_shift_scale_diagnostics=dict(
                combined_shift_scale.get("diagnostics", {})
            ),
            external_monotone_log_offsets=external_monotone_log_offsets,
            external_monotone_effect_bin_edges=external_monotone_effect_bin_edges,
            external_monotone_support_threshold=float(
                external_monotone_activation.get("support_threshold", 0.0)
            ),
            external_monotone_support_width=float(
                external_monotone_activation.get("support_width", 1.0)
            ),
            external_monotone_effect_threshold=float(
                external_monotone_activation.get("effect_threshold", 0.75)
            ),
            external_monotone_effect_width=float(
                external_monotone_activation.get("effect_width", 0.50)
            ),
            external_monotone_selected_shrinkage=float(
                external_monotone.get("selected_shrinkage", 0.0)
            ),
            external_monotone_diagnostics=dict(
                external_monotone.get("diagnostics", {})
            ),
            min_multiplier=float(bounds[0]),
            max_multiplier=float(bounds[1]),
            method=method,
        )


def fit_conditional_beta_scale_calibration(
    posterior: BetaPosterior,
    beta_true: np.ndarray,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    distribution: str,
    coefficient_names: Sequence[str] | None = None,
    baseline_calibration: BetaScaleCalibration | None = None,
    nominal_level: float = 0.95,
    species_mask: np.ndarray | None = None,
    regularization: float = 1e-3,
    epochs: int = 400,
    learning_rate: float = 0.03,
    prevalence_weights: tuple[float, float, float] = (4.0, 2.0, 1.0),
    prevalence_edges: tuple[float, float] = (0.1, 0.3),
    rank_penalty_weight: float = 0.02,
    rank_mean_tolerance: float = 0.025,
    rank_variance_tolerance: float = 0.015,
    support_quantile: float = 0.99,
    fallback_strength: float = 2.0,
    ood_uncertainty_strength: float = 0.75,
    ood_uncertainty_max_multiplier: float = 4.0,
    ood_calibration_batches: Sequence[ConditionalBetaOODCalibrationBatch] | None = None,
    rare_calibration_batches: (
        Sequence[ConditionalBetaOODCalibrationBatch] | None
    ) = None,
    rare_validation_batches: Sequence[ConditionalBetaOODCalibrationBatch] | None = None,
    ood_objective: str = "none",
    ood_objective_weight: float = 1.0,
    ood_in_domain_gate_weight: float = 10.0,
    ood_objective_epochs: int | None = None,
    support_ridge: float = 1e-4,
    min_multiplier: float = 0.1,
    max_multiplier: float = 20.0,
) -> ConditionalBetaScaleCalibration:
    """Fit a structured conditional scale head on simulated calibration truth.

    The head combines prevalence-weighted Gaussian log score with analytic SBC
    rank-moment penalties. A final scalar normalization restores nominal
    marginal coverage, while a feature-support gate falls back to the frozen
    scalar multiplier under covariate shift.
    """
    if not 0.0 < nominal_level < 1.0:
        raise ValueError("nominal_level must be between zero and one")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if len(prevalence_weights) != 3 or any(
        value <= 0.0 for value in prevalence_weights
    ):
        raise ValueError("prevalence_weights must contain three positive values")
    low_prevalence, high_prevalence = (float(value) for value in prevalence_edges)
    if not 0.0 < low_prevalence < high_prevalence < 1.0:
        raise ValueError("prevalence_edges must be ordered values between zero and one")
    if rank_penalty_weight < 0.0:
        raise ValueError("rank_penalty_weight must be non-negative")
    if rank_mean_tolerance <= 0.0 or rank_variance_tolerance <= 0.0:
        raise ValueError("rank tolerances must be positive")
    if not 0.5 < support_quantile < 1.0:
        raise ValueError("support_quantile must be between 0.5 and 1")
    if fallback_strength < 0.0 or support_ridge <= 0.0:
        raise ValueError(
            "fallback_strength must be non-negative and support_ridge positive"
        )
    if ood_uncertainty_strength < 0.0 or ood_uncertainty_max_multiplier < 1.0:
        raise ValueError(
            "ood uncertainty strength must be non-negative and max multiplier at least one"
        )
    if ood_objective not in _OOD_OBJECTIVES:
        raise ValueError(f"unsupported OOD objective: {ood_objective!r}")
    if ood_objective_weight < 0.0 or ood_in_domain_gate_weight < 0.0:
        raise ValueError("OOD objective weights must be non-negative")
    if ood_objective_epochs is not None and ood_objective_epochs <= 0:
        raise ValueError("ood_objective_epochs must be positive")
    if not 0.0 < min_multiplier < max_multiplier:
        raise ValueError("multiplier bounds must be positive and ordered")

    mean, scale, design, response = _validated_arrays(posterior, X=X, Y=Y)
    truth = np.asarray(beta_true, dtype=float)
    if truth.shape != mean.shape or not np.all(np.isfinite(truth)):
        raise ValueError("beta_true must be finite and match the posterior shape")
    names = _coefficient_names(coefficient_names, mean.shape[1])
    mask = _coefficient_mask(mean.shape, species_mask)
    if not np.any(mask):
        raise ValueError("calibration mask selects no coefficients")
    prevalence = _prevalence(response)
    prevalence_by_coefficient = np.broadcast_to(prevalence[:, None, :], mean.shape)
    coefficient_index = np.broadcast_to(
        np.arange(mean.shape[1])[None, :, None], mean.shape
    )
    prevalence_stratum = _prevalence_stratum_index(
        prevalence_by_coefficient,
        prevalence_edges=(low_prevalence, high_prevalence),
    )
    z_value = float(norm.ppf(0.5 + nominal_level / 2.0))
    mean_bias_shrinkage = 32.0
    rare_mean_head_stats: dict[str, Any] = {
        "n_observations": 0,
        "selected_shrinkage": 0.0,
        "validation_rank_error": 0.0,
        "diagnostics": {},
    }
    if rare_calibration_batches:
        (
            mean_bias_correction,
            rare_mean_head_stats,
        ) = _fit_rare_balanced_mean_bias_correction(
            rare_calibration_batches=rare_calibration_batches,
            rare_validation_batches=rare_validation_batches,
            validation_truth=truth,
            validation_mean=mean,
            validation_scale=scale,
            validation_design=design,
            validation_response=response,
            validation_mask=mask,
            validation_prevalence_stratum=prevalence_stratum,
            validation_coefficient_stratum=coefficient_index,
            distribution=distribution,
            prevalence_edges=(low_prevalence, high_prevalence),
            shrinkage=mean_bias_shrinkage,
            nominal_level=nominal_level,
            z_value=z_value,
        )
    else:
        fit_mean_bias_correction = False
        mean_bias_correction = (
            _fit_prevalence_coefficient_mean_bias(
                truth=truth,
                mean=mean,
                scale=scale,
                prevalence_stratum=prevalence_stratum,
                coefficient_stratum=coefficient_index,
                mask=mask,
                shrinkage=mean_bias_shrinkage,
            )
            if fit_mean_bias_correction
            else tuple(tuple(0.0 for _ in range(mean.shape[1])) for _ in range(3))
        )
    mean_bias_array = np.asarray(mean_bias_correction, dtype=float)[
        prevalence_stratum, coefficient_index
    ]
    fit_rank_centering = False
    (
        rank_centering_offsets,
        rank_centering_shrinkage,
    ) = (
        _fit_rank_centering_offsets(
            truth=truth,
            mean=mean + mean_bias_array,
            scale=scale,
            prevalence_stratum=prevalence_stratum,
            coefficient_stratum=coefficient_index,
            mask=mask,
            nominal_level=nominal_level,
            z_value=z_value,
        )
        if fit_rank_centering
        else (
            tuple(tuple(0.0 for _ in range(mean.shape[1])) for _ in range(3)),
            0.0,
        )
    )
    rank_centering_array = (
        np.asarray(rank_centering_offsets, dtype=float)[
            prevalence_stratum, coefficient_index
        ]
        * scale
    )
    corrected_mean = mean + mean_bias_array + rank_centering_array
    corrected_posterior = BetaPosterior(
        mean=tf.convert_to_tensor(corrected_mean, dtype=posterior.mean.dtype),
        scale=posterior.scale,
        scale_tril=posterior.scale_tril,
    )

    baseline = baseline_calibration or fit_beta_scale_calibration(
        corrected_posterior,
        truth,
        nominal_level=nominal_level,
        distribution=distribution,
        species_mask=species_mask,
    )
    baseline.validate_domain(
        distribution=distribution,
        n_covariates=mean.shape[1],
        n_species=mean.shape[2],
    )
    if not np.isclose(baseline.nominal_level, nominal_level):
        raise ValueError("baseline calibration nominal level does not match")

    raw_features = _raw_features(
        mean=corrected_mean,
        scale=scale,
        X=design,
        Y=response,
        distribution=distribution,
    )
    selected_raw = raw_features[mask]
    location = np.mean(selected_raw, axis=0)
    feature_scale = np.std(selected_raw, axis=0)
    feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)
    selected_standardized = (selected_raw - location) / feature_scale
    support_tail = (1.0 - support_quantile) / 2.0
    support_lower = np.quantile(selected_standardized, support_tail, axis=0)
    support_upper = np.quantile(selected_standardized, 1.0 - support_tail, axis=0)
    support_covariance = np.cov(selected_standardized, rowvar=False)
    support_precision = np.linalg.inv(
        support_covariance + float(support_ridge) * np.eye(3)
    )
    support_distance = np.sqrt(
        np.maximum(
            np.einsum(
                "ni,ij,nj->n",
                selected_standardized,
                support_precision,
                selected_standardized,
            ),
            0.0,
        )
    )
    support_radius = float(np.quantile(support_distance, support_quantile))
    selected_mean_magnitude = np.log1p(np.abs(corrected_mean))[mask]
    mean_magnitude_location = float(np.mean(selected_mean_magnitude))
    mean_magnitude_scale = float(np.std(selected_mean_magnitude))
    if mean_magnitude_scale <= 1e-8:
        mean_magnitude_scale = 1.0
    standardized_mean_magnitude = (
        selected_mean_magnitude - mean_magnitude_location
    ) / mean_magnitude_scale
    mean_magnitude_lower = float(np.quantile(standardized_mean_magnitude, support_tail))
    mean_magnitude_upper = float(
        np.quantile(standardized_mean_magnitude, 1.0 - support_tail)
    )
    feature_design, feature_names = _structured_design(
        raw_features,
        location=location,
        scale=feature_scale,
        n_covariates=mean.shape[1],
    )

    signed_standardized_error = (truth - corrected_mean) / scale
    standardized_error = np.abs(signed_standardized_error)
    selected_design = feature_design[mask.reshape(-1)]
    selected_error = standardized_error[mask]
    selected_signed_error = signed_standardized_error[mask]
    selected_prevalence = prevalence_by_coefficient[mask]
    design_signal = _design_information_signal(
        raw_features,
        location=location,
        scale=feature_scale,
    )
    design_stratum = _design_information_stratum_index(design_signal)
    coefficient_stratum = coefficient_index.astype(np.int32)
    observation_weights = _prevalence_observation_weights(
        selected_prevalence,
        prevalence_weights=prevalence_weights,
        prevalence_edges=prevalence_edges,
    )
    rank_groups = _prevalence_group_masks(
        selected_prevalence, prevalence_edges=prevalence_edges
    )
    in_domain_gate_groups = _in_domain_gate_group_masks(
        prevalence=selected_prevalence,
        log_design_information=selected_raw[:, 1],
        coefficient_index=coefficient_index[mask],
        prevalence_edges=prevalence_edges,
    )
    weights = tf.Variable(
        np.zeros(selected_design.shape[1], dtype=np.float64), dtype=tf.float64
    )
    fit_base_scale_strata = ood_objective == "support_effect_gated_rank_coverage"
    base_prevalence_offsets = tf.Variable(np.zeros(3, dtype=np.float64))
    base_design_offsets = tf.Variable(np.zeros(3, dtype=np.float64))
    base_coefficient_offsets = tf.Variable(np.zeros(mean.shape[1], dtype=np.float64))
    design_tensor = tf.constant(selected_design, dtype=tf.float64)
    error_tensor = tf.constant(selected_error, dtype=tf.float64)
    signed_error_tensor = tf.constant(selected_signed_error, dtype=tf.float64)
    observation_weight_tensor = tf.constant(observation_weights, dtype=tf.float64)
    rank_group_tensors = [tf.constant(group) for group in rank_groups]
    selected_prevalence_stratum_tensor = tf.constant(
        prevalence_stratum[mask], dtype=tf.int32
    )
    selected_design_stratum_tensor = tf.constant(design_stratum[mask], dtype=tf.int32)
    selected_coefficient_stratum_tensor = tf.constant(
        coefficient_stratum[mask], dtype=tf.int32
    )
    base_log_scale = tf.constant(np.log(baseline.scale_multiplier), dtype=tf.float64)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for _ in range(epochs):
        with tf.GradientTape() as tape:
            base_stratum_log_offset = (
                tf.gather(base_prevalence_offsets, selected_prevalence_stratum_tensor)
                + tf.gather(base_design_offsets, selected_design_stratum_tensor)
                + tf.gather(
                    base_coefficient_offsets, selected_coefficient_stratum_tensor
                )
                if fit_base_scale_strata
                else tf.constant(0.0, dtype=tf.float64)
            )
            log_multiplier = tf.clip_by_value(
                base_log_scale
                + tf.linalg.matvec(design_tensor, weights)
                + base_stratum_log_offset,
                np.log(min_multiplier),
                np.log(max_multiplier),
            )
            coefficient_nll = log_multiplier + 0.5 * tf.square(error_tensor) * tf.exp(
                -2.0 * log_multiplier
            )
            nll = tf.reduce_sum(
                observation_weight_tensor * coefficient_nll
            ) / tf.reduce_sum(observation_weight_tensor)
            rank_probability = _tf_normal_cdf(
                signed_error_tensor * tf.exp(-log_multiplier)
            )
            rank_loss = _tf_rank_moment_loss(
                rank_probability,
                rank_group_tensors,
                mean_tolerance=rank_mean_tolerance,
                variance_tolerance=rank_variance_tolerance,
            )
            penalty = tf.cast(regularization, tf.float64) * tf.reduce_mean(
                tf.square(weights)
            )
            if fit_base_scale_strata:
                penalty = penalty + tf.cast(regularization, tf.float64) * (
                    tf.reduce_mean(tf.square(base_prevalence_offsets))
                    + tf.reduce_mean(tf.square(base_design_offsets))
                    + tf.reduce_mean(tf.square(base_coefficient_offsets))
                )
            loss = nll + tf.cast(rank_penalty_weight, tf.float64) * rank_loss + penalty
        variables = [weights]
        if fit_base_scale_strata:
            variables.extend(
                [
                    base_prevalence_offsets,
                    base_design_offsets,
                    base_coefficient_offsets,
                ]
            )
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

    fitted_weights = weights.numpy()
    base_scale_stratum_offsets = (
        tuple(float(value) for value in base_prevalence_offsets.numpy())
        + tuple(float(value) for value in base_design_offsets.numpy())
        + tuple(float(value) for value in base_coefficient_offsets.numpy())
        if fit_base_scale_strata
        else tuple(0.0 for _ in range(6 + mean.shape[1]))
    )
    base_stratum_log_offset = _base_scale_stratum_log_offset(
        offsets=base_scale_stratum_offsets,
        prevalence_stratum=prevalence_stratum,
        design_stratum=design_stratum,
        coefficient_stratum=coefficient_stratum,
        n_covariates=mean.shape[1],
    )
    adjustment = np.exp(
        np.clip(
            feature_design @ fitted_weights + base_stratum_log_offset.reshape(-1),
            -20.0,
            20.0,
        )
    ).reshape(mean.shape)
    support_trust = _support_trust(
        raw_features,
        location=location,
        scale=feature_scale,
        lower=support_lower,
        upper=support_upper,
        precision=support_precision,
        radius=support_radius,
        fallback_strength=fallback_strength,
        mean_magnitude=np.log1p(np.abs(corrected_mean)),
        mean_magnitude_location=mean_magnitude_location,
        mean_magnitude_scale=mean_magnitude_scale,
        mean_magnitude_lower=mean_magnitude_lower,
        mean_magnitude_upper=mean_magnitude_upper,
    )
    support_excess = _support_excess(
        raw_features,
        location=location,
        scale=feature_scale,
        lower=support_lower,
        upper=support_upper,
        precision=support_precision,
        radius=support_radius,
        mean_magnitude=np.log1p(np.abs(corrected_mean)),
        mean_magnitude_location=mean_magnitude_location,
        mean_magnitude_scale=mean_magnitude_scale,
        mean_magnitude_lower=mean_magnitude_lower,
        mean_magnitude_upper=mean_magnitude_upper,
    )
    effect_signal = _effect_size_signal(
        np.log1p(np.abs(corrected_mean)),
        mean_magnitude_location=mean_magnitude_location,
        mean_magnitude_scale=mean_magnitude_scale,
    )
    design_signal = _design_information_signal(
        raw_features,
        location=location,
        scale=feature_scale,
    )
    prevalence_stratum = _prevalence_stratum_index(
        prevalence_by_coefficient,
        prevalence_edges=(low_prevalence, high_prevalence),
    )
    design_stratum = _design_information_stratum_index(design_signal)
    coefficient_stratum = coefficient_index.astype(np.int32)
    community_occupancy = _community_occupancy_array(response, mean.shape)
    normalization = _fit_coverage_normalization(
        standardized_error=standardized_error,
        adjustment=adjustment,
        trust=support_trust,
        support_excess=support_excess,
        effect_signal=effect_signal,
        design_signal=design_signal,
        community_occupancy=community_occupancy,
        prevalence_stratum=prevalence_stratum,
        design_stratum=design_stratum,
        coefficient_stratum=coefficient_stratum,
        global_multiplier=baseline.scale_multiplier,
        mask=mask,
        nominal_level=nominal_level,
        z_value=z_value,
        ood_uncertainty_strength=ood_uncertainty_strength,
        ood_uncertainty_max_multiplier=ood_uncertainty_max_multiplier,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
    )
    ood_inflation_parameters = None
    ood_objective_loss = 0.0
    ood_objective_rank_loss = 0.0
    ood_in_domain_gate_loss = 0.0
    ood_objective_n_observations = 0
    ood_objective_domains: tuple[str, ...] = ()
    if ood_objective != "none":
        batches = tuple(ood_calibration_batches or ())
        if not batches:
            raise ValueError(
                "OOD objective requires at least one OOD calibration batch"
            )
        (
            ood_inflation_parameters,
            ood_objective_loss,
            ood_objective_rank_loss,
            ood_in_domain_gate_loss,
            ood_objective_n_observations,
            ood_objective_domains,
        ) = _fit_ood_inflation_parameters(
            batches,
            location=location,
            feature_scale=feature_scale,
            feature_names=feature_names,
            support_lower=support_lower,
            support_upper=support_upper,
            support_precision=support_precision,
            support_radius=support_radius,
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
            normalization=normalization,
            global_multiplier=baseline.scale_multiplier,
            fallback_strength=fallback_strength,
            fitted_weights=fitted_weights,
            mean_bias_correction=mean_bias_correction,
            rank_centering_offsets=rank_centering_offsets,
            base_scale_stratum_offsets=base_scale_stratum_offsets,
            in_domain_signed_error=selected_signed_error,
            in_domain_adjustment=adjustment[mask],
            in_domain_trust=support_trust[mask],
            in_domain_support_excess=support_excess[mask],
            in_domain_effect_signal=effect_signal[mask],
            in_domain_design_signal=design_signal[mask],
            in_domain_prevalence_stratum=prevalence_stratum[mask],
            in_domain_design_stratum=design_stratum[mask],
            in_domain_coefficient_stratum=coefficient_stratum[mask],
            in_domain_rank_groups=(
                in_domain_gate_groups
                if ood_objective == "support_effect_gated_rank_coverage"
                else rank_groups
            ),
            distribution=distribution,
            n_covariates=mean.shape[1],
            n_species=mean.shape[2],
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            max_ood_multiplier=ood_uncertainty_max_multiplier,
            objective_weight=ood_objective_weight,
            in_domain_gate_weight=ood_in_domain_gate_weight,
            epochs=ood_objective_epochs or max(50, epochs // 2),
            learning_rate=learning_rate,
            gate_effect_branch=ood_objective == "support_effect_gated_rank_coverage",
            prevalence_edges=(low_prevalence, high_prevalence),
            rank_mean_tolerance=rank_mean_tolerance,
            rank_variance_tolerance=rank_variance_tolerance,
            nominal_level=nominal_level,
            z_value=z_value,
        )
        normalization = _fit_coverage_normalization(
            standardized_error=standardized_error,
            adjustment=adjustment,
            trust=support_trust,
            support_excess=support_excess,
            effect_signal=effect_signal,
            design_signal=design_signal,
            community_occupancy=community_occupancy,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            coefficient_stratum=coefficient_stratum,
            global_multiplier=baseline.scale_multiplier,
            mask=mask,
            nominal_level=nominal_level,
            z_value=z_value,
            ood_uncertainty_strength=ood_uncertainty_strength,
            ood_uncertainty_max_multiplier=ood_uncertainty_max_multiplier,
            ood_inflation_parameters=ood_inflation_parameters,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
    rare_validation_scale_log_offsets: tuple[float, float, float] = (0.0, 0.0, 0.0)
    combined_shift_scale_log_amplitude = 0.0
    combined_shift_scale_effect_bin_edges = (0.25, 1.0)
    combined_shift_scale_effect_bin_log_amplitudes = (0.0, 0.0, 0.0)
    combined_shift_scale_context_gate_strength = 0.0
    combined_shift_scale_context_gate_intercept = 0.0
    combined_shift_scale_diagnostics: dict[str, Any] = {}
    rare_validation_scale_stats: dict[str, Any] = {
        "selected_shrinkage": 0.0,
        "support_threshold": 0.0,
        "support_width": 1.0,
        "community_threshold": 0.0,
        "community_width": 1.0,
        "diagnostics": {},
    }
    ood_effect_shift_head_selection_diagnostics: dict[str, Any] = {}
    ood_domain_expert_selection_diagnostics: dict[str, Any] = {}
    base_multipliers = _blend_with_scalar_fallback(
        adjustment,
        support_trust,
        support_excess=support_excess,
        effect_signal=effect_signal,
        design_signal=design_signal,
        prevalence_stratum=prevalence_stratum,
        design_stratum=design_stratum,
        coefficient_stratum=coefficient_stratum,
        normalization=normalization,
        global_multiplier=baseline.scale_multiplier,
        ood_uncertainty_strength=ood_uncertainty_strength,
        ood_uncertainty_max_multiplier=ood_uncertainty_max_multiplier,
        ood_inflation_parameters=ood_inflation_parameters,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
    )
    if rare_validation_batches:
        (
            rare_validation_scale_log_offsets,
            rare_validation_scale_stats,
        ) = _fit_rare_validation_scale_correction(
            rare_validation_batches=rare_validation_batches,
            location=location,
            feature_scale=feature_scale,
            feature_names=feature_names,
            support_lower=support_lower,
            support_upper=support_upper,
            support_precision=support_precision,
            support_radius=support_radius,
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
            normalization=normalization,
            global_multiplier=baseline.scale_multiplier,
            fallback_strength=fallback_strength,
            fitted_weights=fitted_weights,
            mean_bias_correction=mean_bias_correction,
            rank_centering_offsets=rank_centering_offsets,
            base_scale_stratum_offsets=base_scale_stratum_offsets,
            distribution=distribution,
            n_covariates=mean.shape[1],
            n_species=mean.shape[2],
            prevalence_edges=(low_prevalence, high_prevalence),
            ood_uncertainty_strength=ood_uncertainty_strength,
            ood_uncertainty_max_multiplier=ood_uncertainty_max_multiplier,
            ood_inflation_parameters=ood_inflation_parameters,
            in_domain_signed_error=selected_signed_error,
            in_domain_base_multiplier=base_multipliers[mask],
            in_domain_support_excess=support_excess[mask],
            in_domain_community_occupancy=community_occupancy[mask],
            in_domain_prevalence_stratum=prevalence_stratum[mask],
            in_domain_design_stratum=design_stratum[mask],
            nominal_level=nominal_level,
            z_value=z_value,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
    if (
        ood_objective == "support_effect_gated_rank_coverage"
        and ood_inflation_parameters is not None
    ):
        (
            ood_inflation_parameters,
            ood_objective_loss,
            ood_objective_rank_loss,
            ood_in_domain_gate_loss,
            ood_objective_n_observations,
            ood_objective_domains,
        ) = _fit_ood_inflation_parameters(
            tuple(ood_calibration_batches or ()),
            location=location,
            feature_scale=feature_scale,
            feature_names=feature_names,
            support_lower=support_lower,
            support_upper=support_upper,
            support_precision=support_precision,
            support_radius=support_radius,
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
            normalization=normalization,
            global_multiplier=baseline.scale_multiplier,
            fallback_strength=fallback_strength,
            fitted_weights=fitted_weights,
            mean_bias_correction=mean_bias_correction,
            rank_centering_offsets=rank_centering_offsets,
            base_scale_stratum_offsets=base_scale_stratum_offsets,
            in_domain_signed_error=selected_signed_error,
            in_domain_adjustment=adjustment[mask],
            in_domain_trust=support_trust[mask],
            in_domain_support_excess=support_excess[mask],
            in_domain_effect_signal=effect_signal[mask],
            in_domain_design_signal=design_signal[mask],
            in_domain_prevalence_stratum=prevalence_stratum[mask],
            in_domain_design_stratum=design_stratum[mask],
            in_domain_coefficient_stratum=coefficient_stratum[mask],
            in_domain_rank_groups=in_domain_gate_groups,
            distribution=distribution,
            n_covariates=mean.shape[1],
            n_species=mean.shape[2],
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            max_ood_multiplier=ood_uncertainty_max_multiplier,
            objective_weight=ood_objective_weight,
            in_domain_gate_weight=ood_in_domain_gate_weight,
            epochs=max(25, (ood_objective_epochs or max(50, epochs // 2)) // 2),
            learning_rate=learning_rate,
            gate_effect_branch=True,
            prevalence_edges=(low_prevalence, high_prevalence),
            rank_mean_tolerance=rank_mean_tolerance,
            rank_variance_tolerance=rank_variance_tolerance,
            nominal_level=nominal_level,
            z_value=z_value,
            initial_parameters=ood_inflation_parameters,
            final_multiplier_aware=True,
            post_scale_log_offsets=rare_validation_scale_log_offsets,
            post_scale_support_threshold=float(
                rare_validation_scale_stats["support_threshold"]
            ),
            post_scale_support_width=float(
                rare_validation_scale_stats["support_width"]
            ),
            post_scale_community_threshold=float(
                rare_validation_scale_stats["community_threshold"]
            ),
            post_scale_community_width=float(
                rare_validation_scale_stats["community_width"]
            ),
            in_domain_community_occupancy=community_occupancy[mask],
        )
        normalization = _fit_coverage_normalization(
            standardized_error=standardized_error,
            adjustment=adjustment,
            trust=support_trust,
            support_excess=support_excess,
            effect_signal=effect_signal,
            design_signal=design_signal,
            community_occupancy=community_occupancy,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            coefficient_stratum=coefficient_stratum,
            global_multiplier=baseline.scale_multiplier,
            mask=mask,
            nominal_level=nominal_level,
            z_value=z_value,
            ood_uncertainty_strength=ood_uncertainty_strength,
            ood_uncertainty_max_multiplier=ood_uncertainty_max_multiplier,
            ood_inflation_parameters=ood_inflation_parameters,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
        base_multipliers = _blend_with_scalar_fallback(
            adjustment,
            support_trust,
            support_excess=support_excess,
            effect_signal=effect_signal,
            design_signal=design_signal,
            community_occupancy=community_occupancy,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            coefficient_stratum=coefficient_stratum,
            normalization=normalization,
            global_multiplier=baseline.scale_multiplier,
            ood_uncertainty_strength=ood_uncertainty_strength,
            ood_uncertainty_max_multiplier=ood_uncertainty_max_multiplier,
            ood_inflation_parameters=ood_inflation_parameters,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
        (
            ood_inflation_parameters,
            ood_domain_expert_selection_diagnostics,
        ) = _fit_and_select_domain_expert_ood_parameters(
            batches=tuple(ood_calibration_batches or ()),
            parameters=ood_inflation_parameters,
            location=location,
            feature_scale=feature_scale,
            feature_names=feature_names,
            support_lower=support_lower,
            support_upper=support_upper,
            support_precision=support_precision,
            support_radius=support_radius,
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
            normalization=normalization,
            global_multiplier=baseline.scale_multiplier,
            fallback_strength=fallback_strength,
            fitted_weights=fitted_weights,
            mean_bias_correction=mean_bias_correction,
            rank_centering_offsets=rank_centering_offsets,
            base_scale_stratum_offsets=base_scale_stratum_offsets,
            post_scale_log_offsets=rare_validation_scale_log_offsets,
            post_scale_support_threshold=float(
                rare_validation_scale_stats["support_threshold"]
            ),
            post_scale_support_width=float(
                rare_validation_scale_stats["support_width"]
            ),
            post_scale_community_threshold=float(
                rare_validation_scale_stats["community_threshold"]
            ),
            post_scale_community_width=float(
                rare_validation_scale_stats["community_width"]
            ),
            distribution=distribution,
            n_covariates=mean.shape[1],
            n_species=mean.shape[2],
            prevalence_edges=(low_prevalence, high_prevalence),
            in_domain_signed_error=selected_signed_error,
            in_domain_adjustment=adjustment[mask],
            in_domain_trust=support_trust[mask],
            in_domain_support_excess=support_excess[mask],
            in_domain_effect_signal=effect_signal[mask],
            in_domain_design_signal=design_signal[mask],
            in_domain_prevalence_stratum=prevalence_stratum[mask],
            in_domain_design_stratum=design_stratum[mask],
            in_domain_coefficient_stratum=coefficient_stratum[mask],
            in_domain_post_scale_multiplier=_rare_validation_scale_multiplier(
                log_offsets=rare_validation_scale_log_offsets,
                prevalence_stratum=prevalence_stratum[mask],
                design_stratum=design_stratum[mask],
                support_excess=support_excess[mask],
                community_occupancy=community_occupancy[mask],
                support_threshold=float(
                    rare_validation_scale_stats["support_threshold"]
                ),
                support_width=float(rare_validation_scale_stats["support_width"]),
                community_threshold=float(
                    rare_validation_scale_stats["community_threshold"]
                ),
                community_width=float(rare_validation_scale_stats["community_width"]),
            ),
            in_domain_rank_groups=in_domain_gate_groups,
            rank_mean_tolerance=rank_mean_tolerance,
            rank_variance_tolerance=rank_variance_tolerance,
            nominal_level=nominal_level,
            z_value=z_value,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            max_ood_multiplier=ood_uncertainty_max_multiplier,
            objective_weight=ood_objective_weight,
            in_domain_gate_weight=ood_in_domain_gate_weight,
            epochs=max(10, (ood_objective_epochs or max(50, epochs // 2)) // 2),
            learning_rate=learning_rate,
            in_domain_community_occupancy=community_occupancy[mask],
        )
        normalization = _fit_coverage_normalization(
            standardized_error=standardized_error,
            adjustment=adjustment,
            trust=support_trust,
            support_excess=support_excess,
            effect_signal=effect_signal,
            design_signal=design_signal,
            community_occupancy=community_occupancy,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            coefficient_stratum=coefficient_stratum,
            global_multiplier=baseline.scale_multiplier,
            mask=mask,
            nominal_level=nominal_level,
            z_value=z_value,
            ood_uncertainty_strength=ood_uncertainty_strength,
            ood_uncertainty_max_multiplier=ood_uncertainty_max_multiplier,
            ood_inflation_parameters=ood_inflation_parameters,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
        base_multipliers = _blend_with_scalar_fallback(
            adjustment,
            support_trust,
            support_excess=support_excess,
            effect_signal=effect_signal,
            design_signal=design_signal,
            community_occupancy=community_occupancy,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            coefficient_stratum=coefficient_stratum,
            normalization=normalization,
            global_multiplier=baseline.scale_multiplier,
            ood_uncertainty_strength=ood_uncertainty_strength,
            ood_uncertainty_max_multiplier=ood_uncertainty_max_multiplier,
            ood_inflation_parameters=ood_inflation_parameters,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
        (
            ood_inflation_parameters,
            ood_effect_shift_head_selection_diagnostics,
        ) = _select_effect_shift_head_shrinkage(
            batches=tuple(ood_calibration_batches or ()),
            parameters=ood_inflation_parameters,
            location=location,
            feature_scale=feature_scale,
            feature_names=feature_names,
            support_lower=support_lower,
            support_upper=support_upper,
            support_precision=support_precision,
            support_radius=support_radius,
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
            normalization=normalization,
            global_multiplier=baseline.scale_multiplier,
            fallback_strength=fallback_strength,
            fitted_weights=fitted_weights,
            mean_bias_correction=mean_bias_correction,
            rank_centering_offsets=rank_centering_offsets,
            base_scale_stratum_offsets=base_scale_stratum_offsets,
            post_scale_log_offsets=rare_validation_scale_log_offsets,
            post_scale_support_threshold=float(
                rare_validation_scale_stats["support_threshold"]
            ),
            post_scale_support_width=float(
                rare_validation_scale_stats["support_width"]
            ),
            post_scale_community_threshold=float(
                rare_validation_scale_stats["community_threshold"]
            ),
            post_scale_community_width=float(
                rare_validation_scale_stats["community_width"]
            ),
            distribution=distribution,
            n_covariates=mean.shape[1],
            n_species=mean.shape[2],
            prevalence_edges=(low_prevalence, high_prevalence),
            in_domain_signed_error=selected_signed_error,
            in_domain_adjustment=adjustment[mask],
            in_domain_trust=support_trust[mask],
            in_domain_support_excess=support_excess[mask],
            in_domain_effect_signal=effect_signal[mask],
            in_domain_design_signal=design_signal[mask],
            in_domain_prevalence_stratum=prevalence_stratum[mask],
            in_domain_design_stratum=design_stratum[mask],
            in_domain_coefficient_stratum=coefficient_stratum[mask],
            in_domain_post_scale_multiplier=_rare_validation_scale_multiplier(
                log_offsets=rare_validation_scale_log_offsets,
                prevalence_stratum=prevalence_stratum[mask],
                design_stratum=design_stratum[mask],
                support_excess=support_excess[mask],
                community_occupancy=community_occupancy[mask],
                support_threshold=float(
                    rare_validation_scale_stats["support_threshold"]
                ),
                support_width=float(rare_validation_scale_stats["support_width"]),
                community_threshold=float(
                    rare_validation_scale_stats["community_threshold"]
                ),
                community_width=float(rare_validation_scale_stats["community_width"]),
            ),
            in_domain_rank_groups=in_domain_gate_groups,
            rank_mean_tolerance=rank_mean_tolerance,
            rank_variance_tolerance=rank_variance_tolerance,
            nominal_level=nominal_level,
            z_value=z_value,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            max_ood_multiplier=ood_uncertainty_max_multiplier,
        )
        normalization = _fit_coverage_normalization(
            standardized_error=standardized_error,
            adjustment=adjustment,
            trust=support_trust,
            support_excess=support_excess,
            effect_signal=effect_signal,
            design_signal=design_signal,
            community_occupancy=community_occupancy,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            coefficient_stratum=coefficient_stratum,
            global_multiplier=baseline.scale_multiplier,
            mask=mask,
            nominal_level=nominal_level,
            z_value=z_value,
            ood_uncertainty_strength=ood_uncertainty_strength,
            ood_uncertainty_max_multiplier=ood_uncertainty_max_multiplier,
            ood_inflation_parameters=ood_inflation_parameters,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
        base_multipliers = _blend_with_scalar_fallback(
            adjustment,
            support_trust,
            support_excess=support_excess,
            effect_signal=effect_signal,
            design_signal=design_signal,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            coefficient_stratum=coefficient_stratum,
            normalization=normalization,
            global_multiplier=baseline.scale_multiplier,
            ood_uncertainty_strength=ood_uncertainty_strength,
            ood_uncertainty_max_multiplier=ood_uncertainty_max_multiplier,
            ood_inflation_parameters=ood_inflation_parameters,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
        (
            combined_shift_scale_log_amplitude,
            combined_shift_scale_effect_bin_log_amplitudes,
            combined_shift_scale_effect_bin_edges,
            combined_shift_scale_context_gate_strength,
            combined_shift_scale_context_gate_intercept,
            combined_shift_scale_diagnostics,
        ) = _select_combined_shift_scale_head(
            batches=tuple(ood_calibration_batches or ()),
            parameters=ood_inflation_parameters,
            location=location,
            feature_scale=feature_scale,
            feature_names=feature_names,
            support_lower=support_lower,
            support_upper=support_upper,
            support_precision=support_precision,
            support_radius=support_radius,
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
            normalization=normalization,
            global_multiplier=baseline.scale_multiplier,
            fallback_strength=fallback_strength,
            fitted_weights=fitted_weights,
            mean_bias_correction=mean_bias_correction,
            rank_centering_offsets=rank_centering_offsets,
            base_scale_stratum_offsets=base_scale_stratum_offsets,
            post_scale_log_offsets=rare_validation_scale_log_offsets,
            post_scale_support_threshold=float(
                rare_validation_scale_stats["support_threshold"]
            ),
            post_scale_support_width=float(
                rare_validation_scale_stats["support_width"]
            ),
            post_scale_community_threshold=float(
                rare_validation_scale_stats["community_threshold"]
            ),
            post_scale_community_width=float(
                rare_validation_scale_stats["community_width"]
            ),
            distribution=distribution,
            n_covariates=mean.shape[1],
            n_species=mean.shape[2],
            prevalence_edges=(low_prevalence, high_prevalence),
            in_domain_signed_error=selected_signed_error,
            in_domain_final_multiplier=np.clip(
                base_multipliers[mask]
                * _rare_validation_scale_multiplier(
                    log_offsets=rare_validation_scale_log_offsets,
                    prevalence_stratum=prevalence_stratum[mask],
                    design_stratum=design_stratum[mask],
                    support_excess=support_excess[mask],
                    community_occupancy=community_occupancy[mask],
                    support_threshold=float(
                        rare_validation_scale_stats["support_threshold"]
                    ),
                    support_width=float(rare_validation_scale_stats["support_width"]),
                    community_threshold=float(
                        rare_validation_scale_stats["community_threshold"]
                    ),
                    community_width=float(
                        rare_validation_scale_stats["community_width"]
                    ),
                ),
                min_multiplier,
                max_multiplier,
            ),
            in_domain_support_excess=support_excess[mask],
            in_domain_effect_signal=effect_signal[mask],
            in_domain_design_signal=design_signal[mask],
            in_domain_community_occupancy=community_occupancy[mask],
            in_domain_log_ood_inflation=_learned_ood_log_inflation_numpy(
                support_excess[mask],
                effect_signal=effect_signal[mask],
                design_signal=design_signal[mask],
                community_occupancy=community_occupancy[mask],
                prevalence_stratum=prevalence_stratum[mask],
                design_stratum=design_stratum[mask],
                coefficient_stratum=coefficient_stratum[mask],
                parameters=ood_inflation_parameters,
                max_multiplier=ood_uncertainty_max_multiplier,
            ),
            in_domain_rank_groups=in_domain_gate_groups,
            rank_mean_tolerance=rank_mean_tolerance,
            rank_variance_tolerance=rank_variance_tolerance,
            nominal_level=nominal_level,
            z_value=z_value,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            max_ood_multiplier=ood_uncertainty_max_multiplier,
        )
    multipliers = np.clip(
        base_multipliers
        * _rare_validation_scale_multiplier(
            log_offsets=rare_validation_scale_log_offsets,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            support_excess=support_excess,
            community_occupancy=community_occupancy,
            support_threshold=float(rare_validation_scale_stats["support_threshold"]),
            support_width=float(rare_validation_scale_stats["support_width"]),
            community_threshold=float(
                rare_validation_scale_stats["community_threshold"]
            ),
            community_width=float(rare_validation_scale_stats["community_width"]),
        )
        * _combined_shift_scale_multiplier(
            support_excess=support_excess,
            effect_signal=effect_signal,
            design_signal=design_signal,
            community_occupancy=community_occupancy,
            log_amplitude=combined_shift_scale_log_amplitude,
            effect_bin_log_amplitudes=combined_shift_scale_effect_bin_log_amplitudes,
            effect_bin_edges=combined_shift_scale_effect_bin_edges,
            context_gate_strength=combined_shift_scale_context_gate_strength,
            context_gate_intercept=combined_shift_scale_context_gate_intercept,
        ),
        min_multiplier,
        max_multiplier,
    )
    uncalibrated_coverage = _coverage(mean, scale, truth, mask, z_value)
    calibrated_coverage = _coverage(
        corrected_mean, scale * multipliers, truth, mask, z_value
    )
    scalar_multiplier = np.full(selected_error.shape, baseline.scale_multiplier)
    conditional_multiplier = multipliers[mask]
    ood_final_multiplier_diagnostics: dict[str, Any] = {}
    if ood_inflation_parameters is not None and ood_calibration_batches:
        ood_final_multiplier_diagnostics = _ood_final_multiplier_diagnostics(
            batches=tuple(ood_calibration_batches),
            location=location,
            feature_scale=feature_scale,
            feature_names=feature_names,
            support_lower=support_lower,
            support_upper=support_upper,
            support_precision=support_precision,
            support_radius=support_radius,
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
            normalization=normalization,
            global_multiplier=baseline.scale_multiplier,
            fallback_strength=fallback_strength,
            fitted_weights=fitted_weights,
            mean_bias_correction=mean_bias_correction,
            rank_centering_offsets=rank_centering_offsets,
            base_scale_stratum_offsets=base_scale_stratum_offsets,
            ood_inflation_parameters=ood_inflation_parameters,
            post_scale_log_offsets=rare_validation_scale_log_offsets,
            post_scale_support_threshold=float(
                rare_validation_scale_stats["support_threshold"]
            ),
            post_scale_support_width=float(
                rare_validation_scale_stats["support_width"]
            ),
            post_scale_community_threshold=float(
                rare_validation_scale_stats["community_threshold"]
            ),
            post_scale_community_width=float(
                rare_validation_scale_stats["community_width"]
            ),
            combined_shift_scale_log_amplitude=combined_shift_scale_log_amplitude,
            distribution=distribution,
            n_covariates=mean.shape[1],
            n_species=mean.shape[2],
            prevalence_edges=(low_prevalence, high_prevalence),
            in_domain_signed_error=selected_signed_error,
            in_domain_final_multiplier=conditional_multiplier,
            in_domain_log_ood_inflation=(
                _learned_ood_log_inflation_numpy(
                    support_excess[mask],
                    effect_signal=effect_signal[mask],
                    design_signal=design_signal[mask],
                    community_occupancy=community_occupancy[mask],
                    prevalence_stratum=prevalence_stratum[mask],
                    design_stratum=design_stratum[mask],
                    coefficient_stratum=coefficient_stratum[mask],
                    parameters=ood_inflation_parameters,
                    max_multiplier=ood_uncertainty_max_multiplier,
                )
                + np.log(
                    _combined_shift_scale_multiplier(
                        support_excess=support_excess[mask],
                        effect_signal=effect_signal[mask],
                        design_signal=design_signal[mask],
                        community_occupancy=community_occupancy[mask],
                        log_amplitude=combined_shift_scale_log_amplitude,
                        effect_bin_log_amplitudes=(
                            combined_shift_scale_effect_bin_log_amplitudes
                        ),
                        effect_bin_edges=combined_shift_scale_effect_bin_edges,
                        context_gate_strength=(
                            combined_shift_scale_context_gate_strength
                        ),
                        context_gate_intercept=(
                            combined_shift_scale_context_gate_intercept
                        ),
                    )
                )
            ),
            in_domain_rank_groups=in_domain_gate_groups,
            rank_mean_tolerance=rank_mean_tolerance,
            rank_variance_tolerance=rank_variance_tolerance,
            nominal_level=nominal_level,
            z_value=z_value,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            max_ood_multiplier=ood_uncertainty_max_multiplier,
            combined_shift_scale_effect_bin_log_amplitudes=(
                combined_shift_scale_effect_bin_log_amplitudes
            ),
            combined_shift_scale_effect_bin_edges=combined_shift_scale_effect_bin_edges,
            combined_shift_scale_context_gate_strength=(
                combined_shift_scale_context_gate_strength
            ),
            combined_shift_scale_context_gate_intercept=(
                combined_shift_scale_context_gate_intercept
            ),
        )
        if ood_effect_shift_head_selection_diagnostics:
            ood_final_multiplier_diagnostics["effect_shift_head_selection"] = (
                ood_effect_shift_head_selection_diagnostics
            )
        if ood_domain_expert_selection_diagnostics:
            ood_final_multiplier_diagnostics["domain_expert_selection"] = (
                ood_domain_expert_selection_diagnostics
            )
        if combined_shift_scale_diagnostics:
            ood_final_multiplier_diagnostics["combined_shift_scale_selection"] = (
                combined_shift_scale_diagnostics
            )

    return ConditionalBetaScaleCalibration(
        global_scale_multiplier=float(baseline.scale_multiplier),
        normalization_multiplier=normalization,
        feature_location=tuple(float(value) for value in location),
        feature_scale=tuple(float(value) for value in feature_scale),
        weights=tuple(float(value) for value in fitted_weights),
        feature_names=feature_names,
        coefficient_names=names,
        nominal_level=float(nominal_level),
        uncalibrated_coverage=uncalibrated_coverage,
        calibrated_coverage=calibrated_coverage,
        n_observations=int(np.count_nonzero(mask)),
        distribution=str(distribution),
        n_covariates=int(mean.shape[1]),
        n_species=int(mean.shape[2]),
        regularization=float(regularization),
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        scalar_nll=_scale_nll(selected_error, scalar_multiplier),
        conditional_nll=_scale_nll(selected_error, conditional_multiplier),
        scalar_rank_loss=_rank_moment_loss(
            selected_signed_error / scalar_multiplier,
            rank_groups,
            mean_tolerance=rank_mean_tolerance,
            variance_tolerance=rank_variance_tolerance,
        ),
        conditional_rank_loss=_rank_moment_loss(
            selected_signed_error / conditional_multiplier,
            rank_groups,
            mean_tolerance=rank_mean_tolerance,
            variance_tolerance=rank_variance_tolerance,
        ),
        prevalence_weights=tuple(float(value) for value in prevalence_weights),
        prevalence_edges=(low_prevalence, high_prevalence),
        rank_penalty_weight=float(rank_penalty_weight),
        rank_mean_tolerance=float(rank_mean_tolerance),
        rank_variance_tolerance=float(rank_variance_tolerance),
        support_lower=tuple(float(value) for value in support_lower),
        support_upper=tuple(float(value) for value in support_upper),
        support_precision=tuple(
            tuple(float(value) for value in row) for row in support_precision
        ),
        support_radius=support_radius,
        support_quantile=float(support_quantile),
        fallback_strength=float(fallback_strength),
        mean_magnitude_location=mean_magnitude_location,
        mean_magnitude_scale=mean_magnitude_scale,
        mean_magnitude_lower=mean_magnitude_lower,
        mean_magnitude_upper=mean_magnitude_upper,
        mean_bias_correction=mean_bias_correction,
        mean_bias_shrinkage=mean_bias_shrinkage,
        rare_mean_head_n_observations=int(rare_mean_head_stats["n_observations"]),
        rare_mean_head_selected_shrinkage=float(
            rare_mean_head_stats["selected_shrinkage"]
        ),
        rare_mean_head_validation_rank_error=float(
            rare_mean_head_stats["validation_rank_error"]
        ),
        rare_mean_head_diagnostics=dict(rare_mean_head_stats.get("diagnostics", {})),
        rank_centering_offsets=rank_centering_offsets,
        rank_centering_shrinkage=rank_centering_shrinkage,
        base_scale_stratum_offsets=base_scale_stratum_offsets,
        rare_validation_scale_log_offsets=rare_validation_scale_log_offsets,
        rare_validation_scale_selected_shrinkage=float(
            rare_validation_scale_stats["selected_shrinkage"]
        ),
        rare_validation_scale_support_threshold=float(
            rare_validation_scale_stats["support_threshold"]
        ),
        rare_validation_scale_support_width=float(
            rare_validation_scale_stats["support_width"]
        ),
        rare_validation_scale_community_threshold=float(
            rare_validation_scale_stats["community_threshold"]
        ),
        rare_validation_scale_community_width=float(
            rare_validation_scale_stats["community_width"]
        ),
        rare_validation_scale_diagnostics=dict(
            rare_validation_scale_stats.get("diagnostics", {})
        ),
        ood_uncertainty_strength=float(ood_uncertainty_strength),
        ood_uncertainty_max_multiplier=float(ood_uncertainty_max_multiplier),
        ood_objective=ood_objective,
        ood_objective_weight=float(
            ood_objective_weight if ood_objective != "none" else 0.0
        ),
        ood_in_domain_gate_weight=float(
            ood_in_domain_gate_weight if ood_objective != "none" else 0.0
        ),
        ood_inflation_parameters=ood_inflation_parameters,
        ood_objective_domains=ood_objective_domains,
        ood_objective_n_observations=ood_objective_n_observations,
        ood_objective_loss=ood_objective_loss,
        ood_objective_rank_loss=ood_objective_rank_loss,
        ood_in_domain_gate_loss=ood_in_domain_gate_loss,
        ood_final_multiplier_diagnostics=ood_final_multiplier_diagnostics,
        combined_shift_scale_log_amplitude=combined_shift_scale_log_amplitude,
        combined_shift_scale_effect_bin_edges=combined_shift_scale_effect_bin_edges,
        combined_shift_scale_effect_bin_log_amplitudes=(
            combined_shift_scale_effect_bin_log_amplitudes
        ),
        combined_shift_scale_context_gate_strength=(
            combined_shift_scale_context_gate_strength
        ),
        combined_shift_scale_context_gate_intercept=(
            combined_shift_scale_context_gate_intercept
        ),
        combined_shift_scale_diagnostics=combined_shift_scale_diagnostics,
        min_multiplier=float(min_multiplier),
        max_multiplier=float(max_multiplier),
    )


def fit_external_context_monotone_calibration(
    calibration: ConditionalBetaScaleCalibration,
    posterior: BetaPosterior,
    beta_true: np.ndarray,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    distribution: str,
    coefficient_names: Sequence[str] | None = None,
    ood_validation_batches: Sequence[ConditionalBetaOODCalibrationBatch] | None = None,
    nominal_level: float = 0.95,
    max_external_multiplier: float = 2.0,
    min_mean_ood_gain: float = 0.005,
    min_combined_shift_gain: float = 0.005,
    min_in_domain_coverage: float = 0.90,
    max_in_domain_coverage: float = 0.99,
    min_rare_coverage: float = 0.70,
) -> ConditionalBetaScaleCalibration:
    """Fit a conservative external monotone context-stratified scale wrapper.

    The wrapper is intentionally low capacity: three non-negative monotone
    effect-size bins activated only by support excess or large posterior effect
    signal. Selection is held-out and gate based; failure returns zero offsets.
    """
    if max_external_multiplier < 1.0:
        raise ValueError("max_external_multiplier must be at least one")
    if not 0.0 < nominal_level < 1.0:
        raise ValueError("nominal_level must be between zero and one")
    names = (
        calibration.coefficient_names
        if coefficient_names is None
        else tuple(str(value) for value in coefficient_names)
    )
    calibration.validate_domain(
        distribution=distribution,
        n_covariates=_as_numpy(posterior.mean).shape[1],
        n_species=_as_numpy(posterior.mean).shape[2],
        coefficient_names=names,
    )
    z_value = float(norm.ppf(0.5 + nominal_level / 2.0))
    zero_offsets = np.zeros(3, dtype=float)
    if not ood_validation_batches:
        return replace(
            calibration,
            method="external_context_monotone_scale",
            external_monotone_log_offsets=(0.0, 0.0, 0.0),
            external_monotone_selected_shrinkage=0.0,
            external_monotone_diagnostics={
                "kind": "heldout_context_stratified_monotone_scale_selection",
                "selected": "baseline",
                "reason": "no_ood_validation_batches",
                "candidate_log_offsets": [0.0, 0.0, 0.0],
            },
        )

    base_arrays = _external_monotone_arrays(
        posterior,
        beta_true,
        calibration,
        X=X,
        Y=Y,
        distribution=distribution,
        coefficient_names=names,
    )
    support_threshold = float(np.quantile(base_arrays["support_excess"], 0.95))
    support_width = max(
        float(np.quantile(base_arrays["support_excess"], 0.99) - support_threshold),
        0.05,
    )
    effect_threshold = float(np.quantile(base_arrays["effect_signal"], 0.80))
    effect_width = max(
        float(np.quantile(base_arrays["effect_signal"], 0.95) - effect_threshold),
        0.05,
    )
    effect_edges = tuple(
        float(value)
        for value in np.quantile(base_arrays["effect_signal"], (0.50, 0.85))
    )
    if effect_edges[1] <= effect_edges[0]:
        effect_edges = (effect_edges[0], effect_edges[0] + 0.25)

    ood_arrays = [
        _external_monotone_arrays(
            batch.posterior,
            batch.beta_true,
            calibration,
            X=batch.X,
            Y=batch.Y,
            distribution=distribution,
            coefficient_names=names,
            label=batch.label,
        )
        for batch in ood_validation_batches
    ]
    candidate_offsets = _external_monotone_candidate_offsets(
        ood_arrays,
        effect_bin_edges=effect_edges,
        support_threshold=support_threshold,
        support_width=support_width,
        effect_threshold=effect_threshold,
        effect_width=effect_width,
        z_value=z_value,
        max_external_multiplier=max_external_multiplier,
    )
    candidate_offsets = np.maximum.accumulate(candidate_offsets)
    candidate_offsets = np.clip(
        candidate_offsets, 0.0, float(np.log(max_external_multiplier))
    )
    baseline_metrics = _external_monotone_selection_metrics(
        zero_offsets,
        in_domain=base_arrays,
        ood=ood_arrays,
        effect_bin_edges=effect_edges,
        support_threshold=support_threshold,
        support_width=support_width,
        effect_threshold=effect_threshold,
        effect_width=effect_width,
        z_value=z_value,
        prevalence_edges=calibration.prevalence_edges,
    )
    best_offsets = zero_offsets
    best_shrinkage = 0.0
    best_metrics = baseline_metrics
    shrinkage_grid = []
    for shrinkage in (0.0, 0.25, 0.5, 0.75, 1.0):
        offsets = candidate_offsets * float(shrinkage)
        metrics = _external_monotone_selection_metrics(
            offsets,
            in_domain=base_arrays,
            ood=ood_arrays,
            effect_bin_edges=effect_edges,
            support_threshold=support_threshold,
            support_width=support_width,
            effect_threshold=effect_threshold,
            effect_width=effect_width,
            z_value=z_value,
            prevalence_edges=calibration.prevalence_edges,
        )
        mean_gain = metrics["mean_ood_coverage"] - baseline_metrics[
            "mean_ood_coverage"
        ]
        combined_gain = metrics["combined_shift_coverage"] - baseline_metrics[
            "combined_shift_coverage"
        ]
        gate_ok = (
            mean_gain >= float(min_mean_ood_gain)
            and combined_gain >= float(min_combined_shift_gain)
            and metrics["worst_ood_coverage"] + 0.005
            >= baseline_metrics["worst_ood_coverage"]
            and float(min_in_domain_coverage)
            <= metrics["in_domain_coverage"]
            <= float(max_in_domain_coverage)
            and metrics["rare_prevalence_coverage"] >= float(min_rare_coverage)
        )
        row = {
            "shrinkage": float(shrinkage),
            "accepted": bool(gate_ok),
            "mean_ood_gain": float(mean_gain),
            "combined_shift_gain": float(combined_gain),
            **metrics,
        }
        shrinkage_grid.append(row)
        if gate_ok and row["mean_ood_gain"] > (
            best_metrics["mean_ood_coverage"] - baseline_metrics["mean_ood_coverage"]
        ):
            best_offsets = offsets
            best_shrinkage = float(shrinkage)
            best_metrics = metrics

    selected = "external_monotone" if best_shrinkage > 0.0 else "baseline"
    diagnostics = {
        "kind": "heldout_context_stratified_monotone_scale_selection",
        "selected": selected,
        "candidate_log_offsets": [float(value) for value in candidate_offsets],
        "selected_log_offsets": [float(value) for value in best_offsets],
        "selected_multipliers": [float(np.exp(value)) for value in best_offsets],
        "selected_shrinkage": float(best_shrinkage),
        "activation": {
            "kind": "support_or_effect_ramp",
            "support_threshold": float(support_threshold),
            "support_width": float(support_width),
            "effect_threshold": float(effect_threshold),
            "effect_width": float(effect_width),
            "effect_bin_edges": [float(value) for value in effect_edges],
        },
        "baseline_metrics": baseline_metrics,
        "selected_metrics": best_metrics,
        "shrinkage_grid": shrinkage_grid,
    }
    return replace(
        calibration,
        method="external_context_monotone_scale",
        external_monotone_log_offsets=tuple(float(value) for value in best_offsets),
        external_monotone_effect_bin_edges=effect_edges,
        external_monotone_support_threshold=float(support_threshold),
        external_monotone_support_width=float(support_width),
        external_monotone_effect_threshold=float(effect_threshold),
        external_monotone_effect_width=float(effect_width),
        external_monotone_selected_shrinkage=float(best_shrinkage),
        external_monotone_diagnostics=diagnostics,
    )


def conditional_beta_scale_multipliers(
    posterior: BetaPosterior,
    calibration: ConditionalBetaScaleCalibration,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    distribution: str | None = None,
    coefficient_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Predict one positive scale multiplier per Beta coefficient."""
    mean, scale, design, response = _validated_arrays(posterior, X=X, Y=Y)
    names = (
        calibration.coefficient_names
        if coefficient_names is None
        else coefficient_names
    )
    calibration.validate_domain(
        distribution=distribution or calibration.distribution,
        n_covariates=mean.shape[1],
        n_species=mean.shape[2],
        coefficient_names=names,
    )
    mean = mean + _mean_bias_correction_array(
        calibration=calibration,
        Y=response,
        shape=mean.shape,
    )
    mean = mean + _rank_centering_correction_array(
        calibration=calibration,
        scale=scale,
        Y=response,
        shape=mean.shape,
    )
    raw_features = _raw_features(
        mean=mean,
        scale=scale,
        X=design,
        Y=response,
        distribution=distribution or calibration.distribution,
    )
    feature_design, feature_names = _structured_design(
        raw_features,
        location=np.asarray(calibration.feature_location),
        scale=np.asarray(calibration.feature_scale),
        n_covariates=mean.shape[1],
    )
    if feature_names != calibration.feature_names:
        raise ValueError("conditional calibration feature specification mismatch")
    adjustment = np.exp(
        np.clip(feature_design @ np.asarray(calibration.weights), -20.0, 20.0)
    ).reshape(mean.shape)
    trust = _support_trust(
        raw_features,
        location=np.asarray(calibration.feature_location),
        scale=np.asarray(calibration.feature_scale),
        lower=np.asarray(calibration.support_lower),
        upper=np.asarray(calibration.support_upper),
        precision=np.asarray(calibration.support_precision),
        radius=calibration.support_radius,
        fallback_strength=calibration.fallback_strength,
        mean_magnitude=np.log1p(np.abs(mean)),
        mean_magnitude_location=calibration.mean_magnitude_location,
        mean_magnitude_scale=calibration.mean_magnitude_scale,
        mean_magnitude_lower=calibration.mean_magnitude_lower,
        mean_magnitude_upper=calibration.mean_magnitude_upper,
    )
    support_excess = _support_excess(
        raw_features,
        location=np.asarray(calibration.feature_location),
        scale=np.asarray(calibration.feature_scale),
        lower=np.asarray(calibration.support_lower),
        upper=np.asarray(calibration.support_upper),
        precision=np.asarray(calibration.support_precision),
        radius=calibration.support_radius,
        mean_magnitude=np.log1p(np.abs(mean)),
        mean_magnitude_location=calibration.mean_magnitude_location,
        mean_magnitude_scale=calibration.mean_magnitude_scale,
        mean_magnitude_lower=calibration.mean_magnitude_lower,
        mean_magnitude_upper=calibration.mean_magnitude_upper,
    )
    effect_signal = _effect_size_signal(
        np.log1p(np.abs(mean)),
        mean_magnitude_location=calibration.mean_magnitude_location,
        mean_magnitude_scale=calibration.mean_magnitude_scale,
    )
    design_signal = _design_information_signal(
        raw_features,
        location=np.asarray(calibration.feature_location),
        scale=np.asarray(calibration.feature_scale),
    )
    prevalence = _prevalence(response)
    community_occupancy = _community_occupancy_array(response, mean.shape)
    prevalence_stratum = _prevalence_stratum_index(
        np.broadcast_to(prevalence[:, None, :], mean.shape),
        prevalence_edges=calibration.prevalence_edges,
    )
    design_stratum = _design_information_stratum_index(design_signal)
    coefficient_stratum = _coefficient_stratum_index(mean.shape)
    adjustment = adjustment * np.exp(
        _base_scale_stratum_log_offset(
            offsets=calibration.base_scale_stratum_offsets,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            coefficient_stratum=coefficient_stratum,
            n_covariates=mean.shape[1],
        )
    )
    multipliers = _blend_with_scalar_fallback(
        adjustment,
        trust,
        support_excess=support_excess,
        effect_signal=effect_signal,
        design_signal=design_signal,
        community_occupancy=community_occupancy,
        prevalence_stratum=prevalence_stratum,
        design_stratum=design_stratum,
        coefficient_stratum=coefficient_stratum,
        normalization=calibration.normalization_multiplier,
        global_multiplier=calibration.global_scale_multiplier,
        ood_uncertainty_strength=calibration.ood_uncertainty_strength,
        ood_uncertainty_max_multiplier=calibration.ood_uncertainty_max_multiplier,
        ood_inflation_parameters=calibration.ood_inflation_parameters,
        min_multiplier=calibration.min_multiplier,
        max_multiplier=calibration.max_multiplier,
    )
    return np.clip(
        multipliers
        * _rare_validation_scale_multiplier(
            log_offsets=calibration.rare_validation_scale_log_offsets,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            support_excess=support_excess,
            community_occupancy=community_occupancy,
            support_threshold=calibration.rare_validation_scale_support_threshold,
            support_width=calibration.rare_validation_scale_support_width,
            community_threshold=calibration.rare_validation_scale_community_threshold,
            community_width=calibration.rare_validation_scale_community_width,
        )
        * _combined_shift_scale_multiplier(
            support_excess=support_excess,
            effect_signal=effect_signal,
            design_signal=design_signal,
            community_occupancy=community_occupancy,
            log_amplitude=calibration.combined_shift_scale_log_amplitude,
            effect_bin_log_amplitudes=(
                calibration.combined_shift_scale_effect_bin_log_amplitudes
            ),
            effect_bin_edges=calibration.combined_shift_scale_effect_bin_edges,
            context_gate_strength=(
                calibration.combined_shift_scale_context_gate_strength
            ),
            context_gate_intercept=(
                calibration.combined_shift_scale_context_gate_intercept
            ),
        )
        * _external_monotone_multiplier(
            support_excess=support_excess,
            effect_signal=effect_signal,
            log_offsets=calibration.external_monotone_log_offsets,
            effect_bin_edges=calibration.external_monotone_effect_bin_edges,
            support_threshold=calibration.external_monotone_support_threshold,
            support_width=calibration.external_monotone_support_width,
            effect_threshold=calibration.external_monotone_effect_threshold,
            effect_width=calibration.external_monotone_effect_width,
        ),
        calibration.min_multiplier,
        calibration.max_multiplier,
    )


def conditional_beta_ood_uncertainty_inflation(
    posterior: BetaPosterior,
    calibration: ConditionalBetaScaleCalibration,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    distribution: str | None = None,
    coefficient_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Return the bounded OOD uncertainty inflation applied to scale multipliers."""
    mean, _, design, response = _validated_arrays(posterior, X=X, Y=Y)
    names = (
        calibration.coefficient_names
        if coefficient_names is None
        else coefficient_names
    )
    calibration.validate_domain(
        distribution=distribution or calibration.distribution,
        n_covariates=mean.shape[1],
        n_species=mean.shape[2],
        coefficient_names=names,
    )
    mean = mean + _mean_bias_correction_array(
        calibration=calibration,
        Y=response,
        shape=mean.shape,
    )
    mean = mean + _rank_centering_correction_array(
        calibration=calibration,
        scale=_as_numpy(posterior.scale),
        Y=response,
        shape=mean.shape,
    )
    raw_features = _raw_features(
        mean=mean,
        scale=_as_numpy(posterior.scale),
        X=design,
        Y=response,
        distribution=distribution or calibration.distribution,
    )
    support_excess = _support_excess(
        raw_features,
        location=np.asarray(calibration.feature_location),
        scale=np.asarray(calibration.feature_scale),
        lower=np.asarray(calibration.support_lower),
        upper=np.asarray(calibration.support_upper),
        precision=np.asarray(calibration.support_precision),
        radius=calibration.support_radius,
        mean_magnitude=np.log1p(np.abs(mean)),
        mean_magnitude_location=calibration.mean_magnitude_location,
        mean_magnitude_scale=calibration.mean_magnitude_scale,
        mean_magnitude_lower=calibration.mean_magnitude_lower,
        mean_magnitude_upper=calibration.mean_magnitude_upper,
    )
    effect_signal = _effect_size_signal(
        np.log1p(np.abs(mean)),
        mean_magnitude_location=calibration.mean_magnitude_location,
        mean_magnitude_scale=calibration.mean_magnitude_scale,
    )
    design_signal = _design_information_signal(
        raw_features,
        location=np.asarray(calibration.feature_location),
        scale=np.asarray(calibration.feature_scale),
    )
    prevalence = _prevalence(response)
    community_occupancy = _community_occupancy_array(response, mean.shape)
    prevalence_stratum = _prevalence_stratum_index(
        np.broadcast_to(prevalence[:, None, :], mean.shape),
        prevalence_edges=calibration.prevalence_edges,
    )
    design_stratum = _design_information_stratum_index(design_signal)
    coefficient_stratum = _coefficient_stratum_index(mean.shape)
    return _ood_uncertainty_inflation(
        support_excess,
        effect_signal=effect_signal,
        design_signal=design_signal,
        community_occupancy=community_occupancy,
        prevalence_stratum=prevalence_stratum,
        design_stratum=design_stratum,
        coefficient_stratum=coefficient_stratum,
        strength=calibration.ood_uncertainty_strength,
        max_multiplier=calibration.ood_uncertainty_max_multiplier,
        learned_parameters=calibration.ood_inflation_parameters,
    )


def conditional_beta_support_trust(
    posterior: BetaPosterior,
    calibration: ConditionalBetaScaleCalibration,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    distribution: str | None = None,
    coefficient_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Return coefficient-level trust in the learned conditional adjustment."""
    mean, scale, design, response = _validated_arrays(posterior, X=X, Y=Y)
    names = (
        calibration.coefficient_names
        if coefficient_names is None
        else coefficient_names
    )
    calibration.validate_domain(
        distribution=distribution or calibration.distribution,
        n_covariates=mean.shape[1],
        n_species=mean.shape[2],
        coefficient_names=names,
    )
    mean = mean + _mean_bias_correction_array(
        calibration=calibration,
        Y=response,
        shape=mean.shape,
    )
    mean = mean + _rank_centering_correction_array(
        calibration=calibration,
        scale=scale,
        Y=response,
        shape=mean.shape,
    )
    raw_features = _raw_features(
        mean=mean,
        scale=scale,
        X=design,
        Y=response,
        distribution=distribution or calibration.distribution,
    )
    return _support_trust(
        raw_features,
        location=np.asarray(calibration.feature_location),
        scale=np.asarray(calibration.feature_scale),
        lower=np.asarray(calibration.support_lower),
        upper=np.asarray(calibration.support_upper),
        precision=np.asarray(calibration.support_precision),
        radius=calibration.support_radius,
        fallback_strength=calibration.fallback_strength,
        mean_magnitude=np.log1p(np.abs(mean)),
        mean_magnitude_location=calibration.mean_magnitude_location,
        mean_magnitude_scale=calibration.mean_magnitude_scale,
        mean_magnitude_lower=calibration.mean_magnitude_lower,
        mean_magnitude_upper=calibration.mean_magnitude_upper,
    )


def conditional_beta_mean_support_diagnostics(
    posterior: BetaPosterior,
    calibration: ConditionalBetaScaleCalibration,
) -> dict[str, float]:
    """Summarize posterior-mean magnitude relative to calibration support."""
    mean = _as_numpy(posterior.mean)
    if mean.ndim != 3:
        raise ValueError("posterior mean must have batch x covariate x species shape")
    calibration.validate_domain(
        n_covariates=mean.shape[1],
        n_species=mean.shape[2],
    )
    standardized = (
        np.log1p(np.abs(mean)) - calibration.mean_magnitude_location
    ) / calibration.mean_magnitude_scale
    outside = (standardized < calibration.mean_magnitude_lower) | (
        standardized > calibration.mean_magnitude_upper
    )
    return {
        "conditional_mean_magnitude_support_outside_fraction": float(np.mean(outside)),
        "conditional_mean_magnitude_support_max_abs_z": float(
            np.max(np.abs(standardized))
        ),
    }


def conditional_beta_effect_size_signal(
    posterior: BetaPosterior,
    calibration: ConditionalBetaScaleCalibration,
) -> np.ndarray:
    """Return the coefficient-level positive posterior-mean magnitude signal."""
    mean = _as_numpy(posterior.mean)
    if mean.ndim != 3:
        raise ValueError("posterior mean must have batch x covariate x species shape")
    calibration.validate_domain(
        n_covariates=mean.shape[1],
        n_species=mean.shape[2],
    )
    return _effect_size_signal(
        np.log1p(np.abs(mean)),
        mean_magnitude_location=calibration.mean_magnitude_location,
        mean_magnitude_scale=calibration.mean_magnitude_scale,
    )


def apply_conditional_beta_scale_calibration(
    posterior: BetaPosterior,
    calibration: ConditionalBetaScaleCalibration,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    distribution: str | None = None,
    coefficient_names: Sequence[str] | None = None,
) -> BetaPosterior:
    """Apply conditional mean-bias correction and coefficient-specific scaling."""
    multipliers = conditional_beta_scale_multipliers(
        posterior,
        calibration,
        X=X,
        Y=Y,
        distribution=distribution,
        coefficient_names=coefficient_names,
    )
    mean_array, scale_array, _, response = _validated_arrays(posterior, X=X, Y=Y)
    mean_array = mean_array + _mean_bias_correction_array(
        calibration=calibration,
        Y=response,
        shape=mean_array.shape,
    )
    mean_array = mean_array + _rank_centering_correction_array(
        calibration=calibration,
        scale=scale_array,
        Y=response,
        shape=mean_array.shape,
    )
    mean = tf.convert_to_tensor(mean_array, dtype=posterior.mean.dtype)
    multiplier_tensor = tf.cast(multipliers, mean.dtype)
    if posterior.scale_tril is None:
        return BetaPosterior(
            mean=mean,
            scale=tf.convert_to_tensor(posterior.scale) * multiplier_tensor,
        )

    scale_tril = tf.convert_to_tensor(posterior.scale_tril)
    per_species = tf.transpose(multiplier_tensor, [0, 2, 1])
    calibrated_tril = scale_tril * per_species[..., :, None]
    marginal = tf.sqrt(tf.reduce_sum(tf.square(calibrated_tril), axis=-1))
    return BetaPosterior(
        mean=mean,
        scale=tf.transpose(marginal, [0, 2, 1]),
        scale_tril=calibrated_tril,
    )


def _validated_arrays(
    posterior: BetaPosterior, *, X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = _as_numpy(posterior.mean)
    scale = _as_numpy(posterior.scale)
    design = np.asarray(X, dtype=float)
    response = np.asarray(Y, dtype=float)
    if mean.ndim != 3 or scale.shape != mean.shape:
        raise ValueError(
            "posterior mean and scale must have batch x covariate x species shape"
        )
    if (
        design.ndim != 3
        or design.shape[0] != mean.shape[0]
        or design.shape[2] != mean.shape[1]
    ):
        raise ValueError("X must have shape batch x sites x covariates")
    if response.ndim != 3 or response.shape != (
        mean.shape[0],
        design.shape[1],
        mean.shape[2],
    ):
        raise ValueError("Y must have shape batch x sites x species")
    if np.any(scale <= 0.0):
        raise ValueError("posterior scales must be positive")
    if not all(np.all(np.isfinite(value)) for value in (mean, scale, design, response)):
        raise ValueError("posterior, X, and Y values must be finite")
    return mean, scale, design, response


def _raw_features(
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    distribution: str,
) -> np.ndarray:
    prevalence = _prevalence(Y)
    epsilon = 0.5 / float(X.shape[1] + 1)
    prevalence = np.clip(prevalence, epsilon, 1.0 - epsilon)
    prevalence_logit = np.log(prevalence / (1.0 - prevalence))
    prevalence_feature = np.broadcast_to(prevalence_logit[:, None, :], mean.shape)
    information = _expected_design_information(mean, X, distribution=distribution)
    return np.stack(
        [
            prevalence_feature,
            np.log(np.maximum(information, 1e-12)),
            np.log(scale),
        ],
        axis=-1,
    )


def _prevalence(Y: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(Y) > 0.0, axis=1)


def _community_occupancy_array(
    Y: np.ndarray, shape: tuple[int, int, int]
) -> np.ndarray:
    """Return batch-level community occupancy broadcast to coefficient shape."""
    occupancy = np.mean(np.asarray(Y) > 0.0, axis=(1, 2))
    return np.broadcast_to(occupancy[:, None, None], shape)


def _prevalence_observation_weights(
    prevalence: np.ndarray,
    *,
    prevalence_weights: tuple[float, float, float],
    prevalence_edges: tuple[float, float],
) -> np.ndarray:
    low, high = prevalence_edges
    rare, intermediate, common = prevalence_weights
    return np.where(
        prevalence <= low,
        rare,
        np.where(prevalence <= high, intermediate, common),
    ).astype(float)


def _prevalence_group_masks(
    prevalence: np.ndarray, *, prevalence_edges: tuple[float, float]
) -> list[np.ndarray]:
    low, high = prevalence_edges
    candidates = [
        np.ones(prevalence.shape, dtype=bool),
        prevalence <= low,
        (prevalence > low) & (prevalence <= high),
        prevalence > high,
    ]
    return [mask for mask in candidates if np.count_nonzero(mask) >= 2]


def _in_domain_gate_group_masks(
    *,
    prevalence: np.ndarray,
    log_design_information: np.ndarray,
    coefficient_index: np.ndarray,
    prevalence_edges: tuple[float, float],
) -> list[np.ndarray]:
    """Return stratified groups used by the learned OOD in-domain gate."""
    prevalence = np.asarray(prevalence, dtype=float).reshape(-1)
    design_information = np.asarray(log_design_information, dtype=float).reshape(-1)
    coefficient_index = np.asarray(coefficient_index, dtype=int).reshape(-1)
    if not (prevalence.shape == design_information.shape == coefficient_index.shape):
        raise ValueError("in-domain gate group inputs must have matching shape")
    low, high = prevalence_edges
    candidates = [
        np.ones(prevalence.shape, dtype=bool),
        prevalence <= low,
        (prevalence > low) & (prevalence <= high),
        prevalence > high,
    ]
    if design_information.size:
        design_low, design_high = np.quantile(
            design_information, (1.0 / 3.0, 2.0 / 3.0)
        )
        candidates.extend(
            [
                design_information <= design_low,
                (design_information > design_low) & (design_information <= design_high),
                design_information > design_high,
            ]
        )
    for index in np.unique(coefficient_index):
        candidates.append(coefficient_index == index)
    unique_groups: list[np.ndarray] = []
    seen: set[bytes] = set()
    for candidate in candidates:
        if np.count_nonzero(candidate) < 2:
            continue
        key = np.packbits(candidate).tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique_groups.append(candidate)
    return unique_groups


def _expected_design_information(
    mean: np.ndarray, X: np.ndarray, *, distribution: str
) -> np.ndarray:
    linear = np.einsum("bnk,bks->bns", X, mean)
    key = str(distribution).lower()
    if key in {"normal", "gaussian"}:
        weight = np.ones(linear.shape, dtype=float)
    elif key in {"probit", "bernoulli", "binomial"}:
        probability = np.clip(ndtr(linear), 1e-9, 1.0 - 1e-9)
        density = np.exp(-0.5 * np.square(linear)) / np.sqrt(2.0 * np.pi)
        weight = np.square(density) / (probability * (1.0 - probability))
    elif key == "poisson":
        weight = np.exp(np.clip(linear, -20.0, 20.0))
    else:
        raise ValueError(
            f"unsupported distribution for conditional calibration: {distribution!r}"
        )
    return np.einsum("bnk,bns->bks", np.square(X), weight)


def _structured_design(
    raw_features: np.ndarray,
    *,
    location: np.ndarray,
    scale: np.ndarray,
    n_covariates: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    standardized = (raw_features - location) / scale
    flattened = standardized.reshape(-1, standardized.shape[-1])
    columns = []
    names = []
    for feature_index, feature_name in enumerate(_RAW_FEATURE_NAMES):
        values = flattened[:, feature_index]
        columns.extend([values, np.maximum(values, 0.0)])
        names.extend([feature_name, f"{feature_name}_positive_hinge"])

    shape = raw_features.shape[:3]
    coefficient_index = np.broadcast_to(
        np.arange(n_covariates)[None, :, None], shape
    ).reshape(-1)
    centered_identity = np.eye(n_covariates)[coefficient_index] - 1.0 / n_covariates
    prevalence = flattened[:, 0]
    for index in range(n_covariates):
        columns.append(centered_identity[:, index])
        names.append(f"coefficient_{index}")
    for index in range(n_covariates):
        columns.append(centered_identity[:, index] * prevalence)
        names.append(f"prevalence_by_coefficient_{index}")
    return np.column_stack(columns), tuple(names)


def _support_trust(
    raw_features: np.ndarray,
    *,
    location: np.ndarray,
    scale: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    precision: np.ndarray,
    radius: float,
    fallback_strength: float,
    mean_magnitude: np.ndarray,
    mean_magnitude_location: float,
    mean_magnitude_scale: float,
    mean_magnitude_lower: float,
    mean_magnitude_upper: float,
) -> np.ndarray:
    if fallback_strength <= 0.0:
        return np.ones(raw_features.shape[:3], dtype=float)
    total_excess = _support_excess(
        raw_features,
        location=location,
        scale=scale,
        lower=lower,
        upper=upper,
        precision=precision,
        radius=radius,
        mean_magnitude=mean_magnitude,
        mean_magnitude_location=mean_magnitude_location,
        mean_magnitude_scale=mean_magnitude_scale,
        mean_magnitude_lower=mean_magnitude_lower,
        mean_magnitude_upper=mean_magnitude_upper,
    )
    return np.exp(-float(fallback_strength) * np.square(total_excess))


def _support_excess(
    raw_features: np.ndarray,
    *,
    location: np.ndarray,
    scale: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    precision: np.ndarray,
    radius: float,
    mean_magnitude: np.ndarray,
    mean_magnitude_location: float,
    mean_magnitude_scale: float,
    mean_magnitude_lower: float,
    mean_magnitude_upper: float,
) -> np.ndarray:
    standardized = (raw_features - location) / scale
    lower_excess = np.maximum(lower - standardized, 0.0)
    upper_excess = np.maximum(standardized - upper, 0.0)
    box_excess = np.sqrt(np.sum(np.square(lower_excess + upper_excess), axis=-1))
    distance = np.sqrt(
        np.maximum(
            np.einsum(
                "...i,ij,...j->...",
                standardized,
                precision,
                standardized,
            ),
            0.0,
        )
    )
    radial_excess = np.maximum(distance - float(radius), 0.0)
    standardized_mean = (mean_magnitude - float(mean_magnitude_location)) / float(
        mean_magnitude_scale
    )
    mean_excess = np.maximum(
        float(mean_magnitude_lower) - standardized_mean, 0.0
    ) + np.maximum(standardized_mean - float(mean_magnitude_upper), 0.0)
    total_excess = np.sqrt(
        np.square(box_excess) + np.square(radial_excess) + np.square(mean_excess)
    )
    return total_excess


def _effect_size_signal(
    mean_magnitude: np.ndarray,
    *,
    mean_magnitude_location: float,
    mean_magnitude_scale: float,
) -> np.ndarray:
    """Return positive standardized posterior-mean magnitude."""
    standardized = (mean_magnitude - float(mean_magnitude_location)) / float(
        mean_magnitude_scale
    )
    return np.maximum(standardized, 0.0)


def _design_information_signal(
    raw_features: np.ndarray,
    *,
    location: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Return positive standardized design information for v8 gate constraints."""
    standardized = (raw_features[..., 1] - float(location[1])) / max(
        float(scale[1]), 1e-8
    )
    return np.maximum(standardized, 0.0)


def _prevalence_stratum_index(
    prevalence: np.ndarray,
    *,
    prevalence_edges: tuple[float, float],
) -> np.ndarray:
    """Return rare/intermediate/common prevalence stratum ids."""
    low, high = prevalence_edges
    prevalence = np.asarray(prevalence, dtype=float)
    return np.where(prevalence <= low, 0, np.where(prevalence <= high, 1, 2)).astype(
        np.int32
    )


def _design_information_stratum_index(design_signal: np.ndarray) -> np.ndarray:
    """Return low/intermediate/high positive design-information stratum ids."""
    design_signal = np.asarray(design_signal, dtype=float)
    return np.where(
        design_signal <= 0.0, 0, np.where(design_signal <= 1.0, 1, 2)
    ).astype(np.int32)


def _tertile_index(values: np.ndarray, *, mask: np.ndarray) -> np.ndarray:
    """Return low/intermediate/high tertile ids using values selected by mask."""
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    selected = values[mask]
    if selected.size < 3:
        return np.zeros(values.shape, dtype=np.int32)
    low, high = np.quantile(selected, (1.0 / 3.0, 2.0 / 3.0))
    return np.where(values <= low, 0, np.where(values <= high, 1, 2)).astype(np.int32)


def _coefficient_stratum_index(shape: tuple[int, int, int]) -> np.ndarray:
    """Return coefficient identity ids broadcast to Beta coefficient shape."""
    return np.broadcast_to(np.arange(shape[1], dtype=np.int32)[None, :, None], shape)


def _fit_prevalence_coefficient_mean_bias(
    *,
    truth: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    prevalence_stratum: np.ndarray,
    coefficient_stratum: np.ndarray,
    mask: np.ndarray,
    shrinkage: float,
) -> tuple[tuple[float, ...], ...]:
    """Fit a shrunk residual-mean correction by prevalence and coefficient."""
    residual = np.asarray(truth, dtype=float) - np.asarray(mean, dtype=float)
    values = np.zeros((3, mean.shape[1]), dtype=float)
    for prevalence_index in range(3):
        for coefficient_index in range(mean.shape[1]):
            group = (
                mask
                & (prevalence_stratum == prevalence_index)
                & (coefficient_stratum == coefficient_index)
            )
            count = int(np.count_nonzero(group))
            if count < 2:
                continue
            group_residual = residual[group]
            group_scale = scale[group]
            shrink = count / (count + float(shrinkage))
            raw_bias = float(np.mean(group_residual)) * shrink
            max_abs = float(np.median(group_scale))
            if max_abs > 0.0:
                raw_bias = float(np.clip(raw_bias, -max_abs, max_abs))
            values[prevalence_index, coefficient_index] = raw_bias
    return tuple(tuple(float(value) for value in row) for row in values)


def _fit_rare_balanced_mean_bias_correction(
    *,
    rare_calibration_batches: Sequence[ConditionalBetaOODCalibrationBatch],
    rare_validation_batches: Sequence[ConditionalBetaOODCalibrationBatch] | None,
    validation_truth: np.ndarray,
    validation_mean: np.ndarray,
    validation_scale: np.ndarray,
    validation_design: np.ndarray,
    validation_response: np.ndarray,
    validation_mask: np.ndarray,
    validation_prevalence_stratum: np.ndarray,
    validation_coefficient_stratum: np.ndarray,
    distribution: str,
    prevalence_edges: tuple[float, float],
    shrinkage: float,
    nominal_level: float,
    z_value: float,
) -> tuple[tuple[tuple[float, ...], ...], dict[str, Any]]:
    """Fit a rare-only residual mean head from balanced rare simulations."""
    rare_truth_arrays = []
    rare_mean_arrays = []
    rare_scale_arrays = []
    rare_design_arrays = []
    rare_response_arrays = []
    rare_labels = []
    for batch in rare_calibration_batches:
        batch_mean, batch_scale, batch_design, batch_response = _validated_arrays(
            batch.posterior,
            X=batch.X,
            Y=batch.Y,
        )
        batch_truth = np.asarray(batch.beta_true, dtype=float)
        if batch_truth.shape != batch_mean.shape or not np.all(
            np.isfinite(batch_truth)
        ):
            raise ValueError(
                "rare calibration beta_true must be finite and match posterior shape"
            )
        if (
            batch_mean.shape[1:] != validation_mean.shape[1:]
            or batch_response.shape[1:] != validation_response.shape[1:]
        ):
            raise ValueError("rare calibration batches must match calibration shape")
        rare_truth_arrays.append(batch_truth)
        rare_mean_arrays.append(batch_mean)
        rare_scale_arrays.append(batch_scale)
        rare_design_arrays.append(batch_design)
        rare_response_arrays.append(batch_response)
        rare_labels.extend([str(batch.label)] * int(batch_mean.shape[0]))
    if not rare_truth_arrays:
        return (
            tuple(
                tuple(0.0 for _ in range(validation_mean.shape[1])) for _ in range(3)
            ),
            {
                "n_observations": 0,
                "selected_shrinkage": 0.0,
                "validation_rank_error": 0.0,
                "diagnostics": {},
            },
        )

    rare_truth = np.concatenate(rare_truth_arrays, axis=0)
    rare_mean = np.concatenate(rare_mean_arrays, axis=0)
    rare_scale = np.concatenate(rare_scale_arrays, axis=0)
    rare_design = np.concatenate(rare_design_arrays, axis=0)
    rare_response = np.concatenate(rare_response_arrays, axis=0)
    rare_label_array = np.asarray(rare_labels, dtype=object)
    rare_prevalence = _prevalence(rare_response)
    rare_prevalence_by_coefficient = np.broadcast_to(
        rare_prevalence[:, None, :], rare_mean.shape
    )
    rare_prevalence_stratum = _prevalence_stratum_index(
        rare_prevalence_by_coefficient,
        prevalence_edges=prevalence_edges,
    )
    rare_coefficient_stratum = _coefficient_stratum_index(rare_mean.shape)
    rare_raw_features = _raw_features(
        mean=rare_mean,
        scale=rare_scale,
        X=rare_design,
        Y=rare_response,
        distribution=distribution,
    )
    rare_log_design = rare_raw_features[..., 1]
    rare_mask = np.isfinite(rare_truth) & np.isfinite(rare_mean) & (rare_scale > 0.0)
    rare_mask &= rare_prevalence_stratum == 0
    rare_design_stratum = _tertile_index(rare_log_design, mask=rare_mask)
    rare_regime = np.broadcast_to(rare_label_array[:, None, None], rare_mean.shape)

    residual = rare_truth - rare_mean
    signed_error = residual / np.maximum(rare_scale, 1e-8)
    candidate = np.zeros((3, validation_mean.shape[1]), dtype=float)
    rare_pool_by_coefficient: list[dict[str, float]] = []
    rare_pool_by_cell: list[dict[str, Any]] = []
    regimes = sorted(str(value) for value in np.unique(rare_label_array))
    for coefficient_index in range(validation_mean.shape[1]):
        group = rare_mask & (rare_coefficient_stratum == coefficient_index)
        count = int(np.count_nonzero(group))
        coefficient_summary = {
            "coefficient_index": int(coefficient_index),
            "count": count,
            "residual_mean": 0.0,
            "residual_sd": 0.0,
            "standardized_residual_mean": 0.0,
            "rank_mean": 0.0,
            "candidate_offset": 0.0,
        }
        if count < 8:
            rare_pool_by_coefficient.append(coefficient_summary)
            continue
        cell_offsets = []
        for regime in regimes:
            for design_index in range(3):
                cell = (
                    group
                    & (rare_regime == regime)
                    & (rare_design_stratum == design_index)
                )
                cell_count = int(np.count_nonzero(cell))
                cell_summary: dict[str, Any] = {
                    "coefficient_index": int(coefficient_index),
                    "regime": str(regime),
                    "design_stratum": int(design_index),
                    "count": cell_count,
                    "residual_mean": 0.0,
                    "standardized_residual_mean": 0.0,
                    "rank_mean": 0.0,
                    "candidate_offset": 0.0,
                }
                if cell_count >= 4:
                    raw_bias = float(np.mean(residual[cell]))
                    max_abs = 0.5 * float(np.median(rare_scale[cell]))
                    if max_abs > 0.0:
                        raw_bias = float(np.clip(raw_bias, -max_abs, max_abs))
                    empirical_shrink = cell_count / (cell_count + float(shrinkage))
                    cell_offset = empirical_shrink * raw_bias
                    cell_offsets.append(cell_offset)
                    cell_summary.update(
                        {
                            "residual_mean": float(np.mean(residual[cell])),
                            "standardized_residual_mean": float(
                                np.mean(signed_error[cell])
                            ),
                            "rank_mean": float(np.mean(ndtr(signed_error[cell]))),
                            "candidate_offset": float(cell_offset),
                        }
                    )
                rare_pool_by_cell.append(cell_summary)
        if cell_offsets:
            candidate[0, coefficient_index] = float(np.mean(cell_offsets))
        coefficient_summary.update(
            {
                "residual_mean": float(np.mean(residual[group])),
                "residual_sd": float(np.std(residual[group])),
                "standardized_residual_mean": float(np.mean(signed_error[group])),
                "rank_mean": float(np.mean(ndtr(signed_error[group]))),
                "candidate_offset": float(candidate[0, coefficient_index]),
            }
        )
        rare_pool_by_coefficient.append(coefficient_summary)

    raw_features = _raw_features(
        mean=validation_mean,
        scale=validation_scale,
        X=validation_design,
        Y=validation_response,
        distribution=distribution,
    )
    log_design = raw_features[..., 1]
    design_low, design_high = np.quantile(
        log_design[validation_mask], (1.0 / 3.0, 2.0 / 3.0)
    )
    rare_validation_mask = validation_mask & (validation_prevalence_stratum == 0)
    intermediate_design_mask = validation_mask & (
        (log_design > design_low) & (log_design <= design_high)
    )
    high_design_mask = validation_mask & (log_design > design_high)

    def metrics(offsets: np.ndarray) -> dict[str, float]:
        offset = offsets[
            validation_prevalence_stratum,
            validation_coefficient_stratum,
        ]
        signed_error = (validation_truth - (validation_mean + offset)) / np.maximum(
            validation_scale, 1e-8
        )
        selected = signed_error[validation_mask]
        rare_selected = signed_error[rare_validation_mask]
        rare_rank_error = (
            abs(float(np.mean(ndtr(rare_selected))) - 0.5)
            if rare_selected.size >= 8
            else abs(float(np.mean(ndtr(selected))) - 0.5)
        )
        overall_rank_error = abs(float(np.mean(ndtr(selected))) - 0.5)
        coverage = float(np.mean(np.abs(selected) <= z_value))
        intermediate_coverage = (
            float(np.mean(np.abs(signed_error[intermediate_design_mask]) <= z_value))
            if np.count_nonzero(intermediate_design_mask) >= 8
            else coverage
        )
        high_coverage = (
            float(np.mean(np.abs(signed_error[high_design_mask]) <= z_value))
            if np.count_nonzero(high_design_mask) >= 8
            else coverage
        )
        objective = rare_rank_error + 0.25 * overall_rank_error
        objective += max(0.0, max(0.90, nominal_level - 0.04) - coverage) * 4.0
        objective += max(0.0, 0.90 - intermediate_coverage) * 4.0
        objective += max(0.0, 0.90 - high_coverage) * 4.0
        return {
            "objective": objective,
            "rare_rank_error": rare_rank_error,
            "overall_rank_error": overall_rank_error,
            "coverage": coverage,
            "intermediate_coverage": intermediate_coverage,
            "high_coverage": high_coverage,
        }

    zero_offsets = np.zeros_like(candidate)
    zero_metrics = metrics(zero_offsets)
    best_offsets = zero_offsets
    best_shrinkage = 0.0
    best_metrics = zero_metrics
    shrinkage_grid = [
        {
            "shrinkage": 0.0,
            **{key: float(value) for key, value in zero_metrics.items()},
        }
    ]
    for shrink_factor in (0.125, 0.25, 0.375, 0.5, 0.75, 1.0):
        offsets = shrink_factor * candidate
        current = metrics(offsets)
        accepted = (
            current["objective"] + 1e-8 < best_metrics["objective"]
            and current["rare_rank_error"] <= zero_metrics["rare_rank_error"] + 1e-8
            and current["coverage"] >= max(0.90, zero_metrics["coverage"] - 0.01)
            and current["intermediate_coverage"]
            >= max(0.90, zero_metrics["intermediate_coverage"] - 0.01)
            and current["high_coverage"]
            >= max(0.90, zero_metrics["high_coverage"] - 0.01)
        )
        shrinkage_grid.append(
            {
                "shrinkage": float(shrink_factor),
                "accepted_against_current_best": bool(accepted),
                **{key: float(value) for key, value in current.items()},
            }
        )
        if accepted:
            best_offsets = offsets
            best_shrinkage = shrink_factor
            best_metrics = current

    if best_shrinkage == 0.0:
        best_offsets = zero_offsets
    rare_prevalence_values = rare_prevalence.reshape(-1)
    rare_pool_summary = {
        "n_batches": int(rare_truth.shape[0]),
        "n_observations": int(np.count_nonzero(rare_mask)),
        "prevalence_min": float(np.min(rare_prevalence_values)),
        "prevalence_mean": float(np.mean(rare_prevalence_values)),
        "prevalence_median": float(np.median(rare_prevalence_values)),
        "prevalence_max": float(np.max(rare_prevalence_values)),
        "rare_species_fraction": float(np.mean(rare_prevalence <= prevalence_edges[0])),
        "regime_counts": {
            str(regime): int(np.count_nonzero(rare_label_array == regime))
            for regime in regimes
        },
        "rare_observations_by_regime": {
            str(regime): int(np.count_nonzero(rare_mask & (rare_regime == regime)))
            for regime in regimes
        },
        "rare_observations_by_design_stratum": {
            str(index): int(
                np.count_nonzero(rare_mask & (rare_design_stratum == index))
            )
            for index in range(3)
        },
    }
    validation_summary = {
        "n_observations": int(np.count_nonzero(validation_mask)),
        "n_rare_observations": int(np.count_nonzero(rare_validation_mask)),
        "n_intermediate_design_observations": int(
            np.count_nonzero(intermediate_design_mask)
        ),
        "n_high_design_observations": int(np.count_nonzero(high_design_mask)),
        "zero_metrics": {key: float(value) for key, value in zero_metrics.items()},
        "best_metrics": {key: float(value) for key, value in best_metrics.items()},
    }
    independent_validation = None
    if rare_validation_batches:
        independent_validation = _rare_head_validation_context_from_batches(
            rare_validation_batches,
            reference_mean=validation_mean,
            distribution=distribution,
            prevalence_edges=prevalence_edges,
            z_value=z_value,
            nominal_level=nominal_level,
        )
        independent_zero_metrics = _rare_head_validation_metrics(
            offsets=zero_offsets,
            context=independent_validation,
        )
        independent_best_metrics = _rare_head_validation_metrics(
            offsets=best_offsets,
            context=independent_validation,
        )
        independent_ok = _rare_head_validation_non_degrading(
            current=independent_best_metrics,
            baseline=independent_zero_metrics,
            nominal_level=nominal_level,
        )
        for row in shrinkage_grid:
            offsets = float(row["shrinkage"]) * candidate
            row["independent_validation"] = _rare_head_validation_metrics(
                offsets=offsets,
                context=independent_validation,
            )
        validation_summary["independent"] = {
            "n_observations": int(np.count_nonzero(independent_validation["mask"])),
            "n_rare_observations": int(
                np.count_nonzero(independent_validation["rare_mask"])
            ),
            "n_intermediate_design_observations": int(
                np.count_nonzero(independent_validation["intermediate_design_mask"])
            ),
            "n_high_design_observations": int(
                np.count_nonzero(independent_validation["high_design_mask"])
            ),
            "zero_metrics": independent_zero_metrics,
            "best_metrics": independent_best_metrics,
            "non_degrading": bool(independent_ok),
        }
        if best_shrinkage > 0.0 and not independent_ok:
            best_offsets = zero_offsets
            best_shrinkage = 0.0
            best_metrics = zero_metrics
            validation_summary["best_metrics"] = {
                key: float(value) for key, value in best_metrics.items()
            }
            validation_summary["independent"]["selected_reset_to_zero"] = True
    return (
        tuple(tuple(float(value) for value in row) for row in best_offsets),
        {
            "n_observations": int(np.count_nonzero(rare_mask)),
            "selected_shrinkage": float(best_shrinkage),
            "validation_rank_error": float(best_metrics["rare_rank_error"]),
            "diagnostics": {
                "candidate_offsets": [
                    [float(value) for value in row] for row in candidate
                ],
                "selected_offsets": [
                    [float(value) for value in row] for row in best_offsets
                ],
                "shrinkage_grid": shrinkage_grid,
                "rare_pool": rare_pool_summary,
                "rare_pool_by_coefficient": rare_pool_by_coefficient,
                "rare_pool_by_cell": rare_pool_by_cell,
                "validation": validation_summary,
            },
        },
    )


def _rare_head_validation_context_from_batches(
    batches: Sequence[ConditionalBetaOODCalibrationBatch],
    *,
    reference_mean: np.ndarray,
    distribution: str,
    prevalence_edges: tuple[float, float],
    z_value: float,
    nominal_level: float,
) -> dict[str, Any]:
    truth_arrays = []
    mean_arrays = []
    scale_arrays = []
    design_arrays = []
    response_arrays = []
    for batch in batches:
        mean, scale, design, response = _validated_arrays(
            batch.posterior,
            X=batch.X,
            Y=batch.Y,
        )
        truth = np.asarray(batch.beta_true, dtype=float)
        if truth.shape != mean.shape or not np.all(np.isfinite(truth)):
            raise ValueError(
                "rare validation beta_true must be finite and match posterior shape"
            )
        if mean.shape[1:] != reference_mean.shape[1:]:
            raise ValueError("rare validation batches must match calibration shape")
        truth_arrays.append(truth)
        mean_arrays.append(mean)
        scale_arrays.append(scale)
        design_arrays.append(design)
        response_arrays.append(response)
    truth = np.concatenate(truth_arrays, axis=0)
    mean = np.concatenate(mean_arrays, axis=0)
    scale = np.concatenate(scale_arrays, axis=0)
    design = np.concatenate(design_arrays, axis=0)
    response = np.concatenate(response_arrays, axis=0)
    prevalence = _prevalence(response)
    prevalence_by_coefficient = np.broadcast_to(prevalence[:, None, :], mean.shape)
    prevalence_stratum = _prevalence_stratum_index(
        prevalence_by_coefficient,
        prevalence_edges=prevalence_edges,
    )
    coefficient_stratum = _coefficient_stratum_index(mean.shape)
    raw_features = _raw_features(
        mean=mean,
        scale=scale,
        X=design,
        Y=response,
        distribution=distribution,
    )
    log_design = raw_features[..., 1]
    mask = np.isfinite(truth) & np.isfinite(mean) & (scale > 0.0)
    if not np.any(mask):
        raise ValueError("rare validation batches contain no valid coefficients")
    design_low, design_high = np.quantile(log_design[mask], (1.0 / 3.0, 2.0 / 3.0))
    rare_mask = mask & (prevalence_stratum == 0)
    intermediate_design_mask = mask & (
        (log_design > design_low) & (log_design <= design_high)
    )
    high_design_mask = mask & (log_design > design_high)
    return {
        "truth": truth,
        "mean": mean,
        "scale": scale,
        "mask": mask,
        "rare_mask": rare_mask,
        "intermediate_design_mask": intermediate_design_mask,
        "high_design_mask": high_design_mask,
        "prevalence_stratum": prevalence_stratum,
        "coefficient_stratum": coefficient_stratum,
        "z_value": float(z_value),
        "nominal_level": float(nominal_level),
    }


def _rare_head_validation_metrics(
    *,
    offsets: np.ndarray,
    context: dict[str, Any],
) -> dict[str, float]:
    offset = offsets[
        context["prevalence_stratum"],
        context["coefficient_stratum"],
    ]
    signed_error = (context["truth"] - (context["mean"] + offset)) / np.maximum(
        context["scale"], 1e-8
    )
    selected = signed_error[context["mask"]]
    rare_selected = signed_error[context["rare_mask"]]
    rare_rank_error = (
        abs(float(np.mean(ndtr(rare_selected))) - 0.5)
        if rare_selected.size >= 8
        else abs(float(np.mean(ndtr(selected))) - 0.5)
    )
    overall_rank_error = abs(float(np.mean(ndtr(selected))) - 0.5)
    coverage = float(np.mean(np.abs(selected) <= context["z_value"]))
    intermediate_coverage = (
        float(
            np.mean(
                np.abs(signed_error[context["intermediate_design_mask"]])
                <= context["z_value"]
            )
        )
        if np.count_nonzero(context["intermediate_design_mask"]) >= 8
        else coverage
    )
    high_coverage = (
        float(
            np.mean(
                np.abs(signed_error[context["high_design_mask"]]) <= context["z_value"]
            )
        )
        if np.count_nonzero(context["high_design_mask"]) >= 8
        else coverage
    )
    objective = rare_rank_error + 0.25 * overall_rank_error
    objective += max(0.0, max(0.90, context["nominal_level"] - 0.04) - coverage) * 4.0
    objective += max(0.0, 0.90 - intermediate_coverage) * 4.0
    objective += max(0.0, 0.90 - high_coverage) * 4.0
    return {
        "objective": float(objective),
        "rare_rank_error": float(rare_rank_error),
        "overall_rank_error": float(overall_rank_error),
        "coverage": float(coverage),
        "intermediate_coverage": float(intermediate_coverage),
        "high_coverage": float(high_coverage),
    }


def _rare_head_validation_non_degrading(
    *,
    current: dict[str, float],
    baseline: dict[str, float],
    nominal_level: float,
) -> bool:
    return (
        current["rare_rank_error"] <= baseline["rare_rank_error"] + 1e-8
        and current["coverage"] >= max(0.90, baseline["coverage"] - 0.01)
        and current["intermediate_coverage"]
        >= max(0.90, baseline["intermediate_coverage"] - 0.01)
        and current["high_coverage"] >= max(0.90, baseline["high_coverage"] - 0.01)
        and current["coverage"] >= max(0.90, float(nominal_level) - 0.05)
    )


def _fit_rare_validation_scale_correction(
    *,
    rare_validation_batches: Sequence[ConditionalBetaOODCalibrationBatch],
    location: np.ndarray,
    feature_scale: np.ndarray,
    feature_names: tuple[str, ...],
    support_lower: np.ndarray,
    support_upper: np.ndarray,
    support_precision: np.ndarray,
    support_radius: float,
    mean_magnitude_location: float,
    mean_magnitude_scale: float,
    mean_magnitude_lower: float,
    mean_magnitude_upper: float,
    normalization: float,
    global_multiplier: float,
    fallback_strength: float,
    fitted_weights: np.ndarray,
    mean_bias_correction: tuple[tuple[float, ...], ...],
    rank_centering_offsets: tuple[tuple[float, ...], ...],
    base_scale_stratum_offsets: tuple[float, ...],
    distribution: str,
    n_covariates: int,
    n_species: int,
    prevalence_edges: tuple[float, float],
    ood_uncertainty_strength: float,
    ood_uncertainty_max_multiplier: float,
    ood_inflation_parameters: tuple[float, ...] | None,
    in_domain_signed_error: np.ndarray,
    in_domain_base_multiplier: np.ndarray,
    in_domain_support_excess: np.ndarray,
    in_domain_community_occupancy: np.ndarray,
    in_domain_prevalence_stratum: np.ndarray,
    in_domain_design_stratum: np.ndarray,
    nominal_level: float,
    z_value: float,
    min_multiplier: float,
    max_multiplier: float,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Fit a gated design-stratum scale correction from rare-validation data."""
    signed_error_arrays = []
    multiplier_arrays = []
    mask_arrays = []
    prevalence_stratum_arrays = []
    design_stratum_arrays = []
    support_excess_arrays = []
    community_occupancy_arrays = []
    label_arrays = []
    for batch in rare_validation_batches:
        mean, scale, design, response = _validated_arrays(
            batch.posterior,
            X=batch.X,
            Y=batch.Y,
        )
        if mean.shape[1] != n_covariates or mean.shape[2] != n_species:
            raise ValueError("rare validation scale batch domain mismatch")
        truth = np.asarray(batch.beta_true, dtype=float)
        if truth.shape != mean.shape or not np.all(np.isfinite(truth)):
            raise ValueError(
                "rare validation scale beta_true must be finite and match posterior shape"
            )
        prevalence = _prevalence(response)
        prevalence_by_coefficient = np.broadcast_to(prevalence[:, None, :], mean.shape)
        prevalence_stratum = _prevalence_stratum_index(
            prevalence_by_coefficient,
            prevalence_edges=prevalence_edges,
        )
        coefficient_stratum = _coefficient_stratum_index(mean.shape)
        if mean_bias_correction:
            bias_values = np.asarray(mean_bias_correction, dtype=float)
            mean = mean + bias_values[prevalence_stratum, coefficient_stratum]
        if rank_centering_offsets:
            centering_values = np.asarray(rank_centering_offsets, dtype=float)
            mean = (
                mean + centering_values[prevalence_stratum, coefficient_stratum] * scale
            )
        raw_features = _raw_features(
            mean=mean,
            scale=scale,
            X=design,
            Y=response,
            distribution=distribution,
        )
        design_matrix, names = _structured_design(
            raw_features,
            location=location,
            scale=feature_scale,
            n_covariates=n_covariates,
        )
        if names != feature_names:
            raise ValueError("rare validation scale feature specification mismatch")
        design_signal = _design_information_signal(
            raw_features,
            location=location,
            scale=feature_scale,
        )
        design_stratum = _design_information_stratum_index(design_signal)
        base_log_offset = _base_scale_stratum_log_offset(
            offsets=base_scale_stratum_offsets,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            coefficient_stratum=coefficient_stratum,
            n_covariates=n_covariates,
        )
        adjustment = np.exp(
            np.clip(
                design_matrix @ fitted_weights + base_log_offset.reshape(-1),
                -20.0,
                20.0,
            )
        ).reshape(mean.shape)
        trust = _support_trust(
            raw_features,
            location=location,
            scale=feature_scale,
            lower=support_lower,
            upper=support_upper,
            precision=support_precision,
            radius=support_radius,
            fallback_strength=fallback_strength,
            mean_magnitude=np.log1p(np.abs(mean)),
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
        )
        support_excess = _support_excess(
            raw_features,
            location=location,
            scale=feature_scale,
            lower=support_lower,
            upper=support_upper,
            precision=support_precision,
            radius=support_radius,
            mean_magnitude=np.log1p(np.abs(mean)),
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
        )
        effect_signal = _effect_size_signal(
            np.log1p(np.abs(mean)),
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
        )
        base_multiplier = _blend_with_scalar_fallback(
            adjustment,
            trust,
            support_excess=support_excess,
            effect_signal=effect_signal,
            design_signal=design_signal,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            coefficient_stratum=coefficient_stratum,
            normalization=normalization,
            global_multiplier=global_multiplier,
            ood_uncertainty_strength=ood_uncertainty_strength,
            ood_uncertainty_max_multiplier=ood_uncertainty_max_multiplier,
            ood_inflation_parameters=ood_inflation_parameters,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
        valid = np.isfinite(truth) & np.isfinite(mean) & (scale > 0.0)
        community_occupancy = _community_occupancy_array(response, mean.shape)
        signed_error_arrays.append((truth - mean) / np.maximum(scale, 1e-8))
        multiplier_arrays.append(base_multiplier)
        mask_arrays.append(valid)
        prevalence_stratum_arrays.append(prevalence_stratum)
        design_stratum_arrays.append(design_stratum)
        support_excess_arrays.append(support_excess)
        community_occupancy_arrays.append(community_occupancy)
        label_arrays.append(np.full(mean.shape, str(batch.label), dtype=object))

    signed_error = np.concatenate(signed_error_arrays, axis=0)
    base_multiplier = np.concatenate(multiplier_arrays, axis=0)
    mask = np.concatenate(mask_arrays, axis=0)
    prevalence_stratum = np.concatenate(prevalence_stratum_arrays, axis=0)
    design_stratum = np.concatenate(design_stratum_arrays, axis=0)
    support_excess = np.concatenate(support_excess_arrays, axis=0)
    community_occupancy = np.concatenate(community_occupancy_arrays, axis=0)
    labels = np.concatenate(label_arrays, axis=0)
    if np.count_nonzero(mask) < 32:
        return (
            (0.0, 0.0, 0.0),
            {
                "selected_shrinkage": 0.0,
                "diagnostics": {
                    "reason": "insufficient_validation_observations",
                    "n_observations": int(np.count_nonzero(mask)),
                },
            },
        )

    selected_in_domain_support = np.asarray(in_domain_support_excess, dtype=float)
    support_threshold = float(np.quantile(selected_in_domain_support, 0.95))
    active_validation_support = support_excess[
        mask & (support_excess > support_threshold)
    ]
    if active_validation_support.size >= 8:
        support_width = float(
            max(np.std(active_validation_support - support_threshold), 0.25)
        )
    else:
        support_width = float(max(np.std(selected_in_domain_support), 0.25))
    selected_in_domain_community = np.asarray(
        in_domain_community_occupancy, dtype=float
    )
    community_threshold = float(np.quantile(selected_in_domain_community, 0.05))
    low_community_validation = community_occupancy[
        mask & (community_occupancy < community_threshold)
    ]
    if low_community_validation.size >= 8:
        community_width = float(
            max(np.std(community_threshold - low_community_validation), 0.02)
        )
    else:
        community_width = float(max(np.std(selected_in_domain_community), 0.02))
    floor = max(0.90, float(nominal_level) - 0.05)
    candidate = np.zeros(3, dtype=float)
    for design_index in range(3):
        group = mask & (design_stratum == design_index)
        if np.count_nonzero(group) < 8:
            continue
        if (
            _coverage_with_multiplier(
                signed_error=signed_error,
                multiplier=base_multiplier,
                mask=group,
                z_value=z_value,
            )
            >= floor
        ):
            continue
        lower, upper = 0.0, np.log(4.0)
        for _ in range(48):
            midpoint = 0.5 * (lower + upper)
            adjusted_multiplier = np.clip(
                base_multiplier
                * _rare_validation_scale_multiplier(
                    log_offsets=np.full(3, midpoint, dtype=float),
                    prevalence_stratum=prevalence_stratum,
                    design_stratum=design_stratum,
                    support_excess=support_excess,
                    community_occupancy=community_occupancy,
                    support_threshold=support_threshold,
                    support_width=support_width,
                    community_threshold=community_threshold,
                    community_width=community_width,
                ),
                min_multiplier,
                max_multiplier,
            )
            coverage = _coverage_with_multiplier(
                signed_error=signed_error,
                multiplier=adjusted_multiplier,
                mask=group,
                z_value=z_value,
            )
            if coverage < floor:
                lower = midpoint
            else:
                upper = midpoint
        candidate[design_index] = upper

    zero_offsets = np.zeros(3, dtype=float)
    zero_metrics = _rare_validation_scale_metrics(
        signed_error=signed_error,
        base_multiplier=base_multiplier,
        mask=mask,
        prevalence_stratum=prevalence_stratum,
        design_stratum=design_stratum,
        support_excess=support_excess,
        community_occupancy=community_occupancy,
        labels=labels,
        log_offsets=zero_offsets,
        support_threshold=support_threshold,
        support_width=support_width,
        community_threshold=community_threshold,
        community_width=community_width,
        z_value=z_value,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
    )
    in_domain_mask = np.ones(in_domain_signed_error.shape, dtype=bool)
    in_domain_zero_metrics = _rare_validation_scale_metrics(
        signed_error=in_domain_signed_error,
        base_multiplier=in_domain_base_multiplier,
        mask=in_domain_mask,
        prevalence_stratum=in_domain_prevalence_stratum,
        design_stratum=in_domain_design_stratum,
        support_excess=in_domain_support_excess,
        community_occupancy=in_domain_community_occupancy,
        labels=np.full(in_domain_signed_error.shape, "in_domain", dtype=object),
        log_offsets=zero_offsets,
        support_threshold=support_threshold,
        support_width=support_width,
        community_threshold=community_threshold,
        community_width=community_width,
        z_value=z_value,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
    )
    best_offsets = zero_offsets
    best_shrinkage = 0.0
    best_metrics = zero_metrics
    best_in_domain_metrics = in_domain_zero_metrics
    shrinkage_grid = [
        {
            "shrinkage": 0.0,
            "in_domain_guard": in_domain_zero_metrics,
            **zero_metrics,
        }
    ]
    for shrinkage in (0.25, 0.5, 0.75, 1.0):
        offsets = shrinkage * candidate
        current = _rare_validation_scale_metrics(
            signed_error=signed_error,
            base_multiplier=base_multiplier,
            mask=mask,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            support_excess=support_excess,
            community_occupancy=community_occupancy,
            labels=labels,
            log_offsets=offsets,
            support_threshold=support_threshold,
            support_width=support_width,
            community_threshold=community_threshold,
            community_width=community_width,
            z_value=z_value,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
        in_domain_current = _rare_validation_scale_metrics(
            signed_error=in_domain_signed_error,
            base_multiplier=in_domain_base_multiplier,
            mask=in_domain_mask,
            prevalence_stratum=in_domain_prevalence_stratum,
            design_stratum=in_domain_design_stratum,
            support_excess=in_domain_support_excess,
            community_occupancy=in_domain_community_occupancy,
            labels=np.full(in_domain_signed_error.shape, "in_domain", dtype=object),
            log_offsets=offsets,
            support_threshold=support_threshold,
            support_width=support_width,
            community_threshold=community_threshold,
            community_width=community_width,
            z_value=z_value,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
        gate_ok = _rare_validation_scale_gate_ok(
            current=current,
            baseline=zero_metrics,
            in_domain_current=in_domain_current,
            in_domain_baseline=in_domain_zero_metrics,
            floor=floor,
        )
        row = {
            "shrinkage": float(shrinkage),
            "accepted_against_current_best": bool(
                gate_ok and current["objective"] + 1e-8 < best_metrics["objective"]
            ),
            "in_domain_guard": in_domain_current,
            **current,
        }
        shrinkage_grid.append(row)
        if gate_ok and current["objective"] + 1e-8 < best_metrics["objective"]:
            best_offsets = offsets
            best_shrinkage = shrinkage
            best_metrics = current
            best_in_domain_metrics = in_domain_current

    if best_shrinkage == 0.0:
        best_offsets = zero_offsets
        best_metrics = zero_metrics
        best_in_domain_metrics = in_domain_zero_metrics
    diagnostics = {
        "n_observations": int(np.count_nonzero(mask)),
        "n_rare_observations": int(np.count_nonzero(mask & (prevalence_stratum == 0))),
        "candidate_log_offsets": [float(value) for value in candidate],
        "candidate_multipliers": [float(np.exp(value)) for value in candidate],
        "selected_log_offsets": [float(value) for value in best_offsets],
        "selected_multipliers": [float(np.exp(value)) for value in best_offsets],
        "activation": {
            "kind": "thresholded_low_community_or_support_excess",
            "threshold": float(support_threshold),
            "width": float(support_width),
            "community_occupancy_threshold": float(community_threshold),
            "community_occupancy_width": float(community_width),
            "in_domain_active_fraction": float(
                np.mean(
                    _rare_validation_scale_activation(
                        in_domain_support_excess,
                        community_occupancy=in_domain_community_occupancy,
                        prevalence_stratum=in_domain_prevalence_stratum,
                        design_stratum=in_domain_design_stratum,
                        support_threshold=support_threshold,
                        support_width=support_width,
                        community_threshold=community_threshold,
                        community_width=community_width,
                    )
                    > 0.5
                )
            ),
            "validation_active_fraction": float(
                np.mean(
                    _rare_validation_scale_activation(
                        support_excess[mask],
                        community_occupancy=community_occupancy[mask],
                        prevalence_stratum=prevalence_stratum[mask],
                        design_stratum=design_stratum[mask],
                        support_threshold=support_threshold,
                        support_width=support_width,
                        community_threshold=community_threshold,
                        community_width=community_width,
                    )
                    > 0.5
                )
            ),
        },
        "coverage_floor": float(floor),
        "zero_metrics": zero_metrics,
        "best_metrics": best_metrics,
        "in_domain_zero_metrics": in_domain_zero_metrics,
        "in_domain_best_metrics": best_in_domain_metrics,
        "shrinkage_grid": shrinkage_grid,
    }
    return (
        tuple(float(value) for value in best_offsets),
        {
            "selected_shrinkage": float(best_shrinkage),
            "support_threshold": float(support_threshold),
            "support_width": float(support_width),
            "community_threshold": float(community_threshold),
            "community_width": float(community_width),
            "diagnostics": diagnostics,
        },
    )


def _rare_validation_scale_multiplier(
    *,
    log_offsets: Sequence[float],
    prevalence_stratum: np.ndarray,
    design_stratum: np.ndarray,
    support_excess: np.ndarray,
    community_occupancy: np.ndarray,
    support_threshold: float,
    support_width: float,
    community_threshold: float,
    community_width: float,
) -> np.ndarray:
    """Return the rare-validation design-stratum scale multiplier."""
    offsets = np.asarray(tuple(float(value) for value in log_offsets), dtype=float)
    if offsets.size != 3:
        raise ValueError("rare validation scale log offsets must have length three")
    activation = _rare_validation_scale_activation(
        support_excess,
        community_occupancy=community_occupancy,
        prevalence_stratum=prevalence_stratum,
        design_stratum=design_stratum,
        support_threshold=support_threshold,
        support_width=support_width,
        community_threshold=community_threshold,
        community_width=community_width,
    )
    return np.exp(offsets[np.clip(design_stratum, 0, 2)] * activation)


def _external_monotone_multiplier(
    *,
    support_excess: np.ndarray,
    effect_signal: np.ndarray,
    log_offsets: Sequence[float],
    effect_bin_edges: Sequence[float],
    support_threshold: float,
    support_width: float,
    effect_threshold: float,
    effect_width: float,
) -> np.ndarray:
    """Return the conservative external context-stratified scale multiplier."""
    support = np.asarray(support_excess, dtype=float)
    effect = np.asarray(effect_signal, dtype=float)
    offsets = np.asarray(tuple(float(value) for value in log_offsets), dtype=float)
    edges = np.asarray(tuple(float(value) for value in effect_bin_edges), dtype=float)
    if offsets.size != 3 or edges.size != 2:
        raise ValueError(
            "external monotone scale requires three offsets and two effect-bin edges"
        )
    offsets = np.maximum.accumulate(np.maximum(offsets, 0.0))
    if not np.any(offsets > 0.0):
        return np.ones_like(support, dtype=float)
    support_activation = np.clip(
        (support - float(support_threshold)) / max(float(support_width), 1e-8),
        0.0,
        1.0,
    )
    effect_activation = np.clip(
        (effect - float(effect_threshold)) / max(float(effect_width), 1e-8),
        0.0,
        1.0,
    )
    activation = np.maximum(support_activation, effect_activation)
    bin_index = np.digitize(effect, edges, right=False)
    log_multiplier = offsets[np.clip(bin_index, 0, 2)] * activation
    return np.exp(log_multiplier)


def _external_monotone_arrays(
    posterior: BetaPosterior,
    beta_true: np.ndarray,
    calibration: ConditionalBetaScaleCalibration,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    distribution: str,
    coefficient_names: Sequence[str],
    label: str = "in_distribution",
) -> dict[str, np.ndarray | str]:
    """Collect base calibrated errors and context features for external fitting."""
    base_calibration = replace(
        calibration,
        external_monotone_log_offsets=(0.0, 0.0, 0.0),
        external_monotone_selected_shrinkage=0.0,
    )
    calibrated = apply_conditional_beta_scale_calibration(
        posterior,
        base_calibration,
        X=X,
        Y=Y,
        distribution=distribution,
        coefficient_names=coefficient_names,
    )
    mean, scale, design, response = _validated_arrays(calibrated, X=X, Y=Y)
    truth = np.asarray(beta_true, dtype=float)
    if truth.shape != mean.shape:
        raise ValueError("beta_true must match posterior mean shape")
    raw_features = _raw_features(
        mean=mean,
        scale=scale,
        X=design,
        Y=response,
        distribution=distribution,
    )
    support_excess = _support_excess(
        raw_features,
        location=np.asarray(calibration.feature_location),
        scale=np.asarray(calibration.feature_scale),
        lower=np.asarray(calibration.support_lower),
        upper=np.asarray(calibration.support_upper),
        precision=np.asarray(calibration.support_precision),
        radius=calibration.support_radius,
        mean_magnitude=np.log1p(np.abs(mean)),
        mean_magnitude_location=calibration.mean_magnitude_location,
        mean_magnitude_scale=calibration.mean_magnitude_scale,
        mean_magnitude_lower=calibration.mean_magnitude_lower,
        mean_magnitude_upper=calibration.mean_magnitude_upper,
    )
    effect_signal = _effect_size_signal(
        np.log1p(np.abs(mean)),
        mean_magnitude_location=calibration.mean_magnitude_location,
        mean_magnitude_scale=calibration.mean_magnitude_scale,
    )
    prevalence = _prevalence(response)
    prevalence_stratum = _prevalence_stratum_index(
        np.broadcast_to(prevalence[:, None, :], mean.shape),
        prevalence_edges=calibration.prevalence_edges,
    )
    return {
        "label": str(label),
        "error": np.abs(truth - mean).reshape(-1),
        "scale": np.maximum(scale.reshape(-1), 1e-8),
        "support_excess": support_excess.reshape(-1),
        "effect_signal": effect_signal.reshape(-1),
        "prevalence_stratum": prevalence_stratum.reshape(-1),
    }


def _external_monotone_candidate_offsets(
    arrays_by_domain: Sequence[dict[str, np.ndarray | str]],
    *,
    effect_bin_edges: Sequence[float],
    support_threshold: float,
    support_width: float,
    effect_threshold: float,
    effect_width: float,
    z_value: float,
    max_external_multiplier: float,
) -> np.ndarray:
    """Estimate monotone effect-bin offsets needed to recover held-out coverage."""
    if not arrays_by_domain:
        return np.zeros(3, dtype=float)
    effect = np.concatenate(
        [np.asarray(arrays["effect_signal"], dtype=float) for arrays in arrays_by_domain]
    )
    support = np.concatenate(
        [
            np.asarray(arrays["support_excess"], dtype=float)
            for arrays in arrays_by_domain
        ]
    )
    error = np.concatenate(
        [np.asarray(arrays["error"], dtype=float) for arrays in arrays_by_domain]
    )
    scale = np.concatenate(
        [np.asarray(arrays["scale"], dtype=float) for arrays in arrays_by_domain]
    )
    activation = np.maximum(
        np.clip((support - support_threshold) / max(support_width, 1e-8), 0.0, 1.0),
        np.clip((effect - effect_threshold) / max(effect_width, 1e-8), 0.0, 1.0),
    )
    needed = np.maximum(error / (float(z_value) * np.maximum(scale, 1e-8)), 1.0)
    required_log = np.log(needed) / np.maximum(activation, 1e-8)
    bin_index = np.digitize(effect, np.asarray(effect_bin_edges, dtype=float), right=False)
    offsets = np.zeros(3, dtype=float)
    for index in range(3):
        mask = (bin_index == index) & (activation > 0.10) & np.isfinite(required_log)
        if np.any(mask):
            offsets[index] = float(np.quantile(required_log[mask], 0.95))
    return np.clip(
        np.maximum.accumulate(np.maximum(offsets, 0.0)),
        0.0,
        float(np.log(max_external_multiplier)),
    )


def _external_monotone_selection_metrics(
    log_offsets: np.ndarray,
    *,
    in_domain: dict[str, np.ndarray | str],
    ood: Sequence[dict[str, np.ndarray | str]],
    effect_bin_edges: Sequence[float],
    support_threshold: float,
    support_width: float,
    effect_threshold: float,
    effect_width: float,
    z_value: float,
    prevalence_edges: tuple[float, float],
) -> dict[str, Any]:
    """Summarize in-domain, rare, and held-out OOD coverage after offsets."""

    def coverage(arrays: dict[str, np.ndarray | str]) -> float:
        multiplier = _external_monotone_multiplier(
            support_excess=np.asarray(arrays["support_excess"], dtype=float),
            effect_signal=np.asarray(arrays["effect_signal"], dtype=float),
            log_offsets=log_offsets,
            effect_bin_edges=effect_bin_edges,
            support_threshold=support_threshold,
            support_width=support_width,
            effect_threshold=effect_threshold,
            effect_width=effect_width,
        )
        covered = np.asarray(arrays["error"], dtype=float) <= (
            float(z_value) * np.asarray(arrays["scale"], dtype=float) * multiplier
        )
        return float(np.mean(covered)) if covered.size else 0.0

    def rare_coverage(arrays: dict[str, np.ndarray | str]) -> float:
        multiplier = _external_monotone_multiplier(
            support_excess=np.asarray(arrays["support_excess"], dtype=float),
            effect_signal=np.asarray(arrays["effect_signal"], dtype=float),
            log_offsets=log_offsets,
            effect_bin_edges=effect_bin_edges,
            support_threshold=support_threshold,
            support_width=support_width,
            effect_threshold=effect_threshold,
            effect_width=effect_width,
        )
        rare = np.asarray(arrays["prevalence_stratum"], dtype=int) == 0
        if not np.any(rare):
            return coverage(arrays)
        covered = np.asarray(arrays["error"], dtype=float) <= (
            float(z_value) * np.asarray(arrays["scale"], dtype=float) * multiplier
        )
        return float(np.mean(covered[rare]))

    in_domain_coverage = coverage(in_domain)
    rare_prevalence_coverage = rare_coverage(in_domain)
    domain_coverages: dict[str, float] = {}
    for arrays in ood:
        label = str(arrays["label"])
        domain_coverages.setdefault(label, [])
        domain_coverages[label].append(coverage(arrays))  # type: ignore[union-attr]
    averaged_domain_coverages = {
        label: float(np.mean(values)) for label, values in domain_coverages.items()
    }
    if averaged_domain_coverages:
        mean_ood = float(np.mean(list(averaged_domain_coverages.values())))
        worst_ood = float(np.min(list(averaged_domain_coverages.values())))
    else:
        mean_ood = 0.0
        worst_ood = 0.0
    combined = float(averaged_domain_coverages.get("combined_shift", worst_ood))
    effect = float(averaged_domain_coverages.get("effect_size_shift", worst_ood))
    return {
        "in_domain_coverage": float(in_domain_coverage),
        "rare_prevalence_coverage": float(rare_prevalence_coverage),
        "mean_ood_coverage": mean_ood,
        "worst_ood_coverage": worst_ood,
        "combined_shift_coverage": combined,
        "effect_size_shift_coverage": effect,
        "domain_coverages": averaged_domain_coverages,
        "prevalence_edges": [float(value) for value in prevalence_edges],
    }


def _combined_shift_scale_multiplier(
    *,
    support_excess: np.ndarray,
    effect_signal: np.ndarray,
    design_signal: np.ndarray | None = None,
    community_occupancy: np.ndarray | None = None,
    log_amplitude: float,
    effect_bin_log_amplitudes: Sequence[float] = (),
    effect_bin_edges: Sequence[float] = (),
    context_gate_strength: float = 0.0,
    context_gate_intercept: float = 0.0,
) -> np.ndarray:
    """Return a domain-specific combined-shift scale multiplier."""
    support = np.asarray(support_excess, dtype=float)
    effect = np.asarray(effect_signal, dtype=float)
    amplitude = np.full_like(
        support,
        float(np.clip(log_amplitude, 0.0, _COMBINED_SHIFT_SCALE_MAX_LOG)),
        dtype=float,
    )
    bin_amplitudes = np.asarray(
        tuple(float(value) for value in effect_bin_log_amplitudes),
        dtype=float,
    )
    bin_edges = np.asarray(
        tuple(float(value) for value in effect_bin_edges), dtype=float
    )
    if bin_amplitudes.size:
        if bin_amplitudes.size != 3 or bin_edges.size != 2:
            raise ValueError(
                "combined-shift effect-bin scale requires three amplitudes and two edges"
            )
        bin_index = np.digitize(effect, bin_edges, right=False)
        amplitude = amplitude + bin_amplitudes[np.clip(bin_index, 0, 2)]
    amplitude = np.clip(amplitude, 0.0, _COMBINED_SHIFT_SCALE_MAX_LOG)
    if not np.any(amplitude > 0.0):
        return np.ones_like(support, dtype=float)
    design = (
        np.zeros_like(support, dtype=float)
        if design_signal is None
        else np.asarray(design_signal, dtype=float)
    )
    community = (
        np.full_like(support, _COMBINED_SHIFT_SCALE_LOW_COMMUNITY_CENTER, dtype=float)
        if community_occupancy is None
        else np.asarray(community_occupancy, dtype=float)
    )
    support_gate = _sigmoid_numpy(
        (support - _COMBINED_SHIFT_SCALE_SUPPORT_CENTER)
        / _COMBINED_SHIFT_SCALE_SUPPORT_WIDTH
    )
    effect_gate = _sigmoid_numpy(
        (effect - _COMBINED_SHIFT_SCALE_EFFECT_CENTER)
        / _COMBINED_SHIFT_SCALE_EFFECT_WIDTH
    )
    low_design_gate = _sigmoid_numpy(
        (_COMBINED_SHIFT_SCALE_LOW_DESIGN_CENTER - design)
        / _COMBINED_SHIFT_SCALE_LOW_DESIGN_WIDTH
    )
    low_community_gate = _sigmoid_numpy(
        (_COMBINED_SHIFT_SCALE_LOW_COMMUNITY_CENTER - community)
        / _COMBINED_SHIFT_SCALE_LOW_COMMUNITY_WIDTH
    )
    context_gate = _combined_shift_context_gate(
        support_excess=support,
        effect_signal=effect,
        design_signal=design,
        community_occupancy=community,
        strength=context_gate_strength,
        intercept=context_gate_intercept,
    )
    return np.exp(
        amplitude
        * support_gate
        * effect_gate
        * low_design_gate
        * low_community_gate
        * context_gate
    )


def _combined_shift_context_gate(
    *,
    support_excess: np.ndarray,
    effect_signal: np.ndarray,
    design_signal: np.ndarray,
    community_occupancy: np.ndarray,
    strength: float,
    intercept: float,
) -> np.ndarray:
    """Return a classifier-style gate for combined-shift scale activation."""
    support = np.asarray(support_excess, dtype=float)
    if float(strength) <= 0.0:
        return np.ones_like(support, dtype=float)
    effect = np.asarray(effect_signal, dtype=float)
    design = np.asarray(design_signal, dtype=float)
    community = np.asarray(community_occupancy, dtype=float)
    support_z = (support - _COMBINED_SHIFT_SCALE_SUPPORT_CENTER) / max(
        _COMBINED_SHIFT_SCALE_SUPPORT_WIDTH, 1e-6
    )
    effect_z = (effect - _COMBINED_SHIFT_SCALE_EFFECT_CENTER) / max(
        _COMBINED_SHIFT_SCALE_EFFECT_WIDTH, 1e-6
    )
    low_design_z = (_COMBINED_SHIFT_SCALE_LOW_DESIGN_CENTER - design) / max(
        _COMBINED_SHIFT_SCALE_LOW_DESIGN_WIDTH, 1e-6
    )
    low_community_z = (_COMBINED_SHIFT_SCALE_LOW_COMMUNITY_CENTER - community) / max(
        _COMBINED_SHIFT_SCALE_LOW_COMMUNITY_WIDTH, 1e-6
    )
    interaction = np.minimum(np.maximum(support_z, 0.0), 6.0) * np.minimum(
        np.maximum(low_community_z, 0.0), 6.0
    )
    score = (
        _COMBINED_SHIFT_CONTEXT_GATE_SUPPORT_WEIGHT * support_z
        + _COMBINED_SHIFT_CONTEXT_GATE_EFFECT_WEIGHT * effect_z
        + _COMBINED_SHIFT_CONTEXT_GATE_LOW_DESIGN_WEIGHT * low_design_z
        + _COMBINED_SHIFT_CONTEXT_GATE_LOW_COMMUNITY_WEIGHT * low_community_z
        + _COMBINED_SHIFT_CONTEXT_GATE_INTERACTION_WEIGHT * interaction
        - float(intercept)
    ) / max(_COMBINED_SHIFT_CONTEXT_GATE_WIDTH, 1e-6)
    gate = _sigmoid_numpy(score)
    strength_value = float(np.clip(strength, 0.0, 1.0))
    return (1.0 - strength_value) + strength_value * gate


def _combined_shift_context_gate_summary(
    *,
    support_excess: np.ndarray,
    effect_signal: np.ndarray,
    design_signal: np.ndarray,
    community_occupancy: np.ndarray,
    strength: float,
    intercept: float,
) -> dict[str, float]:
    gate = _combined_shift_context_gate(
        support_excess=support_excess,
        effect_signal=effect_signal,
        design_signal=design_signal,
        community_occupancy=community_occupancy,
        strength=strength,
        intercept=intercept,
    )
    return {
        "strength": float(strength),
        "intercept": float(intercept),
        "mean": float(np.mean(gate)),
        "max": float(np.max(gate)),
        "active_fraction_over_0_5": float(np.mean(gate > 0.5)),
        "active_fraction_over_0_8": float(np.mean(gate > 0.8)),
    }


def _rare_validation_scale_activation(
    support_excess: np.ndarray,
    *,
    community_occupancy: np.ndarray | None = None,
    prevalence_stratum: np.ndarray | None = None,
    design_stratum: np.ndarray | None = None,
    support_threshold: float,
    support_width: float,
    community_threshold: float = 0.0,
    community_width: float = 1.0,
) -> np.ndarray:
    """Return smooth activation for rare-validation scale contexts."""
    width = max(float(support_width), 1e-6)
    support_sigmoid = 1.0 / (
        1.0
        + np.exp(
            -((np.asarray(support_excess, dtype=float) - support_threshold) / width)
        )
    )
    support_activation = np.clip(2.0 * (support_sigmoid - 0.5), 0.0, 1.0)
    if prevalence_stratum is None or design_stratum is None:
        return support_activation
    prevalence_weight = np.asarray(
        (1.0, 0.65, 0.0),
        dtype=float,
    )[np.clip(np.asarray(prevalence_stratum, dtype=np.int32), 0, 2)]
    design_weight = np.asarray(
        (0.35, 0.75, 1.0),
        dtype=float,
    )[np.clip(np.asarray(design_stratum, dtype=np.int32), 0, 2)]
    regime_proxy = prevalence_weight * design_weight
    if community_occupancy is None:
        community_proxy = 0.0
    else:
        community_scale = max(float(community_width), 1e-6)
        community_sigmoid = 1.0 / (
            1.0
            + np.exp(
                (
                    np.asarray(community_occupancy, dtype=float)
                    - float(community_threshold)
                )
                / community_scale
            )
        )
        community_proxy = np.clip(2.0 * (community_sigmoid - 0.5), 0.0, 1.0)
    low_community_regime_proxy = np.maximum(
        community_proxy, community_proxy * regime_proxy
    )
    return np.maximum.reduce(
        [
            support_activation,
            low_community_regime_proxy,
        ]
    )


def _rare_validation_scale_metrics(
    *,
    signed_error: np.ndarray,
    base_multiplier: np.ndarray,
    mask: np.ndarray,
    prevalence_stratum: np.ndarray,
    design_stratum: np.ndarray,
    support_excess: np.ndarray,
    community_occupancy: np.ndarray,
    labels: np.ndarray,
    log_offsets: np.ndarray,
    support_threshold: float,
    support_width: float,
    community_threshold: float,
    community_width: float,
    z_value: float,
    min_multiplier: float,
    max_multiplier: float,
) -> dict[str, Any]:
    extra = _rare_validation_scale_multiplier(
        log_offsets=log_offsets,
        prevalence_stratum=prevalence_stratum,
        design_stratum=design_stratum,
        support_excess=support_excess,
        community_occupancy=community_occupancy,
        support_threshold=support_threshold,
        support_width=support_width,
        community_threshold=community_threshold,
        community_width=community_width,
    )
    multiplier = np.clip(base_multiplier * extra, min_multiplier, max_multiplier)
    selected = mask
    rare = selected & (prevalence_stratum == 0)
    intermediate = selected & (design_stratum == 1)
    high = selected & (design_stratum == 2)
    overall_coverage = _coverage_with_multiplier(
        signed_error=signed_error,
        multiplier=multiplier,
        mask=selected,
        z_value=z_value,
    )
    rare_coverage = _coverage_with_multiplier(
        signed_error=signed_error,
        multiplier=multiplier,
        mask=rare,
        z_value=z_value,
    )
    intermediate_coverage = _coverage_with_multiplier(
        signed_error=signed_error,
        multiplier=multiplier,
        mask=intermediate,
        z_value=z_value,
    )
    high_coverage = _coverage_with_multiplier(
        signed_error=signed_error,
        multiplier=multiplier,
        mask=high,
        z_value=z_value,
    )
    rank_probability = ndtr(signed_error / np.maximum(multiplier, 1e-8))
    overall_rank_error = abs(float(np.mean(rank_probability[selected])) - 0.5)
    overall_rank_variance = float(np.var(rank_probability[selected]))
    rare_rank_error = (
        abs(float(np.mean(rank_probability[rare])) - 0.5)
        if np.count_nonzero(rare) >= 8
        else overall_rank_error
    )
    rare_rank_variance = (
        float(np.var(rank_probability[rare]))
        if np.count_nonzero(rare) >= 8
        else overall_rank_variance
    )
    label_metrics: dict[str, dict[str, float]] = {}
    for label in sorted(str(value) for value in np.unique(labels[selected])):
        label_mask = selected & (labels == label)
        label_metrics[label] = {
            "n_observations": int(np.count_nonzero(label_mask)),
            "coverage": _coverage_with_multiplier(
                signed_error=signed_error,
                multiplier=multiplier,
                mask=label_mask,
                z_value=z_value,
            ),
            "rank_error": (
                abs(float(np.mean(rank_probability[label_mask])) - 0.5)
                if np.count_nonzero(label_mask)
                else 0.0
            ),
        }
    floor = 0.90
    shortfall = (
        max(0.0, floor - overall_coverage)
        + max(0.0, floor - rare_coverage)
        + max(0.0, floor - intermediate_coverage)
        + max(0.0, floor - high_coverage)
    )
    objective = rare_rank_error + 0.25 * overall_rank_error + 4.0 * shortfall
    objective += 0.01 * float(np.mean(np.square(log_offsets)))
    return {
        "objective": float(objective),
        "overall_coverage": float(overall_coverage),
        "rare_coverage": float(rare_coverage),
        "intermediate_design_coverage": float(intermediate_coverage),
        "high_design_coverage": float(high_coverage),
        "overall_rank_error": float(overall_rank_error),
        "rare_rank_error": float(rare_rank_error),
        "overall_rank_variance": float(overall_rank_variance),
        "rare_rank_variance": float(rare_rank_variance),
        "by_regime": label_metrics,
    }


def _rare_validation_scale_gate_ok(
    *,
    current: dict[str, Any],
    baseline: dict[str, Any],
    in_domain_current: dict[str, Any],
    in_domain_baseline: dict[str, Any],
    floor: float,
) -> bool:
    variance_floor = 0.045
    return (
        current["overall_coverage"] >= floor
        and current["rare_coverage"] >= floor
        and current["intermediate_design_coverage"] >= floor
        and current["high_design_coverage"] >= floor
        and current["overall_coverage"] >= baseline["overall_coverage"] - 1e-8
        and current["rare_coverage"] >= baseline["rare_coverage"] - 1e-8
        and current["intermediate_design_coverage"]
        >= baseline["intermediate_design_coverage"] - 1e-8
        and current["high_design_coverage"] >= baseline["high_design_coverage"] - 1e-8
        and current["rare_rank_error"] <= baseline["rare_rank_error"] + 0.01
        and current["overall_rank_error"] <= baseline["overall_rank_error"] + 0.02
        and in_domain_current["overall_coverage"]
        <= max(0.975, in_domain_baseline["overall_coverage"] + 0.015)
        and in_domain_current["high_design_coverage"]
        <= max(0.985, in_domain_baseline["high_design_coverage"] + 0.02)
        and in_domain_current["overall_rank_variance"]
        >= min(in_domain_baseline["overall_rank_variance"] - 0.005, variance_floor)
        and in_domain_current["rare_rank_error"]
        <= in_domain_baseline["rare_rank_error"] + 0.01
    )


def _coverage_with_multiplier(
    *,
    signed_error: np.ndarray,
    multiplier: np.ndarray,
    mask: np.ndarray,
    z_value: float,
) -> float:
    if np.count_nonzero(mask) == 0:
        return 1.0
    return float(np.mean(np.abs(signed_error[mask]) <= z_value * multiplier[mask]))


def _solve_rank_center_shift(standardized_error: np.ndarray) -> float:
    """Solve c so mean(Phi(z - c)) is close to 0.5."""
    if standardized_error.size < 2:
        return 0.0
    lower, upper = -0.5, 0.5
    for _ in range(48):
        midpoint = 0.5 * (lower + upper)
        rank_mean = float(np.mean(ndtr(standardized_error - midpoint)))
        if rank_mean > 0.5:
            lower = midpoint
        else:
            upper = midpoint
    return float(np.clip(0.5 * (lower + upper), -0.35, 0.35))


def _fit_rank_centering_offsets(
    *,
    truth: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    prevalence_stratum: np.ndarray,
    coefficient_stratum: np.ndarray,
    mask: np.ndarray,
    nominal_level: float,
    z_value: float,
) -> tuple[tuple[tuple[float, ...], ...], float]:
    """Fit held-out-validated standardized rank-centering offsets."""
    if mean.shape[0] < 4:
        return (
            tuple(tuple(0.0 for _ in range(mean.shape[1])) for _ in range(3)),
            0.0,
        )
    batch_index = np.broadcast_to(
        np.arange(mean.shape[0])[:, None, None],
        mean.shape,
    )
    validation_mask = mask & ((batch_index % 4) == 0)
    training_mask = mask & ~validation_mask
    if np.count_nonzero(validation_mask) < 32 or np.count_nonzero(training_mask) < 32:
        return (
            tuple(tuple(0.0 for _ in range(mean.shape[1])) for _ in range(3)),
            0.0,
        )

    signed_error = (truth - mean) / scale
    candidate = np.zeros((3, mean.shape[1]), dtype=float)
    for prevalence_index in range(3):
        for coefficient_index in range(mean.shape[1]):
            group = (
                training_mask
                & (prevalence_stratum == prevalence_index)
                & (coefficient_stratum == coefficient_index)
            )
            if np.count_nonzero(group) < 8:
                continue
            candidate[prevalence_index, coefficient_index] = _solve_rank_center_shift(
                signed_error[group]
            )

    def score(shrinkage: float) -> tuple[float, float, float]:
        offset = shrinkage * candidate[prevalence_stratum, coefficient_stratum]
        adjusted = signed_error - offset
        validation = adjusted[validation_mask]
        validation_coverage = float(np.mean(np.abs(validation) <= z_value))
        overall_rank_error = abs(float(np.mean(ndtr(validation))) - 0.5)
        rare_mask = validation_mask & (prevalence_stratum == 0)
        rare_rank_error = overall_rank_error
        if np.count_nonzero(rare_mask) >= 8:
            rare_rank_error = abs(float(np.mean(ndtr(adjusted[rare_mask]))) - 0.5)
        objective = rare_rank_error + 0.5 * overall_rank_error
        objective += max(0.0, nominal_level - validation_coverage) * 4.0
        return objective, rare_rank_error, validation_coverage

    zero_score = score(0.0)
    best_shrinkage = 0.0
    best_score = zero_score
    for shrinkage in (0.125, 0.25, 0.375, 0.5, 0.75):
        current = score(shrinkage)
        if (
            current[0] + 1e-8 < best_score[0]
            and current[1] <= zero_score[1] + 1e-8
            and current[2] >= max(0.90, zero_score[2] - 0.01)
        ):
            best_shrinkage = shrinkage
            best_score = current
    offsets = best_shrinkage * candidate
    if best_shrinkage == 0.0:
        offsets.fill(0.0)
    return (
        tuple(tuple(float(value) for value in row) for row in offsets),
        float(best_shrinkage),
    )


def _mean_bias_correction_array(
    *,
    calibration: ConditionalBetaScaleCalibration,
    Y: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Return the serialized prevalence/coefficient mean correction array."""
    if not calibration.mean_bias_correction:
        return np.zeros(shape, dtype=float)
    values = np.asarray(calibration.mean_bias_correction, dtype=float)
    if values.shape != (3, shape[1]):
        raise ValueError("mean bias correction does not match posterior shape")
    prevalence = _prevalence(Y)
    prevalence_stratum = _prevalence_stratum_index(
        np.broadcast_to(prevalence[:, None, :], shape),
        prevalence_edges=calibration.prevalence_edges,
    )
    coefficient_stratum = _coefficient_stratum_index(shape)
    return values[prevalence_stratum, coefficient_stratum]


def _rank_centering_correction_array(
    *,
    calibration: ConditionalBetaScaleCalibration,
    scale: np.ndarray,
    Y: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Return prevalence/coefficient rank-centering mean shifts."""
    if not calibration.rank_centering_offsets:
        return np.zeros(shape, dtype=float)
    values = np.asarray(calibration.rank_centering_offsets, dtype=float)
    if values.shape != (3, shape[1]):
        raise ValueError("rank centering offsets do not match posterior shape")
    prevalence = _prevalence(Y)
    prevalence_stratum = _prevalence_stratum_index(
        np.broadcast_to(prevalence[:, None, :], shape),
        prevalence_edges=calibration.prevalence_edges,
    )
    coefficient_stratum = _coefficient_stratum_index(shape)
    return values[prevalence_stratum, coefficient_stratum] * scale


def _base_scale_stratum_log_offset(
    *,
    offsets: Sequence[float],
    prevalence_stratum: np.ndarray,
    design_stratum: np.ndarray,
    coefficient_stratum: np.ndarray,
    n_covariates: int,
) -> np.ndarray:
    """Return additive log-scale offsets by prevalence/design/coefficient strata."""
    if not offsets:
        return np.zeros(prevalence_stratum.shape, dtype=float)
    values = np.asarray(tuple(float(value) for value in offsets), dtype=float)
    expected = 6 + int(n_covariates)
    if values.size != expected:
        raise ValueError("base scale stratum offsets do not match covariate count")
    prevalence_offsets = values[:3]
    design_offsets = values[3:6]
    coefficient_offsets = values[6:]
    return (
        prevalence_offsets[np.clip(prevalence_stratum, 0, 2)]
        + design_offsets[np.clip(design_stratum, 0, 2)]
        + coefficient_offsets[
            np.clip(coefficient_stratum, 0, coefficient_offsets.size - 1)
        ]
    )


def _blend_with_scalar_fallback(
    adjustment: np.ndarray,
    trust: np.ndarray,
    *,
    support_excess: np.ndarray,
    effect_signal: np.ndarray | None = None,
    design_signal: np.ndarray | None = None,
    community_occupancy: np.ndarray | None = None,
    prevalence_stratum: np.ndarray | None = None,
    design_stratum: np.ndarray | None = None,
    coefficient_stratum: np.ndarray | None = None,
    normalization: float,
    global_multiplier: float,
    ood_uncertainty_strength: float,
    ood_uncertainty_max_multiplier: float,
    min_multiplier: float,
    max_multiplier: float,
    ood_inflation_parameters: tuple[float, ...] | None = None,
) -> np.ndarray:
    conditional = np.clip(
        float(normalization) * adjustment,
        min_multiplier,
        max_multiplier,
    )
    log_multiplier = trust * np.log(conditional) + (1.0 - trust) * np.log(
        float(global_multiplier)
    )
    multiplier = np.exp(log_multiplier) * _ood_uncertainty_inflation(
        support_excess,
        effect_signal=effect_signal,
        design_signal=design_signal,
        community_occupancy=community_occupancy,
        prevalence_stratum=prevalence_stratum,
        design_stratum=design_stratum,
        coefficient_stratum=coefficient_stratum,
        strength=ood_uncertainty_strength,
        max_multiplier=ood_uncertainty_max_multiplier,
        learned_parameters=ood_inflation_parameters,
    )
    return np.clip(multiplier, min_multiplier, max_multiplier)


def _ood_uncertainty_inflation(
    support_excess: np.ndarray,
    *,
    effect_signal: np.ndarray | None = None,
    design_signal: np.ndarray | None = None,
    community_occupancy: np.ndarray | None = None,
    prevalence_stratum: np.ndarray | None = None,
    design_stratum: np.ndarray | None = None,
    coefficient_stratum: np.ndarray | None = None,
    strength: float,
    max_multiplier: float,
    learned_parameters: tuple[float, ...] | None = None,
) -> np.ndarray:
    if max_multiplier <= 1.0:
        return np.ones_like(support_excess, dtype=float)
    if learned_parameters is not None:
        parameters = tuple(float(value) for value in learned_parameters)
        effect = (
            np.zeros_like(support_excess, dtype=float)
            if effect_signal is None
            else np.asarray(effect_signal, dtype=float)
        )
        design = (
            np.zeros_like(support_excess, dtype=float)
            if design_signal is None
            else np.asarray(design_signal, dtype=float)
        )
        community = (
            None
            if community_occupancy is None
            else np.asarray(community_occupancy, dtype=float)
        )
        log_inflation = _learned_ood_log_inflation_numpy(
            support_excess,
            effect_signal=effect,
            design_signal=design,
            community_occupancy=community,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            coefficient_stratum=coefficient_stratum,
            parameters=parameters,
            max_multiplier=max_multiplier,
        )
        return np.exp(log_inflation)
    if strength <= 0.0:
        return np.ones_like(support_excess, dtype=float)
    log_inflation = np.minimum(
        float(strength) * np.square(support_excess),
        np.log(float(max_multiplier)),
    )
    return np.exp(log_inflation)


def _fit_coverage_normalization(
    *,
    standardized_error: np.ndarray,
    adjustment: np.ndarray,
    trust: np.ndarray,
    support_excess: np.ndarray,
    effect_signal: np.ndarray | None = None,
    design_signal: np.ndarray | None = None,
    community_occupancy: np.ndarray | None = None,
    prevalence_stratum: np.ndarray | None = None,
    design_stratum: np.ndarray | None = None,
    coefficient_stratum: np.ndarray | None = None,
    global_multiplier: float,
    mask: np.ndarray,
    nominal_level: float,
    z_value: float,
    ood_uncertainty_strength: float,
    ood_uncertainty_max_multiplier: float,
    min_multiplier: float,
    max_multiplier: float,
    ood_inflation_parameters: tuple[float, ...] | None = None,
) -> float:
    lower = float(min_multiplier)
    upper = float(max_multiplier)
    for _ in range(64):
        midpoint = float(np.sqrt(lower * upper))
        multiplier = _blend_with_scalar_fallback(
            adjustment,
            trust,
            support_excess=support_excess,
            effect_signal=effect_signal,
            design_signal=design_signal,
            community_occupancy=community_occupancy,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            coefficient_stratum=coefficient_stratum,
            normalization=midpoint,
            global_multiplier=global_multiplier,
            ood_uncertainty_strength=ood_uncertainty_strength,
            ood_uncertainty_max_multiplier=ood_uncertainty_max_multiplier,
            ood_inflation_parameters=ood_inflation_parameters,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
        coverage = float(
            np.mean(standardized_error[mask] <= z_value * multiplier[mask])
        )
        if coverage < nominal_level:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def _ood_final_multiplier_diagnostics(
    *,
    batches: Sequence[ConditionalBetaOODCalibrationBatch],
    location: np.ndarray,
    feature_scale: np.ndarray,
    feature_names: tuple[str, ...],
    support_lower: np.ndarray,
    support_upper: np.ndarray,
    support_precision: np.ndarray,
    support_radius: float,
    mean_magnitude_location: float,
    mean_magnitude_scale: float,
    mean_magnitude_lower: float,
    mean_magnitude_upper: float,
    normalization: float,
    global_multiplier: float,
    fallback_strength: float,
    fitted_weights: np.ndarray,
    mean_bias_correction: tuple[tuple[float, ...], ...],
    rank_centering_offsets: tuple[tuple[float, ...], ...],
    base_scale_stratum_offsets: tuple[float, ...],
    ood_inflation_parameters: tuple[float, ...],
    post_scale_log_offsets: tuple[float, float, float],
    post_scale_support_threshold: float,
    post_scale_support_width: float,
    post_scale_community_threshold: float,
    post_scale_community_width: float,
    combined_shift_scale_log_amplitude: float,
    distribution: str,
    n_covariates: int,
    n_species: int,
    prevalence_edges: tuple[float, float],
    in_domain_signed_error: np.ndarray,
    in_domain_final_multiplier: np.ndarray,
    in_domain_log_ood_inflation: np.ndarray,
    in_domain_rank_groups: list[np.ndarray],
    rank_mean_tolerance: float,
    rank_variance_tolerance: float,
    nominal_level: float,
    z_value: float,
    min_multiplier: float,
    max_multiplier: float,
    max_ood_multiplier: float,
    combined_shift_scale_effect_bin_log_amplitudes: Sequence[float] = (),
    combined_shift_scale_effect_bin_edges: Sequence[float] = (),
    combined_shift_scale_context_gate_strength: float = 0.0,
    combined_shift_scale_context_gate_intercept: float = 0.0,
) -> dict[str, Any]:
    """Summarize final OOD multiplier behavior without changing calibration."""
    domains: list[dict[str, Any]] = []
    for batch in batches:
        mean, scale, design, response = _validated_arrays(
            batch.posterior, X=batch.X, Y=batch.Y
        )
        if mean.shape[1] != n_covariates or mean.shape[2] != n_species:
            raise ValueError("OOD diagnostics batch domain does not match calibration")
        truth = np.asarray(batch.beta_true, dtype=float)
        if truth.shape != mean.shape or not np.all(np.isfinite(truth)):
            raise ValueError("OOD diagnostics beta_true must match posterior shape")
        prevalence = _prevalence(response)
        prevalence_by_coefficient = np.broadcast_to(prevalence[:, None, :], mean.shape)
        prevalence_stratum = _prevalence_stratum_index(
            prevalence_by_coefficient,
            prevalence_edges=prevalence_edges,
        )
        coefficient_stratum = _coefficient_stratum_index(mean.shape)
        if mean_bias_correction:
            bias_values = np.asarray(mean_bias_correction, dtype=float)
            mean = mean + bias_values[prevalence_stratum, coefficient_stratum]
        if rank_centering_offsets:
            centering_values = np.asarray(rank_centering_offsets, dtype=float)
            mean = (
                mean + centering_values[prevalence_stratum, coefficient_stratum] * scale
            )
        raw_features = _raw_features(
            mean=mean,
            scale=scale,
            X=design,
            Y=response,
            distribution=distribution,
        )
        design_matrix, names = _structured_design(
            raw_features,
            location=location,
            scale=feature_scale,
            n_covariates=n_covariates,
        )
        if names != feature_names:
            raise ValueError("OOD diagnostics feature specification mismatch")
        adjustment = np.exp(
            np.clip(design_matrix @ fitted_weights, -20.0, 20.0)
        ).reshape(mean.shape)
        trust = _support_trust(
            raw_features,
            location=location,
            scale=feature_scale,
            lower=support_lower,
            upper=support_upper,
            precision=support_precision,
            radius=support_radius,
            fallback_strength=fallback_strength,
            mean_magnitude=np.log1p(np.abs(mean)),
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
        )
        support_excess = _support_excess(
            raw_features,
            location=location,
            scale=feature_scale,
            lower=support_lower,
            upper=support_upper,
            precision=support_precision,
            radius=support_radius,
            mean_magnitude=np.log1p(np.abs(mean)),
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
        )
        effect_signal = _effect_size_signal(
            np.log1p(np.abs(mean)),
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
        )
        design_signal = _design_information_signal(
            raw_features,
            location=location,
            scale=feature_scale,
        )
        design_stratum = _design_information_stratum_index(design_signal)
        adjustment = adjustment * np.exp(
            _base_scale_stratum_log_offset(
                offsets=base_scale_stratum_offsets,
                prevalence_stratum=prevalence_stratum,
                design_stratum=design_stratum,
                coefficient_stratum=coefficient_stratum,
                n_covariates=n_covariates,
            )
        )
        base_multiplier = _blend_with_scalar_fallback(
            adjustment,
            trust,
            support_excess=support_excess,
            effect_signal=effect_signal,
            design_signal=design_signal,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            coefficient_stratum=coefficient_stratum,
            normalization=normalization,
            global_multiplier=global_multiplier,
            ood_uncertainty_strength=0.0,
            ood_uncertainty_max_multiplier=1.0,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
        log_ood_inflation = _learned_ood_log_inflation_numpy(
            support_excess,
            effect_signal=effect_signal,
            design_signal=design_signal,
            community_occupancy=_community_occupancy_array(response, mean.shape),
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            coefficient_stratum=coefficient_stratum,
            parameters=ood_inflation_parameters,
            max_multiplier=max_ood_multiplier,
        )
        ood_inflation = np.exp(log_ood_inflation)
        effect_gate = _learned_ood_effect_gate_numpy(
            support_excess,
            effect_signal=effect_signal,
            design_signal=design_signal,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            coefficient_stratum=coefficient_stratum,
            parameters=ood_inflation_parameters,
        )
        post_scale_multiplier = _rare_validation_scale_multiplier(
            log_offsets=post_scale_log_offsets,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            support_excess=support_excess,
            community_occupancy=_community_occupancy_array(response, mean.shape),
            support_threshold=post_scale_support_threshold,
            support_width=post_scale_support_width,
            community_threshold=post_scale_community_threshold,
            community_width=post_scale_community_width,
        )
        final_multiplier = np.clip(
            base_multiplier
            * ood_inflation
            * post_scale_multiplier
            * _combined_shift_scale_multiplier(
                support_excess=support_excess,
                effect_signal=effect_signal,
                design_signal=design_signal,
                community_occupancy=_community_occupancy_array(response, mean.shape),
                log_amplitude=combined_shift_scale_log_amplitude,
                effect_bin_log_amplitudes=(
                    combined_shift_scale_effect_bin_log_amplitudes
                ),
                effect_bin_edges=combined_shift_scale_effect_bin_edges,
                context_gate_strength=combined_shift_scale_context_gate_strength,
                context_gate_intercept=combined_shift_scale_context_gate_intercept,
            ),
            min_multiplier,
            max_multiplier,
        )
        signed_error = (truth - mean) / scale
        covered = np.abs(signed_error) <= z_value * final_multiplier
        label = str(batch.label)
        coverage_floor = (
            0.90
            if ("effect_size_shift" in label or "combined_shift" in label)
            else float(nominal_level)
        )
        domains.append(
            {
                "label": label,
                "n_observations": int(signed_error.size),
                "coverage": float(np.mean(covered)),
                "coverage_floor": float(coverage_floor),
                "coverage_floor_shortfall": float(
                    max(0.0, coverage_floor - float(np.mean(covered)))
                ),
                "rank_mean": float(np.mean(ndtr(signed_error / final_multiplier))),
                "rank_variance": float(np.var(ndtr(signed_error / final_multiplier))),
                "effect_gate_activation": _summary_quantiles(effect_gate),
                "learned_ood_inflation": _summary_quantiles(ood_inflation),
                "rare_post_scale_multiplier": _summary_quantiles(post_scale_multiplier),
                "combined_shift_scale_multiplier": _summary_quantiles(
                    _combined_shift_scale_multiplier(
                        support_excess=support_excess,
                        effect_signal=effect_signal,
                        design_signal=design_signal,
                        community_occupancy=_community_occupancy_array(
                            response, mean.shape
                        ),
                        log_amplitude=combined_shift_scale_log_amplitude,
                        effect_bin_log_amplitudes=(
                            combined_shift_scale_effect_bin_log_amplitudes
                        ),
                        effect_bin_edges=combined_shift_scale_effect_bin_edges,
                        context_gate_strength=(
                            combined_shift_scale_context_gate_strength
                        ),
                        context_gate_intercept=(
                            combined_shift_scale_context_gate_intercept
                        ),
                    )
                ),
                "combined_shift_context_gate": _combined_shift_context_gate_summary(
                    support_excess=support_excess,
                    effect_signal=effect_signal,
                    design_signal=design_signal,
                    community_occupancy=_community_occupancy_array(
                        response, mean.shape
                    ),
                    strength=combined_shift_scale_context_gate_strength,
                    intercept=combined_shift_scale_context_gate_intercept,
                ),
                "learned_combined_shift_context": _summary_quantiles(
                    _learned_combined_shift_context_numpy(
                        support_excess,
                        effect_signal=effect_signal,
                        design_signal=design_signal,
                        community_occupancy=_community_occupancy_array(
                            response,
                            mean.shape,
                        ),
                        parameters=ood_inflation_parameters,
                        n_covariates=n_covariates,
                    )
                ),
                "final_multiplier": _summary_quantiles(final_multiplier),
                "effect_size_quantile_coverage": _effect_quantile_coverage(
                    effect_signal=effect_signal.reshape(-1),
                    covered=covered.reshape(-1),
                    final_multiplier=final_multiplier.reshape(-1),
                ),
                "in_domain_gate": _in_domain_gate_diagnostics(
                    signed_error=in_domain_signed_error,
                    final_multiplier=in_domain_final_multiplier,
                    log_ood_inflation=in_domain_log_ood_inflation,
                    rank_groups=in_domain_rank_groups,
                    rank_mean_tolerance=rank_mean_tolerance,
                    rank_variance_tolerance=rank_variance_tolerance,
                    z_value=z_value,
                ),
            }
        )
    return {
        "kind": "post_scale_final_multiplier_ood_diagnostics",
        "domains": domains,
    }


def _summary_quantiles(values: np.ndarray) -> dict[str, float]:
    """Return compact JSON-safe summary statistics."""
    array = np.asarray(values, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "mean": float("nan"),
            "min": float("nan"),
            "q05": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "q95": float("nan"),
            "max": float("nan"),
        }
    q05, q25, q50, q75, q95 = np.quantile(array, (0.05, 0.25, 0.5, 0.75, 0.95))
    return {
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "q05": float(q05),
        "q25": float(q25),
        "median": float(q50),
        "q75": float(q75),
        "q95": float(q95),
        "max": float(np.max(array)),
    }


def _fit_and_select_domain_expert_ood_parameters(
    *,
    batches: Sequence[ConditionalBetaOODCalibrationBatch],
    parameters: tuple[float, ...],
    location: np.ndarray,
    feature_scale: np.ndarray,
    feature_names: tuple[str, ...],
    support_lower: np.ndarray,
    support_upper: np.ndarray,
    support_precision: np.ndarray,
    support_radius: float,
    mean_magnitude_location: float,
    mean_magnitude_scale: float,
    mean_magnitude_lower: float,
    mean_magnitude_upper: float,
    normalization: float,
    global_multiplier: float,
    fallback_strength: float,
    fitted_weights: np.ndarray,
    mean_bias_correction: tuple[tuple[float, ...], ...],
    rank_centering_offsets: tuple[tuple[float, ...], ...],
    base_scale_stratum_offsets: tuple[float, ...],
    post_scale_log_offsets: tuple[float, float, float],
    post_scale_support_threshold: float,
    post_scale_support_width: float,
    post_scale_community_threshold: float,
    post_scale_community_width: float,
    distribution: str,
    n_covariates: int,
    n_species: int,
    prevalence_edges: tuple[float, float],
    in_domain_signed_error: np.ndarray,
    in_domain_adjustment: np.ndarray,
    in_domain_trust: np.ndarray,
    in_domain_support_excess: np.ndarray,
    in_domain_effect_signal: np.ndarray,
    in_domain_design_signal: np.ndarray,
    in_domain_prevalence_stratum: np.ndarray,
    in_domain_design_stratum: np.ndarray,
    in_domain_coefficient_stratum: np.ndarray,
    in_domain_post_scale_multiplier: np.ndarray,
    in_domain_rank_groups: list[np.ndarray],
    rank_mean_tolerance: float,
    rank_variance_tolerance: float,
    nominal_level: float,
    z_value: float,
    min_multiplier: float,
    max_multiplier: float,
    max_ood_multiplier: float,
    objective_weight: float,
    in_domain_gate_weight: float,
    epochs: int,
    learning_rate: float,
    in_domain_community_occupancy: np.ndarray,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Fit domain-specific OOD experts and select one with held-out gates."""
    if not batches:
        return parameters, {}

    thresholds: dict[str, float] = {
        "min_effect_size_coverage_gain": 0.020,
        "min_combined_shift_coverage_gain": 0.020,
        "max_non_target_coverage_loss": 0.0,
        "max_overlap_excess_loss": 0.12,
        "max_extra_over_1_05_loss_increase": 0.25,
        "max_group_extra_cap_loss_increase": 0.08,
        "max_mean_group_loss_increase": 0.01,
        "max_max_group_loss_increase": 0.04,
    }
    shrinkage_grid = (
        0.0,
        0.03125,
        0.0625,
        0.09375,
        0.125,
        0.1875,
        0.25,
        0.375,
        0.5,
        0.75,
        1.0,
    )
    expert_overlap_penalty_grid = (
        {
            "kind": "domain_localized_overlap_penalty",
            "name": "localized_w4_tol106",
            "fit_mode": "single_stage",
            "weight": 4.0,
            "log_tolerance": float(np.log(1.06)),
            "scale": float(np.log(1.25)),
            "target_coverage_weight": 4.0,
            "effect_quantile_weight": 3.0,
            "margin_weight": 0.0,
            "bin_in_domain_penalty_weight": 0.20,
            "in_domain_gate_weight_multiplier": 1.0,
            "projection_cap_grid": (0.25, 0.5, 0.75, 1.0),
        },
        {
            "kind": "domain_localized_overlap_penalty",
            "name": "localized_w8_tol104",
            "fit_mode": "single_stage",
            "weight": 8.0,
            "log_tolerance": float(np.log(1.04)),
            "scale": float(np.log(1.25)),
            "target_coverage_weight": 6.0,
            "effect_quantile_weight": 4.0,
            "margin_weight": 0.0,
            "bin_in_domain_penalty_weight": 0.35,
            "in_domain_gate_weight_multiplier": 1.0,
            "projection_cap_grid": (0.125, 0.25, 0.5, 0.75, 1.0),
        },
        {
            "kind": "domain_localized_overlap_penalty",
            "name": "localized_w12_tol103",
            "fit_mode": "single_stage",
            "weight": 12.0,
            "log_tolerance": float(np.log(1.03)),
            "scale": float(np.log(1.25)),
            "target_coverage_weight": 8.0,
            "effect_quantile_weight": 5.0,
            "margin_weight": 0.0,
            "bin_in_domain_penalty_weight": 0.50,
            "in_domain_gate_weight_multiplier": 1.0,
            "projection_cap_grid": (0.125, 0.25, 0.5, 0.75, 1.0),
        },
        {
            "kind": "domain_localized_overlap_penalty",
            "name": "combined_target_w14_tol102_projection",
            "target_domains": ("combined_shift",),
            "fit_mode": "single_stage",
            "weight": 14.0,
            "log_tolerance": float(np.log(1.02)),
            "scale": float(np.log(1.20)),
            "target_coverage_weight": 12.0,
            "effect_quantile_weight": 8.0,
            "margin_weight": 4.0,
            "bin_in_domain_penalty_weight": 0.80,
            "in_domain_gate_weight_multiplier": 1.25,
            "projection_cap_grid": (0.0625, 0.125, 0.25, 0.375, 0.5),
        },
        {
            "kind": "two_stage_target_then_projection",
            "name": "two_stage_target_w6_projection",
            "fit_mode": "target_then_projection",
            "weight": 2.0,
            "log_tolerance": float(np.log(1.08)),
            "scale": float(np.log(1.25)),
            "target_coverage_weight": 6.0,
            "effect_quantile_weight": 4.0,
            "margin_weight": 5.0,
            "bin_in_domain_penalty_weight": 0.0,
            "in_domain_gate_weight_multiplier": 0.35,
            "projection_cap_grid": (0.0625, 0.125, 0.25, 0.5, 0.75, 1.0),
        },
    )
    expert_overlap_penalty = expert_overlap_penalty_grid[0]
    baseline_in_domain_log_ood = _learned_ood_log_inflation_numpy(
        in_domain_support_excess,
        effect_signal=in_domain_effect_signal,
        design_signal=in_domain_design_signal,
        community_occupancy=in_domain_community_occupancy,
        prevalence_stratum=in_domain_prevalence_stratum,
        design_stratum=in_domain_design_stratum,
        coefficient_stratum=in_domain_coefficient_stratum,
        parameters=parameters,
        max_multiplier=max_ood_multiplier,
    )

    pure_train, pure_eval, pure_split = _domain_expert_batch_split(
        batches, "effect_size_shift"
    )
    combined_train, combined_eval, combined_split = _domain_expert_batch_split(
        batches, "combined_shift"
    )
    evaluation_batches = tuple(pure_eval + combined_eval)
    if not evaluation_batches:
        evaluation_batches = tuple(batches)

    def candidate_diagnostics(candidate: tuple[float, ...]) -> dict[str, Any]:
        in_domain_base = _blend_with_scalar_fallback(
            in_domain_adjustment,
            in_domain_trust,
            support_excess=in_domain_support_excess,
            effect_signal=in_domain_effect_signal,
            design_signal=in_domain_design_signal,
            community_occupancy=in_domain_community_occupancy,
            prevalence_stratum=in_domain_prevalence_stratum,
            design_stratum=in_domain_design_stratum,
            coefficient_stratum=in_domain_coefficient_stratum,
            normalization=normalization,
            global_multiplier=global_multiplier,
            ood_uncertainty_strength=0.0,
            ood_uncertainty_max_multiplier=max_ood_multiplier,
            ood_inflation_parameters=candidate,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
        in_domain_final = np.clip(
            in_domain_base * in_domain_post_scale_multiplier,
            min_multiplier,
            max_multiplier,
        )
        in_domain_log_ood = _learned_ood_log_inflation_numpy(
            in_domain_support_excess,
            effect_signal=in_domain_effect_signal,
            design_signal=in_domain_design_signal,
            community_occupancy=in_domain_community_occupancy,
            prevalence_stratum=in_domain_prevalence_stratum,
            design_stratum=in_domain_design_stratum,
            coefficient_stratum=in_domain_coefficient_stratum,
            parameters=candidate,
            max_multiplier=max_ood_multiplier,
        )
        return _ood_final_multiplier_diagnostics(
            batches=evaluation_batches,
            location=location,
            feature_scale=feature_scale,
            feature_names=feature_names,
            support_lower=support_lower,
            support_upper=support_upper,
            support_precision=support_precision,
            support_radius=support_radius,
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
            normalization=normalization,
            global_multiplier=global_multiplier,
            fallback_strength=fallback_strength,
            fitted_weights=fitted_weights,
            mean_bias_correction=mean_bias_correction,
            rank_centering_offsets=rank_centering_offsets,
            base_scale_stratum_offsets=base_scale_stratum_offsets,
            ood_inflation_parameters=candidate,
            post_scale_log_offsets=post_scale_log_offsets,
            post_scale_support_threshold=post_scale_support_threshold,
            post_scale_support_width=post_scale_support_width,
            post_scale_community_threshold=post_scale_community_threshold,
            post_scale_community_width=post_scale_community_width,
            combined_shift_scale_log_amplitude=0.0,
            distribution=distribution,
            n_covariates=n_covariates,
            n_species=n_species,
            prevalence_edges=prevalence_edges,
            in_domain_signed_error=in_domain_signed_error,
            in_domain_final_multiplier=in_domain_final,
            in_domain_log_ood_inflation=in_domain_log_ood,
            in_domain_rank_groups=in_domain_rank_groups,
            rank_mean_tolerance=rank_mean_tolerance,
            rank_variance_tolerance=rank_variance_tolerance,
            nominal_level=nominal_level,
            z_value=z_value,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            max_ood_multiplier=max_ood_multiplier,
        )

    baseline_diagnostics = candidate_diagnostics(parameters)
    baseline_summary = _effect_shift_selection_summary(baseline_diagnostics)
    candidate_records: list[dict[str, Any]] = []
    accepted_candidates: list[tuple[float, tuple[float, ...], dict[str, Any]]] = []

    def fit_expert(
        *,
        expert: str,
        target_label: str,
        train_batches: tuple[ConditionalBetaOODCalibrationBatch, ...],
        split_mode: str,
    ) -> None:
        if not train_batches:
            candidate_records.append(
                {
                    "expert": expert,
                    "target_domain": target_label,
                    "accepted": False,
                    "reason": "no_training_batches",
                    "split_mode": split_mode,
                }
            )
            return
        required_gain = (
            thresholds["min_effect_size_coverage_gain"]
            if target_label == "effect_size_shift"
            else thresholds["min_combined_shift_coverage_gain"]
        )
        for overlap_penalty in expert_overlap_penalty_grid:
            target_domains = tuple(
                str(value) for value in overlap_penalty.get("target_domains", ())
            )
            if target_domains and target_label not in target_domains:
                continue
            (
                candidate,
                objective_loss,
                rank_loss,
                gate_loss,
                n_observations,
                training_domains,
            ) = _fit_ood_inflation_parameters(
                train_batches,
                location=location,
                feature_scale=feature_scale,
                feature_names=feature_names,
                support_lower=support_lower,
                support_upper=support_upper,
                support_precision=support_precision,
                support_radius=support_radius,
                mean_magnitude_location=mean_magnitude_location,
                mean_magnitude_scale=mean_magnitude_scale,
                mean_magnitude_lower=mean_magnitude_lower,
                mean_magnitude_upper=mean_magnitude_upper,
                normalization=normalization,
                global_multiplier=global_multiplier,
                fallback_strength=fallback_strength,
                fitted_weights=fitted_weights,
                mean_bias_correction=mean_bias_correction,
                rank_centering_offsets=rank_centering_offsets,
                base_scale_stratum_offsets=base_scale_stratum_offsets,
                in_domain_signed_error=in_domain_signed_error,
                in_domain_adjustment=in_domain_adjustment,
                in_domain_trust=in_domain_trust,
                in_domain_support_excess=in_domain_support_excess,
                in_domain_effect_signal=in_domain_effect_signal,
                in_domain_design_signal=in_domain_design_signal,
                in_domain_prevalence_stratum=in_domain_prevalence_stratum,
                in_domain_design_stratum=in_domain_design_stratum,
                in_domain_coefficient_stratum=in_domain_coefficient_stratum,
                in_domain_rank_groups=in_domain_rank_groups,
                distribution=distribution,
                n_covariates=n_covariates,
                n_species=n_species,
                min_multiplier=min_multiplier,
                max_multiplier=max_multiplier,
                max_ood_multiplier=max_ood_multiplier,
                objective_weight=objective_weight,
                in_domain_gate_weight=(
                    in_domain_gate_weight
                    * float(overlap_penalty["in_domain_gate_weight_multiplier"])
                ),
                epochs=max(1, int(epochs)),
                learning_rate=learning_rate,
                gate_effect_branch=True,
                prevalence_edges=prevalence_edges,
                rank_mean_tolerance=rank_mean_tolerance,
                rank_variance_tolerance=rank_variance_tolerance,
                nominal_level=nominal_level,
                z_value=z_value,
                initial_parameters=parameters,
                final_multiplier_aware=True,
                post_scale_log_offsets=post_scale_log_offsets,
                post_scale_support_threshold=post_scale_support_threshold,
                post_scale_support_width=post_scale_support_width,
                post_scale_community_threshold=post_scale_community_threshold,
                post_scale_community_width=post_scale_community_width,
                in_domain_community_occupancy=in_domain_community_occupancy,
                trust_region_parameters=parameters,
                trust_region_weight=0.0,
                trust_region_log_tolerance=0.0,
                overlap_penalty_domain=target_label,
                overlap_penalty_weight=float(overlap_penalty["weight"]),
                overlap_log_tolerance=float(overlap_penalty["log_tolerance"]),
                bin_in_domain_penalty_weight=float(
                    overlap_penalty["bin_in_domain_penalty_weight"]
                ),
                expert_target_coverage_weight=float(
                    overlap_penalty["target_coverage_weight"]
                ),
                expert_effect_quantile_weight=float(
                    overlap_penalty["effect_quantile_weight"]
                ),
                expert_margin_weight=float(overlap_penalty["margin_weight"]),
            )
            grid_records: list[dict[str, Any]] = []
            best_grid_record: dict[str, Any] | None = None
            best_grid_parameters = parameters
            best_accepted_record: dict[str, Any] | None = None
            best_accepted_parameters = parameters
            best_gate_compatible_record: dict[str, Any] | None = None
            best_gate_compatible_parameters = parameters
            projection_cap_grid = tuple(
                float(value) for value in overlap_penalty["projection_cap_grid"]
            )
            for shrinkage in shrinkage_grid:
                for projection_cap in projection_cap_grid:
                    shrunk_candidate = _interpolate_ood_parameters(
                        parameters,
                        candidate,
                        shrinkage=float(shrinkage),
                    )
                    projected_candidate = _project_ood_effect_bin_parameters(
                        parameters,
                        shrunk_candidate,
                        n_covariates=n_covariates,
                        expert=expert,
                        cap=float(projection_cap),
                    )
                    summary = _effect_shift_selection_summary(
                        candidate_diagnostics(projected_candidate)
                    )
                    domain_gain = {
                        label: float(
                            coverage
                            - baseline_summary["domain_coverages"].get(label, 0.0)
                        )
                        for label, coverage in summary["domain_coverages"].items()
                    }
                    gate_delta = _effect_shift_gate_delta(
                        baseline_summary["in_domain_gate"],
                        summary["in_domain_gate"],
                    )
                    candidate_log_ood = _learned_ood_log_inflation_numpy(
                        in_domain_support_excess,
                        effect_signal=in_domain_effect_signal,
                        design_signal=in_domain_design_signal,
                        community_occupancy=in_domain_community_occupancy,
                        prevalence_stratum=in_domain_prevalence_stratum,
                        design_stratum=in_domain_design_stratum,
                        coefficient_stratum=in_domain_coefficient_stratum,
                        parameters=projected_candidate,
                        max_multiplier=max_ood_multiplier,
                    )
                    overlap_control = _domain_overlap_control_summary(
                        domain=target_label,
                        support_excess=in_domain_support_excess,
                        effect_signal=in_domain_effect_signal,
                        design_signal=in_domain_design_signal,
                        community_occupancy=in_domain_community_occupancy,
                        baseline_log_ood=baseline_in_domain_log_ood,
                        candidate_log_ood=candidate_log_ood,
                        log_tolerance=float(overlap_penalty["log_tolerance"]),
                        scale=float(overlap_penalty["scale"]),
                    )
                    target_gain = float(domain_gain.get(target_label, 0.0))
                    non_target_gains = [
                        gain
                        for label, gain in domain_gain.items()
                        if label != target_label
                    ]
                    non_target_ok = all(
                        gain >= -thresholds["max_non_target_coverage_loss"]
                        for gain in non_target_gains
                    )
                    overlap_ok = (
                        overlap_control["weighted_excess_loss"]
                        <= thresholds["max_overlap_excess_loss"]
                    )
                    accepted = (
                        target_gain >= required_gain
                        and non_target_ok
                        and overlap_ok
                        and _effect_shift_gate_delta_ok(gate_delta, thresholds)
                    )
                    gate_compatible = (
                        target_gain > 0.0
                        and non_target_ok
                        and overlap_ok
                        and _effect_shift_gate_delta_ok(gate_delta, thresholds)
                    )
                    grid_record = {
                        "shrinkage": float(shrinkage),
                        "projection_cap": float(projection_cap),
                        "accepted": bool(accepted),
                        "gate_compatible": bool(gate_compatible),
                        "target_coverage_gain": target_gain,
                        "domain_coverage_gains": domain_gain,
                        "non_target_coverage_ok": bool(non_target_ok),
                        "overlap_control_ok": bool(overlap_ok),
                        "overlap_control": overlap_control,
                        "in_domain_gate_delta": gate_delta,
                        **summary,
                    }
                    grid_records.append(grid_record)
                    if best_grid_record is None or target_gain > float(
                        best_grid_record["target_coverage_gain"]
                    ):
                        best_grid_record = grid_record
                        best_grid_parameters = projected_candidate
                    if accepted and (
                        best_accepted_record is None
                        or target_gain
                        > float(best_accepted_record["target_coverage_gain"])
                    ):
                        best_accepted_record = grid_record
                        best_accepted_parameters = projected_candidate
                    if gate_compatible and (
                        best_gate_compatible_record is None
                        or _domain_expert_projection_sort_key(grid_record)
                        > _domain_expert_projection_sort_key(
                            best_gate_compatible_record
                        )
                    ):
                        best_gate_compatible_record = grid_record
                        best_gate_compatible_parameters = projected_candidate

            selected_grid_record = (
                best_accepted_record or best_gate_compatible_record or best_grid_record
            )
            if selected_grid_record is None:
                selected_grid_record = {
                    "shrinkage": 0.0,
                    "accepted": False,
                    "gate_compatible": True,
                    "target_coverage_gain": 0.0,
                    "domain_coverage_gains": {},
                    "in_domain_gate_delta": {},
                    **baseline_summary,
                }
            selected_parameters_for_record = (
                best_accepted_parameters
                if best_accepted_record is not None
                else (
                    best_gate_compatible_parameters
                    if best_gate_compatible_record is not None
                    else best_grid_parameters
                )
            )
            record = {
                "expert": expert,
                "target_domain": target_label,
                "accepted": bool(best_accepted_record is not None),
                "split_mode": split_mode,
                "training_domains": list(training_domains),
                "n_training_observations": int(n_observations),
                "objective_loss": float(objective_loss),
                "rank_loss": float(rank_loss),
                "in_domain_gate_loss": float(gate_loss),
                "overlap_penalty": overlap_penalty,
                "selected_shrinkage": float(selected_grid_record["shrinkage"]),
                "selected_projection_cap": float(
                    selected_grid_record.get("projection_cap", 1.0)
                ),
                "selected_gate_compatible": bool(
                    selected_grid_record.get("gate_compatible", False)
                ),
                "selection_rule": (
                    "accepted"
                    if best_accepted_record is not None
                    else (
                        "best_gate_compatible"
                        if best_gate_compatible_record is not None
                        else "best_target_gain"
                    )
                ),
                "full_expert_shrinkage": 1.0,
                "shrinkage_grid": grid_records,
                "target_coverage_gain": float(
                    selected_grid_record["target_coverage_gain"]
                ),
                "domain_coverage_gains": selected_grid_record["domain_coverage_gains"],
                "in_domain_gate_delta": selected_grid_record["in_domain_gate_delta"],
                "domain_coverages": selected_grid_record["domain_coverages"],
                "mean_ood_coverage": float(selected_grid_record["mean_ood_coverage"]),
                "worst_ood_domain_coverage": float(
                    selected_grid_record["worst_ood_domain_coverage"]
                ),
                "in_domain_gate": selected_grid_record["in_domain_gate"],
            }
            candidate_records.append(record)
            if best_accepted_record is not None:
                selection_score = float(
                    best_accepted_record["target_coverage_gain"]
                ) + 0.5 * float(
                    best_accepted_record["worst_ood_domain_coverage"]
                    - baseline_summary["worst_ood_domain_coverage"]
                )
                accepted_candidates.append(
                    (selection_score, selected_parameters_for_record, record)
                )

    fit_expert(
        expert="pure_effect",
        target_label="effect_size_shift",
        train_batches=pure_train,
        split_mode=pure_split,
    )
    fit_expert(
        expert="combined_shift",
        target_label="combined_shift",
        train_batches=combined_train,
        split_mode=combined_split,
    )

    if accepted_candidates:
        accepted_candidates.sort(key=lambda item: item[0], reverse=True)
        _, selected_parameters, selected_record = accepted_candidates[0]
    else:
        selected_parameters = parameters
        selected_record = {
            "expert": "baseline",
            "accepted": False,
            "reason": "no_candidate_passed_selection_gate",
            "selected_shrinkage": 0.0,
            **baseline_summary,
        }

    return selected_parameters, {
        "kind": "heldout_domain_expert_ood_selection",
        "thresholds": thresholds,
        "expert_overlap_penalty": expert_overlap_penalty,
        "expert_overlap_penalty_grid": expert_overlap_penalty_grid,
        "shrinkage_grid": [float(value) for value in shrinkage_grid],
        "evaluation_domains": [str(batch.label) for batch in evaluation_batches],
        "split_modes": {
            "pure_effect": pure_split,
            "combined_shift": combined_split,
        },
        "baseline": baseline_summary,
        "selected": selected_record,
        "candidates": candidate_records,
    }


def _domain_expert_projection_sort_key(record: dict[str, Any]) -> tuple[float, ...]:
    """Rank non-accepted projection rows by gain, then low inflation penalties."""
    delta = record.get("in_domain_gate_delta", {})
    return (
        float(record.get("target_coverage_gain", 0.0)),
        -float(delta.get("extra_inflation_over_1_05_loss", 0.0)),
        -float(delta.get("max_group_extra_inflation_cap_loss", 0.0)),
        -float(delta.get("mean_group_loss", 0.0)),
        -float(delta.get("max_group_loss", 0.0)),
        -float(record.get("projection_cap", 1.0)),
        -float(record.get("shrinkage", 1.0)),
    )


def _interpolate_ood_parameters(
    baseline: tuple[float, ...],
    candidate: tuple[float, ...],
    *,
    shrinkage: float,
) -> tuple[float, ...]:
    """Return a convex interpolation between baseline and trained OOD expert."""
    bounded = float(np.clip(shrinkage, 0.0, 1.0))
    if len(baseline) != len(candidate):
        return candidate if bounded >= 1.0 else baseline
    baseline_array = np.asarray(baseline, dtype=float)
    candidate_array = np.asarray(candidate, dtype=float)
    values = baseline_array + bounded * (candidate_array - baseline_array)
    return tuple(float(value) for value in values)


def _project_ood_effect_bin_parameters(
    baseline: tuple[float, ...],
    candidate: tuple[float, ...],
    *,
    n_covariates: int,
    expert: str,
    cap: float,
) -> tuple[float, ...]:
    """Shrink branch-specific effect-bin movement without changing vector shape."""
    bounded = float(np.clip(cap, 0.0, 1.0))
    if bounded >= 1.0 or len(baseline) != len(candidate):
        return tuple(float(value) for value in candidate)
    head_start = 15 + int(n_covariates)
    head_end = head_start + _OOD_EFFECT_SHIFT_HEAD_PARAMETER_COUNT
    if len(candidate) < head_end:
        return tuple(float(value) for value in candidate)

    baseline_array = np.asarray(baseline, dtype=float)
    candidate_array = np.asarray(candidate, dtype=float).copy()
    if expert == "pure_effect":
        projected_indices = (
            head_start + 3,
            head_start + 8,
            head_start + 9,
            head_start + 10,
        )
    elif expert == "combined_shift":
        projected_indices = (
            head_start + 7,
            head_start + 11,
            head_start + 12,
            head_start + 13,
        )
    else:
        projected_indices = ()
    for index in projected_indices:
        candidate_array[index] = baseline_array[index] + bounded * (
            candidate_array[index] - baseline_array[index]
        )
    return tuple(float(value) for value in candidate_array)


def _domain_overlap_context_numpy(
    *,
    domain: str,
    support_excess: np.ndarray,
    effect_signal: np.ndarray,
    design_signal: np.ndarray,
    community_occupancy: np.ndarray,
) -> np.ndarray:
    """Return fixed in-domain context weights for one OOD expert domain."""
    support = np.asarray(support_excess, dtype=float)
    effect = np.asarray(effect_signal, dtype=float)
    design = np.asarray(design_signal, dtype=float)
    community = np.asarray(community_occupancy, dtype=float)
    positive_support = np.maximum(support, 0.0)
    if domain == "effect_size_shift":
        high_effect = 1.0 / (1.0 + np.exp(-((effect - 0.75) / 0.35)))
        support_close = 1.0 / (1.0 + np.exp(-((0.25 - positive_support) / 0.25)))
        return high_effect * support_close
    if domain == "combined_shift":
        support_gate = 1.0 / (1.0 + np.exp(-((positive_support - 0.20) / 0.35)))
        effect_gate = 1.0 / (1.0 + np.exp(-((effect - 0.25) / 0.50)))
        low_design_gate = 1.0 / (1.0 + np.exp(-((0.75 - design) / 0.35)))
        low_community_gate = 1.0 / (1.0 + np.exp(-((0.45 - community) / 0.06)))
        return support_gate * effect_gate * low_design_gate * low_community_gate
    return np.zeros_like(support, dtype=float)


def _domain_overlap_control_summary(
    *,
    domain: str,
    support_excess: np.ndarray,
    effect_signal: np.ndarray,
    design_signal: np.ndarray,
    community_occupancy: np.ndarray,
    baseline_log_ood: np.ndarray,
    candidate_log_ood: np.ndarray,
    log_tolerance: float,
    scale: float,
) -> dict[str, float]:
    """Summarize in-domain OOD-log drift within the target-domain overlap."""
    context = _domain_overlap_context_numpy(
        domain=domain,
        support_excess=support_excess,
        effect_signal=effect_signal,
        design_signal=design_signal,
        community_occupancy=community_occupancy,
    ).reshape(-1)
    baseline = np.asarray(baseline_log_ood, dtype=float).reshape(-1)
    candidate = np.asarray(candidate_log_ood, dtype=float).reshape(-1)
    drift = np.abs(candidate - baseline)
    tolerance = max(float(log_tolerance), 0.0)
    denominator = float(np.sum(context)) + 1e-8
    scaled_excess = np.maximum(drift - tolerance, 0.0) / max(float(scale), 1e-8)
    weighted_excess_loss = float(
        np.sum(context * np.square(scaled_excess)) / denominator
    )
    weighted_mean_abs_log_drift = float(np.sum(context * drift) / denominator)
    active = context > 0.5
    active_fraction = float(np.mean(active)) if context.size else 0.0
    active_mean_abs_log_drift = float(np.mean(drift[active])) if np.any(active) else 0.0
    return {
        "domain": str(domain),
        "mean_context": float(np.mean(context)) if context.size else 0.0,
        "active_fraction_over_0_5": active_fraction,
        "weighted_mean_abs_log_drift": weighted_mean_abs_log_drift,
        "active_mean_abs_log_drift": active_mean_abs_log_drift,
        "weighted_excess_loss": weighted_excess_loss,
    }


def _domain_expert_batch_split(
    batches: Sequence[ConditionalBetaOODCalibrationBatch],
    domain_label: str,
) -> tuple[
    tuple[ConditionalBetaOODCalibrationBatch, ...],
    tuple[ConditionalBetaOODCalibrationBatch, ...],
    str,
]:
    """Return train/evaluation batches for one OOD expert domain."""
    domain_batches = tuple(
        batch for batch in batches if domain_label in str(batch.label)
    )
    if len(domain_batches) >= 2:
        train = domain_batches[::2]
        evaluation = domain_batches[1::2] or domain_batches[-1:]
        return train, evaluation, "alternating_batches"
    if len(domain_batches) == 1:
        split = _split_ood_batch_on_batch_axis(domain_batches[0])
        if split is not None:
            return (split[0],), (split[1],), "within_batch_axis0"
        return domain_batches, domain_batches, "same_batch_fallback"
    return (), (), "no_batches"


def _split_ood_batch_on_batch_axis(
    batch: ConditionalBetaOODCalibrationBatch,
) -> (
    tuple[ConditionalBetaOODCalibrationBatch, ConditionalBetaOODCalibrationBatch] | None
):
    """Split a calibration batch along the simulation axis when possible."""
    mean = _as_numpy(batch.posterior.mean)
    if mean.ndim != 3 or mean.shape[0] < 2:
        return None
    train_index = np.arange(0, mean.shape[0], 2)
    validation_index = np.arange(1, mean.shape[0], 2)
    if train_index.size == 0 or validation_index.size == 0:
        return None

    def slice_array(value: Any, index: np.ndarray) -> np.ndarray:
        return np.take(np.asarray(value), index, axis=0)

    scale_tril = batch.posterior.scale_tril
    train_posterior = BetaPosterior(
        mean=slice_array(batch.posterior.mean, train_index),
        scale=slice_array(batch.posterior.scale, train_index),
        scale_tril=(
            None if scale_tril is None else slice_array(scale_tril, train_index)
        ),
    )
    validation_posterior = BetaPosterior(
        mean=slice_array(batch.posterior.mean, validation_index),
        scale=slice_array(batch.posterior.scale, validation_index),
        scale_tril=(
            None if scale_tril is None else slice_array(scale_tril, validation_index)
        ),
    )
    train_batch = ConditionalBetaOODCalibrationBatch(
        posterior=train_posterior,
        beta_true=slice_array(batch.beta_true, train_index),
        X=slice_array(batch.X, train_index),
        Y=slice_array(batch.Y, train_index),
        label=batch.label,
        weight=batch.weight,
    )
    validation_batch = ConditionalBetaOODCalibrationBatch(
        posterior=validation_posterior,
        beta_true=slice_array(batch.beta_true, validation_index),
        X=slice_array(batch.X, validation_index),
        Y=slice_array(batch.Y, validation_index),
        label=batch.label,
        weight=batch.weight,
    )
    return train_batch, validation_batch


def _select_effect_shift_head_shrinkage(
    *,
    batches: Sequence[ConditionalBetaOODCalibrationBatch],
    parameters: tuple[float, ...],
    location: np.ndarray,
    feature_scale: np.ndarray,
    feature_names: tuple[str, ...],
    support_lower: np.ndarray,
    support_upper: np.ndarray,
    support_precision: np.ndarray,
    support_radius: float,
    mean_magnitude_location: float,
    mean_magnitude_scale: float,
    mean_magnitude_lower: float,
    mean_magnitude_upper: float,
    normalization: float,
    global_multiplier: float,
    fallback_strength: float,
    fitted_weights: np.ndarray,
    mean_bias_correction: tuple[tuple[float, ...], ...],
    rank_centering_offsets: tuple[tuple[float, ...], ...],
    base_scale_stratum_offsets: tuple[float, ...],
    post_scale_log_offsets: tuple[float, float, float],
    post_scale_support_threshold: float,
    post_scale_support_width: float,
    post_scale_community_threshold: float,
    post_scale_community_width: float,
    distribution: str,
    n_covariates: int,
    n_species: int,
    prevalence_edges: tuple[float, float],
    in_domain_signed_error: np.ndarray,
    in_domain_adjustment: np.ndarray,
    in_domain_trust: np.ndarray,
    in_domain_support_excess: np.ndarray,
    in_domain_effect_signal: np.ndarray,
    in_domain_design_signal: np.ndarray,
    in_domain_prevalence_stratum: np.ndarray,
    in_domain_design_stratum: np.ndarray,
    in_domain_coefficient_stratum: np.ndarray,
    in_domain_post_scale_multiplier: np.ndarray,
    in_domain_rank_groups: list[np.ndarray],
    rank_mean_tolerance: float,
    rank_variance_tolerance: float,
    nominal_level: float,
    z_value: float,
    min_multiplier: float,
    max_multiplier: float,
    max_ood_multiplier: float,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Select effect-shift head shrinkage after fitting against explicit gates."""
    head_start = 15 + int(n_covariates)
    if len(parameters) < head_start + _OOD_EFFECT_SHIFT_HEAD_BASE_PARAMETER_COUNT:
        return parameters, {}

    thresholds: dict[str, float] = {
        "min_effect_size_coverage_gain": 0.0125,
        "min_combined_shift_coverage_gain": 0.0125,
        "max_extra_over_1_05_loss_increase": 0.25,
        "max_group_extra_cap_loss_increase": 0.08,
        "max_mean_group_loss_increase": 0.01,
        "max_max_group_loss_increase": 0.04,
    }

    def candidate_diagnostics(
        pure_shrinkage: float,
        combined_shrinkage: float,
    ) -> tuple[tuple[float, ...], dict[str, Any]]:
        candidate = _shrink_effect_shift_head(
            parameters,
            n_covariates=n_covariates,
            pure_shrinkage=pure_shrinkage,
            combined_shrinkage=combined_shrinkage,
        )
        in_domain_base = _blend_with_scalar_fallback(
            in_domain_adjustment,
            in_domain_trust,
            support_excess=in_domain_support_excess,
            effect_signal=in_domain_effect_signal,
            design_signal=in_domain_design_signal,
            prevalence_stratum=in_domain_prevalence_stratum,
            design_stratum=in_domain_design_stratum,
            coefficient_stratum=in_domain_coefficient_stratum,
            normalization=normalization,
            global_multiplier=global_multiplier,
            ood_uncertainty_strength=0.0,
            ood_uncertainty_max_multiplier=max_ood_multiplier,
            ood_inflation_parameters=candidate,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
        in_domain_final = np.clip(
            in_domain_base * in_domain_post_scale_multiplier,
            min_multiplier,
            max_multiplier,
        )
        in_domain_log_ood = _learned_ood_log_inflation_numpy(
            in_domain_support_excess,
            effect_signal=in_domain_effect_signal,
            design_signal=in_domain_design_signal,
            prevalence_stratum=in_domain_prevalence_stratum,
            design_stratum=in_domain_design_stratum,
            coefficient_stratum=in_domain_coefficient_stratum,
            parameters=candidate,
            max_multiplier=max_ood_multiplier,
        )
        diagnostics = _ood_final_multiplier_diagnostics(
            batches=batches,
            location=location,
            feature_scale=feature_scale,
            feature_names=feature_names,
            support_lower=support_lower,
            support_upper=support_upper,
            support_precision=support_precision,
            support_radius=support_radius,
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
            normalization=normalization,
            global_multiplier=global_multiplier,
            fallback_strength=fallback_strength,
            fitted_weights=fitted_weights,
            mean_bias_correction=mean_bias_correction,
            rank_centering_offsets=rank_centering_offsets,
            base_scale_stratum_offsets=base_scale_stratum_offsets,
            ood_inflation_parameters=candidate,
            post_scale_log_offsets=post_scale_log_offsets,
            post_scale_support_threshold=post_scale_support_threshold,
            post_scale_support_width=post_scale_support_width,
            post_scale_community_threshold=post_scale_community_threshold,
            post_scale_community_width=post_scale_community_width,
            combined_shift_scale_log_amplitude=0.0,
            distribution=distribution,
            n_covariates=n_covariates,
            n_species=n_species,
            prevalence_edges=prevalence_edges,
            in_domain_signed_error=in_domain_signed_error,
            in_domain_final_multiplier=in_domain_final,
            in_domain_log_ood_inflation=in_domain_log_ood,
            in_domain_rank_groups=in_domain_rank_groups,
            rank_mean_tolerance=rank_mean_tolerance,
            rank_variance_tolerance=rank_variance_tolerance,
            nominal_level=nominal_level,
            z_value=z_value,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            max_ood_multiplier=max_ood_multiplier,
        )
        return candidate, diagnostics

    baseline_parameters, baseline_diagnostics = candidate_diagnostics(0.0, 0.0)
    baseline_summary = _effect_shift_selection_summary(baseline_diagnostics)
    best_parameters = baseline_parameters
    selected_pure = 0.0
    selected_combined = 0.0
    candidate_records: list[dict[str, Any]] = []

    def build_record(
        *,
        branch: str,
        pure_shrinkage: float,
        combined_shrinkage: float,
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        domain_gain = {
            label: float(
                coverage - baseline_summary["domain_coverages"].get(label, 0.0)
            )
            for label, coverage in summary["domain_coverages"].items()
        }
        gate_delta = _effect_shift_gate_delta(
            baseline_summary["in_domain_gate"],
            summary["in_domain_gate"],
        )
        if branch == "pure_effect":
            target_gain = domain_gain.get("effect_size_shift", 0.0)
            accepted = target_gain >= thresholds[
                "min_effect_size_coverage_gain"
            ] and _effect_shift_gate_delta_ok(gate_delta, thresholds)
        elif branch == "combined_shift":
            target_gain = domain_gain.get("combined_shift", 0.0)
            accepted = target_gain >= thresholds[
                "min_combined_shift_coverage_gain"
            ] and _effect_shift_gate_delta_ok(gate_delta, thresholds)
        else:
            target_gain = 0.0
            accepted = False
        return {
            "branch": branch,
            "pure_shrinkage": float(pure_shrinkage),
            "combined_shrinkage": float(combined_shrinkage),
            "accepted": bool(accepted),
            "target_coverage_gain": float(target_gain),
            "domain_coverage_gains": domain_gain,
            "in_domain_gate_delta": gate_delta,
            **summary,
        }

    best_pure_record: dict[str, Any] | None = None
    for pure_shrinkage in (0.25, 0.5, 0.75, 1.0):
        candidate, diagnostics = candidate_diagnostics(pure_shrinkage, 0.0)
        summary = _effect_shift_selection_summary(diagnostics)
        record = build_record(
            branch="pure_effect",
            pure_shrinkage=pure_shrinkage,
            combined_shrinkage=0.0,
            summary=summary,
        )
        candidate_records.append(record)
        if record["accepted"] and (
            best_pure_record is None
            or record["target_coverage_gain"] > best_pure_record["target_coverage_gain"]
        ):
            best_pure_record = record
            selected_pure = float(pure_shrinkage)
            best_parameters = candidate

    best_combined_record: dict[str, Any] | None = None
    for combined_shrinkage in (0.25, 0.5, 0.75, 1.0):
        candidate, diagnostics = candidate_diagnostics(
            selected_pure,
            combined_shrinkage,
        )
        summary = _effect_shift_selection_summary(diagnostics)
        record = build_record(
            branch="combined_shift",
            pure_shrinkage=selected_pure,
            combined_shrinkage=combined_shrinkage,
            summary=summary,
        )
        candidate_records.append(record)
        if record["accepted"] and (
            best_combined_record is None
            or record["target_coverage_gain"]
            > best_combined_record["target_coverage_gain"]
        ):
            best_combined_record = record
            selected_combined = float(combined_shrinkage)
            best_parameters = candidate

    selected_parameters, selected_diagnostics = candidate_diagnostics(
        selected_pure,
        selected_combined,
    )
    selected_summary = _effect_shift_selection_summary(selected_diagnostics)
    selected_record = {
        "pure_shrinkage": float(selected_pure),
        "combined_shrinkage": float(selected_combined),
        "pure_effect_accepted": best_pure_record is not None,
        "combined_shift_accepted": best_combined_record is not None,
        "domain_coverage_gains": {
            label: float(
                coverage - baseline_summary["domain_coverages"].get(label, 0.0)
            )
            for label, coverage in selected_summary["domain_coverages"].items()
        },
        "in_domain_gate_delta": _effect_shift_gate_delta(
            baseline_summary["in_domain_gate"],
            selected_summary["in_domain_gate"],
        ),
        **selected_summary,
    }
    best_parameters = selected_parameters
    diagnostics = {
        "kind": "post_fit_independent_effect_shift_head_selection",
        "thresholds": thresholds,
        "baseline": baseline_summary,
        "selected": selected_record,
        "candidates": candidate_records,
    }
    return best_parameters, diagnostics


def _shrink_effect_shift_head(
    parameters: tuple[float, ...],
    *,
    n_covariates: int,
    pure_shrinkage: float,
    combined_shrinkage: float,
) -> tuple[float, ...]:
    """Return parameters with only effect-shift head amplitudes shrunk."""
    head_start = 15 + int(n_covariates)
    if len(parameters) < head_start + _OOD_EFFECT_SHIFT_HEAD_BASE_PARAMETER_COUNT:
        return parameters
    values = list(parameters)
    pure_bounded = float(np.clip(pure_shrinkage, 0.0, 1.0))
    combined_bounded = float(np.clip(combined_shrinkage, 0.0, 1.0))
    values[head_start + 3] = float(values[head_start + 3]) * pure_bounded
    values[head_start + 7] = float(values[head_start + 7]) * combined_bounded
    if len(parameters) >= head_start + _OOD_EFFECT_SHIFT_HEAD_PARAMETER_COUNT:
        for index in (8, 9, 10):
            values[head_start + index] = (
                float(values[head_start + index]) * pure_bounded
            )
        for index in (11, 12, 13):
            values[head_start + index] = (
                float(values[head_start + index]) * combined_bounded
            )
    return tuple(float(value) for value in values)


def _effect_shift_gate_delta(
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, float]:
    """Return candidate-minus-baseline in-domain gate penalty deltas."""
    keys = (
        "mean_group_loss",
        "max_group_loss",
        "extra_inflation_over_1_05_loss",
        "max_group_extra_inflation_cap_loss",
    )
    return {
        key: float(candidate.get(key, 0.0) - baseline.get(key, 0.0)) for key in keys
    }


def _effect_shift_gate_delta_ok(
    delta: dict[str, float],
    thresholds: dict[str, float],
) -> bool:
    """Return whether in-domain penalty increases stay within explicit limits."""
    return (
        delta["extra_inflation_over_1_05_loss"]
        <= thresholds["max_extra_over_1_05_loss_increase"]
        and delta["max_group_extra_inflation_cap_loss"]
        <= thresholds["max_group_extra_cap_loss_increase"]
        and delta["mean_group_loss"] <= thresholds["max_mean_group_loss_increase"]
        and delta["max_group_loss"] <= thresholds["max_max_group_loss_increase"]
    )


def _effect_shift_selection_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Summarize OOD and gate diagnostics for post-fit head selection."""
    domains = [
        domain
        for domain in diagnostics.get("domains", [])
        if domain.get("label") in {"effect_size_shift", "combined_shift"}
    ]
    coverages = [float(domain.get("coverage", 0.0)) for domain in domains]
    gate = domains[0].get("in_domain_gate", {}) if domains else {}
    return {
        "domain_coverages": {
            str(domain.get("label")): float(domain.get("coverage", 0.0))
            for domain in domains
        },
        "mean_ood_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "worst_ood_domain_coverage": float(np.min(coverages)) if coverages else 0.0,
        "in_domain_gate": {
            "overall_coverage": float(gate.get("overall_coverage", 0.0)),
            "mean_group_loss": float(gate.get("mean_group_loss", 0.0)),
            "max_group_loss": float(gate.get("max_group_loss", 0.0)),
            "extra_inflation_over_1_05_loss": float(
                gate.get("extra_inflation_over_1_05_loss", 0.0)
            ),
            "max_group_extra_inflation_cap_loss": float(
                gate.get("max_group_extra_inflation_cap_loss", 0.0)
            ),
        },
    }


def _select_combined_shift_scale_head(
    *,
    batches: Sequence[ConditionalBetaOODCalibrationBatch],
    parameters: tuple[float, ...],
    location: np.ndarray,
    feature_scale: np.ndarray,
    feature_names: tuple[str, ...],
    support_lower: np.ndarray,
    support_upper: np.ndarray,
    support_precision: np.ndarray,
    support_radius: float,
    mean_magnitude_location: float,
    mean_magnitude_scale: float,
    mean_magnitude_lower: float,
    mean_magnitude_upper: float,
    normalization: float,
    global_multiplier: float,
    fallback_strength: float,
    fitted_weights: np.ndarray,
    mean_bias_correction: tuple[tuple[float, ...], ...],
    rank_centering_offsets: tuple[tuple[float, ...], ...],
    base_scale_stratum_offsets: tuple[float, ...],
    post_scale_log_offsets: tuple[float, float, float],
    post_scale_support_threshold: float,
    post_scale_support_width: float,
    post_scale_community_threshold: float,
    post_scale_community_width: float,
    distribution: str,
    n_covariates: int,
    n_species: int,
    prevalence_edges: tuple[float, float],
    in_domain_signed_error: np.ndarray,
    in_domain_final_multiplier: np.ndarray,
    in_domain_support_excess: np.ndarray,
    in_domain_effect_signal: np.ndarray,
    in_domain_design_signal: np.ndarray,
    in_domain_community_occupancy: np.ndarray,
    in_domain_log_ood_inflation: np.ndarray,
    in_domain_rank_groups: list[np.ndarray],
    rank_mean_tolerance: float,
    rank_variance_tolerance: float,
    nominal_level: float,
    z_value: float,
    min_multiplier: float,
    max_multiplier: float,
    max_ood_multiplier: float,
) -> tuple[
    float,
    tuple[float, float, float],
    tuple[float, float],
    float,
    float,
    dict[str, Any],
]:
    """Select a domain-specific combined-shift scale head with held-out gates."""
    thresholds: dict[str, float] = {
        "combined_shift_coverage_floor": 0.90,
        "min_combined_shift_coverage_gain": 0.005,
        "max_extra_over_1_05_loss_increase": 0.40,
        "max_group_extra_cap_loss_increase": 0.12,
        "max_mean_group_loss_increase": 0.01,
        "max_max_group_loss_increase": 0.04,
        "max_in_domain_context_gate_mean": 0.74,
        "max_in_domain_context_gate_active_fraction": 0.70,
    }
    combined_batches = tuple(
        batch for batch in batches if "combined_shift" in str(batch.label)
    )
    if not combined_batches:
        return (
            0.0,
            (0.0, 0.0, 0.0),
            (0.25, 1.0),
            0.0,
            0.0,
            {
                "kind": "context_gated_combined_shift_scale_selection",
                "selected_log_amplitude": 0.0,
                "accepted": False,
                "reason": "no_combined_shift_batches",
                "thresholds": thresholds,
                "candidates": [],
            },
        )

    effect_bin_edges = (0.25, 1.0)

    def candidate_diagnostics(
        log_amplitude: float,
        effect_bin_log_amplitudes: tuple[float, float, float],
        context_gate_strength: float,
        context_gate_intercept: float,
    ) -> dict[str, Any]:
        combined_multiplier = _combined_shift_scale_multiplier(
            support_excess=in_domain_support_excess,
            effect_signal=in_domain_effect_signal,
            design_signal=in_domain_design_signal,
            community_occupancy=in_domain_community_occupancy,
            log_amplitude=log_amplitude,
            effect_bin_log_amplitudes=effect_bin_log_amplitudes,
            effect_bin_edges=effect_bin_edges,
            context_gate_strength=context_gate_strength,
            context_gate_intercept=context_gate_intercept,
        )
        diagnostics = _ood_final_multiplier_diagnostics(
            batches=batches,
            location=location,
            feature_scale=feature_scale,
            feature_names=feature_names,
            support_lower=support_lower,
            support_upper=support_upper,
            support_precision=support_precision,
            support_radius=support_radius,
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
            normalization=normalization,
            global_multiplier=global_multiplier,
            fallback_strength=fallback_strength,
            fitted_weights=fitted_weights,
            mean_bias_correction=mean_bias_correction,
            rank_centering_offsets=rank_centering_offsets,
            base_scale_stratum_offsets=base_scale_stratum_offsets,
            ood_inflation_parameters=parameters,
            post_scale_log_offsets=post_scale_log_offsets,
            post_scale_support_threshold=post_scale_support_threshold,
            post_scale_support_width=post_scale_support_width,
            post_scale_community_threshold=post_scale_community_threshold,
            post_scale_community_width=post_scale_community_width,
            combined_shift_scale_log_amplitude=log_amplitude,
            combined_shift_scale_effect_bin_log_amplitudes=(effect_bin_log_amplitudes),
            combined_shift_scale_effect_bin_edges=effect_bin_edges,
            combined_shift_scale_context_gate_strength=context_gate_strength,
            combined_shift_scale_context_gate_intercept=context_gate_intercept,
            distribution=distribution,
            n_covariates=n_covariates,
            n_species=n_species,
            prevalence_edges=prevalence_edges,
            in_domain_signed_error=in_domain_signed_error,
            in_domain_final_multiplier=np.clip(
                in_domain_final_multiplier * combined_multiplier,
                min_multiplier,
                max_multiplier,
            ),
            in_domain_log_ood_inflation=(
                in_domain_log_ood_inflation + np.log(combined_multiplier)
            ),
            in_domain_rank_groups=in_domain_rank_groups,
            rank_mean_tolerance=rank_mean_tolerance,
            rank_variance_tolerance=rank_variance_tolerance,
            nominal_level=nominal_level,
            z_value=z_value,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            max_ood_multiplier=max_ood_multiplier,
        )
        return diagnostics

    zero_bin_amplitudes = (0.0, 0.0, 0.0)
    baseline_diagnostics = candidate_diagnostics(0.0, zero_bin_amplitudes, 0.0, 0.0)
    baseline_summary = _effect_shift_selection_summary(baseline_diagnostics)
    candidate_records: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    best_log_amplitude = 0.0
    best_bin_amplitudes = zero_bin_amplitudes
    best_context_gate_strength = 0.0
    best_context_gate_intercept = 0.0
    amplitude_grid = tuple(
        float(value)
        for value in np.linspace(0.0, _COMBINED_SHIFT_SCALE_MAX_LOG, 13)[1:]
    )
    pattern_builders = {
        "scalar_selective": lambda value: (value, (0.0, 0.0, 0.0)),
        "high_effect": lambda value: (0.0, (0.0, 0.0, value)),
        "mid_high_effect": lambda value: (0.0, (0.0, 0.5 * value, value)),
        "ranked_effect": lambda value: (0.0, (0.25 * value, 0.6 * value, value)),
        "all_effect_bins": lambda value: (0.0, (value, value, value)),
    }
    context_gate_grid = (
        ("legacy_product", 0.0, 0.0),
        ("context_moderate", 1.0, 1.0),
        ("context_strict", 1.0, 2.0),
    )
    for pattern_name, build_pattern in pattern_builders.items():
        for context_name, context_strength, context_intercept in context_gate_grid:
            for raw_amplitude in amplitude_grid:
                log_amplitude, bin_amplitudes = build_pattern(raw_amplitude)
                bin_amplitudes = tuple(float(value) for value in bin_amplitudes)
                diagnostics = candidate_diagnostics(
                    float(log_amplitude),
                    bin_amplitudes,
                    context_strength,
                    context_intercept,
                )
                summary = _effect_shift_selection_summary(diagnostics)
                domain_coverages = summary["domain_coverages"]
                combined_coverage = float(domain_coverages.get("combined_shift", 0.0))
                coverage_gain = combined_coverage - float(
                    baseline_summary["domain_coverages"].get("combined_shift", 0.0)
                )
                gate_delta = _effect_shift_gate_delta(
                    baseline_summary["in_domain_gate"],
                    summary["in_domain_gate"],
                )
                in_domain_context_gate = _combined_shift_context_gate_summary(
                    support_excess=in_domain_support_excess,
                    effect_signal=in_domain_effect_signal,
                    design_signal=in_domain_design_signal,
                    community_occupancy=in_domain_community_occupancy,
                    strength=context_strength,
                    intercept=context_intercept,
                )
                overlap_ok = (
                    in_domain_context_gate["mean"]
                    <= thresholds["max_in_domain_context_gate_mean"]
                    and in_domain_context_gate["active_fraction_over_0_8"]
                    <= thresholds["max_in_domain_context_gate_active_fraction"]
                )
                accepted = (
                    combined_coverage >= thresholds["combined_shift_coverage_floor"]
                    and coverage_gain >= thresholds["min_combined_shift_coverage_gain"]
                    and _effect_shift_gate_delta_ok(gate_delta, thresholds)
                    and overlap_ok
                )
                record = {
                    "pattern": pattern_name,
                    "context_pattern": context_name,
                    "log_amplitude": float(log_amplitude),
                    "multiplier": float(np.exp(float(log_amplitude))),
                    "effect_bin_edges": [float(value) for value in effect_bin_edges],
                    "effect_bin_log_amplitudes": [
                        float(value) for value in bin_amplitudes
                    ],
                    "effect_bin_multipliers": [
                        float(np.exp(float(value))) for value in bin_amplitudes
                    ],
                    "context_gate_strength": float(context_strength),
                    "context_gate_intercept": float(context_intercept),
                    "in_domain_context_gate": in_domain_context_gate,
                    "context_overlap_ok": bool(overlap_ok),
                    "accepted": bool(accepted),
                    "combined_shift_coverage": combined_coverage,
                    "combined_shift_coverage_gain": float(coverage_gain),
                    "domain_coverages": domain_coverages,
                    "in_domain_gate_delta": gate_delta,
                    "in_domain_gate": summary["in_domain_gate"],
                    "mean_ood_coverage": summary["mean_ood_coverage"],
                    "worst_ood_domain_coverage": summary["worst_ood_domain_coverage"],
                }
                candidate_records.append(record)
                if record["accepted"] and (
                    best_record is None
                    or record["combined_shift_coverage"]
                    > best_record["combined_shift_coverage"]
                    or (
                        record["combined_shift_coverage"]
                        == best_record["combined_shift_coverage"]
                        and max(bin_amplitudes + (float(log_amplitude),))
                        < max(
                            tuple(best_record.get("effect_bin_log_amplitudes", ()))
                            + (float(best_record.get("log_amplitude", 0.0)),)
                        )
                    )
                ):
                    best_record = record
                    best_log_amplitude = float(log_amplitude)
                    best_bin_amplitudes = bin_amplitudes
                    best_context_gate_strength = float(context_strength)
                    best_context_gate_intercept = float(context_intercept)

    selected_diagnostics = candidate_diagnostics(
        best_log_amplitude,
        best_bin_amplitudes,
        best_context_gate_strength,
        best_context_gate_intercept,
    )
    selected_summary = _effect_shift_selection_summary(selected_diagnostics)
    selected_record = {
        "log_amplitude": float(best_log_amplitude),
        "multiplier": float(np.exp(best_log_amplitude)),
        "effect_bin_edges": [float(value) for value in effect_bin_edges],
        "effect_bin_log_amplitudes": [float(value) for value in best_bin_amplitudes],
        "effect_bin_multipliers": [
            float(np.exp(float(value))) for value in best_bin_amplitudes
        ],
        "context_gate_strength": float(best_context_gate_strength),
        "context_gate_intercept": float(best_context_gate_intercept),
        "in_domain_context_gate": _combined_shift_context_gate_summary(
            support_excess=in_domain_support_excess,
            effect_signal=in_domain_effect_signal,
            design_signal=in_domain_design_signal,
            community_occupancy=in_domain_community_occupancy,
            strength=best_context_gate_strength,
            intercept=best_context_gate_intercept,
        ),
        "accepted": best_record is not None,
        "domain_coverages": selected_summary["domain_coverages"],
        "combined_shift_coverage_gain": float(
            selected_summary["domain_coverages"].get("combined_shift", 0.0)
            - baseline_summary["domain_coverages"].get("combined_shift", 0.0)
        ),
        "in_domain_gate_delta": _effect_shift_gate_delta(
            baseline_summary["in_domain_gate"],
            selected_summary["in_domain_gate"],
        ),
        "in_domain_gate": selected_summary["in_domain_gate"],
        "mean_ood_coverage": selected_summary["mean_ood_coverage"],
        "worst_ood_domain_coverage": selected_summary["worst_ood_domain_coverage"],
    }
    return (
        best_log_amplitude,
        best_bin_amplitudes,
        effect_bin_edges,
        best_context_gate_strength,
        best_context_gate_intercept,
        {
            "kind": "context_gated_combined_shift_scale_selection",
            "thresholds": thresholds,
            "activation": {
                "support_center": _COMBINED_SHIFT_SCALE_SUPPORT_CENTER,
                "support_width": _COMBINED_SHIFT_SCALE_SUPPORT_WIDTH,
                "effect_center": _COMBINED_SHIFT_SCALE_EFFECT_CENTER,
                "effect_width": _COMBINED_SHIFT_SCALE_EFFECT_WIDTH,
                "low_design_center": _COMBINED_SHIFT_SCALE_LOW_DESIGN_CENTER,
                "low_design_width": _COMBINED_SHIFT_SCALE_LOW_DESIGN_WIDTH,
                "low_community_center": _COMBINED_SHIFT_SCALE_LOW_COMMUNITY_CENTER,
                "low_community_width": _COMBINED_SHIFT_SCALE_LOW_COMMUNITY_WIDTH,
                "context_gate_grid": [
                    {
                        "name": name,
                        "strength": float(strength),
                        "intercept": float(intercept),
                    }
                    for name, strength, intercept in context_gate_grid
                ],
            },
            "baseline": baseline_summary,
            "selected": selected_record,
            "candidates": candidate_records,
        },
    )


def _effect_quantile_coverage(
    *,
    effect_signal: np.ndarray,
    covered: np.ndarray,
    final_multiplier: np.ndarray,
) -> list[dict[str, float | int | str]]:
    effect = np.asarray(effect_signal, dtype=float).reshape(-1)
    is_covered = np.asarray(covered, dtype=bool).reshape(-1)
    multiplier = np.asarray(final_multiplier, dtype=float).reshape(-1)
    if effect.size == 0:
        return []
    edges = np.quantile(effect, (0.0, 0.25, 0.5, 0.75, 1.0))
    rows: list[dict[str, float | int | str]] = []
    for index in range(4):
        if index == 3:
            mask = (effect >= edges[index]) & (effect <= edges[index + 1])
        else:
            mask = (effect >= edges[index]) & (effect < edges[index + 1])
        if np.count_nonzero(mask) == 0:
            continue
        rows.append(
            {
                "label": f"q{index + 1}",
                "n_observations": int(np.count_nonzero(mask)),
                "effect_signal_min": float(np.min(effect[mask])),
                "effect_signal_max": float(np.max(effect[mask])),
                "effect_signal_mean": float(np.mean(effect[mask])),
                "coverage": float(np.mean(is_covered[mask])),
                "final_multiplier_mean": float(np.mean(multiplier[mask])),
            }
        )
    return rows


def _effect_quantile_group_masks(effect_signal: np.ndarray) -> list[np.ndarray]:
    """Return nonempty effect-signal quantile masks for OOD coverage gates."""
    effect = np.asarray(effect_signal, dtype=float).reshape(-1)
    if effect.size < 2:
        return []
    edges = np.quantile(effect, (0.0, 0.25, 0.5, 0.75, 1.0))
    groups: list[np.ndarray] = []
    for index in range(4):
        if index == 3:
            mask = (effect >= edges[index]) & (effect <= edges[index + 1])
        else:
            mask = (effect >= edges[index]) & (effect < edges[index + 1])
        if np.count_nonzero(mask) >= 2:
            groups.append(mask)
    return groups


def _in_domain_gate_diagnostics(
    *,
    signed_error: np.ndarray,
    final_multiplier: np.ndarray,
    log_ood_inflation: np.ndarray,
    rank_groups: list[np.ndarray],
    rank_mean_tolerance: float,
    rank_variance_tolerance: float,
    z_value: float,
) -> dict[str, Any]:
    signed = np.asarray(signed_error, dtype=float).reshape(-1)
    multiplier = np.asarray(final_multiplier, dtype=float).reshape(-1)
    log_extra = np.asarray(log_ood_inflation, dtype=float).reshape(-1)
    ranks = ndtr(signed / multiplier)
    covered = np.abs(signed) <= z_value * multiplier
    expected_rank_variance = 1.0 / 12.0
    group_rows = []
    for index, group in enumerate(rank_groups):
        group_mask = np.asarray(group, dtype=bool).reshape(-1)
        if group_mask.shape != signed.shape or np.count_nonzero(group_mask) < 2:
            continue
        selected_ranks = ranks[group_mask]
        selected_covered = covered[group_mask]
        rank_mean = float(np.mean(selected_ranks))
        rank_variance = float(np.var(selected_ranks))
        coverage = float(np.mean(selected_covered))
        rank_mean_loss = max(0.0, abs(rank_mean - 0.5) / rank_mean_tolerance - 1.0) ** 2
        rank_variance_loss = (
            max(
                0.0,
                abs(rank_variance - expected_rank_variance) / rank_variance_tolerance
                - 1.0,
            )
            ** 2
        )
        coverage_loss = max(0.0, (0.925 - coverage) / 0.05) ** 2
        extra_cap_loss = float(
            np.mean(np.square(np.maximum(log_extra[group_mask] - np.log(1.04), 0.0)))
        )
        group_rows.append(
            {
                "label": f"group_{index}",
                "n_observations": int(np.count_nonzero(group_mask)),
                "rank_mean": rank_mean,
                "rank_variance": rank_variance,
                "coverage": coverage,
                "rank_mean_loss": float(rank_mean_loss),
                "rank_variance_loss": float(rank_variance_loss),
                "coverage_loss": float(coverage_loss),
                "extra_inflation_cap_loss": extra_cap_loss,
            }
        )
    group_losses = [
        row["rank_mean_loss"] + row["rank_variance_loss"] + row["coverage_loss"]
        for row in group_rows
    ]
    overall_coverage = float(np.mean(covered))
    extra_over_105_loss = float(
        np.mean(np.square(np.maximum((log_extra - np.log(1.05)) / np.log(1.25), 0.0)))
    )
    return {
        "overall_coverage": overall_coverage,
        "overall_coverage_loss": float(max(0.0, (0.90 - overall_coverage) / 0.05) ** 2),
        "mean_group_loss": float(np.mean(group_losses)) if group_losses else 0.0,
        "max_group_loss": float(np.max(group_losses)) if group_losses else 0.0,
        "extra_inflation_over_1_05_loss": extra_over_105_loss,
        "max_group_extra_inflation_cap_loss": (
            float(np.max([row["extra_inflation_cap_loss"] for row in group_rows]))
            if group_rows
            else 0.0
        ),
        "groups": group_rows,
    }


def _fit_ood_inflation_parameters(
    batches: Sequence[ConditionalBetaOODCalibrationBatch],
    *,
    location: np.ndarray,
    feature_scale: np.ndarray,
    feature_names: tuple[str, ...],
    support_lower: np.ndarray,
    support_upper: np.ndarray,
    support_precision: np.ndarray,
    support_radius: float,
    mean_magnitude_location: float,
    mean_magnitude_scale: float,
    mean_magnitude_lower: float,
    mean_magnitude_upper: float,
    normalization: float,
    global_multiplier: float,
    fallback_strength: float,
    fitted_weights: np.ndarray,
    mean_bias_correction: tuple[tuple[float, ...], ...],
    rank_centering_offsets: tuple[tuple[float, ...], ...],
    base_scale_stratum_offsets: tuple[float, ...],
    in_domain_signed_error: np.ndarray,
    in_domain_adjustment: np.ndarray,
    in_domain_trust: np.ndarray,
    in_domain_support_excess: np.ndarray,
    in_domain_effect_signal: np.ndarray,
    in_domain_design_signal: np.ndarray,
    in_domain_prevalence_stratum: np.ndarray,
    in_domain_design_stratum: np.ndarray,
    in_domain_coefficient_stratum: np.ndarray,
    in_domain_rank_groups: list[np.ndarray],
    distribution: str,
    n_covariates: int,
    n_species: int,
    min_multiplier: float,
    max_multiplier: float,
    max_ood_multiplier: float,
    objective_weight: float,
    in_domain_gate_weight: float,
    epochs: int,
    learning_rate: float,
    gate_effect_branch: bool,
    prevalence_edges: tuple[float, float],
    rank_mean_tolerance: float,
    rank_variance_tolerance: float,
    nominal_level: float,
    z_value: float,
    initial_parameters: tuple[float, ...] | None = None,
    final_multiplier_aware: bool = False,
    post_scale_log_offsets: tuple[float, float, float] = (0.0, 0.0, 0.0),
    post_scale_support_threshold: float = 0.0,
    post_scale_support_width: float = 1.0,
    post_scale_community_threshold: float = 0.0,
    post_scale_community_width: float = 1.0,
    in_domain_community_occupancy: np.ndarray | None = None,
    trust_region_parameters: tuple[float, ...] | None = None,
    trust_region_weight: float = 0.0,
    trust_region_log_tolerance: float = 0.0,
    overlap_penalty_domain: str | None = None,
    overlap_penalty_weight: float = 0.0,
    overlap_log_tolerance: float = 0.0,
    bin_in_domain_penalty_weight: float = 0.0,
    expert_target_coverage_weight: float = 0.0,
    expert_effect_quantile_weight: float = 0.0,
    expert_margin_weight: float = 0.0,
) -> tuple[tuple[float, ...], float, float, float, int, tuple[str, ...]]:
    """Fit a learned support-excess inflation curve from held-out OOD batches."""
    if max_ood_multiplier <= 1.0:
        raise ValueError(
            "learned OOD inflation requires max multiplier greater than one"
        )

    domain_arrays = []
    labels = []
    n_observations = 0
    for batch in batches:
        if batch.weight <= 0.0:
            raise ValueError("OOD calibration batch weights must be positive")
        mean, scale, design, response = _validated_arrays(
            batch.posterior, X=batch.X, Y=batch.Y
        )
        if mean.shape[1] != n_covariates or mean.shape[2] != n_species:
            raise ValueError("OOD calibration batch domain does not match calibration")
        truth = np.asarray(batch.beta_true, dtype=float)
        if truth.shape != mean.shape or not np.all(np.isfinite(truth)):
            raise ValueError("OOD beta_true must be finite and match posterior shape")
        prevalence = _prevalence(response)
        prevalence_by_coefficient = np.broadcast_to(prevalence[:, None, :], mean.shape)
        prevalence_stratum = _prevalence_stratum_index(
            prevalence_by_coefficient,
            prevalence_edges=prevalence_edges,
        )
        coefficient_stratum = _coefficient_stratum_index(mean.shape)
        if mean_bias_correction:
            bias_values = np.asarray(mean_bias_correction, dtype=float)
            mean = mean + bias_values[prevalence_stratum, coefficient_stratum]
        if rank_centering_offsets:
            centering_values = np.asarray(rank_centering_offsets, dtype=float)
            mean = (
                mean + centering_values[prevalence_stratum, coefficient_stratum] * scale
            )
        raw_features = _raw_features(
            mean=mean,
            scale=scale,
            X=design,
            Y=response,
            distribution=distribution,
        )
        design_matrix, names = _structured_design(
            raw_features,
            location=location,
            scale=feature_scale,
            n_covariates=n_covariates,
        )
        if names != feature_names:
            raise ValueError("OOD calibration feature specification mismatch")
        adjustment = np.exp(
            np.clip(design_matrix @ fitted_weights, -20.0, 20.0)
        ).reshape(mean.shape)
        trust = _support_trust(
            raw_features,
            location=location,
            scale=feature_scale,
            lower=support_lower,
            upper=support_upper,
            precision=support_precision,
            radius=support_radius,
            fallback_strength=fallback_strength,
            mean_magnitude=np.log1p(np.abs(mean)),
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
        )
        support_excess = _support_excess(
            raw_features,
            location=location,
            scale=feature_scale,
            lower=support_lower,
            upper=support_upper,
            precision=support_precision,
            radius=support_radius,
            mean_magnitude=np.log1p(np.abs(mean)),
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
            mean_magnitude_lower=mean_magnitude_lower,
            mean_magnitude_upper=mean_magnitude_upper,
        )
        effect_signal = _effect_size_signal(
            np.log1p(np.abs(mean)),
            mean_magnitude_location=mean_magnitude_location,
            mean_magnitude_scale=mean_magnitude_scale,
        )
        design_signal = _design_information_signal(
            raw_features,
            location=location,
            scale=feature_scale,
        )
        design_stratum = _design_information_stratum_index(design_signal)
        adjustment = adjustment * np.exp(
            _base_scale_stratum_log_offset(
                offsets=base_scale_stratum_offsets,
                prevalence_stratum=prevalence_stratum,
                design_stratum=design_stratum,
                coefficient_stratum=coefficient_stratum,
                n_covariates=n_covariates,
            )
        )
        base_multiplier = _blend_with_scalar_fallback(
            adjustment,
            trust,
            support_excess=support_excess,
            effect_signal=effect_signal,
            design_signal=design_signal,
            prevalence_stratum=prevalence_stratum,
            design_stratum=design_stratum,
            coefficient_stratum=coefficient_stratum,
            normalization=normalization,
            global_multiplier=global_multiplier,
            ood_uncertainty_strength=0.0,
            ood_uncertainty_max_multiplier=1.0,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
        signed_error = (truth - mean) / scale
        rank_groups = _prevalence_group_masks(
            prevalence_by_coefficient.reshape(-1),
            prevalence_edges=prevalence_edges,
        )
        effect_quantile_groups = _effect_quantile_group_masks(effect_signal.reshape(-1))
        community_occupancy = _community_occupancy_array(response, mean.shape)
        if final_multiplier_aware:
            post_scale_multiplier = _rare_validation_scale_multiplier(
                log_offsets=post_scale_log_offsets,
                prevalence_stratum=prevalence_stratum,
                design_stratum=design_stratum,
                support_excess=support_excess,
                community_occupancy=community_occupancy,
                support_threshold=post_scale_support_threshold,
                support_width=post_scale_support_width,
                community_threshold=post_scale_community_threshold,
                community_width=post_scale_community_width,
            )
        else:
            post_scale_multiplier = np.ones_like(base_multiplier, dtype=float)
        label = str(batch.label)
        is_combined_shift = "combined_shift" in label
        is_effect_size_shift = "effect_size_shift" in label
        if final_multiplier_aware and is_combined_shift:
            coverage_floor = 0.90
            coverage_gate_weight = 2.0
            combined_direct_weight = _COMBINED_SHIFT_OBJECTIVE_COVERAGE_WEIGHT
            combined_quantile_weight = _COMBINED_SHIFT_OBJECTIVE_QUANTILE_WEIGHT
            combined_context_weight = _COMBINED_SHIFT_OBJECTIVE_CONTEXT_WEIGHT
        elif final_multiplier_aware and is_effect_size_shift:
            coverage_floor = 0.90
            coverage_gate_weight = 2.0
            combined_direct_weight = 0.0
            combined_quantile_weight = 0.0
            combined_context_weight = 0.0
        else:
            coverage_floor = float(nominal_level)
            coverage_gate_weight = 0.0
            combined_direct_weight = 0.0
            combined_quantile_weight = 0.0
            combined_context_weight = 0.0
        domain_arrays.append(
            {
                "label": label,
                "signed_error": signed_error.reshape(-1),
                "base_multiplier": base_multiplier.reshape(-1),
                "post_scale_multiplier": post_scale_multiplier.reshape(-1),
                "support_excess": support_excess.reshape(-1),
                "effect_signal": effect_signal.reshape(-1),
                "design_signal": design_signal.reshape(-1),
                "community_occupancy": community_occupancy.reshape(-1),
                "prevalence_stratum": prevalence_stratum.reshape(-1),
                "design_stratum": design_stratum.reshape(-1),
                "coefficient_stratum": coefficient_stratum.reshape(-1),
                "rank_groups": rank_groups,
                "effect_quantile_groups": effect_quantile_groups,
                "coverage_floor": float(coverage_floor),
                "coverage_gate_weight": float(coverage_gate_weight),
                "combined_direct_weight": float(combined_direct_weight),
                "combined_quantile_weight": float(combined_quantile_weight),
                "combined_context_weight": float(combined_context_weight),
                "expert_target_coverage_weight": float(
                    expert_target_coverage_weight
                    if overlap_penalty_domain is not None
                    and overlap_penalty_domain in label
                    else 0.0
                ),
                "expert_effect_quantile_weight": float(
                    expert_effect_quantile_weight
                    if overlap_penalty_domain is not None
                    and overlap_penalty_domain in label
                    else 0.0
                ),
                "expert_margin_weight": float(
                    expert_margin_weight
                    if overlap_penalty_domain is not None
                    and overlap_penalty_domain in label
                    else 0.0
                ),
                "weight": float(batch.weight),
            }
        )
        labels.append(label)
        n_observations += int(signed_error.size)

    in_domain_base_multiplier = _blend_with_scalar_fallback(
        in_domain_adjustment,
        in_domain_trust,
        support_excess=in_domain_support_excess,
        effect_signal=in_domain_effect_signal,
        design_signal=in_domain_design_signal,
        prevalence_stratum=in_domain_prevalence_stratum,
        design_stratum=in_domain_design_stratum,
        coefficient_stratum=in_domain_coefficient_stratum,
        normalization=normalization,
        global_multiplier=global_multiplier,
        ood_uncertainty_strength=0.0,
        ood_uncertainty_max_multiplier=1.0,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
    )
    if final_multiplier_aware:
        in_domain_post_scale_multiplier = _rare_validation_scale_multiplier(
            log_offsets=post_scale_log_offsets,
            prevalence_stratum=in_domain_prevalence_stratum,
            design_stratum=in_domain_design_stratum,
            support_excess=in_domain_support_excess,
            community_occupancy=(
                np.zeros_like(in_domain_support_excess, dtype=float)
                if in_domain_community_occupancy is None
                else in_domain_community_occupancy
            ),
            support_threshold=post_scale_support_threshold,
            support_width=post_scale_support_width,
            community_threshold=post_scale_community_threshold,
            community_width=post_scale_community_width,
        )
    else:
        in_domain_post_scale_multiplier = np.ones_like(
            in_domain_base_multiplier, dtype=float
        )

    initial = tuple(float(value) for value in (initial_parameters or ()))
    offset = tf.Variable(initial[0] if len(initial) >= 5 else -4.0, dtype=tf.float64)
    raw_support_linear = tf.Variable(
        _softplus_inverse(max(initial[1], 1e-8) if len(initial) >= 5 else 1e-3),
        dtype=tf.float64,
    )
    raw_support_quadratic = tf.Variable(
        _softplus_inverse(max(initial[2], 1e-8) if len(initial) >= 5 else 0.75),
        dtype=tf.float64,
    )
    raw_effect_linear = tf.Variable(
        _softplus_inverse(max(initial[3], 1e-8) if len(initial) >= 5 else 1e-3),
        dtype=tf.float64,
    )
    raw_effect_quadratic = tf.Variable(
        _softplus_inverse(max(initial[4], 1e-8) if len(initial) >= 5 else 0.1),
        dtype=tf.float64,
    )
    effect_gate_intercept = tf.Variable(
        initial[5] if len(initial) >= 9 else -4.0, dtype=tf.float64
    )
    raw_effect_gate_support_linear = tf.Variable(
        _softplus_inverse(max(initial[6], 1e-8) if len(initial) >= 9 else 2.0),
        dtype=tf.float64,
    )
    raw_effect_gate_effect_linear = tf.Variable(
        _softplus_inverse(max(initial[7], 1e-8) if len(initial) >= 9 else 4.0),
        dtype=tf.float64,
    )
    raw_effect_high_design_suppression = tf.Variable(
        _softplus_inverse(max(initial[8], 1e-8) if len(initial) >= 9 else 1.0),
        dtype=tf.float64,
    )
    prevalence_gate_offsets = tf.Variable(
        np.asarray(
            initial[9:12] if len(initial) >= 12 else np.zeros(3, dtype=np.float64),
            dtype=np.float64,
        )
    )
    design_gate_offsets = tf.Variable(
        np.asarray(
            initial[12:15] if len(initial) >= 15 else np.zeros(3, dtype=np.float64),
            dtype=np.float64,
        )
    )
    coefficient_initial = np.zeros(n_covariates, dtype=np.float64)
    if len(initial) > 15:
        coefficient_values = np.asarray(initial[15:], dtype=np.float64)
        coefficient_initial[: min(n_covariates, coefficient_values.size)] = (
            coefficient_values[:n_covariates]
        )
    coefficient_gate_offsets = tf.Variable(
        coefficient_initial,
        dtype=tf.float64,
    )
    head_start = 15 + n_covariates
    head_initial = tuple(
        initial[head_start : head_start + _OOD_EFFECT_SHIFT_HEAD_PARAMETER_COUNT]
    )
    has_head_base = len(head_initial) >= _OOD_EFFECT_SHIFT_HEAD_BASE_PARAMETER_COUNT
    pure_effect_intercept = tf.Variable(
        head_initial[0] if has_head_base else -3.0,
        dtype=tf.float64,
    )
    raw_pure_effect_linear = tf.Variable(
        _softplus_inverse(max(head_initial[1], 1e-8) if has_head_base else 2.0),
        dtype=tf.float64,
    )
    raw_pure_support_suppression = tf.Variable(
        _softplus_inverse(max(head_initial[2], 1e-8) if has_head_base else 2.0),
        dtype=tf.float64,
    )
    raw_pure_log_amplitude = tf.Variable(
        _softplus_inverse(max(head_initial[3], 1e-8) if has_head_base else 0.05),
        dtype=tf.float64,
    )
    combined_effect_intercept = tf.Variable(
        head_initial[4] if has_head_base else -3.0,
        dtype=tf.float64,
    )
    raw_combined_effect_linear = tf.Variable(
        _softplus_inverse(max(head_initial[5], 1e-8) if has_head_base else 2.0),
        dtype=tf.float64,
    )
    raw_combined_support_linear = tf.Variable(
        _softplus_inverse(max(head_initial[6], 1e-8) if has_head_base else 1.0),
        dtype=tf.float64,
    )
    raw_combined_log_amplitude = tf.Variable(
        _softplus_inverse(max(head_initial[7], 1e-8) if has_head_base else 0.05),
        dtype=tf.float64,
    )
    pure_bin_initial = (
        np.asarray(head_initial[8:11], dtype=np.float64)
        if len(head_initial) >= _OOD_EFFECT_SHIFT_HEAD_PARAMETER_COUNT
        else np.full(3, 0.01, dtype=np.float64)
    )
    combined_bin_initial = (
        np.asarray(head_initial[11:14], dtype=np.float64)
        if len(head_initial) >= _OOD_EFFECT_SHIFT_HEAD_PARAMETER_COUNT
        else np.full(3, 0.01, dtype=np.float64)
    )
    raw_pure_effect_bin_log_amplitudes = tf.Variable(
        np.asarray(
            [_softplus_inverse(max(float(value), 1e-8)) for value in pure_bin_initial],
            dtype=np.float64,
        ),
        dtype=tf.float64,
    )
    raw_combined_effect_bin_log_amplitudes = tf.Variable(
        np.asarray(
            [
                _softplus_inverse(max(float(value), 1e-8))
                for value in combined_bin_initial
            ],
            dtype=np.float64,
        ),
        dtype=tf.float64,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    expected_rank_variance = tf.constant(1.0 / 12.0, dtype=tf.float64)
    target_coverage = tf.constant(float(nominal_level), dtype=tf.float64)

    tf_domains = []
    for arrays in domain_arrays:
        tf_domains.append(
            {
                "signed_error": tf.constant(arrays["signed_error"], dtype=tf.float64),
                "base_multiplier": tf.constant(
                    arrays["base_multiplier"], dtype=tf.float64
                ),
                "post_scale_multiplier": tf.constant(
                    arrays["post_scale_multiplier"], dtype=tf.float64
                ),
                "support_excess": tf.constant(
                    arrays["support_excess"], dtype=tf.float64
                ),
                "effect_signal": tf.constant(arrays["effect_signal"], dtype=tf.float64),
                "design_signal": tf.constant(arrays["design_signal"], dtype=tf.float64),
                "community_occupancy": tf.constant(
                    arrays["community_occupancy"], dtype=tf.float64
                ),
                "prevalence_stratum": tf.constant(
                    arrays["prevalence_stratum"], dtype=tf.int32
                ),
                "design_stratum": tf.constant(arrays["design_stratum"], dtype=tf.int32),
                "coefficient_stratum": tf.constant(
                    arrays["coefficient_stratum"], dtype=tf.int32
                ),
                "rank_groups": [
                    tf.constant(group, dtype=tf.bool) for group in arrays["rank_groups"]
                ],
                "effect_quantile_groups": [
                    tf.constant(group, dtype=tf.bool)
                    for group in arrays["effect_quantile_groups"]
                ],
                "coverage_floor": tf.constant(
                    arrays["coverage_floor"], dtype=tf.float64
                ),
                "coverage_gate_weight": tf.constant(
                    arrays["coverage_gate_weight"], dtype=tf.float64
                ),
                "combined_direct_weight": tf.constant(
                    arrays["combined_direct_weight"], dtype=tf.float64
                ),
                "combined_quantile_weight": tf.constant(
                    arrays["combined_quantile_weight"], dtype=tf.float64
                ),
                "combined_context_weight": tf.constant(
                    arrays["combined_context_weight"], dtype=tf.float64
                ),
                "expert_target_coverage_weight": tf.constant(
                    arrays["expert_target_coverage_weight"], dtype=tf.float64
                ),
                "expert_effect_quantile_weight": tf.constant(
                    arrays["expert_effect_quantile_weight"], dtype=tf.float64
                ),
                "expert_margin_weight": tf.constant(
                    arrays["expert_margin_weight"], dtype=tf.float64
                ),
                "weight": tf.constant(float(arrays["weight"]), dtype=tf.float64),
            }
        )
    in_domain = {
        "signed_error": tf.constant(in_domain_signed_error, dtype=tf.float64),
        "base_multiplier": tf.constant(in_domain_base_multiplier, dtype=tf.float64),
        "post_scale_multiplier": tf.constant(
            in_domain_post_scale_multiplier, dtype=tf.float64
        ),
        "support_excess": tf.constant(in_domain_support_excess, dtype=tf.float64),
        "effect_signal": tf.constant(in_domain_effect_signal, dtype=tf.float64),
        "design_signal": tf.constant(in_domain_design_signal, dtype=tf.float64),
        "community_occupancy": tf.constant(
            (
                np.zeros_like(in_domain_support_excess, dtype=float)
                if in_domain_community_occupancy is None
                else in_domain_community_occupancy
            ),
            dtype=tf.float64,
        ),
        "prevalence_stratum": tf.constant(in_domain_prevalence_stratum, dtype=tf.int32),
        "design_stratum": tf.constant(in_domain_design_stratum, dtype=tf.int32),
        "coefficient_stratum": tf.constant(
            in_domain_coefficient_stratum, dtype=tf.int32
        ),
        "rank_groups": [
            tf.constant(group, dtype=tf.bool) for group in in_domain_rank_groups
        ],
    }
    in_domain_trust_region_baseline = None
    if trust_region_parameters is not None and (
        trust_region_weight > 0.0 or overlap_penalty_weight > 0.0
    ):
        in_domain_trust_region_baseline = tf.constant(
            _learned_ood_log_inflation_numpy(
                in_domain_support_excess,
                effect_signal=in_domain_effect_signal,
                design_signal=in_domain_design_signal,
                community_occupancy=in_domain_community_occupancy,
                prevalence_stratum=in_domain_prevalence_stratum,
                design_stratum=in_domain_design_stratum,
                coefficient_stratum=in_domain_coefficient_stratum,
                parameters=trust_region_parameters,
                max_multiplier=max_ood_multiplier,
            ),
            dtype=tf.float64,
        )

    def localized_overlap_context_for(arrays: dict[str, Any]) -> tf.Tensor:
        """Return fixed in-domain overlap context for a target OOD expert."""
        positive_support = tf.maximum(
            arrays["support_excess"],
            tf.constant(0.0, dtype=tf.float64),
        )
        if overlap_penalty_domain == "effect_size_shift":
            high_effect = tf.sigmoid(
                (arrays["effect_signal"] - tf.constant(0.75, dtype=tf.float64))
                / tf.constant(0.35, dtype=tf.float64)
            )
            support_close = tf.sigmoid(
                (tf.constant(0.25, dtype=tf.float64) - positive_support)
                / tf.constant(0.25, dtype=tf.float64)
            )
            return high_effect * support_close
        if overlap_penalty_domain == "combined_shift":
            support_gate = tf.sigmoid(
                (positive_support - tf.constant(0.20, dtype=tf.float64))
                / tf.constant(0.35, dtype=tf.float64)
            )
            effect_gate = tf.sigmoid(
                (arrays["effect_signal"] - tf.constant(0.25, dtype=tf.float64))
                / tf.constant(0.50, dtype=tf.float64)
            )
            low_design_gate = tf.sigmoid(
                (tf.constant(0.75, dtype=tf.float64) - arrays["design_signal"])
                / tf.constant(0.35, dtype=tf.float64)
            )
            low_community_gate = tf.sigmoid(
                (tf.constant(0.45, dtype=tf.float64) - arrays["community_occupancy"])
                / tf.constant(0.06, dtype=tf.float64)
            )
            return support_gate * effect_gate * low_design_gate * low_community_gate
        return tf.zeros_like(arrays["support_excess"], dtype=tf.float64)

    def effect_bin_basis_for(arrays: dict[str, Any]) -> tf.Tensor:
        """Return smooth effect-bin activations for per-bin gate penalties."""
        return tf.exp(
            -0.5
            * tf.square(
                (
                    tf.expand_dims(arrays["effect_signal"], axis=-1)
                    - tf.constant(_OOD_EFFECT_SHIFT_BIN_CENTERS, dtype=tf.float64)
                )
                / tf.constant(_OOD_EFFECT_SHIFT_BIN_WIDTH, dtype=tf.float64)
            )
        )

    def effect_bin_context_for(arrays: dict[str, Any]) -> tf.Tensor:
        """Return the domain-local context used to cap effect-bin movement."""
        positive_support = tf.maximum(
            arrays["support_excess"],
            tf.constant(0.0, dtype=tf.float64),
        )
        if overlap_penalty_domain == "effect_size_shift":
            support_context = tf.sigmoid(
                (
                    positive_support
                    - tf.constant(
                        _OOD_EFFECT_SHIFT_PURE_CONTEXT_SUPPORT_CENTER,
                        dtype=tf.float64,
                    )
                )
                / tf.constant(
                    _OOD_EFFECT_SHIFT_PURE_CONTEXT_SUPPORT_WIDTH,
                    dtype=tf.float64,
                )
            )
            low_design_context = tf.sigmoid(
                (
                    tf.constant(
                        _OOD_EFFECT_SHIFT_PURE_CONTEXT_LOW_DESIGN_CENTER,
                        dtype=tf.float64,
                    )
                    - arrays["design_signal"]
                )
                / tf.constant(
                    _OOD_EFFECT_SHIFT_PURE_CONTEXT_LOW_DESIGN_WIDTH,
                    dtype=tf.float64,
                )
            )
            context = 1.0 - (1.0 - support_context) * (1.0 - low_design_context)
            return (
                tf.constant(
                    _OOD_EFFECT_SHIFT_PURE_CONTEXT_FLOOR,
                    dtype=tf.float64,
                )
                + tf.constant(
                    1.0 - _OOD_EFFECT_SHIFT_PURE_CONTEXT_FLOOR,
                    dtype=tf.float64,
                )
                * context
            )
        if overlap_penalty_domain == "combined_shift":
            support_context = tf.sigmoid(
                (
                    positive_support
                    - tf.constant(
                        _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_SUPPORT_CENTER,
                        dtype=tf.float64,
                    )
                )
                / tf.constant(
                    _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_SUPPORT_WIDTH,
                    dtype=tf.float64,
                )
            )
            low_design_context = tf.sigmoid(
                (
                    tf.constant(
                        _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_LOW_DESIGN_CENTER,
                        dtype=tf.float64,
                    )
                    - arrays["design_signal"]
                )
                / tf.constant(
                    _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_LOW_DESIGN_WIDTH,
                    dtype=tf.float64,
                )
            )
            low_community_context = tf.sigmoid(
                (
                    tf.constant(
                        _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_LOW_COMMUNITY_CENTER,
                        dtype=tf.float64,
                    )
                    - arrays["community_occupancy"]
                )
                / tf.constant(
                    _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_LOW_COMMUNITY_WIDTH,
                    dtype=tf.float64,
                )
            )
            context = support_context * low_design_context * low_community_context
            return (
                tf.constant(
                    _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_FLOOR,
                    dtype=tf.float64,
                )
                + tf.constant(
                    1.0 - _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_FLOOR,
                    dtype=tf.float64,
                )
                * context
            )
        return tf.zeros_like(arrays["support_excess"], dtype=tf.float64)

    def log_inflation_for(arrays: dict[str, Any]) -> tf.Tensor:
        support_linear = tf.nn.softplus(raw_support_linear)
        support_quadratic = tf.nn.softplus(raw_support_quadratic)
        effect_linear = tf.nn.softplus(raw_effect_linear)
        effect_quadratic = tf.nn.softplus(raw_effect_quadratic)
        effect_gate_support_linear = tf.nn.softplus(raw_effect_gate_support_linear)
        effect_gate_effect_linear = tf.nn.softplus(raw_effect_gate_effect_linear)
        effect_high_design_suppression = tf.nn.softplus(
            raw_effect_high_design_suppression
        )
        pure_effect_linear = tf.nn.softplus(raw_pure_effect_linear)
        pure_support_suppression = tf.nn.softplus(raw_pure_support_suppression)
        pure_log_amplitude = tf.nn.softplus(raw_pure_log_amplitude)
        pure_effect_bin_log_amplitudes = tf.nn.softplus(
            raw_pure_effect_bin_log_amplitudes
        )
        combined_effect_linear = tf.nn.softplus(raw_combined_effect_linear)
        combined_support_linear = tf.nn.softplus(raw_combined_support_linear)
        combined_log_amplitude = tf.nn.softplus(raw_combined_log_amplitude)
        combined_effect_bin_log_amplitudes = tf.nn.softplus(
            raw_combined_effect_bin_log_amplitudes
        )
        return _tf_learned_ood_log_inflation(
            arrays["support_excess"],
            effect_signal=arrays["effect_signal"],
            design_signal=arrays["design_signal"],
            community_occupancy=arrays["community_occupancy"],
            prevalence_stratum=arrays["prevalence_stratum"],
            design_stratum=arrays["design_stratum"],
            coefficient_stratum=arrays["coefficient_stratum"],
            offset=offset,
            support_linear=support_linear,
            support_quadratic=support_quadratic,
            effect_linear=effect_linear,
            effect_quadratic=effect_quadratic,
            effect_gate_intercept=(
                effect_gate_intercept if gate_effect_branch else None
            ),
            effect_gate_support_linear=(
                effect_gate_support_linear if gate_effect_branch else None
            ),
            effect_gate_effect_linear=(
                effect_gate_effect_linear if gate_effect_branch else None
            ),
            effect_high_design_suppression=(
                effect_high_design_suppression if gate_effect_branch else None
            ),
            prevalence_gate_offsets=(
                prevalence_gate_offsets if gate_effect_branch else None
            ),
            design_gate_offsets=(design_gate_offsets if gate_effect_branch else None),
            coefficient_gate_offsets=(
                coefficient_gate_offsets if gate_effect_branch else None
            ),
            pure_effect_intercept=(
                pure_effect_intercept if gate_effect_branch else None
            ),
            pure_effect_linear=(pure_effect_linear if gate_effect_branch else None),
            pure_support_suppression=(
                pure_support_suppression if gate_effect_branch else None
            ),
            pure_log_amplitude=(pure_log_amplitude if gate_effect_branch else None),
            pure_effect_bin_log_amplitudes=(
                pure_effect_bin_log_amplitudes if gate_effect_branch else None
            ),
            combined_effect_intercept=(
                combined_effect_intercept if gate_effect_branch else None
            ),
            combined_effect_linear=(
                combined_effect_linear if gate_effect_branch else None
            ),
            combined_support_linear=(
                combined_support_linear if gate_effect_branch else None
            ),
            combined_log_amplitude=(
                combined_log_amplitude if gate_effect_branch else None
            ),
            combined_effect_bin_log_amplitudes=(
                combined_effect_bin_log_amplitudes if gate_effect_branch else None
            ),
            max_multiplier=max_ood_multiplier,
        )

    def combined_context_for(arrays: dict[str, Any]) -> tf.Tensor:
        if not gate_effect_branch:
            return tf.zeros_like(arrays["support_excess"], dtype=tf.float64)
        positive_support = tf.maximum(
            arrays["support_excess"],
            tf.constant(0.0, dtype=tf.float64),
        )
        combined_gate = tf.sigmoid(
            combined_effect_intercept
            + tf.nn.softplus(raw_combined_effect_linear) * arrays["effect_signal"]
            + tf.nn.softplus(raw_combined_support_linear) * positive_support
        )
        support_gate = tf.sigmoid(
            (
                positive_support
                - tf.constant(
                    _COMBINED_SHIFT_SCALE_SUPPORT_CENTER,
                    dtype=tf.float64,
                )
            )
            / tf.constant(_COMBINED_SHIFT_SCALE_SUPPORT_WIDTH, dtype=tf.float64)
        )
        low_design_gate = tf.sigmoid(
            (
                tf.constant(
                    _COMBINED_SHIFT_SCALE_LOW_DESIGN_CENTER,
                    dtype=tf.float64,
                )
                - arrays["design_signal"]
            )
            / tf.constant(_COMBINED_SHIFT_SCALE_LOW_DESIGN_WIDTH, dtype=tf.float64)
        )
        low_community_gate = tf.sigmoid(
            (
                tf.constant(
                    _COMBINED_SHIFT_SCALE_LOW_COMMUNITY_CENTER,
                    dtype=tf.float64,
                )
                - arrays["community_occupancy"]
            )
            / tf.constant(
                _COMBINED_SHIFT_SCALE_LOW_COMMUNITY_WIDTH,
                dtype=tf.float64,
            )
        )
        return combined_gate * support_gate * low_design_gate * low_community_gate

    def total_multiplier(arrays: dict[str, Any]) -> tf.Tensor:
        log_inflation = log_inflation_for(arrays)
        return tf.clip_by_value(
            arrays["base_multiplier"]
            * tf.exp(log_inflation)
            * arrays["post_scale_multiplier"],
            float(min_multiplier),
            float(max_multiplier),
        )

    last_ood_loss = tf.constant(0.0, dtype=tf.float64)
    last_rank_loss = tf.constant(0.0, dtype=tf.float64)
    last_gate_loss = tf.constant(0.0, dtype=tf.float64)
    for epoch_index in range(epochs):
        progress = (
            float(epoch_index) / max(float(epochs - 1), 1.0)
            if final_multiplier_aware and gate_effect_branch
            else 1.0
        )
        if progress < _COMBINED_SHIFT_OBJECTIVE_WARMUP_FRACTION:
            combined_coverage_stage_weight = (
                _COMBINED_SHIFT_OBJECTIVE_WARMUP_COVERAGE_BOOST
            )
            gate_stage_weight = _COMBINED_SHIFT_OBJECTIVE_WARMUP_GATE_FRACTION
            overlap_stage_weight = 0.0
        else:
            ramp = (progress - _COMBINED_SHIFT_OBJECTIVE_WARMUP_FRACTION) / max(
                1.0 - _COMBINED_SHIFT_OBJECTIVE_WARMUP_FRACTION, 1e-8
            )
            combined_coverage_stage_weight = 1.0
            gate_stage_weight = (
                _COMBINED_SHIFT_OBJECTIVE_WARMUP_GATE_FRACTION
                + (1.0 - _COMBINED_SHIFT_OBJECTIVE_WARMUP_GATE_FRACTION) * ramp
            )
            overlap_stage_weight = ramp
        with tf.GradientTape() as tape:
            ood_losses = []
            rank_losses = []
            for arrays in tf_domains:
                multiplier = total_multiplier(arrays)
                signed_error = arrays["signed_error"]
                nll = tf.reduce_mean(
                    tf.math.log(multiplier) + 0.5 * tf.square(signed_error / multiplier)
                )
                rank_probability = _tf_normal_cdf(signed_error / multiplier)
                rank_loss = _tf_rank_moment_loss(
                    rank_probability,
                    arrays["rank_groups"],
                    mean_tolerance=rank_mean_tolerance,
                    variance_tolerance=rank_variance_tolerance,
                )
                coverage = tf.reduce_mean(
                    tf.cast(
                        tf.abs(signed_error) <= float(z_value) * multiplier,
                        tf.float64,
                    )
                )
                smooth_coverage = tf.reduce_mean(
                    tf.sigmoid(
                        (float(z_value) * multiplier - tf.abs(signed_error))
                        / tf.constant(0.05, dtype=tf.float64)
                    )
                )
                coverage_loss = tf.square(
                    tf.nn.relu((target_coverage - coverage) / 0.05)
                )
                floor_loss = tf.square(
                    tf.nn.relu((arrays["coverage_floor"] - smooth_coverage) / 0.025)
                )
                baseline_boundary = float(z_value) * arrays["base_multiplier"]
                current_boundary = float(z_value) * multiplier
                baseline_miss = tf.abs(signed_error) - baseline_boundary
                near_miss_weight = tf.exp(
                    -tf.square(
                        tf.nn.relu(baseline_miss) / tf.constant(0.35, dtype=tf.float64)
                    )
                )
                near_miss_weight = near_miss_weight * tf.sigmoid(
                    baseline_miss / tf.constant(0.03, dtype=tf.float64)
                )
                margin_shortfall = tf.nn.relu(tf.abs(signed_error) - current_boundary)
                margin_loss = tf.reduce_sum(
                    near_miss_weight
                    * tf.square(margin_shortfall / tf.constant(0.05, dtype=tf.float64))
                ) / (
                    tf.reduce_sum(near_miss_weight)
                    + tf.constant(1e-8, dtype=tf.float64)
                )
                effect_quantile_losses = []
                for group in arrays["effect_quantile_groups"]:
                    selected_error = tf.boolean_mask(signed_error, group)
                    selected_multiplier = tf.boolean_mask(multiplier, group)
                    selected_smooth_coverage = tf.reduce_mean(
                        tf.sigmoid(
                            (
                                float(z_value) * selected_multiplier
                                - tf.abs(selected_error)
                            )
                            / tf.constant(0.05, dtype=tf.float64)
                        )
                    )
                    effect_quantile_losses.append(
                        tf.square(
                            tf.nn.relu(
                                (arrays["coverage_floor"] - selected_smooth_coverage)
                                / 0.025
                            )
                        )
                    )
                effect_quantile_loss = (
                    tf.reduce_mean(tf.stack(effect_quantile_losses))
                    if effect_quantile_losses
                    else tf.constant(0.0, dtype=tf.float64)
                )
                combined_context = combined_context_for(arrays)
                smooth_margin = tf.sigmoid(
                    (float(z_value) * multiplier - tf.abs(signed_error))
                    / tf.constant(0.05, dtype=tf.float64)
                )
                combined_context_coverage = tf.reduce_sum(
                    combined_context * smooth_margin
                ) / (
                    tf.reduce_sum(combined_context)
                    + tf.constant(1e-8, dtype=tf.float64)
                )
                combined_context_coverage_loss = tf.square(
                    tf.nn.relu(
                        (arrays["coverage_floor"] - combined_context_coverage) / 0.025
                    )
                )
                ood_losses.append(
                    arrays["weight"]
                    * (
                        nll
                        + rank_loss
                        + coverage_loss
                        + arrays["coverage_gate_weight"] * floor_loss
                        + arrays["coverage_gate_weight"] * effect_quantile_loss
                        + combined_coverage_stage_weight
                        * arrays["combined_direct_weight"]
                        * floor_loss
                        + combined_coverage_stage_weight
                        * arrays["combined_quantile_weight"]
                        * effect_quantile_loss
                        + combined_coverage_stage_weight
                        * arrays["combined_context_weight"]
                        * combined_context_coverage_loss
                        + arrays["expert_target_coverage_weight"] * floor_loss
                        + arrays["expert_effect_quantile_weight"] * effect_quantile_loss
                        + arrays["expert_margin_weight"] * margin_loss
                    )
                )
                rank_losses.append(rank_loss)
            ood_loss = tf.reduce_mean(tf.stack(ood_losses))
            ood_rank_loss = tf.reduce_mean(tf.stack(rank_losses))

            in_multiplier = total_multiplier(in_domain)
            in_rank_probability = _tf_normal_cdf(
                in_domain["signed_error"] / in_multiplier
            )
            in_rank_losses = []
            for group in in_domain["rank_groups"]:
                selected = tf.boolean_mask(in_rank_probability, group)
                selected_signed_error = tf.boolean_mask(
                    in_domain["signed_error"], group
                )
                selected_multiplier = tf.boolean_mask(in_multiplier, group)
                rank_mean = tf.reduce_mean(selected)
                rank_variance = tf.reduce_mean(tf.square(selected - rank_mean))
                group_coverage = tf.reduce_mean(
                    tf.cast(
                        tf.abs(selected_signed_error)
                        <= float(z_value) * selected_multiplier,
                        tf.float64,
                    )
                )
                in_rank_losses.append(
                    tf.square(
                        tf.nn.relu(tf.abs(rank_mean - 0.5) / rank_mean_tolerance - 1.0)
                    )
                    + tf.square(
                        tf.nn.relu(
                            tf.abs(rank_variance - expected_rank_variance)
                            / rank_variance_tolerance
                            - 1.0
                        )
                    )
                    + tf.square(tf.nn.relu((0.925 - group_coverage) / 0.05))
                )
            in_coverage = tf.reduce_mean(
                tf.cast(
                    tf.abs(in_domain["signed_error"]) <= float(z_value) * in_multiplier,
                    tf.float64,
                )
            )
            in_group_losses = tf.stack(in_rank_losses)
            gate_loss = (
                tf.reduce_mean(in_group_losses)
                + tf.reduce_max(in_group_losses)
                + tf.square(tf.nn.relu((0.90 - in_coverage) / 0.05))
            )
            if gate_effect_branch:
                in_extra_log_inflation = log_inflation_for(in_domain)
                gate_loss = gate_loss + 0.05 * tf.reduce_mean(
                    tf.square(
                        tf.nn.relu(
                            (
                                in_extra_log_inflation
                                - tf.math.log(tf.constant(1.05, dtype=tf.float64))
                            )
                            / tf.math.log(tf.constant(1.25, dtype=tf.float64))
                        )
                    )
                )
                stratum_extra_cap = tf.math.log(tf.constant(1.04, dtype=tf.float64))
                stratum_extra_losses = []
                for group in in_domain["rank_groups"]:
                    selected_extra = tf.boolean_mask(in_extra_log_inflation, group)
                    stratum_extra_losses.append(
                        tf.reduce_mean(
                            tf.square(tf.nn.relu(selected_extra - stratum_extra_cap))
                        )
                    )
                gate_loss = gate_loss + 0.05 * tf.reduce_max(
                    tf.stack(stratum_extra_losses)
                )
                high_design_close_weight = tf.sigmoid(
                    4.0
                    * (in_domain["design_signal"] - tf.constant(1.0, dtype=tf.float64))
                ) * tf.exp(
                    -tf.maximum(
                        in_domain["support_excess"],
                        tf.constant(0.0, dtype=tf.float64),
                    )
                )
                high_design_close_cap = tf.math.log(tf.constant(1.03, dtype=tf.float64))
                gate_loss = gate_loss + 0.2 * (
                    tf.reduce_sum(
                        high_design_close_weight
                        * tf.square(
                            tf.nn.relu(in_extra_log_inflation - high_design_close_cap)
                        )
                    )
                    / (
                        tf.reduce_sum(high_design_close_weight)
                        + tf.constant(1e-8, dtype=tf.float64)
                    )
                )
                if (
                    overlap_penalty_domain is not None
                    and bin_in_domain_penalty_weight > 0.0
                ):
                    bin_context = effect_bin_basis_for(in_domain) * tf.expand_dims(
                        effect_bin_context_for(in_domain),
                        axis=-1,
                    )
                    bin_extra = tf.expand_dims(in_extra_log_inflation, axis=-1)
                    bin_rank_probability = tf.expand_dims(in_rank_probability, axis=-1)
                    bin_smooth_coverage = tf.expand_dims(
                        tf.sigmoid(
                            (
                                float(z_value) * in_multiplier
                                - tf.abs(in_domain["signed_error"])
                            )
                            / tf.constant(0.05, dtype=tf.float64)
                        ),
                        axis=-1,
                    )
                    bin_weight_total = tf.reduce_sum(
                        bin_context,
                        axis=0,
                    ) + tf.constant(1e-8, dtype=tf.float64)
                    bin_mean_extra_loss = tf.reduce_mean(
                        tf.reduce_sum(
                            bin_context
                            * tf.square(
                                tf.nn.relu(
                                    bin_extra
                                    - tf.constant(
                                        _OOD_EFFECT_SHIFT_BIN_IN_DOMAIN_CAP,
                                        dtype=tf.float64,
                                    )
                                )
                                / tf.math.log(tf.constant(1.25, dtype=tf.float64))
                            ),
                            axis=0,
                        )
                        / bin_weight_total
                    )
                    bin_rank_mean = (
                        tf.reduce_sum(bin_context * bin_rank_probability, axis=0)
                        / bin_weight_total
                    )
                    bin_coverage = (
                        tf.reduce_sum(bin_context * bin_smooth_coverage, axis=0)
                        / bin_weight_total
                    )
                    bin_rank_loss = tf.reduce_mean(
                        tf.square(
                            tf.nn.relu(
                                tf.abs(bin_rank_mean - 0.5) / rank_mean_tolerance - 1.0
                            )
                        )
                    )
                    bin_coverage_loss = tf.reduce_mean(
                        tf.square(tf.nn.relu((0.925 - bin_coverage) / 0.05))
                    )
                    gate_loss = gate_loss + tf.constant(
                        float(bin_in_domain_penalty_weight),
                        dtype=tf.float64,
                    ) * (bin_mean_extra_loss + bin_rank_loss + bin_coverage_loss)
                gate_loss = gate_loss + 0.01 * (
                    tf.reduce_mean(tf.square(prevalence_gate_offsets))
                    + tf.reduce_mean(tf.square(design_gate_offsets))
                    + tf.reduce_mean(tf.square(coefficient_gate_offsets))
                )
                gate_loss = gate_loss + 0.01 * (
                    tf.square(tf.nn.softplus(raw_pure_log_amplitude))
                    + tf.square(tf.nn.softplus(raw_combined_log_amplitude))
                    + tf.reduce_mean(
                        tf.square(tf.nn.softplus(raw_pure_effect_bin_log_amplitudes))
                    )
                    + tf.reduce_mean(
                        tf.square(
                            tf.nn.softplus(raw_combined_effect_bin_log_amplitudes)
                        )
                    )
                )
                in_combined_context = combined_context_for(in_domain)
                in_context_active = tf.reduce_mean(
                    tf.sigmoid(
                        (in_combined_context - tf.constant(0.8, dtype=tf.float64))
                        / tf.constant(0.05, dtype=tf.float64)
                    )
                )
                context_overlap_loss = tf.square(
                    tf.nn.relu(
                        (
                            tf.reduce_mean(in_combined_context)
                            - tf.constant(
                                _COMBINED_SHIFT_OBJECTIVE_OVERLAP_MEAN_CAP,
                                dtype=tf.float64,
                            )
                        )
                        / tf.constant(0.05, dtype=tf.float64)
                    )
                ) + tf.square(
                    tf.nn.relu(
                        (
                            in_context_active
                            - tf.constant(
                                _COMBINED_SHIFT_OBJECTIVE_OVERLAP_ACTIVE_CAP,
                                dtype=tf.float64,
                            )
                        )
                        / tf.constant(0.05, dtype=tf.float64)
                    )
                )
                context_extra_inflation_loss = tf.reduce_sum(
                    in_combined_context
                    * tf.square(
                        tf.nn.relu(
                            (
                                in_extra_log_inflation
                                - tf.math.log(tf.constant(1.04, dtype=tf.float64))
                            )
                            / tf.math.log(tf.constant(1.25, dtype=tf.float64))
                        )
                    )
                ) / (
                    tf.reduce_sum(in_combined_context)
                    + tf.constant(1e-8, dtype=tf.float64)
                )
                gate_loss = gate_loss + tf.constant(
                    _COMBINED_SHIFT_OBJECTIVE_OVERLAP_WEIGHT,
                    dtype=tf.float64,
                ) * overlap_stage_weight * (
                    context_overlap_loss + context_extra_inflation_loss
                )
                if in_domain_trust_region_baseline is not None:
                    trust_tolerance = tf.constant(
                        max(float(trust_region_log_tolerance), 0.0),
                        dtype=tf.float64,
                    )
                    trust_scale = tf.math.log(tf.constant(1.25, dtype=tf.float64))
                    trust_excess = tf.nn.relu(
                        tf.abs(in_extra_log_inflation - in_domain_trust_region_baseline)
                        - trust_tolerance
                    )
                    trust_region_loss = tf.reduce_mean(
                        tf.square(trust_excess / trust_scale)
                    )
                    trust_group_losses = []
                    for group in in_domain["rank_groups"]:
                        selected_trust_excess = tf.boolean_mask(trust_excess, group)
                        trust_group_losses.append(
                            tf.reduce_mean(
                                tf.square(selected_trust_excess / trust_scale)
                            )
                        )
                    gate_loss = gate_loss + tf.constant(
                        float(trust_region_weight),
                        dtype=tf.float64,
                    ) * (
                        trust_region_loss + tf.reduce_max(tf.stack(trust_group_losses))
                    )
                if (
                    in_domain_trust_region_baseline is not None
                    and overlap_penalty_weight > 0.0
                    and overlap_penalty_domain is not None
                ):
                    overlap_context = localized_overlap_context_for(in_domain)
                    overlap_tolerance = tf.constant(
                        max(float(overlap_log_tolerance), 0.0),
                        dtype=tf.float64,
                    )
                    overlap_scale = tf.math.log(tf.constant(1.25, dtype=tf.float64))
                    overlap_excess = tf.nn.relu(
                        tf.abs(in_extra_log_inflation - in_domain_trust_region_baseline)
                        - overlap_tolerance
                    )
                    weighted_overlap_loss = tf.reduce_sum(
                        overlap_context * tf.square(overlap_excess / overlap_scale)
                    ) / (
                        tf.reduce_sum(overlap_context)
                        + tf.constant(1e-8, dtype=tf.float64)
                    )
                    overlap_active = tf.reduce_mean(
                        tf.sigmoid(
                            (overlap_context - tf.constant(0.5, dtype=tf.float64))
                            / tf.constant(0.05, dtype=tf.float64)
                        )
                    )
                    gate_loss = gate_loss + tf.constant(
                        float(overlap_penalty_weight),
                        dtype=tf.float64,
                    ) * (
                        weighted_overlap_loss
                        + tf.square(
                            tf.nn.relu(
                                (overlap_active - tf.constant(0.35, dtype=tf.float64))
                                / tf.constant(0.10, dtype=tf.float64)
                            )
                        )
                    )
            loss = (
                float(objective_weight) * ood_loss
                + float(in_domain_gate_weight) * gate_stage_weight * gate_loss
            )
        variables = [
            offset,
            raw_support_linear,
            raw_support_quadratic,
            raw_effect_linear,
            raw_effect_quadratic,
        ]
        if gate_effect_branch:
            variables.extend(
                [
                    effect_gate_intercept,
                    raw_effect_gate_support_linear,
                    raw_effect_gate_effect_linear,
                    raw_effect_high_design_suppression,
                    prevalence_gate_offsets,
                    design_gate_offsets,
                    coefficient_gate_offsets,
                    pure_effect_intercept,
                    raw_pure_effect_linear,
                    raw_pure_support_suppression,
                    raw_pure_log_amplitude,
                    raw_pure_effect_bin_log_amplitudes,
                    combined_effect_intercept,
                    raw_combined_effect_linear,
                    raw_combined_support_linear,
                    raw_combined_log_amplitude,
                    raw_combined_effect_bin_log_amplitudes,
                ]
            )
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        last_ood_loss = ood_loss
        last_rank_loss = ood_rank_loss
        last_gate_loss = gate_loss

    learned_values = [
        float(offset.numpy()),
        float(tf.nn.softplus(raw_support_linear).numpy()),
        float(tf.nn.softplus(raw_support_quadratic).numpy()),
        float(tf.nn.softplus(raw_effect_linear).numpy()),
        float(tf.nn.softplus(raw_effect_quadratic).numpy()),
    ]
    if gate_effect_branch:
        learned_values.extend(
            [
                float(effect_gate_intercept.numpy()),
                float(tf.nn.softplus(raw_effect_gate_support_linear).numpy()),
                float(tf.nn.softplus(raw_effect_gate_effect_linear).numpy()),
                float(tf.nn.softplus(raw_effect_high_design_suppression).numpy()),
                *[float(value) for value in prevalence_gate_offsets.numpy()],
                *[float(value) for value in design_gate_offsets.numpy()],
                *[float(value) for value in coefficient_gate_offsets.numpy()],
                float(pure_effect_intercept.numpy()),
                float(tf.nn.softplus(raw_pure_effect_linear).numpy()),
                float(tf.nn.softplus(raw_pure_support_suppression).numpy()),
                float(tf.nn.softplus(raw_pure_log_amplitude).numpy()),
                float(combined_effect_intercept.numpy()),
                float(tf.nn.softplus(raw_combined_effect_linear).numpy()),
                float(tf.nn.softplus(raw_combined_support_linear).numpy()),
                float(tf.nn.softplus(raw_combined_log_amplitude).numpy()),
                *[
                    float(value)
                    for value in tf.nn.softplus(
                        raw_pure_effect_bin_log_amplitudes
                    ).numpy()
                ],
                *[
                    float(value)
                    for value in tf.nn.softplus(
                        raw_combined_effect_bin_log_amplitudes
                    ).numpy()
                ],
            ]
        )
    learned = tuple(learned_values)
    return (
        learned,
        float(last_ood_loss.numpy()),
        float(last_rank_loss.numpy()),
        float(last_gate_loss.numpy()),
        n_observations,
        tuple(labels),
    )


def _softplus_inverse(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("softplus inverse requires a positive value")
    return float(np.log(np.expm1(value)))


def _effect_shift_head_count(values: Sequence[float]) -> int:
    """Return the recognized experimental effect-shift head length."""
    n_values = len(values)
    if n_values >= _OOD_EFFECT_SHIFT_HEAD_PARAMETER_COUNT:
        return _OOD_EFFECT_SHIFT_HEAD_PARAMETER_COUNT
    if n_values >= _OOD_EFFECT_SHIFT_HEAD_BASE_PARAMETER_COUNT:
        return _OOD_EFFECT_SHIFT_HEAD_BASE_PARAMETER_COUNT
    return 0


def _split_gated_ood_parameter_tail(
    parameters: tuple[float, ...],
    *,
    coefficient_stratum: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, ...]]:
    """Split version-8 stratum offsets from optional experimental head params."""
    prevalence_gate_offsets = np.zeros(3, dtype=float)
    design_gate_offsets = np.zeros(3, dtype=float)
    coefficient_gate_offsets = np.zeros(0, dtype=float)
    effect_shift_head: tuple[float, ...] = ()
    if len(parameters) < 9:
        return (
            prevalence_gate_offsets,
            design_gate_offsets,
            coefficient_gate_offsets,
            effect_shift_head,
        )
    extra_parameters = tuple(float(value) for value in parameters[9:])
    if len(extra_parameters) < 6:
        return (
            prevalence_gate_offsets,
            design_gate_offsets,
            coefficient_gate_offsets,
            effect_shift_head,
        )
    prevalence_gate_offsets = np.asarray(extra_parameters[:3], dtype=float)
    design_gate_offsets = np.asarray(extra_parameters[3:6], dtype=float)
    remaining = extra_parameters[6:]
    if coefficient_stratum is not None:
        coefficient_array = np.asarray(coefficient_stratum, dtype=np.int32)
        n_coefficients = (
            int(np.max(coefficient_array)) + 1 if coefficient_array.size else 0
        )
        n_coefficients = min(max(n_coefficients, 0), len(remaining))
        coefficient_gate_offsets = np.asarray(
            remaining[:n_coefficients],
            dtype=float,
        )
        head_start = n_coefficients
        head_count = _effect_shift_head_count(remaining[head_start:])
        if head_count:
            effect_shift_head = tuple(remaining[head_start : head_start + head_count])
    else:
        head_count = _effect_shift_head_count(remaining)
        coefficient_count = max(0, len(remaining) - head_count)
        coefficient_gate_offsets = np.asarray(
            remaining[:coefficient_count],
            dtype=float,
        )
        if head_count:
            effect_shift_head = tuple(
                remaining[coefficient_count : coefficient_count + head_count]
            )
    return (
        prevalence_gate_offsets,
        design_gate_offsets,
        coefficient_gate_offsets,
        effect_shift_head,
    )


def _learned_ood_log_inflation_numpy(
    support_excess: np.ndarray,
    *,
    effect_signal: np.ndarray,
    design_signal: np.ndarray | None = None,
    community_occupancy: np.ndarray | None = None,
    prevalence_stratum: np.ndarray | None = None,
    design_stratum: np.ndarray | None = None,
    coefficient_stratum: np.ndarray | None = None,
    parameters: tuple[float, ...],
    max_multiplier: float,
) -> np.ndarray:
    design = (
        np.zeros_like(support_excess, dtype=float)
        if design_signal is None
        else np.asarray(design_signal, dtype=float)
    )
    if len(parameters) >= 7:
        (
            prevalence_gate_offsets,
            design_gate_offsets,
            coefficient_gate_offsets,
            effect_shift_head,
        ) = _split_gated_ood_parameter_tail(
            parameters,
            coefficient_stratum=coefficient_stratum,
        )
        if len(parameters) >= 9:
            (
                offset,
                support_linear,
                support_quadratic,
                effect_linear,
                effect_quadratic,
                effect_gate_intercept,
                effect_gate_support_linear,
                effect_gate_effect_linear,
                effect_high_design_suppression,
            ) = parameters[:9]
        elif len(parameters) >= 8:
            (
                offset,
                support_linear,
                support_quadratic,
                effect_linear,
                effect_quadratic,
                effect_gate_intercept,
                effect_gate_support_linear,
                effect_gate_effect_linear,
            ) = parameters[:8]
            effect_high_design_suppression = 0.0
        else:
            (
                offset,
                support_linear,
                support_quadratic,
                effect_linear,
                effect_quadratic,
                effect_gate_intercept,
                effect_gate_support_linear,
            ) = parameters[:7]
            effect_gate_effect_linear = 0.0
            effect_high_design_suppression = 0.0
        support_close_design = design / (1.0 + np.maximum(support_excess, 0.0))
        stratum_gate_offset = np.zeros_like(support_excess, dtype=float)
        if prevalence_stratum is not None:
            stratum_gate_offset = (
                stratum_gate_offset
                + prevalence_gate_offsets[
                    np.clip(np.asarray(prevalence_stratum, dtype=np.int32), 0, 2)
                ]
            )
        if design_stratum is not None:
            stratum_gate_offset = (
                stratum_gate_offset
                + design_gate_offsets[
                    np.clip(np.asarray(design_stratum, dtype=np.int32), 0, 2)
                ]
            )
        if coefficient_stratum is not None and coefficient_gate_offsets.size:
            stratum_gate_offset = (
                stratum_gate_offset
                + coefficient_gate_offsets[
                    np.clip(
                        np.asarray(coefficient_stratum, dtype=np.int32),
                        0,
                        coefficient_gate_offsets.size - 1,
                    )
                ]
            )
        effect_gate = _sigmoid_numpy(
            float(effect_gate_intercept)
            + float(effect_gate_support_linear) * support_excess
            + float(effect_gate_effect_linear) * effect_signal
            - float(effect_high_design_suppression) * support_close_design
            + stratum_gate_offset
        )
    elif len(parameters) >= 5:
        offset, support_linear, support_quadratic, effect_linear, effect_quadratic = (
            parameters[:5]
        )
        effect_gate = 1.0
    elif len(parameters) == 3:
        offset, support_linear, support_quadratic = parameters
        effect_linear = 0.0
        effect_quadratic = 0.0
        effect_gate = 1.0
    else:
        raise ValueError(
            "learned OOD inflation curve requires three, five, seven, eight, "
            "nine, or stratum-conditioned parameters"
        )
    raw = (
        float(offset)
        + float(support_linear) * support_excess
        + float(support_quadratic) * np.square(support_excess)
        + effect_gate
        * (
            float(effect_linear) * effect_signal
            + float(effect_quadratic) * np.square(effect_signal)
        )
    )
    baseline = np.logaddexp(0.0, float(offset))
    log_inflation = np.logaddexp(0.0, raw) - baseline
    if len(parameters) >= 7 and len(effect_shift_head) in (
        _OOD_EFFECT_SHIFT_HEAD_BASE_PARAMETER_COUNT,
        _OOD_EFFECT_SHIFT_HEAD_PARAMETER_COUNT,
    ):
        log_inflation = log_inflation + _effect_shift_head_log_inflation_numpy(
            support_excess,
            effect_signal=effect_signal,
            design_signal=design_signal,
            community_occupancy=community_occupancy,
            parameters=effect_shift_head,
        )
    return np.clip(log_inflation, 0.0, np.log(float(max_multiplier)))


def _learned_ood_effect_gate_numpy(
    support_excess: np.ndarray,
    *,
    effect_signal: np.ndarray,
    design_signal: np.ndarray | None = None,
    prevalence_stratum: np.ndarray | None = None,
    design_stratum: np.ndarray | None = None,
    coefficient_stratum: np.ndarray | None = None,
    parameters: tuple[float, ...],
) -> np.ndarray:
    """Return the learned effect-branch gate activation."""
    support = np.asarray(support_excess, dtype=float)
    if len(parameters) < 7:
        return np.ones_like(support, dtype=float)
    design = (
        np.zeros_like(support, dtype=float)
        if design_signal is None
        else np.asarray(design_signal, dtype=float)
    )
    (
        prevalence_gate_offsets,
        design_gate_offsets,
        coefficient_gate_offsets,
        _effect_shift_head,
    ) = _split_gated_ood_parameter_tail(
        parameters,
        coefficient_stratum=coefficient_stratum,
    )
    if len(parameters) >= 9:
        (
            _offset,
            _support_linear,
            _support_quadratic,
            _effect_linear,
            _effect_quadratic,
            effect_gate_intercept,
            effect_gate_support_linear,
            effect_gate_effect_linear,
            effect_high_design_suppression,
        ) = parameters[:9]
    elif len(parameters) >= 8:
        (
            _offset,
            _support_linear,
            _support_quadratic,
            _effect_linear,
            _effect_quadratic,
            effect_gate_intercept,
            effect_gate_support_linear,
            effect_gate_effect_linear,
        ) = parameters[:8]
        effect_high_design_suppression = 0.0
    else:
        (
            _offset,
            _support_linear,
            _support_quadratic,
            _effect_linear,
            _effect_quadratic,
            effect_gate_intercept,
            effect_gate_support_linear,
        ) = parameters[:7]
        effect_gate_effect_linear = 0.0
        effect_high_design_suppression = 0.0
    support_close_design = design / (1.0 + np.maximum(support, 0.0))
    stratum_gate_offset = np.zeros_like(support, dtype=float)
    if prevalence_stratum is not None:
        stratum_gate_offset = (
            stratum_gate_offset
            + prevalence_gate_offsets[
                np.clip(np.asarray(prevalence_stratum, dtype=np.int32), 0, 2)
            ]
        )
    if design_stratum is not None:
        stratum_gate_offset = (
            stratum_gate_offset
            + design_gate_offsets[
                np.clip(np.asarray(design_stratum, dtype=np.int32), 0, 2)
            ]
        )
    if coefficient_stratum is not None and coefficient_gate_offsets.size:
        stratum_gate_offset = (
            stratum_gate_offset
            + coefficient_gate_offsets[
                np.clip(
                    np.asarray(coefficient_stratum, dtype=np.int32),
                    0,
                    coefficient_gate_offsets.size - 1,
                )
            ]
        )
    return _sigmoid_numpy(
        float(effect_gate_intercept)
        + float(effect_gate_support_linear) * support
        + float(effect_gate_effect_linear) * np.asarray(effect_signal, dtype=float)
        - float(effect_high_design_suppression) * support_close_design
        + stratum_gate_offset
    )


def _effect_shift_head_log_inflation_numpy(
    support_excess: np.ndarray,
    *,
    effect_signal: np.ndarray,
    design_signal: np.ndarray | None = None,
    community_occupancy: np.ndarray | None = None,
    parameters: tuple[float, ...],
) -> np.ndarray:
    """Return experimental context-gated effect-shift log inflation."""
    if len(parameters) not in (
        _OOD_EFFECT_SHIFT_HEAD_BASE_PARAMETER_COUNT,
        _OOD_EFFECT_SHIFT_HEAD_PARAMETER_COUNT,
    ):
        return np.zeros_like(support_excess, dtype=float)
    (
        pure_intercept,
        pure_effect_linear,
        pure_support_suppression,
        pure_amplitude,
        combined_intercept,
        combined_effect_linear,
        combined_support_linear,
        combined_amplitude,
    ) = parameters[:_OOD_EFFECT_SHIFT_HEAD_BASE_PARAMETER_COUNT]
    pure_bin_amplitudes = np.zeros(3, dtype=float)
    combined_bin_amplitudes = np.zeros(3, dtype=float)
    if len(parameters) >= _OOD_EFFECT_SHIFT_HEAD_PARAMETER_COUNT:
        pure_bin_amplitudes = np.asarray(parameters[8:11], dtype=float)
        combined_bin_amplitudes = np.asarray(parameters[11:14], dtype=float)
    support = np.maximum(np.asarray(support_excess, dtype=float), 0.0)
    effect = np.asarray(effect_signal, dtype=float)
    design = (
        np.zeros_like(support, dtype=float)
        if design_signal is None
        else np.asarray(design_signal, dtype=float)
    )
    community = (
        None
        if community_occupancy is None
        else np.asarray(community_occupancy, dtype=float)
    )
    bin_basis = np.exp(
        -0.5
        * np.square(
            (effect[..., None] - np.asarray(_OOD_EFFECT_SHIFT_BIN_CENTERS, dtype=float))
            / _OOD_EFFECT_SHIFT_BIN_WIDTH
        )
    )
    pure_support_context = _sigmoid_numpy(
        (support - _OOD_EFFECT_SHIFT_PURE_CONTEXT_SUPPORT_CENTER)
        / _OOD_EFFECT_SHIFT_PURE_CONTEXT_SUPPORT_WIDTH
    )
    pure_low_design_context = _sigmoid_numpy(
        (_OOD_EFFECT_SHIFT_PURE_CONTEXT_LOW_DESIGN_CENTER - design)
        / _OOD_EFFECT_SHIFT_PURE_CONTEXT_LOW_DESIGN_WIDTH
    )
    pure_context = 1.0 - (1.0 - pure_support_context) * (1.0 - pure_low_design_context)
    pure_context = (
        _OOD_EFFECT_SHIFT_PURE_CONTEXT_FLOOR
        + (1.0 - _OOD_EFFECT_SHIFT_PURE_CONTEXT_FLOOR) * pure_context
    )
    combined_support_context = _sigmoid_numpy(
        (support - _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_SUPPORT_CENTER)
        / _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_SUPPORT_WIDTH
    )
    combined_low_design_context = _sigmoid_numpy(
        (_OOD_EFFECT_SHIFT_COMBINED_CONTEXT_LOW_DESIGN_CENTER - design)
        / _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_LOW_DESIGN_WIDTH
    )
    combined_context = combined_support_context * combined_low_design_context
    if community is not None:
        combined_low_community_context = _sigmoid_numpy(
            (_OOD_EFFECT_SHIFT_COMBINED_CONTEXT_LOW_COMMUNITY_CENTER - community)
            / _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_LOW_COMMUNITY_WIDTH
        )
        combined_context = combined_context * combined_low_community_context
    combined_context = (
        _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_FLOOR
        + (1.0 - _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_FLOOR) * combined_context
    )
    pure_amplitude_by_effect = pure_context * (
        max(float(pure_amplitude), 0.0)
        + np.sum(bin_basis * np.maximum(pure_bin_amplitudes, 0.0), axis=-1)
    )
    combined_amplitude_by_effect = combined_context * (
        max(float(combined_amplitude), 0.0)
        + np.sum(bin_basis * np.maximum(combined_bin_amplitudes, 0.0), axis=-1)
    )
    pure_gate = _sigmoid_numpy(
        float(pure_intercept)
        + float(pure_effect_linear) * effect
        - float(pure_support_suppression) * support
    )
    pure_high_effect_taper = _sigmoid_numpy(
        (_OOD_EFFECT_SHIFT_PURE_HIGH_EFFECT_CENTER - effect)
        / _OOD_EFFECT_SHIFT_PURE_HIGH_EFFECT_WIDTH
    )
    combined_gate = _sigmoid_numpy(
        float(combined_intercept)
        + float(combined_effect_linear) * effect
        + float(combined_support_linear) * support
    )
    combined_support_gate = _sigmoid_numpy(
        (support - _OOD_EFFECT_SHIFT_COMBINED_SUPPORT_CENTER)
        / _OOD_EFFECT_SHIFT_COMBINED_SUPPORT_WIDTH
    )
    pure_contribution = np.minimum(
        pure_gate * pure_high_effect_taper * pure_amplitude_by_effect,
        _OOD_EFFECT_SHIFT_PURE_LOG_CAP,
    )
    combined_contribution = np.minimum(
        combined_gate * combined_support_gate * combined_amplitude_by_effect,
        _OOD_EFFECT_SHIFT_COMBINED_LOG_CAP,
    )
    return pure_contribution + combined_contribution


def _learned_combined_shift_context_numpy(
    support_excess: np.ndarray,
    *,
    effect_signal: np.ndarray,
    design_signal: np.ndarray,
    community_occupancy: np.ndarray,
    parameters: tuple[float, ...],
    n_covariates: int | None = None,
) -> np.ndarray:
    """Return learned combined-shift context activation from OOD head params."""
    support = np.asarray(support_excess, dtype=float)
    effect_shift_head: tuple[float, ...] = ()
    if n_covariates is not None and len(parameters) >= 15 + int(n_covariates):
        remaining = parameters[15 + int(n_covariates) :]
        head_count = _effect_shift_head_count(remaining)
        if head_count:
            effect_shift_head = tuple(remaining[:head_count])
    if not effect_shift_head:
        (
            _prevalence_gate_offsets,
            _design_gate_offsets,
            _coefficient_gate_offsets,
            effect_shift_head,
        ) = _split_gated_ood_parameter_tail(parameters, coefficient_stratum=None)
    if len(effect_shift_head) not in (
        _OOD_EFFECT_SHIFT_HEAD_BASE_PARAMETER_COUNT,
        _OOD_EFFECT_SHIFT_HEAD_PARAMETER_COUNT,
    ):
        return np.zeros_like(support, dtype=float)
    (
        _pure_intercept,
        _pure_effect_linear,
        _pure_support_suppression,
        _pure_amplitude,
        combined_intercept,
        combined_effect_linear,
        combined_support_linear,
        _combined_amplitude,
    ) = effect_shift_head[:_OOD_EFFECT_SHIFT_HEAD_BASE_PARAMETER_COUNT]
    positive_support = np.maximum(support, 0.0)
    combined_gate = _sigmoid_numpy(
        float(combined_intercept)
        + float(combined_effect_linear) * np.asarray(effect_signal, dtype=float)
        + float(combined_support_linear) * positive_support
    )
    support_gate = _sigmoid_numpy(
        (positive_support - _COMBINED_SHIFT_SCALE_SUPPORT_CENTER)
        / _COMBINED_SHIFT_SCALE_SUPPORT_WIDTH
    )
    low_design_gate = _sigmoid_numpy(
        (
            _COMBINED_SHIFT_SCALE_LOW_DESIGN_CENTER
            - np.asarray(design_signal, dtype=float)
        )
        / _COMBINED_SHIFT_SCALE_LOW_DESIGN_WIDTH
    )
    low_community_gate = _sigmoid_numpy(
        (
            _COMBINED_SHIFT_SCALE_LOW_COMMUNITY_CENTER
            - np.asarray(community_occupancy, dtype=float)
        )
        / _COMBINED_SHIFT_SCALE_LOW_COMMUNITY_WIDTH
    )
    return combined_gate * support_gate * low_design_gate * low_community_gate


def _sigmoid_numpy(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


def _tf_learned_ood_log_inflation(
    support_excess: tf.Tensor,
    *,
    effect_signal: tf.Tensor,
    design_signal: tf.Tensor | None = None,
    community_occupancy: tf.Tensor | None = None,
    prevalence_stratum: tf.Tensor | None = None,
    design_stratum: tf.Tensor | None = None,
    coefficient_stratum: tf.Tensor | None = None,
    offset: tf.Tensor,
    support_linear: tf.Tensor,
    support_quadratic: tf.Tensor,
    effect_linear: tf.Tensor,
    effect_quadratic: tf.Tensor,
    effect_gate_intercept: tf.Tensor | None = None,
    effect_gate_support_linear: tf.Tensor | None = None,
    effect_gate_effect_linear: tf.Tensor | None = None,
    effect_high_design_suppression: tf.Tensor | None = None,
    prevalence_gate_offsets: tf.Tensor | None = None,
    design_gate_offsets: tf.Tensor | None = None,
    coefficient_gate_offsets: tf.Tensor | None = None,
    pure_effect_intercept: tf.Tensor | None = None,
    pure_effect_linear: tf.Tensor | None = None,
    pure_support_suppression: tf.Tensor | None = None,
    pure_log_amplitude: tf.Tensor | None = None,
    pure_effect_bin_log_amplitudes: tf.Tensor | None = None,
    combined_effect_intercept: tf.Tensor | None = None,
    combined_effect_linear: tf.Tensor | None = None,
    combined_support_linear: tf.Tensor | None = None,
    combined_log_amplitude: tf.Tensor | None = None,
    combined_effect_bin_log_amplitudes: tf.Tensor | None = None,
    max_multiplier: float,
) -> tf.Tensor:
    if (
        effect_gate_intercept is None
        or effect_gate_support_linear is None
        or effect_gate_effect_linear is None
    ):
        effect_gate = tf.constant(1.0, dtype=tf.float64)
    else:
        design = (
            tf.zeros_like(support_excess, dtype=tf.float64)
            if design_signal is None
            else design_signal
        )
        suppression = (
            tf.constant(0.0, dtype=tf.float64)
            if effect_high_design_suppression is None
            else effect_high_design_suppression
        )
        support_close_design = design / (
            tf.constant(1.0, dtype=tf.float64)
            + tf.maximum(support_excess, tf.constant(0.0, dtype=tf.float64))
        )
        stratum_gate_offset = tf.zeros_like(support_excess, dtype=tf.float64)
        if prevalence_stratum is not None and prevalence_gate_offsets is not None:
            stratum_gate_offset = stratum_gate_offset + tf.gather(
                prevalence_gate_offsets,
                tf.clip_by_value(prevalence_stratum, 0, 2),
            )
        if design_stratum is not None and design_gate_offsets is not None:
            stratum_gate_offset = stratum_gate_offset + tf.gather(
                design_gate_offsets,
                tf.clip_by_value(design_stratum, 0, 2),
            )
        if coefficient_stratum is not None and coefficient_gate_offsets is not None:
            max_index = tf.shape(coefficient_gate_offsets)[0] - 1
            stratum_gate_offset = stratum_gate_offset + tf.gather(
                coefficient_gate_offsets,
                tf.clip_by_value(coefficient_stratum, 0, max_index),
            )
        effect_gate = tf.sigmoid(
            effect_gate_intercept
            + effect_gate_support_linear * support_excess
            + effect_gate_effect_linear * effect_signal
            - suppression * support_close_design
            + stratum_gate_offset
        )
    raw = (
        offset
        + support_linear * support_excess
        + support_quadratic * tf.square(support_excess)
        + effect_gate
        * (effect_linear * effect_signal + effect_quadratic * tf.square(effect_signal))
    )
    baseline = tf.nn.softplus(offset)
    log_inflation = tf.nn.softplus(raw) - baseline
    if (
        pure_effect_intercept is not None
        and pure_effect_linear is not None
        and pure_support_suppression is not None
        and pure_log_amplitude is not None
        and combined_effect_intercept is not None
        and combined_effect_linear is not None
        and combined_support_linear is not None
        and combined_log_amplitude is not None
    ):
        positive_support = tf.maximum(
            support_excess,
            tf.constant(0.0, dtype=tf.float64),
        )
        design_for_context = (
            tf.zeros_like(positive_support, dtype=tf.float64)
            if design_signal is None
            else design_signal
        )
        pure_gate = tf.sigmoid(
            pure_effect_intercept
            + pure_effect_linear * effect_signal
            - pure_support_suppression * positive_support
        )
        pure_high_effect_taper = tf.sigmoid(
            (
                tf.constant(
                    _OOD_EFFECT_SHIFT_PURE_HIGH_EFFECT_CENTER,
                    dtype=tf.float64,
                )
                - effect_signal
            )
            / tf.constant(
                _OOD_EFFECT_SHIFT_PURE_HIGH_EFFECT_WIDTH,
                dtype=tf.float64,
            )
        )
        effect_bins = tf.exp(
            -0.5
            * tf.square(
                (
                    tf.expand_dims(effect_signal, axis=-1)
                    - tf.constant(_OOD_EFFECT_SHIFT_BIN_CENTERS, dtype=tf.float64)
                )
                / tf.constant(_OOD_EFFECT_SHIFT_BIN_WIDTH, dtype=tf.float64)
            )
        )
        pure_amplitude_by_effect = pure_log_amplitude
        if pure_effect_bin_log_amplitudes is not None:
            pure_amplitude_by_effect = pure_amplitude_by_effect + tf.reduce_sum(
                effect_bins * pure_effect_bin_log_amplitudes,
                axis=-1,
            )
        pure_support_context = tf.sigmoid(
            (
                positive_support
                - tf.constant(
                    _OOD_EFFECT_SHIFT_PURE_CONTEXT_SUPPORT_CENTER,
                    dtype=tf.float64,
                )
            )
            / tf.constant(
                _OOD_EFFECT_SHIFT_PURE_CONTEXT_SUPPORT_WIDTH,
                dtype=tf.float64,
            )
        )
        pure_low_design_context = tf.sigmoid(
            (
                tf.constant(
                    _OOD_EFFECT_SHIFT_PURE_CONTEXT_LOW_DESIGN_CENTER,
                    dtype=tf.float64,
                )
                - design_for_context
            )
            / tf.constant(
                _OOD_EFFECT_SHIFT_PURE_CONTEXT_LOW_DESIGN_WIDTH,
                dtype=tf.float64,
            )
        )
        pure_context = 1.0 - (1.0 - pure_support_context) * (
            1.0 - pure_low_design_context
        )
        pure_context = (
            tf.constant(
                _OOD_EFFECT_SHIFT_PURE_CONTEXT_FLOOR,
                dtype=tf.float64,
            )
            + tf.constant(
                1.0 - _OOD_EFFECT_SHIFT_PURE_CONTEXT_FLOOR,
                dtype=tf.float64,
            )
            * pure_context
        )
        pure_amplitude_by_effect = pure_amplitude_by_effect * pure_context
        combined_amplitude_by_effect = combined_log_amplitude
        if combined_effect_bin_log_amplitudes is not None:
            combined_amplitude_by_effect = combined_amplitude_by_effect + tf.reduce_sum(
                effect_bins * combined_effect_bin_log_amplitudes,
                axis=-1,
            )
        combined_support_context = tf.sigmoid(
            (
                positive_support
                - tf.constant(
                    _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_SUPPORT_CENTER,
                    dtype=tf.float64,
                )
            )
            / tf.constant(
                _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_SUPPORT_WIDTH,
                dtype=tf.float64,
            )
        )
        combined_low_design_context = tf.sigmoid(
            (
                tf.constant(
                    _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_LOW_DESIGN_CENTER,
                    dtype=tf.float64,
                )
                - design_for_context
            )
            / tf.constant(
                _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_LOW_DESIGN_WIDTH,
                dtype=tf.float64,
            )
        )
        combined_context = combined_support_context * combined_low_design_context
        if community_occupancy is not None:
            combined_context = combined_context * tf.sigmoid(
                (
                    tf.constant(
                        _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_LOW_COMMUNITY_CENTER,
                        dtype=tf.float64,
                    )
                    - community_occupancy
                )
                / tf.constant(
                    _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_LOW_COMMUNITY_WIDTH,
                    dtype=tf.float64,
                )
            )
        combined_context = (
            tf.constant(
                _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_FLOOR,
                dtype=tf.float64,
            )
            + tf.constant(
                1.0 - _OOD_EFFECT_SHIFT_COMBINED_CONTEXT_FLOOR,
                dtype=tf.float64,
            )
            * combined_context
        )
        combined_amplitude_by_effect = combined_amplitude_by_effect * combined_context
        combined_gate = tf.sigmoid(
            combined_effect_intercept
            + combined_effect_linear * effect_signal
            + combined_support_linear * positive_support
        )
        combined_support_gate = tf.sigmoid(
            (
                positive_support
                - tf.constant(
                    _OOD_EFFECT_SHIFT_COMBINED_SUPPORT_CENTER,
                    dtype=tf.float64,
                )
            )
            / tf.constant(
                _OOD_EFFECT_SHIFT_COMBINED_SUPPORT_WIDTH,
                dtype=tf.float64,
            )
        )
        pure_contribution = tf.minimum(
            pure_gate * pure_high_effect_taper * pure_amplitude_by_effect,
            tf.constant(_OOD_EFFECT_SHIFT_PURE_LOG_CAP, dtype=tf.float64),
        )
        combined_contribution = tf.minimum(
            combined_gate * combined_support_gate * combined_amplitude_by_effect,
            tf.constant(_OOD_EFFECT_SHIFT_COMBINED_LOG_CAP, dtype=tf.float64),
        )
        log_inflation = log_inflation + (pure_contribution + combined_contribution)
    return tf.clip_by_value(log_inflation, 0.0, np.log(float(max_multiplier)))


def _tf_normal_cdf(value: tf.Tensor) -> tf.Tensor:
    return 0.5 * (1.0 + tf.math.erf(value / tf.sqrt(tf.constant(2.0, tf.float64))))


def _tf_rank_moment_loss(
    rank_probability: tf.Tensor,
    groups: list[tf.Tensor],
    *,
    mean_tolerance: float,
    variance_tolerance: float,
) -> tf.Tensor:
    losses = []
    expected_variance = tf.constant(1.0 / 12.0, dtype=tf.float64)
    for group in groups:
        selected = tf.boolean_mask(rank_probability, group)
        rank_mean = tf.reduce_mean(selected)
        rank_variance = tf.reduce_mean(tf.square(selected - rank_mean))
        losses.append(
            tf.square((rank_mean - 0.5) / mean_tolerance)
            + tf.square((rank_variance - expected_variance) / variance_tolerance)
        )
    return tf.reduce_mean(tf.stack(losses))


def _rank_moment_loss(
    signed_standardized_error: np.ndarray,
    groups: list[np.ndarray],
    *,
    mean_tolerance: float,
    variance_tolerance: float,
) -> float:
    ranks = ndtr(signed_standardized_error)
    losses = []
    for group in groups:
        selected = ranks[group]
        losses.append(
            ((float(np.mean(selected)) - 0.5) / mean_tolerance) ** 2
            + ((float(np.var(selected)) - 1.0 / 12.0) / variance_tolerance) ** 2
        )
    return float(np.mean(losses))


def _coefficient_names(
    names: Sequence[str] | None, n_covariates: int
) -> tuple[str, ...]:
    result = (
        tuple(str(name) for name in names)
        if names is not None
        else tuple(f"coefficient_{index}" for index in range(n_covariates))
    )
    if len(result) != n_covariates:
        raise ValueError("coefficient_names length must match posterior covariates")
    return result


def _coefficient_mask(
    shape: tuple[int, ...], species_mask: np.ndarray | None
) -> np.ndarray:
    if species_mask is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(species_mask, dtype=bool)
    if mask.shape != (shape[0], shape[2]):
        raise ValueError("species_mask must have shape batch x species")
    return np.broadcast_to(mask[:, None, :], shape)


def _coverage(
    mean: np.ndarray,
    scale: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    z_value: float,
) -> float:
    covered = np.abs(mean - truth) <= z_value * scale
    return float(np.mean(covered[mask]))


def _scale_nll(error: np.ndarray, multiplier: np.ndarray) -> float:
    return float(np.mean(np.log(multiplier) + 0.5 * np.square(error / multiplier)))


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)
