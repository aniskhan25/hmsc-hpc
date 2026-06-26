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
    "generate_fixed_effect_corpus",
    "simulate_fixed_effect_dataset",
]

