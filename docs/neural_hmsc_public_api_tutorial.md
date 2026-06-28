# Neural-HMSC Public API Tutorial

Milestone 11 stabilizes a small public API around the experimental neural
prototype. The supported surface is intentionally narrow:

- fixed-effect `Beta` posterior inference
- fixed site/covariate/species shape per checkpoint
- `normal`, `probit`, and `poisson` fixed-effect benchmark distributions
- outputs readable by `pyhmsc.posterior.HmscFit`
- per-species full-covariance Normal `Beta` posteriors by default

The API does not yet support trait-mediated `Gamma`, phylogeny, iid latent
factors, spatial latent factors, random effects, random slopes, detection
models, or variable-shape checkpoints. Those pieces remain experimental
modules until their semantics and validation are stable.

## Train, Save, Load, Infer

```python
from pyhmsc.neural import NeuralHmscInference, simulate_fixed_effect_dataset

train = [
    simulate_fixed_effect_dataset(
        n_sites=32,
        n_species=3,
        distribution="normal",
        seed=100 + idx,
    )
    for idx in range(16)
]
test = simulate_fixed_effect_dataset(
    n_sites=32,
    n_species=3,
    distribution="normal",
    seed=999,
)

engine = NeuralHmscInference.for_fixed_effects(
    n_sites=32,
    n_species=3,
    distribution="normal",
    formula="~ x1 + x2",
    posterior_family="full_covariance_normal",
)
engine.fit(train, epochs=40, batch_size=8)
engine.save("run/neural_checkpoint")

engine = NeuralHmscInference.load("run/neural_checkpoint")
fit = engine.infer(
    test,
    draws=1000,
    chains=2,
    seed=1,
    output="run/neural_posterior.h5",
)

beta_mean = fit.beta_mean()
beta_ci = fit.beta_ci(level=0.95)
prediction = fit.predict_mean(test.X)
```

`fit` is a normal `HmscFit`, so existing posterior summaries, prediction
helpers, diagnostics that apply to `Beta`, and HDF5 storage inspection continue
to work.

## Infer From A Compiled Artifact

The same checkpoint can infer from a Python-native compiled `init.json` or its
containing directory:

```python
from pyhmsc.neural import NeuralHmscInference

engine = NeuralHmscInference.load("run/neural_checkpoint")
fit = engine.infer(
    "compiled/init.json",
    draws=1000,
    chains=2,
    output="run/neural_from_compiled.h5",
)
```

The compiled artifact must match the checkpoint dimensions and must be a
fixed-effect model. Unsupported structures raise
`NeuralHmscCompatibilityError` before inference starts.

## Checkpoint Versioning

Each checkpoint directory contains:

```text
neural_checkpoint.json
weights.weights.h5
```

The manifest records:

- `checkpoint_version`
- `training_corpus_version`
- `model_family`
- `posterior_family`
- distribution, formula, dimensions, and names
- public limitations for the checkpoint family

Milestone 11 uses checkpoint version `0.1` and training corpus version `0.1`.
Future incompatible checkpoint formats must bump `checkpoint_version`. Changes
to generated training-data semantics must bump `training_corpus_version`.

Checkpoints created before `posterior_family` was added are loaded as
`diagonal_normal`. New fixed-effect checkpoints default to
`full_covariance_normal`, which stores correlated draws in the existing
`Beta` HDF5 sample shape.

## Compatibility Limits

`NeuralHmscInference` currently rejects compiled artifacts with any of:

- random levels
- spatial random effects
- trait effects
- phylogeny
- unsupported distributions
- dimensions that differ from the checkpoint

This is deliberate. The neural output should not imply HMSC uncertainty
semantics for model families that have not yet been validated through the
benchmark protocol.
