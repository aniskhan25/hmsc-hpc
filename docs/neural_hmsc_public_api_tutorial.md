# Neural-HMSC Public API Tutorial

Milestone 11 stabilizes a small public API around the experimental neural
prototype. The supported surface is intentionally narrow:

- fixed-effect `Beta` posterior inference
- fixed site/covariate/species shape per checkpoint
- `normal`, `probit`, and `poisson` fixed-effect benchmark distributions
- outputs readable by `pyhmsc.posterior.HmscFit`
- distribution-aware `Beta` posterior families by default

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
    posterior_family="auto",
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
- probit anchor type and numerical settings
- distribution, formula, dimensions, and names
- public limitations for the checkpoint family

The IRLS/Laplace-aware encoder uses checkpoint version `0.3`; training corpus
version remains `0.1`. Version `0.2` checkpoints remain loadable with their
legacy ridge anchor.
Future incompatible checkpoint formats must bump `checkpoint_version`. Changes
to generated training-data semantics must bump `training_corpus_version`.

Checkpoints created before `posterior_family` was added are loaded as
`diagonal_normal`. New fixed-effect engines default to `auto`: Poisson resolves
to `full_covariance_normal`, while Gaussian and probit resolve to
`diagonal_normal`. Explicit family overrides remain available. Full-covariance
models store correlated draws in the existing `Beta` HDF5 sample shape.

Poisson checkpoints use `log1p(Y)` encoder summaries before constructing the
ridge anchor. Gaussian and probit checkpoints retain identity response
features. New probit checkpoints use a penalized probit IRLS mode as the mean
anchor and include log Laplace marginal scales in encoder features. The
manifest records `probit_anchor`, iterations, prior precision, and eta clip.
`probit_anchor: auto` selects IRLS/Laplace only for probit. Version `0.2`
checkpoints without these fields load with the exact legacy ridge architecture.

## Coefficient and Predictive Calibration

`apply_beta_scale_calibration` applies only the coefficient-coverage multiplier
and returns the Beta posterior used for inference. For Poisson and probit
models, `apply_beta_predictive_calibration` may apply a separately fitted
predictive multiplier for response-scale scoring. Probit predictive calibration
uses the exact Gaussian-probit expectation. Its output is a predictive-only
surrogate and must not replace, or be reported as, Beta posterior uncertainty.

Calibration metadata uses `semantics_version: 2`, with separate
`scale_multiplier` and optional `predictive_scale_multiplier` fields. Loading
legacy Poisson metadata recovers `coverage_scale_multiplier` as the Beta scale
and preserves the old applied scale as predictive-only.

Milestone 12 adds `fit_conditional_beta_scale_calibration` and
`apply_conditional_beta_scale_calibration` for simulation-trained coefficient
scaling. Current anchored-model conditional artifacts use
`semantics_version: 6`, store their rank-aware objective, feature normalization,
structured-head weights, support geometry, and bounded OOD uncertainty
inflation, and require `X`, `Y`, and matching coefficient names when applied.
They use prevalence, expected design information, raw posterior scale,
posterior-mean magnitude, and coefficient identity.
`conditional_beta_support_trust` exposes the coefficient-level blend weight;
zero trust applies the frozen scalar fallback before any configured bounded OOD
uncertainty inflation.
Full-covariance posteriors are transformed as `D Sigma D`; posterior means are
unchanged. Version 3, 4, and legacy version 5 conditional metadata remain
readable.

The conditional artifact applies only to coefficient uncertainty. Continue to
use the separately fitted `BetaScaleCalibration` with
`apply_beta_predictive_calibration` for predictive-only scoring.

For cross-dataset transfer, load the existing checkpoint with
`NeuralHmscInference.load(...)` and reconstruct calibration from the stored
coefficient-posterior metadata with `BetaScaleCalibration.from_metadata(...)`
or `ConditionalBetaScaleCalibration.from_metadata(...)`, according to its
method. Do not fit either calibrator on the target dataset. Record content
hashes for both frozen artifacts and keep target holdout responses out of all
projection and species-selection decisions.

## SBC and OOD Diagnostics

`beta_sbc_rank_diagnostics(samples, truth)` accepts posterior samples with
shape `replicates x draws x covariates x species` and matching simulated truths.
It returns JSON-ready metadata and flat report fields for rank uniformity,
tail behavior, posterior-mean RMSE, and 80/90/95 percent coverage.

`beta_sbc_stratified_diagnostics(samples, truth, X=..., Y=...,
distribution=...)` adds one overall row plus prevalence, named-coefficient, and
expected design-information rows. Prevalence is computed per simulated
dataset/species. Information is evaluated at the raw posterior mean so it does
not depend on simulated truth at deployment. Qualification code must continue
to select the `sbc_stratum_kind == "overall"` row; stratified rows diagnose
where calibration fails and guide conditional calibration.

The benchmark runner generates independent in-domain simulations plus named
`covariate_shift`, `effect_size_shift`, and `combined_shift` datasets. It writes
`neural_hmsc_sbc_diagnostics.{csv,json,md}` and per-distribution
`sbc_diagnostics.json` files. These are stress diagnostics: OOD calibration is
measured, not assumed to retain in-domain validity.

Benchmark qualification requires predictive acceptance and SBC
non-degradation. A checkpoint that improves predictive RMSE while worsening
coefficient coverage or rank behavior is not qualified.

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
