# Amortized Neural-HMSC Implementation Roadmap

This roadmap describes a staged implementation plan for the first deep learning
research direction selected from `docs/deep_learning_jsdm_directions.md`:

```text
Amortized Neural-HMSC posterior inference
```

The goal is not to replace HMSC with a black-box neural predictor. The goal is
to train a neural inference engine that can approximate HMSC-shaped posterior
quantities much faster than repeated MCMC runs, while preserving the parameters
and summaries that make HMSC scientifically useful.

## Research Question

Can a neural model learn to map compiled HMSC data artifacts to calibrated
posterior distributions over HMSC-like parameters?

```text
(Y, X, traits, phylogeny, study design, random levels)
    -> neural inference model
    -> approximate samples for Beta, Gamma, Eta, Lambda, Sigma
    -> existing pyhmsc summaries, predictions, PPCs, diagnostics
```

The first implementation should target a narrow, testable subset:

```text
fixed-effect Gaussian / probit / Poisson HMSC
  -> posterior over Beta
  -> posterior mean and credible interval approximation
  -> posterior predictive checks using existing pyhmsc machinery
```

Only after this works should the project add traits, random effects, latent
associations, and spatial structure.

## Non-Goals

- Do not build a generic neural SDM as the primary contribution.
- Do not remove or weaken the existing Gibbs sampler path.
- Do not claim exact Bayesian equivalence to HMSC MCMC.
- Do not start with full spatial random effects or latent factor recovery.
- Do not make neural predictions the only validation target.

## Success Criteria

A milestone is successful only if it improves speed while maintaining acceptable
posterior behavior against known truth or MCMC references.

Minimum criteria for early prototypes:

- posterior means are close to MCMC posterior means on matched simulated tasks
- nominal 80%, 90%, and 95% intervals have reasonable empirical coverage
- posterior predictive means and intervals behave similarly to MCMC
- calibration does not collapse under held-out simulated datasets
- failure modes are detectable by diagnostics

Longer-term criteria:

- species association summaries from neural samples match identifiable MCMC
  association summaries when latent factors are included
- spatial holdout predictions retain calibration under blocked splits
- rare species behavior is not worse than the MCMC baseline by unacceptable
  margins
- inference time is materially lower than running MCMC for a comparable model

## Proposed Package Surface

Initial code should live under an experimental namespace so the stable HMSC path
remains untouched.

```text
pyhmsc/neural/
  __init__.py
  datasets.py
  simulator.py
  encoders.py
  posterior_heads.py
  models.py
  train.py
  calibration.py
  evaluation.py
  storage.py
```

Possible CLI entry points:

```bash
python -m pyhmsc neural simulate-training-data CONFIG --output DATASET
python -m pyhmsc neural train CONFIG --dataset DATASET --output MODEL
python -m pyhmsc neural infer MODEL INIT_JSON --output POSTERIOR
python -m pyhmsc neural evaluate MODEL BENCHMARK_CONFIG --output REPORT
```

The neural posterior output should eventually be readable by `HmscFit` or by a
thin `NeuralHmscFit` adapter with the same summary methods where possible.

## Design Principle: Reuse Existing Artifacts

The implementation should reuse the existing Python-native model artifacts
instead of defining a parallel data format.

Inputs already available:

- compiled model metadata JSON
- HDF5 arrays from `pyhmsc.compiler`
- simulated projects under `examples/projects`
- posterior HDF5 output from the Gibbs sampler
- posterior summaries and diagnostics in `pyhmsc.posterior`
- validation and holdout analyzers under `examples/`

This keeps the neural path comparable to the current sampler and avoids
inventing a separate ecosystem.

## Milestone 0: Design Lock and Baseline Audit

Purpose:

Confirm the smallest viable target and freeze the benchmark protocol before
adding neural code.

Status:

Implemented in this feature branch as the benchmark protocol and three
fixed-effect benchmark configuration directories:

```text
docs/neural_hmsc_benchmark_protocol.md
examples/projects/neural_hmsc_fixed_gaussian/
examples/projects/neural_hmsc_fixed_probit/
examples/projects/neural_hmsc_fixed_poisson/
```

Tasks:

- document the exact fixed-effect model family to target first
- identify which compiled arrays are required for fixed-effect inference
- define train/validation/test simulation regimes
- define MCMC reference settings for small benchmark datasets
- choose metrics for posterior accuracy, calibration, and speed
- decide whether the first implementation uses TensorFlow/Keras only or also
  TensorFlow Probability layers/distributions

Recommended first target:

```text
distribution: Gaussian, probit, and Poisson in separate sub-experiments
parameters: Beta only
traits: no
random effects: no
spatial effects: no
```

Deliverables:

- `docs/neural_hmsc_benchmark_protocol.md`
- small benchmark configuration files under `examples/projects/neural_hmsc_*`

Exit criteria:

- a reviewer can run the existing MCMC baseline and know exactly what the neural
  model must match

## Milestone 1: Simulation Dataset Generator

Purpose:

Create many small HMSC-like datasets with known parameters and controlled
variation.

Status:

Implemented in this feature branch for fixed-effect Gaussian, probit, and
Poisson benchmark corpora:

```text
pyhmsc/neural/datasets.py
pyhmsc/neural/simulator.py
examples/generate_neural_hmsc_training_data.py
tests/test_neural_hmsc_simulator.py
```

The generator writes per-dataset `Y.csv`, `X.csv`, `truth_beta.csv`,
`truth_linear_predictor.csv`, `model.yaml`, compiled `init.json`/HDF5 artifacts,
and a top-level `corpus_metadata.json`.

Tasks:

- add a neural training-data generator that calls or extends `pyhmsc.simulate`
- sample model dimensions from configurable ranges:
  - number of sites
  - number of species
  - number of covariates
  - signal-to-noise ratio
  - prevalence/count intensity regimes
  - coefficient sparsity or shrinkage level
- emit compiled JSON/HDF5 artifacts using `pyhmsc.compiler`
- emit true parameter files for supervised evaluation
- optionally emit short MCMC reference posteriors for a subset of datasets
- store generated data in a chunked format suitable for neural training

Likely files:

```text
pyhmsc/neural/simulator.py
pyhmsc/neural/datasets.py
examples/generate_neural_hmsc_training_data.py
tests/test_neural_hmsc_simulator.py
```

Design notes:

- Keep generated datasets small at first.
- Randomize dimensions so the model does not memorize one matrix shape.
- Include explicit seeds and metadata for reproducibility.
- Separate simulation truth from MCMC reference output.

Exit criteria:

- generator can produce a reproducible training corpus
- generated artifacts can be loaded by the existing sampler
- tests verify shape, metadata, and truth alignment

## Milestone 2: Fixed-Shape Beta Posterior Prototype

Purpose:

Prove the basic inference idea on the simplest possible task.

Status:

Implemented in this feature branch for fixed-shape Gaussian `Beta` posterior
prototyping:

```text
pyhmsc/neural/posterior_heads.py
pyhmsc/neural/models.py
pyhmsc/neural/train.py
pyhmsc/neural/evaluation.py
tests/test_neural_hmsc_beta_fixed_shape.py
```

The prototype consumes fixed-shape design/response tensors, encodes `X'Y`,
`X'X`, and response summaries, predicts a diagonal Normal posterior over
`Beta`, trains against simulated truth, and verifies non-collapsed posterior
scales plus improvement over the zero-mean baseline.

Scope:

```text
fixed n_sites
fixed n_species
fixed n_covariates
Gaussian response first
posterior target: Beta mean and scale
```

Model:

```text
Y, X
  -> tensor encoder
  -> posterior head
  -> diagonal Gaussian q(Beta)
```

Training objective options:

- supervised loss against known simulated `Beta`
- negative log probability of true `Beta` under predicted q
- optional posterior moment matching against short MCMC references

Likely files:

```text
pyhmsc/neural/models.py
pyhmsc/neural/posterior_heads.py
pyhmsc/neural/train.py
pyhmsc/neural/evaluation.py
tests/test_neural_hmsc_beta_fixed_shape.py
```

Exit criteria:

- learns nontrivial posterior mean estimates on held-out simulations
- uncertainty is not collapsed to near-zero
- posterior samples can be converted to a minimal posterior-like object

Failure criteria:

- model cannot outperform simple baselines such as ridge regression estimates
- intervals are systematically under-dispersed
- training only works for one hand-picked data shape

## Milestone 3: Variable-Shape Encoder

Purpose:

Move from a toy fixed-shape model to a reusable inference model over variable
numbers of sites and species.

Status:

Implemented in this feature branch for variable site/species counts with fixed
covariates and Gaussian fixed effects:

```text
pyhmsc/neural/train.py
pyhmsc/neural/models.py
pyhmsc/neural/evaluation.py
tests/test_neural_hmsc_variable_shape.py
```

The implementation pads variable-shape datasets, carries `site_mask` and
`species_mask`, predicts masked diagonal Normal `Beta` posteriors, and verifies
site-order invariance plus species-order equivariance.

Candidate architecture:

```text
site encoder:
  X_i plus transformed row of Y

species encoder:
  column summaries of Y

cross encoder:
  low-rank site/species interaction summaries

posterior head:
  Beta covariate/species parameter distribution
```

Recommended implementation:

- start with permutation-aware pooling rather than full transformer attention
- support masks for missing values
- normalize covariates and response summaries consistently
- expose deterministic shape checks and metadata checks

Possible later upgrade:

- Set Transformer or Perceiver-style cross attention

Exit criteria:

- one trained model handles several site/species/covariate dimensions
- predictions are invariant to site order
- predictions are equivariant to species order
- tests verify permutation behavior

## Milestone 4: Posterior Storage Adapter

Purpose:

Make neural posterior samples usable by existing pyhmsc analysis tools.

Tasks:

- define a minimal neural posterior HDF5 schema compatible with `HmscFit`
  where possible
- write predicted `Beta` samples as chains/draws/covariates/species
- store neural metadata:
  - model checkpoint path or hash
  - training corpus metadata
  - calibration method
  - inference seed
  - sampled draws
- add loader tests for posterior summaries
- verify `fit.beta_mean()`, `fit.beta_ci()`, `fit.predict_mean()`, and PPC paths
  where applicable

Design choice:

Prefer writing standard posterior samples over adding many neural-specific
summary methods. If the neural model produces samples in the existing shape,
the existing summary code remains the reference.

Exit criteria:

- neural posterior samples can be summarized with the same public methods used
  for MCMC posterior samples

## Milestone 5: MCMC Reference Benchmark

Purpose:

Compare neural posterior approximation against real MCMC output, not only
simulation truth.

Tasks:

- choose a small benchmark suite:
  - fixed Gaussian recovery
  - fixed probit recovery
  - fixed Poisson recovery
- run MCMC references with controlled samples/transient/thin
- compare:
  - posterior mean error
  - interval overlap
  - empirical coverage of truth
  - posterior predictive summaries
  - runtime
- emit markdown and CSV reports

Likely files:

```text
examples/run_neural_hmsc_benchmark.py
examples/analyze_neural_hmsc_benchmark.py
docs/neural_hmsc_benchmark_results_YYYY-MM-DD.md
```

Exit criteria:

- benchmark reports are reproducible locally for small cases
- longer benchmark can run on LUMI

## Milestone 6: Calibration Layer

Purpose:

Improve uncertainty behavior after the neural posterior model learns useful
means but imperfect intervals.

Candidate methods:

- temperature/scale calibration of posterior standard deviations
- empirical Bayes correction by dimension and distribution family
- conformal calibration for posterior predictive intervals
- ensemble of neural posterior models
- simulation-based calibration diagnostics

Tasks:

- add calibration split to generated training corpus
- implement calibration metadata in posterior output
- evaluate calibrated vs uncalibrated coverage
- fail loudly when the requested model is out of calibration domain

Exit criteria:

- interval coverage improves without destroying posterior mean accuracy
- calibration behavior is visible in benchmark reports

## Milestone 7: Traits and Gamma

Purpose:

Extend the approximate posterior from species-specific fixed effects to
trait-mediated structure.

Target:

```text
Beta | Gamma, traits
Gamma posterior
trait design matrix T
optional phylogenetic covariance C later
```

Tasks:

- include `T` from compiled artifacts
- add species trait encoder
- add posterior head for `Gamma`
- compare neural `Gamma` summaries with MCMC
- verify trait effects with simulated truth

Exit criteria:

- neural posterior recovers trait-mediated coefficients on simulated datasets
- `Gamma` summaries can be emitted through existing summary conventions

## Milestone 8: IID Random Effects and Latent Factors

Purpose:

Add the first random-effect structure while avoiding spatial complexity.

Target:

```text
iid random intercepts
Eta posterior
Lambda posterior
association summaries from Lambda.T @ Lambda
```

Tasks:

- encode study design and group membership
- predict posterior samples for `Eta` and `Lambda`
- handle factor sign/permutation non-identifiability
- prioritize identifiable association matrix recovery over raw factor recovery
- reuse existing association diagnostics

Important design note:

Raw `Eta` and `Lambda` are not uniquely identified. Training directly against
raw MCMC samples is fragile. The roadmap should prefer one or more of:

- train against identifiable `Eta @ Lambda` contributions
- train against `Lambda.T @ Lambda` association matrices
- post-hoc align factors before supervised losses
- use invariant losses for latent factors

Exit criteria:

- association summaries are close to MCMC on small iid random-effect tasks
- PPC improvements from iid random effects are reproduced qualitatively

## Milestone 9: Spatial Random Effects

Purpose:

Approximate HMSC spatial random-effect behavior after the iid path is stable.

Scope order:

1. full spatial
2. GPP
3. NNGP

Tasks:

- encode coordinates and random-level metadata
- approximate posterior over spatial `Eta`
- evaluate held-out spatial prediction under nearest and conditional modes
- compare residual spatial autocorrelation against MCMC baselines
- add spatially blocked validation splits

Potential architectures:

- coordinate MLP with Fourier features
- neural field for `Eta(s)`
- graph encoder over spatial neighbor structure
- low-rank spatial basis encoder

Exit criteria:

- spatial heldout predictions remain calibrated
- residual spatial autocorrelation improves over fixed/iid neural baselines
- speedup over spatial MCMC is material

## Milestone 10: LUMI Training and Benchmark Workflow

Purpose:

Move from local smoke tests to HPC-scale training and validation.

Tasks:

- add sbatch scripts for:
  - training-data generation
  - neural model training
  - MCMC reference generation
  - benchmark evaluation
- store large generated corpora outside the repo
- write reproducible run metadata
- support checkpoint resume
- log GPU utilization and wall time

Likely files:

```text
docs/lumi_neural_hmsc_train_sbatch.sh
docs/lumi_neural_hmsc_benchmark_sbatch.sh
```

Exit criteria:

- LUMI run can train the prototype and emit a benchmark report without manual
  intervention

## Milestone 11: Public API Stabilization

Purpose:

Decide which experimental pieces become supported API.

Candidate API:

```python
from pyhmsc.neural import NeuralHmscInference

engine = NeuralHmscInference.load("checkpoint")
fit = engine.infer(model_or_compiled_artifact, draws=1000, seed=1)
fit.beta_mean()
fit.beta_ci()
fit.predict_mean(X_new)
```

Tasks:

- finalize naming
- document limitations by model family
- add user-facing tutorial
- add compatibility checks for unsupported model structures
- define versioning for neural checkpoints and training corpora

Exit criteria:

- a user can run a documented end-to-end example without touching internal
  training code

## Testing Strategy

Unit tests:

- generated dataset shapes and metadata
- posterior head output shapes
- posterior sample writer/loader compatibility
- permutation invariance/equivariance checks
- calibration math

Integration tests:

- tiny train/infer loop over synthetic Gaussian data
- neural posterior samples readable by `HmscFit`
- benchmark script produces expected files

Slow tests:

- simulated fixed-effect recovery with multiple seeds
- MCMC comparison on small datasets
- iid random-effect association recovery
- spatial holdout recovery once implemented

## Evaluation Metrics

Posterior parameter metrics:

- posterior mean RMSE against truth
- posterior mean RMSE against MCMC mean
- interval coverage of truth
- interval width ratio against MCMC
- simulation-based calibration rank diagnostics

Predictive metrics:

- held-out log likelihood
- RMSE or classification metrics by distribution
- species PPC coverage
- site richness PPC coverage
- calibration curves for occurrence probabilities

Latent/random-effect metrics:

- recovery of `Eta @ Lambda`
- recovery of `Lambda.T @ Lambda`
- association sign probability agreement
- residual spatial autocorrelation

Operational metrics:

- training time
- inference time per dataset
- memory usage
- speedup over MCMC baseline

## Risks and Mitigations

Risk: posterior intervals are too narrow.

Mitigation:

- use calibration layer
- use ensembles
- validate with simulation-based calibration before claiming uncertainty

Risk: model memorizes simulation settings.

Mitigation:

- randomize dimensions and priors
- reserve out-of-distribution simulation regimes
- benchmark on real-data validation projects

Risk: latent factor targets are non-identifiable.

Mitigation:

- train on identifiable products or association matrices
- align factors only for diagnostics, not as the sole objective

Risk: neural path becomes a disconnected project.

Mitigation:

- use compiled artifacts as inputs
- write posterior samples in existing storage shapes
- evaluate through existing pyhmsc summaries and analyzers

Risk: predictive performance improves but inference quality fails.

Mitigation:

- separate predictive benchmarks from posterior benchmarks
- do not promote the model beyond "predictive neural JSDM" unless posterior
  calibration passes

## First Concrete Pull Request

The first implementation PR should be intentionally small.

Suggested scope:

- add `pyhmsc/neural/` package skeleton
- add simulation corpus generator for fixed Gaussian models
- add fixed-shape `Beta` posterior model
- add tiny training loop
- add posterior sample writer for `Beta`
- add tests proving:
  - generator emits valid compiled artifacts
  - model output has correct shape
  - posterior samples can be summarized

Out of scope for the first PR:

- traits
- phylogeny
- random effects
- spatial effects
- LUMI scripts
- publication-grade benchmarks

## Decision Points Before Coding

Before implementation starts, decide:

- TensorFlow-only or TensorFlow + TensorFlow Probability distribution layers?
- supervised training against simulated truth first, MCMC moments first, or both?
- posterior output as direct samples, distribution parameters, or both?
- whether to introduce `NeuralHmscFit` or write standard posterior HDF5 only
- which distribution family is the first target: Gaussian, probit, or Poisson
- how much API surface should be experimental/private in the first PR

Recommended choices:

- use TensorFlow plus TensorFlow Probability where it simplifies distributions
- start with Gaussian fixed-effect `Beta`
- train against simulated truth first
- write standard posterior samples as early as possible
- keep user-facing API minimal until calibration is proven
