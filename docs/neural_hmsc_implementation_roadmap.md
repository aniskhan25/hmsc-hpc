# Amortized Neural-HMSC Implementation Roadmap

> **Current outcome (2026-07-27):** the completed branch qualifies bounded
> fixed-effect probit neural approximations, not structural Neural HMSC or
> MCMC near-equivalence. Neural model-family development on this branch is
> closed after terminal trait, variable-design, covariance, and joint
> Student-t results. See
> `docs/neural_hmsc_branch_closure_audit_2026-07-27.md`. Any new structural
> attempt requires a new branch and fresh generative-model preregistration.

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

## Expected Outcome and Meaning of MCMC Equivalence

The neural method is amortized approximate inference. It will not reproduce the
MCMC algorithm, finite-chain draws, or exact Bayesian posterior. The roadmap
therefore distinguishes four claims that must not be collapsed into one:

1. **Interface equivalence** means neural output uses the existing HDF5 schema
   and can be consumed by `HmscFit` summaries and prediction APIs. This is
   already achieved for the supported fixed-effect `Beta` path.
2. **Operational equivalence** means a pretrained checkpoint accepts the same
   supported compiled model boundary, emits the same named parameter families,
   and provides materially faster inference with explicit compatibility
   rejection outside its scope. This is the target for Neural-HMSC v0.1.
3. **Summary-level near-equivalence** means predeclared posterior and predictive
   summaries remain within quantitative tolerances of qualified MCMC over
   independent simulations and real-data diagnostics. This must be earned
   separately for each model family; it is not yet established generally.
4. **Joint-posterior equivalence** would require matching posterior dependence,
   multimodality, latent-factor uncertainty, and all derived HMSC semantics.
   This is not promised by the current architecture and is not a v0.1 goal.

Following this roadmap will first produce a fast, documented fixed-effect
approximation, not a drop-in statistical replacement for MCMC. Later versions
may qualify as summary-level near-equivalent for specific fixed, trait, iid, or
spatial families if they pass their family-specific gates. MCMC remains the
reference and fallback whenever a model lies outside the qualified domain or a
near-equivalence gate fails.

For a supported family, a future summary-level near-equivalence claim requires
all of the following on frozen independent evaluations:

- coefficient posterior means track MCMC and known simulation truth under a
  predeclared error/correlation tolerance;
- 80%, 90%, and 95% marginal intervals pass SBC coverage and rank gates;
- interval widths and overlap with MCMC do not show systematic collapse or
  uncontrolled inflation;
- held-out Brier/log-loss or distribution-appropriate proper scores remain
  within a predeclared tolerance of MCMC;
- identifiable association or spatial summaries pass when those parameter
  families are included;
- inference is materially faster after checkpoint training is amortized.

Passing only prediction, coverage, or file compatibility is insufficient for a
general MCMC-equivalence claim.

The default frozen thresholds for a future fixed-effect summary-level
near-equivalence decision are:

- posterior-mean correlation with qualified MCMC at least `0.95`, with truth
  RMSE no more than `1.10` times the MCMC truth RMSE on simulations;
- 95% SBC coverage between `0.925` and `0.975`, normalized-rank mean error no
  greater than `0.025`, and rank-variance error no greater than `0.025`;
- median MCMC/neural marginal interval-width ratio between `0.80` and `1.25`
  and median 95% interval overlap at least `0.80`;
- held-out proper-score ratios no greater than `1.05` relative to qualified
  MCMC on every declared real-data benchmark;
- inference-only speedup at least `10x`, reported together with training cost
  and amortization break-even.

These thresholds must be frozen before the qualifying evaluation. A structural
family may declare stricter thresholds, but may not weaken them after observing
test or real-data outcomes.

Current evidence does not satisfy this near-equivalence definition. Interface
equivalence is achieved and operational equivalence is close for fixed effects,
but Big Spatial neural/MCMC proper-score ratios remain about `1.079/1.073` and
the previously measured 95% coefficient-interval overlap was `0.1648` despite
useful posterior-mean correlation. Neural-HMSC v0.1 must therefore be labelled
an amortized approximate alternative, not near-equivalent MCMC.

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

Status:

Implemented in this feature branch for single-dataset fixed-effect `Beta`
posterior storage:

```text
pyhmsc/neural/storage.py
tests/test_neural_hmsc_storage.py
```

The adapter samples from a neural diagonal Normal `Beta` posterior, writes HDF5
using the existing `chains x draws x covariates x species` shape, records
neural inference metadata in `pyhmsc_metadata`, and verifies that `HmscFit` plus
storage inspection can read the result.

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

Status:

Implemented in this feature branch for fixed-effect `Beta` posterior reference
comparisons:

```text
pyhmsc/neural/benchmark.py
examples/run_neural_hmsc_benchmark.py
examples/analyze_neural_hmsc_benchmark.py
tests/test_neural_hmsc_benchmark.py
```

The benchmark module compares neural posterior HDF5 files with MCMC reference
HDF5 files through `HmscFit`, reports posterior mean/sd agreement, credible
interval overlap, truth recovery, optional fixed-effect predictive summaries,
runtime, and speedup. The runner can generate the small Gaussian/probit/Poisson
suite locally and optionally launch Python-native MCMC references; the analyzer
compares already-produced posterior files, which is the intended path for
longer LUMI runs.

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

Status:

Implemented in this feature branch for post-hoc scalar temperature/scale
calibration of diagonal Normal fixed-effect `Beta` posteriors:

```text
pyhmsc/neural/calibration.py
tests/test_neural_hmsc_calibration.py
```

The calibrator fits a posterior scale multiplier on a calibration split using
standardized absolute `Beta` errors at a requested nominal credible level. It
preserves posterior means, rescales posterior standard deviations, stores
calibration method/domain/coverage metadata in neural HDF5 output, and raises
on response-family or shape mismatches. The benchmark runner writes both
uncalibrated and calibrated posterior files and exposes calibration metadata in
CSV/Markdown reports.

Post-milestone Poisson hardening replaces the public fixed-effect diagonal
posterior for Poisson with a per-species full-covariance Normal parameterized
by Cholesky factors. Gaussian and probit retain diagonal posteriors after scaled
validation showed a Gaussian regression under universal full covariance.
Poisson calibration now keeps two explicit semantics. The coefficient
`scale_multiplier` targets nominal Beta coverage and is the only multiplier
written to `neural_posterior.h5`. A separate `predictive_scale_multiplier` is
selected by balancing independent simulation-replicate log score against
truth-rate RMSE and is written to a clearly labelled predictive-only artifact.
It never replaces Beta uncertainty. Benchmark reports include eta-clipping,
predictive acceptance, SBC non-degradation, and a combined qualification gate.

Multi-seed qualification exposed Poisson sensitivity caused by applying the
Gaussian raw-count ridge anchor to log-link data. The fixed-shape Poisson
encoder now uses `log1p(Y)` response summaries and checkpoint version `0.2`.
Across LUMI seeds 20260626 through 20260630, the calibrated Poisson predictive
RMSE averaged 1.3616 versus 1.1897 for MCMC, with a worst neural-to-MCMC ratio
of 1.287 and five of five predictive checks passing. Those historical runs
predate the SBC qualification gate and must not be described as fully
qualified. Before the encoder change,
the same seeds averaged 4.0748 with a worst ratio of 9.244. Gaussian and probit
results were unchanged by the distribution-specific transform.

Split-calibration validation on LUMI job `19637239` used the previously
problematic seed 20260627. The coefficient multiplier was 1.507 and the
predictive-only multiplier was 0.299. Independent SBC coverage improved from
86.2% to 95.1%, with calibrated rank variance 0.067 versus the discrete-uniform
expectation 0.083 and no rank-error degradation. Predictive RMSE improved from
2.045 to 1.601 versus MCMC 1.515. Predictive acceptance, SBC acceptance, and
combined qualification all passed.

Candidate methods:

- temperature/scale calibration of posterior standard deviations
- empirical Bayes correction by dimension and distribution family
- conformal calibration for posterior predictive intervals
- ensemble of neural posterior models
- simulation-based calibration diagnostics

Simulation-based calibration rank diagnostics are now implemented in
`pyhmsc/neural/diagnostics.py`. The benchmark evaluates independent simulated
datasets with randomized tie-aware ranks and reports rank histograms, expected
discrete-rank counts, rank moments, tail imbalance, coefficient-level bias,
chi-square uniformity diagnostics, posterior-mean RMSE, and interval coverage.
The chi-square p-value is descriptive because coefficients within a simulated
community are not independent tests.

The same benchmark evaluates three explicit OOD regimes without retraining:

- `covariate_shift`: covariate mean 2.0 and standard deviation 1.5
- `effect_size_shift`: coefficient standard deviation 1.5
- `combined_shift`: both changes together

Each OOD row records posterior-mean RMSE relative to the matching in-domain
posterior variant. Generated corpora may also request these regimes under
`simulation.ood`, and LUMI runs expose `SBC_DATASETS`, `SBC_DRAWS`, `SBC_BINS`,
and `OOD_REGIMES` controls.

Tasks:

- add calibration split to generated training corpus
- implement calibration metadata in posterior output
- evaluate calibrated vs uncalibrated coverage
- fail loudly when the requested model is out of calibration domain
- run SBC ranks on independent in-domain and OOD simulation batches

Exit criteria:

- interval coverage improves without destroying posterior mean accuracy
- calibration behavior is visible in benchmark reports

## Milestone 7: Traits and Gamma

Purpose:

Extend the approximate posterior from species-specific fixed effects to
trait-mediated structure.

Status:

Implemented in this feature branch for fixed-shape trait-mediated `Gamma`
posterior prototypes:

```text
pyhmsc/neural/simulator.py
pyhmsc/neural/models.py
pyhmsc/neural/storage.py
tests/test_neural_hmsc_traits_gamma.py
```

The implementation simulates species traits with `Beta = Gamma @ T.T`, reads
compiler-emitted `T` arrays into neural training data, predicts a diagonal
Normal `Gamma` posterior with a trait encoder, writes `Gamma` samples in the
existing `chains x draws x covariates x traits` HDF5 shape, and compares neural
`Gamma` summaries against MCMC-style reference posterior files. Phylogenetic
covariance remains explicitly deferred.

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

Status:

Implemented in this feature branch for fixed-shape iid random-intercept latent
factor prototypes:

```text
pyhmsc/neural/simulator.py
pyhmsc/neural/models.py
pyhmsc/neural/storage.py
tests/test_neural_hmsc_iid_latent.py
```

The implementation simulates iid group membership, `Eta`, `Lambda`, and
low-rank residual contributions; encodes group membership in neural training
data; predicts `Beta`, `Eta`, and `Lambda` with an SVD-style residual encoder;
writes samples to the existing `random_levels/0/Eta` and
`random_levels/0/Lambda` HDF5 layout; and compares identifiable association
summaries from `Lambda.T @ Lambda`. Raw factor sign/permutation is treated as
non-identifiable, so tests prioritize contribution and association recovery.

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

Status:

Implemented in this feature branch for a first full-spatial random-intercept
latent-factor prototype:

```text
pyhmsc/neural/simulator.py
pyhmsc/neural/models.py
pyhmsc/neural/evaluation.py
pyhmsc/neural/storage.py
tests/test_neural_hmsc_spatial_latent.py
```

The implementation simulates coordinates, spatial `Eta`, `Lambda`, blocked
held-out sites, and residual spatial contributions; encodes coordinates in
spatial training data; predicts `Beta`, spatial `Eta`, and `Lambda` with a
coordinate-kernel residual encoder; evaluates nearest and conditional held-out
spatial interpolation; reports residual nearest-neighbor autocorrelation; and
writes samples to the existing random-level posterior layout with
`spatial_full` metadata. GPP and NNGP variants remain deferred.

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

Status:

- Implemented `docs/lumi_neural_hmsc_train_sbatch.sh` for scratch-backed neural
  prototype training with checkpoint-style reuse of existing posterior
  artifacts.
- Implemented `docs/lumi_neural_hmsc_benchmark_sbatch.sh` for end-to-end
  neural plus Python-native MCMC reference benchmarking and report emission.
- Extended `examples/run_neural_hmsc_benchmark.py` to write reproducible run
  metadata, reuse existing per-distribution outputs with `--skip-existing`, and
  resume missing MCMC references when neural artifacts already exist.
- Added GPU utilization and wall-time logging to both LUMI workflows.

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

Status:

- Added `NeuralHmscInference` as the first stable public facade for fixed-shape
  fixed-effect `Beta` neural posterior inference.
- Added versioned checkpoint manifests with explicit checkpoint and training
  corpus versions.
- Added compatibility checks that reject unsupported compiled model structures
  before inference.
- Added `docs/neural_hmsc_public_api_tutorial.md` with a documented
  train/save/load/infer workflow and explicit limitations.

## Milestone 12: Conditional Coefficient Calibration

Purpose:

Replace the single global probit coefficient-scale multiplier with a
simulation-trained conditional calibration model that can represent different
uncertainty corrections for rare and common species and for intercept and
environmental coefficients.

Why this is next:

- Whittaker predictive-only calibration transferred successfully to an
  independent ecological dataset, so the predictive path is now a frozen
  baseline rather than the immediate blocker.
- Whittaker calibrated SBC coverage reached 95.5%, but rank variance remained
  `0.05081` versus the `0.08333` expectation and the rank histogram remained
  strongly nonuniform.
- On the independent Big Spatial Plant dataset, coefficient posterior means
  correlated `0.9322` with MCMC, while 95% interval overlap was only `0.1648`.
  This indicates useful mean recovery but poorly transferred uncertainty.
- Applying the global `5.241370` coefficient scale to predictions degraded both
  real-data benchmarks, confirming that coefficient and predictive calibration
  must remain separate.

Implementation direction:

- keep the current `0.934528` probit predictive-only scale and both real-data
  workflows unchanged as regression baselines
- add coefficient-level SBC summaries stratified by training prevalence,
  coefficient type, and design information
- derive calibration features only from simulated training/calibration data,
  including species prevalence, effective sample size, raw posterior scale,
  coefficient identity, and fixed-effect design curvature
- learn positive coefficient-specific scale corrections with a small monotone
  calibrator or structured scale head; do not alter posterior means
- preserve covariance structure when scaling full-covariance posterior families
- add a probit-aware IRLS/Laplace encoder anchor only if conditional scale
  correction cannot satisfy the coefficient gates
- never fit coefficient calibration from Whittaker, Big Spatial, or their
  MCMC reference posteriors

Acceptance gates:

- at least 90% empirical 95% coverage overall and in every sufficiently
  populated prevalence stratum
- absolute normalized-rank mean error no greater than `0.025`
- absolute normalized-rank variance error no greater than `0.015`
- no material degradation in posterior-mean RMSE relative to the uncalibrated
  checkpoint
- improvement over the global scalar in multi-seed in-domain and OOD SBC
- no regression in frozen Whittaker or Big Spatial predictive-only metrics
- real-data MCMC interval overlap and coefficient-SD RMSE reported as transfer
  diagnostics, never used to fit the calibrator

Deliverables:

```text
pyhmsc/neural/conditional_calibration.py
tests/test_neural_hmsc_conditional_calibration.py
examples/run_neural_hmsc_conditional_calibration.py
docs/neural_hmsc_conditional_calibration_YYYY-MM-DD.md
```

Status:

- Active milestone.
- Implemented mask-aware coefficient SBC diagnostics and stratified report rows
  for prevalence, coefficient identity, and expected design-information
  quantiles.
- Integrated stratified rows into the general in-domain/OOD benchmark and the
  Whittaker shape-matched SBC workflow. Qualification remains tied to the
  `overall` row.
- Completed the five-seed global scalar baseline on LUMI for in-domain,
  covariate-shift, effect-size-shift, and combined-shift simulations. The
  detailed record is `docs/neural_hmsc_stratified_sbc_baseline_2026-07-01.md`.
- Scalar calibration produced in-domain 95% coverage `0.9420 +/- 0.0034`, but
  rank-variance error was `0.0243 +/- 0.0015` and failed the `0.015` gate in
  all five seeds. Rare and intermediate prevalence rank-mean errors were
  `0.1070` and `0.0680`; rare coverage passed in only one seed.
- Locked the first conditional architecture as a regularized structured
  residual scale head over prevalence, raw posterior scale, coefficient
  identity, expected design information, and a prevalence-by-coefficient
  interaction. Posterior means remain fixed and covariance scaling uses
  `D Sigma D`.
- Implemented the conditional calibration module, serializable version 3
  metadata, exact diagonal and full-covariance application, a dedicated
  benchmark entry point, and end-to-end coefficient/predictive semantics tests.
  Implementation details are recorded in
  `docs/neural_hmsc_conditional_calibration_2026-07-01.md`.
- Completed the frozen-checkpoint five-seed comparison on LUMI. Uncalibrated
  diagnostics reproduced the scalar runs exactly. Conditional scaling reduced
  in-domain rank-variance error from `0.0243` to `0.0055`, but rare coverage
  was `0.8718`, rare/intermediate rank-mean errors worsened to `0.1291` and
  `0.0823`, and every OOD regime degraded relative to scalar calibration. The
  comparison is recorded in
  `docs/neural_hmsc_conditional_comparison_2026-07-02.md`.
- The first conditional architecture is not qualified. Next substep: add a
  rank-aware, prevalence-weighted calibration objective and a support-aware
  gate that shrinks conditional adjustments to the scalar multiplier under
  feature shift, then repeat the same frozen five-seed comparison.
- Implemented the rank-aware revision with analytic posterior-CDF rank moments,
  explicit rare/intermediate prevalence weighting, robust feature bounds,
  regularized Mahalanobis support, and log-scale fallback to the frozen scalar
  multiplier. Version 3 artifacts remain readable; new artifacts use
  `semantics_version: 4` and expose support-trust diagnostics in SBC rows.
- Completed the frozen five-seed version 4 comparison on LUMI. Rare coverage
  improved from `0.8718` to `0.9372`, but rare/intermediate rank-mean errors
  remained `0.0917` and `0.0643`, intercept rank-variance error was `0.0189`,
  and OOD results did not improve over scalar calibration. The result is
  recorded in `docs/neural_hmsc_rankaware_v4_comparison_2026-07-09.md`.
- Version 4 is not qualified. Scale-only conditional calibration is now
  exhausted. Next substep: add the reserved probit-aware IRLS/Laplace encoder
  anchor to improve amortized posterior means, add posterior-mean magnitude to
  support diagnostics, and repeat the frozen five-seed comparison.
- Implemented the penalized probit IRLS mode and Laplace information anchor for
  new probit amortizers. New checkpoints use version `0.3`; version `0.2`
  checkpoints retain the exact ridge architecture. Anchored calibration uses
  semantics version 5 and adds posterior-mean magnitude to support fallback.
  Implementation details are recorded in
  `docs/neural_hmsc_irls_laplace_anchor_2026-07-09.md`.
- Completed the paired five-seed IRLS/Laplace LUMI retraining comparison against
  frozen scalar and version 4 references. Version 5 IRLS passed all overall
  in-domain gates and improved in-domain `Beta` RMSE from roughly `0.59` to
  `0.325`, but it regressed OOD coefficient coverage and rank-variance under
  covariate and combined shifts. The result is recorded in
  `docs/neural_hmsc_irls_v5_comparison_2026-07-10.md`.
- Version 5 IRLS is not qualified as the default calibration path. Keep the
  IRLS/Laplace anchor as an experimental candidate, but decouple its sharper
  posterior means from the OOD uncertainty scale. Next substep: add an
  OOD-aware variance inflation or stronger support-distance uncertainty term on
  top of the IRLS anchor, then repeat the paired five-seed in-domain/OOD LUMI
  comparison against the frozen scalar, version 4, and version 5 IRLS
  references.
- Implemented version 6 OOD-aware support-excess uncertainty inflation and
  completed the paired five-seed LUMI comparison. Version 6 preserved all
  in-domain gates and improved version 5 IRLS OOD coverage and rank-variance
  errors, but every OOD regime still failed the absolute 95% coverage gate. The
  result is recorded in `docs/neural_hmsc_oodv6_comparison_2026-07-12.md`.
- Version 6 OOD is not qualified as the default calibration path. Next substep:
  run a small LUMI tuning sweep over OOD uncertainty strength and maximum
  multiplier, then compare against scalar, version 4, version 5 IRLS, and the
  current version 6 default.
- Completed the version 6 OOD tuning sweep with strength/cap settings
  `1.0/8.0`, `1.5/8.0`, and `1.5/12.0`. The jobs retrained because the
  frozen-checkpoint environment variables did not propagate through Slurm, so
  the result is interpreted as a retrained tuning sweep rather than a pure
  frozen-checkpoint comparison. All three settings preserved in-domain gates
  and improved OOD coverage versus version 6 default, but every OOD regime
  still failed the absolute 95% coverage gate. The result is recorded in
  `docs/neural_hmsc_oodv6_sweep_comparison_2026-07-12.md`.
- Fixed support-excess multiplier tuning is now exhausted. Next substep:
  implement an explicit OOD calibration objective that learns regime-aware
  uncertainty inflation from held-out OOD simulations while keeping the
  in-domain coefficient SBC gates as hard acceptance constraints.
- Implemented the opt-in version 7 learned OOD calibration objective. It fits a
  bounded softplus support-excess inflation curve from held-out OOD simulation
  batches, records OOD objective metadata, and keeps posterior means fixed. The
  benchmark runner can now generate OOD calibration batches with
  `--conditional-calibration-ood-objective support_excess_rank_coverage` and
  `--conditional-calibration-ood-datasets`. Next substep: run a production-like
  local sanity check if feasible, then submit the five-seed LUMI comparison
  against scalar, version 4, version 5 IRLS, version 6 default, and the
  conservative version 6 strength-1.5/cap-8 candidate.
- Completed a production-shape local sanity run at `40` sites and `75` species
  with all three OOD regimes. Version 7 metadata, predictive/coefficient
  calibration separation, artifact writing, and SBC row generation worked, and
  overall in-domain SBC passed. The rare-prevalence in-domain rank-mean gate
  failed locally, so the next substep is to strengthen the rare-prevalence
  in-domain gate in the OOD objective or increase the in-domain gate weight
  before spending a full five-seed LUMI comparison. The result is recorded in
  `docs/neural_hmsc_v7_local_sanity_2026-07-12.md`.
- Submitted the requested five-seed version 7 LUMI comparison on `standard-g`
  despite the local rare-prevalence warning. Job IDs are `19831708`,
  `19831709`, `19831710`, `19831711`, and `19831712`.
- Completed and aggregated the five-seed version 7 LUMI comparison. Version 7
  preserved all overall and stratified in-domain gates, including rare
  prevalence, and improved OOD coverage/rank-variance over both version 6
  default and the version 6 strength-1.5/cap-8 sweep candidate. It still failed
  the absolute OOD coverage gate in every OOD regime, so it is not qualified as
  the default calibration path. The result is recorded in
  `docs/neural_hmsc_v7_lumi_comparison_2026-07-12.md`.
- Support-excess-only OOD calibration is now likely exhausted. Next substep:
  add regime-specific OOD calibration features or an effect-size-shift detector,
  because effect-size shift retains high support trust and remains
  under-inflated.
- Implemented an effect-size-aware revision of the version 7 OOD objective. The
  learned OOD inflation curve now uses both support excess and positive
  standardized posterior-mean magnitude, while legacy support-only version 7
  metadata remains loadable. SBC rows now expose effect-size signal summaries.
- Completed the effect-size-aware version 7 production-shape local sanity run.
  The overall in-domain acceptance gate passed, the calibration record used
  `support_effect_learned_softplus`, and the effect-size signal separated
  effect-size shift from covariate shift. The result is recorded in
  `docs/neural_hmsc_v7_effect_size_local_sanity_2026-07-13.md`.
- Initially submitted the five-seed effect-size-aware version 7 LUMI comparison
  on `standard-g`, matching the support-only version 7 comparison, but those
  jobs remained pending due to priority. The pending jobs were canceled and the
  run was moved to `dev-g`, which is better suited to the roughly 12 to 13
  minute workflow. Jobs `19835554`, `19835555`, `19835716`, `19835717`, and
  `19835779` completed for seeds `20260626` through `20260630`.
- Completed and aggregated the five-seed effect-size-aware version 7 LUMI
  comparison. The effect-size signal substantially improved the intended
  effect-size-shift regime, raising coefficient coverage from `0.7715` to
  `0.8481` versus support-only version 7 and reducing rank-variance error from
  `0.0366` to `0.0146`. Combined-shift coverage also improved from `0.6278` to
  `0.6439`. The candidate still fails the OOD coverage requirement and weakens
  the high-design-information in-domain stratum, which passed coverage in only
  `3/5` seeds. The result is recorded in
  `docs/neural_hmsc_v7_effect_size_lumi_comparison_2026-07-13.md`.
- Effect-aware version 7 is not qualified as the default calibration path. Next
  substep: make the effect-size branch OOD-context-aware, for example by gating
  effect-size inflation by support trust, prevalence/design strata, or an
  explicit in-domain penalty on the effect-size branch.
- Implemented an opt-in version 8 OOD objective,
  `support_effect_gated_rank_coverage`. Version 8 writes
  `support_effect_gated_learned_softplus` metadata, gates the learned
  effect-size branch by support excess and effect-signal magnitude, and adds a
  direct in-domain extra-inflation penalty while keeping legacy version 7
  metadata loadable. Focused conditional-calibration and LUMI workflow tests
  pass. A production-shape local sanity run at `40` sites and `75` species is
  recorded in `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- The version 8 candidate is not ready for five-seed LUMI submission. Overall
  in-domain SBC passes locally, but stratified diagnostics still fail the local
  precondition: rare-prevalence rank mean remains high, high-design-information
  coverage is only `0.9108`, and effect-size OOD coverage remains below the
  previous ungated effect-aware version 7 local result. Next substep: add
  explicit stratified in-domain gates to the OOD objective, especially
  design-information groups, before rerunning local sanity and submitting LUMI.
- Implemented the stratified in-domain OOD gate follow-up. The learned OOD
  objective now gates in-domain behavior over prevalence strata,
  design-information tertiles, and coefficient identity, with per-stratum
  coverage penalties and a worst-stratum gate term. Focused tests pass, and the
  follow-up local sanity is recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- The stratified-gate v8 candidate still is not ready for LUMI. It reduced
  in-domain inflation from `1.1730` to `1.1371` and nudged high-design coverage
  from `0.9108` to `0.9138`, but rare-prevalence rank mean remains high and
  effect-size OOD coverage dropped from `0.8410` to `0.8326`. Next substep:
  split the scalar learned inflation into stratum-conditioned or constrained
  branches so high-design-information in-domain coefficients can be capped
  without suppressing true effect-size OOD inflation.
- Implemented the constrained v8 branch. The gated curve now carries a
  backward-compatible ninth parameter, `effect_high_design_suppression`, and
  the objective adds a high-design/support-close cap on in-domain extra
  inflation. Focused conditional-calibration and LUMI workflow tests pass, and
  the production-shape local sanity result is appended to
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- The constrained v8 branch still is not ready for LUMI. It learned nonzero
  high-design suppression (`0.3327`), but local high-design coverage remained
  `0.9138`, intermediate-design coverage remained `0.9200`, rare-prevalence
  rank mean error remained `0.0432`, and effect-size OOD coverage remained
  `0.8326`. Next substep: replace the single globally learned curve plus
  constraints with a genuinely stratum-conditioned v8 objective, such as
  prevalence/design/coefficient-specific gate intercepts or caps, then rerun
  the same local sanity gate before any five-seed LUMI comparison.
- Implemented the stratum-conditioned v8 objective. Version 8 metadata now
  serializes prevalence, design-information, and coefficient gate offsets in
  addition to the high-design suppression term, and the OOD objective includes
  per-stratum in-domain extra-inflation caps. Focused tests pass, and the
  production-shape local sanity result is appended to
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- The stratum-conditioned v8 branch still is not ready for LUMI. The learned
  offsets were active but small: prevalence offsets `[0.0002, 0.0185, 0.0135]`,
  design offsets `[0.0271, 0.0146, -0.0048]`, and coefficient offsets
  `[0.0170, 0.0093, 0.0083]`. Local high-design coverage remained `0.9138`,
  intermediate-design coverage remained `0.9200`, rare-prevalence rank mean
  error remained `0.0430`, and effect-size OOD coverage fell to `0.8308`. Next
  substep: address the residual in-domain failures outside the OOD inflation
  gate, either with rare-prevalence signed-bias correction before scale
  calibration or stratum-specific base scale/normalization terms.
- Implemented and locally tested both residual in-domain follow-ups. The
  prevalence-by-coefficient signed-bias correction is serialized and applicable
  but automatic fitting is disabled because local transfer failed: rare
  rank-mean error worsened from `0.0430` to `0.0666`, and high-design coverage
  fell to `0.9079`. The stratum-specific base scale offset candidate is active
  in the v8 path and learned prevalence, design-information, and coefficient
  log-scale offsets outside the OOD gate. It improved high-design coverage to
  `0.9229` and lowered in-domain inflation to `1.1166`, but intermediate-design
  coverage fell to `0.9133`, rare-prevalence rank mean error remained `0.0412`,
  and effect-size OOD coverage fell to `0.8285`. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- The residual in-domain follow-up is not ready for LUMI. Next substep: target
  rare-prevalence rank mean with a transfer-robust objective, such as a
  rank-mean-aware neural posterior mean penalty during training or a
  calibration-time monotone rank-centering transform validated on held-out SBC,
  rather than additional scale-only or OOD-inflation terms.
- Implemented the calibration-time monotone rank-centering candidate with an
  internal held-out calibration split. The selector chose shrinkage `0.75`, but
  local SBC transfer failed: rare-prevalence rank mean error worsened from
  `0.0412` under the base-strata candidate to `0.0554`, and
  intermediate-design coverage remained below gate at `0.9113`. Automatic
  rank-centering fitting is disabled; metadata/application machinery remains
  available with zero default offsets. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- The rank-centering follow-up is not ready for LUMI. Next substep: move the
  rare-prevalence rank objective into neural training itself by adding a
  rank-mean-aware posterior-mean penalty evaluated on held-out simulation
  batches, then rerun the same local gate before any LUMI comparison.
- Implemented the opt-in training-time rare-prevalence rank-mean penalty. The
  public training API and benchmark runner now accept a holdout rank penalty
  weight and holdout fraction, and benchmark records include the rank-penalty
  history. Two local sanity runs were completed with weights `0.05` and `0.02`.
  The better run reduced rare-prevalence rank mean error from `0.0412` to
  `0.0316`, but intermediate-design coverage remained low at `0.9146` and
  effect-size OOD coverage remained weak at `0.8285`. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- The training-rank-penalty follow-up is not ready for LUMI. Next substep:
  tune or redesign the training penalty locally, likely with a
  prevalence-weighted rank objective plus a design-coverage guard, or a
  two-stage schedule that activates the penalty only after posterior scale has
  stabilized.
- Implemented and tuned the redesigned training-rank penalty locally. The
  public training API and benchmark runner now expose delayed activation via
  `rank_mean_penalty_start_fraction` and an optional design-information
  coverage guard via `rank_mean_penalty_design_guard_weight` and
  `rank_mean_penalty_design_guard_floor`. Five local probit sanity variants
  were run. The best rare-rank variant used a small design guard from epoch 0
  and reduced rare-prevalence rank mean error to `0.0302`, but it still missed
  the `0.025` rank gate and left intermediate-design coverage at `0.9163`.
  The longer two-stage variant improved intermediate-design coverage to
  `0.9217`, but rare-prevalence rank error regressed to `0.0428` and
  high-design coverage fell to `0.9175`. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- The redesigned training-rank-penalty candidates are not ready for LUMI. Next
  substep: stop tuning scale-only or coverage-guard terms and redesign the
  posterior-mean part of training, likely with a signed rare-prevalence
  posterior-mean correction or auxiliary loss that targets rank direction
  directly while constraining intermediate/high design-information strata.
- Implemented the signed posterior-mean training objective. The public
  training API and benchmark runner now expose
  `rank_mean_penalty_signed_mean_weight`,
  `rank_mean_penalty_design_mean_guard_weight`, and
  `rank_mean_penalty_design_mean_guard_tolerance`. The objective uses held-out
  rare-prevalence rank imbalance to drive a signed normalized mean-bias penalty
  and constrains medium/high expected design-information strata. Focused tests
  pass, and six local probit sanity variants were run.
- The signed posterior-mean candidates are not ready for LUMI. The weak
  pseudo-shift form was nearly neutral. The stronger signed-bias form improved
  some coverage metrics, but rare-prevalence rank mean transferred in the wrong
  direction on independent SBC data: the previous best rare rank mean was
  `0.4698`, while signed-bias weights `0.1`, `0.25`, and `0.5` yielded
  `0.4587`, `0.4572`, and `0.4554`. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: address holdout-transfer failure in the training rank
  objective. Use cross-fit or multi-holdout rank training so signed rare-rank
  corrections are applied only when the direction is stable across rotating
  simulation folds and medium/high design-information gates remain satisfied.
- Implemented crossfit/multi-holdout rank training. The public training API
  and benchmark runner now expose `rank_mean_penalty_holdout_folds` and
  `rank_mean_penalty_crossfit_min_agreement`. Multi-fold runs evaluate the
  base rank/design penalty on each holdout fold, enable signed posterior-mean
  correction only when rare-prevalence rank directions are stable across
  folds, and suppress signed correction when medium/high design-information
  rank gates fail. Focused tests pass.
- The crossfit rank-training candidates are not ready for LUMI. Large-holdout
  variants degraded base training and high-design coverage. Smaller-holdout
  variants were less damaging, but the best rare-rank error was `0.0311`,
  still worse than the previous `0.0302` local candidate and above the `0.025`
  rank gate; intermediate design coverage remained below the local guard
  target. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: move away from signed posterior-mean correction for rare
  species. Use larger, explicitly balanced rare-prevalence simulation batches
  or a separate rare-species calibration head trained across many simulation
  batches, instead of noisy per-run holdout rank signs.
- Implemented a guarded rare-balanced calibration head. The simulator now
  supports an `intercept_mean` override for rare-prevalence calibration
  simulations, and the benchmark runner exposes `rare_calibration_datasets`
  and `rare_calibration_intercept_mean`. Conditional calibration can fit a
  rare-only prevalence-by-coefficient residual mean head from the extra
  calibration pool and serializes rare-head diagnostics in
  `mean_bias_correction`.
- The rare-balanced head is not ready for LUMI. Local runs with intercept means
  `-1.0`, `-1.75`, and `-2.5` produced large rare pools, but the validation
  gate selected zero shrinkage in every case. Combined with the previous best
  training-rank penalty, the head also selected zero shrinkage and left the
  benchmark unchanged. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: instrument the rare-head fitting path before changing the
  objective again. Record candidate rare-head offsets, validation-gate metrics,
  rare-pool prevalence summaries, and rare-pool residual/rank summaries in
  benchmark metadata, then run a small local diagnostic sweep to identify why
  rare-balanced residuals disagree with independent SBC rare-rank behavior.
- Implemented rare-head fitting instrumentation. The calibration metadata now
  records candidate and selected rare-head offsets, shrinkage-grid validation
  scores, rare-pool prevalence summaries, per-coefficient rare-pool
  residual/rank summaries, and validation group sizes under
  `mean_bias_correction.rare_balanced_diagnostics`.
- The diagnostic local run explains why the current rare head stays off. The
  rare-balanced pool produced a strong negative intercept candidate offset
  (`-0.2235`) and rare-pool intercept rank mean `0.2423`, but applying any
  shrinkage worsened ordinary validation rare-rank error monotonically from
  `0.0369` at zero shrinkage to `0.0893` at full shrinkage. Results are
  recorded in `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: change the rare-calibration simulation design rather than
  relaxing the validation gate. Build a stratified rare-calibration pool that
  matches the SBC rare-failure context: balanced by rare prevalence,
  coefficient identity, and design-information tertile, with intercept-shift,
  low-detection, and small-sample rare regimes recorded separately in
  diagnostics.
- Implemented the stratified rare-calibration pool. The simulator now supports
  detection thinning and effective sample-fraction masking in addition to
  intercept shifts. The benchmark runner exposes rare-calibration regimes
  `intercept_shift`, `low_detection`, and `small_sample`, and the rare-head
  fitter balances candidate offsets by regime, design-information tertile, and
  coefficient identity before validating shrinkage.
- The stratified rare pool is not ready for LUMI. It balanced rare observations
  exactly by design tertile and coefficient, and the ordinary validation gate
  selected full shrinkage with validation rare-rank error improving from
  `0.0369` to `0.0277`. Independent SBC rare-rank error nevertheless worsened
  from `0.0412` to `0.0496`, so the ordinary calibration validation batch is
  insufficient for rare-head acceptance. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: add an independent rare-head validation gate. Before applying
  nonzero rare-head offsets, evaluate candidates on a separate SBC-style
  in-domain validation pool with rare/prevalence/design stratification, and
  require non-degradation of independent rare-rank error and intermediate/high
  design coverage.
- Implemented the independent rare-head validation gate. The benchmark runner
  now exposes `rare_validation_datasets`, generated independently from
  `rare_calibration_datasets`, and conditional calibration records independent
  rare-head gate metrics under
  `mean_bias_correction.rare_balanced_diagnostics.validation.independent`.
  Failed independent validation resets selected rare-head shrinkage and applied
  offsets to zero while preserving rejected candidate offsets in diagnostics.
- The independent rare-head gate is not ready for LUMI. In the local sanity run
  the ordinary validation pool again selected the stratified rare candidate
  offsets and improved ordinary rare-rank error from `0.0369` to `0.0277`.
  Independent validation improved rare-rank error from `0.0163` to `0.0107`,
  but absolute independent coverage remained too low: overall `0.7900`,
  intermediate-design `0.7988`, and high-design `0.7000`. The gate therefore
  reset rare-head shrinkage to `0.0` and preserved the previous independent
  SBC result. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: address rare-validation regime coverage before revisiting
  nonzero rare-head mean offsets. The likely direction is a rare-regime-aware
  scale or normalization correction evaluated on the independent rare-validation
  pool, with the independent rare-head mean-offset gate kept strict.
- Implemented a rare-validation scale correction. When independent
  rare-validation batches are supplied, conditional calibration can fit a
  design-stratum log-scale multiplier from that independent pool. The scale
  correction is separate from the rare-head mean offsets; the independent
  rare-head mean-offset gate remains strict and still resets failed nonzero
  rare-head offsets to zero. Metadata is stored under `rare_validation_scale`.
- The rare-validation scale local sanity run fixed the immediate independent
  rare-validation coverage failure: overall coverage improved from `0.6857` to
  `0.9001`, intermediate-design coverage from `0.5240` to `0.9001`, and
  high-design coverage from `0.4750` to `0.9000`. The selected design
  multipliers were `1.187`, `2.643`, and `3.320`. Local SBC also improved
  rare-prevalence rank error from `0.0412` to `0.0345` and effect-size OOD
  coverage from `0.8285` to `0.9119`.
- The rare-validation scale candidate is not ready for LUMI. It is too
  conservative in-domain: overall coverage rose to `0.9921`, high-design
  coverage to `1.0000`, and rank variance remains compressed. Combined-shift
  OOD coverage improved but still failed at `0.8767`. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: constrain the rare-validation scale correction. Target the
  failed low-detection/small-sample regimes and high-risk design contexts more
  selectively, add an in-domain overcoverage/rank-variance guard, and retest
  locally before any five-seed LUMI comparison.
- Implemented a constrained rare-validation scale correction. The design-stratum
  scale now uses a support-excess activation derived from the in-domain
  calibration support, and candidate shrinkage is rejected if it over-inflates
  the original in-domain calibration pool or compresses in-domain rank variance.
- The constrained scale local sanity run selected zero shrinkage. The support
  gate activated on only `3.3%` of in-domain coefficients and `13.5%` of
  rare-validation coefficients. It prevented the always-on overcoverage failure,
  but validation coverage remained too low before the in-domain guard would
  fail: at full shrinkage validation high-design coverage was only `0.7298`,
  while in-domain high-design coverage already reached `1.0000` and rank
  variance compressed to `0.0449`. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: replace the support-excess-only activation with a more
  discriminative low-detection/small-sample regime proxy. Candidate directions
  are a learned undercoverage classifier from rare-validation features, a
  prevalence-by-effective-sample-size scale gate, or a two-part correction that
  separates rare-regime OOD coverage from in-domain design strata.
- Implemented an observable rare-regime proxy for rare-validation scale
  activation. The proxy combines support excess, rare/intermediate prevalence
  by design-information stratum, and low community occupancy estimated from the
  observed response matrix. It does not use hidden rare-validation regime
  labels at application time.
- The community-occupancy proxy local sanity run selected zero shrinkage. Full
  shrinkage cleared the independent rare-validation floors, with overall
  coverage `0.9001`, intermediate-design coverage `0.9001`, and high-design
  coverage `0.9000`. The in-domain guard correctly rejected it because
  in-domain high-design coverage rose to `1.0000` and overall coverage to
  `0.9894`. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: change the correction shape rather than only the activation.
  A single positive scale multiplier by design stratum remains too blunt. Use
  regime-proxy-conditioned slope caps or a two-part correction that separates
  high-design in-domain coefficients from low-community rare-validation
  coefficients, while keeping the independent rare-validation and in-domain
  guards.
- Implemented a thresholded low-community scale shape. Support-excess and
  low-community activations are now zero at their in-domain thresholds and grow
  only outside those thresholds. Low-community stress can activate the
  design-stratum scale directly, avoiding the previous damping of common
  high-design coefficients in low-detection/small-sample batches.
- The thresholded low-community scale local sanity run selected full shrinkage.
  It passed the independent rare-validation gate: validation overall coverage
  `0.9001`, intermediate-design coverage `0.9001`, and high-design coverage
  `0.9000`. It also passed the in-domain guard: overall coverage changed only
  from `0.9500` to `0.9508`, high-design coverage from `0.9371` to `0.9392`,
  and rank variance from `0.0815` to `0.0811`.
- The candidate is still not ready for LUMI because OOD coverage remains below
  target. Local calibrated effect-size OOD coverage was `0.8324`, and
  combined-shift OOD coverage was `0.8015`. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: evaluate whether the thresholded low-community scale should be
  combined with the learned OOD objective or whether OOD inflation must be
  refit after this scale correction. Run a local OOD-focused sanity check first,
  then decide whether a five-seed LUMI comparison is justified.
- Ran the OOD-focused local sanity check. Increasing the learned OOD inflation
  cap from `8` to `16`, doubling OOD calibration batches, and increasing OOD
  objective epochs did not improve OOD coverage. Covariate-shift coverage
  changed from `0.9146` to `0.9128`, effect-size coverage from `0.8324` to
  `0.8253`, and combined-shift coverage from `0.8015` to `0.7953`. Mean
  inflation increased for covariate and combined shifts, but effect-size mean
  inflation stayed near `1.5`.
- Reconstructed OOD diagnostics showed that the thresholded low-community scale
  is not the pure effect-size fix. Rare-scale activation was `50.3%` under
  covariate shift and `46.7%` under combined shift, but only `3.8%` under pure
  effect-size shift. The remaining OOD blocker is the learned OOD effect-size
  objective, not the rare-validation scale shape or multiplier cap alone.
  Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: implement a post-scale or final-multiplier-aware OOD objective
  focused on effect-size coverage. The objective should evaluate the final
  calibrated multiplier after rare-validation scale is applied, include
  effect-size and combined-shift coverage floors as acceptance gates, and keep
  the existing in-domain and rare-validation gates.
- Implemented the final-multiplier-aware OOD refinement as a second OOD fitting
  pass after rare-validation scale selection. The pass initializes from the
  first OOD fit, evaluates the post-scale final multiplier, and adds targeted
  coverage-floor penalties for effect-size and combined-shift OOD domains.
  Focused tests pass, including conditional calibration, public API, and LUMI
  workflow coverage.
- The final-aware local sanity run is not ready for LUMI. It preserved the
  in-domain acceptance gate with coverage `0.9301` and rank variance `0.0838`,
  but effect-size OOD coverage remained `0.8256` and combined-shift OOD
  coverage remained `0.8036`. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: add OOD final-multiplier diagnostics before another objective
  change. Benchmark metadata should report learned effect-gate activation,
  learned OOD inflation, rare-validation post-scale multiplier, final
  multiplier quantiles, coverage by effect-size quantile, and in-domain gate
  penalty components for each OOD regime. Use those diagnostics to decide
  whether the next objective should be an effect-shift-specific scale head or a
  domain-classifier-gated multiplier.
- Implemented OOD final-multiplier diagnostics in conditional calibration
  metadata. Each OOD calibration regime now records learned effect-gate
  activation, learned OOD inflation, rare-validation post-scale multiplier,
  final multiplier quantiles, coverage by effect-size quantile, and in-domain
  gate penalty components. A tiny benchmark smoke run confirmed the fields are
  present in `benchmark_manifest.json`; focused conditional calibration, public
  API, and LUMI workflow tests pass.
- Next substep: run the production-like local sanity workflow again and inspect
  the new diagnostics from the failed effect-size and combined-shift regimes.
  Use the observed effect-gate activation, final multiplier quantiles, and
  effect-quantile coverage to choose the next objective shape, likely an
  effect-shift-specific scale head or domain-classifier-gated multiplier.
- Ran the production-like diagnostics-enabled local sanity workflow. Held-out
  SBC coverage remained failed for effect-size shift (`0.8256`) and combined
  shift (`0.8036`) while the in-domain gate still passed. The OOD calibration
  diagnostics showed that effect-size shift has median final multiplier only
  `0.8035` and low middle-effect coverage (`0.7844` in the q3 effect bin),
  whereas combined shift has much larger inflation but still fails high-effect
  coverage (`0.7711` in the q4 effect bin). In-domain group losses remain tiny,
  but extra-inflation penalties are already active, so broad in-domain inflation
  is constrained. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: implement an experimental effect-shift-specific scale head or
  domain-classifier-gated multiplier. The objective should target
  effect-quantile coverage directly, with separate pure effect-size and combined
  shift constraints, while preserving the in-domain and rare-validation gates.
- Implemented an experimental context-gated effect-shift scale head in the
  version 8 OOD objective. The head adds separate pure-effect and combined-shift
  positive log-scale components and trains them with differentiable
  effect-quantile coverage losses. Existing in-domain and rare-validation gates
  remain active. Focused conditional calibration, public API, and LUMI workflow
  tests pass; a tiny benchmark smoke run confirmed the head serializes in
  `benchmark_manifest.json`.
- Next substep: run the production-like local sanity workflow with the
  experimental effect-shift head enabled. Compare held-out OOD coverage,
  effect-quantile coverage, final-multiplier diagnostics, and in-domain/rare
  validation gates against the previous final-aware run before considering any
  LUMI submission.
- Ran the production-like local sanity workflow for the experimental
  effect-shift head. The head improved held-out OOD coverage but did not qualify:
  effect-size shift improved from `0.8256` to `0.8656`, combined shift improved
  from `0.8036` to `0.8333`, and covariate shift improved from `0.9143` to
  `0.9199`. In-domain overall coverage still passed at `0.9311`, and the
  independent rare-validation scale gate still passed. Diagnostics showed the
  intended effect-bin improvements, but in-domain extra-inflation penalties rose
  substantially (`extra_inflation_over_1_05_loss` from `0.8885` to `2.9035`).
  Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: constrain the effect-shift head rather than increasing its
  strength. Candidate fixes are a scheduled/two-stage head fit, stronger
  in-domain extra-inflation normalization, or effect-bin-specific amplitude caps
  that improve OOD middle/high-effect coverage without increasing in-domain
  extra-inflation penalties.
- Implemented a constrained effect-shift head. The pure-effect branch now has a
  fixed log-amplitude cap plus a high-effect taper, and the combined branch now
  has its own fixed log-amplitude cap plus a support-excess activation gate.
  The constrained head metadata is serialized as
  `constrained_context_gated_effect_quantile_scale`. Focused conditional
  calibration, public API, and LUMI workflow tests pass; a tiny benchmark smoke
  run confirmed the constrained head fields in `benchmark_manifest.json`.
- Next substep: run the production-like local sanity workflow for the
  constrained effect-shift head. Compare held-out OOD coverage,
  effect-quantile coverage, final-multiplier diagnostics, and in-domain/rare
  validation gates against both the unconstrained effect-head run and the
  previous final-aware run.
- Ran the production-like local sanity workflow for the constrained effect-shift
  head. The constraint kept most of the OOD gain but did not qualify:
  effect-size shift coverage was `0.8621` and combined-shift coverage was
  `0.8293`, versus `0.8656` and `0.8333` for the unconstrained head. In-domain
  overall coverage still passed at `0.9306`, and rare-validation still passed,
  but extra-inflation penalties remained high (`extra_inflation_over_1_05_loss`
  `2.7131`, max group extra-cap loss `0.6623`). Intermediate/high
  design-stratum held-out coverage stayed low at `0.9133` and `0.9138`.
  Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: move away from a globally applied learned effect-shift head.
  Implement a two-stage post-fit selection or shrinkage step that accepts head
  offsets only when OOD effect-quantile coverage improves enough and in-domain
  extra-inflation/gate penalties remain below explicit thresholds.
- Implemented post-fit effect-head selection. The OOD objective still fits the
  effect-shift head, but the calibrator now evaluates a shrinkage grid for head
  amplitudes and accepts nonzero head offsets only when mean/worst-domain OOD
  coverage gains pass thresholds and in-domain extra-inflation/gate penalties
  remain below explicit limits. The selection decision and candidate grid are
  stored under
  `ood_objective.final_multiplier_diagnostics.effect_shift_head_selection`.
  Focused conditional calibration, public API, and LUMI workflow tests pass; a
  tiny benchmark smoke run confirmed the selection metadata in
  `benchmark_manifest.json`.
- Next substep: run the production-like local sanity workflow with post-fit
  head selection enabled. Compare held-out OOD coverage, effect-quantile
  coverage, selected shrinkage, and in-domain/rare-validation gates against the
  final-aware, unconstrained-head, and constrained-head runs.
- Ran the production-like local sanity workflow with post-fit head selection
  enabled. The selector rejected the head and selected shrinkage `0.0`.
  Held-out coverage was `0.8481` for effect-size shift and `0.8168` for
  combined shift, better than the original final-aware run but worse than the
  unconstrained and constrained heads. In-domain overall SBC still passed at
  `0.9294`, and rare-validation still passed, but selector diagnostics still
  showed gate pressure: extra-inflation-over-1.05 loss `1.8772`, max group
  extra-cap loss `0.5724`, and max group loss `0.1267`. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Implemented independent pure-effect and combined-shift post-fit head
  selection. The OOD objective still fits separate pure-effect and
  combined-shift effect-head amplitudes, but post-fit selection now evaluates
  pure-effect shrinkage against the `effect_size_shift` validation domain and
  combined-shift shrinkage against the `combined_shift` validation domain. Each
  branch has its own coverage-gain gate and shared in-domain extra-inflation
  gate. The selected pure/combined shrinkage values, branch acceptance flags,
  domain coverage gains, in-domain gate deltas, and candidate records are
  stored under
  `ood_objective.final_multiplier_diagnostics.effect_shift_head_selection`
  with kind `post_fit_independent_effect_shift_head_selection`. Focused
  conditional calibration, public API, and LUMI workflow tests pass; a tiny
  benchmark smoke run confirmed the independent selector metadata in
  `/private/tmp/neural_hmsc_independent_heads_smoke`.
- Ran the production-like local sanity workflow with independent
  pure-effect/combined-shift head selection enabled. The selector accepted the
  pure-effect branch with shrinkage `0.5` and rejected the combined-shift branch
  with shrinkage `0.0`. Held-out coverage was `0.8507` for effect-size shift
  and `0.8175` for combined shift, slightly above shared post-fit selection
  (`0.8481` and `0.8168`) but still below the `0.90` OOD floor. In-domain
  overall SBC still passed at `0.9294`, and rare-validation still passed with
  selected shrinkage `1.0`, validation coverage `0.9001`, and rare-scale
  in-domain guard coverage `0.9506`. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Implemented a domain-specific combined-shift scale head. The new serialized
  `ood_objective.combined_shift_scale` block applies a bounded
  support-excess-by-effect-signal log multiplier after the learned OOD curve
  and rare-validation scale. It is selected after independent effect-head
  selection and requires both a held-out `combined_shift` coverage floor of
  `0.90` and a minimum coverage gain of `0.005`, while preserving the existing
  in-domain gate deltas. The selected decision is recorded under
  `ood_objective.final_multiplier_diagnostics.combined_shift_scale_selection`,
  and final diagnostics now report `combined_shift_scale_multiplier` by OOD
  domain. Focused conditional-calibration, public API, and LUMI workflow tests
  pass; a tiny metadata smoke run confirmed the block in
  `/private/tmp/neural_hmsc_combined_shift_scale_smoke`.
- Ran the production-like local sanity workflow with the combined-shift scale
  head enabled. The selector rejected every nonzero candidate and selected
  `log_amplitude = 0.0`, leaving held-out SBC identical to the independent
  selector run: effect-size shift coverage `0.8507` and combined-shift coverage
  `0.8175`. Candidate diagnostics showed that large amplitudes can raise
  diagnostic combined-shift coverage above `0.90` (`0.9117` at log amplitude
  `1.4931` and `0.9222` at log amplitude `1.7918`), but only with severe
  in-domain gate violations. Even the smallest nonzero candidate improved
  combined coverage by `0.0089` while exceeding the max group-loss delta gate.
  Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Implemented a more selective combined-shift correction shape. The
  `combined_shift_scale` multiplier now requires joint support-excess,
  effect-size, low-design-information, and low-community-occupancy activation
  instead of applying across the combined regime globally. The serialized
  activation metadata records `low_design_center`, `low_design_width`,
  `low_community_center`, and `low_community_width`, and the selection
  diagnostics use the same activation as application. Focused
  conditional-calibration, public API, and LUMI workflow tests pass; a tiny
  metadata smoke run confirmed the selective activation in
  `/private/tmp/neural_hmsc_selective_combined_shift_scale_smoke`.
- Ran the production-like local sanity workflow with the selective
  combined-shift scale enabled. The selector again selected `log_amplitude =
  0.0`, leaving held-out SBC unchanged from the globally shaped combined-shift
  scale run: effect-size shift coverage `0.8507` and combined-shift coverage
  `0.8175`. The selective activation reduced in-domain gate pressure for
  nonzero candidates, but it also reduced OOD gain: the strongest tested
  selective candidate reached only `0.8561` diagnostic combined-shift coverage,
  versus `0.9222` for the global shape. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Implemented an effect-bin-specific combined-shift scale selector. The
  `combined_shift_scale` block now serializes effect-bin edges, per-bin log
  amplitudes, and per-bin multipliers, while retaining the selective
  support/effect/low-design/low-community activation. Selection evaluates
  scalar, high-effect, mid/high-effect, ranked-effect, and all-bin candidate
  shapes against the same combined-shift coverage floor and in-domain gate
  deltas. Focused conditional-calibration, public API, and LUMI workflow tests
  pass; a tiny metadata smoke run confirmed 60 effect-bin candidates in
  `/private/tmp/neural_hmsc_effect_bin_combined_shift_scale_smoke`. The smoke
  selected zero amplitude because no candidate satisfied the strict held-out
  gate in that tiny setting.
- Next substep: run the production-like local sanity workflow with the
  effect-bin-specific combined-shift scale enabled. Compare selected candidate
  pattern, per-bin amplitudes, held-out combined-shift/effect-size coverage,
  effect-quantile diagnostics, final multiplier quantiles, in-domain gate
  deltas, and rare-validation gates against the selective scalar
  combined-shift scale run before considering LUMI.
- Ran the production-like local sanity workflow with the effect-bin-specific
  combined-shift selector enabled and corrected the run to use the same rare
  calibration and validation settings as the selective scalar baseline
  (`rare_calibration_datasets = 32`, `rare_validation_datasets = 32`). The
  selector evaluated 60 candidates but again selected zero amplitude, so
  held-out 95% coverage was unchanged from the selective scalar run:
  in-distribution `0.9294`, covariate shift `0.9161`, effect-size shift
  `0.8507`, and combined shift `0.8175`. The best scalar/all-bin candidate
  reached diagnostic combined-shift coverage `0.8561`, but violated the
  in-domain gate with max group-loss delta `0.6311` and extra-inflation delta
  `1.2254`. More selective effect-bin patterns reduced the gate deltas but
  dropped combined-shift coverage to `0.8389`-`0.8450`, still below the `0.90`
  floor. Rare-validation gates matched the selective scalar run, with selected
  shrinkage `1.0`, overall rare-validation coverage `0.9001`, and rare-scale
  in-domain guard coverage `0.9506`. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: stop tuning effect-bin amplitudes and implement a more
  domain-discriminative combined-shift activation. The likely implementation is
  a combined-shift domain/context gate, or a low-community-by-support-excess
  classifier-style gate, trained on combined-shift validation batches with
  explicit in-domain overlap penalties and the existing rare-validation guard.
- Implemented a context-gated combined-shift activation. The
  `combined_shift_scale` metadata now serializes a
  support/effect/low-design/low-community classifier-style `context_gate`, with
  strength/intercept selected by held-out combined-shift validation. The
  selector evaluates legacy-product, moderate-context, and strict-context gate
  variants across the existing scalar/effect-bin correction shapes, records
  `in_domain_context_gate` overlap diagnostics for every candidate, and rejects
  nonzero candidates whose in-domain context-gate mean or active fraction
  exceeds explicit thresholds. Legacy metadata remains behaviorally compatible
  because missing context-gate fields default to strength `0.0`. Focused
  conditional-calibration, public API, and LUMI workflow tests pass; a tiny
  metadata smoke run confirmed 180 context-gated candidates in
  `/private/tmp/neural_hmsc_context_gated_combined_shift_smoke`.
- Next substep: run the production-like local sanity workflow with the
  context-gated combined-shift selector enabled, using the same rare
  calibration/validation settings as the previous selective scalar and
  effect-bin runs. Compare selected context pattern, context overlap metrics,
  held-out combined-shift/effect-size coverage, effect-quantile diagnostics,
  final multiplier quantiles, in-domain gate deltas, and rare-validation gates
  before considering any LUMI comparison.
- Ran the production-like local sanity workflow with the context-gated
  combined-shift selector enabled, using the same rare calibration/validation
  settings as the previous selective scalar and effect-bin runs. The selector
  evaluated 180 candidates and selected zero amplitude, leaving held-out 95%
  coverage unchanged: in-distribution `0.9294`, covariate shift `0.9161`,
  effect-size shift `0.8507`, and combined shift `0.8175`. The best legacy
  product candidate reached diagnostic combined-shift coverage `0.8561`, but
  had full in-domain context overlap and violated the in-domain gate. The
  moderate context gate reduced in-domain context mean to `0.1896` and active
  fraction over `0.8` to `0.0061`, but combined-shift coverage fell to
  `0.8361` and gate deltas still failed. The strict context gate reduced
  context mean to `0.0900` and active fraction over `0.8` to `0.0033`, but
  combined-shift coverage fell to `0.8328` and the gain was below the minimum
  threshold. Rare-validation gates matched prior runs, with selected shrinkage
  `1.0`, overall rare-validation coverage `0.9001`, and rare-scale in-domain
  guard coverage `0.9506`. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: stop adding post-hoc combined-shift scale gates. Move the
  combined-shift signal earlier into the learned OOD objective, for example by
  adding a direct combined-shift coverage term with a learned context classifier
  and explicit in-domain overlap regularization during OOD-objective fitting,
  instead of selecting another post-scale multiplier.
- Implemented the combined-shift objective inside the final-multiplier-aware
  OOD fitting path. The learned OOD objective now adds direct combined-shift
  coverage pressure, effect-quantile coverage pressure, and a
  context-weighted combined-shift coverage term using the existing learned
  combined branch as a support/effect/low-design/low-community context
  classifier. The in-domain gate now also penalizes learned combined-context
  overlap and context-weighted extra inflation, so the combined branch is
  trained under the same in-domain protection rather than selected afterward as
  a post-scale multiplier. The serialized `ood_objective` metadata records the
  `combined_shift_training_objective` block, and final multiplier diagnostics
  now report `learned_combined_shift_context` by OOD domain. Focused
  conditional-calibration, public API, and LUMI workflow tests pass; a tiny
  metadata smoke run confirmed the objective block and diagnostics in
  `/private/tmp/neural_hmsc_combined_objective_smoke`.
- Next substep: run the production-like local sanity workflow with the
  combined-shift-aware OOD objective enabled, using the same rare
  calibration/validation settings as the prior local sanity runs. Compare
  held-out combined-shift/effect-size coverage, effect-quantile diagnostics,
  learned combined-context activation, in-domain gate loss/deltas, and
  rare-validation gates against the context-gated post-scale run.
- Ran the production-like local sanity workflow with the combined-shift-aware
  OOD objective enabled, using the same rare calibration/validation settings as
  the prior local sanity runs. The run reduced diagnostic in-domain gate
  penalties but worsened held-out OOD coverage. Compared with the context-gated
  post-scale run, 95% coverage changed from `0.8507` to `0.8303` for
  effect-size shift and from `0.8175` to `0.8060` for combined shift.
  Diagnostic selector coverage also fell from `0.8672` to `0.8511` for
  effect-size shift and from `0.8289` to `0.8178` for combined shift. The
  learned combined context activated more on combined shift than pure
  effect-size shift (`0.1019` versus `0.0751` mean activation), but not enough
  to recover coverage. In-domain and rare-validation gates remained acceptable,
  with rare-validation selected shrinkage `1.0`, rare-validation coverage
  `0.9001`, and rare-scale in-domain guard coverage `0.9508`. Results are
  recorded in `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: rebalance or stage the combined-shift-aware OOD objective
  instead of adding another gate. The likely implementation is a two-stage or
  constrained schedule: first fit the combined branch to recover
  combined-shift/effect-quantile coverage, then apply an in-domain overlap
  constraint or shrinkage step that preserves the OOD coverage gain.
- Implemented a staged combined-shift-aware OOD objective. The final-aware OOD
  refit now uses a `coverage_warmup_then_overlap_ramp` schedule: early epochs
  boost direct combined-shift, effect-quantile, and context-weighted coverage
  losses while down-weighting the in-domain gate and disabling the explicit
  combined-context overlap penalty; later epochs ramp the in-domain gate and
  overlap penalty back to full strength. The existing post-fit
  pure-effect/combined-shift shrinkage selector still runs afterward, so any
  recovered OOD gain must survive the same in-domain gate checks before it is
  applied. The serialized `combined_shift_training_objective.schedule` block
  records the warmup fraction, coverage boost, and warmup gate fraction.
  Focused conditional-calibration, public API, and LUMI workflow tests pass; a
  tiny metadata smoke run confirmed the staged schedule in
  `/private/tmp/neural_hmsc_staged_combined_objective_smoke`.
- Next substep: run the production-like local sanity workflow with the staged
  combined-shift-aware OOD objective enabled, using the same rare
  calibration/validation settings as the prior local sanity runs. Compare
  held-out combined-shift/effect-size coverage, effect-quantile diagnostics,
  learned combined-context activation, post-fit shrinkage selection, in-domain
  gate deltas, and rare-validation gates against the unstaged combined-aware
  objective.
- Ran the production-like local sanity workflow with the staged
  combined-shift-aware OOD objective enabled. The staged schedule did not
  recover OOD coverage. Compared with the context-gated post-scale baseline,
  95% coverage remained worse for effect-size shift (`0.8290` versus `0.8507`)
  and combined shift (`0.8056` versus `0.8175`). Compared with the unstaged
  combined objective, the staged schedule slightly reduced diagnostic
  in-domain inflation penalties but did not improve selector coverage:
  effect-size selector coverage stayed `0.8511` and combined-shift selector
  coverage stayed `0.8178`. Learned combined-context activation remained
  selective but weak, with mean activation `0.0746` for effect-size shift and
  `0.1014` for combined shift. Rare-validation gates remained acceptable, with
  selected shrinkage `1.0`, rare-validation coverage `0.9001`, and rare-scale
  in-domain guard coverage `0.9508`. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: stop this combined-shift objective family. Add a new
  representation or data split for combined shift instead, likely a
  domain-adversarial or mixture-of-experts OOD head with separate pure-effect
  and combined-shift experts plus a held-out expert-selection gate.
- Implemented the first representation/data-split replacement for the failed
  combined-shift objective family. The final-multiplier-aware OOD path now
  trains separate pure-effect and combined-shift OOD expert candidates on
  domain-specific calibration splits, evaluates them with explicit held-out
  expert-selection gates when enough simulation batches are available, and
  records a `domain_expert_selection` diagnostic block under
  `ood_objective.final_multiplier_diagnostics`. The selector keeps the
  existing serialized OOD parameter format stable: if no expert improves its
  target OOD coverage while respecting in-domain gate deltas, it falls back to
  the baseline OOD parameters. A tiny smoke run at
  `/private/tmp/neural_hmsc_domain_expert_smoke` confirmed the metadata block,
  within-batch split fallback, candidate losses, target coverage gains, and
  selected baseline fallback. Focused conditional-calibration, public API, and
  LUMI workflow tests pass.
- Next substep: run the production-like local sanity workflow with the
  held-out domain-expert OOD selector enabled, using the same rare
  calibration/validation settings as the staged combined-objective run. Compare
  selected expert, split mode, held-out target gains, effect-size and
  combined-shift coverage, effect-quantile diagnostics, final multiplier
  quantiles, in-domain gate deltas, and rare-validation gates against the
  staged combined-objective and context-gated post-scale baselines before any
  LUMI comparison.
- Ran the production-like local sanity workflow with the held-out domain-expert
  OOD selector enabled in
  `/private/tmp/neural_hmsc_v8_domain_expert_local_sanity_rare32`. The selector
  correctly rejected both candidates and preserved the baseline parameters:
  pure-effect target coverage improved by `0.0489` and combined-shift target
  coverage improved by `0.0578`, but their in-domain gate deltas were far above
  limits. Final 95% coverage therefore matched the staged combined-objective
  result and remained below OOD floor: effect-size shift `0.8290` and combined
  shift `0.8056`. Rare-validation gates remained acceptable. Results are
  recorded in `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: keep the domain-expert split but make expert acceptance
  constrained or shrinkage-aware. Evaluate a baseline-to-expert shrinkage grid,
  or add a trust-region/overlap penalty during expert fitting, so useful OOD
  directions can be partially accepted only when in-domain gate deltas stay
  within explicit limits. Rerun the same production-like local sanity workflow
  before any LUMI comparison.
- Implemented baseline-to-expert shrinkage-aware expert acceptance. Each
  domain expert now records a shrinkage grid (`0.0`, `0.125`, `0.25`, `0.5`,
  `0.75`, `1.0`) and selects only a shrinkage point that clears target OOD
  coverage gain and in-domain gate-delta thresholds. Focused
  conditional-calibration, public API, and LUMI workflow tests pass.
- Ran the production-like local sanity workflow with shrinkage-aware domain
  experts in
  `/private/tmp/neural_hmsc_v8_domain_expert_shrinkage_local_sanity_rare32`.
  No shrinkage point qualified. At shrinkage `0.125`, the pure-effect branch
  was close to the in-domain gate but target gain was only `0.0067`; by
  shrinkage `0.25`, target gain cleared the minimum but extra-inflation delta
  rose to `0.6077`, above the `0.25` limit. Final 95% coverage stayed at the
  staged combined-objective values: effect-size shift `0.8290` and combined
  shift `0.8056`. Rare-validation gates remained acceptable. Results are
  recorded in `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: keep the domain-expert data split, but move the constraint into
  expert fitting itself. Add a trust-region or overlap-regularized expert
  objective that penalizes in-domain final-multiplier drift during expert
  training, then rerun the same production-like local sanity workflow before
  any LUMI comparison.
- Implemented an in-domain log-inflation trust-region penalty inside
  domain-expert fitting. The trust-region uses the pre-expert OOD log-inflation
  as the in-domain baseline, tolerance `log(1.08)`, scale `log(1.25)`, and
  weight `3.0`; the post-fit shrinkage grid still runs afterward. Focused
  conditional-calibration, public API, and LUMI workflow tests pass.
- Ran the production-like local sanity workflow with the trust-region
  domain-expert objective in
  `/private/tmp/neural_hmsc_v8_domain_expert_trust_region_local_sanity_rare32`.
  The trust region reduced expert aggressiveness but no shrinkage point
  qualified. The full pure-effect expert's extra-inflation delta fell from
  `3.1203` to `0.6216`, but still exceeded the `0.25` limit; shrinkage `0.25`
  stayed below the extra-inflation limit but target gain was only `0.0056`.
  Final 95% coverage stayed at effect-size shift `0.8290` and combined shift
  `0.8056`. Rare-validation gates remained acceptable. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: tune or redesign the constrained expert objective locally
  before another production-like run. Start with a small trust-region sweep over
  stronger weights/tighter tolerances, or replace the global trust region with a
  domain-localized overlap penalty that penalizes in-domain overlap contexts
  while allowing more effect-specific OOD movement.
- Implemented a compact trust-region sweep inside domain-expert fitting. The
  selector now trains each domain expert with three settings:
  `moderate_w3_tol108`, `strong_w6_tol106`, and `tight_w10_tol104`, and then
  evaluates the usual post-fit shrinkage grid for each trained candidate.
  Focused conditional-calibration, public API, and LUMI workflow tests pass.
- Ran a smaller local tuning check in
  `/private/tmp/neural_hmsc_v8_domain_expert_trust_sweep_tuning`. The compact
  run accepted the combined-shift expert at shrinkage `1.0` for all three trust
  settings; the tightest setting reduced in-domain extra-inflation delta to
  `0.0931`. This is not a qualification result because the run uses smaller
  dimensions and fewer rare/OOD/SBC batches, but it verifies that the sweep can
  produce accepted candidates. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: rerun the production-like rare32 local sanity workflow with the
  trust-region sweep enabled. If no trust setting qualifies under rare32
  settings, replace the global trust region with a domain-localized overlap
  penalty.
- Ran the production-like rare32 local sanity workflow with the trust-region
  sweep enabled in
  `/private/tmp/neural_hmsc_v8_domain_expert_trust_sweep_local_sanity_rare32`.
  The selector accepted one boundary candidate: pure-effect expert with
  `tight_w10_tol104` at shrinkage `1.0`, target gain `0.0100`, and
  extra-inflation delta `0.2377`. Final 95% coverage did not improve:
  effect-size shift was `0.8282` and combined shift was `0.8056`.
  Rare-validation gates remained acceptable. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: replace the global trust region with a domain-localized overlap
  penalty and strengthen the expert-selection gate. Boundary candidates should
  require a practical held-out target gain, no OOD-domain degradation, and
  explicit control of in-domain contexts that overlap the target OOD domain
  before application. Rerun a compact tuning check before another production-like
  rare32 workflow.
- Implemented the domain-localized overlap penalty and strengthened expert
  gate. The selector now requires target gain at least `0.0200`, no non-target
  OOD coverage degradation, and localized overlap excess loss at most `0.1200`.
  Expert fitting penalizes in-domain OOD log-inflation drift only in target
  overlap contexts: high-effect/support-close coefficients for pure-effect
  experts and support-excess/low-design/low-community coefficients for
  combined-shift experts. Focused conditional-calibration, public API, and LUMI
  workflow tests pass.
- Ran a compact tuning check in
  `/private/tmp/neural_hmsc_v8_domain_expert_overlap_tuning`. No candidate
  passed the strengthened gate. The strongest combined-shift candidate had no
  non-target degradation and localized overlap excess loss `0.0238`, but target
  gain was only `0.0148`, below the practical-gain floor. Results are recorded
  in `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: improve the domain-expert objective so it can clear the
  strengthened gate before another rare32 run. Likely directions are stronger
  target-domain coverage pressure, effect-quantile-specific expert losses, or a
  target-domain curriculum, while preserving the localized overlap penalty and
  no-degradation selection gate.
- Added expert-only target-domain coverage and effect-quantile coverage
  pressure while preserving the localized overlap penalty and no-degradation
  selection gate. Focused conditional-calibration, public API, and LUMI workflow
  tests pass.
- Ran the compact tuning check in
  `/private/tmp/neural_hmsc_v8_domain_expert_target_pressure_tuning`. The
  stronger scalar pressure did not change selector behavior: no candidate
  passed the strengthened gate, and the best combined-shift target gain stayed
  at `0.0148`, below the `0.0200` practical-gain floor. Results are recorded in
  `docs/neural_hmsc_v8_gated_local_sanity_2026-07-13.md`.
- Next substep: change the domain-expert representation rather than adding more
  scalar objective weight. Add effect-bin-specific expert amplitudes or a
  target-domain-specific slope/cap parameterization that can move weak OOD
  effect quantiles without broad in-domain inflation. Rerun compact tuning
  before any rare32 workflow.
- Implemented effect-bin-specific expert amplitudes in the learned OOD
  effect-shift head. New fits emit a 14-parameter head: the previous
  pure-effect and combined-shift context gates plus three localized
  effect-bin log-amplitudes for each expert. Metadata records the bin centers,
  bin width, per-expert bin amplitudes, and parameter count. Legacy
  8-parameter heads remain loadable. Focused conditional-calibration tests and
  the public API / conditional calibration / LUMI workflow trio pass.
- Ran the compact tuning check in
  `/private/tmp/neural_hmsc_v8_domain_expert_bin_head_tuning`. The
  combined-shift expert passed the strengthened held-out selection gate with
  selected shrinkage `1.0`, combined-shift target gain `0.0222`, non-target
  effect-size gain `0.0037`, and localized overlap excess loss `0.0000`. The
  pure-effect expert still did not pass because its effect-size target gain was
  only `0.0074`. The compact run remains a selector/representation check, not
  a production-like qualification.
- Next substep: run the production-like rare32 local sanity workflow with the
  effect-bin-specific domain-expert head enabled, using the same rare
  calibration/validation settings as prior rare32 runs. Compare selected expert
  branch, bin amplitudes, held-out OOD coverage, effect-quantile coverage,
  final multiplier quantiles, in-domain gate deltas, and rare-validation gates
  against the previous target-pressure and localized-overlap compact baselines.
  Do not submit a LUMI comparison unless the rare32 local gates hold.
- Ran the production-like rare32 local sanity workflow in
  `/private/tmp/neural_hmsc_v8_domain_expert_bin_head_local_sanity_rare32`.
  The compact selector gain did not transfer: the selected expert remained
  `baseline` with `selected_shrinkage = 0.0` and no candidate passed the
  strengthened gate. Final coverage was `0.9350` for covariate shift,
  `0.8517` for effect-size shift, and `0.8183` for combined shift. The best
  combined-shift candidates improved held-out selector coverage, but failed
  localized overlap and in-domain extra-inflation controls; the strongest
  profile had combined-shift gain `0.0400` but localized overlap excess loss
  `1.2964`. Rare-validation gates remained acceptable with selected shrinkage
  `1.0`, overall coverage `0.9001`, rare coverage `0.9285`, and rare rank
  error `0.0194`.
- Next substep: do not submit a LUMI comparison. Make the effect-bin expert
  movement more domain-local and gate-compatible before another rare32 run.
  A plausible implementation is a support/design/community-conditioned
  bin-amplitude cap or selection step, plus direct penalties on per-bin
  in-domain extra-inflation and group-loss deltas during expert fitting.
- Implemented the first context-capped effect-bin expert. The serialized
  14-parameter head shape is unchanged, but pure-effect bin amplitudes are
  applied through support-excess or low-design context caps, and combined-shift
  bin amplitudes are applied through support-excess, low-design, and optional
  low-community caps. Expert fitting now adds per-effect-bin in-domain
  penalties for extra log inflation, rank-mean drift, and low coverage in each
  bin's active context. Focused conditional-calibration tests and the public
  API / conditional calibration / LUMI workflow trio pass.
- Ran the compact tuning check in
  `/private/tmp/neural_hmsc_v8_domain_expert_context_capped_bin_tuning`. The
  new cap/penalty controlled in-domain leakage but over-constrained target
  movement. No candidate passed the strengthened selector gate: the best
  combined-shift target gain fell to `0.0111`, below the `0.0200` floor, while
  overlap excess loss stayed controlled at `0.0271` to `0.0356` and
  extra-inflation delta stayed around `0.098`.
- Next substep: do not run rare32 yet. Tune or redesign the context-cap shape
  locally before another production-like workflow. Likely directions are softer
  community/design caps, target-domain-specific cap floors, or a two-stage fit
  that first learns target-domain bin movement and then projects it through the
  per-bin in-domain gate.
- Tested the first cap-shape redesign with fixed target-domain context floors
  while preserving the same 14-parameter effect-bin head shape. Pure-effect
  contexts now have a `0.20` floor and combined-shift contexts have a `0.35`
  floor; per-bin in-domain penalties remain active. Focused
  conditional-calibration tests and the public API / conditional calibration /
  LUMI workflow trio still pass.
- Ran the compact soft-floor tuning check in
  `/private/tmp/neural_hmsc_v8_domain_expert_soft_floor_bin_tuning`. The
  soft floors increased learned combined-bin amplitudes but did not improve the
  selected held-out outcome. No candidate passed the strengthened selector
  gate. The best combined-shift target gain stayed at `0.0111`, below the
  `0.0200` practical-gain floor; overlap excess loss remained controlled
  (`0.0274` to `0.0364`) and extra-inflation delta was about `0.103`. The
  selected branch remained `baseline` with mean held-out OOD `0.7185` and worst
  held-out OOD `0.7148`.
- Next substep: do not run rare32 yet. Implement a two-stage
  target-then-projection expert fit. First learn target-domain effect-bin
  movement without the full in-domain bin gate, then project or shrink those
  learned amplitudes through explicit per-bin in-domain gate constraints before
  selector acceptance.
- Implemented the two-stage target-then-projection expert profile inside the
  existing held-out domain-expert selector. The new profile keeps the serialized
  14-parameter effect-bin head unchanged. Stage 1 fits the target-domain expert
  with moderate target/effect-quantile pressure, reduced in-domain gate weight,
  relaxed overlap tolerance, and no per-bin in-domain penalty. Stage 2 evaluates
  the learned expert through the existing selector gates using the standard
  shrinkage grid crossed with branch-specific effect-head projection caps
  `(0.25, 0.5, 0.75, 1.0)`. Projection caps shrink only the active branch's
  scalar effect-head amplitude and three effect-bin amplitudes back toward the
  baseline vector before acceptance.
- Validation passed:
  `python -m py_compile pyhmsc/neural/conditional_calibration.py`,
  `pytest tests/test_neural_hmsc_conditional_calibration.py -q` (`18 passed`),
  and
  `pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q`
  (`41 passed`). The focused metadata test now asserts that the
  `two_stage_target_then_projection` selector profile and per-candidate
  projection-cap diagnostics are present.
- Next substep: run a compact local tuning check for the two-stage projection
  profile using the same rare calibration/validation settings as the soft-floor
  and hard-cap compact runs. Compare target gains, selected projection caps,
  overlap-control loss, in-domain gate deltas, and held-out OOD coverage against
  the soft-floor context-capped run before considering rare32.
- Ran the compact two-stage projection check in
  `/private/tmp/neural_hmsc_v8_domain_expert_two_stage_projection_tuning`. The
  new projection profile did not qualify and did not improve selected held-out
  coverage. The selector again kept `baseline` with mean held-out OOD `0.7185`,
  worst held-out OOD `0.7148`, effect-size-shift coverage `0.7222`, and
  combined-shift coverage `0.7148`. The best two-stage pure-effect candidate
  used shrinkage `0.25` and projection cap `0.75`, but target gain stayed
  `0.0037`. The best two-stage combined-shift candidate used shrinkage `1.0`
  and projection cap `0.25`; overlap excess loss fell to `0.0245`, but target
  gain stayed `0.0111` and max group extra-cap delta was `0.0889`, above the
  `0.0800` gate.
- Next substep: do not run rare32. Stop relying on scalar/bin-amplitude
  projection as the main fix. The compact result suggests projection can reduce
  in-domain leakage but cannot create enough held-out target-domain coverage
  movement. Change the target signal or calibration pool before another
  production-like workflow: either fit the expert on a larger/harder
  target-domain compact pool, or replace interval-coverage pressure with a
  margin-aware OOD loss that upweights near-miss OOD coefficients before the
  existing projection gate is applied.
- Implemented the margin-aware target-signal variant for the two-stage
  projection profile. The new `margin_weight` profile field is zero for the
  legacy localized profiles and positive for `two_stage_target_then_projection`.
  The loss upweights target-domain coefficients that are outside but near the
  baseline nominal interval, while keeping projection and held-out selector
  gates unchanged. Validation passed:
  `python -m py_compile pyhmsc/neural/conditional_calibration.py`,
  `pytest tests/test_neural_hmsc_conditional_calibration.py -q` (`18 passed`),
  and
  `pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q`
  (`41 passed`).
- Ran the compact margin-aware tuning check in
  `/private/tmp/neural_hmsc_v8_domain_expert_margin_projection_tuning`. The
  selector still kept `baseline`, and selected held-out OOD coverage was
  unchanged from the previous two-stage projection run: mean `0.7185`, worst
  `0.7148`, effect-size-shift `0.7222`, and combined-shift `0.7148`. The best
  margin-aware pure-effect candidate had target gain `0.0037`; the best
  margin-aware combined-shift candidate had target gain `0.0111`, overlap
  excess loss `0.0246`, extra-inflation delta `0.1015`, and max group
  extra-cap delta `0.0890`.
- Next substep: do not run rare32. Change the compact calibration data rather
  than adding another scalar target-loss term. Build a larger or explicitly
  harder target-domain OOD calibration pool, enriched for near-boundary misses
  and separated by pure-effect versus combined-shift regimes, then rerun the
  same projection selector locally.
- Implemented a hard target-domain OOD calibration pool in the benchmark
  runner. New CLI options
  `--conditional-calibration-ood-hard-target-multiplier` and
  `--conditional-calibration-ood-hard-target-candidate-multiplier` preserve the
  old behavior by default. When enabled, `effect_size_shift` and
  `combined_shift` calibration datasets are drawn from an over-sampled
  candidate pool, scored by near-boundary coefficient misses under the current
  posterior, and the hardest subset is retained before fitting the same
  projection selector. Non-target OOD regimes keep the original sampling path.
  Validation passed:
  `python -m py_compile examples/run_neural_hmsc_benchmark.py`,
  `pytest tests/test_neural_hmsc_lumi_workflow.py -q` (`5 passed`), and
  `pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q`
  (`42 passed`).
- Ran the compact hard-pool projection check in
  `/private/tmp/neural_hmsc_v8_domain_expert_hard_pool_projection_tuning` with
  `--conditional-calibration-ood-datasets 4`,
  `--conditional-calibration-ood-hard-target-multiplier 3`, and
  `--conditional-calibration-ood-hard-target-candidate-multiplier 2`. The
  selector still kept `baseline`, but held-out target coverage moved materially:
  mean held-out OOD `0.7568`, worst held-out OOD `0.7420`, effect-size-shift
  `0.7716`, and combined-shift `0.7420`. The best pure-effect candidate reached
  target gain `0.0185`, overlap excess loss `0.0003`, extra-inflation delta
  `0.1600`, and max group extra-cap delta `0.0912`. The best combined-shift
  candidate reached target gain `0.0160`, overlap excess loss `0.0752`,
  extra-inflation delta `0.1667`, and max group extra-cap delta `0.0913`.
- Next substep: do not run rare32 yet. Refine the hard-pool/projection
  interaction locally. The hard pool is the first recent change that materially
  increases target-domain gains, but it still misses the `0.0200` practical-gain
  floor and breaches the max group extra-cap delta gate. Either increase
  target-pool hardness modestly while adding a stricter projection cap for
  in-domain extra-cap loss, or split hard target batches into independent
  train/evaluation batches so accepted gains are not dominated by one
  within-batch split.
- Implemented independent hard target-pool batch grouping. When hard target
  selection is enabled for `effect_size_shift` or `combined_shift`, the
  selected target-domain datasets are emitted as two calibration batches. This
  makes the existing domain-expert selector use `alternating_batches` rather
  than `within_batch_axis0`. Validation passed:
  `python -m py_compile examples/run_neural_hmsc_benchmark.py`,
  `pytest tests/test_neural_hmsc_lumi_workflow.py -q` (`6 passed`), and
  `pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q`
  (`43 passed`).
- Ran compact split hard-pool checks in
  `/private/tmp/neural_hmsc_v8_domain_expert_hard_pool_split_projection_tuning`
  and
  `/private/tmp/neural_hmsc_v8_domain_expert_hard_pool_split_x4_projection_tuning`.
  The split fixed the extra-cap gate problem but did not qualify. For split x3,
  the selector kept `baseline` with mean held-out OOD `0.7630`, worst `0.7469`,
  effect-size-shift `0.7790`, and combined-shift `0.7469`. Best pure-effect
  target gain was `0.0111` with max group extra-cap delta `0.0075`; best
  combined-shift target gain was `0.0173` with max group extra-cap delta
  `0.0112`. Split x4 did not improve the result: mean held-out OOD fell to
  `0.7537`, worst to `0.7343`, and best target gains were `0.0130` for both
  pure-effect and combined-shift.
- Next substep: do not run rare32. Make hard-pool selection gate-aware rather
  than simply harder or split differently. Score candidate target-domain
  datasets by near-boundary misses while penalizing high in-domain-overlap
  contexts, or build separate hard pools for train and validation with matched
  near-boundary difficulty so target gain does not collapse under independent
  evaluation.
- Implemented gate-aware hard target-pool scoring. The selector still scores
  target-domain OOD candidates by near-boundary coefficient misses, but now
  subtracts a regime-specific overlap proxy for `effect_size_shift` and
  `combined_shift`. Validation passed:
  `python -m py_compile examples/run_neural_hmsc_benchmark.py`,
  `pytest tests/test_neural_hmsc_lumi_workflow.py -q` (`7 passed`), and
  `pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q`
  (`44 passed`).
- Ran the compact gate-aware hard-pool check in
  `/private/tmp/neural_hmsc_v8_domain_expert_gate_aware_hard_pool_tuning`. The
  selector still kept `baseline`, but selected held-out OOD diagnostics improved
  versus split x3: mean `0.7710`, worst `0.7642`, effect-size-shift `0.7778`,
  and combined-shift `0.7642`. Best pure-effect target gain was `0.0148` with
  non-target gain `0.0198`, overlap excess loss `0.0008`, extra-inflation delta
  `0.1891`, and max group extra-cap delta `0.0205`. Best combined-shift target
  gain was `0.0173` with non-target gain `0.0136`, overlap excess loss
  `0.0895`, extra-inflation delta `0.1843`, and max group extra-cap delta
  `0.0196`.
- Next substep: do not run rare32. Build matched train/validation hard pools
  rather than changing a scalar score again. The target selector needs training
  batches and evaluation batches with similar near-boundary difficulty;
  otherwise target gain either collapses under independent evaluation or
  remains below the `0.0200` practical-gain floor despite better gate control.
- Implemented score-balanced hard target-pool grouping. Target-domain
  calibration batches are now balanced by the same gate-aware near-boundary
  score used for candidate selection. Validation passed:
  `python -m py_compile examples/run_neural_hmsc_benchmark.py`,
  `pytest tests/test_neural_hmsc_lumi_workflow.py -q` (`8 passed`), and
  `pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q`
  (`45 passed`).
- Ran the compact matched hard-pool check in
  `/private/tmp/neural_hmsc_v8_domain_expert_matched_hard_pool_tuning`. The
  selector still kept `baseline`, and matched grouping worsened the selected
  held-out OOD result relative to gate-aware x3: mean `0.7611`, worst `0.7519`,
  effect-size-shift `0.7704`, and combined-shift `0.7519`. Best pure-effect
  target gain stayed `0.0148` with max group extra-cap delta `0.0095`; best
  combined-shift target gain fell to `0.0148` with max group extra-cap delta
  `0.0100`.
- Next substep: do not run rare32. Instrument the hard-pool selection path
  before another heuristic change. Record selected candidate score
  distributions, train/evaluation score summaries, near-boundary miss
  summaries, and overlap-proxy summaries by target regime so the
  training/evaluation mismatch can be diagnosed directly.
- Implemented hard-pool selection diagnostics in `benchmark_record.json`.
  Target hard-pool runs now record candidate and selected score distributions,
  raw near-boundary score summaries before overlap penalty, overlap-proxy
  summaries, miss-rate and excess-miss summaries, and matched train/evaluation
  group summaries by target regime. Validation passed:
  `python -m py_compile examples/run_neural_hmsc_benchmark.py`,
  `pytest tests/test_neural_hmsc_lumi_workflow.py -q` (`9 passed`), and
  `pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q`
  (`46 passed`).
- Ran the compact instrumentation check in
  `/private/tmp/neural_hmsc_v8_hard_pool_instrumentation_check`. The selector
  again kept `baseline` with mean held-out OOD `0.7611`, worst OOD-domain
  coverage `0.7519`, effect-size-shift coverage `0.7704`, and combined-shift
  coverage `0.7519`. The new diagnostics show effect-size-shift selected score
  mean `-0.0485`, selected overlap mean `0.2815`, and matched group score delta
  `0.0605`; combined-shift selected score mean `-0.0293`, selected overlap mean
  `0.2458`, and matched group score delta `0.1345`.
- Next substep: do not run rare32. Inspect the new diagnostic arrays in detail
  and redesign hard-pool construction around two explicit constraints: enough
  raw near-boundary misses and low target-domain overlap, with separate
  train/evaluation matching for each target regime.
- Implemented constrained hard-pool construction. Target-domain selection now
  separates raw near-boundary difficulty from target-domain overlap, relaxes
  overlap before relaxing raw difficulty if the eligible pool is too small, and
  matches train/evaluation hard pools by raw difficulty, overlap, and final
  score within each target regime. Validation passed:
  `python -m py_compile examples/run_neural_hmsc_benchmark.py`,
  `pytest tests/test_neural_hmsc_lumi_workflow.py -q` (`11 passed`), and
  `pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q`
  (`48 passed`).
- Ran the compact constrained hard-pool check in
  `/private/tmp/neural_hmsc_v8_constrained_hard_pool_check`. The selector still
  kept `baseline`, and held-out OOD worsened relative to the instrumentation
  baseline: mean `0.7525`, worst `0.7457`, effect-size shift `0.7593`, and
  combined shift `0.7457`. Matching improved substantially, with score deltas
  `0.0132` for effect-size shift and `0.0107` for combined shift, but both
  regimes had to relax the overlap threshold to the `0.95` quantile to keep
  enough raw near-boundary misses.
- Next substep: do not run rare32. Change candidate-pool generation, not
  selection. Generate or oversample target-domain candidates in low-overlap
  contexts first, then apply the constrained hard-pool selector and rerun the
  same compact local check.
- Implemented low-overlap target candidate-pool generation. Target regimes now
  generate a wider seed window, prefilter the generated pool into a low-overlap
  candidate pool while preserving a raw near-boundary miss floor, and then apply
  the constrained selector plus regime-specific train/evaluation matching.
  Validation passed: `python -m py_compile examples/run_neural_hmsc_benchmark.py`,
  `pytest tests/test_neural_hmsc_lumi_workflow.py -q` (`12 passed`), and
  `pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q`
  (`49 passed`).
- Ran the compact low-overlap candidate-pool check in
  `/private/tmp/neural_hmsc_v8_low_overlap_candidate_pool_check`. The selector
  still kept `baseline`. Mean held-out OOD improved relative to constrained
  hard-pool (`0.7562` versus `0.7525`) but remained below the instrumentation
  baseline (`0.7611`). Combined-shift overlap improved from generated overlap
  mean `0.2683` to candidate-pool overlap mean `0.2377`; selected combined
  score mean improved to `-0.0184`. In-domain extra-inflation loss improved to
  `0.1339`, and max extra-cap loss improved to `0.1060`, but no expert passed
  the selection gate.
- Next substep: do not run rare32. Change the simulated target-domain
  candidate distribution itself, not the seed-window prefilter. Add an explicit
  low-overlap target candidate regime or context-controlled OOD simulator
  variant that creates more low-overlap hard misses before applying the
  constrained selector.
- Implemented a simulator-level low-overlap target candidate context. Default
  OOD simulations remain unchanged, but hard target calibration candidate
  generation now uses `candidate_context="low_overlap"` for `effect_size_shift`
  and `combined_shift`. The effect-size target candidate context shifts
  covariate support to reduce pure-effect overlap, and the combined-shift
  target candidate context raises intercept context to reduce low-community
  overlap. Validation passed:
  `python -m py_compile pyhmsc/neural/simulator.py examples/run_neural_hmsc_benchmark.py`,
  `pytest tests/test_neural_hmsc_simulator.py tests/test_neural_hmsc_lumi_workflow.py -q`
  (`21 passed`), and
  `pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py tests/test_neural_hmsc_simulator.py -q`
  (`58 passed`).
- Ran the compact context-controlled candidate check in
  `/private/tmp/neural_hmsc_v8_context_controlled_low_overlap_check`. The
  selector still kept `baseline`, but target held-out OOD improved sharply:
  mean OOD `0.7969`, worst OOD `0.7580`, effect-size shift `0.8358`, and
  combined shift `0.7580`. Candidate quality improved: effect-size generated
  overlap mean `0.1308`, pool overlap mean `0.1050`, selected score mean
  `0.0347`; combined-shift generated overlap mean `0.2311`, pool overlap mean
  `0.1970`, selected score mean `-0.0012`. The remaining blocker is expert
  acceptance: best pure-effect target gain was `0.0173` with extra-inflation
  delta about `0.3090`, and best combined-shift target gain was `0.0136` with
  extra-inflation delta about `0.2643`.
- Next substep: do not run rare32. Keep the context-controlled candidate
  distribution, but redesign expert fitting/acceptance for it. Reduce
  in-domain extra inflation and improve combined-shift target gain, likely via
  a smaller expert-amplitude schedule or a combined-shift-specific target loss
  followed by strict extra-inflation projection before selection.
- Implemented a finer expert shrinkage schedule, strict branch-specific
  projection caps, a combined-shift-specific target-loss profile, and
  gate-compatible projection selection. Candidate records now prefer the best
  gate-compatible projection row when no row clears the full target-gain floor,
  instead of reporting the highest-gain high-inflation row. Validation passed:
  `python -m py_compile pyhmsc/neural/conditional_calibration.py`,
  `pytest tests/test_neural_hmsc_conditional_calibration.py -q` (`18 passed`),
  and
  `pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py tests/test_neural_hmsc_simulator.py -q`
  (`58 passed`).
- Ran the compact gate-compatible projection check in
  `/private/tmp/neural_hmsc_v8_gate_compatible_projection_check`. The selector
  still kept `baseline` with the same selected held-out OOD diagnostics as the
  context-controlled candidate run: mean `0.7969`, worst `0.7580`,
  effect-size shift `0.8358`, and combined shift `0.7580`. The best
  gate-compatible pure-effect candidate had target gain `0.0136`, extra-
  inflation delta `0.2369`, and max extra-cap delta `0.0177`; the best
  gate-compatible combined-shift candidate had target gain `0.0099`, extra-
  inflation delta `0.1837`, and max extra-cap delta `0.0137`.
- Next substep: do not run rare32. Move the extra-inflation constraint into
  expert fitting itself instead of relying on post-fit projection. Fit a
  combined-shift expert with an in-objective gate-compatible amplitude penalty
  or target-domain curriculum that can recover combined-shift gain while
  staying below the extra-inflation gate.

## Roadmap Reset: Competing With The Baseline

The recent v8 work established useful diagnostics, but it is no longer
productive to continue the same incremental selector/projection loop. The
central result is that the domain-expert path has not yet produced a deployed
calibration that beats the frozen baseline under the acceptance gates. In most
compact checks the selector still chose `baseline`, so the final applied model
was unchanged even when candidate diagnostics looked better.

The key retrospective results are:

| Step | Best Result | What It Proved | Limitation |
| --- | --- | --- | --- |
| hard target-pool selection | target gains reached about `0.0185` pure-effect and `0.0160` combined-shift | near-boundary OOD examples can create useful signal | missed the `0.0200` gain floor and breached extra-cap gates |
| split/matched hard pools | max group extra-cap deltas fell near `0.01` | train/evaluation splitting can control obvious overfit | target gains collapsed or held-out OOD worsened |
| gate-aware scoring | mean internal OOD rose to `0.7710`; combined-shift to `0.7642` | overlap-aware target selection is better than raw hardness | still selected baseline |
| constrained selection | group deltas fell below about `0.014` | pool matching worked technically | overlap constraints had to relax to `0.95`, showing pool scarcity |
| low-overlap seed prefilter | extra-inflation loss improved to `0.1339` | low-overlap filtering reduced in-domain inflation pressure | did not improve final selected model |
| context-controlled simulator | effect-size internal coverage reached `0.8358` | candidate distribution quality improved materially | changed the evaluation/calibration pool, so it was not an apples-to-apples baseline comparison |
| gate-compatible projection | extra-inflation deltas became gate-compatible | projection can control risk | target gains dropped to `0.0136` pure-effect and `0.0099` combined-shift |

The main shortcomings are:

- The selected model has usually remained `baseline`; candidate improvements
  therefore do not translate into a better final calibration.
- Several comparisons changed the target calibration/evaluation pool itself.
  Those diagnostics are useful for understanding candidate quality, but they do
  not prove improvement over the frozen scalar/v4/v5/v6/v8 baseline on an
  independent OOD suite.
- The current acceptance gate requires both practical target gain and low
  in-domain inflation. The expert family has shown a tradeoff: high-gain rows
  increase in-domain extra inflation, while gate-compatible projections lose
  target gain.
- Most recent changes are post-hoc scale manipulations. They are not changing
  the posterior mean, posterior shape, or representation that creates the
  misspecification, so there is limited headroom.
- Combined-shift remains the hard regime. The current expert representation is
  too blunt: it moves enough to help only when it also inflates in-domain
  contexts, and strict projection removes the gain.

The revised objective is:

> Produce a final selected calibration that beats the frozen scalar baseline on
> a fixed, independent OOD validation suite while preserving in-domain and rare
> validation gates.

This changes the development rule. Do not treat internal calibration-batch
coverage as success. A change only advances when it improves a fixed
independent comparison against the frozen baseline.

### Revised Acceptance Gates

Use a frozen evaluation bundle before any more model work:

- fixed in-domain SBC suite,
- fixed rare-validation suite,
- fixed OOD suite with `covariate_shift`, `effect_size_shift`, and
  `combined_shift`,
- fixed seeds and fixed data-generating settings,
- frozen scalar/v4/v5/v6/v8 baseline checkpoints,
- final selected calibration output, not merely candidate diagnostics.

A candidate is promotable only if:

- final selected model is not equivalent to `baseline`, or it beats baseline as
  an external wrapper in the fixed evaluation harness,
- mean OOD coverage improves by at least `0.010` over frozen scalar baseline,
- worst OOD-domain coverage improves or is no worse than baseline by more than
  `0.005`,
- combined-shift coverage improves by at least `0.010`,
- in-domain coverage remains within the existing acceptance window,
- rare-validation gates remain satisfied,
- extra-inflation and max extra-cap deltas remain below current gates,
- the result holds on at least three compact local seeds before LUMI.

### Revised Technical Direction

Stop prioritizing domain-expert projection tuning. The next candidate should be
a clearly different competitor to the scalar baseline:

1. Fixed Evaluation Harness

   Build a local script that evaluates frozen baselines and candidate
   calibrations on the same independent suites. It should write one comparison
   table with in-domain, rare, effect-size, combined-shift, worst-domain, and
   extra-inflation metrics. This is now the first required step.

2. Conservative External Calibrator

   Implement a post-hoc external calibrator that does not mutate the learned
   OOD head. A good first competitor is a monotone, context-stratified scale
   calibrator fitted on calibration batches and evaluated on independent
   validation batches. It should have very few degrees of freedom:

   - separate intercepts for effect-size and combined-shift,
   - monotone bins by effect-size and support,
   - explicit in-domain inflation budget,
   - shrinkage selected by independent validation,
   - fallback to scalar baseline when gates fail.

   This creates a stronger baseline competitor than the current neural expert
   family because it can be evaluated as a wrapper with transparent degrees of
   freedom and strict external selection.

3. Representation-Level Alternative

   If the conservative calibrator cannot beat scalar baseline, stop tuning
   scale heads and change the posterior model itself. The likely representation
   issue is posterior mean/scale misspecification under rare and combined-shift
   contexts. The next representation-level candidate should train the neural
   posterior with OOD-aware simulation batches and an explicit combined-shift
   validation loss, rather than fitting OOD scale corrections after training.

4. Competing Model Track

   Keep a second track for a genuinely different method:

   - ensemble or MC-dropout uncertainty expansion,
   - conformalized coefficient intervals by context strata,
   - mixture-of-experts posterior head with a learned domain/context router,
   - simulation-trained quantile regression for coefficient intervals rather
     than Gaussian scale inflation.

   Each track must plug into the same fixed evaluation harness.

### Revised Immediate Roadmap

1. Build the fixed independent evaluation harness and regenerate one compact
   report comparing frozen scalar/v4/v5/v6/v8/default candidates on identical
   data.

   Status: implemented a first fixed-evaluation harness in
   `examples/compare_neural_hmsc_fixed_evaluation.py`. The harness consumes
   existing benchmark output directories, requires identical SBC/OOD row keys
   across runs, compares final calibrated rows against a named baseline, and
   writes JSON, CSV, and Markdown reports with mean OOD, worst OOD,
   combined-shift, in-domain, rare-prevalence, and stratum-level deltas. It
   applies explicit acceptance gates so candidate diagnostics cannot be
   mistaken for a selected model improvement.

   Validation:

   - `python -m py_compile examples/compare_neural_hmsc_fixed_evaluation.py`
   - `pytest tests/test_neural_hmsc_lumi_workflow.py -q` (`13 passed`)

   Smoke check:

   - Compared
     `/private/tmp/neural_hmsc_v8_context_controlled_low_overlap_check` against
     `/private/tmp/neural_hmsc_v8_gate_compatible_projection_check` with
     `context` as the baseline.
   - Output written to `/private/tmp/neural_hmsc_fixed_eval_smoke`.
   - The harness reported `80` fixed row keys and zero OOD/combined-shift delta
     for the projection run, confirming the main retrospective finding: the
     final calibrated evaluation did not improve even when internal candidate
     diagnostics changed.

   Compact fixed-bundle check:

   - Generated a local compact probit bundle under
     `/private/tmp/neural_hmsc_fixed_eval_compact_bundle_20260716` with
     `scalar`, `v4`, `v5`, `v6`, `v8`, and `default` candidate directories.
     All non-scalar candidates reused the frozen scalar checkpoint and shared
     `--seed 20260716`, `--model-seed 20260716`, `--sbc-datasets 8`,
     `--sbc-draws 64`, and the same `covariate_shift`, `effect_size_shift`,
     and `combined_shift` OOD regimes. Historical v4/v5/v6 result cache
     entries were present locally only as empty directory shells, so `v4` and
     `v5` in this compact bundle are current-code generated analogues rather
     than restored historical binaries.
   - Ran
     `examples/compare_neural_hmsc_fixed_evaluation.py` with `scalar` as the
     baseline. Output was written to
     `/private/tmp/neural_hmsc_fixed_eval_compact_bundle_20260716/comparison`.
     The harness accepted `80` fixed row keys, confirming that all candidates
     were evaluated on identical independent SBC/OOD rows.
   - Scalar failed the improvement gates by construction
     (`mean_ood_coverage_95 = 0.7148`, `combined_shift_coverage_95 = 0.5972`).
     All generated non-scalar candidates passed the fixed-evaluation gates.
     The strongest compact result was the rare-aware/default conditional path:
     `mean_ood_coverage_95 = 0.8392`,
     `worst_ood_coverage_95 = 0.7704`,
     `effect_size_shift_coverage_95 = 0.7704`,
     `combined_shift_coverage_95 = 0.8407`,
     `in_domain_coverage_95 = 0.9306`, and
     `rare_prevalence_coverage_95 = 0.8095`.
   - The generated `v8` hard-target/effect-gated path passed but did not beat
     the simpler rare-aware/default path in this fixed local bundle
     (`mean_ood_coverage_95 = 0.8272`,
     `effect_size_shift_coverage_95 = 0.7546`,
     `combined_shift_coverage_95 = 0.8324`). This reinforces the revised
     conclusion: the next competitor should not be another mutation of the
     domain-expert/hard-target family.

   Status: complete for local compact validation.

   External monotone competitor:

   - Implemented `--coefficient-calibration external_monotone` as a separate
     competitor. It first fits the ordinary conditional/default calibration,
     then fits a held-out external context-stratified monotone scale wrapper
     with three ordered effect-size bins and an activation ramp from support
     excess or large posterior effect signal. The wrapper is serialized in
     calibration metadata as `external_context_monotone`; failed gates select
     zero offsets and fall back to the baseline conditional calibration.
   - Added `--external-monotone-datasets`,
     `--external-monotone-max-multiplier`,
     `--external-monotone-min-ood-gain`, and
     `--external-monotone-min-combined-gain` to the benchmark runner.
   - Validation passed:
     `python -m py_compile pyhmsc/neural/conditional_calibration.py examples/run_neural_hmsc_benchmark.py`,
     `pytest tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q`
     (`32 passed`), and `pytest tests/test_neural_hmsc_public_api.py -q`
     (`19 passed`).

   Three-seed compact local gate:

   - Ran seeds `20260716`, `20260717`, and `20260718` under
     `/private/tmp/neural_hmsc_external_monotone_3seed_20260716`. Each seed
     used a fresh scalar checkpoint, then compared `scalar`, `default`, and
     `external_monotone` on identical fixed SBC/OOD rows. The external wrapper
     selected nonzero offsets in all three seeds with selected shrinkage `1.0`
     and selected log offsets `[0.0, 0.0, 0.6931]`.
   - Mean across the three compact seeds:
     `external_monotone` achieved `mean_ood_coverage_95 = 0.8636`,
     `worst_ood_coverage_95 = 0.8191`,
     `effect_size_shift_coverage_95 = 0.8191`,
     `combined_shift_coverage_95 = 0.8448`,
     `in_domain_coverage_95 = 0.9392`, and
     `rare_prevalence_coverage_95 = 0.8613`.
   - The same seeds gave `default` mean OOD `0.8277`, worst OOD `0.7420`,
     effect-size shift `0.7420`, and combined shift `0.8228`; scalar mean OOD
     was `0.7204` and combined shift `0.6009`. `external_monotone` therefore
     passed the local gate and outperformed both scalar and default on the
     target OOD summaries while keeping in-domain and rare-prevalence coverage
     inside the acceptance window.

   Five-seed LUMI fixed-evaluation gate:

   - Submitted and completed LUMI job `19940765` on `dev-g`; elapsed time was
     `00:06:08`, exit code `0:0`, workflow wall time `317` seconds.
   - Results are documented in
     `docs/neural_hmsc_external_monotone_lumi_comparison_2026-07-16.md`.
   - The run root was
     `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_external_monotone_fixed_eval_20260716`.
   - Across five seeds, `external_monotone` selected nonzero offsets in every
     seed with shrinkage `1.0` and log offsets `[0.0, 0.0, 0.6931]`.
   - Five-seed means: `external_monotone` achieved mean OOD `0.8596`,
     worst OOD `0.8120`, effect-size shift `0.8120`, combined shift `0.8472`,
     in-domain `0.9404`, and rare-prevalence `0.8741`. The corresponding
     `default` means were mean OOD `0.8241`, worst OOD `0.7356`, effect-size
     shift `0.7356`, and combined shift `0.8244`. Scalar mean OOD was `0.7249`
     and combined shift was `0.6063`.

   Status: complete for implementation, three-seed compact local validation,
   and five-seed LUMI fixed-evaluation validation.

   Promotion decision: do not promote immediately from the compact gate alone.
   The five-seed result is strong, but the evaluation used compact settings
   (`sbc_datasets = 8`, `sbc_draws = 64`, `train_datasets = 8`). Before
   changing the default recommendation, run one production-shape confirmation
   with larger independent SBC/OOD evaluation counts. The confirmation should
   keep the same fixed-evaluation discipline and compare `scalar`, `default`,
   and `external_monotone` on shared seeds.

   Next substep: submit a production-shape LUMI confirmation, preferably with
   `sbc_datasets >= 32`, `sbc_draws >= 256`, and the same five seeds. Promote
   `external_monotone` only if it still beats `default` on mean OOD, worst OOD,
   effect-size shift, and combined shift while preserving in-domain and
   rare-prevalence acceptance.

2. Mark the current domain-expert/projection family as paused. Keep the code
   and diagnostics, but do not add another projection or shrinkage heuristic
   until an external evaluation report shows it is the bottleneck.

3. Implement the conservative external context-stratified monotone calibrator
   as a new competitor, not as another mutation of the domain-expert selector.
   Status: complete locally.

4. Run three compact local seeds against the frozen scalar baseline. Only if
   the candidate wins on mean OOD, worst OOD, and combined-shift while passing
   in-domain and rare gates should it move to a five-seed LUMI comparison.
   Status: complete; local gate passed.

5. Run a five-seed LUMI fixed-evaluation comparison for `scalar`, `default`,
   and `external_monotone`.
   Status: complete; `external_monotone` passed and beat both baselines.

6. Decide promotion policy for `external_monotone`: either promote it as the
   default compact competitor now, or run one production-shape confirmation with
   larger SBC/OOD evaluation counts first.
   Status: decided; run production-shape confirmation before promotion.

7. Run the production-shape LUMI confirmation with larger SBC/OOD evaluation
   counts for `scalar`, `default`, and `external_monotone`.
   Status: complete; LUMI job `19942240` completed with exit code `0:0`.
   Results are documented in
   `docs/neural_hmsc_external_monotone_production_confirmation_2026-07-16.md`.
   `external_monotone` passed all five fixed-evaluation seeds and beat
   `default` on mean OOD, worst OOD, effect-size shift, and combined shift
   while preserving in-domain and rare-prevalence acceptance. It qualifies for
   promotion as the default compact competitor, with the caveat that the
   combined-shift margin over `default` was small.

8. Promote `external_monotone` in the benchmark workflow default path while
   keeping `default` available as the legacy conditional baseline for direct
   comparisons.
   Status: implemented in `docs/lumi_neural_hmsc_benchmark_sbatch.sh`. The LUMI
   benchmark workflow now defaults `COEFFICIENT_CALIBRATION` to
   `external_monotone`, passes the external monotone calibration controls, and
   includes rare calibration/validation dataset controls. The legacy
   conditional baseline remains available with
   `COEFFICIENT_CALIBRATION=conditional`; scalar remains available with
   `COEFFICIENT_CALIBRATION=scalar`.

9. If the conservative calibrator fails on later larger or real-data
   confirmation, pivot to
   representation-level training changes rather than more post-hoc scale
   tuning.

10. Run Whittaker promoted-default real-data requalification with the Python
    MCMC reference retained as comparator.
    Status: complete; LUMI job `19948534` completed with exit code `0:0`.
    Results are documented in
    `docs/neural_hmsc_whittaker_external_monotone_requalification_2026-07-16.md`.
    The promoted `external_monotone` coefficient calibration passed coefficient
    SBC acceptance, held-out predictive acceptance, and the combined
    qualification gate on the fixed-effect Whittaker split. The predictive-only
    artifact passed the real held-out gate, but MCMC remained better on Brier
    score, log loss, prevalence MAE, and richness MAE. This supports real-data
    transfer utility for the promoted neural calibration, not exact posterior
    equivalence.

11. Run a direct Whittaker Python-only HMSC parity comparison against the
    original R-created HMSC model exported through the R+Python HMSC-HPC
    boundary. This should use the same split, formula, MCMC settings, posterior
    summaries, held-out predictions, and qualitative HMSC book checks. Keep
    this separate from the neural `external_monotone` evidence.
    Status: complete for fixed-effect Whittaker trait/phylogeny. Initial LUMI
    job `19967782` found preprocessing mismatches in `XScaled` and `TrScaled`;
    LUMI job `19983202` passed after Python-native preprocessing was corrected.

12. Extend direct R/Python parity beyond Whittaker trait/phylogeny to compact
    fixture coverage: fixed-effect no-trait first, then iid random intercept,
    then spatial random effects only after inspecting R/Hmsc spatial boundary
    semantics explicitly.
    Status: implementation added. `examples/run_direct_r_python_parity.py`
    provides a reusable direct parity workflow for YAML configs,
    `docs/lumi_direct_r_python_parity_sbatch.sh` runs
    `examples/projects/simulated_poisson_recovery/model.yaml` and
    `examples/projects/simulated_spatial_validation/model_iid.yaml`, and the R
    bridge now emits iid `studyDesign`/`ranLevels` instead of silently dropping
    random levels. Status: complete; results are documented in
    `docs/direct_r_python_hmsc_fixture_parity_2026-07-18.md`. First LUMI
    attempt `19983642` used overly small smoke
    fixtures and strict absolute-error deltas; it passed fixed-effect boundary
    equality but failed posterior/predictive gates. The rerun should use the
    corrected fixture set and non-degradation predictive semantics. Attempt
    `19983758` exposed a second preprocessing rule: R/Hmsc leaves binary 0/1
    indicator columns unscaled. Attempt `19983981` passed the fixed-effect
    boundary arrays after that fix, but failed posterior-summary correlations;
    compact fixture posterior correlations are now treated as diagnostics so
    this workflow can enforce boundary parity and predictive non-degradation
    while still reporting stochastic posterior disagreement. LUMI job
    `19984923` passed the corrected fixture workflow for
    `simulated_poisson_recovery` and `simulated_spatial_validation/model_iid.yaml`.
    Spatial boundary inspection is now documented in
    `docs/spatial_r_python_hmsc_boundary_inspection_2026-07-18.md`: Full and
    GPP distance arrays match the R boundary exactly, NNGP requires ragged-list
    versus padded-tensor normalization, and Python-native spatial compilation
    now uses the R/Hmsc 101-point default `alphapw` grid instead of the previous
    one-point compact support unless `alphapw` or `alpha` is explicitly
    provided. Corrected inspection job `19995051` passed with exact `alphapw`
    equality for Full, GPP, and NNGP. Direct spatial parity is now complete for
    compact Full, GPP, and NNGP fixtures: Full passed in retry job `19995352`;
    GPP and NNGP passed in job `19999784` after padding variable random-level
    posterior shapes and applying the native GPP jitter/clipping stabilization
    to the legacy RDS import path. Decision recorded in
    `docs/python_only_hmsc_parity_decision_2026-07-19.md`: compact fixtures are
    sufficient for a bounded boundary/parity implementation claim, but not for
    the broader Python-only HMSC ecological-spatial parity claim. Larger
    real-data full-spatial requalification was then added for
    `examples/projects/big_spatial_plants_validation/model_spatial_full.yaml`.
    LUMI job `20000066` completed both samplers; the final report was
    regenerated from existing posteriors using `--reuse-existing-posteriors`
    after correcting stale remote source/data sync. The Big Spatial run passed
    boundary and predictive gates: `Y`, `X`, `T`, and `Pi` matched; `Beta`
    diagnostic correlation was `0.984128`; `Gamma` diagnostic correlation was
    `0.999570`; spatial association diagnostic correlation was `0.749967`; and
    Python-native predictive MAE was better than the R-boundary comparator
    (`0.099725` vs `0.113449`). Next action: return to neural work with
    Python-only parity scaffolding considered adequate for the current feature
    scope.

13. Use the qualified Python-only/R-boundary comparator when evaluating the
    promoted neural `external_monotone` path on real-data transfer.
    Status: complete. Implemented in the Whittaker and Big Spatial neural real-data
    runners. `examples/run_neural_hmsc_whittaker.py` and
    `examples/run_neural_hmsc_big_spatial_transfer.py` now accept
    `--reference-parity-metrics`; when a passed direct R/Python parity metrics
    JSON is supplied, the MCMC reference row is labelled
    `qualified_python_mcmc_fixed` and the generated report, acceptance JSON,
    and run metadata record the boundary parity provenance. The corresponding
    LUMI sbatch scripts expose `REFERENCE_PARITY_METRICS` and
    `QUALIFIED_REFERENCE_LABEL` without changing the default local behavior.
    LUMI job `20000918` completed the promoted Whittaker `external_monotone`
    requalification with the passed Whittaker parity metrics attached; it
    passed coefficient SBC, held-out predictive, combined qualification, and
    reference parity gates. Whittaker predictive-only metrics were Brier
    `0.077161`, log loss `0.273998`, macro AUC `0.549285`, prevalence MAE
    `0.077720`, and richness MAE `3.716901`; the qualified Python MCMC
    comparator remained stronger on Brier/log loss/prevalence/richness while
    matching macro AUC. Big Spatial dependent job `20000925` and retry
    `20001335` exposed transfer-runner assumptions about promoted calibration
    artifact separation; after fixing the runner to load coefficient
    calibration from `neural_posterior.h5` and predictive calibration from
    `neural_predictive_distribution.h5`, LUMI job `20001432` completed the
    Big Spatial frozen-transfer check with the Big Spatial parity metrics
    attached. Big Spatial predictive transfer passed: predictive-only metrics
    were Brier `0.052811`, log loss `0.211215`, macro AUC `0.632069`,
    prevalence MAE `0.059852`, and richness MAE `5.148121`; the qualified
    Python MCMC comparator remained stronger.

14. Decide whether to run one multi-seed real-data sensitivity check or return
    directly to simulated neural competitor development.
    Status: complete. The bounded three-seed real-data sensitivity check ran
    first, and the result supports returning to simulated neural competitor
    development. The workflow was deliberately limited: three Whittaker seeds
    with attached Whittaker parity metrics, each feeding a matching Big Spatial
    frozen-transfer run with attached Big Spatial parity metrics. It aggregated
    pass rate, predictive-only Brier/log-loss ratios versus uncalibrated and
    qualified Python MCMC, macro AUC deltas, prevalence/richness MAE ratios,
    source SBC coverage/rank moments, and runtime. It did not introduce new
    calibration objectives or tune thresholds. The workflow is implemented as
    `docs/lumi_neural_hmsc_realdata_sensitivity_sbatch.sh`, with aggregation in
    `examples/aggregate_neural_hmsc_realdata_sensitivity.py`.
    LUMI job `20001710` completed on `dev-g` with seeds `20260721`,
    `20260722`, and `20260723`, elapsed wall time `1962` seconds, run root
    `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_realdata_sensitivity_20260719`.
    The downloaded aggregate is in
    `/private/tmp/neural_realdata_sensitivity_20001710`, and the summarized
    report is `docs/neural_hmsc_realdata_sensitivity_2026-07-19.md`.
    All six dataset-seed rows completed and passed. The paired pass count was
    `3/3`, and the qualified Python MCMC comparator retained a Brier/log-loss
    advantage in all three paired seeds. Whittaker means were Brier ratio
    versus qualified MCMC `1.0386`, log-loss ratio `1.0576`, macro AUC delta
    `0.0019`, SBC coverage `0.9550`, rank mean `0.4943`, and rank variance
    `0.0702`. Big Spatial means were Brier ratio versus qualified MCMC
    `1.0945`, log-loss ratio `1.0866`, and macro AUC delta `-0.0167`.
    Decision: `stable_return_to_competitor_development`. The evidence supports
    stable neural predictive transfer with qualified comparator provenance, but
    it confirms that qualified Python MCMC remains stronger on core real-data
    proper scores.

15. Resume simulated neural competitor development with the qualified real-data
    evidence treated as a benchmark constraint.
    Status: implementation added. The first competitor in this return-to-neural
    phase is a conservative predictive-only Beta-mean calibrator, exposed as
    `--predictive-mean-calibration affine_shrinkage` in
    `examples/run_neural_hmsc_benchmark.py`. It fits a scalar affine/shrinkage
    correction on simulated calibration posterior means, gates selection on an
    independent simulated validation pool, and falls back to identity if the
    validation RMSE improvement is not positive enough. It writes metadata under
    `predictive_mean_calibration` on `neural_predictive_distribution.h5` with
    `artifact_role = predictive_only_mean`; it does not alter
    `neural_posterior.h5`, coefficient-posterior calibration, SBC rank
    diagnostics, OOD gates, rare-validation gates, or real-data acceptance
    thresholds. A local smoke run passed with:
    `python3 examples/run_neural_hmsc_benchmark.py --output /private/tmp/neural_mean_calibration_smoke --suite probit --n-sites 12 --n-species 2 --train-datasets 4 --calibration-datasets 3 --predictive-mean-calibration affine_shrinkage --predictive-mean-calibration-validation-datasets 2 --predictive-mean-calibration-min-improvement 0.0001 --epochs 2 --batch-size 2 --neural-chains 1 --neural-draws 8 --sbc-datasets 0`.
    The smoke selected a non-identity calibrator with validation RMSE ratio
    `0.9420` while retaining the existing predictive-only scale calibration
    metadata separately.

16. Run a compact fixed-evaluation simulated comparison for the new
    predictive-mean competitor.
    Status: complete; do not submit to LUMI. The compact local comparison is
    documented in
    `docs/neural_hmsc_predictive_mean_compact_comparison_2026-07-19.md`.
    It compared `external_monotone` against `external_monotone +
    affine_shrinkage` on shared probit seeds, reused the baseline neural
    checkpoint for the affine run, and wrote both MCMC-reference predictive
    rows and fixed SBC/OOD rows. The affine calibrator correctly wrote
    predictive-only metadata, but it rejected the non-identity candidate on the
    independent validation pool and fell back to identity: selected `false`,
    slope `1.0000`, intercept `0.0000`, validation RMSE ratio `1.0000`.
    Neural predictive RMSE was unchanged (`0.3493` uncalibrated and `0.3529`
    calibrated in both runs), and coefficient SBC/OOD rows were unchanged as
    expected. The compact fixed-evaluation gate also did not qualify because
    in-domain coverage was `0.8750`, below the frozen `0.9000` gate. This
    candidate is therefore mechanically valid but not promotable.

17. Redesign the predictive-mean competitor around response-scale proper-score
    selection instead of global coefficient-RMSE affine fitting.
    Status: implementation added. The compact comparison showed that a global
    affine correction to Beta means can improve calibration-pool coefficient
    RMSE while failing independent validation and producing no predictive
    movement. The new implementation adds
    `--predictive-mean-calibration probit_response_affine`, which fits bounded
    slope/intercept candidates against simulated response-scale Brier plus
    log-loss, then accepts the selected candidate only if an independent
    validation pool improves the combined score without exceeding configured
    Brier/log-loss degradation ratios. It falls back to identity otherwise.
    The implementation keeps the same separation: `neural_posterior.h5`,
    coefficient uncertainty calibration, SBC rank diagnostics, OOD gates,
    rare-validation gates, Whittaker gates, and Big Spatial gates stay frozen.
    Only `neural_predictive_distribution.h5` receives the predictive-only mean
    metadata. A local smoke run passed:
    `python3 examples/run_neural_hmsc_benchmark.py --output /private/tmp/neural_response_mean_smoke --suite probit --n-sites 12 --n-species 2 --train-datasets 4 --calibration-datasets 3 --predictive-mean-calibration probit_response_affine --predictive-mean-calibration-validation-datasets 2 --predictive-mean-calibration-min-improvement 0.0001 --epochs 2 --batch-size 2 --neural-chains 1 --neural-draws 8 --sbc-datasets 0`.
    The smoke selected a non-identity correction with slope `1.25`, intercept
    `-0.0250`, validation Brier ratio `0.9569`, and validation log-loss ratio
    `0.9530`, and wrote these metrics under `predictive_mean_calibration`.

18. Run a compact fixed-evaluation simulated comparison for the
    response-scale predictive-mean competitor.
    Status: complete; not yet LUMI-ready. The compact comparison is documented
    in `docs/neural_hmsc_response_mean_compact_comparison_2026-07-19.md`. It
    compared `external_monotone` against `external_monotone +
    probit_response_affine` on shared probit seeds, reusing the baseline neural
    checkpoint and fixed SBC/OOD settings. The response-scale selector accepted
    a non-identity predictive-only correction: slope `1.2500`, intercept
    `0.0250`, validation Brier ratio `0.9911`, and validation log-loss ratio
    `0.9877`. Calibrated predictive RMSE improved from `0.3529` to `0.3505`
    (`0.9934x`). Coefficient SBC/OOD rows were unchanged, as intended. The
    compact fixed-evaluation acceptance flag did not pass because in-domain
    coverage was `0.8750`, below the frozen `0.9000` gate, identical to the
    baseline. This is likely sensitive to the very small `sbc_datasets = 8`
    compact check, but the result is not enough to submit a five-seed LUMI
    comparison.

19. Run a larger local fixed-evaluation confirmation for
    `probit_response_affine`.
    Status: complete; not promotable on this compact checkpoint. The larger
    local confirmation is documented in
    `docs/neural_hmsc_response_mean_larger_local_confirmation_2026-07-19.md`.
    It reused the same frozen checkpoint and seed schedule as the compact
    comparison, but increased fixed SBC/OOD evaluation from `8 x 64` to
    `24 x 128`. The response-scale selector again accepted the non-identity
    predictive-only correction: slope `1.2500`, intercept `0.0250`, validation
    Brier ratio `0.9911`, and validation log-loss ratio `0.9877`. Calibrated
    predictive RMSE again improved from `0.3529` to `0.3505` (`0.9934x`).
    Coefficient SBC/OOD rows were unchanged, as required. However, in-domain
    coefficient coverage was `0.8519` for both baseline and response candidate,
    below the frozen `0.9000` gate. This shows the compact in-domain failure was
    not rescued by larger local evaluation and is not caused by the
    response-mean layer; it reflects the underqualified compact checkpoint or
    coefficient calibration setup used for this local experiment. Do not submit
    this result to five-seed LUMI.

20. Evaluate `probit_response_affine` on a previously qualified
    external-monotone baseline.
    Status: complete; viable predictive-only competitor, not default yet. Added
    `examples/compare_neural_hmsc_predictive_scores.py` to compute direct
    probit Brier, log-loss, predictive RMSE, prevalence MAE, and richness MAE
    from `neural_predictive_distribution.h5` plus benchmark `data/X.csv` and
    `data/Y.csv`, avoiding an unnecessary MCMC-reference rerun. Added
    `docs/lumi_neural_hmsc_response_mean_production_eval_sbatch.sh` to reuse
    the previously qualified production-shape external-monotone baseline from
    LUMI job `19942240`:
    `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_external_monotone_production_confirm_20260716`.
    The script runs only `external_monotone + probit_response_affine` for the
    same five seeds (`20260716` through `20260720`), reusing each seed's scalar
    checkpoint, shape `40 x 75`, `sbc_datasets = 32`, `sbc_draws = 256`,
    `external_monotone_datasets = 4`, and the same fixed OOD regimes. It then
    compares fixed SBC/OOD gates against the existing qualified
    `external_monotone` baseline and writes predictive-score comparisons for
    baseline versus response candidate. Remote validation passed, and LUMI job
    `20005059` completed on `dev-g`; run root:
    `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_response_mean_production_eval_20005059`.
    The aggregate is documented in
    `docs/neural_hmsc_response_mean_production_eval_2026-07-19.md`. Wall time
    was `205` seconds. Fixed coefficient/SBC/OOD metrics were identical between
    baseline and response candidate, as intended: in-domain 95% coverage
    `0.9442`, rare 95% coverage `0.9520`, mean OOD 95% coverage `0.9203`,
    worst OOD/effect-size-shift 95% coverage `0.8214`, and combined-shift 95%
    coverage `0.9684`. The response candidate passed all five fixed-comparison
    acceptance rows under the zero-delta reuse comparison. Predictive proper
    scores improved modestly on average: Brier `0.147620 -> 0.147153`
    (`0.9968x`), log-loss `0.446577 -> 0.443869` (`0.9938x`), predictive RMSE
    `0.384179 -> 0.383563` (`0.9984x`), prevalence MAE
    `0.043553 -> 0.042621`, and richness MAE `3.023169 -> 2.982169`.
    Four of five seeds improved Brier/log-loss/RMSE, while seed `20260718`
    worsened slightly. Keep `probit_response_affine` as a viable
    predictive-only competitor, but do not promote it as the default until it
    is checked on real-data transfer.

21. Run real-data transfer validation for the response-scale predictive-mean
    competitor.
    Status: complete; do not promote as default. Extended the Whittaker and Big Spatial real-data
    qualification workflows so they can compare promoted `external_monotone`
    against `external_monotone + probit_response_affine` while attaching the
    existing Python-only/R-boundary parity metrics. The report must keep the
    response layer labelled predictive-only and must not treat its predictive
    mean improvement as evidence for coefficient-posterior calibration. Gate
    the candidate on Whittaker and Big Spatial held-out Brier/log-loss/proper
    scores, while requiring the existing Whittaker/Big-Spatial acceptance,
    Python-only HMSC parity context, SBC/OOD, and rare-validation gates to
    remain frozen. Implementation and submission details are documented in
    `docs/neural_hmsc_response_mean_realdata_transfer_2026-07-19.md`. Local
    py_compile, sbatch syntax checks, help checks, focused tests, Whittaker
    response-mean smokes, and a Big Spatial transfer smoke passed. LUMI
    Whittaker job `20006616` completed on `dev-g` with
    `PREDICTIVE_MEAN_CALIBRATION=probit_response_affine`,
    `PREDICTIVE_MEAN_CALIBRATION_VALIDATION_DATASETS=128`, and the qualified
    Whittaker R/Python parity metrics attached. LUMI Big Spatial job `20006620`
    completed with dependency `afterok:20006616`, using the Whittaker
    response-mean run as its frozen source and the qualified Big Spatial
    R/Python parity metrics attached. Both jobs exited `0:0`. Whittaker passed
    coefficient SBC, held-out predictive, combined qualification, and reference
    parity gates, but response-mean calibrated prediction slightly worsened
    Brier/log-loss relative to scale-only predictive calibration: Brier ratio
    `1.0030`, log-loss ratio `1.0005`. It improved macro AUC, prevalence MAE,
    richness MAE, and rare-species metrics, but remained worse than the
    qualified Python MCMC comparator on core proper scores: Brier ratio
    `1.0461`, log-loss ratio `1.0454`. Big Spatial passed source, target
    predictive, transfer, and reference parity gates. There the response-mean
    layer improved scale-only transfer: Brier ratio `0.9962`, log-loss ratio
    `0.9944`, macro AUC ratio `1.0050`, prevalence MAE ratio `0.9729`, and
    richness MAE ratio `0.9823`, but it still trailed qualified Python MCMC:
    Brier ratio `1.1083`, log-loss ratio `1.0971`. The candidate remains a
    valid experimental predictive-only competitor but should not become the
    default. Next action: return to predictive-mean competitor development with
    an explicit cross-dataset no-degradation gate against the promoted
    `external_monotone` scale-only predictive path on both Whittaker and Big
    Spatial.

22. Redesign predictive-mean competitor selection with cross-dataset
    no-degradation gates.
    Status: complete. Added an executable promotion gate in
    `pyhmsc/neural/predictive_selection.py`, exposed by
    `examples/evaluate_neural_hmsc_predictive_promotion.py`, with focused tests
    in `tests/test_neural_hmsc_predictive_selection.py`. The gate compares
    `neural_predictive_mean_calibrated` against
    `neural_predictive_only_calibrated` across named held-out real-data metric
    CSVs. By default, every dataset must have Brier and log-loss ratios
    `<= 1.0`; optional simulated summary rows can also require positive
    simulated Brier/log-loss gains, but simulated improvement cannot override a
    real-data degradation. Running the gate on the completed Whittaker and Big
    Spatial response-mean validation bundle wrote
    `/private/tmp/neural_response_mean_realdata_20006616_20006620/promotion_gate`.
    It rejected `probit_response_affine` for default promotion exactly because
    Whittaker degraded: Brier ratio `1.0030`, log-loss ratio `1.0005`. Big
    Spatial passed: Brier ratio `0.9962`, log-loss ratio `0.9944`. The gate
    result is documented in
    `docs/neural_hmsc_predictive_mean_promotion_gate_2026-07-19.md`. Keep
    `probit_response_affine` available as an experimental predictive-only path,
    but do not tune it into the default.

23. Implement a domain-conditional predictive-mean selector.
    Status: implemented locally; production validation pending. Added
    `domain_conditional_predictive_mean_selector_metadata` and
    `select_predictive_mean_calibration_for_context` in
    `pyhmsc/neural/mean_calibration.py`. The Whittaker runner now accepts
    `--predictive-mean-selection-policy {apply_selected,domain_conditional}`.
    In `domain_conditional` mode, Whittaker/source-like contexts use identity
    and write final predictive samples identical to scale-only while retaining
    the response-affine candidate in `predictive_mean_selector` metadata.
    Big Spatial reads that selector metadata from the frozen Whittaker artifact
    and applies the candidate for `big_spatial_transfer` if active. The LUMI
    Whittaker sbatch wrapper exposes `PREDICTIVE_MEAN_SELECTION_POLICY`.
    Focused tests passed, and a local transfer-shape smoke confirmed the
    intended behavior: Whittaker selector-final Brier/log-loss ratios versus
    scale-only were exactly `1.0000/1.0000`, while Big Spatial selector-final
    ratios were `0.9716/0.9584`. The smoke cross-dataset promotion gate passed.
    Details are documented in
    `docs/neural_hmsc_domain_conditional_predictive_mean_selector_2026-07-19.md`.

24. Run production-shape real-data validation for the domain-conditional
    predictive-mean selector.
    Status: complete; promotion gate passed. Submitted Whittaker with
    `PREDICTIVE_MEAN_CALIBRATION=probit_response_affine`,
    `PREDICTIVE_MEAN_SELECTION_POLICY=domain_conditional`, the same validation
    dataset count (`128`) and parity metrics as the previous response-mean
    real-data run. Whittaker LUMI job `20008206` ran on `dev-g`; run
    root:
    `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_whittaker_domain_selector_realdata_20260719`.
    Submitted dependent Big Spatial transfer job `20008208` with
    `afterok:20008206`, using that frozen Whittaker artifact and the Big
    Spatial parity metrics; run root:
    `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_big_spatial_domain_selector_realdata_20260719`.
    Remote syntax/import verification passed under the TensorFlow venv before
    submission. Both jobs completed successfully: Whittaker `20008206` elapsed
    `00:10:28`, Big Spatial `20008208` elapsed `00:01:50`, both exit `0:0`.
    Downloaded results to
    `/private/tmp/neural_domain_selector_realdata_20008206_20008208` and ran
    the frozen cross-dataset promotion gate from step 22. The gate passed.
    Whittaker selected identity/source fallback and was exactly unchanged
    relative to scale-only: Brier ratio `1.0000`, log-loss ratio `1.0000`.
    Big Spatial selected the response-affine candidate and retained the
    transfer gain: Brier ratio `0.9962`, log-loss ratio `0.9944`, macro AUC
    ratio `1.0050`, prevalence MAE ratio `0.9729`, and richness MAE ratio
    `0.9823`. Whittaker and Big Spatial acceptance gates and reference parity
    attachments passed. Qualified Python MCMC remains stronger on core
    real-data proper scores, so this is a neural scale-only predictive-transfer
    improvement, not an HMSC-comparator superiority claim. Detailed results are
    documented in
    `docs/neural_hmsc_domain_conditional_predictive_mean_selector_2026-07-19.md`.

25. Decide promotion policy for the domain-conditional predictive-mean
    selector.
    Status: complete; require bounded sensitivity confirmation before default
    promotion. The selector has passed the two-dataset production-shape
    real-data gate, but only on one paired seed. Because the earlier raw
    `probit_response_affine` evidence was mixed across Whittaker and Big
    Spatial, do not change defaults from one paired context-rule result. Keep
    `PREDICTIVE_MEAN_SELECTION_POLICY=apply_selected` as the sbatch default for
    now. Run a bounded three-seed Whittaker plus dependent Big Spatial
    sensitivity confirmation with
    `PREDICTIVE_MEAN_SELECTION_POLICY=domain_conditional`,
    `PREDICTIVE_MEAN_CALIBRATION=probit_response_affine`,
    `PREDICTIVE_MEAN_CALIBRATION_VALIDATION_DATASETS=128`, and the same
    Whittaker/Big Spatial parity attachments. Every seed pair must pass the
    existing real-data gates and the frozen cross-dataset no-degradation gate.
    If it passes, promote the domain-conditional selector policy, not raw
    `probit_response_affine`, as the default predictive-mean deployment
    policy. Decision details are recorded in
    `docs/neural_hmsc_domain_conditional_predictive_mean_selector_2026-07-19.md`.

26. Submit bounded three-seed real-data sensitivity confirmation for the
    domain-conditional selector.
    Status: submitted/running on LUMI. Reused the existing bounded
    real-data sensitivity harness and extended it to run
    `PREDICTIVE_MEAN_CALIBRATION=probit_response_affine`,
    `PREDICTIVE_MEAN_SELECTION_POLICY=domain_conditional`, per-seed
    Whittaker plus dependent Big Spatial transfer, and the frozen
    cross-dataset no-degradation promotion gate. The aggregator now records
    selector decisions, per-dataset promotion ratios, per-seed gate pass
    counts, and a final promotion-candidate decision without tuning
    thresholds. Local and remote checks passed:
    `bash -n`, `py_compile`, and focused pytest for
    `tests/test_neural_hmsc_realdata_sensitivity.py` plus
    `tests/test_neural_hmsc_predictive_selection.py`. Submitted job
    `20010991` to `dev-g` with run root
    `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_domain_selector_realdata_sensitivity_20260719`
    and seeds `20260721 20260722 20260723`. Initial state was running on
    `nid007962`. Required output remains: per-seed Whittaker and Big Spatial
    held-out metrics, acceptance JSON, selector decisions, promotion gate
    JSON/CSV/Markdown, and aggregate `realdata_sensitivity.{csv,json,md}`.

27. Monitor and aggregate bounded domain-conditional real-data sensitivity.
    Status: complete; do not promote `domain_conditional` as default. LUMI job
    `20010991` completed successfully in `00:34:09` with exit code `0:0`.
    Downloaded the run root to
    `/private/tmp/neural_domain_selector_realdata_sensitivity_20010991/neural_domain_selector_realdata_sensitivity_20260719`.
    Inspected `realdata_sensitivity.{csv,json,md}` and all per-seed
    `promotion_gate/` outputs. All six Whittaker and Big Spatial dataset
    acceptance gates passed, and Whittaker used identity/source fallback in
    all three seeds. The frozen cross-dataset predictive-mean promotion gate
    passed for seeds `20260721` and `20260722`, but failed for seed
    `20260723` because Big Spatial degraded relative to the scale-only
    baseline: Brier ratio `1.004801`, log-loss ratio `1.004174`. Paired
    promotion-gate pass count was therefore `2 / 3`, below the every-seed
    no-degradation rule. Keep promoted scale-only `external_monotone` as the
    default predictive path and keep `domain_conditional` experimental. Full
    result details are recorded in
    `docs/neural_hmsc_domain_conditional_predictive_mean_selector_2026-07-19.md`.

28. Add a conservative transfer-stability guard for predictive-mean
    calibration.
    Status: complete; implementation added. The bounded sensitivity failure shows source validation
    proper-score gains are too small to justify unconditional transfer-side
    response-affine movement: every seed selected a candidate, but one Big
    Spatial transfer seed degraded. Implement a stricter selector guard that
    keeps identity unless response-affine validation gains exceed a practical
    margin and candidate movement stays within conservative amplitude
    constraints. The first design should be simple and auditable: expose
    configurable minimum Brier/log-loss gain margins and optional intercept or
    slope caps in the selector metadata, preserve Whittaker identity fallback,
    and only apply the candidate to transfer-like contexts when the guard
    passes. Validate locally/fixed-sim first; only rerun the bounded
    three-seed real-data sensitivity if those gates still hold.
    The implementation is in `pyhmsc/neural/mean_calibration.py` and stores
    `transfer_stability_guard` metadata with thresholds, measured validation
    gains, candidate movement, pass/fail status, and failure reasons. Default
    guard thresholds are Brier gain `>= 0.0001`, log-loss gain `>= 0.0005`,
    slope delta `<= 0.05`, and absolute intercept `<= 0.025`. The Whittaker
    runner exposes these as `--predictive-mean-transfer-*` flags, and both
    Whittaker plus real-data sensitivity LUMI sbatch wrappers expose matching
    environment variables. Focused validation passed:
    `py_compile`, `bash -n`, `tests/test_neural_hmsc_mean_calibration.py`,
    `tests/test_neural_hmsc_realdata_sensitivity.py`, and
    `tests/test_neural_hmsc_predictive_selection.py`. Details are recorded in
    `docs/neural_hmsc_domain_conditional_predictive_mean_selector_2026-07-19.md`.

29. Run a fixed/local transfer-stability guard sanity check.
    Status: complete. Before submitting another LUMI real-data sensitivity run,
    run a small local or fixed-evaluation workflow that exercises
    `PREDICTIVE_MEAN_SELECTION_POLICY=domain_conditional` with the new guard.
    Confirm that near-zero source validation gains produce transfer identity
    decisions, that strong synthetic validation gains can still apply the
    candidate, and that coefficient-posterior calibration, SBC/OOD semantics,
    Whittaker source identity, and scale-only predictive outputs remain
    unchanged. If this passes, rerun the bounded three-seed real-data
    sensitivity with the guarded selector.
    The local fixed replay used the downloaded LUMI job `20010991` artifacts in
    `/private/tmp/neural_domain_selector_realdata_sensitivity_20010991/neural_domain_selector_realdata_sensitivity_20260719`.
    It reconstructed guarded selector metadata from each seed's saved
    Whittaker response-affine candidate. All three prior source-selected
    candidates were withheld from Big Spatial transfer because validation gains
    were below the new practical margins; seed `20260723` was also blocked by
    the intercept cap. Whittaker source decisions remained identity in all
    three seeds, and guarded transfer fallback would produce Brier/log-loss
    ratios of `1.0000/1.0000` for each seed instead of the previous
    `20260723` degradation. A strong synthetic candidate with Brier/log-loss
    gains of `0.0010/0.0010`, slope `1.025`, and intercept `0.020` still
    selected `apply_candidate`. Details are recorded in
    `docs/neural_hmsc_domain_conditional_predictive_mean_selector_2026-07-19.md`.

30. Tighten guarded-selector promotion reporting before another LUMI
    sensitivity submission.
    Status: complete. The guard fixes no-degradation mechanically, but the fixed
    replay shows the prior real-data candidates would all fall back to
    identity. Before rerunning the bounded three-seed LUMI sensitivity, update
    the sensitivity aggregator or promotion report to separate two outcomes:
    safe identity fallback versus genuine transfer improvement. The promotion
    decision should require the existing every-seed no-degradation gate plus a
    practical nonzero transfer-improvement signal, for example at least two of
    three Big Spatial transfer seeds applying the candidate with Brier and
    log-loss ratios below `1.0`. Without this distinction, a guarded selector
    could appear promotable only because it behaves identically to the current
    scale-only default.
    Implemented in `examples/aggregate_neural_hmsc_realdata_sensitivity.py`.
    The aggregate rows now include `predictive_mean_transfer_outcome`,
    `predictive_mean_genuine_transfer_improvement`,
    `predictive_mean_safe_identity_fallback`, and direct predictive-mean versus
    scale-only Brier/log-loss ratios. The summary now reports paired genuine
    transfer-improvement and safe identity-fallback counts. The promotion
    decision requires all paired dataset gates, all paired no-degradation gates,
    and at least two Big Spatial transfer seeds with genuine transfer
    improvement; an all-identity guarded run is classified as
    `safe_identity_fallback_not_promotable`. Focused validation passed:
    `py_compile`, `bash -n`, and
    `tests/test_neural_hmsc_realdata_sensitivity.py` plus
    `tests/test_neural_hmsc_predictive_selection.py`. Replaying the tightened
    aggregator on downloaded LUMI job `20010991` produced
    `inspect_seed_level_no_degradation`, with two genuine Big Spatial transfer
    improvements and one applied degradation, matching the prior failure.
    Details are recorded in
    `docs/neural_hmsc_domain_conditional_predictive_mean_selector_2026-07-19.md`.

31. Submit guarded bounded three-seed real-data sensitivity with tightened
    reporting.
    Status: complete. Sync the guarded selector implementation, tightened
    aggregator, updated Whittaker runner, and LUMI sbatch wrappers to LUMI.
    Run the same bounded Whittaker plus dependent Big Spatial sensitivity
    workflow with `PREDICTIVE_MEAN_SELECTION_POLICY=domain_conditional` and the
    default transfer-stability guard. Required decision outputs:
    per-seed selector action, transfer outcome, no-degradation gate, genuine
    transfer-improvement count, safe identity-fallback count, and final
    aggregate decision. Promotion should occur only if the run clears both
    no-degradation and genuine-transfer-improvement requirements; identity-only
    fallback should remain non-promotable.
    Local validation passed (`py_compile`, `bash -n`, focused pytest, and
    `git diff --check`), then remote LUMI validation passed under the
    TensorFlow venv with the same focused checks. Submitted LUMI job
    `20020582` on `dev-g`, initial state running on `nid007956`, run root
    `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_domain_selector_guarded_realdata_sensitivity_20260720`,
    seeds `20260721 20260722 20260723`, predictive mean calibration
    `probit_response_affine`, selection policy `domain_conditional`, and
    default transfer-stability guard. Submission details are recorded in
    `docs/neural_hmsc_domain_conditional_predictive_mean_selector_2026-07-19.md`.

32. Monitor and aggregate guarded bounded real-data sensitivity.
    Status: complete; do not promote guarded `domain_conditional`. Monitored
    LUMI job `20020582`, which completed in `00:31:41` with exit code `0:0`.
    Downloaded the result root to
    `/private/tmp/neural_domain_selector_guarded_realdata_sensitivity_20020582/neural_domain_selector_guarded_realdata_sensitivity_20260720`.
    Inspected `realdata_sensitivity.{csv,json,md}` and all per-seed
    `promotion_gate/` outputs. All six dataset acceptance gates passed, and
    all three paired no-degradation promotion gates passed. However, the
    tightened reporting classified all three Big Spatial transfer rows as
    `safe_identity_fallback`: paired genuine transfer-improvement count was
    `0`, paired safe identity-fallback count was `3`, and the aggregate
    decision was `safe_identity_fallback_not_promotable`. The guard failure
    reasons matched the local replay: all three candidates had validation Brier
    and log-loss gains below the practical margins; seed `20260723` also
    exceeded the intercept cap. Keep `external_monotone` scale-only as the
    default predictive path and keep `domain_conditional` experimental. Details
    are recorded in
    `docs/neural_hmsc_domain_conditional_predictive_mean_selector_2026-07-19.md`.

33. Redesign the predictive-mean candidate around transfer-aware validation.
    Status: complete; implementation added. The guarded selector family now has correct no-degradation
    behavior, but it does not create a competing solution because it falls back
    to identity on every real-data transfer seed. Stop tuning the current
    source-only `probit_response_affine` selector. Implement a new
    predictive-only response-mean candidate that is selected against
    transfer-like or multi-domain validation rather than Whittaker-source
    validation alone. A suitable first design is a conservative
    transfer-aware response calibrator trained on simulated source plus
    transfer-like OOD validation batches, with selection requiring
    no-degradation on source-like contexts and a practical Brier/log-loss gain
    on transfer-like contexts. The candidate must keep coefficient posterior
    calibration, SBC/OOD/rare gates, and Python-only HMSC parity semantics
    unchanged. Validate on compact/fixed simulations first; only submit another
    real-data LUMI sensitivity run if the candidate clears both no-degradation
    and genuine-transfer-improvement reporting gates.
    Implemented `probit_transfer_response_affine`, a predictive-only
    response-mean calibrator selected with separate source-like and
    transfer-like response validation. `BetaResponseCalibrationBatch` carries
    transfer-like validation batches, and
    `fit_beta_transfer_response_mean_calibration` records source
    `response_validation` separately from `transfer_response_validation` in
    `BetaPredictiveMeanCalibration` metadata. The transfer-stability guard now
    uses transfer-validation gains when available. The simulated benchmark
    runner exposes the new candidate via
    `--predictive-mean-calibration probit_transfer_response_affine`, generating
    transfer-like validation batches from the configured OOD regimes. Focused
    validation passed (`py_compile`, `tests/test_neural_hmsc_mean_calibration.py`,
    CLI help), and a tiny local probit smoke at
    `/private/tmp/neural_transfer_mean_candidate_smoke` selected the new method
    with source Brier/log-loss ratios `0.9736/0.9714` and transfer ratios
    `0.9214/0.9306` while keeping the artifact role predictive-only.
    Details are recorded in
    `docs/neural_hmsc_domain_conditional_predictive_mean_selector_2026-07-19.md`.

34. Run compact fixed-evaluation comparison for the transfer-aware
    predictive-mean candidate.
    Status: complete; candidate remains experimental. Compare promoted
    `external_monotone` scale-only against
    `external_monotone + probit_transfer_response_affine` on shared compact
    simulated probit seeds, with fixed SBC/OOD evaluation rows and no real-data
    submission. Required outputs: coefficient calibration/SBC rows unchanged
    relative to scale-only, predictive-only response metadata with separate
    source and transfer validation scores, transfer-like predictive Brier and
    log-loss ratios, and a decision on whether the candidate shows enough
    no-degradation plus genuine transfer improvement to justify a bounded LUMI
    five-seed or real-data sensitivity check.
    Ran the candidate at
    `/private/tmp/neural_transfer_mean_fixed_eval_20260720/external_monotone_transfer_response`
    against the promoted compact baseline
    `/private/tmp/neural_mean_fixed_eval_20260719/external_monotone`, then
    wrote fixed-evaluation and predictive-score comparisons under
    `/private/tmp/neural_transfer_mean_fixed_eval_20260720/{comparison,predictive_scores}`.
    The predictive-only candidate selected `slope = 1.15` and
    `intercept = 0.0`, with source validation Brier/log-loss ratios
    `0.9924/0.9894` and transfer validation ratios `0.9874/0.9807` across
    `covariate_shift`, `effect_size_shift`, and `combined_shift` batches.
    Fixed coefficient/SBC/OOD rows were identical for both arms:
    in-domain coverage `0.8750`, mean OOD `0.7731`, effect-size shift
    `0.7222`, combined shift `0.7361`, and rare coverage `1.0000`.
    Both arms therefore failed the compact fixed-evaluation acceptance gate on
    in-domain coverage, so the result is not promotable and should not trigger
    real-data LUMI sensitivity. Predictive scores improved versus scale-only
    on Brier (`0.9849` ratio), log loss (`0.9784`), predictive RMSE
    (`0.9924`), and prevalence MAE (`0.7302`), while richness MAE worsened
    slightly (`1.0194`). Details are recorded in
    `docs/neural_hmsc_domain_conditional_predictive_mean_selector_2026-07-19.md`.

35. Run a larger fixed-evaluation simulated confirmation for the transfer-aware
    predictive-mean candidate.
    Status: complete; not promotable on this compact checkpoint. Re-run
    `external_monotone` scale-only versus
    `external_monotone + probit_transfer_response_affine` on shared simulated
    probit seeds with enough SBC/OOD rows to re-qualify the in-domain gate
    rather than inheriting the compact bundle failure. Keep coefficient
    calibration/SBC/OOD/rare gates frozen and treat the response-affine layer
    as predictive-only. Promotion criteria for this step: no coefficient-gate
    degradation relative to scale-only, in-domain coverage within the frozen
    acceptance interval for both arms, nonzero predictive improvement on Brier
    and log loss, no material predictive RMSE or richness MAE degradation, and
    retained transfer-validation metadata. Only if this larger simulated
    confirmation passes should we consider a bounded five-seed LUMI or
    real-data sensitivity run.
    Ran the larger local confirmation under
    `/private/tmp/neural_transfer_mean_larger_eval_20260720`, increasing fixed
    SBC/OOD evaluation from `8 x 64` to `24 x 128` while reusing the same
    frozen compact checkpoint and seed schedule. The transfer-aware candidate
    again selected `slope = 1.15` and `intercept = 0.0`, with source
    validation Brier/log-loss ratios `0.9924/0.9894` and transfer validation
    ratios `0.9874/0.9807`. Coefficient/SBC/OOD rows were identical between
    scale-only and the predictive candidate, but the shared fixed gate still
    failed: in-domain coverage was `0.8519`, mean OOD `0.7701`, worst
    OOD/effect-size shift `0.6898`, combined shift `0.7130`, and rare coverage
    `1.0000`. Predictive scores again improved versus scale-only on Brier
    (`0.9841` ratio), log loss (`0.9771`), predictive RMSE (`0.9920`), and
    prevalence MAE (`0.7290`), while richness MAE worsened slightly (`1.0200`).
    The result confirms a repeatable predictive-only gain, but it also confirms
    that this compact checkpoint/coefficient calibration setup is
    underqualified. Do not submit real-data sensitivity from this local
    checkpoint. Details are recorded in
    `docs/neural_hmsc_domain_conditional_predictive_mean_selector_2026-07-19.md`.

36. Evaluate the transfer-aware predictive-mean candidate on a qualified
    production-shape external-monotone baseline.
    Status: complete; qualified for real-data transfer validation. Reused the
    five-seed production-shape
    `external_monotone` fixed-evaluation baseline and ran only
    `external_monotone + probit_transfer_response_affine` on the same seed
    schedule, scalar checkpoints, external-monotone settings, SBC/OOD rows, and
    OOD regimes. Compare candidate coefficient/SBC/OOD rows against the frozen
    qualified baseline and compute predictive Brier, log-loss, predictive RMSE,
    prevalence MAE, and richness MAE against scale-only. Required decision:
    the response-affine candidate must preserve the qualified coefficient gates,
    improve Brier/log-loss on average, avoid material RMSE/richness
    degradation, and keep transfer-validation metadata attached. If it passes,
    then consider a bounded five-seed LUMI or real-data sensitivity check; if it
    fails, keep `probit_transfer_response_affine` experimental and redesign
    the predictive-mean candidate rather than tuning this compact checkpoint.
    LUMI job `20023454` completed on `dev-g` using seeds `20260716` through
    `20260720`, shape `40 x 75`, `32` SBC datasets, `256` SBC draws, and the
    frozen qualified production baseline. The candidate passed the fixed
    comparison in all five seeds and preserved identical in-domain (`0.9442`),
    rare (`0.9520`), mean OOD (`0.9203`), effect-size (`0.8214`), and combined
    shift (`0.9684`) coverage. Mean predictive ratios versus scale-only were
    `0.9957` Brier, `0.9928` log-loss, and `0.9979` RMSE; prevalence MAE
    improved from `0.04355` to `0.04237` and richness MAE from `3.0232` to
    `2.9889`. Brier and RMSE worsened marginally in one seed, while log-loss
    improved in all five. The candidate selected in all seeds and retained
    source and three-regime transfer-validation metadata. Details are recorded
    in `docs/neural_hmsc_transfer_response_mean_production_eval_2026-07-20.md`.

37. Validate the qualified transfer-aware predictive mean on both real-data
    domains.
    Status: complete; rejected for global promotion. Extended the
    Whittaker/Big Spatial real-data evaluation path to
    accept `probit_transfer_response_affine`, then compare it against promoted
    scale-only `external_monotone` on Whittaker and dependent Big Spatial runs.
    Keep Python-only HMSC parity metrics attached, label the affine layer
    predictive-only, and apply a frozen cross-dataset decision rule: no material
    Brier/log-loss degradation on either dataset, no material RMSE/richness
    degradation, and a genuine improvement rather than identity fallback. If
    the paired run passes, submit a bounded three-seed real-data sensitivity
    confirmation before considering promotion; otherwise keep the method as a
    simulated-only experimental competitor.
    LUMI job `20026496` completed the paired seed `20260721` workflow with
    qualified Whittaker and Big Spatial R/Python parity metrics attached. The
    candidate selected slope `1.025` and intercept `0.025`; independent source
    validation ratios were `0.9997/0.9992` for Brier/log-loss and transfer
    validation ratios were `0.9984/0.9980`. Big Spatial improved on Brier
    (`0.9969` ratio), log-loss (`0.9948`), RMSE (`0.9984`), and richness MAE
    (`0.9845`). Whittaker degraded on Brier (`1.0049`), log-loss (`1.0015`),
    and RMSE (`1.0024`) while richness improved (`0.9915`). The frozen
    simulated gate passed, but the strict cross-dataset gate failed and the
    mean real-data Brier gain was negative. Do not launch a three-seed run.
    Details are recorded in
    `docs/neural_hmsc_transfer_response_realdata_pair_2026-07-20.md`.

38. Split transfer-aware predictive mean calibration into independently gated
    source and transfer branches.
    Status: complete; passed the one-seed production replay. Replaced the
    globally applied affine candidate with a
    two-branch predictive-only calibration artifact. Fit and select the source
    branch only on independent source-shaped simulations, requiring a material
    validation margin before any nonidentity movement. Fit and select the
    transfer branch on balanced covariate-shift, effect-size-shift, and
    combined-shift simulations. Deployment chooses the branch from an explicit
    predeclared context (`whittaker` source, `big_spatial_transfer` transfer),
    never from real held-out outcomes. Preserve the qualified coefficient/SBC,
    parity, simulated production, and cross-dataset no-degradation gates. First
    run a compact serialization/context-selection test and replay the paired
    real-data workflow for one seed; only a passing pair may advance to the
    bounded three-seed sensitivity confirmation.
    Implemented `probit_source_transfer_response_affine` with a source branch
    fitted/selected on independent shape-matched simulations and a transfer
    branch fitted/selected on separate balanced covariate-shift,
    effect-size-shift, and combined-shift pools. The serialized selector maps
    `whittaker` to the source branch and `big_spatial_transfer` to the transfer
    branch. LUMI job `20029081` replayed seed `20260721` with all parity,
    coefficient/SBC, simulated-production, and real-data gates frozen. The
    source branch missed its `0.0005` margin and used identity, reproducing all
    Whittaker scale-only metrics exactly. The transfer branch selected slope
    `1.05` and intercept `0.05`; Big Spatial ratios were `0.9954` Brier,
    `0.9911` log-loss, `0.9977` RMSE, and `0.9706` richness MAE. Both dataset
    gates and the complete cross-dataset gate passed. Details are recorded in
    `docs/neural_hmsc_source_transfer_branch_realdata_pair_2026-07-20.md`.

39. Run bounded three-seed sensitivity for independent source/transfer branch
    selection.
    Status: complete; failed the frozen promotion gate. LUMI job `20029856`
    ran seeds `20260721`, `20260722`, and `20260723` with
    `PREDICTIVE_MEAN_CALIBRATION=probit_source_transfer_response_affine`, source
    margin `0.0005`, transfer margin `0.0001`, and the same parity references,
    simulated production summary, coefficient/SBC settings, and strict
    cross-dataset gates. All source branches selected identity and Whittaker
    reproduced scale-only metrics exactly. All transfer branches were applied.
    Seeds `20260721` and `20260722` improved all four Big Spatial metrics, but
    seed `20260723` degraded Brier (`1.0048` ratio), log loss (`1.0042`), RMSE
    (`1.0024`), and richness MAE (`1.0147`). The run achieved the required two
    genuine improvements but only two of three cross-dataset gate passes, so
    the aggregate decision was `inspect_seed_level_no_degradation` and the
    method remains experimental. The failed seed had the strongest independent
    simulated validation ratios, showing target-context mismatch rather than a
    weak scalar margin. Details are recorded in
    `docs/neural_hmsc_source_transfer_branch_realdata_sensitivity_2026-07-20.md`.

40. Add a target-context-conditioned independent simulation gate for transfer
    branch application.
    Status: complete; implemented but rejected by the frozen replay. Use the target dataset's unlabeled covariates, study design,
    community size, and prevalence context to construct separate synthetic
    calibration and validation responses. Preserve the existing generic OOD
    gate and require agreement from both gates before applying a nonidentity
    transfer branch. Never use target held-out responses during fitting or
    selection. First replay the resulting selector against the three frozen
    checkpoints from job `20029856`; only a three-of-three no-degradation result
    with at least two genuine Big Spatial improvements may return to LUMI
    training or promotion.
    The implementation adds a reusable target-context response gate, a
    dual-gated source/transfer selector, Big Spatial target-support simulation
    wiring, diagnostics, scheduler controls, and a no-training frozen replay
    harness. Held-out target `Y` is loaded only after selection. LUMI job
    `20031969` evaluated `32` independent datasets per target calibration and
    validation pool against all three frozen checkpoints. Every target gate
    passed. Seeds `20260721` and `20260722` retained their real Big Spatial
    gains, but seed `20260723` also passed both synthetic pools and retained its
    real degradation. The replay therefore achieved only `2/3` no-degradation
    passes and decided `target_context_gate_failed_no_degradation`. This shows
    simulator-to-ecology misspecification rather than insufficient synthetic
    sample size. Do not tune or promote this gate family. Details are recorded
    in `docs/neural_hmsc_target_context_gate_replay_2026-07-20.md`.

41. Evaluate a probability-level deep-ensemble predictive-mean competitor.
    Status: complete; passed the frozen promotion gate. Stop trying to infer checkpoint-specific real transfer
    direction from the current simulator. Average predictive probabilities
    across the three frozen checkpoints, comparing the scale-only ensemble to
    the affine-branch ensemble on Whittaker and Big Spatial. Repeat the frozen
    comparison for all three leave-one-seed-out ensembles to expose dependence
    on any checkpoint. Keep target outcomes unavailable until final scoring and
    retain the existing coefficient/SBC/parity provenance. Advance only if the
    full ensemble and every leave-one-out ensemble preserve Brier, log-loss,
    RMSE, and richness-MAE no degradation, with genuine Big Spatial Brier and
    log-loss improvement for the full ensemble; otherwise abandon this affine
    branch family rather than add another selector gate.
    Implemented a frozen evaluator that constructs scale-only and affine
    probability ensembles over identical members, scores the full three-seed
    set and every leave-one-out pair on Whittaker and Big Spatial, verifies
    source parity/acceptance provenance, and keeps outcomes unavailable until
    all member predictions are frozen. Focused validation passed (`25` tests,
    Python compilation, shell syntax, and `git diff --check`). LUMI job
    `20032201` completed in `35` seconds. All eight full/leave-one-out dataset
    rows passed Brier, log-loss, RMSE, and richness-MAE no degradation.
    Whittaker remained exactly identity. The full Big Spatial ensemble ratios
    were `0.9975` Brier, `0.9949` log loss, `0.9988` RMSE, and `0.9857`
    richness MAE; all three leave-one-out Big Spatial ensembles also improved
    all four metrics. Provenance passed and the decision was
    `probability_ensemble_promotion_candidate`. Details are recorded in
    `docs/neural_hmsc_probability_ensemble_2026-07-20.md`.

42. Package the probability ensemble as a reusable predictive deployment
    artifact and API.
    Status: complete. Added an ordered ensemble manifest containing member artifact
    paths and hashes, seeds, predictive calibration roles, species/formula
    compatibility, and qualified parity provenance. Provide a `predict_mean`
    API that averages member response probabilities and rejects incompatible
    members. Integrate the artifact into Whittaker/Big Spatial reporting, then
    run one clean frozen requalification against both the matched scale-only
    ensemble and qualified Python MCMC. Keep the ensemble explicitly
    predictive-only and do not change the default deployment policy until this
    API/provenance requalification passes. `PredictiveProbabilityEnsemble` is
    now available from `pyhmsc`, supports ordered subsets, and validates neural
    member, acceptance, run-metadata, parity-metrics, and MCMC-reference hashes.
    LUMI job `20032745` completed in `79` seconds with decision
    `predictive_ensemble_api_requalification_passed`. All full and leave-one-out
    neural no-degradation gates reproduced; full Big Spatial affine-versus-scale
    ratios were `0.9975` Brier and `0.9949` log loss, while Whittaker remained
    identity. Qualified Python MCMC remained stronger: affine-versus-MCMC ratios
    were `1.0213`/`1.0327` Brier/log loss on Whittaker and
    `1.0790`/`1.0731` on Big Spatial. This is a neural predictive deployment
    qualification, not an HMSC parity claim. Details are recorded in
    `docs/neural_hmsc_probability_ensemble_api_requalification_2026-07-20.md`.

43. Promote the manifest-backed affine probability ensemble within the neural
    predictive deployment path.
    Status: complete. Made the qualified `affine_branch` manifest the default
    neural predictive-mean policy, retained the matched `scale_only` manifest
    as an explicit fallback, and kept qualified Python MCMC as the statistical
    reference. Wired deployment and scheduler entry points to load manifests
    through `PredictiveProbabilityEnsemble.from_manifest`, add one clean default
    wiring smoke, and state explicitly that this policy does not replace or
    claim equivalence to Python-only/R-boundary HMSC inference. The public
    `load_predictive_mean_ensemble` loader now applies the promoted default and
    rejects manifests without compatible members, predictive-only semantics,
    outcome-independent selection, qualified parity files, or an ordered MCMC
    reference matching the neural seeds. `scale_only` requires an explicit
    policy selection. LUMI `dev-g` smoke job `20032978` completed in `34`
    seconds with decision `predictive_deployment_smoke_passed`. Whittaker
    default/fallback predictions were identical; Big Spatial had maximum
    response-probability movement `0.040348`. Both datasets validated parity
    provenance, target outcomes were not opened, and MCMC was not used for
    neural prediction. Details are in
    `docs/neural_hmsc_predictive_deployment_promotion_2026-07-20.md`.

44. Freeze the promoted ensemble as the neural deployment baseline and resume
    competitor development against the remaining proper-score gap.
    Status: complete. Gave the promoted manifest bundle the stable versioned
    baseline identifier `neural_predictive_affine_v1`, preserved the qualified
    `scale_only` fallback and Python MCMC comparator, and required future
    simulated and real-data competitors to report deltas against this exact
    ensemble. The atomic freeze rejects an existing destination and pins all
    four deployment manifests plus API requalification and scheduler-smoke
    evidence by SHA-256. The public `load_predictive_deployment_baseline` API
    resolves the stable ID and revalidates its evidence and manifest hashes
    before loading either ensemble. LUMI job `20033698` completed in `33`
    seconds and its stable-ID Whittaker/Big Spatial smoke passed without opening
    target outcomes or using MCMC for neural prediction. The bundle embeds seven
    frozen competitor gates and the exact full-ensemble MCMC gaps, including
    Big Spatial `1.0790` Brier and `1.0731` log loss. Details are recorded in
    `docs/neural_hmsc_predictive_baseline_v1_2026-07-20.md`.

45. Implement a simulation-trained MCMC-teacher residual competitor against
    `neural_predictive_affine_v1`.
    Status: complete; compact gate failed safely. Built an offline
    response-scale teacher corpus from independent simulated communities with
    qualified Python MCMC predictive probabilities.
    Train a bounded neural logit-residual head on top of the frozen affine
    ensemble using site, species, design-information, prevalence, and community
    context, with identity regularization and no real-data outcomes. Keep the
    coefficient posterior and scale calibration untouched. First run a compact
    shared-seed fixed evaluation against the exact versioned baseline; advance
    only if simulated Brier/log loss improve, all SBC/OOD/rare gates remain
    unchanged, Whittaker is no worse, and the independent Big-Spatial-shaped
    simulation gate improves before any real-data or five-seed LUMI run.
    The retained run verified all three ordered Big Spatial transfer-affine
    member artifact hashes against the frozen manifest, fitted each neural/MCMC
    posterior on 40 simulated sites, and scored 20 disjoint holdout sites and
    75 species across five regimes. It used independent training seed
    `20260731`, validation seed `20260732`, and evaluation seeds `20260733`-
    `20260735`. No nonzero shrinkage passed the strict validation gate;
    shrinkage `0.25` missed only because covariate-shift Brier was `1.000145` of
    baseline. The raw head improved aggregate Big-Spatial-shaped Brier/log loss
    to `0.9835`/`0.9857`, with target improvement in all three evaluation
    seeds, but degraded covariate-shift Brier in two seeds and rare-validation
    scores in one seed. The selector therefore chose identity. Real-data and
    five-seed LUMI evaluation remain blocked. Details are recorded in
    `docs/neural_hmsc_mcmc_teacher_residual_compact_2026-07-20.md`.

46. Stabilize the MCMC-teacher signal with cross-fitted simulated evidence.
    Status: complete; compact gate passed. Built a larger teacher corpus with
    multiple independent
    training and validation communities per regime and out-of-fold MCMC
    predictive targets. Measure residual direction and proper-score stability
    by fold before fitting the shared head. Add a regime/context-conditioned
    identity expert and allow nonzero movement only when every held-out fold
    preserves outcome Brier/log loss and the target-shaped fold improves.
    Rerun the compact fixed evaluation against
    `neural_predictive_affine_v1`; do not open real-data outcomes or submit a
    five-seed LUMI comparison unless the nonzero cross-fitted head passes all
    frozen gates.
    The retained exact-ensemble run used four leave-one-community-out
    calibration communities and three untouched evaluation communities across
    five regimes, 35 datasets, and 52,500 held-out response probabilities. A
    nearest-prototype identity expert with label-specific support caps selected
    shrinkage `0.5`, effect cap `3.0`, and Big-Spatial-shaped cap `2.5`.
    In-domain, covariate-shift, and rare contexts remained identity. Independent
    effect-size-shift Brier/log-loss ratios were `0.9917`/`0.9926`; target-shaped
    ratios were `0.9861`/`0.9878`, with improvement in every evaluation seed.
    All-regime and all-seed no-degradation gates passed. No real outcomes were
    opened and no LUMI job was submitted. Details are recorded in
    `docs/neural_hmsc_mcmc_teacher_crossfit_context_compact_2026-07-21.md`.

47. Verify outcome-blind real-context routing before real-data scoring.
    Status: complete; failed closed. Added a covariate-only harness that
    verified the exact six frozen member hashes, computed ensemble
    probabilities and context diagnostics, and did not open target responses,
    MCMC predictions, or scoring inputs. Whittaker correctly selected the rare
    fallback and remained numerical identity (`1.11e-16` maximum movement).
    Big Spatial also remained identity: its nearest approved effect context was
    at distance `11.9382`, outside the `3.0` cap, and its nearest fallback was
    in distribution at `10.7183`. The dominant mismatch was normalized mean log
    design information (`10.12` for Big Spatial versus `0.02` at the approved
    effect prototype), caused by applying a corpus fitted only on 20-site
    holdouts to 360 sites. Probability/prevalence summaries also remained
    outside approved target support. Paired real-data scoring remains blocked.
    Details are recorded in
    `docs/neural_hmsc_mcmc_teacher_real_context_routing_2026-07-21.md`.

48. Rebuild the teacher representation and simulation corpus for real-context
    support.
    Status: complete; both required gates failed closed. Added a versioned v3
    representation using per-site Bernoulli information and bounded sample-size
    context while retaining version 2 artifact compatibility. Rebuilt the
    cross-fitted corpus with 12-, 20-, and 360-site profiles, low-prevalence
    target/effect simulations, shared MCMC fits across nested holdouts, and
    balanced per-batch residual targets. Aggregate effect and target proper
    scores improved; both compact and 360-site target profiles improved, while
    in-domain, covariate-shift, and rare profiles stayed identity. The compact
    gate nevertheless failed because effect-shift evaluation seed `20260741`
    degraded Brier/log loss to `1.001300/1.001508`. Outcome-blind routing then
    kept Whittaker on identity but also routed Big Spatial to the rare fallback:
    approved target distance `4.1895` exceeded its `2.5` cap and the rare
    fallback distance was `3.6174`. The remaining support mismatch is
    prevalence rather than raw sample size: frozen Big Spatial mean probability
    is `0.1002`, versus about `0.185-0.210` in the new target simulations. No
    ecological outcomes were opened. Details are recorded in
    `docs/neural_hmsc_mcmc_teacher_sample_size_v3_2026-07-21.md`.

## Active Roadmap Reset: Deliver A Supported Neural-HMSC

Steps 1-48 above are retained as implementation and experiment history. They
established a working fixed-effect neural posterior path, qualified Python-MCMC
references, and a frozen predictive ensemble, but later work over-concentrated
on post-hoc calibration, selector, and teacher/router variants. The active
roadmap below supersedes the previous continuation into another teacher-head
support experiment.

The simulation-trained MCMC-teacher residual family is paused. Versions 2 and 3
remain reproducible negative/diagnostic results, but no additional support cap,
router, shrinkage, or target-specific simulator tuning should be undertaken by
default. Resuming that family requires a new representation-level hypothesis
and an explicit decision separate from this roadmap.

49. Reset scope and close the teacher/router experiment family.
    Status: complete. The supported deliverable is now defined as amortized
    fixed-effect `Beta` inference plus the already frozen predictive ensemble,
    with qualified Python MCMC retained as the statistical reference. Teacher
    residual v2 passed compact simulation but failed real-context support;
    sample-size-stable v3 failed an independent effect-shift seed and still
    routed Big Spatial to identity. These are useful negative results, not the
    next production path. No real-data outcome was used to fit or rescue them.

50. Produce a Neural-HMSC v0.1 release-readiness audit.
    Status: complete, not release-ready. Do not add a new calibration model. Exercise the public
    `NeuralHmscInference` and frozen `neural_predictive_affine_v1` artifacts in
    one clean workflow covering checkpoint load, compiled-artifact validation,
    posterior emission, `HmscFit` summaries, prediction, and provenance. Record
    a support matrix that distinguishes public fixed-effect functionality from
    experimental variable-shape, trait, iid, and spatial prototypes.

    The audit must report, without hiding failures:

    - Gaussian, probit, and Poisson fixed-effect simulation results;
    - coefficient SBC coverage, rank mean/variance, and posterior-mean error;
    - Whittaker and Big Spatial frozen predictive scores against qualified MCMC;
    - checkpoint training time, per-dataset inference time, MCMC time, and
      amortization break-even count;
    - exact artifact, manifest, parity, and report hashes;
    - unsupported structures rejected by the public compatibility boundary.

    v0.1 acceptance means the workflow is reproducible, inference-only speedup
    is material, fixed-effect marginal calibration passes the existing frozen
    in-domain gates, and real-data proper-score degradation versus MCMC remains
    inside the declared bounded approximation envelope. It does not require or
    claim joint-posterior equivalence or predictive superiority over MCMC.

    The v0.1 bounded approximation envelope is deliberately weaker than the
    future near-equivalence gate: at least `100x` inference-only speedup on the
    declared ecological workflows, 95% SBC coverage in `[0.925, 0.975]`, rank
    mean and variance errors no greater than `0.025`, and Brier/log-loss ratios
    no greater than `1.10` versus qualified MCMC on both Whittaker and Big
    Spatial. Passing this envelope qualifies a useful accelerated
    approximation; it does not qualify MCMC posterior equivalence.

    The 2026-07-21 audit validated the frozen bundle and all six affine member
    hashes, exercised checkpoint load, compiled-artifact compatibility,
    posterior emission, `HmscFit` summaries/prediction, and unsupported-trait
    rejection. The narrowed probit envelope passed on all three production-shape
    seeds. Whittaker Brier/log-loss ratios were `1.0213`/`1.0327`, Big Spatial
    ratios were `1.0790`/`1.0731`, and mean inference-only speedups were `256x`
    and `211x`, respectively.

    The initial audit failed because the public checkpoint contained no
    `external_monotone` coefficient-calibration artifact. That packaging defect
    is resolved in checkpoint schema `0.4`; the unchanged re-audit now returns
    `release_ready`. Gaussian and Poisson remain implemented without retained
    production-shape release evidence, so the release scope is fixed-shape
    fixed-effect probit and those likelihoods remain experimental. See
    `docs/neural_hmsc_v0_1_release_readiness_audit_2026-07-21.md`.

51. Freeze and document Neural-HMSC v0.1.
    Status: complete.
    Checkpoint schema `0.4` binds the retained `external_monotone` metadata in a
    SHA-256-validated artifact. `NeuralHmscInference.load()` validates method,
    `Beta` parameter, distribution, dimensions, coefficient names, canonical
    metadata hash, and independent-simulation provenance, then applies the
    frozen correction before posterior draws and `HmscFit` emission. Regression
    tests prove direct/public application parity, unchanged weights, domain and
    provenance rejection, and artifact-tamper rejection.

    All three checkpoints were packaged without fitting or reselection; their
    original weights and twelve predictive artifacts are byte-identical. The
    unchanged audit returned `release_ready`, with all narrowed probit release
    gates passing and no blockers. The complete bundle is now frozen atomically
    under `neural_hmsc_v0_1`. Its hash-complete 36-file inventory contains the
    calibrated checkpoints, affine and scale-only members for both named
    datasets, original and release-local predictive manifests, qualification
    evidence, audit, and support matrix. Release-local manifests preserve the
    original member bytes and provenance while removing runtime dependence on
    LUMI absolute paths. `load_neural_hmsc_release()` resolves the stable ID;
    the end-to-end example exercises compiled input, calibrated posterior
    emission, `HmscFit`, and predictive ensemble loading. The content digest is
    `affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8`.
    The same inventory was rehashed and atomically published in the durable
    LUMI deployment registry at
    `/scratch/project_462000131/anisrahm/hmsc-hpc-deployments/neural_hmsc_v0_1`.
    Checkpoint training can cost more than one compact MCMC fit, while repeated
    inference is expected to be substantially faster. Gaussian and Poisson
    remain unqualified. Details are in
    `docs/neural_hmsc_v0_1_release.md`.

52. Promote variable-shape fixed-effect inference to the public API.
    Status: complete. Advanced the existing variable-shape prototype so
    one distribution-specific checkpoint supports predeclared ranges of site
    and species counts. Train across broad simulation priors and compare against
    ridge/Laplace and MCMC baselines. No target-dataset-specific manifests,
    routers, or outcome-conditioned deployment branches are allowed. This is
    the next representation-level research milestone because it expands actual
    usability instead of tuning a fixed checkpoint after inference.

    The public `VariableShapeNeuralHmscInference` now supports fixed-effect
    probit models with 12-48 sites, 2-10 species, and the fixed ordered
    `Intercept`, `x1`, `x2` design. Its mask-aware IRLS/Laplace anchor excludes
    padding from prevalence and information; a permutation-aware neural head
    learns bounded mean/log-scale corrections. Checkpoint schema `0.1` records
    the shape range, anchor, covariates, limitations, and hash-bound independent
    simulation scale calibration. Compiled inputs outside the range or with
    traits, random effects, changed formula/covariates, or another distribution
    are rejected before inference.

    Three independent production-shape local runs each used 64 training, 32
    calibration, and 32 test simulations plus two Python-MCMC fits scored on
    held-out sites. All runs passed boundary, checkpoint parity, coefficient
    coverage, rank, IRLS/Laplace, and MCMC proper-score gates. Mean coverage was
    `0.9512`; mean/worst neural-to-MCMC Brier ratios were `0.9997`/`1.0339`, and
    log-loss ratios were `1.0013`/`1.0297`. Seed `20260730` was predeclared as
    the release candidate; two later seeds were sensitivity-only. The candidate
    and evidence are frozen under `neural_hmsc_variable_probit_v1`, content
    digest `badaf8b8244cbd850693723147c6094bf196482237c52d2cc279ce42b286d2f9`,
    at `/scratch/project_462000131/anisrahm/hmsc-hpc-deployments/neural_hmsc_variable_probit_v1`.
    `neural_hmsc_v0_1` remains unchanged. See
    `docs/neural_hmsc_variable_shape_probit_v1.md`.

53. Qualify structural HMSC families one at a time.
    Status: paused after a documented non-promotion result. The order is
    trait-mediated `Gamma`, iid latent `Eta`/`Lambda` association summaries,
    then full-spatial latent effects with blocked holdouts. Each family receives
    its own simulation truth, MCMC, calibration, identifiability, real-data,
    speed, and API-compatibility gate. Phylogeny, GPP, and NNGP remain deferred
    until the preceding family is qualified. A prototype implementation is not
    sufficient for promotion.

    The first bounded trait-Gamma candidate matched the qualified Whittaker
    shape and semantics: probit, 40 sites, 75 species, `~ TMG`, and one scaled
    non-intercept trait from `~ CN`. It emits separate Beta and Gamma marginal
    approximations through `HmscFit`; it does not claim a coupled joint
    posterior. The initial two-stage Beta-to-Gamma anchor failed the simulated
    MCMC Gamma gate. Its permitted representation redesign used a bounded joint
    site-species probit IRLS/Laplace anchor over `X (x) T`, retained the
    species-level Beta anchor, and added only bounded zero-initialized neural
    corrections.

    Predeclared seed `20260801` passed every gate: Gamma coverage `0.929688`,
    rank mean `0.484436`, rank variance `0.082419`, simulated neural/MCMC Gamma
    RMSE ratio `0.930566`, Whittaker Brier ratio `1.012636`, and Whittaker log
    loss ratio `0.966303`. The intended sensitivity seed `20260802`
    passed mean, rank, MCMC, and real-data predictive gates but failed the
    predeclared Gamma coverage floor with coverage `0.882812` versus `0.90`.
    Coverage was `0.890625` for the intercept coefficient and `0.875000` for
    TMG with negligible signed bias, identifying cross-seed uncertainty
    calibration transfer as the blocker.

    A bounded decision review then found that the intended sensitivity run was
    not independent: advancing the base seed by one overlapped 63/64 training,
    31/32 calibration, and 63/64 test communities. It remains a failed result
    and prevents promotion, but it cannot consume the fresh independent
    evaluation allowed by the stop rule. The implementation remains
    experimental under `pyhmsc.neural` and is not exported from the stable
    top-level API. Both `neural_hmsc_v0_1` and
    `neural_hmsc_variable_probit_v1` remain immutable. See
    `docs/neural_hmsc_trait_gamma_m53_negative_result_2026-07-22.md`.

53A. Requalify frozen trait-Gamma uncertainty with genuinely disjoint evidence.
    Status: terminal failure; neural trait-Gamma v1 is closed and Python MCMC
    remains the only qualified trait-Gamma path.
    Freeze candidate weights at SHA-256
    `bc869b8a92e7d9ea0bf11acb565e571816a68dcff220f0a003f22d2d753cdcac`.
    Do not retrain the neural model or alter posterior means. Fit one
    finite-sample split-conformal scalar from 384 new communities spanning six
    disjoint seed blocks and a balanced 3 by 3 Gamma-scale/residual-scale
    factorial. Evaluate only after calibration is frozen on three untouched
    258-community blocks beginning at `41000001`, `42000001`, and `43000001`,
    plus three Whittaker MCMC replays. Promotion requires every overall,
    coefficient, regime, rank, mean, MCMC, predictive, and real-data gate to
    pass in every block. Any failure permanently ends the v1 neural
    trait-Gamma path and retains Python MCMC as the only qualified path. The
    complete frozen protocol is in
    `docs/neural_hmsc_trait_gamma_calibration_decision_2026-07-22.md`.

    `examples/run_neural_hmsc_trait_gamma_m53a.py` now provides separate
    `smoke`, `calibrate`, and sealed `evaluate` commands. Calibration packaging
    copies the source weights byte-for-byte and binds a finite-sample conformal
    artifact to their hash. The evaluation command validates the production
    calibration manifest and requires the exact confirmation token before it
    constructs any reserved simulation corpus. Focused tests cover the order
    statistic, package round-trip, factorial balance, disposable/reserved seed
    separation, and confirmation barrier. The disposable smoke used only
    `51000001`/`52000001`, preserved the candidate weight hash, completed both
    MCMC paths, and reported `reserved_seed_opened=false`. Its metric values are
    not promotion evidence and did not alter the preregistered protocol.

    The production calibration subsequently used all 384 preregistered
    communities in the six `31000001` through `36000001` blocks. It froze a
    finite-sample conformal multiplier of `1.3018141270106574`, with 42 or 43
    communities in each factorial cell. Independent validation confirmed
    calibration artifact SHA-256
    `3ab539e117827a73718a03b19cee3e1c1191484038d2c67b3b58a5f7746f40a9`,
    checkpoint manifest SHA-256
    `aed26718e224fea37c29a8701249f7c149fc98eccb17a2fe1bbbf91ae8612554`,
    and the unchanged frozen weight hash. The freeze records
    `reserved_evaluation_opened=false` before the authorized evaluation.

    The authorized one-shot evaluation opened all three reserved blocks and
    all three Whittaker replays without changing any frozen component or gate.
    Blocks `41000001` and `42000001` passed every gate. Block `43000001`
    passed all gates except the preregistered MCMC Gamma-RMSE gate: the
    neural-to-MCMC Gamma RMSE ratio was `1.426715144159916`, exceeding the
    fixed `1.25` maximum. All three Whittaker replays passed. Evaluation
    artifact SHA-256 is
    `af55e54172465893b3dbfde4a04a392cbdb55a9f875646f0aacd7bb30b0a467b`.
    The recorded decision is `trait_gamma_probit_terminal_failure`, so no
    trait-Gamma v1 baseline may be frozen and no post-result tuning or rerun is
    permitted. iid Eta/Lambda qualification remains blocked.

    Next, make a bounded scope decision outside trait-Gamma v1: retain Python
    MCMC as the trait-Gamma implementation and return to qualified fixed-effect
    neural work, or preregister a genuinely representation-level structural
    family milestone with fresh evidence. Do not reopen scalar Gamma
    calibration or start iid Eta/Lambda from this failed prerequisite.

54. Generalize qualified variable-shape fixed-effect probit inference to a
    variable-design v2 representation.
    Status: preregistered candidate failed its one-shot 103M production
    evaluation; the single permitted `v2_1` representation redesign is now
    preregistered with untouched seeds and has not been implemented or run.
    Trait-Gamma v1 is terminally closed, Python MCMC remains the structural
    reference, and iid Eta/Lambda work stays blocked. Development returns to
    the already-qualified fixed-effect probit family. The purpose of this
    milestone is usability expansion, not another post-hoc calibration search
    or an MCMC-equivalence claim. The bounded scope decision is recorded in
    `docs/neural_hmsc_post_m53a_scope_decision_2026-07-22.md`.

    Implement a separate coefficient-wise, mask-aware representation with one
    checkpoint covering 12-128 sites, 2-100 species, and 2-8 ordered design
    columns including one leading intercept. Add a covariate mask and shared
    coefficient head so output dimensionality is not baked into the network.
    Preserve site-permutation invariance, species-permutation equivariance, and
    non-intercept covariate-permutation equivariance. Keep the probit
    IRLS/Laplace anchor and bounded residual correction, but compute
    coefficient-local sufficient statistics and explicit design-support
    diagnostics. Compiled covariate names and formula provenance must roundtrip
    without being fixed to `x1` and `x2`.

    This is a new checkpoint schema and baseline candidate. Do not modify or
    repackage `neural_hmsc_v0_1` or `neural_hmsc_variable_probit_v1`. Do not add
    traits, phylogeny, random effects, spatial effects, detection models,
    Gaussian/Poisson qualification, target-outcome routing, or dataset-specific
    calibration. Inputs with missing intercepts, unsupported dimensions,
    rank-deficient or out-of-support designs, or structural HMSC terms must fail
    compatibility checks before inference.

    Qualification must use disjoint training, coefficient-posterior
    calibration, and fixed-evaluation seed blocks with balanced boundary and
    interior shapes, covariate counts, prevalence, coefficient magnitudes, and
    design-conditioning strata. Predeclare one candidate and two independent
    sensitivity runs. Every run must preserve checkpoint/API roundtrip, mask
    parity, site/species/covariate permutation properties, overall 95% coverage
    in `[0.925, 0.975]`, rank-mean and rank-variance errors no greater than
    `0.025`, IRLS/Laplace posterior-mean no-degradation, and held-out MCMC
    Brier/log-loss ratios no greater than `1.10`. Report coverage and rank
    separately by covariate count, intercept/non-intercept role, shape boundary,
    and design-information stratum; no failing stratum may be hidden by the
    aggregate.

    Only after all simulated gates pass, replay Whittaker and Big Spatial
    through the same target-agnostic v2 checkpoint. Both real-data proper-score
    ratios versus qualified MCMC must remain at or below `1.10`, and neither may
    degrade by more than `0.02` relative to its applicable frozen neural
    baseline. Real outcomes are evaluation-only. Promotion freezes a distinct
    `neural_hmsc_variable_design_probit_v2` artifact and keeps v1 as fallback.
    Failure after the permitted representation redesign and fresh evaluation
    closes v2 and leaves v1 as the qualified endpoint; it does not trigger
    another calibration family.

    Immediate implementation step: introduce a separate variable-design
    training tensor and model path with `covariate_mask`, coefficient-local
    features, and a shared coefficient head. Add deterministic invariance,
    padding, compatibility, and v1 hash-regression tests before any model
    training or qualification corpus is generated.

    The separate skeleton is now implemented in
    `pyhmsc/neural/variable_design_inference.py` and checkpoint schema `0.1`.
    `VariableDesignTrainingData` pads sites, species, and covariates with three
    independent masks. `VariableDesignBetaPosteriorModel` retains the probit
    IRLS/Laplace anchor and applies one shared two-value head to
    coefficient/species-local features, so output dimensionality no longer
    depends on a constructor-time covariate count. Its nonzero-head tests prove
    site-padding invariance, site-order invariance, species equivariance, and
    non-intercept covariate equivariance. The experimental facade validates
    leading-intercept semantics, dimensional bounds, binary response, full
    column rank, condition-number support, formula/name provenance, checkpoint
    weight hashes, and unsupported compiled structures. It remains outside the
    stable top-level `pyhmsc` API and contains no fitted calibration.

    Focused validation passed 26 tests across the new skeleton, existing
    variable-shape model/API, and compiler. Local hash-regression checks
    revalidated `neural_hmsc_v0_1` at
    `affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8`
    and `neural_hmsc_variable_probit_v1` at
    `badaf8b8244cbd850693723147c6094bf196482237c52d2cc279ce42b286d2f9`.
    No training or qualification corpus was generated.

    The fixed qualification harness is implemented in
    `examples/qualify_neural_hmsc_variable_design.py` with separate `smoke`,
    role-specific `train-calibrate`, sealed `evaluate`, and `aggregate`
    operations. The initial protocol draft assigned 61M-69M production blocks,
    but a balance unit test generated the candidate 61M training simulations in
    memory without training or metric inspection. Those blocks are retired.
    Protocol `neural_hmsc_variable_design_m54_v1_1` uses untouched 101M-109M
    blocks, and tests now inspect the pure schedule without generating any
    production corpus. Full details and exact gates are in
    `docs/neural_hmsc_m54_qualification_preregistration_2026-07-22.md`.

    The final `v1_1` disposable smoke used only 91M-93M blocks and returned
    `smoke_passed`; all eight plumbing checks passed and
    `production_seed_opened=false`. Aggregate coverage/rank diagnostics were
    `0.952459`/`0.499613`/`0.077149`. The compact species-count stratum rank
    diagnostic failed but is nonpromotional for smoke and remains an unchanged
    production gate. The smoke report SHA-256 is
    `4f4e6ef4cff276039a1ddd6f80af09066fd918cfbe407bbac63dc5d8de99e775`.
    Seeded duplicate disposable runs produced identical weights, calibration,
    training history, and evaluation metrics. Neither immutable baseline was
    modified.

    LUMI `dev-g` job `20129822` ran candidate `train-calibrate` with exact
    confirmation `GENERATE_M54_CANDIDATE_TRAIN_CALIBRATION`. It completed in
    `00:04:07` and froze 243 training communities from 101000001-101000243 and
    243 independent calibration communities from 102000001-102000243. The
    split-conformal coefficient-posterior scale is `0.9808582145420995` over
    49,410 coefficients. The immutable fixed and variable baseline hashes
    remained exact. Remote post-freeze validation and an independent local
    checkpoint load both passed, with the 103000001-103000243 evaluation block
    still unopened.

    Candidate freeze SHA-256 is
    `021488d1868b773232112bfa9199aad74602e26ef119bcd7a7f38bb2ea90728e`.
    Checkpoint manifest, weights, and calibration SHA-256 values are
    `d735ce55c95bddb9df56992d1b6be7f3d8f4f95ae602397e524485997b017df4`,
    `d6f5923873e77c63e51f8b17a65a91e667b97d02fb4681457d7c08d57fbab52b`,
    and `21041be868e38f4d1209f56a8e42336c6b61d54fd2243874f76d5cd1d82da88f`.
    The run is frozen at
    `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_m54_candidate_train_calibration_20129822`.

    LUMI `dev-g` job `20134138` opened the authorized candidate 103M block with
    exact confirmation `OPEN_M54_CANDIDATE_EVALUATION` and completed in
    `00:03:27`. The frozen report decision is
    `variable_design_role_failed`. Coverage (`0.950293`), rank mean
    (`0.510994`), rank variance (`0.074783`), every calibration stratum,
    checkpoint roundtrip, factorial balance, immutable hashes, coefficient
    RMSE versus anchor, and six-context proper scores versus MCMC all passed.
    Aggregate response-scale no-degradation versus the IRLS/Laplace anchor did
    not: Brier ratio was `1.047000` and log-loss ratio was `1.036589`, above
    the frozen `1.02` limits. The report SHA-256 is
    `a0c2ef66365a9a8875ae123941fe8f08f5bcb2c84ababa5581844374a5d2bdbd`.

    Degradation is concentrated in low-support contexts: 12-site Brier/log-loss
    ratios were `1.100853`/`1.066665`, and 8-covariate ratios were
    `1.092878`/`1.069380`; 128-site and 2-covariate strata remained within the
    gate. The shared residual head improves Beta RMSE but moves predictive
    probabilities too broadly when sites are scarce relative to active design
    columns. This is not a coefficient-scale calibration failure. The full
    negative result is recorded in
    `docs/neural_hmsc_m54_candidate_negative_result_2026-07-22.md`.

    Do not open sensitivity blocks 104M-109M: they cannot select or rescue a
    failed candidate. The one permitted redesign is frozen as protocol
    `neural_hmsc_variable_design_m54_v2_1` in
    `docs/neural_hmsc_m54_v2_1_redesign_preregistration_2026-07-22.md`.
    Production roles are coefficient training 111M, predictive auxiliary
    contexts 112M, independent heldout partners 113M, coefficient calibration
    114M, and sealed evaluation 115M. Disposable smoke uses 191M-195M. All
    ranges were unused when preregistered and remain unopened. The frozen
    preregistration SHA-256 is
    `900af8719fc73947cd7addf3b7dc9fe2f233eadbbd2bf9f37bac1286fc15e54d`.

    The frozen representation is a learned coefficient-local support gate over
    a convex IRLS/Laplace-anchor/residual mean mixture. The gate is trained
    jointly with the unchanged coefficient objective and an independent
    heldout probit log-loss/Brier objective. Existing gates are unchanged;
    additional guards require per-site and per-covariate proper-score ratios no
    greater than `1.02`, aggregate Beta RMSE ratio no greater than `0.98` so
    identity is not promotion, and ordered median support movement between the
    12-site/8-covariate and 128-site/2-covariate corners. There is no threshold,
    loss-weight, gate-cap, or post-fit selector search.

    The frozen three-output gated shared head, paired predictive auxiliary
    tensors/objective, separate checkpoint schema `0.2`, and sealed harness are
    now implemented. Anchor-parity, gate-bound, invariance,
    heldout-independence, exact-loss, roundtrip, compatibility-rejection, seed
    barrier, and immutable-baseline regression tests pass. The implementation
    remains separate from the immutable fixed-shape v0.1 and variable-design v1
    inference paths.

    The authorized 191M-195M disposable smoke returned `smoke_passed`; all
    eleven operational smoke checks passed, and inspection confirmed that only
    the disposable seed millions were opened. The production 111M-115M roles
    remain sealed. The smoke is explicitly nonpromotional: its one-epoch
    production-only genuine Beta-RMSE and ordered support-movement diagnostics
    are not acceptance evidence and do not permit retuning. The result and
    artifact hashes are recorded in
    `docs/neural_hmsc_m54_v2_1_disposable_smoke_2026-07-22.md`.

    Production train/auxiliary/calibration generation was explicitly
    authorized with `GENERATE_M54_V2_1_TRAIN_AUX_CALIBRATION`. LUMI dev-g job
    `20144482` completed in `00:08:08` and froze the 111M coefficient-training,
    112M auxiliary-context, 113M independent-heldout, and 114M calibration
    roles. Both the job-local validator and an independent downloaded replay
    passed exact manifest/checkpoint/calibration hashes, seed-role and
    disjointness checks, factorial and marginal balance, paired-heldout
    independence, immutable baseline hashes, and the preregistration hash. The
    freeze SHA-256 is
    `bb32afd655db277064c5c6fcbdf53e2d89a9f42c24a0690c50a494967f46d816`.
    The 115M evaluation block remains sealed. Full evidence is recorded in
    `docs/neural_hmsc_m54_v2_1_production_freeze_2026-07-22.md`.

    The one-shot 115M evaluation was separately authorized with
    `OPEN_M54_V2_1_RESERVED_EVALUATION`. Its wrapper pinned the freeze SHA-256,
    revalidated all artifacts before opening the block, and LUMI dev-g job
    `20179655` completed in `00:04:31`. The exact 115M block, six
    preregistered MCMC seeds, immutable baselines, and gate decision were
    independently validated.

    The immutable decision is `variable_design_v2_1_terminal_failure`.
    Aggregate Brier and log-loss ratios were `1.041475` and `1.028093`, above
    the frozen `1.02` limits. Per-site and per-covariate proper-score gates also
    failed, concentrated at 12 sites and 8 covariates. Support ordering was
    reversed: median gate movement was `0.988575` at 12 sites/8 covariates
    versus `0.384689` at 128 sites/2 covariates. Coefficient coverage, ranks,
    Beta RMSE, MCMC comparison, provenance, and operational gates passed, but
    they cannot average away the predictive and ordering failures. Full
    evidence is recorded in
    `docs/neural_hmsc_m54_v2_1_terminal_result_2026-07-23.md`.

    Milestone 54 is closed under its preregistered stop rule. There is no
    retuning, rerun, real-data evaluation, or third variable-design
    representation. `neural_hmsc_variable_probit_v1` remains the qualified
    endpoint. Next, return to the already-qualified fixed-effect neural scope
    and make a bounded, freshly preregistered capability decision outside the
    failed variable-design family. Do not advance the deferred MCMC
    near-equivalence claim from this result.

55. Reassess summary-level MCMC near-equivalence.
    Status: deferred. Structural near-equivalence cannot advance because no
    neural structural family is qualified. After any future successful
    representation-level structural milestone, publish an explicit evidence
    table for interface, operational, marginal-posterior, predictive,
    association/spatial, and joint-posterior equivalence. Claim
    near-equivalence only for rows and model families that pass. The expected
    mature outcome remains a fast neural approximation with automatic MCMC
    fallback, not elimination of MCMC or unconditional full-HMSC equivalence.

56. Qualify within-species fixed-effect Beta covariance without changing the
    qualified marginal posterior.
    Status: implementation and disposable smoke complete; production
    train-validation and all reserved evaluation remain sealed. The bounded
    post-Milestone 54 decision is recorded in
    `docs/neural_hmsc_post_m54_capability_decision_2026-07-23.md`. The fixed
    qualification protocol is
    `docs/neural_hmsc_m56_covariance_preregistration_2026-07-23.md`, SHA-256
    `d99b63da87103c3d8891cb2fab5bb7ffad30a188ed7be920950345581f8b2d4b`.
    Its byte-level artifact and unused-seed audit is
    `docs/neural_hmsc_m56_artifact_seed_audit_2026-07-23.json.md`, SHA-256
    `5bb9236967afb5a2a1adc166781f4a34359a7469150aa2e19117752dd1fce29c`.

    Keep the exact fixed-shape v0.1 probit boundary: 40 sites, 75 species, and
    two ordered coefficients including the intercept. Bind one exact packaged
    v0.1 checkpoint and coefficient-calibration artifact, preserve its
    posterior means and calibrated marginal standard deviations, and add only
    a per-species positive-definite `2 x 2` coefficient covariance. The
    intended competitor exposes the full IRLS/Laplace covariance anchor and
    learns a bounded context-conditioned residual correction to its
    correlation before reconstructing `scale_tril`.

    This milestone targets the remaining within-species joint-Beta uncertainty
    gap. It does not add cross-species covariance, traits, random effects,
    spatial effects, another likelihood, variable-design generalization, or
    target-outcome routing. Existing full-covariance code is unqualified
    scaffolding, not evidence.

    The preregistration binds v0.1 member `20260721`, its manifest, unchanged
    weights, and calibration hashes. It freezes a nine-feature shared
    correlation head, Fisher-z residual bound, fixed NLL objective, exact
    40-site/75-species/two-coefficient simulator, 100-epoch training, untouched
    211M training, 212M go/no-go validation, 213M-215M one-shot evaluation, and
    291M-292M disposable roles. Mandatory evidence includes exact v0.1
    mean/marginal-scale parity, unchanged marginal SBC, positive-definite
    roundtrip, multivariate coverage/ranks, covariance accuracy and joint
    proper scores versus diagonal v0.1, raw Laplace, and qualified Python MCMC,
    predictive no-degradation, and immutable baseline/provenance checks.
    Identity correlation or merely reproducing the raw Laplace anchor is not
    improvement.

    If successful, freeze a separate covariance-aware fixed-shape artifact and
    claim only bounded within-species coefficient-covariance qualification.
    This is one additional row toward summary-level equivalence, not full-HMSC
    or unconditional MCMC equivalence. Python MCMC remains the statistical
    reference and fallback.

    The implementation now exposes the full per-species IRLS/Laplace
    covariance while preserving the legacy marginal-anchor return values. A
    separate hash-bound overlay supplies the exact nine-feature shared head,
    frozen feature normalization, bounded Fisher-z residual, unchanged v0.1
    means and calibrated marginal scales, and reconstructed positive-definite
    `scale_tril`. The sealed qualification driver implements distinct
    `smoke`, `train-validate`, `evaluate`, and `realdata` confirmation
    boundaries and computes the frozen aggregate and stratum diagnostics.

    The covariance, initialization, loss, HDF5 sampling, artifact roundtrip,
    compatibility, hash, and seed-barrier tests pass together with the
    fixed-shape, calibration, storage, and release regressions: 51 tests
    passed. The 291M-292M disposable smoke passed every operational check with
    exact zero mean/scale deltas, zero overlay roundtrip deltas, minimum
    covariance eigenvalue `0.000470678`, and maximum absolute correlation
    `0.905283`. It opened no 211M-215M seed. Its statistical values are
    explicitly non-promotional and are recorded in
    `docs/neural_hmsc_m56_disposable_smoke_2026-07-24.md`.

    Next, explicitly authorize production correlation-head training and fixed
    validation with `GENERATE_M56_CORRELATION_TRAIN_VALIDATION`. Run the exact
    211M training and 212M go/no-go validation once, freeze the resulting
    overlay and report hashes, and independently verify every aggregate,
    stratum, parity, and provenance gate. Do not open 213M-215M unless every
    212M gate passes and a separate
    `OPEN_M56_RESERVED_COVARIANCE_EVALUATION` authorization is given.

    Production train-validation was explicitly authorized on 2026-07-24 with
    `GENERATE_M56_CORRELATION_TRAIN_VALIDATION`. The sealed LUMI `dev-g`
    wrapper is
    `docs/lumi_neural_hmsc_covariance_m56_train_validation_sbatch.sh`.
    Preflight revalidated both frozen protocol-document hashes and the exact
    v0.1 release, checkpoint, weights, and calibration binding. Job `20192218`
    was submitted with only the 211M training and 212M fixed-validation roles;
    it entered running state on `nid007961`. The 213M-215M blocks remain
    sealed. On completion, download `freeze.json`,
    `postfreeze_validation.json`, the overlay manifest and weights, then
    independently recompute hash, seed-role, gate-consistency, aggregate, and
    stratum checks before making the reserved-evaluation decision.

    Job `20192218` completed in `00:02:39` with exit code `0:0`. The complete
    freeze bundle was downloaded and independently validated. All 20
    operational/provenance checks passed, and local recomputation reproduced
    all 123 gate decisions and the same 47 failures. Mean and marginal-scale
    parity remained exact, covariance was positive definite, predictive
    no-degradation passed, and the learned head improved substantially over
    raw Laplace correlation. It nevertheless failed the fixed prerequisites:
    marginal coverage was `0.826955`, joint coverage was `0.735021`,
    candidate/diagonal joint NLL ratio was `1.169488`, and radial-rank mean was
    `0.600648`. Failures were broad across effect, location, scale, and
    prevalence strata.

    Milestone 56 is terminally closed under its frozen 212M stop rule. The
    213M-215M blocks were not opened and must remain sealed; do not run the
    MCMC subsets or real-data replay. The result and all artifact hashes are
    recorded in
    `docs/neural_hmsc_m56_fixed_validation_terminal_result_2026-07-24.md`.
    The next step is a bounded capability decision outside Milestone 56.
    Retain `neural_hmsc_v0_1` as the qualified neural endpoint and qualified
    Python MCMC as the statistical reference. Any future joint-posterior
    attempt must use a genuinely different representation that can address
    marginal calibration and covariance together, with a fresh
    preregistration and unused seeds; do not retune this correlation-overlay
    family.

57. Attempt one joint heavy-tailed fixed-effect posterior.
    Status: terminal fixed-validation failure; reserved evaluation and
    real-data replay remain sealed. The decision is recorded in
    `docs/neural_hmsc_post_m56_capability_decision_2026-07-24.md`, SHA-256
    `a1a7bc4a54eca4c78f6b32537f1afff662a524557accbd99d7267a28bc2cb2ba`.

    Milestone 56 established that correlation movement is learnable but a
    fixed-marginal overlay is structurally incapable of repairing the broad
    marginal undercoverage observed on its frozen factorial. Retuning
    correlation, adding another scale calibrator, or reopening 213M-215M is
    forbidden.

    The single permitted next representation is a per-species conditional
    bivariate Student-t posterior for the exact 40-site, 75-species,
    two-coefficient fixed-effect probit scope. It must jointly own posterior
    location, both marginal scales, within-species correlation, and degrees of
    freedom. Immutable v0.1 outputs may be input anchors and comparators but
    may not remain fixed candidate marginals. No separate mean, scale,
    correlation, tail, selector, or target-outcome calibration is allowed.

    This is one final bounded joint-posterior attempt, not another iterative
    model family. It receives one preregistered representation and one
    production train-validation opening, with no redesign or calibration-only
    retry. Any fixed-validation failure closes neural joint-posterior
    development for this scope and returns the project to applicability
    detection and automatic Python-MCMC fallback.

    The exact v0.1 and variable-v1 inventories were validated byte-for-byte
    locally and on LUMI, with no extra files. Candidate seed searches covered
    6,215 local repository files, 1,673 retained local evidence files, 977
    LUMI repository files, and 5,016 retained LUMI run files, with zero
    candidate-token collisions. The frozen machine-readable audit is
    `docs/neural_hmsc_m57_artifact_seed_audit_2026-07-24.json.md`, SHA-256
    `1e1150a04cd17643db37988bfc010b611f8f49d638dbd40ead49cd5329b9b25c`.

    The qualification protocol is frozen in
    `docs/neural_hmsc_m57_student_t_preregistration_2026-07-24.md`, SHA-256
    `10878c65bb16746a4a9c57fa91d6a4fd3cbcc753739a816f6cc8b9b738f1a388`.
    It binds a 15-feature shared head that jointly predicts bounded location
    movement, both marginal scales, correlation, and degrees of freedom for a
    bivariate Student-t posterior. It freezes a proper multivariate NLL
    objective, paired response realizations, exact 321M-325M production roles,
    391M-392M disposable roles, aggregate and stratum gates, MCMC subsets,
    real-data boundaries, and a no-redesign terminal stop rule.

    The frozen Student-t posterior math, exact density and sampling, 15-feature
    extractor, 64-64-32 shared six-output head, immutable overlay schema,
    HDF5 semantics, sealed command harness, full non-MCMC aggregate/stratum
    gate calculator, and regression tests are implemented. The targeted M57
    and fixed-effect regression pass contains 62 passing tests.

    The final disposable run used 54 paired training realizations from
    391000001-391000027 and 27 evaluation realizations from
    392000001-392000027. Every operational check passed, all checkpoint
    roundtrip deltas were zero, and no production seed was opened. The result
    is recorded in
    `docs/neural_hmsc_m57_disposable_smoke_2026-07-24.md`. Its one-epoch,
    32-draw calibration diagnostics are explicitly non-promotional.

    The one permitted production train/fixed-validation opening ran as LUMI
    job `20272020` on `dev-g` and completed in `00:06:27` with exit code
    `0:0`. It generated only paired 321M training and 322M validation data.
    Independent local validation reproduced all 176 gate booleans exactly,
    validated the freeze and overlay hashes, and confirmed that no reserved
    evaluation artifact exists.

    The candidate passed 173 gates but failed three frozen requirements:
    geometric width ratio `0.771923` below the allowed `0.80`, degrees-of-
    freedom bound fraction `0.141399` above `0.10`, and strong-effect radial-
    rank mean `0.565492`, whose error from `0.5` exceeds `0.05`. Aggregate
    marginal and joint 95% coverage passed, and location/proper scores improved
    materially, but the frozen stop rule does not permit those gains to
    override interval-width, tail-saturation, or stratum-rank failures.

    The complete result and immutable hashes are recorded in
    `docs/neural_hmsc_m57_fixed_validation_terminal_result_2026-07-26.md`.
    Keep 323M-325M, MCMC subsets, and real-data replay permanently sealed. Do
    not retune, recalibrate, clip, or redesign this Student-t family.

    Next, move outside neural joint-posterior development and implement an
    explicit applicability decision plus automatic qualified Python-MCMC
    fallback at the public inference boundary. Retain `neural_hmsc_v0_1` for
    its qualified fixed-shape neural claim and preserve the failed M57
    artifact only as immutable negative evidence.

58. Close the current neural-model branch and reset the research claim.
    Status: complete. The branch-level outcome is audited in
    `docs/neural_hmsc_branch_closure_audit_2026-07-27.md`, SHA-256
    `e17701174b51ff8714b83bb6935e55b48bb00854e004bbef0c64724f16e1707e`.

    The branch did not produce a neural replacement or summary-level
    near-equivalent of HMSC/MCMC. It produced two qualified fixed-effect probit
    approximations: exact fixed-shape `neural_hmsc_v0_1` and bounded
    variable-shape `neural_hmsc_variable_probit_v1`. It also produced a
    qualified predictive-only affine ensemble for the frozen Whittaker and Big
    Spatial workflows. Qualified Python MCMC remains stronger on the retained
    real-data proper scores and remains the statistical reference.

    Structural and joint-posterior development did not qualify. Trait-Gamma,
    variable design, covariance-only, and joint Student-t families ended under
    terminal preregistered stop rules. iid Eta/Lambda, spatial latent effects,
    cross-species association, traits/phylogeny, detection, and full MCMC
    equivalence remain unsupported. More calibration or selector tuning on the
    current amortized fixed-effect model cannot close those representation
    gaps.

    Retain the simulation/SBC/OOD diagnostics, deterministic seed ledgers,
    HDF5/API adapters, immutable artifact framework, R/Python parity fixtures,
    real-data workflows, LUMI harnesses, qualified baselines, and terminal
    negative evidence. Retire further tuning of M53, M54, M56, M57, failed
    OOD/effect heads, and failed MCMC-teacher residual families.

    Reframe the current branch as `neural fixed-effect JSDM approximation and
    validation infrastructure`. Applicability detection and automatic
    qualified Python-MCMC fallback are maintenance/safety work; they do not
    complete the original Neural-HMSC claim.

    If the original structural research goal continues, start it on a new
    `feature/generative-neural-hmsc` branch only after a fresh design
    preregistration. The first bounded family should combine fixed effects with
    iid latent site factors and species loadings under an explicit generative
    probit likelihood and an end-to-end structured posterior. It must support
    variable site/species dimensions by construction and may not use v0.1
    posterior means/scales or post-hoc scalar calibration as its primary
    uncertainty representation.

    Next, write and review the new-branch design preregistration before
    creating the branch or allocating any simulation seed. The preregistration
    must freeze the generative model, posterior factorization, identifiability
    constraints, first structural scope, fresh seed ledger, joint and
    association recovery gates, MCMC comparator, runtime budget, real-data
    boundary, and one-redesign stop rule.

59. Preregister the first generative Neural-HMSC structural family.
    Status: complete. The frozen design is
    `docs/generative_neural_hmsc_iid_v1_preregistration_2026-07-27.md`,
    SHA-256
    `09c6a195ca139bdf168816b4f50db321c789bfdd061628e4f99a28cca81cea3f`.
    Its independent design review is
    `docs/generative_neural_hmsc_iid_v1_design_review_2026-07-27.md`,
    SHA-256
    `d271caed64dc1346b1f8d9e192534949adedd3122c1e311638e912ca868990cc`.
    The machine-readable unused-seed ledger is
    `docs/generative_neural_hmsc_iid_v1_seed_audit_2026-07-27.json.md`,
    SHA-256
    `39e8763bf8a4fd525dc624570cd2f2b3392dbd1f62d7fa2e3c326f9340194cd6`.

    The first family is a two-coefficient Bernoulli-probit model with one iid
    site random level, exactly two latent site factors, species loadings, a
    shared community intercept hyperparameter, and inferred loading scale. A
    three-round permutation-equivariant site/species message-passing encoder
    emits a rank-16 joint Normal posterior over alpha, Beta, Eta, Lambda, and
    log(tau). Training uses an eight-sample importance-weighted variational
    objective from raw X, masked Y, and masks only. No v0.1/variable-v1
    posterior, IRLS/Laplace output, MCMC teacher, calibration, or ecological
    target outcome may enter the candidate.

    Raw Eta/Lambda recovery is not a gate because the factors are rotationally
    non-identifiable. Qualification instead uses random-effect contribution
    `R = Eta @ Lambda`, association covariance `A = Lambda.T @ Lambda`, its
    correlation form C, invariant SBC projections, energy scores, masked-cell
    prediction, PPCs, and exact-model MCMC comparisons. Qualified Python
    HMSC-HPC remains a separate behavioral comparator because its native
    loading prior differs from the candidate prior.

    Production blocks contain 324 balanced contexts over site count, species
    count, covariate shape, latent strength, prevalence, and replicate. The
    candidate ledger is 501M training, 502M fixed validation, and sealed
    503M-505M evaluation; 591M-592M are disposable only. One separately
    preregistered representation redesign may use sealed 511M-515M after a
    candidate failure. A second fresh production failure closes the iid
    family. Whittaker no-trait iid-site scoring remains sealed until every
    simulation and comparator gate passes.

    The review corrected two design issues before freezing: exact-model MCMC
    and native HMSC are now distinct references, and a shared alpha parameter
    makes rare/common community-prevalence strata generatable from the
    declared prior. The remaining primary risk is whether one low-rank
    Gaussian can approximate the invariant latent posterior geometry. That is
    the hypothesis to test; it may not be hidden by predictive gains or
    repaired by post-hoc calibration.

    Next, create and switch to `feature/generative-neural-hmsc` without opening
    any seed, then implement the frozen simulator, variable-shape tensor
    contract, bipartite encoder, rank-16 joint posterior,
    importance-weighted objective, exact-model MCMC reference, immutable
    artifact schema, sealed harness, and preregistered tests. After tests pass,
    run only the 591M-592M disposable smoke and keep 501M-515M sealed.

60. Implement the frozen generative iid structural candidate and sealed
    disposable harness.
    Status: complete on `feature/generative-neural-hmsc`. The implementation
    report is
    `docs/generative_neural_hmsc_iid_v1_implementation_2026-07-27.md`,
    SHA-256
    `533f92cad0eb85bad85ae34a317c6dcc3cc53d88347518aff1f7ae20d582fe0c`.

    New isolated modules implement the prior-faithful structural simulator,
    outcome-blind response masks, padded variable site/species tensors, the
    three-round bipartite encoder, one rank-16 joint Normal over alpha/Beta/
    Eta/Lambda/log(tau), masked Woodbury density, eight-sample IWAE objective,
    frozen optimizer schedule, deterministic gauge fixing, float64 exact-model
    NUTS reference, invariant chain diagnostics, and immutable checkpoint
    schema. Public lazy imports expose the generative API without modifying the
    qualified legacy model paths.

    The sealed harness validates all frozen document hashes before seed access,
    has no production/reserved mode, and requires exact confirmation before
    opening the 18-cell 591M/592M disposable factorial. Its no-seed seal check
    reports `production_seed_ranges_opened = false`.

    The new suite covers prior/log-density agreement, all 18 factorial cells
    using ordinary non-ledger seeds, masking, padding, site/species
    permutations, low-rank dense-reference parity, finite gradients and one
    optimizer step, gauge invariants, exact-target parity, MCMC execution,
    checkpoint roundtrip/tamper rejection, legacy incompatibility, and seal
    refusal. Together with legacy iid and fixed/variable public API
    regressions, 52 tests passed. Bytecode compilation, public imports, frozen
    hashes, and `git diff --check` also passed.

    No 501M-515M, 591M, or 592M simulation dataset was generated. This is
    implementation evidence only and does not qualify calibration,
    association recovery, prediction, MCMC agreement, or real-data transfer.

    Next, explicitly authorize and run only the 591M-592M disposable smoke
    with
    `OPEN_GENERATIVE_IID_DISPOSABLE_SMOKE=GENERATE_591M_592M_DISPOSABLE_ONLY`.
    Validate its freeze, checkpoint hashes, finite optimization, exact-target
    check, and seal booleans. Keep 501M-515M unopened.

61. Run and independently validate the frozen generative iid disposable smoke.
    Status: complete. The result report is
    `docs/generative_neural_hmsc_iid_v1_disposable_smoke_2026-07-27.md`,
    SHA-256
    `1513407aa02d78f456d89c315dfd919c3e303abaec217a6960c2420054195963`.

    The exact disposable confirmation opened only the 18-cell 591M training
    and 18-cell 592M validation blocks for two epochs. The accepted artifact
    passed checkpoint roundtrip, freeze/report hash checks, finite objective
    checks, independent validation-IWELBO and exact-target recomputation, and
    a nonzero optimizer-movement check. Its checkpoint content SHA-256 is
    `e827df53e27b239f082166c5760f7e39625ce464bdd9d94961ec499737ce5609`;
    its weight SHA-256 is
    `13cde23b67528567bb7207754ae0d2b27833293cb2bc60abe535ba33ff762939`.
    Both `production_seed_ranges_opened` and
    `reserved_seed_ranges_opened` are false, so 501M-515M remain sealed.

    The first disposable artifact was superseded after review found that it
    recorded only the base commit for a dirty worktree. The accepted retry
    requires and validates exact source-file hashes, branch and worktree
    state, and runtime versions. Its independently recomputed validation loss
    is 5716.5771484375, exact truth log joint is -175.4917007215717, and
    maximum weight movement from seeded initialization is
    0.0017552822828292847.

    This is plumbing and optimization evidence only. It does not qualify
    posterior calibration, latent association recovery, MCMC agreement,
    prediction, or real-data transfer.

    Next, freeze the implementation and disposable evidence in a clean source
    commit. Then implement and review a separate production authorization path
    pinned to that commit for 501M training and 502M fixed validation, while
    keeping 503M-515M sealed. Do not open a production seed during the
    implementation or authorization review.

62. Freeze the branch baseline and implement the separated production
    authorization boundary.
    Status: complete without opening a production seed. The branch baseline is
    clean commit `c90aab13806248d9bc339ba921b30201ba870d81`. The production
    authorization review is
    `docs/generative_neural_hmsc_iid_v1_production_authorization_review_2026-07-27.md`,
    SHA-256
    `9de5e72edc34e7dbc16fe5b51254db74bd31c6cd07bffe6bc749bba4c8c57bd2`.

    The new production harness and `dev-g` scheduler wrapper make 501M
    candidate training conditional on an exact confirmation, a full pinned
    source commit, a clean worktree, unchanged frozen protocol hashes, and a
    new output root. The generated corpus is fixed to 324 owning contexts,
    two independently sampled responses per owner, 200 epochs, batch size
    four, eight IWAE samples, model seed 501900001, and final-epoch weights.
    Its immutable freeze records exact source, corpus, checkpoint, weight,
    optimization, and unopened-seed provenance.

    Review showed that 502M cannot honestly be implemented as a simple
    validation-loss mode. The frozen gates require candidate posterior
    diagnostics, a separately trained no-latent ablation, exact-model MCMC,
    qualified Python HMSC-HPC, immutable v0.1, invariance, masked-cell and
    new-site prediction, posterior-predictive checks, runtime measurements,
    and all aggregate/stratum decisions. The harness therefore supplies a
    read-only 502M preflight but deliberately refuses to generate 502M even
    when its token is present until those components are reviewed.

    Focused verification reports 19 passed and one optional exact-MCMC test
    skipped. The scheduler passes Bash syntax validation. A correctly tokened
    501M dry run from the uncommitted production implementation was rejected
    before output creation by the clean-worktree barrier. The seal reports
    501M, 502M, 503M-505M, and 511M-515M unopened.

    Next, commit this production authorization implementation. Then implement
    and test the complete 502M comparator evaluator using only ordinary
    non-ledger fixtures. Do not authorize 501M until the evaluator and exact
    gate report are executable and reviewed; keep 501M-515M sealed.

63. Implement and review the complete fixed 502M qualification evaluator.
    Status: complete without opening a ledger seed. The evaluator review is
    `docs/generative_neural_hmsc_iid_v1_502m_evaluator_review_2026-07-27.md`,
    SHA-256
    `f8e016f020fdd65a8a5cfd1d1847747daab931ebefffdfb594580a930db4926a`.
    The exact evaluator source boundary is frozen in
    `docs/generative_neural_hmsc_iid_v1_502m_evaluator_freeze_2026-07-27.json.md`,
    SHA-256
    `1bd2a8c50ba3f4cb03c47a92ec371f10100636aae9bfa059d1c69925cc160253`.

    Production training now freezes both the generative candidate and a
    separately trained same-architecture R=0 likelihood ablation from the
    exact 501M corpus. The fixed-validation evaluator consumes 256 candidate
    draws per context and computes all preregistered marginal, rank,
    projection, invariant, association, masked-cell, new-site,
    posterior-predictive, conditioning, and runtime diagnostics.

    The fixed comparator paths are concrete: exact-model MCMC on the 36-context
    subset with one same-chain-state continuation, qualified Python-native
    HMSC-HPC with two iid factors on the same masked contexts, and immutable
    v0.1 on only the matched 40-by-75 cells. Exact and Python comparator
    artifacts are retained under complete SHA-256 inventories. The v0.1
    adapter preserves its frozen `TMG` coefficient naming while using the
    unchanged numeric simulated second covariate.

    Every operational, marginal, joint, association, predictive,
    posterior-predictive, and runtime threshold is an explicit named Boolean.
    Exact candidate, ablation, MCMC, Python-HMSC, and v0.1 seed ownership is
    validated before gate evaluation. Missing or non-finite comparator
    evidence cannot become an implicit pass. The report decision is derived
    only from the conjunction of all gates and never opens 503M-515M.

    Ordinary-seed validation reports 31 passed and one optional test skipped.
    The complete synthetic 324-cell gate fixture, exact-MCMC continuation
    adapter, Python-native HMSC-HPC adapter, immutable v0.1 adapter, Python
    compilation, scheduler syntax, read-only 501M preflight, container
    host-source attestation, and missing-token seed refusals passed.
    No production scheduler job was submitted. The long 501M and 502M
    schedulers use `standard-g`; these workflows are not short `dev-g` jobs.

    Next, commit and hash-freeze this evaluator implementation. From that clean
    commit, run the no-seed seal check and read-only production preflight.
    Then explicitly decide whether to authorize one-shot 501M
    candidate-plus-ablation training. Keep 502M-515M sealed until the resulting
    501M freeze and both checkpoint hashes are independently validated.

64. Freeze the evaluator and decide the 501M production-training boundary.
    Status: complete without opening a ledger seed. The evaluator,
    comparators, production harnesses, tests, review, and source-freeze
    manifest were initially committed at
    `773ee4846c6d35851ff6a75d7ceb9debee7b9fad`. The source-freeze manifest
    was subsequently refreshed for the fail-closed CSC TensorFlow-container
    source-attestation path and is now
    `docs/generative_neural_hmsc_iid_v1_502m_evaluator_freeze_2026-07-27.json.md`,
    SHA-256
    `1bd2a8c50ba3f4cb03c47a92ec371f10100636aae9bfa059d1c69925cc160253`.

    From that exact clean commit, `check-seal` reported candidate training,
    fixed validation, reserved evaluation, and redesign ranges unopened.
    The read-only `preflight-training` command passed with both opening tokens
    unset. It revalidated the 13-file production source inventory, frozen
    protocol documents, 324-context/648-realization training contract,
    candidate and no-latent schedules, and the complete 502M evaluator
    component list without creating output or reading a production seed.

    Decision: authorize the one-shot 501M candidate-plus-ablation training
    only. This decision does not authorize 502M fixed validation or any
    503M-515M block. The 501M run must pin the reviewed clean source commit,
    use a new output root, retain final-epoch candidate and ablation artifacts,
    and complete independent freeze, checkpoint-content, weight-hash,
    optimization, corpus-role, and unopened-seed validation before 502M can be
    considered.

    The first LUMI submission preflight found that CSC's TensorFlow container
    does not include Git. It stopped before seed access and before scheduler
    submission. The wrappers now verify commit and clean status with host Git,
    pass a strict source attestation into the container, and retain all
    in-container frozen-document and production-file hash checks. Missing,
    malformed, or non-clean attestations fail closed. This is an operational
    portability correction only; simulator, representation, objective,
    training schedule, comparator, gate, threshold, and seed roles are
    unchanged.

    The corrected in-container no-seed preflight passed at clean source commit
    `fc2ac5aff84f2fbed2c3604f3001f3647618fdc0`. The one-shot 501M
    candidate-plus-ablation training was then submitted as LUMI job
    `20301852` on `standard-g`, using an isolated detached worktree pinned to
    that commit and exact confirmation
    `OPEN_GENERATIVE_IID_501M_TRAINING=GENERATE_501M_CANDIDATE_TRAINING_ONLY`.
    The scheduler reported the job pending for priority. No 502M-515M token
    was set.

    Job `20301852` later exited during its first read-only validation call.
    The failure and artifact disposition are recorded in Milestone 65.

65. Independently validate and disposition the 501M training artifacts.
    Status: complete without reopening 501M or opening a later seed. The
    validation report is
    `docs/generative_neural_hmsc_iid_v1_501m_validation_2026-07-28.md`,
    SHA-256
    `929fd0dbf3e6b92f60acde33ef92bea2097765f39eb0784bd2929a101bad2250`.
    The machine-readable evidence is
    `docs/generative_neural_hmsc_iid_v1_501m_validation_2026-07-28.json.md`,
    SHA-256
    `fe10e6fc76f029cb8fc4d6dd135bbc3e4e49718d27c1fc7c93663c9cf0b6d107`.

    Slurm job `20301852` reported `FAILED` after 11:03:29, but both frozen
    200-epoch models and every planned training artifact had already been
    written. The failure occurred only in read-only validation: the corpus
    manifest used the false seal key `fixed_validation_opened`, while the
    validator required `fixed_validation_seed_ranges_opened`. The validator
    now accepts either exact key, requires every present alias to be false,
    and rejects missing, true, or conflicting values. Future generated corpus
    manifests use the canonical key.

    The complete run root was downloaded and independently validated. The
    freeze and sidecar SHA-256 both resolve to
    `93f11221c9bbbd3b8ced541888397541ab61f0b88ae23eebc3431e969512ae39`.
    Candidate content/weights are
    `d36dd3b23ccdba36041792716b9fb2cb21a437265870e686cdef1f01b9d05e30`
    and
    `43b4eded085b0213f53ffa795e5bf91f367a2dc86cd17a2915da7e404f8043c7`.
    Ablation content/weights are
    `691f8c992ec709ac241af32ea0fd7e94e43c3ed9d79c768e01a23a4a1e8193bc`
    and
    `1ab01e332b7b23609fb0bdb7a41e978a29c3f237c94eb02c0ab0276bb541232d`.

    Both checkpoints pass schema, content, file-set, size, and weight-hash
    checks; load successfully; contain only finite weights; and emit finite
    posterior parameters with positive diagonal scales on an ordinary
    non-ledger fixture. Corpus ownership is exactly 324 contexts and 648
    realizations. All optimization metrics are finite. Candidate and ablation
    clean-source inventories match. The corrected focused suite reports 37
    passed and one optional skip. The independent
    `postfreeze_validation.json` SHA-256 is
    `0f6ac100df4497d7df8636962cf5c67a76dbefcb4915dcd77a4df6446c3c87c6`.

    Decision: accept the immutable 501M artifacts and do not retrain or reopen
    501M. This is not a candidate-quality decision; quality remains reserved
    for 502M. The refreshed evaluator source-freeze manifest SHA-256 is
    `1bd2a8c50ba3f4cb03c47a92ec371f10100636aae9bfa059d1c69925cc160253`.

    The read-only 502M preflight initially validated the accepted hashes but
    also exposed that the scheduler conflated training-source and
    evaluator-source commits. The boundary now requires distinct immutable
    pins: `fc2ac5aff84f2fbed2c3604f3001f3647618fdc0` owns the 501M
    checkpoint provenance, while the clean post-correction commit owns the
    executable 502M evaluator. The preflight and fixed report retain both.

    The final read-only preflight passed from clean evaluator commit
    `b3d6cd10b045dd52d5513b80519b04220d614f07`, separately bound to
    training source `fc2ac5aff84f2fbed2c3604f3001f3647618fdc0`, training
    freeze
    `93f11221c9bbbd3b8ced541888397541ab61f0b88ae23eebc3431e969512ae39`,
    candidate content
    `d36dd3b23ccdba36041792716b9fb2cb21a437265870e686cdef1f01b9d05e30`,
    and ablation content
    `691f8c992ec709ac241af32ea0fd7e94e43c3ed9d79c768e01a23a4a1e8193bc`.
    The preflight retained every 502M-515M seal flag as false.

    Decision: 502M is technically eligible for explicit one-shot
    authorization, but is not authorized by this validation record.

    The one-shot 502M fixed-validation block was explicitly authorized with
    `OPEN_GENERATIVE_IID_502M_FIXED_VALIDATION=EVALUATE_502M_FIXED_VALIDATION_ONCE`
    and submitted as LUMI job `20351142` on `standard-g`. It uses an isolated
    clean detached worktree at evaluator commit
    `b3d6cd10b045dd52d5513b80519b04220d614f07`, the accepted 501M run root,
    and the exact training source, freeze, candidate, and ablation hashes
    above. The scheduler reported the job pending for priority. No
    503M-515M token was set.

    Next, monitor job `20351142`. On completion, download and independently
    validate `freeze.json`, `fixed_validation_report.json`,
    `context_metrics.json.gz`, every exact-MCMC and Python-HMSC artifact
    inventory, immutable v0.1 provenance, all named gate booleans, and the
    read-only validation output. Open 503M-505M only if every fixed 502M gate
    passes; keep 511M-515M sealed.

66. Classify the 502M monolithic timeout and freeze a recovery boundary.
    Status: complete without a 502M gate decision and without opening
    503M-515M. The timeout report is
    `docs/generative_neural_hmsc_iid_v1_502m_timeout_2026-07-30.md`,
    SHA-256
    `a64884b09da18ae85b7076682949fcd09f65cb312502621be24b0b50b190ac89`.

    LUMI job `20351142` ended as `TIMEOUT` after `1-00:00:11`; the batch step
    was cancelled with exit code `0:15`. It wrote 11 of the fixed 36
    exact-MCMC artifacts and no Python-HMSC comparator output, context-metrics
    bundle, report, freeze, or read-only validation. All 11 partial files are
    members of the preregistered subset and pass archive-integrity checks.

    This is an incomplete infrastructure attempt, not a qualification result
    and not a candidate failure. Serial exact MCMC projects to roughly 75
    hours before the remaining comparators, while `standard-g` permits at most
    48 hours. Another monolithic submission is therefore prohibited.

    The original partial run root remains immutable and is excluded wholesale
    from the final decision. A recovery may reuse only the same deterministic
    502M seeds under unchanged simulator, candidate, ablation, exact-MCMC,
    Python-HMSC, immutable-v0.1, metric, gate, threshold, and continuation
    semantics. It must use independently hash-frozen comparator shards,
    exact seed ownership, complete artifact inventories, and a separately
    authorized finalizer. No partial/recovery result selection is allowed.

    The sealed sharded recovery implementation is complete locally. The
    production harness now assigns the unchanged 36-context exact-MCMC/Python-
    HMSC subset to 36 immutable one-seed shards, validates exact artifact
    inventories and all training/checkpoint/evaluator bindings, excludes the
    timed-out partial root, and atomically reconstructs the unchanged 324-
    context fixed-validation report through the existing
    `fixed_validation_gates` evaluator. Shard execution and finalization use
    distinct authorization tokens.

    Three scheduler stages are frozen:

    - `docs/lumi_generative_neural_hmsc_iid_v1_recovery_preflight_sbatch.sh`
      runs token-free on `dev-g`;
    - `docs/lumi_generative_neural_hmsc_iid_v1_recovery_shard_sbatch.sh`
      owns array indices `0-35` on `standard-g`;
    - `docs/lumi_generative_neural_hmsc_iid_v1_recovery_finalize_sbatch.sh`
      requires a separate finalizer token and reruns the unchanged gate
      evaluator on `dev-g`.

    Ordinary non-ledger fixtures cover token ordering, all 36 fixed seed
    owners, shard roundtrip, artifact tampering, uninventoried files,
    incomplete ownership, and scheduler contracts. The focused generative iid
    and fixed-gate suites pass with `49 passed, 1 skipped`; Python compilation,
    shell syntax, and whitespace validation also pass. No 502M recovery or
    503M-515M seed was opened by these tests.

    The implementation was frozen at evaluator commit
    `d32093b367e0af40a9d9bd583d0812419b83667f` and pushed to the fork.
    Token-free LUMI preflight job `20430583` completed on `dev-g` in 31 seconds
    with exit code `0:0`. Its JSON evidence SHA-256 is
    `8d07dfdcbf37a14ecfc1ebbcf3e45cecd8f1375662e5da22ab1de0c021ce17f3`.
    It matched the accepted 501M freeze, candidate, and ablation hashes; the
    timeout-report hash; all 36 unchanged recovery owners; and all 11 excluded
    partial exact-MCMC artifact hashes. Every recovery/reserved/redesign
    opening flag remained false.

    Based on that exact preflight, the comparator-shard token was explicitly
    authorized and LUMI array job `20430754` was submitted on `standard-g`
    with indices `0-35` and concurrency limit 12. The immutable authorization
    record is
    `docs/generative_neural_hmsc_iid_v1_502m_recovery_authorization_2026-07-30.md`.
    Its SHA-256 is
    `92747de1deb02386b97593b21cf77f8513bd7aa664da9f83cefd5d739c6c26e8`.
    Job `20430754` completed all 36 shards with exit code `0:0`. Standalone,
    token-free validation job `20461335` then independently re-hashed all
    freezes, results, 36 exact-MCMC files, and 108 Python-HMSC files. The
    validation artifact SHA-256 is
    `b19c1b5e45cdc27b9e7cc41bacdbef07af15e35b22125c908085ee5cf2a5b623`;
    its shard-binding SHA-256 is
    `4d2ec028f3eef8d60e75d40fa91d8f47fbbebd8ae52ac2b42d21cb263e50df98`.
    It found exact ownership, exact inventories, no extra files, no partial
    reuse, and no opened 503M-515M block.

    That complete evidence separately authorized the one-shot finalizer with
    `OPEN_GENERATIVE_IID_502M_TIMEOUT_RECOVERY_FINALIZER=FINALIZE_502M_TIMEOUT_RECOVERY_ONCE`.
    LUMI job `20461616` was submitted on `dev-g`; the shard token was unset.
    The immutable authorization record is
    `docs/generative_neural_hmsc_iid_v1_502m_finalizer_authorization_2026-07-30.md`.
    Its SHA-256 is
    `db9443711a3449e7f6ab2cc203d116eb050c46e6b64407092bb283e9a837f79f`.
    Blocks 503M-515M remain sealed.

    Next, monitor job `20461616`. On completion, independently validate the
    final freeze, report, compressed context metrics, reconstructed comparator
    inventories, shard bindings, immutable-v0.1 provenance, and every
    unchanged 502M gate. Open no later seed unless the complete final report
    passes.

67. Close the first generative iid candidate on fixed-validation failure.
    Status: complete as a valid negative result. LUMI finalizer job `20461616`
    completed on `dev-g` in 10:16 with exit code `0:0`. The frozen report
    contains 65 named gates: 26 passed and 39 failed. Its decision is
    `stop_before_reserved_evaluation`.

    Independent reconstruction validation confirmed all 36 shard owners, 36
    exact-MCMC files, 108 Python-HMSC files, all 144 same-inode shard
    hardlinks, immutable-v0.1 provenance, and no partial-attempt reuse. The
    metrics bundle was downloaded and the unchanged evaluator was rerun
    locally; every gate Boolean matched exactly.

    The failure is scientific and material. Beta, R, alpha, and log(tau) 95%
    coverage were 0.6740, 0.3062, 0.3333, and 0.0247. Median association truth
    correlation was -0.0015, candidate/Python association correlation was
    -0.0044, and random-effect RMSE was effectively identical to the no-latent
    ablation. Masked Brier/log-loss ratios were 1.148/1.161 against exact MCMC
    and 1.211/1.217 against Python HMSC. Site-richness coverage was 0.751.
    Runtime gates passed, including 0.139-second maximum-shape inference, but
    speed does not compensate for posterior and predictive failure.

    This blocks 503M-505M, real-data replay, and promotion. Blocks 511M-515M
    remain sealed. The full disposition is
    `docs/generative_neural_hmsc_iid_v1_502m_failure_2026-07-30.md`.
    Its SHA-256 is
    `36f04ee135974f549e5544c33dc911f213fa0536c9ec902a2c71e0046c09bb91`.

    Next, conduct one bounded no-seed representation decision. Either
    preregister the single permitted representation-level redesign with a
    genuinely different posterior representation and encoder, or close the
    iid generative family and retain Python MCMC as the qualified structural
    path. Do not open 511M-515M before a concrete redesign is reviewed and
    hash-frozen.

68. Preregister the single permitted generative iid representation redesign.
    Status: complete without implementation or seed access. The bounded
    decision is
    `docs/generative_neural_hmsc_iid_v2_representation_decision_2026-07-31.md`,
    SHA-256
    `13041f6368eeaa64d4eae4446782c99c7a0b8af2a13bb13be9a69bec040df7ea`.
    The frozen redesign protocol is
    `docs/generative_neural_hmsc_iid_v2_orbit_preregistration_2026-07-31.md`,
    SHA-256
    `a2eaee0441833167f707f7cb9ae6b1162ba4e118ee3dfc1a245983cc9ada24c2`.
    The redesign seed reaudit is
    `docs/generative_neural_hmsc_iid_v2_seed_reaudit_2026-07-31.json.md`,
    SHA-256
    `9a463943508651e74855701cdbd9870961efd3fd3c07a444674da36a67d49344`.

    Decision: use the one redesign for
    `generative_neural_hmsc_iid_probit_v2_orbit`. The posterior replaces the
    raw joint Gaussian with a low-rank multivariate Student-t global block and
    an exact O(2)-orbit-symmetrized matrix-Normal latent block over concatenated
    Eta and transpose(Lambda). A four-block permutation-equivariant attention
    encoder initializes the joint conditional posterior, followed by exactly
    four fixed common-random IWAE refinement steps.

    This directly targets the demonstrated failure: the raw Gaussian averaged
    over factor rotations, ignored the latent path, and produced severe global
    and random-effect undercoverage. A raw-coordinate flow was rejected because
    it does not enforce the continuous symmetry. Direct R/A inference was
    rejected because its manifold measure and induced prior would exceed the
    permitted posterior/encoder redesign.

    The generative model, simulator, prior, likelihood, factor count, objective
    class, factorial, comparators, metrics, thresholds, real-data boundary, and
    stop rules remain unchanged. Calibration, truth losses, MCMC teachers,
    routing, fallback, and gate changes remain prohibited.

    The local and retained LUMI evidence reaudit found no actual use of
    511M-515M or 593M-594M. One broad lexical LUMI match was GPU telemetry
    value `515747354`, outside the reserved 515M interval, and was classified
    as a non-seed false positive. All redesign ranges remain sealed.

    Next, implement only the v2 posterior-math and encoder skeleton using
    ordinary non-ledger fixtures. Before any disposable seed, prove analytic
    orbit-density/quadrature parity, dense Student-t and matrix-Normal parity,
    finite refinement gradients, nondecreasing accepted refinement steps,
    permutation/padding gates, checkpoint incompatibility with v1, immutable
    v1 hash regression, and maximum-shape inference without dense state
    covariance materialization.

69. Implement the generative iid v2 orbit posterior and pass every
    ordinary-fixture feasibility gate.
    Status: complete without ledger-seed access. The implementation evidence
    is
    `docs/generative_neural_hmsc_iid_v2_implementation_2026-07-31.md`,
    SHA-256
    `0d54f04ea5ec5c654df73594b7ff6614157152ec87bdfc3ecfd09c2401550cab`.

    The implementation is isolated in
    `pyhmsc/neural/generative_iid_v2.py` and
    `pyhmsc/neural/generative_iid_v2_artifact.py`. It adds the frozen masked
    low-rank Student-t global block, exact O(2)-orbit matrix-Normal latent
    block, theta-conditioned FiLM mean, four edge-aware bipartite attention
    blocks, four common-random IWAE refinement steps, unchanged raw-state
    assembly, immutable schema-v2 artifacts, and v1/v2 compatibility
    rejection. It imports the existing prior and likelihood rather than
    defining a changed target.

    The complete v2 feasibility suite passed 9/9 tests, including dense
    Student-t and matrix-Normal references, 4096-angle orbit quadrature,
    O(2) target invariance, finite gradients through all four refinement
    steps, nondecreasing accepted common-random IWELBOs, hidden-response
    isolation, permutation/padding gates, exact checkpoint roundtrip,
    immutable v1 hashes, and full refined `96 x 75` inference without a dense
    state covariance. The unchanged non-slow v1 generative suite passed 45/45,
    and public/release API regressions passed 26/26.

    Two pre-freeze defects were found and closed: the conditional latent-mean
    layer was initially absent from unrefined checkpoint builds, and the
    stateful Student-t gamma draw duplicated the batch dimension. Exact
    conditional-mean roundtrip and both stateful/stateless sample shapes are
    now tested. A one-epoch ordinary-fixture outer-training smoke completed
    with finite diagnostics.

    This is implementation feasibility only, not statistical qualification.
    No calibration, truth loss, teacher, routing, fallback, selector, gate
    change, or threshold change was added. The 593M-594M disposable roles and
    every 511M-515M role remain sealed.

    Next, freeze the implementation and evidence in one clean commit. Then
    implement and review a sealed 593M-594M disposable harness using the
    unchanged preregistered architecture, objective, gates, and seed roles.
    Run a token-free/no-seed preflight before separately authorizing that
    smoke.

70. Implement and review the sealed 593M-594M disposable harness.
    Status: complete through clean-commit, token-free preflight.
    The review is
    `docs/generative_neural_hmsc_iid_v2_disposable_harness_review_2026-07-31.md`,
    SHA-256
    `2830238c927a3a2d90174b0b200c38ffa423483ec658004a73de33122a9f4208`.

    The harness fixes the 18-cell `593000001-593000018` training block,
    18-cell masked `594000001-594000018` validation block, model seed
    `511900001`, two smoke epochs, and the unchanged v2 objective and
    refinement. It exposes no 511M-515M execution mode.

    Preflight refuses every `OPEN_GENERATIVE_IID*` token before source
    inspection, calls no simulation function, creates no output, and requires
    a clean pinned commit. Disposable execution and replay require only
    `OPEN_GENERATIVE_IID_V2_593M_594M_DISPOSABLE_SMOKE` with the exact frozen
    value and reject every unrelated opening token.

    Authorized output would include byte-level fingerprints for all 36
    disposable datasets and a complete hash/size inventory for the corpus,
    report, checkpoint manifest, and weights. Independent validation replays
    all corpus fingerprints, the fixed validation objective, exact target,
    optimizer movement, source provenance, and all later-seed seal booleans.
    Disposable metrics remain operational evidence only.

    The v2 implementation and harness suites pass 19/19 tests. The no-token
    seal check reports every 593M-594M and 511M-515M role unopened. No ledger
    simulation was generated.

    The clean preflight ran against
    `ba48fc93c53447cf4277f9c15946bb95f00d332e` and returned
    `generative_iid_v2_disposable_preflight_sealed`. It confirmed 18 unique
    cells, model seed `511900001`, two epochs, exact source-file hashes, no
    simulation generation, no output creation, and every 593M-594M and
    511M-515M open flag false. The evidence is
    `docs/generative_neural_hmsc_iid_v2_disposable_preflight_2026-07-31.md`,
    SHA-256
    `bd895a7bfb51f240bdc5cdc4c710959322015b258dfd0996a4b6b4ce042aed3a`.

    Next, conduct a separate authorization decision for only the 593M-594M
    disposable smoke. If authorized, pin a clean reviewed commit and use only
    `OPEN_GENERATIVE_IID_V2_593M_594M_DISPOSABLE_SMOKE=GENERATE_593M_594M_DISPOSABLE_ONLY`.
    After completion, independently validate the freeze, all 36 corpus
    fingerprints, checkpoint artifacts, finite optimization, validation
    IWELBO, exact target, optimizer movement, source provenance, and all
    511M-515M seal booleans. Disposable output cannot tune the candidate.

71. Authorize one 593M-594M disposable smoke and independent replay.
    Status: attempted in LUMI job `20518403`; failed during token-free
    preflight before any disposable seed or artifact was opened. The
    authorization record is
    `docs/generative_neural_hmsc_iid_v2_disposable_authorization_2026-08-01.md`,
    SHA-256
    `2d4e8068be49c4b21a4c9dacc3067074b92d2e7572e093a7df8fd72c2a63988c`.

    Source is pinned to
    `940d73d6de6e032797e4d695bd9799a74ef0b943` and packaged as isolated archive
    SHA-256
    `76911182c1d34bcd4c979f70b1340af126ddd89baafdc821c4024cc6f846a43a`.
    The shared dirty LUMI checkout must not be modified or used. The reviewed
    scheduler wrapper SHA-256 is
    `7a0bf9ecf89a5e1896ba254d24e916978cfe8e76caf0898a7dffef1df679cf07`.

    Only `593000001-593000018` training and `594000001-594000018` masked
    validation may open, with exact token
    `OPEN_GENERATIVE_IID_V2_593M_594M_DISPOSABLE_SMOKE=GENERATE_593M_594M_DISPOSABLE_ONLY`.
    The job runs token-free preflight, two-epoch candidate training, frozen
    validation, and independent replay. Blocks 511M-515M remain sealed.

    The isolated archive was verified, but the scheduler's host-absolute
    `PYTHONPATH` did not survive the TensorFlow Singularity `/scratch` to
    `/pfs/lustrep4/scratch` namespace translation. Python resolved another
    installed `pyhmsc` package and raised `ModuleNotFoundError` for
    `pyhmsc.neural`. Independent inventory found an absent run root, a zero-byte
    preflight file, and no corpus, checkpoint, freeze, or post-freeze artifact.
    Thus 593M-594M were not opened and 511M-515M remain sealed. The complete
    negative result is recorded in
    `docs/generative_neural_hmsc_iid_v2_disposable_attempt_20518403.md`.

    A bounded scheduler correction now uses `PYTHONPATH=.` after changing to
    the isolated source root and launches the harness with `python -m`, avoiding
    script-mode path preemption by `examples/`. The full token-free LUMI
    preflight passed with no simulation generation, no output creation, and all
    later blocks sealed. No representation, objective, optimization, artifact,
    gate, threshold, or seed role changed. The failed authorization is consumed
    under its preregistered retry rule. The corrected, unsubmitted scheduler is
    hash-frozen at
    `ff618d92c4d4f616507aaa31e2f434cb2cdaa9b2d985bcc0e7e567bc6735cdb7`.

    Next, make a separate explicit retry authorization decision for only
    593M-594M using the reviewed scheduler hash above. If authorized and
    completed, independently validate all 36 corpus fingerprints, freeze
    inventory, checkpoint hashes, finite optimization, fixed validation
    objective, exact-target replay, optimizer movement, source provenance, and
    every 511M-515M seal boolean.

72. Separately authorize one corrected 593M-594M disposable retry.
    Status: completed with numerical failure in LUMI job `20518775`; stop before
    511M. The immutable decision is recorded
    in
    `docs/generative_neural_hmsc_iid_v2_disposable_retry_authorization_2026-08-01.md`,
    SHA-256
    `257cba945a3d7a40697190812021a69d01be5e7a831d8f2fca0e69018ef4770f`.
    It retains source commit
    `940d73d6de6e032797e4d695bd9799a74ef0b943`, archive SHA-256
    `76911182c1d34bcd4c979f70b1340af126ddd89baafdc821c4024cc6f846a43a`,
    and corrected scheduler SHA-256
    `ff618d92c4d4f616507aaa31e2f434cb2cdaa9b2d985bcc0e7e567bc6735cdb7`.

    Only disposable training `593000001-593000018` and masked validation
    `594000001-594000018` may open. The fresh retry run root is
    `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/generative_iid_v2_disposable_retry1_940d73d_20260801`.
    The corrected full token-free LUMI preflight already passed without
    simulation generation or output creation. Blocks 511M-515M remain sealed.

    The corrected token-free preflight passed. The run generated the authorized
    18 training and 18 masked-validation communities, and independent replay
    matched all 36 corpus fingerprints. Both preflight and corpus manifest keep
    511M-515M sealed. Training then emitted failed batched Cholesky operations
    and stopped at the frozen guard with `FloatingPointError: non-finite v2
    gradient`. No checkpoint, smoke report, freeze, post-freeze validation,
    validation IWELBO, or exact-target replay was produced. The artifact and
    finite-optimization conjunction therefore fails. Full evidence is in
    `docs/generative_neural_hmsc_iid_v2_disposable_retry_failure_2026-08-01.md`.

    Decision: `stop_before_511m_numerical_failure`. The retry authorization is
    consumed. Do not open 511M-515M and do not report the disposable smoke as a
    pass.

    The frozen failure record SHA-256 is
    `ac9881c9574cae15eea1a4ff51c8effcd7b4bf57500abf555b9e8f0c89760f5f`.

73. Conduct a bounded no-seed numerical decision for generative iid v2.
    Status: complete; bounded implementation repair accepted. The decision and
    evidence are frozen in
    `docs/generative_neural_hmsc_iid_v2_numerical_review_2026-08-01.md`,
    SHA-256
    `e3b708a09b0c920676e592759f44f7457cc75decff66138b6b29c509254a6192`.

    Ordinary mixed-shape training exposed a missing feasibility test. LUMI job
    `20519231` showed that float32 and symmetric float64 GPU Cholesky each
    emitted 16 rejected decompositions, while symmetric float64 CPU Cholesky
    emitted none and preserved identical finite metrics. The accepted repair
    moves only the small rank-16 Woodbury factorization to symmetric float64 CPU
    arithmetic. It adds no jitter, clipping, calibration, fallback, posterior,
    objective, refinement, architecture, schedule, gate, threshold, simulator,
    or seed-role change.

    The repaired source SHA-256 is
    `87828857ee1718a8825a1a15e7af99abe49a86ee4d179f6cbce6591162aa71bc`.
    LUMI job `20519352` validated the actual repaired source over ordinary seeds
    `983001-983018`: two epochs were finite, repaired metrics exactly matched
    both explicit float64 reference modes, failed-Cholesky warnings were zero,
    and `ledger_seeds_opened = false`. The complete local slow suite passed
    29/29 tests. The family remains open, but the prior disposable failure is
    not converted into a pass. Blocks 511M-515M remain sealed.

74. Freeze the numerical repair before any disposable decision.
    Status: complete. Repair and no-ledger evidence were committed at
    `a057721`; the harness refreeze is commit
    `80a0a35c57eaf1f0786c473eea5185ff149b72cf`, and the final clean preflight
    boundary is `cca9e97518e77c5ca958dfdc3bee753997ed7ac5`.

    The harness now validates the immutable numerical-review hash, inventories
    repaired model SHA-256
    `87828857ee1718a8825a1a15e7af99abe49a86ee4d179f6cbce6591162aa71bc`,
    and carries the numerical-review digest into future freeze metadata. The
    complete local slow suite passed 30/30 tests.

    LUMI token-free preflight job `20520889` completed at clean commit
    `cca9e97518e77c5ca958dfdc3bee753997ed7ac5`. Its JSON SHA-256 is
    `a9f3c3f0f535f31217da279f9907f8c1d0fcf11001a7337ffbb4a4fdade9fe6f`,
    byte-identical to local output. All 11 source records passed independent
    path/size/hash validation. No simulation or output generation occurred and
    every 593M-594M and 511M-515M opening flag remained false. Full evidence is
    in
    `docs/generative_neural_hmsc_iid_v2_repaired_preflight_2026-08-01.md`,
    SHA-256
    `f17c0ac5d4c57519331c44495820d7a4312c6a2f473b93633a42fd8d8c4a784f`.

75. Decide separately whether to authorize one final repaired disposable run.
    Status: complete; one final repaired disposable verification was
    authorized and consumed by terminal LUMI job `20521366`. The decision is
    recorded in
    `docs/generative_neural_hmsc_iid_v2_final_disposable_authorization_2026-08-01.md`,
    SHA-256
    `a3e051a916798a41ea53cdb3d63bcf6f3685a342986003bef01d1284c477d858`.

    The candidate remains pinned to clean commit
    `cca9e97518e77c5ca958dfdc3bee753997ed7ac5`, archive SHA-256
    `bb343bcef927455b5ffedb0483015f75f3da053176d58c1f032b3fece7790eb1`,
    and repaired model SHA-256
    `87828857ee1718a8825a1a15e7af99abe49a86ee4d179f6cbce6591162aa71bc`.
    The reviewed terminal scheduler SHA-256 is
    `9ca1238d7e88560e58b0e92727c821933e90ec704d1ea69e61a86c4aef31066c`.

    Only `593000001-593000018` training and `594000001-594000018` masked
    validation may open, using fresh run root
    `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/generative_iid_v2_disposable_final_cca9e97_20260801`.
    Any scheduler, numerical, artifact, replay, provenance, warning, or seal
    failure closes v2 before production and does not authorize another retry.
    Passing permits only a separate 511M authorization decision.

76. Submit and independently validate the final repaired disposable run.
    Status: terminal failure; generative iid v2 is closed before production.
    The final scheduler was submitted exactly once as LUMI job `20521366` and
    failed after `00:02:12` with exit code `1:0`.

    The token-free preflight passed and independently validated all 11 frozen
    source records. The authorized run generated exactly 18 training and 18
    masked-validation communities. Independent local regeneration from the
    exact `cca9e975` archive matched all 36 fingerprints and the complete corpus
    manifest. All 511M-515M flags remained false, and the logs contained zero
    Cholesky warnings.

    Training nevertheless stopped at the frozen guard with
    `FloatingPointError: non-finite v2 gradient`. Only the corpus manifest was
    produced: checkpoint, weights, smoke report, freeze, deterministic replay,
    and exact-target evidence are absent. The conjunctive terminal gate fails.
    The full record is
    `docs/generative_neural_hmsc_iid_v2_final_disposable_failure_2026-08-01.md`,
    SHA-256
    `ed835cab3a3dd492a29463604da61afb6fbef6664db6f28c9ab6e98bab1fb993`.
    Do not retry v2 and do not open 511M-515M.

77. Conduct a no-seed branch-level closure audit.
    Status: complete. The audit is recorded in
    `docs/generative_neural_hmsc_branch_closure_audit_2026-08-01.md`, SHA-256
    `696aecb72290cbbf2fe907f78a61f526a270ca1d5766ff60f135b31b3db7763b`.

    The branch added no qualified structural neural capability. Retain the
    immutable v0.1 and variable-probit-v1 marginal baselines, the
    predictive-only affine ensemble, qualified Python MCMC, and the reusable
    simulator/comparator/artifact infrastructure. Archive generative iid v1
    and v2 as negative research evidence. Retire 511M-515M without opening
    them, and close this branch for model-family development.

78. Conduct a bounded no-seed go/no-go review for any future generative family.
    Status: complete, reviewed, and accepted with a scoped pivot. Do not build
    a third standalone amortized posterior. Proceed only to preregistration for
    `neural_transport_hmsc_iid_probit_v0_1`: a neural warm start and frozen
    data-conditioned affine transport around the existing corrected HMC/Gibbs
    target. The network may improve initialization and geometry but may not
    define the accepted posterior.

    The decision is recorded in
    `docs/neural_transport_hmsc_go_no_go_2026-08-01.md`, SHA-256
    `135adb8b2614d75f1aab17f2fbe0d2d379b9c971aec62e2fbfa5968acb6fc887`.
    The first scope is fixed at probit, two covariates, 40 sites, 12 species,
    two latent factors, and one iid site-level random intercept. Exact posterior
    parity is a prerequisite to scoring time-to-convergence or ESS/second. The
    identity transport, warm-start-only path, and ordinary Python MCMC are
    mandatory controls.

    The acceptance record is
    `docs/neural_transport_hmsc_milestone_78_acceptance_2026-08-01.md`, SHA-256
    `85940ebda51bbb3e4892f931c9d3beec041afd1c80ccebe4ae35bdf072174f61`.
    The new branch was created directly from clean accepted commit
    `253e7802642192b0d72427b461bf9fc9cc30fa99`.

79. Preregister Neural-Transport HMSC on a new branch.
    Status: complete on `feature/neural-transport-hmsc`; no implementation or
    simulation generation occurred. The complete preregistration is
    `docs/neural_transport_hmsc_iid_probit_v0_1_preregistration_2026-08-01.md`,
    SHA-256
    `6badb6391af537ec4d8886d08e76ef6e000766f492d333628cf1441db165ec89`.
    The unopened seed audit is
    `docs/neural_transport_hmsc_iid_probit_v0_1_seed_audit_2026-08-01.json.md`,
    SHA-256
    `cf876a599ce93ed807dc8c939b7c3fca4b6168f716c71bc94d03fdeb0b227330`.

    The preregistration freezes the native HMSC prior-predictive law, target and
    kernel composition, two-stream DeepSets encoder, deterministic rank-two
    warm start, positive affine transport, exact Jacobian, supervised robust
    center/scale objective, four controls, MCMC settings, 18-cell factorial,
    exactness and efficiency gates, artifact semantics, fallbacks, and stop
    rules. Fresh 711M-717M, 719M, and 791M-792M roles remain sealed. Retired
    511M-515M remains forbidden.

80. Implement the ordinary-fixture transport kernel.
    Status: pending separate implementation authorization. Extract the reusable
    HMC target/state adapter, then implement the frozen simulator, context
    encoder, warm start, positive affine bijector, transformed corrected HMC,
    artifact schema, and explicit Gibbs fallback using only ordinary fixtures
    below one million. All prior-law, target/Jacobian, identity,
    finite-gradient, stationary-moment, permutation, compatibility, fallback,
    checkpoint, and immutable-regression tests must pass before a disposable
    seed is considered. No seed or scheduler is authorized by Milestone 79.

81. Run fresh disposable exactness and efficiency qualification.
    Status: blocked by Milestones 79-80. Compare ordinary Python MCMC,
    warm-start-only MCMC, and corrected neural-transport HMC/Gibbs. Every
    posterior-parity gate must pass before scoring the proposed 25% reduction
    in time to convergence and 1.25x median ESS/second target.

82. Run fixed validation and bounded real-data confirmation.
    Status: blocked by Milestone 81. Open fixed validation only after a complete
    disposable pass. Real ecological outcomes are last and may evaluate the
    frozen transport but may not train or select it. Traits, phylogeny, spatial
    effects, broader likelihoods, and variable shapes remain out of scope.

### Active Stop Rules

- A failed candidate family may receive one representation-level redesign and
  one fresh independent evaluation. A second failure pauses the family.
- Identity fallback counts as safety, not improvement.
- Real ecological outcomes may evaluate a frozen candidate but may not train or
  select target-specific movement.
- No new five-seed LUMI comparison follows a failed local or compact gate.
- Post-hoc scale, cap, router, and selector changes cannot substitute for a
  representation change after repeated failures.
- Every active step must end in a frozen artifact, a qualified capability, or a
  documented negative result; experiment plumbing alone is not a milestone.

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

The first real-data benchmark is recorded in
`docs/neural_hmsc_whittaker_validation_2026-06-29.md`. LUMI job `19609057`
completed a fixed-effect Whittaker plant holdout comparison. It found useful
neural ranking performance and a 624x inference-only speedup, but also showed
that scalar simulation calibration worsened real probability and richness
metrics despite restoring nominal synthetic coverage. Calibrated neural draws
must therefore remain experimental until calibration transfers to real data.

The Whittaker requalification workflow now separates three neural artifacts:
the uncalibrated coefficient posterior, the coefficient-calibrated posterior,
and a probit predictive-only distribution fitted on independent simulations.
Coefficient acceptance is determined only by SBC; predictive acceptance is
determined only on untouched Whittaker holdout observations. The combined
qualification gate requires both decisions to pass. Results from the earlier
job `19609057` must not be interpreted under these corrected semantics.

LUMI requalification job `19637813` completed on 2026-07-01. The independent
probit predictive scale was `0.934528`, avoiding the severe held-out
degradation caused by applying the `5.241370` coefficient scale to prediction.
Coefficient SBC, held-out predictive, and combined qualification gates passed.
Calibrated SBC ranks remain nonuniform, however, with variance `0.05081`
against the `0.08333` expectation and a chi-square p-value effectively zero.
This is qualification under the current coverage/non-degradation policy, not a
claim of posterior equivalence.

The second real-data validation uses the independent Big Spatial Plant
community dataset. A deterministic projection selects 40 spatially distributed
training sites, 75 species by training prevalence, and holds out the remaining
360 sites. Standardized maximum temperature is mapped to the checkpoint's
single environmental-gradient input. The workflow reuses the exact Whittaker
checkpoint and both calibration scales, records content hashes, and forbids
neural or calibration fitting. Target results must be documented separately
from the Whittaker qualification result.

LUMI job `19638224` completed the frozen Big Spatial Plant transfer on
2026-07-01. The predictive-only artifact passed the independent held-out gate,
with Brier `0.05672` versus `0.05698` uncalibrated and `0.04750` for MCMC.
Checkpoint and calibration hashes matched the frozen source artifacts, and no
parameters were updated. Coefficient-posterior agreement remained incomplete:
95% interval overlap with MCMC was `0.1648`. The result qualifies predictive
transfer only; target coefficient calibration remains not assessable without
coefficient truth.

The direct Whittaker Python-only HMSC parity workflow is recorded in
`docs/whittaker_r_python_hmsc_parity_2026-07-17.md`. Initial LUMI job
`19967782` showed that response `Y` and phylogeny `C` matched the R-created
`Hmsc` object, but native preprocessing did not match R/Hmsc `XScaled` and
`TrScaled` semantics. Python-native compile now centers and scales
non-intercept covariates with sample standard deviation, drops trait intercepts,
scales trait values, stores scale parameters, and applies fixed-effect scaling
during prediction. LUMI requalification job `19983202` passed the direct
Whittaker parity gate: all boundary arrays matched, `Beta` mean correlation was
`0.999832`, `Gamma` mean correlation was `1.000000`, and held-out Brier/log-loss
deltas were effectively zero. This supports Python-only HMSC equivalence to the
original R+Python HMSC-HPC boundary for the fixed-effect Whittaker
trait/phylogeny model. The next parity step is to extend the same direct
R/Python protocol to simpler fixed-effect no-trait cases and then iid
random-effect fixtures. That implementation is now present in
`examples/run_direct_r_python_parity.py` and
`docs/lumi_direct_r_python_parity_sbatch.sh`. LUMI job `19984923` completed the
corrected fixture run successfully. Spatial random-level semantics were then
inspected explicitly in
`docs/spatial_r_python_hmsc_boundary_inspection_2026-07-18.md`. The inspection
confirmed exact Full/GPP distance-array boundary equality, identified NNGP's
ragged R list versus padded Python tensor representation, and corrected the
Python-native spatial default `alphapw` to match R/Hmsc's 101-point alpha grid.
Corrected LUMI inspection job `19995051` passed the spatial boundary checks.
Direct spatial parity is now complete for compact Full, GPP, and NNGP fixtures:
Full passed in retry job `19995352`, and GPP/NNGP passed in job `19999784`.
The parity decision is recorded in
`docs/python_only_hmsc_parity_decision_2026-07-19.md`: compact fixtures are
adequate for controlled boundary/parity coverage, but the broader Python-only
HMSC equivalence claim needed one larger real-data spatial requalification.
That requalification is now complete: LUMI job `20000066` ran Big Spatial Plant
full-spatial direct R/Python parity with the R-created HMSC boundary retained as
the comparator. The final report was regenerated from completed posteriors with
`--reuse-existing-posteriors`; boundary arrays passed, Python-native predictive
MAE improved over the R-boundary comparator (`0.099725` vs `0.113449`), and
posterior summaries were retained as diagnostics. The next roadmap step is to
return to neural work rather than extend parity again by default. That neural
return path is now wired into the real-data runners: pass
`--reference-parity-metrics` to attach a passed direct R/Python parity metrics
JSON, relabel the MCMC comparator as `qualified_python_mcmc_fixed`, and record
the parity provenance in the neural report/metadata. Qualified-comparator
real-data reruns are now complete for Whittaker (`20000918`) and Big Spatial
transfer (`20001432`). Both passed their neural acceptance gates, and both
continue to show that qualified Python MCMC is the stronger real-data predictor
on core proper scores.

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

## Historical First Concrete Pull Request

This section records the initial proposed scope. Milestones 0 through 11 and
the subsequent real-data qualification work have superseded it.

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
