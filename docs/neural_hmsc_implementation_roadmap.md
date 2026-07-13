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
  Next substep: rerun the production-shape local sanity check and then the
  five-seed LUMI comparison if in-domain gates remain intact.

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
