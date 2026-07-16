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
