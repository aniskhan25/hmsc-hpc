# Neural-HMSC Benchmark Protocol

This protocol implements Milestone 0 from
`docs/neural_hmsc_implementation_roadmap.md`. It freezes the first benchmark
target, simulation regimes, MCMC reference settings, and evaluation metrics for
the initial amortized Neural-HMSC work.

The selected feature direction is:

```text
Amortized Neural-HMSC posterior inference
```

The first implementation target is deliberately narrow:

```text
fixed-effect HMSC
Gaussian response first
posterior target: Beta
no traits
no phylogeny
no random effects
no spatial effects
```

Probit and Poisson fixed-effect benchmarks are specified here as secondary
targets, but they should not be implemented before the Gaussian `Beta` posterior
prototype has passed the checks below.

## Benchmark Goals

The benchmark should answer four questions:

1. Can the neural model recover fixed-effect posterior means?
2. Are neural posterior intervals calibrated against simulated truth and MCMC?
3. Can neural posterior samples be stored in shapes compatible with existing
   `pyhmsc` posterior summaries?
4. Is neural inference materially faster than running a comparable MCMC fit?

This is not a predictive-only benchmark. Predictive performance is useful, but
the primary target is approximate posterior inference for HMSC-shaped
parameters.

## Model Scope

Supported in the first benchmark:

- fixed effects only
- formula: `~ x1 + x2`
- target parameter: `Beta`
- generated response matrix `Y`
- generated covariate table `X`
- generated truth table `truth_beta.csv`
- compiled JSON/HDF5 artifacts from `pyhmsc.compiler`
- optional MCMC reference posterior from the existing TensorFlow Gibbs sampler

Explicitly unsupported in the first benchmark:

- `Gamma`
- species traits
- phylogenetic covariance
- `Eta`
- `Lambda`
- species association summaries
- iid random effects
- spatial random effects
- detection/occupancy structure

## Required Compiled Inputs

The neural fixed-effect benchmark should read the same compiled artifacts used
by the Python-native sampler.

Required metadata:

- `distribution`
- `formula.X`
- `dimensions.n_sites`
- `dimensions.n_species`
- `dimensions.n_covariates`
- `names.sites`
- `names.species`
- `names.covariates`

Required arrays:

- `Y`
- `X`
- `Beta_init` only for shape/reference compatibility

Arrays present but not used in Milestone 0:

- `T`, except as an intercept-only trait design produced by the compiler
- random-level arrays, which should be absent
- `C`, which should be absent

Any benchmark loader should fail if the compiled artifact has:

- non-empty `random_levels`
- `capabilities.traits = true`
- `capabilities.phylogeny = true`
- `capabilities.spatial = true`

## Dataset Regimes

Each benchmark config defines three splits:

- `train`: many simulated datasets for fitting the neural amortizer
- `validation`: simulated datasets for model selection and calibration
- `test`: held-out simulated datasets for final reporting

The first generator should sample dimensions rather than use one fixed matrix
shape, but the ranges should stay small enough for fast local smoke tests.

Initial dimension ranges:

```text
n_sites:      24, 40, 64
n_species:    2, 4, 8
n_covariates: fixed at 3 including intercept
covariates:   Intercept, x1, x2
```

Initial corpus sizes:

```text
train:       512 datasets
validation: 128 datasets
test:        128 datasets
```

Local smoke corpus sizes:

```text
train:       16 datasets
validation:  4 datasets
test:         4 datasets
```

LUMI-scale corpus sizes should be decided only after the local prototype learns
nontrivial posterior means.

## Simulation Specification

For each generated dataset:

1. Sample `n_sites` and `n_species` from the configured allowed values.
2. Draw covariates:

   ```text
   x1_i ~ Normal(0, 1)
   x2_i ~ Normal(0, 1)
   X_i = [1, x1_i, x2_i]
   ```

3. Draw species coefficients:

   ```text
   Beta_kj ~ Normal(0, beta_scale)
   ```

4. Optionally apply coefficient sparsity:

   ```text
   Beta_kj = 0 with probability beta_zero_probability
   ```

5. Generate the linear predictor:

   ```text
   eta_ij = X_i @ Beta_j
   ```

6. Generate `Y` from the configured response family.

Gaussian:

```text
y_ij ~ Normal(eta_ij, sigma)
```

Probit:

```text
y_ij ~ Bernoulli(Phi(eta_ij))
```

Poisson:

```text
y_ij ~ Poisson(exp(clip(eta_ij, -6, 6)))
```

The same `poisson_eta_clip` bounds must be passed to predictive benchmark
metrics. This is part of the declared simulation model, not a post-hoc metric
clamp; omitting it allows Gaussian posterior tails to dominate response-scale
means after exponentiation.

Poisson reports must also include the fraction of neural and MCMC linear
predictors outside those bounds. Calibration passes predictive acceptance only
when calibrated RMSE is at most 1.25 times uncalibrated RMSE, at most twice the
MCMC-reference RMSE, and fewer than 1% of neural predictors require clipping.

Coefficient-posterior and predictive calibration are separate artifacts. The
Beta posterior uses the coefficient-coverage multiplier. Poisson and probit
predictive-only artifacts may use a different multiplier for response-scale
scoring, but must be labelled `predictive_only` and must not be interpreted as
posterior uncertainty. Probit scale selection uses the exact Gaussian-probit
expectation and is fitted only on independent simulated calibration data.
Conditional coefficient calibration may replace the global coefficient scale,
but it must retain this separation: conditional version 4 metadata belongs to
the coefficient artifact, while the predictive-only artifact retains its
scalar version 2 metadata. Conditional features and weights must be fitted only
from simulated calibration data and must be computable without coefficient
truth when applied.
Rank-aware fitting must report its prevalence weights and rank-moment penalty.
Support-aware fitting must store the calibration support definition and scalar
fallback. SBC rows must report mean/minimum support trust and the fraction of
coefficients with trust below `0.5`, separately for in-domain and OOD data.
Overall qualification requires both predictive acceptance and SBC acceptance.
SBC acceptance requires at least 90% empirical 95% coverage and no material
degradation in coverage error, normalized-rank mean error, or normalized-rank
variance error relative to the uncalibrated posterior.

SBC reports also include diagnostic strata. These rows do not replace the
`overall` row used by qualification:

- prevalence: rare (`<= 0.10`), intermediate (`> 0.10` and `<= 0.30`), and
  common (`> 0.30`) occurrence or nonzero-count prevalence
- coefficient: one row for each named fixed-effect coefficient
- design information: low, intermediate, and high within-coefficient thirds
  of diagonal expected Fisher information

Expected information is evaluated at the raw neural posterior mean, never at
simulated coefficient truth. Gaussian diagnostics use unit observation weight,
probit uses `phi(eta)^2 / (Phi(eta) * (1 - Phi(eta)))`, and Poisson uses
`exp(eta)`. This makes the same information feature computable for eventual
real-data inference and conditional calibration. Each row records its selected
rank count, rank moments, tail fractions, coverage, posterior-mean RMSE, and
histogram diagnostics. Empty or undersized strata are omitted.

For a real occurrence holdout, predictive acceptance requires predictive-only
Brier score and log loss no worse than 1.10 times the uncalibrated neural
values and no worse than twice the MCMC values. Prevalence and richness MAE
must each be no worse than 1.25 times the uncalibrated neural values. These
thresholds evaluate a frozen calibration procedure; held-out observations
must never be used to select the predictive multiplier.

### Independent real-data transfer

A second ecological dataset must evaluate a frozen checkpoint and frozen
calibration procedure. The transfer workflow must record content hashes for
the checkpoint, coefficient source artifact, calibration metadata, and source
acceptance report. It must not call neural training or calibration-fitting
APIs. Target species may be selected using target training observations, but
target holdout observations may not influence site selection, species
selection, weights, calibration scales, or acceptance thresholds.

Predictive-transfer acceptance requires the source run to have passed its
coefficient SBC and combined qualification gates, and the target
predictive-only artifact to pass the unchanged real-occurrence predictive
gate. Reports must use the explicit key
`predictive_transfer_acceptance_passed`; a generic transfer-qualification label
is not permitted. Coefficient-posterior calibration cannot be claimed for a
real target dataset without simulated coefficient truth.

Initial simulation hyperparameters:

```text
beta_scale: 0.75
beta_zero_probability: 0.0
gaussian_sigma: 0.35
poisson_eta_clip: [-6, 6]
```

The first benchmark should avoid sparse coefficients. Sparsity can be introduced
later as an out-of-distribution stress test.

## Simulation-Based Calibration and OOD Stress Tests

SBC uses independent simulated datasets that are excluded from training and
post-hoc scale fitting. For each `Beta` coefficient, its true simulated value is
ranked among posterior draws. Ties are randomized uniformly over their valid
rank positions. Reports include the observed and discrete-uniform expected rank
histograms, normalized-rank mean and variance, lower/upper tail fractions,
maximum coefficient-level mean-rank deviation, and a chi-square uniformity
diagnostic. The chi-square p-value is descriptive because ranks from coefficients
within the same community are dependent.

The fixed-effect benchmark also evaluates these named OOD regimes with the same
trained checkpoint and calibration layer:

```text
covariate_shift:   x ~ Normal(2.0, 1.5)
effect_size_shift: Beta ~ Normal(0.0, 1.5)
combined_shift:    both shifts
```

The in-domain reference remains `x ~ Normal(0, 1)` and
`Beta ~ Normal(0, 0.75)`. OOD results must remain separate from in-domain SBC
summaries and report posterior-mean RMSE degradation relative to the matching
in-domain calibrated or uncalibrated posterior.

## MCMC Reference Settings

MCMC references are expensive and should be generated only for a subset of the
validation and test splits.

Local reference setting:

```text
chains: 2
samples: 500
transient: 250
thin: 5
rng_seed: fixed per dataset
```

Publication-grade reference setting:

```text
chains: 4
samples: 2000
transient: 1000
thin: 10
rng_seed: fixed per dataset
```

Reference subset sizes:

```text
validation: 16 datasets
test:       32 datasets
```

The neural model may train against simulation truth before MCMC references
exist. MCMC references are required before claiming posterior approximation
quality.

## Neural Target

Poisson fixed-effect posterior family:

```text
q_phi(Beta_j | Y, X) = Normal(mean_phi,j, L_phi,j L_phi,j^T)
```

Output shape:

```text
Beta mean:  n_covariates x n_species
Beta marginal scale: n_covariates x n_species
Beta scale_tril:     n_species x n_covariates x n_covariates
```

The initial model should sample `Beta` draws and write them as standard
posterior samples:

```text
chains x draws x n_covariates x n_species
```

The first prototype may use one synthetic chain if needed, but it should record
metadata indicating that samples are neural posterior draws, not MCMC chains.

## Training Objective

Initial objective:

```text
negative log probability of simulated truth Beta under q_phi
```

Recommended auxiliary losses:

- posterior mean squared error against true `Beta`
- weak penalty against collapsed posterior scales

Optional later objective:

- moment matching against MCMC posterior mean and standard deviation

Do not optimize only predictive log likelihood in Milestone 0 or Milestone 1.
That would turn the work into a predictive neural JSDM rather than amortized
posterior inference.

## Metrics

Posterior accuracy against truth:

- `beta_mean_rmse_truth`
- `beta_mean_mae_truth`
- `beta_interval_coverage_truth_80`
- `beta_interval_coverage_truth_90`
- `beta_interval_coverage_truth_95`
- `beta_interval_width_mean`

Posterior agreement with MCMC:

- `beta_mean_rmse_mcmc`
- `beta_sd_rmse_mcmc`
- `beta_ci_overlap_90`
- `beta_ci_overlap_95`
- `posterior_sample_mean_correlation`

Predictive checks:

- held-out response log likelihood where available
- posterior predictive mean RMSE
- species-level PPC coverage
- site-level total abundance/richness PPC coverage, depending on distribution

Operational metrics:

- neural inference wall time per dataset
- MCMC wall time per dataset
- speedup factor
- peak memory where available

Calibration diagnostics:

- coverage by response family
- coverage by `n_sites`
- coverage by `n_species`
- coverage by coefficient magnitude bin

## Baselines

The neural model should be compared against simple non-neural baselines before
being compared only to MCMC.

Required baselines:

- zero posterior mean with fixed prior scale
- ordinary least squares for Gaussian data
- per-species generalized linear model approximation for probit/Poisson where
  available
- MCMC posterior reference for the selected subset

If the neural model cannot beat the simple deterministic baselines on posterior
mean recovery, the architecture should not be extended to traits or random
effects.

## First Distribution

The first implementation should use Gaussian responses.

Reasons:

- fixed-effect Gaussian recovery is the easiest posterior target to debug
- posterior mean errors are interpretable
- likelihood issues are less likely to mask architecture problems
- MCMC comparisons are cheaper and more stable

After Gaussian passes, add probit, then Poisson.

## Framework Decision

Use TensorFlow as the implementation framework and TensorFlow Probability where
it simplifies distribution handling.

Initial choices:

- TensorFlow/Keras for encoders and training loops
- TensorFlow Probability for distribution objects and log probability
- no PyTorch/JAX dependency in the first implementation
- no normalizing flow in the first implementation

The fixed-effect benchmark uses a per-species Cholesky factor so correlated
coefficient uncertainty is retained through the log-link response prediction.
Gaussian and probit retain the diagonal Gaussian, which performed better in
scaled validation. Normalizing flows remain a later option if these Normal
families are too restrictive.

## Expected Files

Milestone 0 adds:

```text
docs/neural_hmsc_benchmark_protocol.md
examples/projects/neural_hmsc_fixed_gaussian/benchmark.yaml
examples/projects/neural_hmsc_fixed_gaussian/README.md
examples/projects/neural_hmsc_fixed_probit/benchmark.yaml
examples/projects/neural_hmsc_fixed_probit/README.md
examples/projects/neural_hmsc_fixed_poisson/benchmark.yaml
examples/projects/neural_hmsc_fixed_poisson/README.md
```

Later milestones may add generated data under these project directories, but
Milestone 0 should not check in large generated corpora.

## Pass/Fail Gate for Milestone 1

Milestone 1 can begin when:

- the benchmark protocol is committed
- fixed Gaussian, probit, and Poisson benchmark configs exist
- Gaussian is marked as the first implementation target
- MCMC reference settings are frozen
- metrics and baselines are frozen

Milestone 1 should not begin if:

- the target includes random effects or spatial effects
- the first model target is predictive likelihood only
- posterior sample storage compatibility is undefined
