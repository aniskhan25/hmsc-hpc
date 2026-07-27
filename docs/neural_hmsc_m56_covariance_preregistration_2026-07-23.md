# Neural-HMSC Milestone 56 Covariance Qualification Preregistration

Date: 2026-07-23

Protocol: `neural_hmsc_fixed_probit_covariance_m56_v1`

Status: frozen before implementation, simulation generation, model fitting, or
evaluation.

## Decision Boundary

Milestone 56 is a single bounded attempt to add within-species coefficient
covariance to the already-qualified fixed-shape fixed-effect probit Beta
posterior. It follows the terminal closure of trait-Gamma v1 and
variable-design Milestone 54. It may not reopen either family or add:

- variable site, species, or design-column shapes;
- Gaussian, Poisson, or another likelihood;
- traits, phylogeny, latent factors, random effects, spatial effects, or
  detection;
- cross-species covariance or ecological association semantics;
- target-outcome routing;
- another coefficient mean, marginal scale, or predictive calibration search;
- a full-HMSC or unconditional MCMC-equivalence claim.

There is one candidate and no post-validation representation redesign. Failure
of the fixed 212M validation or any reserved evaluation gate closes Milestone
56 and retains the existing release unchanged.

## Audited Immutable Inputs

The machine-readable artifact and seed audit is:

`docs/neural_hmsc_m56_artifact_seed_audit_2026-07-23.json.md`

Its SHA-256 is:

`5bb9236967afb5a2a1adc166781f4a34359a7469150aa2e19117752dd1fce29c`

The complete 36-file `neural_hmsc_v0_1` inventory validated byte-for-byte:

- release content SHA-256:
  `affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8`;
- package manifest SHA-256:
  `d2daa81ec841390df59324a208216ffa0032ac514e6c679649d98815490bdbc7`.

The candidate is bound only to packaged member seed `20260721`, the member used
by the public v0.1 release audit and example:

| Artifact | Frozen value |
| --- | --- |
| Checkpoint path | `checkpoints/20260721/neural_checkpoint` |
| Checkpoint version | `0.4` |
| Checkpoint manifest SHA-256 | `f62cd2217df6cc71cbe9f915c0cfbd3a3327b6684b3c5452bd9399aa130133a8` |
| Weights SHA-256 | `bb6e76d3ec9bc5e294ceac3051c3b2d7e5273db5053cfa5ceac676913d6265d9` |
| Source weights SHA-256 | `bb6e76d3ec9bc5e294ceac3051c3b2d7e5273db5053cfa5ceac676913d6265d9` |
| Calibration file SHA-256 | `595fc0796d36802002cee09b270d53162f1fce100b83aecd32476e0958a0fd94` |
| Internal calibration SHA-256 | `81041eb9075b32c4c0f848927c1feea1d49e5cdcde7fb4e3aa7c4f566865a0a4` |
| Calibration method | `external_context_monotone_scale`, semantics version 9 |
| Distribution | probit |
| Formula | `~ TMG` |
| Ordered coefficients | `Intercept`, `TMG` |
| Dimensions | 40 sites, 75 species, 2 coefficients |
| Current posterior family | diagonal Normal |
| Hidden units | 192, 192 |
| Probit anchor | IRLS/Laplace, 8 iterations, prior precision 1.0, eta clip 6.0 |

Members `20260722` and `20260723` remain immutable sensitivity evidence and
cannot select the correlation representation, loss, threshold, or candidate.
`neural_hmsc_variable_probit_v1` remains an immutable regression baseline with
content SHA-256
`badaf8b8244cbd850693723147c6094bf196482237c52d2cc279ce42b286d2f9`.

## Frozen Capability

For each species, the candidate adds one posterior correlation between the
intercept and TMG coefficients. The qualified v0.1 posterior mean and calibrated
marginal standard deviations remain unchanged.

Let:

- `mu = (mu_0, mu_1)` be the calibrated v0.1 posterior mean;
- `sigma = (sigma_0, sigma_1)` be its calibrated marginal standard deviations;
- `rho_L` be the correlation from the full IRLS/Laplace covariance using the
  frozen eight-iteration anchor;
- `delta_z` be the learned bounded Fisher-z residual.

The exact correlation parameterization is:

```text
rho_anchor = clip(rho_L, -0.979, 0.979)
z_anchor   = atanh(rho_anchor / 0.98)
delta_z    = 0.75 * tanh(raw_delta)
rho        = 0.98 * tanh(z_anchor + delta_z)
```

The per-species Cholesky factor is reconstructed without altering marginal
scales:

```text
L_00 = sigma_0
L_10 = rho * sigma_1
L_11 = sigma_1 * sqrt(max(1 - rho^2, 1e-6))
```

Therefore `diag(L @ L.T)` is exactly `(sigma_0^2, sigma_1^2)` up to floating
point tolerance. The candidate output is a `BetaPosterior` with the unchanged
`mean` and `scale` plus `scale_tril`.

The bound `0.98` and Fisher residual bound `0.75` are frozen. There is no
post-fit correlation cap, shrinkage grid, identity selector, or MCMC-teacher
target.

## Frozen Correlation Head

The head is shared across all 75 species. It receives exactly nine
species/context features:

1. `z_anchor`;
2. logit observed prevalence, clipped to `[1e-4, 1 - 1e-4]`;
3. frozen v0.1 posterior intercept mean;
4. frozen v0.1 posterior TMG mean;
5. log frozen calibrated intercept standard deviation;
6. log frozen calibrated TMG standard deviation;
7. sample mean of TMG;
8. log sample standard deviation of TMG;
9. log condition number of the two-column design matrix.

Features are standardized using means and standard deviations computed only
from the 211M training block. Standard deviations are floored at `1e-6`. The
normalizer is part of the frozen candidate artifact.

The head architecture is:

```text
9 inputs
Dense(32, activation="relu")
Dense(16, activation="relu")
Dense(1, activation=None, kernel_initializer="zeros",
      bias_initializer="zeros")
```

Hidden kernels use TensorFlow's seeded Glorot-uniform initializer under model
seed `211900001`; hidden biases are zero. The final zero initialization makes
the untrained candidate exactly equal to the clipped raw Laplace correlation.
The base v0.1 checkpoint and calibration tensors are `stop_gradient` inputs.
Only the correlation-head weights are trainable.

## Frozen Simulation Design

Every production corpus has the exact qualified shape, names, formula, and
probit response. The one TMG predictor is generated as:

```text
z = independent standard Normal values, centered and RMS-standardized
TMG = predictor_location + predictor_scale * z
```

The full factorial is:

- predictor location: `-1.5`, `0.0`, `1.5`;
- predictor scale: `0.5`, `1.0`, `2.0`;
- target prevalence: rare `0.05`, balanced `0.30`, common `0.65`;
- slope magnitude: weak `0.25`, moderate `0.75`, strong `1.50`.

There are 81 cells and four independent replicates per cell, for 324
communities per production block. For each species, slope sign is sampled with
equal probability and slope magnitude receives Normal jitter with standard
deviation `0.10 * magnitude`. The intercept is:

```text
normal_quantile(target_prevalence)
  - slope * predictor_location
  + Normal(0, 0.15)
```

This keeps prevalence interpretable at the predictor mean while inducing a
range of intercept/slope posterior correlations through non-centered designs.
The response is an independent Bernoulli draw from the probit probability.
There is no outcome filtering, rejection, enrichment, or difficulty selection.

Corpus order is the Cartesian product:

```text
predictor_location
predictor_scale
prevalence
effect
replicate
```

with levels in the order shown above and replicate varying fastest.

The 27-community disposable schedule uses all combinations of predictor
location, prevalence, and effect at predictor scale `1.0`, with one replicate.
Its metrics are plumbing diagnostics only.

## Fresh Frozen Seed Ledger

Repository, retained local evidence, and LUMI repository/run searches found
zero exact integer-token collisions for these roles:

| Role | Start | End | Count |
| --- | ---: | ---: | ---: |
| Correlation-head training | 211000001 | 211000324 | 324 |
| Fixed go/no-go validation | 212000001 | 212000324 | 324 |
| Reserved evaluation A | 213000001 | 213000324 | 324 |
| Reserved evaluation B | 214000001 | 214000324 | 324 |
| Reserved evaluation C | 215000001 | 215000324 | 324 |
| Disposable training | 291000001 | 291000027 | 27 |
| Disposable evaluation | 292000001 | 292000027 | 27 |

Model seed: `211900001`.

No separate stochastic seed block is permitted. Heldout-response, posterior
draw, and MCMC chain RNG streams must be deterministically derived with
`numpy.random.SeedSequence` from the owning community seed plus fixed protocol
tags, never from another free integer range.

Production training and fixed validation require:

`GENERATE_M56_CORRELATION_TRAIN_VALIDATION`

Opening all three reserved evaluation blocks requires a separate exact
confirmation:

`OPEN_M56_RESERVED_COVARIANCE_EVALUATION`

Frozen Whittaker and Big Spatial replay requires a third confirmation after
all simulation gates pass:

`OPEN_M56_FROZEN_REALDATA_REPLAY`

## Frozen Training Objective

Training is exactly:

- 100 epochs;
- batch size 9 communities;
- Adam learning rate `0.001`;
- no early stopping or checkpoint selection;
- deterministic shuffling from model seed `211900001`;
- no MCMC or real ecological outcomes.

For each species, the primary loss is the bivariate Gaussian negative log
probability of the known simulated truth Beta under the unchanged v0.1 mean,
unchanged calibrated marginal scales, and candidate correlation. The total
loss is:

```text
L = mean_bivariate_beta_nll + 0.01 * mean(delta_z^2)
```

There is no marginal NLL, coefficient MSE, coverage penalty, rank penalty,
predictive loss, correlation label, or validation-derived loss weight.

After epoch 100, weights and the training feature normalizer are frozen. The
212M block is evaluated once. It cannot alter weights, normalization, bounds,
loss, architecture, or thresholds.

## Fixed 212M Go/No-Go Validation

The reserved 213M-215M evaluation may be authorized only if every fixed
validation gate passes:

- exact bound release, package, checkpoint, weight, and calibration hashes;
- maximum absolute mean and marginal-scale delta versus calibrated v0.1 at
  most `1e-7`;
- all `scale_tril` entries finite, all covariance minimum eigenvalues above
  `1e-8`, and all absolute correlations at most `0.98`;
- checkpoint roundtrip maximum mean, scale, and correlation delta at most
  `1e-7`;
- 95% marginal coefficient coverage in `[0.925, 0.975]`;
- marginal rank-mean and rank-variance errors from `0.5` and `1/12` at most
  `0.025`;
- 95% two-dimensional Mahalanobis-ellipse coverage in `[0.925, 0.975]`;
- radial-rank mean and variance errors from `0.5` and `1/12` at most `0.025`;
- candidate/diagonal-v0.1 joint Gaussian NLL ratio at most `0.99`;
- candidate/raw-Laplace-correlation joint Gaussian NLL ratio at most `0.995`;
- aggregate mean absolute Fisher-z movement from raw Laplace at least `0.01`;
- heldout Brier and log-loss ratios versus diagonal v0.1 each at most `1.02`;
- all marginal and joint stratum gates below;
- immutable variable-v1 hash and provenance checks.

Failure closes Milestone 56 without opening 213M-215M.

## Reserved Evaluation Metrics

Each reserved block uses 256 neural posterior draws per community. For a truth
vector `b`, mean `mu`, and covariance `Sigma`, ellipse coverage uses:

```text
(b - mu).T @ inv(Sigma) @ (b - mu) <= 5.991464547107979
```

The radial rank is the fraction of 256 posterior draws whose squared
Mahalanobis radius from `mu` is below the truth radius.

Three correlation competitors use the same frozen means and marginal scales:

- diagonal v0.1: `rho = 0`;
- raw Laplace: `rho = rho_anchor`;
- candidate: learned `rho`.

Joint Gaussian NLL is averaged over communities and species. Candidate
correlation is also compared with qualified Python MCMC using Fisher-z RMSE,
with all correlations clipped to `[-0.98, 0.98]` before transformation.

Heldout predictive probabilities integrate coefficient uncertainty:

```text
linear_mean = x.T @ mu
linear_variance = x.T @ Sigma @ x
p = normal_cdf(linear_mean / sqrt(1 + linear_variance))
```

Heldout designs and responses are independently generated from each
community's truth using its deterministic protocol-tagged RNG stream.

## Exact MCMC Subsets

Each evaluation block runs Python-native HMSC MCMC on six fixed cells:

| Offset | Predictor location | Predictor scale | Prevalence | Effect |
| ---: | ---: | ---: | --- | --- |
| 1 | -1.5 | 0.5 | rare | weak |
| 105 | -1.5 | 2.0 | common | strong |
| 161 | 0.0 | 1.0 | balanced | moderate |
| 185 | 0.0 | 2.0 | rare | moderate |
| 241 | 1.5 | 0.5 | common | weak |
| 297 | 1.5 | 2.0 | rare | strong |

Exact seeds are:

- block A: `213000001`, `213000105`, `213000161`, `213000185`,
  `213000241`, `213000297`;
- block B: `214000001`, `214000105`, `214000161`, `214000185`,
  `214000241`, `214000297`;
- block C: `215000001`, `215000105`, `215000161`, `215000185`,
  `215000241`, `215000297`.

MCMC uses four chains, 500 retained samples per chain, 250 transient samples,
and thinning 1. MCMC correlation is the posterior sample correlation between
intercept and TMG Beta for each species. Chain seeds are deterministic
`SeedSequence` children of the community seed.

## Production Acceptance Gates

Every operational, marginal, joint, predictive, MCMC, stratum, and provenance
gate must pass. No aggregate may hide a failed block.

### Every evaluation block

- exact seed block, factorial balance, artifact hashes, and no target outcomes;
- mean/marginal-scale parity maximum delta at most `1e-7`;
- finite positive-definite covariance, absolute correlation at most `0.98`,
  and checkpoint/HDF5 roundtrip;
- marginal 95% coverage in `[0.925, 0.975]`;
- marginal rank-mean and rank-variance errors at most `0.025`;
- joint ellipse coverage in `[0.925, 0.975]`;
- radial-rank mean and variance errors at most `0.025`;
- candidate/diagonal joint-NLL ratio at most `0.99`;
- candidate/raw-Laplace joint-NLL ratio at most `1.00`;
- heldout Brier and log-loss ratios versus diagonal v0.1 at most `1.02`;
- heldout Brier and log-loss ratios versus qualified MCMC at most `1.10`;
- candidate/diagonal Fisher-z correlation RMSE ratio versus MCMC at most
  `0.90`;
- candidate/raw-Laplace Fisher-z correlation RMSE ratio versus MCMC at most
  `1.00`;
- correlation sign agreement at least `0.90` for MCMC correlations with
  absolute value at least `0.10`;
- all stratum gates.

### Aggregate across three blocks

- candidate/raw-Laplace joint-NLL ratio at most `0.995`;
- candidate/raw-Laplace Fisher-z correlation RMSE ratio at most `0.98`;
- at least two blocks have candidate/raw-Laplace joint-NLL ratio at most
  `0.995`;
- at least two blocks have candidate/raw-Laplace Fisher-z RMSE ratio at most
  `0.98`;
- mean absolute Fisher-z movement from raw Laplace at least `0.01`.

These improvement gates prevent zero movement or unchanged raw Laplace from
qualifying.

### Strata

Report marginal and joint coverage, radial rank, joint NLL, correlation
movement, and heldout proper scores by:

- predictor location;
- predictor scale;
- prevalence;
- effect magnitude;
- MCMC absolute-correlation bins `[0, 0.1)`, `[0.1, 0.3)`, and `[0.3, 0.98]`
  where MCMC is available.

For every non-MCMC stratum:

- marginal and joint coverage must be in `[0.90, 0.99]`;
- marginal and radial rank-mean error must be at most `0.05`;
- marginal and radial rank-variance error must be at most `0.04`;
- candidate/diagonal joint-NLL ratio must be at most `1.02`;
- candidate/diagonal Brier and log-loss ratios must be at most `1.02`.

No MCMC correlation bin may have candidate Fisher-z RMSE more than `1.05`
times raw Laplace.

### Operational

- correlation artifact schema and bound base hashes validate before inference;
- copied or packaged base weights and calibration remain byte-identical;
- 10,000-draw HDF5 empirical correlations differ from requested correlations
  by at most `0.03` in a fixed roundtrip fixture;
- fixed-shape public compatibility rejects changed dimensions, formula,
  coefficient order, distribution, traits, and random effects;
- mean inference latency and peak memory are each at most `2.0` times the
  corresponding v0.1 path on the same hardware;
- all existing v0.1 and variable-v1 hash regression tests pass.

## Real-Data Boundary

Only a complete simulated pass may authorize frozen Whittaker and Big Spatial
replay. The correlation head and normalizer remain unchanged. Real responses
cannot fit, select, calibrate, cap, or route correlation movement.

For both datasets:

- candidate/diagonal-v0.1 Brier and log-loss ratios must be at most `1.02`;
- candidate/qualified-MCMC Brier and log-loss ratios must be at most `1.10`;
- candidate Fisher-z correlation RMSE versus MCMC must not exceed raw Laplace;
- base mean and marginal-scale parity must remain within `1e-7`;
- all parity provenance and immutable artifact hashes must validate.

Both datasets must pass. Identity fallback is unavailable and would not count
as improvement.

## Artifact And Claim On Success

A complete simulated and real-data pass freezes a separate immutable artifact:

`neural_hmsc_fixed_probit_covariance_v1`

Its manifest must bind the unchanged v0.1 member, calibration, correlation-head
weights, feature normalizer, exact representation, qualification reports, and
MCMC provenance by SHA-256. `neural_hmsc_v0_1` remains unchanged and available
as the diagonal fallback.

The qualified claim would be limited to context-conditioned within-species
intercept/TMG covariance for the exact fixed 40-site/75-species probit scope.
It would not establish cross-species association, structural HMSC, full joint
posterior, or unconditional MCMC equivalence.

## Stop Rules

- The 291M-292M smoke validates plumbing only and cannot select any setting.
- The 212M result is a fixed go/no-go gate, not a tuning set.
- There is one candidate and no representation redesign under this protocol.
- No loss, bound, feature, architecture, seed, threshold, normalization, or
  comparator may change after 212M is opened.
- A failed 212M validation closes Milestone 56 without opening reserved seeds.
- The three evaluation blocks open together once, under their exact
  confirmation. Any failed block or aggregate gate is terminal.
- Real-data replay is forbidden unless every simulated gate passes.
- A zero-movement, diagonal, or raw-Laplace-equivalent result is not promotion.
- Failure retains `neural_hmsc_v0_1` and qualified Python MCMC as endpoints.

## Immediate Implementation Step

Implement only the correlation representation and sealed harness:

1. expose the full IRLS/Laplace covariance while preserving exact regression
   behavior of the existing marginal anchor;
2. add the frozen nine-feature correlation head and `scale_tril`
   reconstruction around a loaded immutable v0.1 checkpoint;
3. add mean/scale parity, positive-definite, initialization, sampling,
   HDF5/checkpoint roundtrip, compatibility, hash, and loss-formula tests;
4. implement separate `smoke`, `train-validate`, `evaluate`, and `realdata`
   commands with exact confirmation barriers;
5. run only the 291M-292M disposable smoke after tests pass.

Do not generate 211M-215M before the smoke implementation and all immutable
regression tests pass.
