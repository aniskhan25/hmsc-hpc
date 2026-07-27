# Neural-HMSC Milestone 57 Student-t Qualification Preregistration

Date: 2026-07-24

Protocol: `neural_hmsc_fixed_probit_student_t_m57_v1`

Status: frozen before implementation, simulation generation, model fitting, or
evaluation.

## Decision Boundary

Milestone 57 is one final bounded attempt to qualify a within-species joint
fixed-effect posterior in the exact fixed-shape probit scope. It follows the
terminal closure of trait-Gamma, variable-design, and correlation-only
Milestone 56.

The governing capability decision is
`docs/neural_hmsc_post_m56_capability_decision_2026-07-24.md`, SHA-256:

`a1a7bc4a54eca4c78f6b32537f1afff662a524557accbd99d7267a28bc2cb2ba`

This protocol may not add or reopen:

- variable site, species, or design-column dimensions;
- Gaussian, Poisson, or another response likelihood;
- traits, phylogeny, latent factors, random effects, spatial effects, or
  detection;
- cross-species posterior covariance or ecological association semantics;
- target-outcome routing, MCMC teachers, ensembles, or mixture experts;
- separate mean, scale, correlation, tail, predictive, or post-fit
  calibration;
- a representation redesign after production validation;
- full-HMSC or unconditional MCMC-equivalence claims.

There is one representation, one production train-validation opening, and no
retry. Failure of any fixed 322M gate closes Milestone 57 without opening
323M-325M.

## Audited Immutable Inputs

The machine-readable artifact and seed audit is:

`docs/neural_hmsc_m57_artifact_seed_audit_2026-07-24.json.md`

Its SHA-256 is:

`1e1150a04cd17643db37988bfc010b611f8f49d638dbd40ead49cd5329b9b25c`

The audit validated every inventory member locally and on LUMI:

| Artifact | Frozen value |
| --- | --- |
| Fixed release ID | `neural_hmsc_v0_1` |
| Fixed release content | `affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8` |
| Fixed release manifest | `31ee489898e3657b97919803c0e850dc20494ef9118e9b963fe4a20365822e98` |
| Fixed package manifest | `d2daa81ec841390df59324a208216ffa0032ac514e6c679649d98815490bdbc7` |
| Bound member seed | `20260721` |
| Bound checkpoint manifest | `f62cd2217df6cc71cbe9f915c0cfbd3a3327b6684b3c5452bd9399aa130133a8` |
| Bound weights | `bb6e76d3ec9bc5e294ceac3051c3b2d7e5273db5053cfa5ceac676913d6265d9` |
| Bound calibration | `595fc0796d36802002cee09b270d53162f1fce100b83aecd32476e0958a0fd94` |
| Variable baseline ID | `neural_hmsc_variable_probit_v1` |
| Variable baseline content | `badaf8b8244cbd850693723147c6094bf196482237c52d2cc279ce42b286d2f9` |
| Variable checkpoint manifest | `cf46ebfdfc457e71a0da28f48f7709613f7e47b101b946553f711d5e1e4f47a5` |
| Variable weights | `70ef4548eeb1dc3a0d9367cb8edaedb5a2030370179241f35b372aecd8d5c4cd` |
| Variable calibration | `c3c8fd4ff50583ced5273c009e501ea0b6f400ff144a74f510513633edd7b771` |

The failed M56 overlay is retained only as a negative comparator:

- freeze:
  `c4fcb04cf1ebd7123be12144803de319ce1ff16a31e4fc5a1fb3e224f361a526`;
- manifest:
  `24f7eafa4a886afab94711bab77c56e76aef726fc93c0911c372b639bfa0121d`;
- weights:
  `66033d4f84cd443abf94053923e929180c0307fb08ac2a1bb9eaa75fe32ccde5`.

It is not a qualified baseline and cannot select or initialize the candidate.
Its sealed 213M-215M roles remain permanently unavailable.

## Exact Supported Scope

The candidate accepts only:

- 40 sites;
- 75 species;
- two ordered coefficients, `Intercept` then `TMG`;
- formula `~ TMG`;
- binary probit response;
- no traits, phylogeny, study design, random levels, coordinates, or detection
  inputs.

Changed dimensions, distribution, formula, coefficient order, structural
input, or missing v0.1 anchor provenance must fail before inference.

## Frozen Student-t Representation

For each species `j`, the candidate emits:

```text
q(Beta_j | X, Y) = StudentT_nu_j(mu_j, A_j)
```

where `A_j` is the Student-t scale matrix. The separately reported covariance
is:

```text
Sigma_j = nu_j / (nu_j - 2) * A_j
```

and is parameterized through posterior marginal standard deviations
`sigma_0`, `sigma_1` and correlation `rho`.

Let `mu_base` and `sigma_base` be the calibrated v0.1 posterior tensors. The
six raw head outputs are transformed exactly as:

```text
mu_k = mu_base_k + 2.0 * sigma_base_k * tanh(raw_mu_k)

sigma_k = sigma_base_k * exp(1.5 * tanh(raw_log_scale_k))

rho = 0.98 * tanh(raw_rho)

nu = 2.1 + 27.9 * sigmoid(raw_nu)
```

Thus:

- location movement is bounded to two base standard deviations per
  coefficient;
- each marginal standard-deviation multiplier lies in
  `[exp(-1.5), exp(1.5)]`;
- absolute correlation is below `0.98`;
- degrees of freedom lies in `(2.1, 30.0)`, so covariance exists.

The covariance Cholesky factor is:

```text
L_cov[0, 0] = sigma_0
L_cov[0, 1] = 0
L_cov[1, 0] = rho * sigma_1
L_cov[1, 1] = sigma_1 * sqrt(max(1 - rho^2, 1e-6))
```

The Student-t scale Cholesky factor used in the density is:

```text
L_t = sqrt((nu - 2) / nu) * L_cov
```

Sampling uses:

```text
z ~ Normal(0, I)
u ~ ChiSquare(nu)
Beta = mu + sqrt((nu - 2) / u) * L_cov @ z
```

The public posterior object must expose `mean`, marginal standard deviation,
`covariance_tril`, `student_t_scale_tril`, and `degrees_of_freedom`.
Ordinary HDF5 output remains sampled `Beta` draws in the existing
chain-by-draw-by-coefficient-by-species shape, with Student-t semantics and
artifact provenance in metadata.

## Frozen Fifteen-Feature Input

The shared species head receives exactly these ordered features:

1. calibrated v0.1 intercept mean;
2. calibrated v0.1 TMG mean;
3. log calibrated v0.1 intercept standard deviation;
4. log calibrated v0.1 TMG standard deviation;
5. eight-iteration IRLS/Laplace intercept mode;
6. eight-iteration IRLS/Laplace TMG mode;
7. log IRLS/Laplace intercept marginal standard deviation;
8. log IRLS/Laplace TMG marginal standard deviation;
9. raw Laplace Fisher-z correlation,
   `atanh(clip(rho_L, -0.979, 0.979) / 0.98)`;
10. observed-prevalence logit, with prevalence clipped to
    `[1e-4, 1 - 1e-4]`;
11. sample mean of TMG;
12. log sample standard deviation of TMG using `ddof=1`;
13. log condition number of the two-column design matrix;
14. normalized probit score for the intercept at the calibrated v0.1 mean;
15. normalized probit score for TMG at the calibrated v0.1 mean.

For coefficient `k`, the normalized score is:

```text
eta = X @ mu_base
p = clip(normal_cdf(eta), 1e-6, 1 - 1e-6)
phi = max(normal_pdf(eta), 1e-6)

score_k = sum_i X_ik * phi_i * (Y_i - p_i) / (p_i * (1 - p_i))

information_k =
  1.0 + sum_i X_ik^2 * phi_i^2 / (p_i * (1 - p_i))

normalized_score_k = score_k / sqrt(information_k)
```

All base and Laplace tensors are `stop_gradient` inputs. Features are
standardized with means and population standard deviations fitted only on the
321M training realizations. Standard deviations are floored at `1e-6`. The
normalizer is part of the candidate artifact.

## Frozen Neural Head

The head is shared across all communities and species:

```text
15 inputs
Dense(64, activation="relu")
Dense(64, activation="relu")
Dense(32, activation="relu")
Dense(6, activation=None)
```

Hidden kernels use TensorFlow seeded Glorot-uniform initialization under model
seed `321900001`; hidden biases are zero. The final kernel is zero. Final
biases are:

```text
[0, 0, 0, 0, 0, logit((10.0 - 2.1) / 27.9)]
```

Therefore the untrained candidate starts at:

- calibrated v0.1 location;
- calibrated v0.1 marginal standard deviations;
- zero correlation;
- degrees of freedom `10`.

Only this head is trainable. The v0.1 checkpoint, calibration, IRLS/Laplace
anchor, normalizer statistics after fitting, and all transforms are frozen.

## Frozen Simulation Design

Every context uses the exact 40-site, 75-species, two-coefficient probit
boundary. TMG is:

```text
z = independent standard Normal values
z = (z - mean(z)) / sqrt(mean(z^2))
TMG = predictor_location + predictor_scale * z
```

The factorial is:

- predictor location: `-1.5`, `0.0`, `1.5`;
- predictor scale: `0.5`, `1.0`, `2.0`;
- target prevalence: rare `0.05`, balanced `0.30`, common `0.65`;
- slope magnitude: weak `0.25`, moderate `0.75`, strong `1.50`;
- four independent parameter-context replicates per cell.

There are 81 cells and 324 owning context seeds per production block. For
each species:

```text
slope_sign ~ Uniform({-1, +1})
slope = slope_sign * (
  effect_magnitude + Normal(0, 0.10 * effect_magnitude)
)

intercept =
  normal_quantile(target_prevalence)
  - slope * predictor_location
  + Normal(0, 0.15)
```

The 321M training block creates two independent Bernoulli-probit response
realizations from each fixed `(X, Beta)` context, yielding 648 training
realizations. Both members of a pair remain in the same minibatch and may not
be split for validation. This repeated-response design identifies conditional
posterior spread and tail behavior without outcome filtering.

Validation and each reserved evaluation block use one observed response and
one independently derived heldout response per context. There is no rejection,
difficulty selection, prevalence filtering, or target enrichment.

Context order is:

```text
predictor_location
predictor_scale
prevalence
effect
context_replicate
```

with the last index varying fastest.

The disposable schedule contains all 27 combinations of predictor location,
prevalence, and effect at predictor scale `1.0`, with one context replicate.
Disposable training creates two responses per context; disposable evaluation
uses one.

## Frozen RNG Contract And Seed Ledger

Fresh owning seeds are:

| Role | Start | End | Contexts |
| --- | ---: | ---: | ---: |
| Training | `321000001` | `321000324` | 324 |
| Fixed validation | `322000001` | `322000324` | 324 |
| Reserved evaluation A | `323000001` | `323000324` | 324 |
| Reserved evaluation B | `324000001` | `324000324` | 324 |
| Reserved evaluation C | `325000001` | `325000324` | 324 |
| Disposable training | `391000001` | `391000027` | 27 |
| Disposable evaluation | `392000001` | `392000027` | 27 |

Model seed: `321900001`.

No additional free RNG range is permitted. Every stream is a
`numpy.random.SeedSequence` child of the owning context seed and these fixed
integer tags:

| Stream | Tag |
| --- | ---: |
| Context/X/Beta | `0x4D5701` |
| Observed response | `0x4D5702` |
| Heldout response | `0x4D5703` |
| Neural posterior draws | `0x4D5704` |
| MCMC chains | `0x4D5705` |
| Real-data replay draws | `0x4D5706` |

The second entropy item is the stream tag and the third, where needed, is the
zero-based response, draw, or chain index.

Real-data replay has no owning simulation seed. Its entropy is the eight
big-endian 32-bit words of:

```text
SHA256(candidate_manifest_bytes + b":" + dataset_name_utf8)
```

followed by stream tag `0x4D5706` and the zero-based draw index. The only
permitted dataset names are `whittaker` and `big_spatial`.

Exact confirmations are:

- train plus fixed validation:
  `GENERATE_M57_STUDENT_T_TRAIN_VALIDATION`;
- all three reserved simulation/MCMC blocks:
  `OPEN_M57_RESERVED_STUDENT_T_EVALUATION`;
- frozen Whittaker and Big Spatial replay:
  `OPEN_M57_FROZEN_REALDATA_REPLAY`.

## Frozen Training Objective

Training is exactly:

- 150 epochs;
- batch size 9 owning contexts, containing both response realizations;
- Adam learning rate `0.0005`;
- deterministic context-pair shuffling under model seed `321900001`;
- no early stopping, checkpoint selection, learning-rate schedule, gradient
  clipping, or weight decay;
- no MCMC samples or real ecological outcomes.

For `d=2`, known simulated truth `b`, location `mu`, degrees of freedom `nu`,
and Student-t scale matrix `A`, the loss per species is:

```text
delta = (b - mu).T @ inverse(A) @ (b - mu)

NLL =
  -lgamma((nu + d) / 2)
  +lgamma(nu / 2)
  +(d / 2) * log(nu * pi)
  +0.5 * log_determinant(A)
  +((nu + d) / 2) * log1p(delta / nu)
```

The total loss is the unweighted mean NLL across both responses, all species,
and all minibatch contexts. There is no marginal loss, coefficient MSE,
coverage/rank penalty, predictive loss, correlation label, anchor penalty, or
validation-derived weight.

After epoch 150, the head weights and feature normalizer freeze. The 322M
block is evaluated once and cannot change any artifact or threshold.

## Posterior And Predictive Evaluation

Simulation diagnostics use 512 deterministic Student-t posterior draws per
context. Marginal ranks are empirical fractions below truth. Joint radial rank
is the fraction of posterior draws whose squared Mahalanobis radius under the
reported covariance is below truth's radius.

Joint 95% coverage uses the species-specific analytic covariance-radius
threshold:

```text
threshold_95 =
  ((nu - 2) / nu) * 2 * F_quantile(0.95; df1=2, df2=nu)
```

The multivariate log score is the frozen Student-t NLL. The energy score uses
512 candidate draws and a second independently derived set of 512 candidate
draws:

```text
ES = mean(||draw - truth||) - 0.5 * mean(||draw_a - draw_b||)
```

Because a continuous-distribution NLL may be negative, log-score comparisons
never use raw ratios. For candidate score `S_c` and comparator score `S_b`, the
normalized delta is:

```text
normalized_log_score_delta =
  (mean(S_c) - mean(S_b)) / max(abs(mean(S_b)), 1e-6)
```

Negative values improve on the comparator.

Diagonal v0.1 and raw Laplace comparisons use 512 draws from their declared
Gaussian posteriors. The M56 negative comparator uses its frozen Gaussian
posterior where available.

For MCMC subsets, energy distance uses two independently derived candidate
sets of 512 draws. After deterministic chain-major concatenation, the first
512 retained MCMC draws form `mcmc_draw_a` and the next 512 form
`mcmc_draw_b`:

```text
ED(candidate, MCMC) =
  2 * mean(||candidate_draw_a - mcmc_draw_a||)
  - mean(||candidate_draw_a - candidate_draw_b||)
  - mean(||mcmc_draw_a - mcmc_draw_b||)
```

All means use index-matched deterministic pairs rather than an unbounded
all-pairs expansion. The same estimator and draw ordering are used for every
competitor.

Heldout prediction integrates coefficient uncertainty with the same 512
posterior draws:

```text
p = mean_draw normal_cdf(X_heldout @ Beta_draw)
```

No response-scale affine or predictive-only calibration is applied.

## Fixed 322M Go/No-Go Gates

Reserved evaluation remains sealed unless every gate below passes.

### Operational and provenance

- exact decision, audit, v0.1, variable-v1, and negative-M56 hashes;
- exact seed roles, factorial balance, pair grouping, and derived RNG streams;
- all parameters and draws finite;
- every covariance minimum eigenvalue above `1e-8`;
- absolute correlation below or equal to `0.98`;
- degrees of freedom in `[2.1, 30.0]`;
- checkpoint roundtrip maximum parameter delta at most `1e-7`;
- on a fixed `nu=10` fixture, 10,000-draw HDF5 empirical
  marginal-standard-deviation error at most `0.03` relative and correlation
  error at most `0.03`;
- changed dimensions, formula, coefficient order, distribution, traits, or
  random effects rejected before inference;
- immutable v0.1 and variable-v1 hash regressions.

### Aggregate calibration and recovery

- 95% marginal coefficient coverage in `[0.925, 0.975]`;
- 50% marginal coefficient coverage in `[0.475, 0.525]`;
- marginal rank-mean and rank-variance errors from `0.5` and `1/12` at most
  `0.025`;
- 95% joint credible-region coverage in `[0.925, 0.975]`;
- radial-rank mean and variance errors from `0.5` and `1/12` at most `0.025`;
- posterior-location RMSE no more than `1.05` times calibrated v0.1 mean RMSE;
- geometric mean marginal-width ratio versus calibrated v0.1 in `[0.80, 2.00]`;
- at most 10% of degrees-of-freedom values within `0.25` of either frozen
  bound;
- at most 10% of marginal-scale multipliers within `0.02` log units of either
  frozen bound.

### Proper scores and prediction

- candidate/diagonal-v0.1 normalized joint-log-score delta at most `-0.02`;
- candidate/raw-Laplace normalized joint-log-score delta at most `-0.05`;
- candidate/failed-M56 normalized joint-log-score delta at most `-0.05`;
- candidate/diagonal-v0.1 energy-score ratio at most `0.99`;
- candidate/failed-M56 energy-score ratio at most `0.99`;
- heldout Brier and log-loss ratios versus diagonal v0.1 each at most `1.02`.

### Non-MCMC strata

Report all calibration, recovery, proper-score, prediction, scale-multiplier,
correlation, and degrees-of-freedom diagnostics by:

- predictor location;
- predictor scale;
- prevalence;
- effect magnitude;
- intercept versus TMG coefficient.

For every stratum:

- 95% marginal and joint coverage in `[0.90, 0.99]`;
- marginal and radial rank-mean error at most `0.05`;
- marginal and radial rank-variance error at most `0.04`;
- candidate/diagonal normalized joint-log-score delta at most `0.02`;
- candidate/diagonal energy-score ratio at most `1.02`;
- candidate/diagonal Brier and log-loss ratios at most `1.02`;
- location RMSE no more than `1.10` times calibrated v0.1.

Any failure closes Milestone 57. A broad interval expansion that restores
coverage but fails log score, energy score, rank, width, prediction, or stratum
gates cannot advance.

## Exact Reserved MCMC Subsets

Each evaluation block runs qualified Python MCMC on six fixed offsets:

| Offset | Location | Scale | Prevalence | Effect |
| ---: | ---: | ---: | --- | --- |
| 1 | `-1.5` | `0.5` | rare | weak |
| 105 | `-1.5` | `2.0` | common | strong |
| 161 | `0.0` | `1.0` | balanced | moderate |
| 185 | `0.0` | `2.0` | rare | moderate |
| 241 | `1.5` | `0.5` | common | weak |
| 297 | `1.5` | `2.0` | rare | strong |

Exact owning seeds are:

- block A: `323000001`, `323000105`, `323000161`, `323000185`,
  `323000241`, `323000297`;
- block B: `324000001`, `324000105`, `324000161`, `324000185`,
  `324000241`, `324000297`;
- block C: `325000001`, `325000105`, `325000161`, `325000185`,
  `325000241`, `325000297`.

MCMC uses four chains, 500 retained draws per chain, 250 transient draws,
thinning 1, and chain RNG children under tag `0x4D5705`.

## Reserved Evaluation Gates

Every block must independently pass all 322M operational, calibration,
recovery, proper-score, predictive, and non-MCMC stratum gates.

Additionally, for every block:

- candidate/MCMC heldout Brier and log-loss ratios each at most `1.10`;
- candidate posterior-location RMSE versus MCMC no more than `0.95` times
  calibrated v0.1;
- candidate marginal-standard-deviation RMSE versus MCMC no more than `0.90`
  times calibrated v0.1;
- candidate Fisher-z correlation RMSE versus MCMC no more than `0.90` times
  diagonal v0.1 and no more than `0.95` times raw Laplace;
- correlation sign agreement at least `0.90` where absolute MCMC correlation
  is at least `0.10`;
- candidate-to-MCMC energy distance no more than `0.90` times diagonal v0.1;
- no MCMC absolute-correlation bin `[0, 0.1)`, `[0.1, 0.3)`, or
  `[0.3, 0.98]` has candidate Fisher-z RMSE more than `1.02` times raw
  Laplace.

Across all three blocks:

- candidate/diagonal normalized joint-log-score delta at most `-0.03`;
- candidate/diagonal energy-score ratio at most `0.98`;
- candidate-to-MCMC energy-distance ratio at most `0.88`;
- candidate/diagonal Fisher-z RMSE ratio versus MCMC at most `0.88`;
- at least two blocks satisfy each corresponding improvement threshold;
- every aggregate and stratum calibration gate passes.

The three blocks open together once. Any block or aggregate failure is
terminal.

## Real-Data Boundary

Real-data replay is forbidden unless every simulated and MCMC gate passes.
The head, normalizer, bounds, and all thresholds remain unchanged. Real
responses cannot fit, calibrate, select, shrink, or route parameters.

Whittaker and Big Spatial must each report:

- candidate/diagonal-v0.1 Brier and log-loss ratios at most `1.02`;
- candidate/qualified-MCMC Brier and log-loss ratios at most `1.10`;
- candidate-to-MCMC energy distance no greater than diagonal v0.1;
- posterior-location and marginal-scale RMSE versus MCMC no greater than
  calibrated v0.1;
- Fisher-z correlation RMSE versus MCMC no greater than raw Laplace;
- immutable Python/R parity metrics and provenance;
- exact artifact hashes and zero target-outcome fitting or selection.

Both datasets must pass. There is no identity fallback.

## Runtime Gates

On identical hardware:

- mean parameter-inference latency at most `2.5` times v0.1;
- peak parameter-inference memory at most `2.0` times v0.1;
- 512-draw generation latency reported separately and finite;
- GPU and CPU sampling agree within Monte Carlo tolerance on the fixed
  roundtrip fixture.

## Artifact And Claim On Success

A complete simulated, MCMC, and real-data pass freezes:

`neural_hmsc_fixed_probit_student_t_v1`

The immutable manifest binds the decision, audit, preregistration, v0.1
anchor, variable-v1 regression, head weights, feature normalizer, Student-t
semantics, qualification reports, MCMC provenance, and real-data parity by
SHA-256.

The claim is limited to a calibrated heavy-tailed two-coefficient posterior
with within-species intercept/TMG dependence for the exact fixed-shape probit
scope. It does not establish cross-species association, structural HMSC, or
full joint-posterior MCMC equivalence.

## Stop Rules

- No implementation or simulation precedes this frozen protocol.
- Disposable 391M-392M smoke is plumbing evidence only.
- There is one architecture, one objective, one candidate, and no redesign.
- Production 321M-322M and reserved 323M-325M require separate exact
  confirmations.
- Any 322M gate failure closes Milestone 57 before MCMC or real data.
- Any reserved block or aggregate failure closes Milestone 57 before real
  data.
- Any real-data failure retains v0.1 and Python MCMC without promotion.
- No post-hoc calibration, cap, selector, threshold adjustment, new seed
  block, or calibration-only retry is permitted.
- Production failure closes neural joint-posterior development for this
  fixed-shape scope. Subsequent work is limited to maintenance, applicability
  detection, and automatic Python-MCMC fallback.

## Immediate Implementation Step

Implement only:

1. the Student-t posterior object, exact density, sampling, covariance, and
   HDF5 metadata;
2. the frozen fifteen-feature extractor and shared six-output head;
3. the immutable checkpoint/normalizer/provenance schema bound to v0.1;
4. exact math, initialization, gradient, covariance, empirical sampling,
   roundtrip, compatibility, hash, seed, and confirmation-barrier tests;
5. separate `smoke`, `train-validate`, `evaluate`, `realdata`, and
   read-only-validation commands.

After tests pass, run only the 391M-392M disposable smoke. Keep 321M-325M
unopened.
