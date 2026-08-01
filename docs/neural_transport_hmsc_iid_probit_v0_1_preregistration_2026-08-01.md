# Neural-Transport HMSC iid Probit v0.1 Preregistration

Date: 2026-08-01

Protocol: `neural_transport_hmsc_iid_probit_v0_1`

Decision: `preregister_exact_corrected_transport_before_implementation`

## Authorization Boundary

This document completes Milestone 79 documentation only. It does not authorize
model implementation, simulator execution, MCMC generation, scheduler
submission, or opening any seed.

The accepted Milestone 78 record is:

```text
docs/neural_transport_hmsc_milestone_78_acceptance_2026-08-01.md
sha256 = 85940ebda51bbb3e4892f931c9d3beec041afd1c80ccebe4ae35bdf072174f61
```

The fresh unopened seed audit is:

```text
docs/neural_transport_hmsc_iid_probit_v0_1_seed_audit_2026-08-01.json.md
sha256 = cf876a599ce93ed807dc8c939b7c3fca4b6168f716c71bc94d03fdeb0b227330
```

The implementation base is clean commit
`253e7802642192b0d72427b461bf9fc9cc30fa99` on
`feature/neural-transport-hmsc`.

## Scientific Question

Can a data-conditioned neural warm start and affine transport reduce warmup
and improve effective samples per second for one HMSC iid-probit structural
family while corrected HMC/Gibbs retains posterior behavior equivalent to the
ordinary Python sampler within frozen Monte Carlo tolerances?

The candidate is successful only if it passes both:

1. exactness, convergence, posterior-parity, and fallback gates; and
2. useful end-to-end efficiency gates after all exactness gates pass.

Posterior parity without acceleration is not a new neural capability.
Acceleration with posterior degradation is invalid.

## Claim If Successful

Success would support only this statement:

> A frozen neural warm start and affine transport improve sampling efficiency
> for a bounded 40-site, 12-species, two-factor iid-probit HMSC model while the
> Metropolis-corrected HMC/Gibbs output remains consistent with qualified
> Python MCMC under the preregistered diagnostics.

It would not support a standalone neural posterior, universal speedup, finite
sample exactness without convergence checks, full Neural HMSC, traits,
phylogeny, spatial effects, detection, variable shapes, random slopes,
Gaussian, Poisson, or MCMC replacement.

## Frozen Model Scope

| Item | Frozen value |
|---|---|
| response | probit occurrence |
| sites | 40 |
| species | 12 |
| fixed covariates | `Intercept`, standardized `x1` |
| trait design | intercept only; no measured traits |
| random levels | one iid site-level random intercept |
| random-level units | 40, one per site |
| latent factors | exactly 2 |
| adaptive factor count | disabled; `nfMin = nfMax = 2` |
| response mask | complete fitting response; deterministic 20% predictive mask used only for evaluation |
| numeric target dtype | float64 |
| neural encoder dtype | float32; outputs cast to float64 before transport |
| unsupported | every scope not listed above |

Compatibility must reject changed dimensions, formula, covariate order,
distribution, traits, phylogeny, random-level count/type, factor count,
spatial metadata, or prior hyperparameters before network execution.

## Frozen Generative Law

The implementation will provide a prior-predictive simulator for the bounded
native HMSC submodel. It must reproduce the following law, not reuse the failed
v1/v2 posterior target:

1. Draw raw `x1` from either standard Normal or log-Normal with Normal log scale
   `0.75`, then center and divide by sample standard deviation.
2. Set `X = [1, x1]` and trait design `T = [1]` for every species.
3. Use fixed-effect prior hyperparameters:
   - `mGamma = 0` with dimension 2;
   - `UGamma = 100 I_2` and `iUGamma = 0.01 I_2`;
   - `f0 = 4`; and
   - `V0 = I_2`.
4. Draw `iV` from the native Wishart prior and set `V = inverse(iV)`.
5. Draw `Gamma` from its native Normal prior and each species' `Beta` from the
   native `Beta | Gamma, V, T` prior.
6. Use iid random-level hyperparameters `nu = 3`, `a1 = 2`, `b1 = 1`,
   `a2 = 3`, and `b2 = 1`.
7. Draw two shrinkage increments `Delta`, their cumulative precision, and
   `Lambda` from the corresponding native marginal Student-t loading prior.
8. Draw `Eta` independently from standard Normal and form
   `U = Eta @ Lambda` and `R = Lambda.T @ Lambda`.
9. Draw `Y_ij ~ Bernoulli(Phi((X @ Beta + U)_ij))`.

Prior draws are accepted only when they enter their assigned factorial cell
within 2048 attempts. A failure is recorded; it cannot be replaced with a new
seed.

## Frozen Simulation Factorial

Every context block is balanced across 18 cells:

```text
covariate_shape   = {normal, right_skewed}
loading_strength = {weak, medium, strong}
prevalence       = {rare, moderate, common}
```

Cell definitions:

| Stratum | Frozen interval |
|---|---:|
| weak loading RMS | `[0.15, 0.35]` |
| medium loading RMS | `[0.50, 0.80]` |
| strong loading RMS | `[0.95, 1.30]` |
| rare expected prevalence | `[0.08, 0.20]` |
| moderate expected prevalence | `[0.30, 0.50]` |
| common expected prevalence | `[0.60, 0.78]` |

Loading RMS is `sqrt(mean(square(Lambda)))`. Expected prevalence is the mean
probit probability before drawing `Y`.

- disposable blocks: one context per cell;
- production training: six contexts per cell, 108 total;
- fixed validation: two contexts per cell, 36 total; and
- reserved evaluation: two contexts per cell, 36 total.

The response realization is `0`. The evaluation-only predictive mask uses the
existing deterministic stratified 20% cell-mask contract but is generated from
the context seed under a new protocol-specific namespace.

## Frozen Statistical Target And Kernel Composition

The statistical authority is the native Python HMSC target. The following
source files are immutable regression boundaries until an implementation
freeze explicitly replaces them:

```text
a7885c9123ac4e52beb1ed366fd5c09857f132789e21cac540be6c96663b8d52  pyhmsc/neural/generative_iid.py
fbafecc267a327f3097906b25f4543fbd3614ad20fb646643c3b07f1fc7aed17  hmsc/updaters/updateHMC.py
5f7f3074db6114022776ac2bc1e4431df75baa6d4ee3672de0a4c73a4fabc312  hmsc/gibbs_sampler.py
558e40a6e98639899588f56c42f7595b81c9e05c34467a87ec5502eca794ee7c  pyhmsc/neural/generative_iid_mcmc.py
```

The new simulator may reuse utilities, strata, masking, and fingerprints from
`generative_iid.py`, but its prior draws must match the native HMSC law above.

One candidate transition cycle is frozen as:

1. every fifth Gibbs sweep, perform one HMC transition over `Beta`, `Eta`, and
   `Lambda`, conditional on the remaining current state;
2. evaluate that transition in transported coordinates with the exact
   Jacobian;
3. retain HMC's Metropolis accept/reject correction; and
4. perform the unchanged native Gibbs sweep for latent response augmentation,
   `Beta/Lambda`, `Gamma/iV`, loading shrinkage, `Eta`, and other applicable
   fixed-scope parameters.

`Delta`, `Gamma`, and `iV` are not transported in v0.1. Adaptive factor-count
updates are disabled. Probit residual scale remains fixed according to the
native target.

Frozen HMC settings:

- transport frequency: every 5 Gibbs sweeps;
- leapfrog steps: 5;
- target acceptance: `0.80`;
- dual-averaging adaptation: first 80% of warmup only;
- no adaptation after warmup; and
- no learned transition during the Gibbs fallback path.

## Frozen Candidate And Controls

Four algorithms are mandatory:

| ID | Initialization | HMC transport | Purpose |
|---|---|---|---|
| `native_gibbs` | existing native initializer | disabled | statistical and operational baseline |
| `identity_hmc_gibbs` | existing native initializer | identity bijector | isolates the benefit of HMC from neural geometry |
| `neural_warmstart_gibbs` | frozen neural warm start | disabled | isolates initialization benefit |
| `neural_affine_hmc_gibbs` | frozen neural warm start | frozen neural affine bijector | only promotion candidate |

All four use four chains, at most 1000 warmup sweeps, 1000 retained draws, and
thin 1 for qualification. Random streams are paired by context and chain role,
but algorithms may consume them differently.

## Frozen Neural Representation

### Inputs

The network receives only compiled observed-model inputs and prior metadata:

- `Y`, observed-response mask, and deterministic fitting mask;
- `X` and ordered covariate metadata;
- `Pi` and random-level dimensions;
- dimensions, distribution, factor count, and frozen prior hyperparameters.

It cannot receive simulation truth, MCMC convergence diagnostics, heldout
responses, ecological outcomes, dataset identifiers, gate values, or algorithm
timings at inference.

### Encoder

Use a permutation-aware two-stream DeepSets encoder:

- site features: `x1`, observed richness fraction, observed-cell fraction;
- species features: observed prevalence, observed-cell fraction, and the
  normalized `x1`-presence cross moment;
- shared site MLP: two width-64 `swish` layers;
- shared species MLP: two width-64 `swish` layers;
- mean-pooled site and species context concatenated into a width-64 global
  context layer;
- no IDs, positional embeddings, attention, recurrence, dropout, batch
  normalization, or stochastic inference-time layer.

### Warm start

The species head predicts `Beta` location. Shared site/species rank-two heads
predict a `40 x 12` random-effect product. Its deterministic rank-two SVD gives
initial `Eta` and `Lambda`; singular values are descending, and each component
sign is fixed by making its largest-magnitude left-singular-vector entry
positive. Ties use the lowest index.

Only `Beta`, `Eta`, and `Lambda` native initial values are replaced. Every
other state block uses the existing initializer.

### Affine transport

For each transported block `b`:

```text
T_b(z_b; context) = location_b(context)
                    + exp(log_scale_b(context)) * z_b
```

`log_scale = 3 * tanh(raw_log_scale / 3)`, so every element lies in `[-3, 3]`.
The exact log absolute determinant is the sum of active element log scales.
Padded or inactive entries are forbidden in the fixed first scope.

The transport is frozen for the complete chain. It cannot inspect the current
state, acceptance history, diagnostics, or outcomes after initialization.

## Frozen Training Targets And Objective

Training references use only production-training contexts and their assigned
Python-MCMC chains.

For each context:

1. calculate posterior medians and MAD-derived robust scales for `Beta`;
2. calculate every draw's identifiable `U = Eta @ Lambda`;
3. calculate the median `U`, then apply the same deterministic rank-two SVD to
   obtain canonical center targets for `Eta` and `Lambda`;
4. apply deterministic SVD to each `U` draw and calculate MAD-derived canonical
   factor scales; and
5. floor target scales at `1e-3` for the logarithm only.

Teacher context eligibility requires four-chain maximum split R-hat `<= 1.05`
and minimum bulk ESS `>= 200` for every `Beta` coefficient and registered
upper-triangle association entry. At least 90 of 108 contexts must qualify.
Ineligible contexts remain recorded and are not replaced. Fewer than 90 closes
production training before fitting the encoder.

The normalized training objective is:

```text
L = Huber(beta_location, beta_median)
  + Huber(random_effect_location, median_U)
  + 0.25 * MSE(beta_log_scale, beta_target_log_scale)
  + 0.25 * MSE(eta_log_scale, eta_target_log_scale)
  + 0.25 * MSE(lambda_log_scale, lambda_target_log_scale)
```

Huber delta is `1.0`. Every term is the mean over its active elements. There is
no truth loss, ELBO, IWAE, SBC, coverage, rank, predictive, gate, timing,
ecological, or reserved-evaluation loss.

Frozen optimizer:

- Adam;
- 200 epochs;
- batch size 12 contexts;
- initial learning rate `1e-3`, cosine decay to `1e-5`;
- global gradient-norm clip `5.0`;
- network seed `719900001`;
- deterministic shuffle seed `719900002`;
- final epoch selected; no early stopping, checkpoint selection, ensemble, or
  hyperparameter sweep.

## Frozen Seed Roles

The seed-audit file is authoritative. Summary:

| Role | Range | Status |
|---|---|---|
| disposable training contexts | `791000001-791000018` | sealed |
| disposable training chains | `791100001-791100072` | sealed |
| disposable validation contexts | `792000001-792000018` | sealed |
| disposable validation chains | `792100001-792100072` | sealed |
| production training contexts | `711000001-711000108` | sealed |
| production training chains | `712000001-712000432` | sealed |
| fixed validation contexts | `713000001-713000036` | sealed |
| fixed validation paired chains | `714000001-714000144` | sealed |
| reserved evaluation contexts | `715000001-715000036` | sealed |
| reserved evaluation paired chains | `716000001-716000144` | sealed |
| Whittaker real-data chains | `717000001-717000008` | sealed |
| network/shuffle | `719900001-719900002` | sealed |

No role may borrow, replace, or recycle a seed. The retired 511M-515M blocks
remain forbidden. Fixed validation cannot open with production training.
Reserved evaluation and real data each require a separate later authorization.

## Disposable Gate

Disposable evidence is operational only. It must establish:

- exact 18-cell corpus fingerprints for training and validation;
- finite simulator, teacher, network, transform, Jacobian, target, gradient,
  and transition outputs;
- nonzero optimizer movement;
- checkpoint roundtrip and content hashes;
- direct versus transformed target equality;
- deterministic identity-transport transition equality;
- finite maximum-shape warm start and transport;
- successful forced-rejection fallback to ordinary Gibbs;
- all later seed flags false; and
- no model, architecture, objective, gate, threshold, or seed-role selection
  from disposable results.

A disposable failure closes the frozen candidate before production unless the
failure is conclusively a scheduler-only defect that occurred before any seed
or artifact opened. Numerical or model failures do not authorize a repair.

## Fixed Validation Gates

All gates are conjunctive and evaluated across aggregate results, all 18
factorial cells, four chains, and registered primary summaries.

### Convergence and numerical gates

- no non-finite value, invalid Jacobian, HMC divergence, or silent fallback;
- maximum split R-hat `<= 1.05` for `Beta`, `Gamma`, `V`, `Delta`, upper-triangle
  association `R`, and registered random-effect projections;
- minimum bulk ESS `>= 200` for the same summaries;
- post-warmup candidate HMC acceptance in `[0.50, 0.95]`; and
- all four algorithms complete under the same resource envelope.

### Posterior-parity gates

Candidate means and intervals are compared with `native_gibbs`:

- Beta mean normalized RMSE `<= 0.10` pooled posterior SD;
- Beta 95% interval-width ratio in `[0.90, 1.10]`;
- association-mean correlation `>= 0.95` and normalized RMSE `<= 0.15`;
- random-effect-product mean correlation `>= 0.95` and normalized RMSE
  `<= 0.15`;
- absolute aggregate 95% truth-coverage difference `<= 0.03` for Beta,
  association, and registered random-effect projections;
- absolute normalized-rank mean difference `<= 0.03` and rank-variance
  difference `<= 0.02` for the same targets;
- no factorial-cell coverage degradation greater than `0.05`;
- heldout Brier and log-loss ratios versus native Gibbs `<= 1.02`; and
- site-richness and species-prevalence predictive coverage differences
  `<= 0.05`.

The identity-HMC and warm-start-only controls must also pass all numerical and
posterior-parity gates. A control failure invalidates attribution and blocks
promotion.

### Efficiency gates

Efficiency is scored only after every preceding gate passes. End-to-end time
includes network loading, encoding, initialization, warmup, sampling, and
posterior writing.

- candidate median time to the frozen convergence target `<= 0.75` times
  `native_gibbs`;
- candidate median ESS/second across primary identifiable summaries `>= 1.25`
  times `native_gibbs`;
- no primary summary ESS/second below `0.90` times `native_gibbs`;
- candidate median ESS/second `>= 1.10` times `identity_hmc_gibbs`;
- candidate time to convergence `<= 0.90` times `neural_warmstart_gibbs`;
- peak memory `<= 1.50` times native Gibbs; and
- training cost and break-even compatible-dataset count reported separately.

Passing parity while missing any efficiency gate is a safe but failed neural
acceleration result. It does not permit threshold tuning or a more expressive
flow under this preregistration.

## Reserved Evaluation Gate

Reserved evaluation repeats every fixed-validation gate unchanged on 36 fresh
contexts. It opens only after a complete fixed-validation pass and independent
freeze/hash verification. There is no retraining, recalibration, architecture
change, threshold change, or checkpoint selection.

Promotion requires every reserved gate to pass. A single failure closes v0.1.

## Real-Data Confirmation

Real data opens only after reserved evaluation passes. Use the existing frozen
Whittaker 40-site training boundary, formula `presence ~ TMG`, the first 12
species in its frozen species order, no traits or phylogeny, and one unique iid
random-level unit per site. Use four native-Gibbs chains and four paired
candidate chains from the reserved 717M block.

The existing 12-site Whittaker holdout is evaluation-only. Real outcomes cannot
train, select, recalibrate, or set support. Required gates are posterior parity,
no predictive proper-score degradation above `1.02`, convergence, and at least
`1.10x` candidate ESS/second. Real data may block deployment but cannot rescue
a failed simulation gate.

## Artifact And API Contract

The future immutable artifact must contain:

- protocol and schema version;
- exact model support boundary and prior fingerprint;
- encoder and affine-transport architecture;
- weights file hash and complete content hash;
- training corpus and eligible-teacher manifest hashes;
- all source-file hashes and source commit;
- TensorFlow, TensorFlow Probability, NumPy, Python, and platform provenance;
- explicit `neural_posterior = false`;
- explicit `mcmc_target_corrected = true`;
- explicit identity and ordinary-Gibbs fallbacks;
- no calibration, selector, router, ensemble, external checkpoint, or
  ecological-outcome dependency; and
- immutable hashes for both existing qualified neural releases.

The public API must require explicit transport selection. The default Python
MCMC path remains unchanged until promotion. Every report must identify which
of the four algorithms ran and whether fallback occurred.

## Ordinary-Fixture Requirements Before Any Seed Opens

Milestone 80 must pass, without ledger seeds:

1. prior-predictive simulator law and 18-cell construction checks;
2. state pack/unpack and active-element contracts;
3. affine forward/inverse/Jacobian dense parity;
4. direct and transformed target equality in float64;
5. identity-transport deterministic transition equality;
6. finite gradients and transitions at the complete declared shape;
7. stationary-moment parity on analytic Gaussian and tiny probit targets;
8. site/species permutation and padding rejection behavior;
9. checkpoint roundtrip, tamper rejection, and incompatible-prior rejection;
10. forced neural failure followed by explicit ordinary-Gibbs fallback;
11. no change to default MCMC or either qualified neural release hash; and
12. a sealed harness whose token-free preflight proves every seed flag false.

Ordinary fixtures must use values below one million and must be listed in the
implementation evidence. Passing these tests permits only a separate
disposable authorization decision.

## Stop Rules

- No implementation begins until this preregistration, seed audit, acceptance
  record, roadmap bindings, and static tests are hash-frozen in a clean commit.
- No seed opens automatically after implementation.
- No post-hoc calibration, gate tuning, threshold tuning, architecture sweep,
  flow upgrade, diffusion model, teacher residual, selector, or target routing
  is permitted.
- A numerical/model disposable failure closes the candidate before production.
- A fixed-validation failure keeps reserved and real-data blocks sealed.
- A reserved-evaluation failure closes v0.1.
- Posterior parity without efficiency improvement is a documented negative
  result, not a qualified feature.
- One future representation redesign may be considered only outside v0.1 under
  a new branch, new preregistration, and fresh seeds; it is not implied by this
  roadmap.

## Next Authorized Step

Hash-freeze this preregistration, seed audit, acceptance record, roadmap, and
static seal tests in one clean commit. Then implement Milestone 80 using only
ordinary non-ledger fixtures. Do not generate simulation or MCMC corpora and do
not submit a scheduler until all Milestone 80 tests and implementation review
pass under a separate authorization decision.
