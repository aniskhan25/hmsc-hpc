# Generative Neural-HMSC IID Probit v2 Orbit Preregistration

Date: 2026-07-31

Protocol: `generative_neural_hmsc_iid_probit_v2_orbit`

## Purpose

This document freezes the single representation-level redesign permitted after
the v1 candidate failed fixed validation. The redesign targets the two failure
mechanisms demonstrated by the untouched 502M result:

1. a Gaussian in raw Eta/Lambda coordinates averaged across a continuous
   orthogonal posterior orbit and lost identifiable association signal; and
2. the one-shot amortized Gaussian was severely biased and underdispersed for
   Beta, R, alpha, and log(tau).

The redesign changes the posterior representation and encoder only. It does
not add calibration, supervised truth loss, an MCMC teacher, target routing,
gate tuning, or a new ecological model.

## Frozen Evidence

The inherited v1 preregistration is:

`docs/generative_neural_hmsc_iid_v1_preregistration_2026-07-27.md`

SHA-256:

`09c6a195ca139bdf168816b4f50db321c789bfdd061628e4f99a28cca81cea3f`

The candidate failure is:

`docs/generative_neural_hmsc_iid_v1_502m_failure_2026-07-30.md`

SHA-256:

`36f04ee135974f549e5544c33dc911f213fa0536c9ec902a2c71e0046c09bb91`

The redesign seed reaudit is:

`docs/generative_neural_hmsc_iid_v2_seed_reaudit_2026-07-31.json.md`

SHA-256:

`9a463943508651e74855701cdbd9870961efd3fd3c07a444674da36a67d49344`

All redesign seed roles remain unused. This preregistration opens none of them.

## Inherited Statistical Model

The complete v1 generative model remains unchanged:

- Bernoulli-probit observations;
- two fixed-effect coefficients per species;
- one shared community intercept hyperparameter alpha;
- one iid site random level with exactly two latent factors;
- site scores Eta with standard Normal prior;
- species loadings Lambda with loading scale tau;
- the same priors for alpha, Beta, Eta, Lambda, and log(tau);
- the same masked likelihood and deterministic new-site split;
- the same maximum support of 96 sites, 75 species, and two covariates.

The simulator, prior density, likelihood density, response realization
contract, masks, exact-model MCMC target, Python HMSC-HPC comparator, immutable
v0.1 comparator, and no-latent likelihood ablation remain byte-for-byte
identical unless a documented compatibility wrapper is required.

## Posterior Factorization

For active species, define the global state

```text
theta = [alpha, vec(Beta), log(tau)].
```

For active sites and species, combine the raw factors into

```text
Z = concat_rows(Eta, transpose(Lambda)),
```

so `Z` has shape `(n_sites + n_species, 2)`. The likelihood and priors are
invariant to the right action

```text
Z -> Z Q,  Q in O(2).
```

The redesigned posterior is

```text
q(theta, Z | X, Y, masks)
  = q_student(theta | encoder)
    q_orbit(Z | theta, encoder).
```

This is one joint conditional posterior. It is not a collection of
independently calibrated marginal heads.

## Global Student-t Block

`q_student(theta)` is one masked multivariate Student-t distribution with:

- encoder-produced mean;
- diagonal plus rank-16 covariance;
- one encoder-produced degrees-of-freedom value per dataset;
- degrees of freedom constrained to `[4, 30]`;
- exact masked sampling and log density using Woodbury and determinant-lemma
  algebra.

The active state order remains alpha, species-major Beta, then log(tau).
Padding is excluded from the density. The heavy-tailed family is trained
inside the generative objective and is not a post-hoc uncertainty scale.

## Orbit-Symmetrized Latent Block

Before symmetrization, the latent block is a matrix Normal

```text
q0(Z | theta) = MatrixNormal(M(theta), K, I_2),
K = diag(d^2) + U U^T,
```

where `U` has fixed rank 16. Rows of `M`, `d`, and `U` are emitted by shared
site/species heads. Site and species rows may covary through `K`.

The conditional mean uses a fixed permutation-invariant summary of sampled
theta:

```text
[alpha, log(tau),
 mean(Beta_intercept), mean(Beta_slope),
 second_moment(Beta_intercept), second_moment(Beta_slope)].
```

That summary enters row-local FiLM layers. Covariance parameters do not depend
on sampled theta.

The actual latent posterior averages the base density over the full
orthogonal orbit:

```text
q_orbit(Z | theta) = integral_O(2) q0(Z Q | theta) dQ.
```

Sampling first draws `Z0` from `q0`, draws a Haar-uniform `Q` from `O(2)`,
and returns `Z = Z0 Q`.

For exact log density, let

```text
C = transpose(Z) inverse(K) M.
r_plus^2  = (C11 + C22)^2 + (C21 - C12)^2.
r_minus^2 = (C11 - C22)^2 + (C21 + C12)^2.
```

The orbit integral is

```text
0.5 * [I0(r_plus) + I0(r_minus)],
```

where `I0` is the modified Bessel function. The implementation must use a
stable log-domain form. The remaining matrix-Normal terms use masked
Woodbury algebra. This gives an exact O(2)-invariant density for the frozen
two-factor family rather than selecting an arbitrary raw-factor orientation.

Raw Eta and Lambda draws are recovered from the site and species rows of Z.
R, A, and C remain the inferential targets. Gauge fixing remains compatibility
metadata only.

## Encoder Redesign

The v1 three-round mean-message encoder is replaced by four masked bipartite
cross-attention blocks:

- token width 96;
- four attention heads;
- feed-forward width 192;
- pre-layer normalization and residual connections;
- shared site and species parameters;
- no entity IDs, positional embeddings, or shape-specific output matrices;
- response-edge features formed only from X, masked Y, observation masks, and
  observed-count summaries;
- float64 accumulation for masked attention normalization and invariant
  pooling, with model projections allowed in float32.

Each block updates sites from species and species from sites. Community
features are masked symmetric means and maxima. Shared heads emit global,
coefficient-local, site-local, and species-local posterior parameters.

## Fixed Semi-Amortized Refinement

The encoder initializes posterior parameters. Both training and inference then
apply exactly four first-order refinement steps to:

- global mean and log diagonal scale;
- latent M and log diagonal scale.

Degrees of freedom and low-rank factors remain amortized and are not refined.
Each step uses the same eight-sample IWAE objective as training. Per-block
gradients are RMS-normalized and clipped to unit norm. Proposed step sizes are
fixed at:

```text
[0.05, 0.025, 0.0125, 0.00625].
```

Each proposal uses common random numbers and may be halved at most three
times. A proposal is accepted only if its common-random IWELBO is nondecreasing;
otherwise that block retains its previous value. Gradient values through the
inner update are stop-gradient, giving a first-order unrolled encoder.

This refinement optimizes the declared variational objective. It cannot read
truth, comparator output, a calibration target, or a later seed.

## Frozen Training Contract

The outer production contract remains:

- 324 owning training communities;
- two independent response realizations per community;
- 200 epochs;
- batch size four;
- eight IWAE samples;
- the existing KL warmup;
- AdamW, cosine learning-rate schedule, weight decay, and gradient clipping
  inherited from v1;
- candidate model seed `511900001`;
- final-epoch weights;
- a separately trained same-representation R=0 likelihood ablation.

The objective remains the probit log likelihood plus the exact unchanged prior
minus the redesigned joint log q. No supervised simulator truth enters
training.

## Implementation Feasibility Gates

Before any disposable or production seed opens, ordinary non-ledger fixtures
must prove:

1. orbit sampling preserves R, A, C, likelihood, and prior under random O(2)
   transforms;
2. analytic orbit log density matches 4096-point angular quadrature within
   `1e-6` on tiny float64 fixtures;
3. orbit log density is invariant within `1e-6`;
4. matrix-Normal low-rank sampling/log density matches a dense reference;
5. masked multivariate Student-t sampling/log density matches a dense
   reference;
6. the joint reparameterized gradient is finite through all four refinement
   steps;
7. accepted refinement steps never lower their common-random IWELBO;
8. site/species permutation and padding deltas are at most `2e-5`;
9. active state, masks, checkpoint roundtrip, and artifact tamper rejection
   are exact;
10. v1 checkpoints cannot load as v2 and immutable v1 hashes remain unchanged;
11. the exact-model log joint remains identical for identical raw states;
12. maximum-shape ordinary-fixture inference completes without dense
    `(state_dimension x state_dimension)` covariance materialization.

Failure of an orbit-density, gradient, invariance, or checkpoint gate closes
the redesign before any ledger seed is used. These are implementation
correctness checks, not substitutes for qualification metrics.

## Seed Roles and Authorization Sequence

The only permitted redesign ledger is:

- disposable training: `593000001-593000018`;
- disposable validation: `594000001-594000018`;
- production training: `511000001-511000324`;
- fixed validation: `512000001-512000324`;
- reserved A: `513000001-513000324`;
- reserved B: `514000001-514000324`;
- reserved C: `515000001-515000324`.

No additional seed range is permitted.

Authorization is sequential:

1. implement and pass ordinary-fixture feasibility gates;
2. separately authorize 593M-594M disposable smoke;
3. freeze implementation and disposable evidence;
4. separately authorize 511M candidate-plus-ablation training;
5. validate all training artifacts and hashes;
6. separately authorize sharded 512M fixed validation;
7. open 513M-515M only after every unchanged 512M gate passes.

Disposable scores cannot tune architecture, posterior family, refinement,
loss, schedule, gates, or thresholds. They may expose only implementation,
numerical, artifact, or scheduler defects.

## Unchanged Qualification Boundary

Every v1 metric, threshold, stratum, comparator, continuation rule, operational
gate, decision rule, and real-data boundary is inherited unchanged from the
v1 preregistration.

In particular, the redesign must still pass:

- marginal and registered-stratum coverage and ranks;
- invariant projection coverage and ranks;
- exact-MCMC interval-width and energy-score comparisons;
- association truth, exact-MCMC, and Python-HMSC comparisons;
- no-latent ablation comparisons;
- exact/Python proper scores;
- site-richness and species-prevalence PPCs;
- immutable-v0.1 matched-shape checks;
- permutation, padding, conditioning, finite-output, runtime, memory, and
  speed gates.

All gates are conjunctive. Runtime success cannot rescue statistical failure.
Identity/no-latent behavior is not improvement.

Fixed validation and exact/Python comparator execution must be sharded from
the outset. Partial shards cannot enter a decision.

## Prohibited Changes

The redesign may not use:

- v1 posterior outputs as candidate inputs or anchors;
- simulator truths in training loss;
- MCMC or Python-HMSC teachers;
- coefficient, predictive, or association calibration;
- post-fit scaling, caps, routing, fallback, or selection;
- target ecological outcomes;
- new metrics that replace a failed inherited gate;
- threshold, stratum, prior, likelihood, factor-count, or data-scope changes;
- another representation redesign after a fresh production failure.

## Stop Rule

If ordinary-fixture feasibility fails, close the iid redesign without opening
593M-594M.

If 512M fixed validation fails any inherited gate, close the generative iid
family permanently. Do not tune this representation and do not open
513M-515M or real data.

If all fixed and reserved simulation gates pass, run the unchanged Whittaker
no-trait iid-site replay. Whittaker may veto but cannot select or modify the
candidate.

## Claim If Successful

A complete pass would support the same bounded v1 claim, with the posterior
identified as an orbit-symmetrized, semi-amortized approximation. It would not
establish full HMSC equivalence or equality to MCMC.

## Current Authorization

This document authorizes no implementation run and opens no seed. After
hash-freeze and review, the next permissible task is implementation plus
ordinary non-ledger feasibility tests only.
