# Generative Neural-HMSC IID Probit v1 Preregistration

Date: 2026-07-27

Protocol: `generative_neural_hmsc_iid_probit_v1`

Status: design-frozen before implementation, simulation generation, fitting, or
evaluation.

Target branch: `feature/generative-neural-hmsc`

## Decision Boundary

This protocol opens a new structural model family. It is not an extension,
calibration, or repair of `neural_hmsc_v0_1`,
`neural_hmsc_variable_probit_v1`, or the failed Milestones 53-57.

The first candidate is limited to:

- binary probit responses;
- two ordered fixed effects, `Intercept` and one continuous covariate;
- one iid site random level;
- exactly two latent site factors;
- species-specific loadings;
- variable numbers of sites and species within the frozen support;
- an end-to-end neural variational posterior over the structural parameters.

The candidate may not use any v0.1 or variable-v1 posterior mean, scale, draw,
feature, calibration, checkpoint activation, or predictive ensemble member as
an input, target, initialization, residual anchor, fallback inside the
candidate, or training teacher. Those artifacts may appear only as separately
scored immutable comparators.

The candidate also may not use:

- IRLS or Laplace posterior outputs;
- MCMC posterior outputs as training targets;
- traits, phylogeny, spatial coordinates, temporal effects, detection, random
  slopes, or another response likelihood;
- post-hoc scalar, monotone, affine, OOD, prevalence, or stratum calibration;
- real ecological outcomes for training, architecture choice, threshold
  choice, routing, or shrinkage;
- a predictor-only success claim in place of posterior and association gates.

## Research Question

Can a permutation-equivariant neural inference network, trained directly
against an explicit fixed-effect-plus-iid-latent-factor generative model,
produce a useful amortized approximation to the identifiable posterior
summaries of that model?

The candidate succeeds only if it jointly demonstrates:

1. calibrated fixed-effect uncertainty;
2. calibrated latent random-effect contribution uncertainty;
3. recovery of residual species association;
4. non-degraded heldout predictive proper scores;
5. bounded agreement with an exact-model MCMC reference;
6. bounded behavioral agreement with qualified Python HMSC-HPC;
7. a material runtime advantage after amortized training.

Prediction alone is insufficient.

## Literature Basis And Deliberate Difference

The representation combines four established deep-learning ideas:

- amortized variational inference and reparameterized latent-variable
  training;
- an importance-weighted evidence bound rather than supervised regression to
  MCMC output;
- Deep-Sets-style permutation-equivariant aggregation;
- shared site/species message passing for variable community shapes.

Relevant starting references are:

- Deep Sets: <https://arxiv.org/abs/1703.06114>
- Set Transformer: <https://arxiv.org/abs/1810.00825>
- Towards a Neural Statistician: <https://arxiv.org/abs/1606.02185>
- Importance Weighted Autoencoders: <https://arxiv.org/abs/1509.00519>

The intended domain contribution is not another multi-output occurrence
network. It is a structured, exchangeable amortized posterior whose stochastic
state contains fixed effects, iid site factors, species loadings, and their
dependence. The old repository prototype computes a residual SVD and assigns
fixed scales. That prototype is retained only as historical plumbing and is
not an implementation template or qualification baseline.

## Exact Generative Family

For site `i = 1,...,N`, species `j = 1,...,S`, fixed-effect index
`k = 0,1`, and factor `h = 1,2`:

```text
alpha ~ Normal(-0.50, 0.85^2)
log(tau) ~ Normal(log(0.65), 0.45^2)

Beta[0,j] | alpha ~ Normal(alpha, 0.35^2)
Beta[1,j] ~ Normal( 0.00, 0.50^2)

Eta[i,h] ~ Normal(0, 1)
Lambda[h,j] | tau ~ Normal(0, tau^2)

R[i,j] = sum_h Eta[i,h] * Lambda[h,j]
z[i,j] = X[i,:] @ Beta[:,j] + R[i,j]
p[i,j] = Phi(z[i,j])
Y[i,j] ~ Bernoulli(p[i,j])
```

`Phi` is the standard Normal CDF. `X[:,0]` is exactly one. The continuous
column is centered and scaled using training-site covariates only; its stored
center and scale are part of the compiled input and artifact provenance.

The loading-scale prior is shared by the community and is inferred. The
candidate posterior state is:

```text
Theta = {alpha, Beta, Eta, Lambda, log(tau)}
```

The first candidate has no adaptive factor count and no loading shrinkage
process. Exactly two factors are used in simulation, neural inference, and the
exact-model MCMC comparator. Qualified Python HMSC-HPC is configured with
`nfMin = nfMax = 2` for the behavioral comparison, but its native loading
prior is not treated as the exact posterior reference.

## Supported Input Contract

The neural candidate accepts:

- `24 <= N <= 96` observed training sites;
- `12 <= S <= 75` species;
- exactly two ordered design columns;
- binary `Y`;
- a Boolean response-observation mask;
- a Boolean site mask and species mask for padding;
- one iid random-level unit per observed site;
- no missing covariates.

Site and species dimensions may vary inside a batch. Inputs are padded to the
batch maximum and all encoder, likelihood, posterior, loss, and diagnostic
operations must honor the masks.

Inputs outside this contract must fail before inference. In particular,
traits, phylogeny, coordinates, more than one random level, repeated group
codes, changed factor count, or changed likelihood may not silently use this
candidate.

## Permutation-Equivariant Encoder

The encoder receives only raw compiled tensors:

```text
X, Y, response_mask, site_mask, species_mask
```

It uses hidden width `64` and three alternating bipartite message-passing
rounds.

Initial edge features are:

```text
[Y_ij when observed else 0, response_mask_ij]
```

Initial site features are:

```text
[X_i0, X_i1, observed response fraction, observed prevalence]
```

Initial species features are:

```text
[observed prevalence, observed count / max(N, 1)]
```

Each message round applies shared two-layer `64 -> 64 -> 64` GELU MLPs:

```text
site_message_i =
  masked_mean_j edge_to_site(site_i, species_j, edge_ij)

species_message_j =
  masked_mean_i edge_to_species(site_i, species_j, edge_ij)
```

Site and species states are updated by residual MLP blocks followed by layer
normalization. A community token is the concatenation of masked means and
masked maxima of final site and species states. There are no learned site IDs,
species IDs, positional encodings, or shape-specific dense output matrices.

The implementation must be equivariant to independent site and species
permutations and invariant to padding. Tests compare posterior moments and
identifiable summaries after undoing the permutations.

## Frozen Posterior Family

The first candidate emits one joint low-rank multivariate Normal over the
unconstrained flattened state:

```text
q_phi(Theta | X, Y, masks)
  = Normal(mu, diag(d^2) + U @ U.T)
```

where:

- `Theta` is ordered as `alpha`, `Beta`, `Eta`, `Lambda`, `log(tau)`;
- every positive diagonal scale is `softplus(raw_scale) + 1e-4`;
- `U` has rank `16`;
- parameter-local shared heads emit the corresponding rows of `mu`, `d`, and
  `U`;
- a community head emits the `alpha` and `log(tau)` rows;
- masked padding rows are never part of `Theta`;
- sampling and `log q` use the matrix determinant lemma and Woodbury identity,
  without materializing a dense covariance matrix.

The coefficient head is shared across species and receives the species state,
community state, coefficient index embedding, and observed design-column
summaries. The Eta head is shared across sites and factors. The Lambda head is
shared across species and factors. Factor index embeddings have dimension
`8`; coefficient index embeddings have dimension `8`.

All local heads use:

```text
Dense(64, GELU)
Dense(64, GELU)
Dense(18, linear)
```

The 18 outputs are one mean, one raw diagonal scale, and 16 low-rank
coordinates. The pooled global head emits two rows with the same per-row output
shape, one for `alpha` and one for `log(tau)`.

This posterior is deliberately more expressive than independent marginal
Normals but remains a bounded first attempt. It can represent global linear
dependence across `alpha`, `Beta`, `Eta`, `Lambda`, and `log(tau)`; it is not
claimed to represent arbitrary multimodality.

## Factor Non-Identifiability

For any orthogonal `Q`:

```text
Eta @ Lambda = (Eta @ Q) @ (Q.T @ Lambda)
Lambda.T @ Lambda = (Q.T @ Lambda).T @ (Q.T @ Lambda)
```

Raw factor signs, ordering, and rotations therefore cannot be qualification
targets.

The primary identifiable quantities are:

```text
R = Eta @ Lambda
A = Lambda.T @ Lambda
C = diag(A)^(-1/2) @ A @ diag(A)^(-1/2)
```

`R` is the site-by-species random-effect contribution, `A` is residual
association covariance, and `C` is its correlation form. Diagonal
stabilization for `C` is exactly `max(A_jj, 1e-8)`.

For HDF5/API compatibility only, every posterior draw is gauge-fixed by:

1. eigendecomposing `Lambda @ Lambda.T`;
2. sorting rows by descending eigenvalue;
3. applying the same rotation to Eta and Lambda;
4. fixing each row sign so its largest-absolute loading is positive.

Gauge-fixed Eta/Lambda summaries are descriptive only. No raw Eta/Lambda RMSE,
coverage, or rank gate is permitted.

## Training Objective

Training maximizes an eight-sample importance-weighted variational lower bound:

```text
L_K =
  E[logmeanexp_k(
    log p(Y_obs | Theta_k, X)
    + log p(Theta_k)
    - log q_phi(Theta_k | X, Y_obs, masks)
  )]

K = 8
```

Only observed response cells contribute to the likelihood. Probabilities are
clipped to `[1e-6, 1 - 1e-6]` inside log likelihood evaluation.

There is no supervised truth loss, MCMC loss, v0.1 loss, response-affine loss,
coverage penalty, rank penalty, association penalty, or real-data loss.
Simulation truth is reserved for evaluation.

Optimization is frozen as:

- TensorFlow float32 model computation;
- float64 metric and exact-MCMC computation;
- AdamW;
- initial learning rate `3e-4`;
- cosine decay to `3e-5`;
- weight decay `1e-5`;
- global gradient norm clip `5.0`;
- batch size `4` communities;
- exactly `200` epochs;
- deterministic context shuffle under model seed `501900001`;
- linear likelihood/KL warm-up multiplier from `0.25` to `1.0` over epochs
  1-20, then exactly `1.0`;
- no early stopping and no validation-based checkpoint choice;
- final epoch weights are the candidate.

The warm-up applies to `log p(Theta) - log q(Theta)` and does not alter the
likelihood. Training aborts on non-finite loss, scale, covariance solve, or
gradient.

## Simulation Factorial

Each 324-context block contains two replicates of every cell in:

| Axis | Levels |
| --- | --- |
| observed sites | `24`, `40`, `96` |
| species | `12`, `36`, `75` |
| covariate shape | `normal`, `right_skewed` |
| loading-scale stratum | `weak`, `medium`, `strong` |
| expected community prevalence | `rare`, `moderate`, `common` |

This gives `3 * 3 * 2 * 3 * 3 * 2 = 324` owning contexts.

Covariate generation is:

- `normal`: standard Normal followed by exact sample centering/scaling;
- `right_skewed`: `exp(Normal(0, 0.75^2))` followed by exact sample
  centering/scaling.

Loading-scale strata are obtained by drawing from the frozen `log(tau)` prior
until:

| Stratum | Accepted tau |
| --- | --- |
| weak | `[0.15, 0.35]` |
| medium | `[0.50, 0.80]` |
| strong | `[0.95, 1.30]` |

Expected prevalence is the mean of `Phi(X Beta + Eta Lambda)` before response
sampling. The shared `alpha` makes community-level prevalence variation part
of the declared generative model rather than a post-hoc intercept shift.
Parameter draws are repeated through named child streams until:

| Stratum | Expected prevalence |
| --- | --- |
| rare | `[0.08, 0.20]` |
| moderate | `[0.30, 0.50]` |
| common | `[0.60, 0.78]` |

At most 512 parameter attempts are allowed per owning context. Failure to fill
a cell is an operational failure; ranges, bins, priors, or attempts may not be
changed after production authorization.

Each training context has two independently sampled response matrices from the
same `X` and `Theta`. Validation and evaluation contexts have one response
matrix.

## Holdout Protocol

Every validation/evaluation community has:

- an inference view with 20% of response cells hidden;
- a target view retaining those hidden outcomes;
- at least one observed and one hidden cell for every site and species.

The mask is generated independently of values, stratified by site and species,
and owned by a named RNG child. The neural and both MCMC comparators receive
the identical inference view.

Posterior predictive scoring uses:

1. masked-cell prediction, where site factors can be inferred from other
   species at the same site;
2. new-site prediction on a deterministic 20% site split, where a fresh Eta is
   drawn from its prior and integrated.

The masked-cell task is the primary association-aware predictive target.

## Comparators

### Exact-model MCMC

A separate TensorFlow Probability HMC/NUTS implementation uses the exact
generative model and priors in this document. It is the posterior-semantics
reference.

The fixed subset contains 36 validation contexts: each of the nine site/species
shape pairs crossed with:

- weak/rare;
- weak/common;
- strong/rare;
- strong/common.

Covariate shape alternates deterministically by shape-pair index, and replicate
zero owns the subset.

Each run uses:

- four chains;
- 1,000 warm-up iterations;
- 1,000 retained draws per chain;
- target acceptance `0.85`;
- chain seeds derived from the owning context;
- no neural initialization.

Every scalar non-gauge parameter and every registered invariant projection
must have split R-hat at most `1.05` and bulk ESS at least `200`. If not, one
preauthorized continuation of 1,000 warm-up-equivalent adaptation iterations
and 1,000 retained draws is allowed with the same chain states and seeds.
Failure after continuation invalidates the comparison; it does not count as a
neural pass or fail.

### Qualified Python HMSC-HPC

Qualified Python-native HMSC-HPC is fit to the same 36 contexts with:

- probit likelihood;
- formula `~ x1`;
- one iid site random level;
- `nfMin = nfMax = 2`;
- four chains, 1,000 transient, 1,000 retained, thin `1`;
- no traits or phylogeny.

Its native priors differ from the exact candidate prior. It is therefore used
for predictive behavior and association direction, not elementwise posterior
equivalence.

### Neural fixed-effect comparators

Two fixed-effect comparators are retained:

- a no-latent ablation of the new architecture, trained under the same 501M
  contexts and schedule with `R = 0`;
- immutable `neural_hmsc_v0_1` only for the matched 40-site, 75-species,
  two-coefficient cell.

The v0.1 artifact is loaded in a separate process after candidate inference.
Its values may not cross the candidate API boundary.

## Posterior Draw And Diagnostic Contract

Every scored candidate emits 256 posterior draws. Registered diagnostics are:

- marginal 50%, 80%, 90%, and 95% intervals for Beta and R;
- SBC ranks for every Beta element;
- SBC ranks for 16 fixed hash-derived projections of Beta;
- SBC ranks for 16 fixed hash-derived projections of R;
- SBC ranks for 16 fixed hash-derived off-diagonal projections of C;
- posterior mean and interval summaries for A and C;
- energy score for the concatenated 48 invariant projections;
- masked-cell and new-site Brier score and log loss;
- site richness and species prevalence posterior predictive intervals;
- posterior means and intervals for alpha and tau;
- final low-rank covariance condition diagnostics.

Projection vectors are Rademacher vectors generated from the SHA-256 of the
protocol string plus the projection family and index. They are normalized to
unit Euclidean norm and cannot be regenerated from evaluation outcomes.

## Fixed 502M Go/No-Go Gates

All gates are conjunctive.

### Operational

- all 324 factorial cells exist with exact metadata and seed ownership;
- all inference outputs are finite;
- checkpoint roundtrip changes posterior means/scales by at most `1e-6`;
- site and species permutation tests change unpermuted Beta, R, and C moments
  by at most `2e-5`;
- padding invariance changes those moments by at most `2e-5`;
- artifact dependency inventory contains no v0.1, variable-v1, IRLS, Laplace,
  MCMC, calibration, ensemble, or real-data artifact;
- all exact-model MCMC subset runs pass their diagnostics.

### Marginal calibration

- aggregate Beta 95% truth coverage is in `[0.925, 0.975]`;
- aggregate R 95% truth coverage is in `[0.90, 0.98]`;
- aggregate alpha and log(tau) 95% truth coverages are each in
  `[0.90, 0.99]`;
- every site-count, species-count, covariate-shape, loading-strength, and
  prevalence stratum has Beta 95% coverage in `[0.89, 0.99]`;
- the same strata have R 95% coverage in `[0.87, 0.995]`;
- aggregate Beta and R normalized rank means are within `0.04` of `0.5`;
- every registered marginal stratum rank mean is within `0.07` of `0.5`;
- aggregate normalized rank variances for Beta and R are each in
  `[0.060, 0.108]`;
- candidate/exact-MCMC median 95% interval-width ratios are in `[0.75, 1.35]`
  for both Beta and R.

### Joint and invariant calibration

- each of the Beta, R, and C projection families has aggregate 95% coverage in
  `[0.91, 0.985]`;
- each projection family has normalized rank mean within `0.05` of `0.5` and
  rank variance in `[0.055, 0.115]`;
- candidate/exact-MCMC invariant energy-score ratio is at most `1.10`
  aggregate and at most `1.20` in every one of the four MCMC regime groups;
- the fraction of posterior draws requiring covariance jitter above `1e-5` is
  at most `0.01`;
- no low-rank covariance condition estimate exceeds `1e8`.

### Association and random-effect recovery

On medium and strong loading contexts:

- median correlation between posterior-mean and true off-diagonal C is at
  least `0.65`;
- the 10th percentile of that correlation is at least `0.25`;
- median R posterior-mean RMSE is at most `0.85` times the fixed-effect
  ablation RMSE;
- candidate association RMSE is at most `1.15` times exact-model MCMC
  association RMSE;
- candidate and Python HMSC-HPC posterior-mean off-diagonal C correlation is
  at least `0.70` aggregate on the MCMC subset.

Weak contexts may not be hidden by aggregate gains:

- candidate masked-cell Brier and log loss may be no more than `1.01` times
  the fixed-effect ablation;
- mean absolute off-diagonal C may be no more than `0.05` above exact-model
  MCMC.

### Predictive and posterior predictive checks

- on medium/strong contexts, aggregate masked-cell Brier is at most `0.98`
  times the fixed-effect ablation;
- on medium/strong contexts, aggregate masked-cell log loss is at most `0.99`
  times the fixed-effect ablation;
- candidate/exact-MCMC Brier and log-loss ratios are each at most `1.10`
  aggregate and `1.20` in every shape, prevalence, and loading stratum;
- candidate/Python-HMSC-HPC Brier and log-loss ratios are each at most `1.10`
  on the 36-context subset;
- new-site Brier and log loss are no more than `1.03` times exact-model MCMC;
- 90% posterior predictive coverage for site richness and species prevalence
  is in `[0.84, 0.96]` aggregate and `[0.78, 0.99]` per registered stratum;
- on the matched 40-by-75 cell, Beta coverage and masked-cell proper scores do
  not degrade relative to immutable v0.1 by more than `0.02` absolute coverage
  or `1.03` score ratio.

### Runtime

- training completes within 24 `dev-g` GPU-hours;
- warm-cache candidate inference plus 256 draws at 96 sites and 75 species has
  median wall time at most 5 seconds and peak device memory at most 32 GiB;
- candidate inference is at least 20 times faster than exact-model MCMC on the
  same maximum-shape context, excluding candidate training time.

Failure of any 502M gate blocks 503M-505M, Python-HMSC real-data replay, and
promotion.

## Reserved Evaluation

The 503M, 504M, and 505M blocks repeat the complete factorial with untouched
parameters and responses. They may be opened only after a complete 502M pass
and an immutable candidate freeze.

The candidate must pass every 502M gate independently in all three blocks,
except MCMC comparison is run on the corresponding 36-context subset in each
block. Aggregate pooling cannot rescue a failed block.

Reserved evaluation A also adds a frozen null-association OOD appendix with
`Lambda = 0` under child streams of its 36 weak-regime MCMC subset. On this
appendix:

- masked-cell Brier and log loss may not exceed the fixed-effect ablation by
  more than 1%;
- mean absolute off-diagonal C may not exceed exact-model MCMC by more than
  `0.03`;
- no null-association claim is made outside this appendix.

## Frozen Real-Data Boundary

Real-data scoring opens only after all three reserved simulation blocks pass.

The sole first-family real-data workflow is the existing Whittaker plant split,
refit under a deliberately narrowed no-trait model:

```text
Y: presence/absence
X formula: ~ TMG
random level: iid site
factors: 2
traits: excluded
phylogeny: excluded
training sites: existing frozen 40-site split
heldout sites: existing frozen 12-site split
species: 75
```

The candidate, exact-model MCMC, Python HMSC-HPC, the no-latent ablation, and
immutable v0.1 receive identical fixed-effect covariates and response splits.
An additional outcome-blind masked-cell split of the 40 training sites is
derived from the candidate content hash before any Whittaker response is read.

Real-data gates are:

- candidate/exact-MCMC heldout-site Brier and log-loss ratios at most `1.10`;
- candidate/Python-HMSC-HPC heldout-site Brier and log-loss ratios at most
  `1.10`;
- candidate/no-latent masked-cell Brier and log-loss ratios at most `0.99`;
- candidate/exact-MCMC posterior-mean off-diagonal C correlation at least
  `0.70`;
- candidate/Python-HMSC-HPC posterior-mean off-diagonal C correlation at least
  `0.65`;
- no fixed-effect Beta 95% interval-width ratio outside `[0.70, 1.40]` against
  exact-model MCMC;
- all model applicability checks pass without target-specific routing or
  fallback.

Whittaker can veto deployment but cannot select a model, threshold, projection,
calibration, or redesign.

## Artifacts And Public API

The immutable candidate bundle must contain:

- protocol and preregistration SHA-256;
- seed-audit SHA-256;
- exact architecture configuration;
- training and model seeds;
- weights;
- optimizer-independent inference graph;
- X scaling contract;
- ordered parameter layout;
- posterior-family and rank metadata;
- simulator and prior schema;
- training-corpus manifest;
- dependency inventory proving absence of earlier neural posterior artifacts;
- validation and reserved-evaluation reports;
- exact-MCMC and Python-HMSC-HPC comparator provenance;
- source commit and environment inventory.

The public inference result must expose:

- structured posterior samples for alpha, Beta, Eta, Lambda, and tau;
- identifiable R, A, and C summaries;
- gauge-fix metadata;
- posterior predictive probabilities;
- applicability limits;
- a statement that inference is approximate and bounded to this family.

Legacy HDF5 output remains readable by `HmscFit`, but metadata must distinguish
gauge-fixed compatibility draws from identifiable association summaries.

## Stop Rule

The first candidate receives:

1. implementation and disposable 591M-592M smoke;
2. one 501M training opening;
3. one 502M fixed-validation opening;
4. 503M-505M only after a complete fixed-validation pass.

If 502M fails, one representation-level redesign may be preregistered against
the already-audited 511M-515M ledger. It must retain the same generative model,
scope, factorial, comparators, metrics, thresholds, and real-data boundary.
The redesign may change the posterior representation and encoder only; it may
not add calibration, supervised truth losses, MCMC teachers, or target
routing.

A second fresh production failure closes this iid family. No series of scale,
cap, selector, loss-weight, or stratum-specific repairs is permitted.

## Claim If Successful

A complete pass would support this statement:

> `generative_neural_hmsc_iid_probit_v1` is a GPU-amortized approximate
> posterior for a bounded two-factor iid latent probit JSDM. Within its frozen
> site/species/design support, it recovers calibrated fixed-effect and
> identifiable latent-association summaries, remains close to exact-model
> MCMC and qualified Python HMSC-HPC on preregistered diagnostics, and provides
> materially faster repeated inference.

It would not support:

- full HMSC equivalence;
- equality of complete neural and MCMC posterior distributions;
- raw Eta/Lambda identifiability;
- trait, phylogeny, spatial, temporal, detection, random-slope, Gaussian, or
  count-model claims;
- use outside the frozen dimension and prior-predictive support;
- replacement of Python HMSC-HPC as the general statistical reference.

Success creates one qualified structural neural family, not a near-universal
MCMC replacement.

## Seed Ledger

The machine-readable seed audit is:

`docs/generative_neural_hmsc_iid_v1_seed_audit_2026-07-27.json.md`

SHA-256:

`39e8763bf8a4fd525dc624570cd2f2b3392dbd1f62d7fa2e3c326f9340194cd6`

It records zero matching candidate or redesign seed tokens in the audited
local repository, retained local evidence, LUMI repository, and retained LUMI
run ledger before this preregistration was written.

## Immediate Authorization State

No simulation seed is open.

The only authorized next work after design review is implementation of:

- the simulator and exact priors;
- masked variable-shape tensor preparation;
- the bipartite encoder;
- the joint low-rank posterior math;
- the importance-weighted objective;
- exact-model MCMC reference;
- immutable artifact and sealed harness;
- unit and deterministic integration tests.

After those tests pass, only disposable 591M-592M may be opened.
