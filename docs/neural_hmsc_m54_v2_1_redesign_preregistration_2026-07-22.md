# Neural-HMSC Milestone 54 v2.1 Redesign Preregistration

Date: 2026-07-22
Protocol: `neural_hmsc_variable_design_m54_v2_1`
Status: frozen before implementation or simulation generation

## Decision Boundary

This is the single representation-level redesign permitted after the
`neural_hmsc_variable_design_m54_v1_1` candidate failed its one-shot 103M
evaluation. It is not a calibration adjustment, post-hoc selector, or rerun of
the failed candidate. The opened 103M outcomes motivate the representation
class only; they may not select a weight, threshold, seed, checkpoint, or gate.

The old 104M-109M sensitivity blocks remain unopened and retired from active
Milestone 54 work. They cannot rescue or select a replacement for the failed
candidate. Existing `neural_hmsc_v0_1` and
`neural_hmsc_variable_probit_v1` artifacts remain immutable.

This protocol gets one disposable smoke and one production
train/calibrate/evaluate sequence. A failed production gate closes Milestone
54 and retains `neural_hmsc_variable_probit_v1` as the qualified endpoint.

## Scope

The supported family remains target-agnostic fixed-effect probit with:

- 12-128 sites;
- 2-100 species;
- 2-8 ordered numerical design columns;
- one required leading intercept;
- no traits, phylogeny, random effects, spatial effects, detection model, other
  likelihood, target-outcome routing, or dataset-specific calibration.

The redesign changes only the coefficient posterior-mean representation and
its training objective. The variable-design masks, coefficient-local shared
head, probit IRLS/Laplace anchor, posterior-scale head, split-conformal Beta
calibration, compatibility checks, and formula/name provenance remain.

## Frozen Representation

For each coefficient and species, define:

```text
anchor_mean      = probit_irls_laplace_mean(X, Y)
residual_mean    = anchor_mean + 0.5 * tanh(raw_residual)
support_gate     = sigmoid(raw_gate)
posterior_mean   = (1 - support_gate) * anchor_mean
                   + support_gate * residual_mean
posterior_scale  = max(anchor_scale * exp(clipped_log_scale), 1e-3)
```

The gate is coefficient/species-local and is produced by the shared head. Its
features are frozen to the existing local feature vector plus:

- `log1p(n_sites / n_covariates)` as a smooth global support ratio;
- `n_sites / max(n_covariates, 1)` clipped to `[1.5, 64.0]`, the exact ratio
  range implied by the supported shape corners;
- coefficient-local anchor standard deviation;
- coefficient-local design information divided by active covariate count;
- coefficient-local cross-design RMS;
- intercept indicator, prevalence summary, and active masks already present in
  the v1 representation.

The projection emits exactly three learned values per active
coefficient/species pair: bounded residual mean, clipped log-scale adjustment,
and gate logit. Zero-initialized residual and scale outputs preserve exact
anchor mean/scale at initialization regardless of the initial gate. The gate
must remain in `[0, 1]` by construction. Padding and site, species, and
non-intercept covariate permutation properties remain mandatory.

There is no hard support threshold, hand-authored low-support cap, post-fit
shrinkage grid, or dataset router. Support-dependent fallback is learned inside
the posterior model from independent simulated training evidence.

## Fresh Frozen Seeds

Repository search confirmed that the following production and disposable
ranges were unused when this protocol was frozen. Every block is contiguous,
contains 243 production communities or 27 disposable communities, and is
mutually disjoint.

### Production

| Purpose | Start | End | Count |
| --- | ---: | ---: | ---: |
| coefficient-posterior training | 111000001 | 111000243 | 243 |
| predictive auxiliary contexts | 112000001 | 112000243 | 243 |
| predictive heldout RNG partners | 113000001 | 113000243 | 243 |
| coefficient calibration | 114000001 | 114000243 | 243 |
| reserved fixed evaluation | 115000001 | 115000243 | 243 |

Model seed: `111900001`.

The 112M context at offset `i` is paired only with the 113M RNG seed at the
same offset. The 112M seed generates its latent coefficient truth and observed
context `(X_context, Y_context)`. Conditional on that frozen truth, the paired
113M seed generates an independent heldout `(X_score, Y_score)` with the same
shape, prevalence, effect, and design-condition stratum. The heldout response
is never an inference input or coefficient target.

Production training plus calibration requires exact confirmation:

`GENERATE_M54_V2_1_TRAIN_AUX_CALIBRATION`

The 115M evaluation cannot be generated until the resulting freeze validates.
It then requires separate exact confirmation:

`OPEN_M54_V2_1_RESERVED_EVALUATION`

### Disposable Smoke

| Purpose | Start | End | Count |
| --- | ---: | ---: | ---: |
| coefficient-posterior training | 191000001 | 191000027 | 27 |
| predictive auxiliary contexts | 192000001 | 192000027 | 27 |
| predictive heldout RNG partners | 193000001 | 193000027 | 27 |
| coefficient calibration | 194000001 | 194000027 | 27 |
| evaluation plumbing | 195000001 | 195000027 | 27 |

Smoke model seed: `191900001`. Smoke uses one epoch, batch size 9, and 32 SBC
draws. It runs no MCMC, opens no production seed, and provides no statistical
promotion evidence.

## Frozen Simulation Design

The 111M, 112M, 114M, and 115M community schedules each use the unchanged full
factorial from protocol v1.1:

- sites: 12, 40, 128;
- species: 2, 20, 100;
- covariates: 2, 5, 8;
- target design condition: 2, 10, 50;
- rare, balanced, and common prevalence once per base cell;
- weak, moderate, and strong coefficient scale once per base cell.

Each of the 81 shape/design cells occurs three times per corpus. The paired
113M heldout design preserves its 112M partner's declared stratum but is an
independent draw. No pool enrichment, filtering, rejection by outcome, or
near-boundary selection is permitted.

## Frozen Training Objective

Training remains 40 epochs, batch size 9, Adam learning rate `0.001`, and model
seed `111900001`. Each optimization step pairs one 111M coefficient batch with
one 112M/113M predictive batch after independent deterministic shuffling.

The coefficient term is unchanged:

```text
L_beta = Gaussian coefficient NLL + 0.25 * coefficient MSE
```

For each 112M posterior, heldout response probabilities are computed from the
posterior mean and diagonal scale using the probit Gaussian integral:

```text
linear_mean     = X_score @ mean
linear_variance = (X_score ** 2) @ (scale ** 2)
p_score         = Phi(linear_mean / sqrt(1 + linear_variance))
L_score = 0.5 * Bernoulli log loss(Y_score, p_score)
          + 0.5 * Brier(Y_score, p_score)
L_total = L_beta + 1.0 * L_score
```

All means are over active, unpadded observations or coefficients. Probability
clipping is fixed at `[1e-7, 1 - 1e-7]` for log loss. There is no gate-label,
103M-derived penalty, manual support target, post-fit gate tuning, or alternate
loss-weight candidate. The coefficient calibration remains one 95%
finite-sample split-conformal scalar fitted only on 114M after all weights are
frozen.

## Frozen Evaluation And Gates

The 115M evaluation uses 256 SBC draws. Qualified Python MCMC comparison is
fixed to the same six factorial schedule offsets as v1.1:

| Evaluation seed | Sites | Species | Covariates | Condition |
| ---: | ---: | ---: | ---: | ---: |
| 115000109 | 40 | 20 | 2 | 2 |
| 115000148 | 40 | 100 | 5 | 10 |
| 115000133 | 40 | 20 | 8 | 50 |
| 115000178 | 128 | 2 | 5 | 50 |
| 115000211 | 128 | 20 | 8 | 10 |
| 115000217 | 128 | 100 | 2 | 2 |

Every v1.1 gate remains unchanged:

- checkpoint mean/scale roundtrip maximum delta at most `1e-6`;
- exact factorial cell count three and complete marginal balance;
- overall 95% Beta coverage in `[0.925, 0.975]`;
- absolute rank-mean error from `0.5` at most `0.025`;
- absolute rank-variance error from `1/12` at most `0.025`;
- neural Beta RMSE no greater than `1.05` times anchor RMSE;
- aggregate neural Brier and log loss no greater than `1.02` times anchor;
- heldout neural/MCMC Brier and log-loss ratios no greater than `1.10`;
- every existing covariate-count, coefficient-role, site, species, and
  design-condition calibration stratum within its frozen coverage/rank limits;
- exact immutable baseline hashes and no target ecological outcome use.

The redesign adds three preregistered guards; none uses a threshold estimated
from 103M:

- Brier and log-loss ratios are each at most `1.02` separately for every site
  count and every active covariate count;
- aggregate Beta RMSE is at most `0.98` times anchor RMSE, so global identity
  fallback cannot count as a successful competitor;
- median support gate for `(128 sites, 2 covariates)` is no lower than median
  support gate for `(12 sites, 8 covariates)`, recorded on 115M before outcome
  scoring.

Gate distributions, mean movement, and proper scores must also be reported by
site count, covariate count, prevalence, effect scale, coefficient role, and
design condition. These reports cannot replace or average away a failed gate.

## Real-Data And Promotion Boundary

Only a complete simulated pass may open frozen Whittaker and Big Spatial
evaluation. The v1.1 real-data gates remain unchanged: each proper-score ratio
must be at most `1.10` versus qualified Python MCMC and may degrade by no more
than `0.02` versus its applicable frozen neural baseline. Real outcomes remain
evaluation-only and cannot fit or select the support gate.

A complete simulated and real-data pass freezes the distinct
`neural_hmsc_variable_design_probit_v2` artifact with a new checkpoint schema;
`neural_hmsc_variable_probit_v1` remains the fallback. Any production failure
is terminal for Milestone 54. It permits no loss-weight sweep, gate-cap search,
post-hoc calibration family, sensitivity rescue, or third representation.

## Immediate Implementation Step

Implement the three-output gated shared head, paired predictive auxiliary
training tensors, and fixed objective without generating any production seed.
Extend tests for exact anchor parity at initialization, gate bounds,
padding/permutation properties, paired-heldout independence, probability/loss
calculation, compatibility rejection, checkpoint roundtrip, and immutable
v0.1/v1 hash regressions. Then implement the sealed harness and run only the
191M-195M disposable smoke.
