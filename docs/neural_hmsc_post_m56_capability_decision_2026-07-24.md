# Neural-HMSC Post-Milestone 56 Capability Decision

Date: 2026-07-24

Decision status: resolved. No Milestone 57 model code, simulation corpus, seed
block, checkpoint, calibration artifact, or evaluation has been created.

## Decision

Permit one final fixed-scope joint-posterior attempt: a **conditional
bivariate Student-t posterior transport** that learns posterior location,
marginal uncertainty, within-species coefficient correlation, and tail weight
together.

This becomes Milestone 57. It is not an extension or redesign of Milestone 56.
Milestone 56 remains terminally closed, and its sealed `213M` through `215M`
blocks remain permanently unavailable.

The immutable endpoints remain:

- `neural_hmsc_v0_1` as the qualified fixed-shape neural release;
- `neural_hmsc_variable_probit_v1` as the qualified variable-shape neural
  release;
- qualified Python MCMC as the statistical reference and fallback.

Milestone 57 may begin only after a fresh artifact/seed audit and a separate
hash-frozen preregistration. This document does not authorize implementation or
simulation generation.

## Evidence Behind The Decision

Milestone 56 isolated within-species correlation while fixing the v0.1
posterior means and calibrated marginal standard deviations. Its 212M
validation established three relevant facts:

1. The implementation was operationally sound. Mean and marginal-scale parity
   were exact, covariance was positive definite, artifact roundtrip passed,
   and predictive Brier/log-loss ratios remained within `1.02`.
2. Correlation movement contained useful signal. Learned correlation reduced
   joint NLL by approximately half relative to raw Laplace correlation:
   candidate/raw-Laplace ratio `0.496555`.
3. Fixed v0.1 marginals were not adequate on the frozen factorial. Marginal
   coverage was `0.826955`, joint ellipse coverage was `0.735021`, and the
   candidate/diagonal-v0.1 joint-NLL ratio was `1.169488`.

The third result is decisive. A correlation-only representation cannot repair
marginal undercoverage. More Fisher-z tuning, correlation caps, selectors, or
post-hoc scale corrections would repeat the exhausted calibration loop rather
than address the failed representation.

At the same time, the large improvement over raw Laplace correlation means the
joint-posterior line is not disproven in principle. One representation that
learns marginals and dependence jointly is technically justified. The attempt
must be bounded because trait-Gamma, variable design, and fixed-marginal
covariance have already failed independent preregistered gates.

## Options Considered

| Direction | Decision | Reason |
| --- | --- | --- |
| Retune the Milestone 56 correlation head | Reject | Violates the terminal stop rule and cannot repair marginal coverage |
| Add another external scale or SBC calibrator | Reject | Returns to post-hoc calibration and separates marginals from dependence again |
| Free only the Gaussian Cholesky scales | Reject | Changes the failed overlay incrementally but retains thin Gaussian tails |
| Conditional normalizing flow | Defer | More expressive, but substantially harder to diagnose and preregister for a two-coefficient target |
| Mixture of Gaussians | Defer | Introduces component non-identifiability and unstable component-selection semantics |
| Conditional bivariate Student-t transport | Select | Minimal identifiable heavy-tailed family that learns location, marginal scales, correlation, and tails jointly |
| Stop neural joint-posterior work now | Retain as fallback decision | Becomes mandatory if the single Milestone 57 production attempt fails |

## Milestone 57 Capability Boundary

The candidate remains within the exact qualified fixed-shape family:

- 40 sites;
- 75 species;
- two ordered coefficients: `Intercept`, `TMG`;
- fixed-effect probit response;
- no traits, phylogeny, random effects, spatial effects, detection model, or
  cross-species posterior covariance.

For species `j`, the candidate posterior is:

```text
q(Beta_j | X, Y) = StudentT_nu_j(mu_j, L_j L_j.T)
```

The representation jointly predicts:

- a two-dimensional posterior location `mu_j`;
- two positive marginal scale parameters;
- one bounded within-species correlation;
- degrees of freedom `nu_j > 2`.

The exact distinction from Milestone 56 is that v0.1 means and scales are no
longer the candidate output. They may be immutable input anchors and mandatory
comparators, but the Student-t transport owns the complete two-coefficient
posterior. No separately fitted mean calibration, scale calibration,
correlation calibration, tail calibration, selector, or identity fallback is
allowed.

The output remains species-factorized. It does not represent residual
cross-species associations or the complete HMSC joint posterior.

## Representation Requirements For Preregistration

The fresh preregistration must freeze, before implementation:

- exact v0.1 release/member/checkpoint/calibration hashes used as anchors;
- exact Student-t parameterization and whether `L` is a scale or covariance
  factor;
- lower and upper numerical bounds for marginal scales, correlation, and
  degrees of freedom;
- all input summaries and their normalization;
- encoder/head architecture, initialization, optimizer, epoch count, and
  deterministic shuffling;
- one fixed multivariate Student-t log-score objective;
- any identity or anchor regularization and its fixed weight;
- posterior sampling, HDF5 representation, and predictive integration method;
- checkpoint schema, compatibility boundary, and artifact hash contract.

The primary loss must be a proper multivariate score for the full Student-t
posterior. Marginal coverage, ranks, or evaluation gates may not be optimized
post hoc. Real ecological outcomes and MCMC samples may not train, calibrate,
select, or tune the candidate.

## Evidence Design

The seed audit must establish entirely unused roles. No `211M` through `215M`
seed may be reused.

The production evidence must contain:

- a sufficiently large prior-predictive training corpus with the exact
  40-by-75-by-2 shape;
- independent fixed validation used once as a go/no-go gate;
- at least three untouched evaluation blocks;
- deterministic heldout-response streams derived from owning community seeds;
- a preregistered qualified Python MCMC subset in each evaluation block;
- frozen Whittaker and Big Spatial replay only after every simulated gate
  passes.

Training and validation must be balanced over prevalence, effect magnitude,
predictor location/scale, and design information. The design should include
repeated response realizations across matched parameter/context cells so
posterior spread and tail behavior are identifiable without target filtering.

Disposable smoke may validate plumbing only. It cannot select architecture,
scale bounds, degrees-of-freedom bounds, regularization, or thresholds.

## Mandatory Qualification Categories

Exact thresholds belong in the preregistration, but none of these categories
may be removed:

- immutable v0.1 and variable-v1 hash regressions;
- finite Student-t parameters, positive-definite scale matrices, and
  checkpoint/HDF5 roundtrip;
- aggregate and stratum marginal coverage and rank calibration;
- aggregate and stratum joint credible-region coverage and radial ranks;
- multivariate log score and energy score versus diagonal v0.1, raw Laplace,
  the failed M56 overlay, and qualified Python MCMC;
- location and marginal-scale accuracy versus truth and MCMC;
- Fisher-z correlation accuracy and sign agreement versus MCMC;
- heldout Brier/log-loss no degradation versus qualified neural and MCMC
  comparators;
- site/species permutation behavior and exact fixed-shape compatibility
  rejection;
- latency, memory, provenance, and seed-role validation;
- Whittaker and Big Spatial no-degradation with parity metrics attached.

Passing aggregate metrics may not hide a failed prevalence, effect,
location/scale, design-information, or MCMC-correlation stratum.

## Stop Rules

- Milestone 57 gets one preregistered representation and one production
  train-validation opening.
- There is no representation redesign, calibration-only retry, threshold
  change, or second production seed allocation.
- Any fixed-validation marginal, joint, proper-score, predictive, provenance,
  or stratum failure closes Milestone 57 before reserved evaluation.
- Reserved simulation, MCMC, and real-data roles require separate exact
  confirmations and remain sealed unless every preceding gate passes.
- Identity behavior, merely widening intervals, or merely matching v0.1 does
  not establish a new capability.
- A production failure closes neural joint-posterior development for the
  current fixed-shape scope. Future work then becomes maintenance,
  applicability detection, and automatic Python-MCMC fallback rather than
  another posterior family.

## Expected Outcome And Claim Boundary

Success would freeze a separate fixed-shape Student-t artifact that supports
calibrated heavy-tailed marginal uncertainty and within-species
intercept/TMG dependence. It could add one bounded joint-Beta row to the
Neural-HMSC capability matrix.

It would not be a near-equivalent implementation of HMSC MCMC. It would still
lack traits, phylogeny, latent factors, random effects, spatial effects,
cross-species associations, and the complete joint posterior. Python MCMC
would remain the statistical reference and automatic fallback.

Failure would produce a useful negative result: simulation-trained neural
fixed-effect inference remains qualified for its existing narrowed marginal
and predictive scope, but not for joint posterior replacement.

## Next Step

Audit the exact v0.1 and variable-v1 artifacts plus all retained local and LUMI
seed evidence. Then write and SHA-256 freeze a Milestone 57 qualification
preregistration with fresh unused seed blocks and exact representation,
training, validation, MCMC, real-data, and stop-rule semantics.

Do not implement the Student-t head or generate a disposable corpus before
both artifacts are frozen.
