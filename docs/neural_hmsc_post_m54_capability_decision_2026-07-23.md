# Neural-HMSC Post-Milestone 54 Capability Decision

Date: 2026-07-23

Decision status: resolved into a fresh frozen preregistration. No
implementation, training corpus, calibration corpus, or evaluation seed has
been generated.

## Decision Boundary

Milestone 54 is terminally closed after its single permitted representation
redesign failed the one-shot 115M evaluation. Trait-Gamma v1 is also closed,
and iid/spatial qualification remains blocked. The next milestone must
therefore remain inside the already-qualified fixed-shape fixed-effect probit
scope and must not reopen:

- variable-design generalization;
- traits, phylogeny, random effects, spatial effects, or detection;
- Gaussian or Poisson qualification;
- response-mean or post-hoc scale calibration searches;
- target-outcome routing;
- a full-HMSC or unconditional MCMC-equivalence claim.

The immutable regression baselines remain:

- `neural_hmsc_v0_1`, content SHA-256
  `affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8`;
- `neural_hmsc_variable_probit_v1`, content SHA-256
  `badaf8b8244cbd850693723147c6094bf196482237c52d2cc279ce42b286d2f9`;
- predictive-only `neural_predictive_affine_v1`;
- qualified Python MCMC as the statistical reference and fallback.

## Options Considered

| Option | Expected value | Decision |
| --- | --- | --- |
| Qualify Gaussian or Poisson | Expands likelihood coverage but leaves the current probit joint-posterior gap unresolved | Defer |
| Add another predictive-mean or uncertainty calibrator | Re-enters the exhausted calibration/selection loop without changing posterior representation | Reject |
| Add only an applicability certificate and MCMC fallback | Useful operational maintenance, but identity fallback is not a new statistical capability | Keep as maintenance, not the next research milestone |
| Reopen variable-design or trait structure | Violates terminal stop rules | Reject |
| Add within-species full covariance to fixed-shape probit Beta | Directly addresses a missing posterior semantic while preserving the qualified family and exact shape | Select for preregistration |

## Selected Capability

Milestone 56 will attempt a **correlation-only, covariance-aware fixed-shape
probit Beta posterior**.

The candidate must retain the exact fixed-shape v0.1 model boundary used by the
qualified checkpoint:

- 40 sites;
- 75 species;
- two ordered coefficients, including the leading intercept;
- fixed-effect probit response;
- no structural HMSC terms.

The fresh preregistration must bind one exact packaged v0.1 checkpoint and its
coefficient-calibration artifact by SHA-256. The candidate must preserve that
checkpoint's posterior mean and calibrated marginal standard deviations
numerically, with a preregistered parity tolerance. It may add only a
per-species `2 x 2` positive-definite coefficient covariance.

The intended representation is:

1. recover the full `2 x 2` IRLS/Laplace covariance currently reduced to its
   diagonal by the qualified probit anchor;
2. convert its off-diagonal term to a bounded correlation anchor;
3. use a small context-conditioned neural head to predict a bounded residual
   correction to that correlation;
4. reconstruct `scale_tril` from the unchanged calibrated marginal scales and
   the corrected correlation;
5. emit correlated Beta draws through the existing `BetaPosterior`,
   `sample_beta_posterior`, HDF5, and `HmscFit` paths.

There is no cross-species covariance. In a fixed-effect model without latent
species factors, the bounded target is posterior dependence between the two
coefficients within each species. Species association, latent co-occurrence,
and structural joint-posterior semantics remain outside scope.

The existing full-covariance head, negative log probability, sampling,
calibration, and storage code is implementation scaffolding only. It has never
been qualified for probit and may not be treated as evidence or silently
promoted.

## Why This Is The First Choice

The qualified v0.1 release already passes marginal coefficient SBC and bounded
real-data prediction gates, but its checkpoint advertises a diagonal Normal
Beta posterior. That leaves coefficient correlation and multivariate credible
regions unrepresented.

The selected capability is narrow enough to isolate:

- means cannot improve or degrade because they remain frozen;
- marginal uncertainty cannot be rescued post hoc because calibrated standard
  deviations remain frozen;
- any claimed gain must come from learned covariance;
- a zero-correlation or unchanged-Laplace result cannot count as improvement;
- the existing release remains a byte-identical fallback.

This is more informative than another response calibrator and substantially
less risky than changing likelihood, design dimensionality, or structural HMSC
terms.

## Required Fresh Preregistration

No implementation may begin until a separate Milestone 56 preregistration
freezes all of the following:

### Artifact and representation

- exact v0.1 member/checkpoint and calibration hashes;
- exact formula, coefficient names, dimensions, and compatibility boundary;
- correlation-anchor calculation and numerical stabilization;
- neural correlation-head inputs, architecture, output bound, and
  initialization;
- whether the head predicts Fisher-z residuals or another fixed
  positive-definite parameterization;
- training loss and every fixed loss weight;
- checkpoint schema and provenance fields;
- exact mean and marginal-scale parity tolerance.

### Evidence roles

- repository-wide audit proving all proposed seed blocks are untouched;
- disjoint simulation training, optional correlation calibration, fixed
  evaluation, and disposable-smoke roles;
- at least three independent production evaluation blocks;
- an untouched qualified Python MCMC subset in every production block;
- real ecological outcomes unavailable to fitting, selection, or threshold
  choice.

### Mandatory gates

- byte-identical v0.1 base weights and calibration artifacts;
- posterior-mean and marginal-standard-deviation parity to the bound v0.1
  checkpoint;
- finite, positive-definite covariance and checkpoint/HDF5 roundtrip;
- unchanged marginal coverage and rank gates;
- preregistered two-dimensional credible-ellipse coverage and multivariate
  rank diagnostics;
- covariance/correlation error versus qualified Python MCMC;
- a joint proper score, such as multivariate log score or energy score, that
  materially improves over diagonal v0.1;
- no degradation versus the raw IRLS/Laplace correlation anchor;
- heldout Brier and log-loss no degradation versus v0.1 and the existing MCMC
  bound;
- Whittaker and Big Spatial no-degradation only after every simulation gate
  passes;
- runtime, memory, API compatibility, provenance, and immutable-baseline
  regressions.

Exact numerical thresholds, aggregation rules, and stratum gates must be
declared before seeds are generated. Aggregate improvement may not hide a
failing species, prevalence, effect-size, or design-information stratum.

## Stop Rules

- One fixed candidate and at most one representation-level redesign are
  permitted.
- Disposable smoke may validate plumbing only and may not select a threshold,
  loss, correlation bound, or architecture.
- Production training and evaluation require separate exact confirmations.
- A zero-correlation identity result is safety, not improvement.
- A candidate that only reproduces the raw Laplace correlation is not a neural
  improvement.
- Any marginal SBC, proper-score, provenance, or joint-covariance gate failure
  blocks real-data evaluation.
- A second fresh production failure closes Milestone 56 and retains
  `neural_hmsc_v0_1` unchanged.

## Expected Outcome And Claim

If successful, the milestone will freeze a separate fixed-shape artifact that
adds qualified within-species coefficient covariance to the existing probit
Beta approximation. It would support correlated Beta draws and a bounded
summary-level claim for coefficient covariance within the exact fixed-effect
scope.

It would **not** produce a near-equivalent implementation of full HMSC MCMC.
It would improve one missing row of the equivalence matrix: joint uncertainty
between fixed-effect coefficients within a species. Python MCMC would remain
the reference for full joint posterior, traits, random effects, spatial
structure, and species associations.

## Resolved Next Step

The artifact and unused-seed audit is frozen in
`docs/neural_hmsc_m56_artifact_seed_audit_2026-07-23.json.md`, SHA-256
`5bb9236967afb5a2a1adc166781f4a34359a7469150aa2e19117752dd1fce29c`.
The qualification protocol is frozen in
`docs/neural_hmsc_m56_covariance_preregistration_2026-07-23.md`, SHA-256
`d99b63da87103c3d8891cb2fab5bb7ffad30a188ed7be920950345581f8b2d4b`.
No implementation or simulation was performed before either document was
frozen.

The next step is to implement the frozen correlation representation and sealed
harness, add the preregistered tests, and run only the 291M-292M disposable
smoke. Production roles 211M-215M remain unopened.
