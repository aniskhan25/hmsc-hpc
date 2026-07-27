# Generative Neural-HMSC IID Probit v1 Design Review

Date: 2026-07-27

Reviewed protocol: `generative_neural_hmsc_iid_probit_v1`

Preregistration:

`docs/generative_neural_hmsc_iid_v1_preregistration_2026-07-27.md`

Preregistration SHA-256:

`09c6a195ca139bdf168816b4f50db321c789bfdd061628e4f99a28cca81cea3f`

Seed audit:

`docs/generative_neural_hmsc_iid_v1_seed_audit_2026-07-27.json.md`

Seed-audit SHA-256:

`39e8763bf8a4fd525dc624570cd2f2b3392dbd1f62d7fa2e3c326f9340194cd6`

## Review Decision

Approved for implementation and disposable-seed smoke only.

No production, validation, reserved-evaluation, redesign, MCMC-subset, or
real-data seed is authorized by this review.

The design is materially different from the closed fixed-effect/calibration
families:

- it starts from an explicit structural probit model;
- its candidate receives raw X, masked Y, and masks rather than an earlier
  posterior;
- Beta, Eta, Lambda, alpha, and tau occupy one stochastic posterior state;
- uncertainty is trained through a generative variational objective;
- association and random-effect contribution are primary qualification
  targets;
- site and species dimensions are exchangeable by construction.

It therefore satisfies the branch entry gate in the closure audit.

## Findings Resolved During Review

### Exact posterior and native HMSC were initially conflated

Qualified Python HMSC-HPC uses its native hierarchical loading prior. It cannot
serve as an elementwise posterior-equivalence reference for a candidate with a
different explicit prior.

Resolution:

- exact-model HMC/NUTS is the posterior-semantics reference;
- Python HMSC-HPC is the ecological behavioral comparator;
- the protocol forbids elementwise posterior-equivalence claims against the
  latter.

### Independent species intercepts could not reliably fill prevalence strata

With many species, independent intercepts would concentrate community-average
prevalence and make rare/common 75-species cells impractical.

Resolution:

- the generative model now includes shared `alpha`;
- species intercepts are conditionally distributed around alpha;
- alpha is part of the joint posterior;
- prevalence strata are prior-conditional draws, not post-hoc shifts.

### Raw factor recovery is not a valid target

Eta and Lambda have sign, permutation, and rotational ambiguity.

Resolution:

- primary targets are R, A, and C;
- factor gauge fixing is API compatibility only;
- raw Eta/Lambda RMSE, coverage, and rank gates are prohibited.

### Predictor-only success could repeat the earlier branch failure mode

A latent predictor can improve heldout scores without approximating the
posterior or residual association.

Resolution:

- marginal, projection-rank, energy-score, association, MCMC, PPC, and runtime
  gates are conjunctive;
- prediction cannot compensate for a failed structural gate;
- identity/no-latent behavior is a comparator, not a promotion result.

## Architecture Review

The bipartite shared-message encoder is appropriate for the frozen variable
site/species scope. It avoids shape-specific output matrices and learned
entity IDs. Explicit permutation and padding tests are sufficient to catch the
most likely implementation violations.

The rank-16 Normal posterior is implementable at the maximum flattened
dimension and provides a tractable first form of cross-parameter dependence.
Using the determinant lemma and Woodbury identity avoids dense covariance
materialization.

The main representation risk is that one Gaussian in raw factor coordinates
may underrepresent curved or multimodal factor geometry. That risk is accepted
because:

- the model is a bounded first candidate;
- qualification uses invariant summaries;
- the protocol permits one separately preregistered representation redesign;
- the response to failure is not post-hoc calibration.

The implementation must not quietly replace the joint distribution with
independent parameter heads. Shared encoder features do not by themselves
constitute a joint posterior; the rank-16 covariance and joint `log q` are
required.

## Statistical Review

The declared prior is complete enough to simulate, evaluate likelihoods, and
run an independent exact-model MCMC reference. The loading-scale hyperparameter
prevents three fixed latent-strength models from being mislabeled as one
posterior family.

The exact-model MCMC reference must diagnose only:

- alpha, tau, and Beta;
- R, A, C, and registered invariant projections.

Raw Eta/Lambda R-hat or posterior-mean agreement cannot block or rescue a run.

The fixed-validation thresholds are demanding but aligned with the intended
claim. In particular:

- broad coverage without rank behavior does not pass;
- interval narrowing without MCMC-width agreement does not pass;
- association recovery without masked-cell predictive gain does not pass;
- proper-score gain without posterior calibration does not pass.

No additional metric may be introduced as a substitute after 502M opens.

## Evidence Review

The 324-context factorial covers:

- three site counts;
- three species counts;
- two covariate shapes;
- three latent strengths;
- three prevalence regimes;
- two replicates.

The ledger separates:

- disposable smoke;
- candidate training;
- fixed validation;
- three reserved evaluations;
- one sealed redesign family.

The local and LUMI scans found no prior use of any assigned seed token before
the preregistration was written.

The 36-context exact-MCMC/Python-HMSC subset deliberately spans all shape pairs
and weak/strong rare/common corners. Medium regimes remain covered by
truth-based SBC and predictive gates in the full 324 contexts.

## Real-Data Review

Whittaker is retained only as a frozen no-trait iid-site requalification:

- it is inside the 40-site, 75-species, two-coefficient shape;
- it does not authorize trait or phylogeny claims;
- it opens only after all simulation blocks pass;
- outcomes cannot select movement, thresholds, architecture, or a fallback.

This is enough for a first real-data boundary. Big Spatial is outside the
frozen design-column, site-count, and spatial scope and must not be used to
claim success or failure for this candidate.

## Implementation Conditions

Before disposable seeds open, tests must prove:

1. simulator log-density and generated tensors agree with the declared model;
2. alpha and prevalence-stratum generation fills all 18 disposable cells
   within the fixed attempt limit;
3. low-rank sampling and log probability match a dense reference on tiny
   dimensions;
4. gradients are finite through probit likelihood, prior, and joint log q;
5. site/species permutations and padding preserve identifiable summaries;
6. gauge fixing preserves R, A, and C numerically;
7. masked likelihood excludes every hidden/padded cell;
8. checkpoint roundtrip and dependency inventory are exact;
9. the exact-model MCMC target log density matches the variational target log
   density on identical Theta;
10. old SVD-style iid prototype artifacts cannot load as the new candidate.

The disposable smoke is plumbing and optimization evidence only. Its scores
cannot change the model, objective, factorial, thresholds, or production
authorization. A code defect may be fixed before production only if the
preregistered mathematical design remains unchanged and the fix is documented.

## Claim Review

The success claim is correctly bounded. Even a complete pass would establish
one amortized iid latent probit family, not full HMSC or complete MCMC
equivalence.

The work would be scientifically useful if it passes because it would add the
first qualified neural residual-association posterior in this repository,
while retaining exact-model MCMC and Python HMSC-HPC as references.

If it fails twice under the frozen stop rule, the result will still answer the
bounded question: this generative neural representation did not qualify for
the declared iid structural scope. It must then close rather than start
another calibration sequence.

## Authorized Next Step

Implement the frozen simulator, variable-shape tensor contract, bipartite
encoder, rank-16 joint posterior, importance-weighted objective, exact-model
MCMC reference, immutable artifact schema, sealed harness, and the ten test
conditions above.

After tests pass, run only the 591M-592M disposable smoke. Keep 501M-515M
sealed.
