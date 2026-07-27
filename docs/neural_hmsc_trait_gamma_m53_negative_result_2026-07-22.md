# Neural-HMSC Milestone 53 Trait-Gamma Result

Date: 2026-07-22

## Decision

The fixed-shape trait-mediated Gamma candidate is **not promoted**. The
implementation remains experimental in `pyhmsc.neural`; it is not exported by
the stable top-level `pyhmsc` API and no immutable deployment baseline was
created.

Both existing regression baselines remain unchanged:

- `neural_hmsc_v0_1`:
  `affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8`
- `neural_hmsc_variable_probit_v1`:
  `badaf8b8244cbd850693723147c6094bf196482237c52d2cc279ce42b286d2f9`

## Implemented Candidate

The bounded candidate covers probit models with 40 sites, 75 species,
`~ TMG`, and one compiler-scaled non-intercept trait from `~ CN`. It uses:

- the existing species-level probit IRLS/Laplace anchor for Beta;
- a bounded joint site-species probit IRLS/Laplace Gamma anchor over `X (x) T`;
- species-permutation-invariant aggregate neural features;
- bounded, zero-initialized posterior-mean and log-scale corrections;
- independent simulation calibration with coefficient-posterior semantics;
- separate Beta and Gamma marginal samples in ordinary `HmscFit` HDF5 output.

The separate marginals are not a coupled joint posterior.

## Evidence

The predeclared seed `20260801` passed every frozen gate:

| Metric | Result |
| --- | ---: |
| Gamma 95% coverage | 0.929688 |
| Gamma rank mean | 0.484436 |
| Gamma rank variance | 0.082419 |
| Neural/MCMC simulated Gamma RMSE ratio | 0.930566 |
| Whittaker neural/MCMC Brier ratio | 1.012636 |
| Whittaker neural/MCMC log-loss ratio | 0.966303 |

The intended sensitivity seed `20260802` failed the predeclared Gamma coverage
floor:

| Metric | Result | Gate |
| --- | ---: | ---: |
| Gamma 95% coverage | 0.882812 | >= 0.90 |
| Intercept coverage | 0.890625 | diagnostic |
| TMG coverage | 0.875000 | diagnostic |
| Gamma rank mean | 0.482788 | 0.40-0.60 |
| Gamma rank variance | 0.086877 | 0.06-0.11 |
| Neural/MCMC simulated Gamma RMSE ratio | 0.675752 | <= 1.25 |
| Whittaker neural/MCMC Brier ratio | 1.022812 | <= 1.05 |
| Whittaker neural/MCMC log-loss ratio | 0.988551 | <= 1.05 |

Bias was negligible for both Gamma coefficients, so the failure is uncertainty
calibration transfer rather than posterior-mean direction. Increasing or
stratifying the scale after observing this sensitivity result would be
post-hoc tuning.

## Stop Rule

This family used its one representation redesign, from a two-stage weighted
Beta projection to a bounded joint trait likelihood. A subsequent decision
review found that the intended sensitivity corpora overlapped the candidate by
63/64 training, 31/32 calibration, and 63/64 test communities because their
base seeds differed by only one. The result cannot count as the fresh
independent evaluation required by the stop rule. Milestone 53 remains paused,
and no baseline is frozen or published, but one genuinely disjoint
calibration-only requalification is preregistered as Milestone 53A.

## Next Step

Implement the frozen-weight Milestone 53A protocol in
`docs/neural_hmsc_trait_gamma_calibration_decision_2026-07-22.md`. Do not
proceed to iid Eta/Lambda qualification while the prerequisite trait-Gamma
family remains unqualified.
