# Neural-HMSC Milestone 54 v2.1 Terminal Result

Date: 2026-07-23

Status: terminal simulated qualification failure; Milestone 54 closed.

## One-Shot Evaluation

Exact authorization: `OPEN_M54_V2_1_RESERVED_EVALUATION`

LUMI dev-g job `20179655` completed in `00:04:31` with exit code `0`.
The process-level run and all provenance checks passed. The one-shot evaluation
used:

- frozen production report SHA-256:
  `bb32afd655db277064c5c6fcbdf53e2d89a9f42c24a0690c50a494967f46d816`;
- exact reserved evaluation block 115000001-115000243;
- exact six preregistered Python MCMC comparison seeds;
- unchanged checkpoint, calibration, objective, gates, thresholds, seed roles,
  immutable baselines, and preregistration.

The result root is:

`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_m54_v2_1_evaluation_20179655`

The evaluation report SHA-256 is
`c410476ee77f7815fe5dbcde0cdd807cb352c9764802da930b50e339cd8dee4f`.
The post-evaluation validation SHA-256 is
`3b9065d4653a26d40e722c31dcafea1c2c2fbfe6371fc1e40d1e7a47e4c91fdc`.
An independent downloaded replay confirmed the report hash, exact 115M block,
freeze hash, MCMC seed subset, gate consistency, and terminal decision.

## Decision

The immutable decision is:

`variable_design_v2_1_terminal_failure`

Five preregistered gates failed:

- aggregate Brier score versus the anchor;
- aggregate log loss versus the anchor;
- proper-score limits for every covariate-count stratum;
- proper-score limits for every site-count stratum;
- ordered support-gate movement.

The aggregate neural/anchor Brier ratio was `1.041475`, above `1.02`. The
aggregate log-loss ratio was `1.028093`, also above `1.02`.

The predictive degradation was concentrated in the low-support strata:

| Stratum | Brier ratio | Log-loss ratio |
| --- | ---: | ---: |
| 12 sites | 1.090547 | 1.051274 |
| 40 sites | 1.034718 | 1.028662 |
| 128 sites | 1.007805 | 1.007267 |
| 2 covariates | 1.004253 | 1.001484 |
| 5 covariates | 1.040762 | 1.027626 |
| 8 covariates | 1.088154 | 1.059872 |

The support gate learned the opposite ordering from the intended
support-aware fallback. Its median was `0.988575` at the low-support
12-site/8-covariate corner and `0.384689` at the high-support
128-site/2-covariate corner. The model therefore applied the largest residual
mean movement where the anchor had the least design support.

## What Passed

The failure is not an operational, posterior-scale, or coefficient-mean
failure:

- checkpoint roundtrip and all provenance/hash checks passed;
- factorial balance and every calibration stratum passed;
- 95% coefficient coverage was `0.949787`;
- rank mean was `0.510716`;
- rank variance was `0.075209`;
- coefficient Beta RMSE was `0.296233` versus anchor `0.377353`, a ratio of
  `0.785029`;
- the six-seed neural/MCMC Brier and log-loss ratios were `0.951649` and
  `0.947369`, within the frozen `1.10` limits;
- the support gate remained bounded.

These passing metrics cannot average away the failed aggregate, stratum, and
support-ordering gates.

## Interpretation And Stop Decision

The predictive auxiliary loss successfully improved coefficient posterior
means but did not teach a transfer-stable notion of design support. It learned
to trust residual movement most in small-sample, high-dimensional contexts,
causing the same low-support predictive degradation that the v2.1
representation was designed to prevent.

This was the single permitted representation redesign after the v1.1 failure.
Under the frozen stop rule:

- Milestone 54 is closed;
- no loss-weight, threshold, cap, calibration, seed, or router tuning is
  permitted;
- no rerun or third variable-design representation is permitted;
- Whittaker and Big Spatial real-data evaluation must not be opened for v2.1;
- `neural_hmsc_variable_probit_v1` remains the qualified variable-shape
  endpoint;
- Python MCMC remains the statistical reference and fallback outside qualified
  neural scope.

## Next Roadmap Step

Freeze this negative result as the Milestone 54 endpoint and return to the
already-qualified neural fixed-effect scope. The next development milestone
must be outside the failed variable-design family and must begin with a bounded
capability decision and fresh preregistration. Do not reopen Milestone 54 or
advance the deferred MCMC near-equivalence claim from this result.
