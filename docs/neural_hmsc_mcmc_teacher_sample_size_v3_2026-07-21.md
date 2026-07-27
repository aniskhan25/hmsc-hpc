# Sample-Size-Stable MCMC-Teacher Corpus V3

Date: 2026-07-21

## Decision

The revised compact simulation gate and outcome-blind real-context routing gate
both failed closed. The v3 representation removed unsupported total-information
extrapolation and the new 360-site target simulations improved proper scores,
but one independent effect-shift seed degraded slightly and the simulated
target prevalence still did not cover frozen Big Spatial predictions. No real
ecological responses were opened and real-data scoring remains blocked.

## Implementation

The teacher artifact schema is now version 3. Existing version 2 artifacts
remain loadable with their original feature semantics. Version 3 replaces
total `n * p * (1-p)` information with per-site mean Bernoulli information and
replaces raw log site count with a bounded log-ratio context. The router stores
the same representation version as the residual head and rejects mixed
representations.

The rebuilt corpus uses:

- 20-site compact profiles for every regime;
- a 360-site profile for the Big-Spatial-shaped regime;
- a 12-site profile for the rare/Whittaker identity control;
- low-prevalence effect-shift target simulations rather than the previous
  high-prevalence combined-shift approximation;
- one shared 40-site MCMC fit per community/regime across nested holdouts;
- balanced residual-training targets, capped to 900 site-species values per
  batch so the 360-site batches do not dominate the objective.

Four cross-fit communities produced 28 calibration batches. Three independent
communities produced 21 evaluation batches. The corpus contains 12-, 20-, and
360-site contexts across 75 species.

## Compact Simulation Gate

The cross-fit selector retained shrinkage `0.75`, margin `0.0`, effect cap
`3.0`, and target cap `2.5`. Aggregate ratios remained favorable:

| Regime | Outcome Brier | Outcome log loss | Action |
|---|---:|---:|---|
| In distribution | 1.000000 | 1.000000 | identity |
| Covariate shift | 1.000000 | 1.000000 | identity |
| Effect-size shift | 0.992629 | 0.993254 | residual |
| Big-Spatial shaped | 0.993869 | 0.995058 | residual |
| Rare validation | 1.000000 | 1.000000 | identity |

Both target profiles improved independently: the 20-site profile had
Brier/log-loss ratios `0.992696/0.993942`, and the 360-site profile had
`0.995042/0.996174`. However, effect-shift evaluation seed `20260741`
degraded to `1.001300` Brier and `1.001508` log loss. Therefore
`all_seed_regime_no_degradation=false` and the compact decision is
`mcmc_teacher_residual_compact_gate_failed`.

## Outcome-Blind Routing

The exact frozen affine ensembles and held-out covariates were then passed to
the v3 router without opening outcomes or MCMC predictions.

| Dataset | Required | Selected | Approved distance / cap | Pass |
|---|---|---|---:|---:|
| Whittaker | identity | rare validation | 2.7608 / 2.5 | true |
| Big Spatial | approved target/effect | rare validation | 4.1895 / 2.5 | false |

Whittaker remained numerical identity. Big Spatial also remained identity: its
nearest rare fallback distance was `3.6174`, less than its approved target
distance `4.1895`.

The sample-size coordinate is now inside the fitted representation, so the
previous 360-versus-20 extrapolation is no longer the dominant issue. The
remaining mismatch is prevalence/context support. Frozen Big Spatial baseline
probability mean is `0.1002`; calibration target profiles average about
`0.185`, and independent evaluation target profiles average about `0.210`.
The new target simulator moved in the correct direction but did not bracket the
actual outcome-blind target context.

## Provenance

Retained compact run:

`/private/tmp/neural_hmsc_mcmc_teacher_sample_size_v3_compact_20260721`

Retained routing run:

`/private/tmp/neural_hmsc_teacher_context_routing_v3_20260721`

- Comparison JSON SHA-256:
  `7e554feefd2fc3475f6c5708309a6df151296087322204de09792c85884b362c`
- Teacher metadata SHA-256:
  `57a250edfde923a479454df118332b79919236176f1be382ef3f2082efa08ca3`
- Teacher weights SHA-256:
  `b7131014a83ef91a8a607206b345f30f9174a7b85472cedc2319f5a251298731`
- Routing JSON SHA-256:
  `d9448252cced4c13e699cb0ef9b8ccd81e23cb779f9b116873e45abdfb8c6727`

## Verification

- `15` focused teacher and routing tests passed during implementation.
- The frozen version 2 teacher artifact loaded with representation version 2
  and its original 14-feature context gate.
- Python compilation and `git diff --check` passed.
- The routing output records no target-response access and no proper scoring.

## Next Step

Add an outcome-blind simulation-support qualification before another teacher
fit. Generated target profiles must bracket the frozen Big Spatial probability
mean, prevalence quantiles, covariate summaries, and bounded site-count context;
rare/Whittaker profiles must independently cover the identity context. Reject
the corpus before MCMC fitting if those support requirements fail. Once support
passes, rerun with fresh cross-fit/evaluation communities and a more
conservative shrinkage-selection rule that requires a practical per-fold
effect-shift margin, rather than selecting the largest accepted movement by
average objective. Only a fresh all-seed compact pass may repeat routing.
