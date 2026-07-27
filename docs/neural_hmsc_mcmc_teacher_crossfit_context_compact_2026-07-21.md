# Cross-Fitted MCMC-Teacher Context Gate

Date: 2026-07-21

## Decision

The larger cross-fitted compact gate passed against the exact frozen
`neural_predictive_affine_v1` Big Spatial affine ensemble. A nonzero residual
head was selected without changing coefficient posteriors or uncertainty
calibration. Real-data outcomes were not opened and no LUMI job was submitted.

The selected predictive-only competitor uses:

- residual shrinkage `0.5`;
- context margin `0.0`;
- effect-size-shift approved-distance cap `3.0`;
- Big-Spatial-shaped approved-distance cap `2.5`;
- exact identity fallback for in-domain, covariate-shift, rare, and ambiguous
  contexts.

## Method

The calibration corpus contains four independent communities per regime. Each
leave-one-community-out fold trains the residual head only on simulated MCMC
predictive probabilities from the other three communities. Fold outcomes are
used only for model selection and no-degradation gates, never in a gradient.

The context expert uses 14 summaries computed from frozen baseline
probabilities and design covariates. It stores normalized regime prototypes,
approved labels, prototype-distance margins, and label-specific support caps.
The residual is applied only when an input is closer to an approved prototype
than every fallback prototype and remains within that approved context's
support cap. Otherwise its multiplier is exactly zero.

The retained corpus has:

| Partition | Communities | Regimes | Datasets | Held-out probabilities |
|---|---:|---:|---:|---:|
| Cross-fit calibration | 4 | 5 | 20 | 30,000 |
| Independent evaluation | 3 | 5 | 15 | 22,500 |
| Total | 7 | 5 | 35 | 52,500 |

Every dataset fits the neural ensemble and Python-native MCMC teacher on 40
sites and scores 20 disjoint sites across 75 species. MCMC uses two chains, 40
transient iterations, and 60 retained iterations per chain. Calibration seeds
are `20260731`-`20260734`; untouched evaluation seeds are `20260741`-
`20260743`.

The first run reused the 20 previously qualified calibration teachers and
generated all 15 evaluation teachers. The final paired rerun reused those exact
posterior files after the support-cap implementation; their hashes and all
ensemble-member hashes are recorded in the comparison JSON.

## Frozen Baseline

- Baseline: `neural_predictive_affine_v1`
- Baseline bundle SHA-256:
  `858e6843a29c462eeb5dbc8299112293fe416278fc5a9e9f97eb65944f5bff36`
- Big Spatial affine manifest SHA-256:
  `903f04b9ed66908f19c6dfd6c7f47c41bee2e7f75648373d0255fadb1dd9c51f`
- Ordered members: `20260721`, `20260722`, `20260723`
- Aggregation: arithmetic mean response probability

The three predictive-artifact hashes matched the immutable manifest. The
source checkpoint metadata and weight hashes are also retained as simulation
provenance.

## Cross-Fit Selection

Twelve candidate combinations passed every fold: shrinkages `0.1`, `0.25`, and
`0.5`, margins `0.0` and `0.25`, and the two tight support-cap profiles. The
minimum approved target/effect objective selected shrinkage `0.5`, margin
`0.0`, effect cap `3.0`, and target cap `2.5`.

For every cross-fit fold and true regime, the selected candidate preserved
outcome Brier score and log loss. Effect and target directions improved in
every fold. Covariate-shift and rare-validation contexts were inactive in every
fold. The support caps resolve the previous overlap where one covariate-shift
community was marginally closer to the target prototype.

## Independent Evaluation

Ratios are selected candidate divided by the frozen affine baseline; lower is
better.

| Regime | Teacher Brier | Teacher cross entropy | Outcome Brier | Outcome log loss | Action |
|---|---:|---:|---:|---:|---|
| In distribution | 1.0000 | 1.0000 | 1.0000 | 1.0000 | identity |
| Covariate shift | 1.0000 | 1.0000 | 1.0000 | 1.0000 | identity |
| Effect-size shift | 0.8994 | 0.9934 | 0.9917 | 0.9926 | residual |
| Big-Spatial shaped | 0.8983 | 0.9858 | 0.9861 | 0.9878 | residual |
| Rare validation | 1.0000 | 1.0000 | 1.0000 | 1.0000 | identity |

Effect and target outcome Brier/log loss improved in all three untouched
evaluation communities. In-domain, covariate-shift, and rare-validation
predictions remained identity within numerical precision. The final decision
is `mcmc_teacher_residual_compact_gate_passed`; all-regime and all-seed
no-degradation gates passed with the declared `1e-10` identity tolerance.

## Artifacts

Local retained run:

`/private/tmp/neural_hmsc_mcmc_teacher_crossfit_context_compact_20260721`

- Head metadata SHA-256:
  `dd68ad26d46cb5667835744b080e800e19f956db41144c6fd8a8de3d291b60c4`
- Head weights SHA-256:
  `6ff5659b5cb264b9ab74b39ae4524ff0529f0031fdf337705758b37e32b588c8`
- Comparison JSON SHA-256:
  `a13ce8600e73fd6d18c7fcc9c8270f7130d790199c87e04ca194ed799f4c91bf`

The JSON hash above records the final paired rerun. These local artifact hashes
are evidence for this compact decision, not a promoted deployment baseline.

## Verification

- `58` broader posterior, ensemble, deployment, real-data integration, and
  teacher-residual tests passed.
- Tests cover context features without outcomes, prototype gate round-trip,
  exact rare identity fallback, stable cross-fit selection, residual artifact
  round-trip, disjoint simulation sites, and legacy ungated artifacts.
- Python compilation and `git diff --check` passed.

## Next Step

The outcome-blind routing check is complete and failed closed. Whittaker routed
to identity, but Big Spatial was outside approved support, primarily because
the 20-site simulation corpus did not cover its 360-site design-information
context. Details are in
`docs/neural_hmsc_mcmc_teacher_real_context_routing_2026-07-21.md`. The next
step is to rebuild the representation and cross-fitted simulation support, then
repeat compact simulation and outcome-blind routing gates before any real-data
scoring, multi-seed LUMI comparison, or deployment-policy change.
