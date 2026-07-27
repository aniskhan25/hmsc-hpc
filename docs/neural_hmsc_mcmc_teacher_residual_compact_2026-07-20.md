# MCMC-Teacher Residual Compact Evaluation

Date: 2026-07-20

## Decision

The compact shared-seed gate failed. The independent validation selector chose
shrinkage `0.0`, so the deployable result is the identity mapping over
`neural_predictive_affine_v1`. Real-data evaluation and a five-seed LUMI run
remain blocked.

The nonzero raw head is retained as an experimental artifact, not as a
promotion candidate. It improved the Big-Spatial-shaped regime, but movement
was not uniformly safe across covariate-shift and rare-validation folds.

## Competitor

`McmcTeacherResidualHead` is a bounded response-logit residual MLP applied after
the frozen affine probability ensemble. Its ten inputs are deployment-safe
summaries of baseline probability, predicted species prevalence, predicted
site richness, design information, covariate support, and community shape. The
head:

- trains only against Python-native HMSC MCMC response probabilities generated
  from independent simulations;
- starts at identity and applies an explicit residual penalty;
- selects shrinkage on an independent validation simulation seed;
- does not use real outcomes, alter coefficient posterior draws, or replace
  coefficient/SBC uncertainty calibration;
- preserves the frozen coefficient SBC, OOD, rare-validation, real-data,
  leave-one-out, and provenance gates.

The implementation is in `pyhmsc/neural/teacher_residual.py`; the compact
harness is `examples/evaluate_neural_hmsc_mcmc_teacher_residual.py`.

## Exact Frozen Baseline

The retained run uses the ordered three-member Big Spatial transfer-affine
ensemble, not a single checkpoint proxy. Each predictive artifact hash was
validated against the immutable `neural_predictive_affine_v1` manifest before
inference. Member response probabilities were combined by arithmetic mean.

- Baseline bundle SHA-256:
  `858e6843a29c462eeb5dbc8299112293fe416278fc5a9e9f97eb65944f5bff36`
- Big Spatial affine manifest SHA-256:
  `903f04b9ed66908f19c6dfd6c7f47c41bee2e7f75648373d0255fadb1dd9c51f`

| Seed | Predictive artifact SHA-256 | Active transfer affine |
|---:|---|---|
| `20260721` | `12a108f22e6128fdd6bda41c4cc480b07bd85ae0a5207e8181cdf8240474756e` | slope `1.05`, intercept `0.05` |
| `20260722` | `6c8456eafe7e690586c7710f4f3a04a5b52b071085f638670b3644036d188ce5` | slope `1.05`, intercept `0.05` |
| `20260723` | `9b7a1b82974cf9f5f427e05fd08c4ee5e68f0a014e34020eede1c0aad2e04fa5` | slope `1.025`, intercept `0.05` |

The harness additionally records each source checkpoint metadata and weight
hash. The baseline registry did not originally pin source checkpoint weight
hashes, so the three downloaded source-run checkpoints are recorded as added
simulation provenance rather than silently treated as registry fields.

## Evaluation Design

Each neural ensemble member and Python-native MCMC teacher is fitted on 40
simulated training sites. Teacher targets and outcome scores use 20 disjoint
holdout sites and 75 species, matching the frozen member shape. The five
regimes are `in_distribution`, `covariate_shift`, `effect_size_shift`,
`big_spatial_shape`, and `rare_validation`.

| Role | Seed(s) |
|---|---|
| Head training | `20260731` |
| Shrinkage validation | `20260732` |
| Final compact evaluation | `20260733`, `20260734`, `20260735` |

Each seed expands to one independent simulation per regime. The corpus
contains 25 simulated datasets and 37,500 held-out response probabilities.
MCMC used two chains, 40 transient iterations, and 60 retained iterations per
chain. The complete local workflow took `459.8` seconds.

Two development runs are excluded from the retained decision: the first used
the fitting sites for scoring; the second corrected the holdout but used one
compact proxy checkpoint rather than the frozen ensemble. Neither contributes
to the metrics below.

## Validation Selection

Ratios below are candidate divided by frozen baseline; lower is better.

| Shrinkage | Overall Brier | Overall log loss | Covariate Brier | Effect Brier | Target Brier | Target log loss | Rare Brier | Accepted |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | no |
| 0.25 | 0.9979 | 0.9982 | 1.0001 | 0.9989 | 0.9945 | 0.9965 | 0.9987 | no |
| 0.50 | 0.9974 | 0.9977 | 1.0026 | 0.9995 | 0.9910 | 0.9948 | 0.9982 | no |
| 0.75 | 0.9985 | 0.9985 | 1.0073 | 1.0017 | 0.9896 | 0.9949 | 0.9985 | no |
| 1.00 | 1.0012 | 1.0007 | 1.0142 | 1.0056 | 0.9903 | 0.9968 | 0.9998 | no |

The strict rule requires no outcome-proper-score degradation in every
validation regime plus target teacher and outcome improvement. Shrinkage
`0.25` was close, but its covariate-shift Brier ratio was `1.000145`; no
nonzero candidate met the rule. Identity is a safe fallback but is not counted
as improvement.

## Independent Evaluation

The raw, unshrunk head produced these aggregate ratios on the three untouched
evaluation seeds:

| Regime | Teacher Brier | Teacher cross entropy | Outcome Brier | Outcome log loss |
|---|---:|---:|---:|---:|
| In distribution | 0.8907 | 0.9941 | 0.9932 | 0.9929 |
| Covariate shift | 1.0455 | 0.9982 | 1.0038 | 0.9978 |
| Effect-size shift | 0.8090 | 0.9841 | 0.9798 | 0.9809 |
| Big-Spatial shaped | 0.8530 | 0.9807 | 0.9835 | 0.9857 |
| Rare validation | 1.1209 | 1.0018 | 0.9981 | 0.9974 |

The target-shaped improvement held in all three evaluation seeds, but the raw
head degraded covariate-shift Brier in seeds `20260734` and `20260735` and
rare-validation Brier/log loss in seed `20260733`. The selected identity head
therefore failed the genuine-improvement and per-seed gates by construction.

The result supports the residual-head direction but not this global fit. The
remaining issue is context/fold stability: the learned movement is useful in
the target, effect-shift, and most in-domain contexts, but the same global
function moves a few covariate/rare contexts in the wrong direction.

## Verification

- `56` broader regression and teacher-residual tests passed after the
  exact-ensemble correction.
- Python compilation and `git diff --check` passed.
- The tests cover identity fallback, bounded residuals, artifact round-trip,
  feature construction without outcomes, gradient-target semantics, and
  disjoint training/holdout sites.
- The retained decision is `mcmc_teacher_residual_compact_gate_failed`.

## Next Step

Build a larger cross-fitted MCMC-teacher corpus with multiple independent
training and validation communities per regime. Fit a regime/context-
conditioned identity expert that can retain the stable target/effect movement
while falling back to identity for covariate-shift and rare contexts whose
out-of-fold proper-score direction is unstable. Require every held-out fold to
preserve outcome Brier/log loss and the target-shaped folds to improve. Repeat
the compact shared-seed gate; do not run real-data or five-seed LUMI validation
unless a nonzero cross-fitted head passes.
