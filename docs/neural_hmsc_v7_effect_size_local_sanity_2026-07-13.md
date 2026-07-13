# Neural HMSC v7 effect-size-aware local sanity run

Date: 2026-07-13

Branch: `feature/neural-hmsc`

Candidate commit: `2563e4c Add effect-size-aware v7 OOD uncertainty`

## Objective

Run a local production-shape sanity check for the effect-size-aware revision of
the version 7 learned OOD calibration objective before submitting the five-seed
LUMI comparison.

This is not a qualification benchmark. The local machine did not expose the
LUMI TensorFlow GPU environment, and the run used reduced
training/calibration/SBC counts. It was intended to verify metadata semantics,
artifact writing, diagnostics, the in-domain acceptance gate, and whether the
new effect-size signal separates coefficient-magnitude shifts from ordinary
covariate support shifts.

## Configuration

Final sanity command used:

```bash
MPLCONFIGDIR=/private/tmp/pyhmsc-mpl \
XDG_CACHE_HOME=/private/tmp/pyhmsc-cache \
python examples/run_neural_hmsc_conditional_calibration.py \
  --output /private/tmp/neural_hmsc_effect_ood_local_sanity \
  --suite probit \
  --n-sites 40 \
  --n-species 75 \
  --train-datasets 16 \
  --calibration-datasets 16 \
  --epochs 4 \
  --batch-size 4 \
  --conditional-calibration-epochs 60 \
  --conditional-calibration-ood-objective support_excess_rank_coverage \
  --conditional-calibration-ood-datasets 8 \
  --conditional-calibration-ood-objective-epochs 60 \
  --conditional-calibration-ood-uncertainty-max-multiplier 8 \
  --neural-chains 1 \
  --neural-draws 20 \
  --sbc-datasets 32 \
  --sbc-draws 128 \
  --sbc-bins 8 \
  --ood-regimes covariate_shift effect_size_shift combined_shift
```

## Metadata Checks

- coefficient calibration semantics: `7`
- coefficient calibration method: `conditional_rank_aware_anchor_scale`
- OOD objective: `support_excess_rank_coverage`
- OOD objective domains: `covariate_shift`, `effect_size_shift`,
  `combined_shift`
- OOD objective observations: `5400`
- OOD inflation transform: `support_effect_learned_softplus`
- OOD curve terms: `offset`, `support_linear`, `support_quadratic`,
  `effect_linear`, `effect_quadratic`
- predictive calibration semantics: `2`
- probit anchor: `irls_laplace`
- SBC rows: `80`

## Overall SBC Rows

Coverage is coefficient posterior 95% interval coverage, not predictive
calibration.

| Domain | Coverage 95 | Rank mean error | Rank variance error | Beta RMSE | Pass | Inflation mean | Inflation max | Support trust mean | Effect signal mean | Effect signal max | Effect positive fraction |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| In-domain | 0.9296 | 0.0054 | 0.0004 | 0.2915 | yes | 1.1960 | 8.0000 | 0.9883 | 0.3948 | 2.8560 | 0.4629 |
| OOD covariate | 0.9028 | n/a | n/a | 0.4160 | n/a | 3.6904 | 8.0000 | 0.5415 | 0.3334 | 2.5146 | 0.4476 |
| OOD effect-size | 0.8462 | n/a | n/a | 0.6403 | n/a | 1.8032 | 8.0000 | 0.9624 | 0.7906 | 3.1240 | 0.6403 |
| OOD combined | 0.7890 | n/a | n/a | 0.8876 | n/a | 3.5170 | 8.0000 | 0.5624 | 0.5507 | 2.9133 | 0.5867 |

## In-domain Stratified Diagnostics

The current diagnostic artifact emits the explicit acceptance flag on the
overall calibrated in-domain row. Stratified errors below are computed directly
from the SBC rank mean and rank variance targets.

| Stratum | Coverage 95 | Rank mean error | Rank variance error | Beta RMSE |
| --- | ---: | ---: | ---: | ---: |
| Prevalence: rare | 0.9565 | 0.0445 | 0.0194 | 0.3500 |
| Prevalence: intermediate | 0.9432 | 0.0004 | 0.0018 | 0.2983 |
| Prevalence: common | 0.9258 | 0.0053 | 0.0005 | 0.2878 |
| Coefficient: Intercept | 0.9275 | 0.0022 | 0.0006 | 0.2773 |
| Coefficient: x1 | 0.9392 | 0.0165 | 0.0023 | 0.2967 |
| Coefficient: x2 | 0.9221 | 0.0021 | 0.0016 | 0.3000 |
| Design information: low | 0.9587 | 0.0095 | 0.0142 | 0.3399 |
| Design information: intermediate | 0.9192 | 0.0071 | 0.0051 | 0.2827 |
| Design information: high | 0.9108 | 0.0003 | 0.0079 | 0.2439 |

## Decision

The effect-size-aware v7 implementation passes the technical local sanity
checks. The calibration record uses `support_effect_learned_softplus`, legacy
predictive calibration remains on semantics version 2, the benchmark diagnostic
rows include effect-size signal summaries, and the calibrated in-domain overall
row passes the acceptance gate.

The effect-size signal moves in the intended direction: it is lower for pure
covariate shift (`0.3334` mean) and higher for effect-size shift (`0.7906`
mean), where support trust remains high (`0.9624`). That is the structural gap
left by the support-only v7 objective.

This local run is still not a statistical qualification. The rare-prevalence
and low-design-information stratified rows remain noisy at the reduced local
SBC size, and every OOD regime remains below nominal 95% coefficient coverage.
The result is sufficient to justify the planned five-seed LUMI comparison, not
to promote the method as the default.

## Recommended Next Step

Submit the five-seed LUMI comparison against the frozen scalar, version 4,
version 5 IRLS, version 6 default, conservative version 6 strength-1.5/cap-8,
and support-only version 7 references. Acceptance should require preserving all
in-domain SBC gates and reducing OOD coverage/rank degradation, with special
attention to the effect-size-shift and combined-shift regimes.

## LUMI Submission

The first submission mirrored the support-only version 7 comparison and used
`standard-g`, but those jobs remained pending due to priority. Because the
workflow takes about 12 to 13 minutes and fits within the development queue, the
pending `standard-g` jobs were canceled and replaced with `dev-g` submissions.
The synced code writes the `support_effect_learned_softplus` curve and
effect-size diagnostics.

Canceled `standard-g` jobs:

| Seed | Job |
| --- | ---: |
| 20260626 | 19835386 |
| 20260627 | 19835387 |
| 20260628 | 19835388 |
| 20260629 | 19835389 |
| 20260630 | 19835390 |

Replacement `dev-g` jobs:

| Seed | Job | State at submission check |
| --- | ---: | --- |
| 20260626 | 19835554 | completed |
| 20260627 | 19835555 | completed |
| 20260628 | 19835716 | completed |
| 20260629 | 19835717 | completed |
| 20260630 | 19835779 | completed |

The completed five-seed comparison is aggregated in
`docs/neural_hmsc_v7_effect_size_lumi_comparison_2026-07-13.md`.

Run names:

```text
neural_hmsc_v7_effect_ood_2563e4c_seed_20260626
neural_hmsc_v7_effect_ood_2563e4c_seed_20260627
neural_hmsc_v7_effect_ood_2563e4c_seed_20260628
neural_hmsc_v7_effect_ood_2563e4c_seed_20260629
neural_hmsc_v7_effect_ood_2563e4c_seed_20260630
```
