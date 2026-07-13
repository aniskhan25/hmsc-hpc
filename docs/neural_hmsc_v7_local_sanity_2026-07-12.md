# Neural HMSC v7 local production-shape sanity run

Date: 2026-07-12

Branch: `feature/neural-hmsc`

Candidate commit: `cdc9c6a Add learned OOD calibration objective`

## Objective

Run a local production-shape sanity check for the opt-in version 7 learned OOD
calibration objective before submitting the five-seed LUMI comparison.

The local machine did not expose a TensorFlow GPU, so this was not intended as
a qualification benchmark. It used production dimensions and all OOD regimes,
but reduced training/calibration/SBC counts.

## Configuration

Final sanity command used:

```bash
python examples/run_neural_hmsc_conditional_calibration.py \
  --output /private/tmp/neural_hmsc_v7_local_sanity_cdc9c6a_sbc32 \
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
- OOD objective domains: `covariate_shift`, `effect_size_shift`, `combined_shift`
- OOD objective observations: `5400`
- OOD inflation transform: `support_excess_learned_softplus`
- predictive calibration semantics: `2`
- probit anchor: `irls_laplace`
- SBC rows: `80`

## Overall SBC Rows

| Domain | Coverage 95 | Rank mean error | Rank variance error | Beta RMSE | Pass | Inflation mean | Support trust mean |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| In-domain | 0.9276 | 0.0051 | 0.0023 | 0.2899 | yes | 1.0178 | 0.9883 |
| OOD covariate | 0.8981 | 0.0042 | 0.0168 | 0.4155 | no | 3.5804 | 0.5415 |
| OOD effect-size | 0.7585 | 0.0024 | 0.0391 | 0.6344 | no | 1.0584 | 0.9624 |
| OOD combined | 0.7522 | 0.0002 | 0.0215 | 0.8856 | no | 3.2040 | 0.5624 |

## In-domain Stratified Rows

| Stratum | Coverage 95 | Rank mean error | Rank variance error | Pass |
| --- | ---: | ---: | ---: | --- |
| Prevalence: rare | 0.9469 | 0.0364 | 0.0060 | no |
| Prevalence: intermediate | 0.9390 | 0.0024 | 0.0022 | yes |
| Prevalence: common | 0.9246 | 0.0055 | 0.0026 | yes |
| Coefficient: Intercept | 0.9263 | 0.0031 | 0.0021 | yes |
| Coefficient: x1 | 0.9387 | 0.0160 | 0.0000 | yes |
| Coefficient: x2 | 0.9179 | 0.0024 | 0.0046 | yes |
| Design information: low | 0.9321 | 0.0084 | 0.0003 | yes |
| Design information: intermediate | 0.9258 | 0.0067 | 0.0028 | yes |
| Design information: high | 0.9250 | 0.0002 | 0.0039 | yes |

## Decision

The production-shape local sanity run passes the technical checks: v7 metadata,
separate predictive semantics, artifact writing, OOD diagnostics, and SBC row
generation all work at the target `40 x 75` shape.

It is not yet a clean pre-LUMI statistical pass. Overall in-domain SBC passes,
but the rare-prevalence in-domain stratum fails the rank-mean gate. OOD
coverage improves substantially under covariate shift and nearly reaches the
coverage gate, but effect-size and combined shifts remain well below the OOD
coverage target.

## Recommended Next Step

Before spending a full five-seed LUMI comparison, add rare-prevalence in-domain
gate penalties to the learned OOD objective or increase the in-domain gate
weight and rerun this same local sanity command. If the rare-prevalence gate
passes locally, submit the five-seed LUMI comparison against scalar, version 4,
version 5 IRLS, version 6 default, and the conservative version 6
strength-1.5/cap-8 candidate.

## LUMI Submission

The user requested proceeding with the five-seed LUMI comparison despite the
rare-prevalence warning. Jobs were submitted on `standard-g` with the same
production settings as the version 6 comparisons plus the version 7 learned OOD
objective:

- `20260626`: `19831708`
- `20260627`: `19831709`
- `20260628`: `19831710`
- `20260629`: `19831711`
- `20260630`: `19831712`

Run names:

```text
neural_hmsc_v7_ood_objective_cdc9c6a_seed_20260626
neural_hmsc_v7_ood_objective_cdc9c6a_seed_20260627
neural_hmsc_v7_ood_objective_cdc9c6a_seed_20260628
neural_hmsc_v7_ood_objective_cdc9c6a_seed_20260629
neural_hmsc_v7_ood_objective_cdc9c6a_seed_20260630
```
