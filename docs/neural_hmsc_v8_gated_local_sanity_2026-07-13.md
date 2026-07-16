# Neural HMSC v8 gated effect-size local sanity run

Date: 2026-07-13

Branch: `feature/neural-hmsc`

## Objective

Implement and locally test a version 8 OOD objective where effect-size
inflation is gated by OOD context rather than applied directly to every positive
posterior-mean magnitude signal.

The new opt-in objective is:

```text
support_effect_gated_rank_coverage
```

It serializes coefficient calibration as semantics version `8` with transform:

```text
support_effect_gated_learned_softplus
```

The learned curve keeps the support-excess branch from version 7 and multiplies
the effect-size branch by a learned context gate over support excess and
effect-signal magnitude. A v8-only in-domain extra-inflation penalty was also
added so the learned effect branch is discouraged from broadly inflating
in-domain coefficients.

## Validation

Focused tests:

```text
pytest tests/test_neural_hmsc_conditional_calibration.py -q
12 passed

pytest tests/test_neural_hmsc_lumi_workflow.py -q
4 passed
```

Production-shape local sanity command:

```bash
MPLCONFIGDIR=/private/tmp/pyhmsc-mpl \
XDG_CACHE_HOME=/private/tmp/pyhmsc-cache \
python examples/run_neural_hmsc_conditional_calibration.py \
  --output /private/tmp/neural_hmsc_v8_gated_penalty_ood_local_sanity \
  --suite probit \
  --n-sites 40 \
  --n-species 75 \
  --train-datasets 16 \
  --calibration-datasets 16 \
  --epochs 4 \
  --batch-size 4 \
  --conditional-calibration-epochs 60 \
  --conditional-calibration-ood-objective support_effect_gated_rank_coverage \
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

- coefficient calibration semantics: `8`
- OOD objective: `support_effect_gated_rank_coverage`
- OOD uncertainty transform: `support_effect_gated_learned_softplus`
- curve terms: `offset`, `support_linear`, `support_quadratic`,
  `effect_linear`, `effect_quadratic`, `effect_gate_intercept`,
  `effect_gate_support_linear`, `effect_gate_effect_linear`
- predictive calibration semantics: `2`
- OOD objective observations: `5400`

## Overall SBC Rows

Coverage is coefficient posterior 95% interval coverage, not predictive
calibration.

| Domain | Coverage 95 | Rank mean error | Rank variance error | Beta RMSE | Pass | Inflation mean | Inflated fraction | Support trust mean | Effect signal mean |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| In-domain | 0.9292 | 0.0053 | 0.0016 | 0.2912 | yes | 1.1730 | 0.4883 | 0.9883 | 0.3948 |
| OOD covariate | 0.9018 | n/a | n/a | 0.4158 | n/a | 3.6640 | 0.7831 | 0.5415 | 0.3334 |
| OOD effect-size | 0.8410 | n/a | n/a | 0.6399 | n/a | 1.7271 | 0.6774 | 0.9624 | 0.7906 |
| OOD combined | 0.7842 | n/a | n/a | 0.8874 | n/a | 3.4672 | 0.8226 | 0.5624 | 0.5507 |

## In-domain Stratified Diagnostics

| Stratum | Coverage 95 | Rank mean error | Rank variance error |
| --- | ---: | ---: | ---: |
| Prevalence: rare | 0.9565 | 0.0444 | 0.0184 |
| Prevalence: intermediate | 0.9432 | 0.0006 | 0.0010 |
| Prevalence: common | 0.9253 | 0.0053 | 0.0010 |
| Coefficient: Intercept | 0.9271 | 0.0023 | 0.0000 |
| Coefficient: x1 | 0.9388 | 0.0165 | 0.0018 |
| Coefficient: x2 | 0.9217 | 0.0021 | 0.0022 |
| Design information: low | 0.9579 | 0.0098 | 0.0127 |
| Design information: intermediate | 0.9188 | 0.0068 | 0.0053 |
| Design information: high | 0.9108 | 0.0003 | 0.0080 |

## Decision

Do not submit the five-seed LUMI comparison for this v8 candidate yet.

The overall in-domain row passes, and covariate-shift OOD coverage is near the
local stress target. However, the local precondition does not hold at the
stratified level: rare-prevalence rank mean remains high, high-design
information coverage is only `0.9108`, and effect-size-shift coverage remains
below the previous effect-aware v7 local result.

The gated objective is therefore implemented and validated mechanically, but it
is not statistically ready for the five-seed LUMI comparison.

## Recommended Next Step

Before LUMI, add explicit stratified in-domain gates to the OOD objective,
especially design-information groups, instead of relying on prevalence-only
rank groups and overall coverage. The next candidate should preserve the v7
effect-size OOD gain while requiring local high-design-information and
rare-prevalence diagnostics to hold before submission.

## Stratified Gate Follow-Up

The follow-up implementation added explicit in-domain OOD-objective gate groups
for:

- prevalence strata,
- design-information tertiles,
- coefficient identity.

It also added per-stratum coverage penalties with a stricter `0.925` local
coverage floor and used both mean and worst-stratum gate losses so rare or
high-information strata are not diluted by easier groups.

Focused validation:

```text
pytest tests/test_neural_hmsc_conditional_calibration.py -q
13 passed

pytest tests/test_neural_hmsc_lumi_workflow.py -q
4 passed
```

Fresh local sanity command used the same shape and objective settings, writing
artifacts to:

```text
/private/tmp/neural_hmsc_v8_stratified_strict_gate_local_sanity
```

Overall rows:

| Domain | Coverage 95 | Beta RMSE | Inflation mean | Effect signal mean |
| --- | ---: | ---: | ---: | ---: |
| In-domain | 0.9299 | 0.2906 | 1.1371 | 0.3948 |
| OOD covariate | 0.9011 | 0.4157 | 3.6361 | 0.3334 |
| OOD effect-size | 0.8326 | 0.6389 | 1.5872 | 0.7906 |
| OOD combined | 0.7792 | 0.8870 | 3.4007 | 0.5507 |

In-domain weak strata after the stratified gate:

| Stratum | Coverage 95 | Rank mean error | Rank variance error |
| --- | ---: | ---: | ---: |
| Prevalence: rare | 0.9565 | 0.0431 | 0.0174 |
| Design information: intermediate | 0.9200 | 0.0068 | 0.0049 |
| Design information: high | 0.9138 | 0.0002 | 0.0073 |

The stricter stratified gate reduced in-domain inflation from `1.1730` to
`1.1371` and slightly improved high-design-information coverage from `0.9108`
to `0.9138`, but it still did not satisfy the local precondition. It also
lowered effect-size OOD coverage from `0.8410` to `0.8326`, further from the
ungated effect-aware version 7 result.

Do not submit this stratified-gate v8 candidate to LUMI.

The next candidate needs more than a stronger penalty on the same scalar
inflation curve. The likely next direction is to separate effect-size OOD
inflation from in-domain coefficient uncertainty by adding stratum-conditioned
gate parameters or a constrained two-branch curve with hard caps for
high-design-information in-domain coefficients.

## Constrained Branch Follow-Up

The constrained-branch implementation extends the version 8 gated curve with a
backward-compatible ninth parameter:

```text
effect_high_design_suppression
```

The parameter suppresses the learned effect-size branch when design information
is high and support excess is close to the training domain. The OOD objective
also adds a direct high-design/support-close cap on in-domain extra inflation.
Legacy seven- and eight-parameter gated curves remain loadable with zero
suppression.

Focused validation after formatting:

```text
pytest tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q
18 passed
```

Fresh local sanity artifacts:

```text
/private/tmp/neural_hmsc_v8_constrained_branch_local_sanity
```

Overall rows:

| Domain | Coverage 95 | Rank mean error | Rank variance | Inflation mean |
| --- | ---: | ---: | ---: | ---: |
| In-domain | 0.9299 | 0.0054 | 0.0837 | 1.1368 |
| OOD covariate | 0.9011 | 0.0041 | 0.0649 | 3.6357 |
| OOD effect-size | 0.8326 | 0.0016 | 0.1054 | 1.5862 |
| OOD combined | 0.7792 | 0.0001 | 0.0991 | 3.3999 |

In-domain weak strata:

| Stratum | Coverage 95 | Rank mean error | Rank variance error |
| --- | ---: | ---: | ---: |
| Prevalence: rare | 0.9565 | 0.0432 | 0.0173 |
| Design information: intermediate | 0.9200 | 0.0068 | 0.0049 |
| Design information: high | 0.9138 | 0.0002 | 0.0073 |

The constrained branch learned a nonzero high-design suppression
(`0.3327`), but the local gate still fails. It did not improve the prior
stratified-gate blocker: high-design coverage remains `0.9138`, intermediate
design coverage remains `0.9200`, and rare-prevalence rank mean error remains
`0.0432`. Effect-size OOD coverage is unchanged at `0.8326`.

Do not submit this constrained v8 candidate to LUMI.

The next roadmap step is to change the objective from a single globally learned
curve with constraints into a genuinely stratum-conditioned model, for example
separate prevalence/design/coefficient intercepts or caps for the in-domain
gate, then rerun the same local sanity gate before any five-seed LUMI
comparison.

## Stratum-Conditioned Branch Follow-Up

The stratum-conditioned implementation extends the version 8 gated curve beyond
the constrained ninth parameter with learned gate offsets for:

- prevalence strata: rare, intermediate, common
- design-information strata: low, intermediate, high
- coefficient identity

It also adds per-stratum in-domain extra-inflation caps, so the objective can
penalize a weak in-domain stratum directly instead of only through the global
extra-inflation cap. Earlier seven-, eight-, and nine-parameter version 8 curves
remain loadable; missing stratum offsets default to zero.

Focused validation:

```text
pytest tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q
19 passed
```

Fresh local sanity artifacts:

```text
/private/tmp/neural_hmsc_v8_stratum_conditioned_local_sanity
```

Learned stratum terms:

| Parameter | Value |
| --- | --- |
| `effect_high_design_suppression` | 0.3285 |
| `effect_prevalence_gate_offsets` | [0.0002, 0.0185, 0.0135] |
| `effect_design_gate_offsets` | [0.0271, 0.0146, -0.0048] |
| `effect_coefficient_gate_offsets` | [0.0170, 0.0093, 0.0083] |

Overall rows:

| Domain | Coverage 95 | Rank mean error | Rank variance | Inflation mean |
| --- | ---: | ---: | ---: | ---: |
| In-domain | 0.9297 | 0.0055 | 0.0839 | 1.1324 |
| OOD covariate | 0.9008 | 0.0041 | 0.0651 | 3.6273 |
| OOD effect-size | 0.8308 | 0.0017 | 0.1058 | 1.5696 |
| OOD combined | 0.7782 | 0.0001 | 0.0994 | 3.3850 |

In-domain weak strata:

| Stratum | Coverage 95 | Rank mean error | Rank variance error |
| --- | ---: | ---: | ---: |
| Prevalence: rare | 0.9565 | 0.0430 | 0.0171 |
| Design information: intermediate | 0.9200 | 0.0068 | 0.0050 |
| Design information: high | 0.9138 | 0.0002 | 0.0073 |

The stratum-conditioned branch is implemented and mechanically validated, but
it is not ready for LUMI. The learned offsets are active but too small to change
the local gate outcome. Compared with the constrained branch, in-domain
inflation fell from `1.1368` to `1.1324`, high-design coverage stayed
`0.9138`, rare-prevalence rank mean error stayed about `0.043`, and
effect-size OOD coverage fell from `0.8326` to `0.8308`.

Do not submit this stratum-conditioned v8 candidate to LUMI.

The next roadmap step should move away from small additive gate offsets and
target the actual residual failure mode: either add a signed-bias correction
for rare-prevalence coefficient ranks before scale calibration, or fit
stratum-specific base scale/normalization terms rather than only
stratum-specific OOD inflation gates.

## Residual In-Domain Follow-Up

Two outside-the-OOD-gate candidates were implemented and tested locally.

First, the conditional calibration code gained an optional serialized
prevalence-by-coefficient mean-bias correction. The correction is fitted as a
shrunk residual mean in coefficient units and applied before scale calibration.
The local sanity run wrote artifacts to:

```text
/private/tmp/neural_hmsc_v8_mean_bias_local_sanity
```

That candidate was rejected as a default behavior. It worsened the residual
rare-prevalence rank issue and high-design coverage:

| Metric | Stratum-conditioned v8 | Mean-bias candidate |
| --- | ---: | ---: |
| In-domain coverage 95 | 0.9297 | 0.9264 |
| Rare-prevalence rank mean error | 0.0430 | 0.0666 |
| High-design coverage 95 | 0.9138 | 0.9079 |
| Effect-size OOD coverage 95 | 0.8308 | 0.8421 |

Because the signed-bias correction did not transfer from the calibration
datasets to the local SBC datasets, automatic mean-bias fitting remains
disabled. The serialization/application machinery remains present for explicit
future experiments, and the stored correction is zero by default.

Second, the code gained stratum-specific base scale offsets outside the OOD
inflation gate. These offsets are learned in the main conditional scale
objective, before OOD inflation, for:

- prevalence strata,
- design-information strata,
- coefficient identity.

Fresh local sanity artifacts:

```text
/private/tmp/neural_hmsc_v8_base_strata_local_sanity
```

Learned base offsets:

| Offset group | Values |
| --- | --- |
| Prevalence | [0.0760, -0.0618, -0.0975] |
| Design information | [0.0260, -0.0339, -0.0242] |
| Coefficient | [0.0574, 0.0160, 0.0267] |

Overall rows:

| Domain | Coverage 95 | Rank mean error | Rank variance | Inflation mean |
| --- | ---: | ---: | ---: | ---: |
| In-domain | 0.9304 | 0.0053 | 0.0839 | 1.1166 |
| OOD covariate | 0.9082 | 0.0041 | 0.0636 | 3.6046 |
| OOD effect-size | 0.8285 | 0.0021 | 0.1070 | 1.5077 |
| OOD combined | 0.7825 | 0.0001 | 0.0985 | 3.3409 |

In-domain weak strata:

| Stratum | Coverage 95 | Rank mean error | Rank variance error |
| --- | ---: | ---: | ---: |
| Prevalence: rare | 0.9614 | 0.0412 | 0.0206 |
| Design information: intermediate | 0.9133 | 0.0060 | 0.0069 |
| Design information: high | 0.9229 | 0.0002 | 0.0047 |

The base-strata candidate is mechanically valid and improves high-design
coverage from `0.9138` to `0.9229`, while lowering in-domain inflation from
`1.1324` to `1.1166`. It still fails the local precondition because
intermediate-design coverage falls to `0.9133` and rare-prevalence rank mean
error remains `0.0412`. Effect-size OOD coverage also falls to `0.8285`.

Do not submit this candidate to LUMI.

The next roadmap step should target the rare-prevalence rank mean directly with
a transfer-robust objective, not residual-mean fitting. A defensible next
candidate is a rank-mean-aware training penalty on the neural posterior mean
or calibration-time monotone rank-centering constrained by held-out SBC, rather
than another scale-only or OOD-inflation adjustment.

## Held-Out Rank-Centering Follow-Up

The calibration code gained a held-out rank-centering mechanism that learns
standardized mean shifts by prevalence stratum and coefficient identity. The
candidate solves monotone shifts that move training rank means toward `0.5`,
then selects a shrinkage only if a deterministic held-out calibration split
improves rare-prevalence rank error without materially degrading validation
coverage.

Fresh local sanity artifacts:

```text
/private/tmp/neural_hmsc_v8_rank_centering_local_sanity
```

The held-out selector chose shrinkage `0.75` with rank-centering values:

| Prevalence stratum | Intercept | x1 | x2 |
| --- | ---: | ---: | ---: |
| Rare | 0.0174 | 0.0133 | 0.0872 |
| Intermediate | 0.0183 | 0.0073 | -0.0260 |
| Common | -0.0343 | -0.0236 | 0.0306 |

Overall rows:

| Domain | Coverage 95 | Rank mean error | Rank variance | Inflation mean |
| --- | ---: | ---: | ---: | ---: |
| In-domain | 0.9292 | 0.0029 | 0.0835 | 1.1405 |
| OOD covariate | 0.9069 | 0.0038 | 0.0636 | 3.6323 |
| OOD effect-size | 0.8314 | 0.0004 | 0.1052 | 1.6050 |
| OOD combined | 0.7810 | 0.0023 | 0.0990 | 3.3518 |

In-domain weak strata:

| Stratum | Coverage 95 | Rank mean error | Rank variance error |
| --- | ---: | ---: | ---: |
| Prevalence: rare | 0.9614 | 0.0554 | 0.0211 |
| Design information: intermediate | 0.9113 | 0.0045 | 0.0073 |
| Design information: high | 0.9208 | 0.0031 | 0.0047 |

This candidate is rejected. The internal held-out calibration split did not
transfer to the local SBC datasets: rare-prevalence rank error worsened from
`0.0412` under the base-strata candidate to `0.0554`, and
intermediate-design coverage remained below the local gate at `0.9113`.

Automatic rank-centering fitting is therefore disabled. The metadata and
application machinery remain implemented for explicit experiments, but stored
rank-centering offsets default to zero in the current v8 path.

Do not submit this candidate to LUMI.

The next roadmap step should move this correction into model training rather
than post-hoc calibration: add a rank-mean-aware posterior-mean penalty during
neural training, evaluated on held-out simulation batches, so rare-prevalence
bias is discouraged before the calibration layer sees the posterior.

## Training Rank-Penalty Follow-Up

The rare-prevalence rank objective was moved into neural training as an
opt-in holdout penalty. The neural training loop now supports:

```text
--rank-mean-penalty-weight
--rank-mean-penalty-holdout-fraction
```

The base NLL/MSE objective trains on the non-holdout simulation batches, while
the rank-mean penalty is evaluated on held-out simulation batches and
backpropagated as a regularizer. The penalty targets rare-prevalence species
rank means overall and by coefficient.

Focused validation:

```text
pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q
33 passed
```

Two local sanity runs were tested:

```text
/private/tmp/neural_hmsc_v8_training_rank_penalty_local_sanity
/private/tmp/neural_hmsc_v8_training_rank_penalty_w002_local_sanity
```

Training penalty comparison:

| Weight | Final training loss | Final rank penalty | Final beta RMSE | Final scale mean |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 35.3335 | 157.8778 | 0.2747 | 0.2783 |
| 0.02 | 29.4310 | 339.6504 | 0.2672 | 0.2630 |

Overall calibrated rows:

| Weight | Domain | Coverage 95 | Rank mean error | Inflation mean |
| ---: | --- | ---: | ---: | ---: |
| 0.05 | In-domain | 0.9303 | 0.0071 | 1.1045 |
| 0.05 | OOD covariate | 0.9107 | 0.0032 | 3.6994 |
| 0.05 | OOD effect-size | 0.8256 | 0.0031 | 1.5172 |
| 0.05 | OOD combined | 0.7907 | 0.0011 | 3.5242 |
| 0.02 | In-domain | 0.9304 | 0.0037 | 1.1106 |
| 0.02 | OOD covariate | 0.9113 | 0.0074 | 3.7000 |
| 0.02 | OOD effect-size | 0.8285 | 0.0043 | 1.5446 |
| 0.02 | OOD combined | 0.7921 | 0.0036 | 3.5334 |

Key in-domain strata:

| Weight | Stratum | Coverage 95 | Rank mean error | Rank variance error |
| ---: | --- | ---: | ---: | ---: |
| 0.05 | Prevalence: rare | 0.9565 | 0.0380 | 0.0168 |
| 0.05 | Design information: intermediate | 0.9150 | 0.0165 | 0.0072 |
| 0.05 | Design information: high | 0.9233 | 0.0031 | 0.0022 |
| 0.02 | Prevalence: rare | 0.9565 | 0.0316 | 0.0174 |
| 0.02 | Design information: intermediate | 0.9146 | 0.0033 | 0.0075 |
| 0.02 | Design information: high | 0.9221 | 0.0146 | 0.0015 |

The training-time penalty is mechanically validated and directionally useful:
the best local run reduced rare-prevalence rank mean error from `0.0412`
under the base-strata candidate to `0.0316`. It still does not clear the local
gate, and the design-information coverage failure remains. Effect-size OOD
coverage also remains weak at `0.8285`.

Do not submit this candidate to LUMI.

The next roadmap step is to tune or redesign the training penalty before any
larger run. The most defensible next candidate is a prevalence-weighted
rank-mean penalty with a design-coverage guard, or a two-stage training scheme
that activates the rank penalty only after the posterior scale has stabilized.

## Training Rank-Penalty Redesign Follow-Up

The training-time penalty was extended with three opt-in controls:

```text
--rank-mean-penalty-start-fraction
--rank-mean-penalty-design-guard-weight
--rank-mean-penalty-design-guard-floor
```

The rank component now uses prevalence-weighted rank centering over rare,
intermediate, and common species, while keeping rare-prevalence species as the
dominant signal. The optional design guard adds a smooth 95% coefficient
coverage floor over expected design-information tertiles. The
`start-fraction` option supports a two-stage schedule in which the base
posterior scale is allowed to stabilize before rank centering is activated.

Focused validation:

```text
python -m py_compile pyhmsc/neural/inference.py examples/run_neural_hmsc_benchmark.py examples/run_neural_hmsc_conditional_calibration.py pyhmsc/neural/conditional_calibration.py
pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q
35 passed
```

Local tuning runs:

```text
/private/tmp/neural_hmsc_v8_training_rank_penalty_guard_local_sanity
/private/tmp/neural_hmsc_v8_training_rank_penalty_guard_w005_start0_local_sanity
/private/tmp/neural_hmsc_v8_training_rank_penalty_guard_w01_start025_local_sanity
/private/tmp/neural_hmsc_v8_training_rank_penalty_w003_start025_local_sanity
/private/tmp/neural_hmsc_v8_training_rank_penalty_guard_w005_epochs8_start05_local_sanity
```

Calibrated local SBC comparison:

| Run | Rare rank error | Rare coverage 95 | Overall coverage 95 | Design low coverage 95 | Design intermediate coverage 95 | Design high coverage 95 | Effect-size OOD coverage 95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| guard 0.25, start 0.5, 4 epochs | 0.0403 | 0.9517 | 0.9315 | 0.9508 | 0.9137 | 0.9300 | 0.8314 |
| guard 0.05, start 0.0, 4 epochs | 0.0302 | 0.9565 | 0.9311 | 0.9529 | 0.9163 | 0.9242 | 0.8282 |
| guard 0.10, start 0.25, 4 epochs | 0.0360 | 0.9469 | 0.9278 | 0.9463 | 0.9104 | 0.9267 | 0.8236 |
| no guard, weight 0.03, start 0.25, 4 epochs | 0.0377 | 0.9469 | 0.9271 | 0.9463 | 0.9108 | 0.9242 | 0.8222 |
| guard 0.05, start 0.5, 8 epochs | 0.0428 | 0.9420 | 0.9306 | 0.9525 | 0.9217 | 0.9175 | 0.8340 |

The best local variant was the small design guard active from epoch 0. It
reduced rare-prevalence rank error to `0.0302`, slightly better than the
previous `0.0316` training-penalty run, and preserved rare coverage. It still
does not clear the `0.025` rank-mean gate, and intermediate design-information
coverage remains below the stricter `0.925` local guard target. The longer
two-stage run improved intermediate-design coverage to `0.9217`, but it
worsened rare rank error and high-design coverage.

Do not submit these candidates to LUMI.

The next roadmap step should stop tuning scale-only or coverage-guard terms
and redesign the posterior-mean part of training. The most likely next
candidate is a signed rare-prevalence posterior-mean correction or auxiliary
loss that targets rank direction directly, with design-stratum constraints so
the rare-prevalence fix does not trade off against intermediate/high design
coverage.

## Signed Posterior-Mean Objective Follow-Up

The posterior-mean part of training was extended with two opt-in controls:

```text
--rank-mean-penalty-signed-mean-weight
--rank-mean-penalty-design-mean-guard-weight
--rank-mean-penalty-design-mean-guard-tolerance
```

The signed mean term reads the held-out rare-prevalence rank imbalance, freezes
that sign, and penalizes normalized posterior-mean bias in the direction that
should move rare ranks toward `0.5`. The design mean guard adds medium/high
expected design-information constraints on rank mean and normalized mean bias,
so rare-prevalence correction cannot ignore design strata.

Focused validation:

```text
python -m py_compile pyhmsc/neural/inference.py examples/run_neural_hmsc_benchmark.py
pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q
36 passed
```

Local sanity runs:

```text
/private/tmp/neural_hmsc_v8_signed_mean_w4_design1_local_sanity
/private/tmp/neural_hmsc_v8_signed_mean_w8_design1_local_sanity
/private/tmp/neural_hmsc_v8_signed_mean_w4_design4_local_sanity
/private/tmp/neural_hmsc_v8_signed_bias_w01_design1_local_sanity
/private/tmp/neural_hmsc_v8_signed_bias_w025_design1_local_sanity
/private/tmp/neural_hmsc_v8_signed_bias_w05_design1_local_sanity
```

Calibrated local SBC comparison:

| Run | Rare rank error | Rare coverage 95 | Overall coverage 95 | Design low coverage 95 | Design intermediate coverage 95 | Design high coverage 95 | Effect-size OOD coverage 95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous best guard 0.05, start 0.0 | 0.0302 | 0.9565 | 0.9311 | 0.9529 | 0.9163 | 0.9242 | 0.8282 |
| pseudo-shift signed mean 4.0, design guard 1.0 | 0.0308 | 0.9517 | 0.9289 | 0.9521 | 0.9142 | 0.9204 | 0.8294 |
| signed bias 0.1, design guard 1.0 | 0.0413 | 0.9517 | 0.9310 | 0.9558 | 0.9142 | 0.9229 | 0.8300 |
| signed bias 0.25, design guard 1.0 | 0.0428 | 0.9662 | 0.9333 | 0.9563 | 0.9150 | 0.9287 | 0.8297 |
| signed bias 0.5, design guard 1.0 | 0.0446 | 0.9710 | 0.9328 | 0.9546 | 0.9167 | 0.9271 | 0.8424 |

The signed posterior-mean objective is mechanically valid but does not pass the
local gate. The pseudo-shift form was nearly neutral. The stronger signed-bias
form improved coverage in some strata and effect-size OOD coverage at the
highest tested weight, but it moved rare-prevalence rank mean farther below
`0.5` on independent SBC data. Rare rank mean changed from `0.4698` under the
previous best local candidate to `0.4587`, `0.4572`, and `0.4554` as signed
bias weight increased.

Do not submit these candidates to LUMI.

The next roadmap step should address holdout-transfer failure in the training
rank objective. The most defensible next candidate is cross-fit or
multi-holdout rank training: evaluate the signed rare-prevalence signal over
rotating held-out simulation folds, average the sign/magnitude across folds,
and only apply a posterior-mean correction when the rare-rank direction is
stable across folds and does not violate medium/high design-information gates.

## Crossfit Rank-Training Follow-Up

The rank-training objective was extended with multi-holdout fold support:

```text
--rank-mean-penalty-holdout-folds
--rank-mean-penalty-crossfit-min-agreement
```

With more than one fold, the training loop evaluates the base rank/design
penalty on each holdout fold, computes rare-prevalence rank-direction vectors
overall and by coefficient, and enables the signed posterior-mean correction
only when fold directions agree above the configured threshold. A second gate
requires medium/high expected design-information rank means to remain within
the design mean tolerance. Single-fold behavior is unchanged.

Focused validation:

```text
python -m py_compile pyhmsc/neural/inference.py examples/run_neural_hmsc_benchmark.py examples/run_neural_hmsc_conditional_calibration.py pyhmsc/neural/conditional_calibration.py
pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q
38 passed
```

Local sanity runs:

```text
/private/tmp/neural_hmsc_v8_crossfit_signed_w025_agree075_local_sanity
/private/tmp/neural_hmsc_v8_crossfit_signed_w025_agree05_local_sanity
/private/tmp/neural_hmsc_v8_crossfit_signed_w01_h025_f4_local_sanity
/private/tmp/neural_hmsc_v8_crossfit_signed_w01_h025_f2_local_sanity
```

Calibrated local SBC comparison:

| Run | Rare rank mean | Rare rank error | Rare coverage 95 | Overall coverage 95 | Design low coverage 95 | Design intermediate coverage 95 | Design high coverage 95 | Effect-size OOD coverage 95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous best guard 0.05, start 0.0 | 0.4698 | 0.0302 | 0.9565 | 0.9311 | 0.9529 | 0.9163 | 0.9242 | 0.8282 |
| single-holdout signed bias 0.25 | 0.4572 | 0.0428 | 0.9662 | 0.9333 | 0.9563 | 0.9150 | 0.9287 | 0.8297 |
| crossfit signed 0.25, holdout 0.5, 4 folds, agreement 0.75 | 0.4375 | 0.0625 | 0.9614 | 0.9156 | 0.9521 | 0.9104 | 0.8842 | 0.8732 |
| crossfit signed 0.25, holdout 0.5, 4 folds, agreement 0.5 | 0.4286 | 0.0714 | 0.9324 | 0.9113 | 0.9525 | 0.9050 | 0.8762 | 0.8789 |
| crossfit signed 0.1, holdout 0.25, 4 folds, agreement 0.75 | 0.4689 | 0.0311 | 0.9614 | 0.9265 | 0.9475 | 0.9104 | 0.9217 | 0.8231 |
| crossfit signed 0.1, holdout 0.25, 2 folds, agreement 1.0 | 0.4622 | 0.0378 | 0.9517 | 0.9311 | 0.9487 | 0.9117 | 0.9329 | 0.8268 |

The crossfit gate prevented neither the rare-rank transfer failure nor the
design-information coverage blocker. The large-holdout variants degraded base
training and high-design coverage. The smaller-holdout variants were less
damaging, but the best rare-rank result was `0.0311`, still worse than the
previous `0.0302` candidate and above the `0.025` rank gate. Intermediate
design coverage remained below the stricter local guard target in all runs.

Do not submit these candidates to LUMI.

The next roadmap step should move away from signed posterior-mean correction
for rare species. A more defensible next candidate is to change the training
data/objective so rare-prevalence rank bias is learned from larger, explicitly
balanced rare-prevalence simulation batches, or to add a separate rare-species
calibration head trained on many simulation batches instead of noisy per-run
holdout rank signs.

## Rare-Balanced Calibration Head Follow-Up

The conditional calibration path now supports an explicit rare-balanced
calibration pool:

```text
--rare-calibration-datasets
--rare-calibration-intercept-mean
```

The simulator also accepts `intercept_mean`, allowing calibration simulations
with rare-prevalence species without shifting every slope coefficient. The
rare head fits a rare-only prevalence-by-coefficient residual mean correction
from the extra calibration pool, then validates shrinkage against the ordinary
in-domain calibration batch. The head writes diagnostics under
`mean_bias_correction`:

```text
rare_balanced_n_observations
rare_balanced_selected_shrinkage
rare_balanced_validation_rank_error
```

Focused validation:

```text
python -m py_compile pyhmsc/neural/simulator.py pyhmsc/neural/conditional_calibration.py examples/run_neural_hmsc_benchmark.py examples/run_neural_hmsc_conditional_calibration.py
pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q
39 passed
```

Local sanity runs:

```text
/private/tmp/neural_hmsc_v8_rare_head_32_i100_local_sanity
/private/tmp/neural_hmsc_v8_rare_head_32_local_sanity
/private/tmp/neural_hmsc_v8_rare_head_32_i250_local_sanity
/private/tmp/neural_hmsc_v8_rank_penalty_rare_head_32_local_sanity
```

Calibrated local SBC comparison:

| Run | Rare rank mean | Rare rank error | Rare coverage 95 | Overall coverage 95 | Design intermediate coverage 95 | Design high coverage 95 | Effect-size OOD coverage 95 | Rare observations | Selected shrinkage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous best guard 0.05, start 0.0 | 0.4698 | 0.0302 | 0.9565 | 0.9311 | 0.9163 | 0.9242 | 0.8282 | 0 | 0.000 |
| rare head, intercept -1.0 | 0.4588 | 0.0412 | 0.9614 | 0.9304 | 0.9133 | 0.9229 | 0.8285 | 1596 | 0.000 |
| rare head, intercept -1.75 | 0.4588 | 0.0412 | 0.9614 | 0.9304 | 0.9133 | 0.9229 | 0.8285 | 3720 | 0.000 |
| rare head, intercept -2.5 | 0.4588 | 0.0412 | 0.9614 | 0.9304 | 0.9133 | 0.9229 | 0.8285 | 5706 | 0.000 |
| previous best plus rare head | 0.4698 | 0.0302 | 0.9565 | 0.9311 | 0.9163 | 0.9242 | 0.8282 | 3720 | 0.000 |

The rare-balanced head is mechanically valid and safely guarded, but it did
not produce a qualifying local improvement. Across intercept means `-1.0`,
`-1.75`, and `-2.5`, the validation gate selected zero shrinkage. Combined
with the previous best training penalty, it also selected zero shrinkage and
left the benchmark unchanged. This confirms that the current rare-balanced
pool residual direction is not transferable under the local validation gate.

Do not submit this candidate to LUMI.

The next roadmap step should diagnose why rare-balanced residuals disagree
with independent SBC rare-rank behavior. The most useful next implementation
is instrumentation: record candidate rare-head offsets, validation gate
metrics, and rare-pool prevalence/effect summaries in benchmark metadata, then
run a small local diagnostic sweep before changing the objective again.

## Rare-Head Diagnostics Follow-Up

The rare-balanced calibration head now records a diagnostic payload in
`mean_bias_correction.rare_balanced_diagnostics`:

```text
candidate_offsets
selected_offsets
shrinkage_grid
rare_pool
rare_pool_by_coefficient
validation
```

Focused validation:

```text
python -m py_compile pyhmsc/neural/conditional_calibration.py examples/run_neural_hmsc_benchmark.py
pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q
39 passed
```

Diagnostic local run:

```text
/private/tmp/neural_hmsc_v8_rare_head_diagnostics_local_sanity
```

Rare-head metadata summary:

| Diagnostic | Value |
| --- | ---: |
| Rare observations | 3720 |
| Selected shrinkage | 0.000 |
| Validation rare rank error at zero | 0.0369 |
| Candidate rare intercept offset | -0.2235 |
| Candidate rare x1 offset | -0.0146 |
| Candidate rare x2 offset | 0.0098 |
| Rare-pool prevalence mean | 0.1410 |
| Rare-pool prevalence median | 0.1000 |
| Rare species fraction | 0.5167 |

Shrinkage-grid validation:

| Shrinkage | Rare rank error | Overall rank error | Coverage 95 | Design intermediate coverage 95 | Design high coverage 95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.0369 | 0.0001 | 0.9861 | 0.9917 | 0.9967 |
| 0.125 | 0.0437 | 0.0003 | 0.9861 | 0.9917 | 0.9967 |
| 0.250 | 0.0504 | 0.0006 | 0.9861 | 0.9917 | 0.9967 |
| 0.375 | 0.0571 | 0.0008 | 0.9856 | 0.9917 | 0.9967 |
| 0.500 | 0.0638 | 0.0010 | 0.9856 | 0.9917 | 0.9967 |
| 0.750 | 0.0767 | 0.0015 | 0.9856 | 0.9917 | 0.9967 |
| 1.000 | 0.0893 | 0.0019 | 0.9858 | 0.9917 | 0.9967 |

Rare-pool residual summaries:

| Coefficient | Count | Residual mean | Standardized residual mean | Rank mean | Candidate offset |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intercept | 1240 | -0.4604 | -1.0354 | 0.2423 | -0.2235 |
| x1 | 1240 | -0.0150 | -0.0359 | 0.4904 | -0.0146 |
| x2 | 1240 | 0.0100 | 0.0228 | 0.5075 | 0.0098 |

The instrumentation explains why the rare head stays off. The rare-balanced
pool induces a strong negative intercept correction, but the ordinary
validation batch already has high coverage and its rare-rank error worsens
monotonically as the candidate correction is applied. The local independent
SBC rare-rank failure is therefore not solved by this simple intercept-shift
rare pool.

Do not submit this candidate to LUMI.

The next roadmap step should change the rare-calibration simulation design
rather than relax the validation gate. The likely next candidate is a
stratified rare-calibration pool that matches the SBC rare-failure context:
balanced by rare prevalence, coefficient identity, and design-information
tertile, with intercept-shift, low-detection, and small-sample rare regimes
recorded separately in diagnostics.

## Stratified Rare-Calibration Pool Follow-Up

The rare-calibration simulation design was changed from a single intercept
shift to a stratified rare pool. The simulator now supports:

```text
intercept_mean
detection_probability
sample_fraction
```

The benchmark runner exposes:

```text
--rare-calibration-regimes
--rare-calibration-detection-probability
--rare-calibration-sample-fraction
```

The default rare-calibration regimes are:

```text
intercept_shift
low_detection
small_sample
```

The rare-head fitter now balances candidate offsets by rare prevalence,
coefficient identity, design-information tertile, and rare regime. It computes
cell-level residual/rank summaries for each regime × design-tertile ×
coefficient cell, then averages cell offsets instead of pooling all rare
residuals.

Focused validation:

```text
python -m py_compile pyhmsc/neural/simulator.py pyhmsc/neural/conditional_calibration.py examples/run_neural_hmsc_benchmark.py examples/run_neural_hmsc_conditional_calibration.py
pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q
39 passed
```

Local sanity run:

```text
/private/tmp/neural_hmsc_v8_stratified_rare_pool_local_sanity
```

Rare-head metadata summary:

| Diagnostic | Value |
| --- | ---: |
| Rare observations | 2436 |
| Selected shrinkage | 1.000 |
| Validation rare rank error at zero | 0.0369 |
| Validation rare rank error at selected shrinkage | 0.0277 |
| Candidate rare intercept offset | 0.0323 |
| Candidate rare x1 offset | 0.0023 |
| Candidate rare x2 offset | 0.0040 |
| Rare-pool prevalence mean | 0.1632 |
| Rare-pool prevalence median | 0.1500 |
| Rare species fraction | 0.3383 |

Rare-pool balance:

| Balance axis | Counts |
| --- | --- |
| Regime batches | intercept-shift 11, low-detection 11, small-sample 10 |
| Rare observations by regime | intercept-shift 1221, low-detection 621, small-sample 594 |
| Rare observations by design tertile | low 812, intermediate 812, high 812 |
| Rare observations by coefficient | Intercept 812, x1 812, x2 812 |

Shrinkage-grid validation:

| Shrinkage | Rare rank error | Overall rank error | Coverage 95 | Design intermediate coverage 95 | Design high coverage 95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.0369 | 0.0001 | 0.9861 | 0.9917 | 0.9967 |
| 0.125 | 0.0357 | 0.0001 | 0.9858 | 0.9917 | 0.9967 |
| 0.250 | 0.0346 | 0.0000 | 0.9858 | 0.9917 | 0.9967 |
| 0.375 | 0.0334 | 0.0000 | 0.9858 | 0.9917 | 0.9967 |
| 0.500 | 0.0323 | 0.0001 | 0.9858 | 0.9917 | 0.9967 |
| 0.750 | 0.0300 | 0.0001 | 0.9858 | 0.9917 | 0.9967 |
| 1.000 | 0.0277 | 0.0002 | 0.9858 | 0.9917 | 0.9967 |

Calibrated local SBC comparison:

| Run | Rare rank mean | Rare rank error | Rare coverage 95 | Overall coverage 95 | Design intermediate coverage 95 | Design high coverage 95 | Effect-size OOD coverage 95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous rare-head diagnostic | 0.4588 | 0.0412 | 0.9614 | 0.9304 | 0.9133 | 0.9229 | 0.8285 |
| stratified rare pool | 0.4504 | 0.0496 | 0.9614 | 0.9297 | 0.9125 | 0.9213 | 0.8285 |

The stratified rare pool is mechanically valid and solved the immediate design
problem: the pool is balanced across design tertiles and coefficients, and the
ordinary validation gate selected a nonzero correction. However, independent
SBC rare rank worsened from `0.0412` to `0.0496`. This means the ordinary
calibration validation batch is not sufficient for rare-head acceptance.

Do not submit this candidate to LUMI.

The next roadmap step should add an independent rare-head validation gate.
Before applying nonzero rare-head offsets, evaluate the candidate on a separate
SBC-style in-domain validation pool with rare/prevalence/design stratification,
and require non-degradation of independent rare-rank error and
intermediate/high design coverage.

## Independent Rare-Head Gate Follow-Up

The rare-head validation path now accepts an independent rare-validation pool
through `--rare-validation-datasets`. Nonzero rare-head offsets are selected on
the ordinary calibration validation pool first, then re-evaluated on the
independent pool. If the independent gate fails, the candidate offsets remain in
diagnostics but the serialized and applied offsets are reset to zero.

Focused validation:

```text
python -m py_compile pyhmsc/neural/conditional_calibration.py examples/run_neural_hmsc_benchmark.py examples/run_neural_hmsc_conditional_calibration.py
pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q
39 passed
```

Local sanity run:

```text
/private/tmp/neural_hmsc_v8_independent_rare_gate_local_sanity
```

The run used 32 rare-calibration datasets and 32 independent rare-validation
datasets across the same intercept-shift, low-detection, and small-sample rare
regimes.

| Run | Rare-head shrinkage | Independent reset | Rare rank mean | Rare rank error | Rare coverage 95 | Overall coverage 95 | Design intermediate coverage 95 | Design high coverage 95 | Effect-size OOD coverage 95 | Combined OOD coverage 95 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stratified rare pool | 1.000 | false | 0.4504 | 0.0496 | 0.9614 | 0.9297 | 0.9125 | 0.9213 | 0.8346 | 0.7832 |
| independent rare gate | 0.000 | true | 0.4588 | 0.0412 | 0.9614 | 0.9304 | 0.9133 | 0.9229 | 0.8285 | 0.7825 |

Ordinary rare-head validation still selected the same candidate offsets:

| Offset | Candidate | Applied after independent gate |
| --- | ---: | ---: |
| rare intercept | 0.0323 | 0.0000 |
| rare x1 | 0.0023 | 0.0000 |
| rare x2 | 0.0040 | 0.0000 |

Independent validation metrics:

| Metric | Zero offsets | Candidate offsets |
| --- | ---: | ---: |
| rare rank error | 0.0163 | 0.0107 |
| overall rank error | 0.0724 | 0.0703 |
| coverage 95 | 0.7894 | 0.7900 |
| intermediate-design coverage 95 | 0.7963 | 0.7988 |
| high-design coverage 95 | 0.6996 | 0.7000 |

The candidate improved independent rare-rank error, but the independent
validation pool did not clear the absolute coverage floors. The gate therefore
reset rare-head shrinkage to `0.0` and preserved the previous local SBC
behavior rather than applying an offset that is not coverage-qualified.

Do not submit this candidate to LUMI.

The next roadmap step should address the rare-validation regime coverage
failure before revisiting rare-head mean offsets. The likely direction is a
rare-regime-aware scale or normalization correction evaluated on the independent
rare-validation pool, with the mean-offset gate remaining strict.

## Rare-Validation Scale Follow-Up

The rare-validation coverage fix was implemented as a scale-side correction,
not a mean-offset correction. When `--rare-validation-datasets` are supplied,
conditional calibration now fits a design-stratum log-scale multiplier from the
independent rare-validation pool. The multiplier is accepted only if the
independent rare-validation pool clears absolute coverage floors for overall,
rare-prevalence, intermediate-design, and high-design strata without material
rank degradation. The independent rare-head mean-offset gate remains strict;
failed nonzero rare-head mean offsets are still reset to zero.

Focused validation:

```text
pytest tests/test_neural_hmsc_conditional_calibration.py -q
17 passed

pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q
40 passed
```

Local sanity run:

```text
/private/tmp/neural_hmsc_v8_rare_validation_scale_local_sanity
```

The rare-validation scale selected full shrinkage with design-stratum
multipliers:

| Design stratum | Multiplier |
| --- | ---: |
| low | 1.187 |
| intermediate | 2.643 |
| high | 3.320 |

Independent rare-validation scale metrics:

| Metric | Before scale | After scale |
| --- | ---: | ---: |
| overall coverage 95 | 0.6857 | 0.9001 |
| rare coverage 95 | 0.8785 | 0.9407 |
| intermediate-design coverage 95 | 0.5240 | 0.9001 |
| high-design coverage 95 | 0.4750 | 0.9000 |
| overall rank error | 0.0768 | 0.0687 |
| rare rank error | 0.0259 | 0.0241 |

Local SBC comparison:

| Run | Rare scale shrinkage | Rare rank mean | Rare rank error | Rare coverage 95 | Overall coverage 95 | Design intermediate coverage 95 | Design high coverage 95 | Effect-size OOD coverage 95 | Combined OOD coverage 95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| independent rare gate | 0.000 | 0.4588 | 0.0412 | 0.9614 | 0.9304 | 0.9133 | 0.9229 | 0.8285 | 0.7825 |
| rare-validation scale | 1.000 | 0.4655 | 0.0345 | 0.9903 | 0.9921 | 0.9904 | 1.0000 | 0.9119 | 0.8767 |

This fixes the immediate rare-validation coverage failure and improves
rare-prevalence rank mean and OOD coverage, but it is too conservative for
promotion. In-domain coverage is inflated to `0.9921`, high-design coverage is
`1.0000`, rank variance remains compressed, and combined-shift OOD coverage is
still below `0.90`.

Do not submit this candidate to LUMI yet.

The next roadmap step should constrain the rare-validation scale correction:
target the failed low-detection/small-sample regimes and high-risk design
contexts more selectively, add an in-domain overcoverage/rank-variance guard,
and retest locally before any five-seed LUMI comparison.

## Constrained Rare-Validation Scale Follow-Up

The rare-validation scale correction now uses a support-excess activation and
an in-domain guard. The correction is still fit from the independent
rare-validation pool, but the design-stratum multiplier is only activated for
coefficients that exceed the in-domain support-excess threshold. Candidate
shrinkage is rejected if it over-inflates the original in-domain calibration
pool or compresses in-domain rank variance beyond tolerance.

Focused validation:

```text
pytest tests/test_neural_hmsc_conditional_calibration.py -q
17 passed

pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q
40 passed
```

Local sanity run:

```text
/private/tmp/neural_hmsc_v8_constrained_rare_validation_scale_local_sanity
```

The constrained gate selected zero shrinkage:

| Run | Rare scale shrinkage | Active fraction in-domain / validation | Rare rank error | Rare coverage 95 | Overall coverage 95 | Overall rank variance | Design high coverage 95 | Effect-size OOD coverage 95 | Combined OOD coverage 95 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| independent rare gate | 0.000 | 0.000 / 0.000 | 0.0412 | 0.9614 | 0.9304 | 0.0839 | 0.9229 | 0.8285 | 0.7825 |
| always-on rare-validation scale | 1.000 | 0.000 / 0.000 | 0.0345 | 0.9903 | 0.9921 | 0.0362 | 1.0000 | 0.9119 | 0.8767 |
| constrained rare-validation scale | 0.000 | 0.033 / 0.135 | 0.0412 | 0.9614 | 0.9304 | 0.0839 | 0.9229 | 0.8285 | 0.7825 |

Shrinkage-grid diagnostics:

| Shrinkage | Validation overall coverage 95 | Validation intermediate coverage 95 | Validation high coverage 95 | In-domain overall coverage 95 | In-domain high coverage 95 | In-domain rank variance | Accepted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.000 | 0.6857 | 0.5240 | 0.4750 | 0.9500 | 0.9371 | 0.0815 | baseline |
| 0.250 | 0.7235 | 0.5891 | 0.5405 | 0.9742 | 0.9706 | 0.0708 | false |
| 0.500 | 0.7638 | 0.6568 | 0.6060 | 0.9861 | 0.9895 | 0.0611 | false |
| 0.750 | 0.8038 | 0.7323 | 0.6667 | 0.9908 | 1.0000 | 0.0524 | false |
| 1.000 | 0.8421 | 0.8017 | 0.7298 | 0.9939 | 1.0000 | 0.0449 | false |

This confirms the constrained guard works, but the support-excess gate alone is
too weak to recover rare-validation coverage before the in-domain guard fails.
The always-on scale fixed rare-validation coverage by making the whole
posterior too wide; the constrained version prevents that but reverts to the
previous qualified-in-domain behavior.

Do not submit this candidate to LUMI.

The next roadmap step should replace the support-excess-only activation with a
more discriminative low-detection/small-sample regime proxy. Candidate options
are a learned undercoverage classifier from rare-validation features, a
prevalence-by-effective-sample-size scale gate, or a two-part correction that
separates rare-regime OOD coverage from in-domain design strata.

## Community-Occupancy Rare-Regime Proxy Follow-Up

The rare-validation scale activation was extended from support-excess alone to
an observable rare-regime proxy:

- support-excess activation for coefficients outside calibration support
- rare/intermediate prevalence by design-information activation
- low community-occupancy activation, using the lower tail of in-domain
  community occupancy as the threshold

This targets low-detection and small-sample rare-validation regimes without
using hidden regime labels at application time.

Focused validation:

```text
pytest tests/test_neural_hmsc_conditional_calibration.py -q
17 passed

pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q
40 passed
```

Local sanity run:

```text
/private/tmp/neural_hmsc_v8_community_proxy_rare_validation_scale_local_sanity
```

Local comparison:

| Run | Rare scale shrinkage | Active fraction in-domain / validation | Rare rank error | Rare coverage 95 | Overall coverage 95 | Overall rank variance | Design high coverage 95 | Effect-size OOD coverage 95 | Combined OOD coverage 95 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| independent rare gate | 0.000 | 0.000 / 0.000 | 0.0412 | 0.9614 | 0.9304 | 0.0839 | 0.9229 | 0.8285 | 0.7825 |
| always-on rare-validation scale | 1.000 | 0.000 / 0.000 | 0.0345 | 0.9903 | 0.9921 | 0.0362 | 1.0000 | 0.9119 | 0.8767 |
| support-constrained scale | 0.000 | 0.033 / 0.135 | 0.0412 | 0.9614 | 0.9304 | 0.0839 | 0.9229 | 0.8285 | 0.7825 |
| community-occupancy proxy | 0.000 | 0.048 / 1.000 | 0.0412 | 0.9614 | 0.9304 | 0.0839 | 0.9229 | 0.8285 | 0.7825 |

Shrinkage-grid diagnostics:

| Shrinkage | Validation overall coverage 95 | Validation intermediate coverage 95 | Validation high coverage 95 | In-domain overall coverage 95 | In-domain high coverage 95 | In-domain rank variance | Accepted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.000 | 0.6857 | 0.5240 | 0.4750 | 0.9500 | 0.9371 | 0.0815 | baseline |
| 0.250 | 0.7394 | 0.6167 | 0.5857 | 0.9694 | 0.9665 | 0.0738 | false |
| 0.500 | 0.7961 | 0.7133 | 0.6952 | 0.9806 | 0.9874 | 0.0667 | false |
| 0.750 | 0.8506 | 0.8150 | 0.7952 | 0.9858 | 0.9916 | 0.0600 | false |
| 1.000 | 0.9001 | 0.9001 | 0.9000 | 0.9894 | 1.0000 | 0.0540 | false |

The community-occupancy proxy correctly identifies the rare-validation stress
context: full shrinkage clears the independent rare-validation coverage floors.
However, that same correction still over-inflates the in-domain high-design
stratum to `1.0000`, so the in-domain guard rejects it and selected shrinkage
remains `0.0`.

Do not submit this candidate to LUMI.

The next roadmap step should change the correction shape, not just the
activation. A single positive scale multiplier by design stratum is too blunt.
The next candidate should use regime-proxy-conditioned slope caps or a
two-part correction that separates high-design in-domain coefficients from
low-community rare-validation coefficients, with the same independent
rare-validation and in-domain guards.

## Thresholded Low-Community Scale-Shape Follow-Up

The rare-validation scale correction was changed from a half-on sigmoid gate to
a thresholded two-part shape:

- support-excess activation is zero at the in-domain support boundary and grows
  only outside support
- low-community activation is zero at the in-domain low-community threshold and
  grows only for lower community occupancy
- low-community stress can activate the design-stratum scale directly, so
  common high-design coefficients in low-detection/small-sample batches are not
  damped

Focused validation:

```text
pytest tests/test_neural_hmsc_conditional_calibration.py -q
17 passed

pytest tests/test_neural_hmsc_public_api.py tests/test_neural_hmsc_conditional_calibration.py tests/test_neural_hmsc_lumi_workflow.py -q
40 passed
```

Local sanity run:

```text
/private/tmp/neural_hmsc_v8_thresholded_community_direct_scale_local_sanity
```

Local comparison:

| Run | Rare scale shrinkage | Active fraction in-domain / validation | Rare rank error | Rare coverage 95 | Overall coverage 95 | Overall rank variance | Design high coverage 95 | Effect-size OOD coverage 95 | Combined OOD coverage 95 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| independent rare gate | 0.000 | 0.000 / 0.000 | 0.0412 | 0.9614 | 0.9304 | 0.0839 | 0.9229 | 0.8285 | 0.7825 |
| always-on rare-validation scale | 1.000 | 0.000 / 0.000 | 0.0345 | 0.9903 | 0.9921 | 0.0362 | 1.0000 | 0.9119 | 0.8767 |
| thresholded damped proxy | 0.000 | 0.013 / 1.000 | 0.0412 | 0.9614 | 0.9304 | 0.0839 | 0.9229 | 0.8285 | 0.7825 |
| thresholded direct proxy | 1.000 | 0.013 / 1.000 | 0.0405 | 0.9614 | 0.9311 | 0.0834 | 0.9242 | 0.8324 | 0.8015 |

Selected rare-validation scale multipliers:

| Design stratum | Multiplier |
| --- | ---: |
| low | 1.187 |
| intermediate | 2.643 |
| high | 3.320 |

Independent gate metrics:

| Metric | Before scale | Selected scale |
| --- | ---: | ---: |
| validation overall coverage 95 | 0.6857 | 0.9001 |
| validation rare coverage 95 | 0.8785 | 0.9407 |
| validation intermediate-design coverage 95 | 0.5240 | 0.9001 |
| validation high-design coverage 95 | 0.4750 | 0.9000 |
| in-domain overall coverage 95 | 0.9500 | 0.9508 |
| in-domain high-design coverage 95 | 0.9371 | 0.9392 |
| in-domain rank variance | 0.0815 | 0.0811 |

This resolves the immediate scale-shape problem: the independent rare-validation
coverage gate passes, while the in-domain overcoverage/rank-variance guard also
passes. The local SBC result is still not LUMI-ready because OOD coverage
remains low: effect-size OOD coverage is `0.8324` and combined-shift OOD
coverage is `0.8015`.

Do not submit this candidate to LUMI yet.

The next roadmap step should evaluate whether the thresholded low-community
scale should be combined with the learned OOD objective or whether OOD
inflation must be refit after this scale correction. Run a local OOD-focused
sanity check first, then decide whether a five-seed LUMI comparison is
justified.

## OOD-Focused Sanity Follow-Up

The OOD-focused local check compared the selected thresholded low-community
scale with a higher-cap learned OOD inflation run:

```text
/private/tmp/neural_hmsc_v8_thresholded_community_direct_scale_local_sanity
/private/tmp/neural_hmsc_v8_thresholded_scale_ood_cap16_local_sanity
```

The cap-16 run used twice as many OOD calibration batches per regime, more OOD
objective epochs, and `--conditional-calibration-ood-uncertainty-max-multiplier
16`.

| Run | Rare scale shrinkage | In-domain coverage 95 | In-domain rank variance | Rare rank error | Rare coverage 95 | Covariate-shift coverage 95 / inflation mean | Effect-size coverage 95 / inflation mean | Combined-shift coverage 95 / inflation mean |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| cap-8 thresholded scale | 1.000 | 0.9311 | 0.0834 | 0.0405 | 0.9614 | 0.9146 / 3.60 | 0.8324 / 1.51 | 0.8015 / 3.34 |
| cap-16 OOD-focused | 1.000 | 0.9303 | 0.0839 | 0.0406 | 0.9614 | 0.9128 / 6.06 | 0.8253 / 1.54 | 0.7953 / 5.23 |

The higher OOD cap and larger OOD calibration pool did not improve OOD
coverage. It increased inflation for covariate and combined shifts, but
coverage slightly worsened. Effect-size shift remained the main failure:
coverage stayed around `0.83` and mean OOD inflation stayed near `1.5`.

Reconstructing the deterministic OOD batches showed why the rare-validation
scale is not the effect-size fix:

| OOD regime | Rare-scale activation mean | Rare-scale active fraction | Community occupancy mean | Support-excess mean |
| --- | ---: | ---: | ---: | ---: |
| covariate shift | 0.4786 | 0.5029 | 0.5005 | 1.1614 |
| effect-size shift | 0.0374 | 0.0383 | 0.5015 | 0.0422 |
| combined shift | 0.4422 | 0.4665 | 0.5005 | 0.8114 |

The thresholded low-community scale activates for covariate and combined
shifts because they move outside support, but it barely activates for pure
effect-size shift. The remaining OOD blocker is therefore the learned OOD
effect-size objective, not rare-validation scale shape or the OOD multiplier
cap alone.

Do not submit this candidate to LUMI.

The next roadmap step should implement a post-scale or final-multiplier-aware
OOD objective focused on effect-size coverage. The objective should evaluate
the final calibrated multiplier after rare-validation scale is applied, include
effect-size and combined-shift coverage floors as acceptance gates, and keep the
existing in-domain and rare-validation gates.

## Final-Multiplier-Aware OOD Objective Follow-Up

The learned OOD objective was updated with a second refinement pass after
rare-validation scale selection. This pass initializes from the first OOD fit,
evaluates the final multiplier after the rare-validation scale multiplier is
applied, and adds targeted coverage-floor penalties for `effect_size_shift` and
`combined_shift` batches. The in-domain rank/coverage gate and independent
rare-validation scale gate remain in force.

Focused validation:

```text
python -m py_compile pyhmsc/neural/conditional_calibration.py \
  examples/run_neural_hmsc_benchmark.py \
  examples/run_neural_hmsc_conditional_calibration.py

pytest tests/test_neural_hmsc_conditional_calibration.py -q
17 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
40 passed

git diff --check
passed
```

Local sanity run:

```text
/private/tmp/neural_hmsc_v8_final_aware_ood_local_sanity
```

Overall calibrated SBC summary:

| Domain | Coverage 95 | Rank mean | Rank variance | Acceptance |
| --- | ---: | ---: | ---: | --- |
| in distribution | 0.9301 | 0.4946 | 0.0838 | pass |
| covariate shift | 0.9143 | 0.5039 | 0.0588 | diagnostic |
| effect-size shift | 0.8256 | 0.4979 | 0.1078 | fail |
| combined shift | 0.8036 | 0.5001 | 0.0895 | fail |

The implementation is wired correctly enough to preserve the existing
in-domain acceptance gate locally, but it does not solve the OOD failure.
Effect-size coverage remains near the previous `0.83` level and combined-shift
coverage remains near `0.80`. This means the blocker is no longer just whether
the OOD objective sees the final multiplier; the current effect-gated
support/effect curve cannot learn enough effect-size inflation under the
existing in-domain constraints.

Do not submit this candidate to LUMI.

The next roadmap step should instrument final-multiplier diagnostics for the
OOD fit before changing the objective again: record learned effect-gate
activation, final multiplier quantiles, post-scale multiplier quantiles,
coverage by effect-size quantile, and in-domain gate penalties for each OOD
regime. Then redesign the effect-size branch using those diagnostics, likely as
an effect-shift-specific scale head or domain-classifier-gated multiplier
rather than another global support/effect softplus curve.

## OOD Final-Multiplier Diagnostics

The conditional calibration metadata now records final-multiplier diagnostics
under `ood_objective.final_multiplier_diagnostics`. The diagnostics are emitted
for each OOD calibration regime and include:

- learned effect-gate activation summaries
- learned OOD inflation summaries
- rare-validation post-scale multiplier summaries
- final multiplier summaries
- coverage by effect-size quantile
- in-domain gate penalty components attached to each OOD regime

Focused validation:

```text
python -m py_compile pyhmsc/neural/conditional_calibration.py \
  examples/run_neural_hmsc_benchmark.py \
  examples/run_neural_hmsc_conditional_calibration.py

pytest tests/test_neural_hmsc_conditional_calibration.py -q
17 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
40 passed
```

Benchmark metadata smoke run:

```text
/private/tmp/neural_hmsc_ood_diagnostics_smoke
```

The smoke manifest confirmed diagnostics for `effect_size_shift` and
`combined_shift`, including effect-gate, learned-inflation, rare-post-scale,
final-multiplier, effect-quantile coverage, and in-domain gate fields. The run
was intentionally tiny and emitted NumPy empty-slice warnings, so it should be
treated only as a metadata-shape check, not as a calibration result.

The next roadmap step should run the production-like local sanity workflow
again and inspect the new diagnostics from the failed effect-size and
combined-shift regimes. Use those diagnostics to choose the next objective
shape, likely an effect-shift-specific scale head or domain-classifier-gated
multiplier.

## Production-Like OOD Diagnostic Local Sanity

The production-like local sanity workflow was rerun with diagnostics-enabled
metadata:

```text
/private/tmp/neural_hmsc_v8_ood_diagnostics_local_sanity
```

Held-out SBC coverage remained unchanged from the previous final-aware run:

| Domain | Held-out SBC coverage 95 | Rank mean | Rank variance | Acceptance |
| --- | ---: | ---: | ---: | --- |
| in distribution | 0.9301 | 0.4946 | 0.0838 | pass |
| covariate shift | 0.9143 | 0.5039 | 0.0588 | diagnostic |
| effect-size shift | 0.8256 | 0.4979 | 0.1078 | fail |
| combined shift | 0.8036 | 0.5001 | 0.0895 | fail |

The new `final_multiplier_diagnostics` summarize the OOD calibration regimes
used by the objective. They show different failure modes for the two failed
regimes:

| Regime | Diagnostic coverage | Floor shortfall | Effect gate mean / median | OOD inflation mean / median | Rare post-scale mean / median | Final multiplier mean / median |
| --- | ---: | ---: | --- | --- | --- | --- |
| effect-size shift | 0.8494 | 0.0506 | 0.5996 / 0.8328 | 1.4597 / 1.0255 | 1.0174 / 1.0000 | 1.2025 / 0.8035 |
| combined shift | 0.8172 | 0.0828 | 0.7463 / 0.9892 | 3.8320 / 1.4726 | 1.5665 / 1.1396 | 5.7567 / 1.4315 |

Effect-quantile coverage:

| Regime | Low/mid effect coverage | High effect coverage | Interpretation |
| --- | ---: | ---: | --- |
| effect-size shift | q2 `0.8556`, q3 `0.7844` | q4 `0.9022` | the current curve opens mostly for the largest effects, while the middle effect stratum remains under-inflated |
| combined shift | q2 `0.8244`, q3 `0.8489` | q4 `0.7711` | even large final multipliers do not protect the highest-effect combined-shift coefficients |

In-domain gate diagnostics were stable across OOD regimes: in-domain coverage
was `0.9506`, mean group loss `0.00043`, and max group loss `0.00430`. The
extra-inflation penalties were nonzero (`extra_inflation_over_1_05_loss`
`0.8885`, max group extra-cap loss `0.4010`), which means the current gate is
already constraining broad in-domain inflation.

Conclusion: a single global support/effect softplus curve is the wrong shape.
Pure effect-size shift needs targeted inflation for middle effect-signal
coefficients without relying on support excess. Combined shift needs a separate
high-effect correction because support/rare post-scale inflation is already
large but still insufficient for high-effect coverage.

Do not submit this candidate to LUMI.

The next roadmap step should implement an experimental effect-shift-specific
scale head or domain-classifier-gated multiplier. The objective should target
effect-quantile coverage directly, with separate constraints for pure
effect-size shift and combined shift, while preserving the in-domain and
rare-validation gates.

## Experimental Effect-Shift Head

Implemented an experimental context-gated effect-shift scale head inside the
version 8 learned OOD inflation path. The head adds two learned positive
log-scale components:

- a pure-effect gate that increases with effect signal and is suppressed by
  support excess
- a combined-shift gate that increases with both effect signal and support
  excess

The OOD objective now also includes differentiable effect-quantile coverage
losses, so pure `effect_size_shift` and `combined_shift` regimes can train
against coverage shortfalls inside effect-signal quantile bins rather than only
through a global regime-level loss. Existing in-domain and rare-validation
gates remain active.

Focused validation:

```text
python -m py_compile pyhmsc/neural/conditional_calibration.py \
  examples/run_neural_hmsc_benchmark.py \
  examples/run_neural_hmsc_conditional_calibration.py

pytest tests/test_neural_hmsc_conditional_calibration.py -q
17 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
40 passed

git diff --check
passed
```

Benchmark metadata smoke run:

```text
/private/tmp/neural_hmsc_effect_shift_head_smoke
```

The smoke manifest confirmed the serialized
`support.ood_uncertainty.curve.effect_shift_head` block and OOD diagnostic
domains for `effect_size_shift` and `combined_shift`. The run was intentionally
tiny and emitted NumPy empty-slice warnings, so it is only a metadata/application
shape check.

Do not submit this candidate to LUMI yet.

The next roadmap step should run the production-like local sanity workflow with
the experimental effect-shift head enabled and compare held-out OOD coverage,
effect-quantile coverage, final-multiplier diagnostics, and in-domain/rare
validation gates against the previous final-aware run.

## Effect-Shift Head Local Sanity

The production-like local sanity workflow was rerun with the experimental
effect-shift head enabled:

```text
/private/tmp/neural_hmsc_v8_effect_shift_head_local_sanity
```

Held-out SBC comparison against the previous final-aware run:

| Domain | Final-aware coverage 95 | Effect-head coverage 95 | Delta | Effect-head rank variance |
| --- | ---: | ---: | ---: | ---: |
| in distribution | 0.9301 | 0.9311 | +0.0010 | 0.0810 |
| covariate shift | 0.9143 | 0.9199 | +0.0056 | 0.0560 |
| effect-size shift | 0.8256 | 0.8656 | +0.0400 | 0.0957 |
| combined shift | 0.8036 | 0.8333 | +0.0297 | 0.0832 |

Rare-validation scale remained selected at full shrinkage. The independent
rare-validation gate still passed with validation overall coverage `0.9001`,
intermediate-design coverage `0.9001`, high-design coverage `0.9000`, and
in-domain guard overall coverage `0.9508`.

OOD calibration diagnostics:

| Regime | Final-aware diagnostic coverage | Effect-head diagnostic coverage | Final multiplier median | Final multiplier q95 | Key effect-bin change |
| --- | ---: | ---: | ---: | ---: | --- |
| effect-size shift | 0.8494 | 0.8761 | 0.8625 | 6.5668 | q3 coverage improved from `0.7844` to `0.8444`; q4 rose from `0.9022` to `0.9800` |
| combined shift | 0.8172 | 0.8400 | 1.8405 | 19.7134 | q4 coverage improved from `0.7711` to `0.8600`; q3 rose from `0.8489` to `0.8689` |

The head learned nonzero components:

| Component | Value |
| --- | ---: |
| pure effect intercept | -1.7220 |
| pure effect linear | 3.0579 |
| pure support suppression | 0.5922 |
| pure log amplitude | 0.1995 |
| combined intercept | -1.6855 |
| combined effect linear | 2.5551 |
| combined support linear | 2.4796 |
| combined log amplitude | 0.2106 |

The improvement is real but not sufficient. Held-out OOD coverage still misses
the `0.90` floor for both effect-size and combined shifts. The new head also
increased in-domain extra-inflation pressure: the diagnostic
`extra_inflation_over_1_05_loss` rose from `0.8885` to `2.9035`, and max
group extra-cap loss rose from `0.4010` to `0.6523`. In-domain overall
acceptance still passes, but the head is now pressing against the in-domain
inflation guard.

Do not submit this candidate to LUMI.

The next roadmap step should constrain the effect-shift head rather than simply
increase its strength. Likely directions are a scheduled/two-stage head fit,
stronger in-domain extra-inflation normalization, or effect-bin-specific
amplitude caps that improve OOD middle/high-effect coverage without increasing
in-domain extra-inflation penalties.

## Constrained Effect-Shift Head

The experimental effect-shift head was constrained without adding new trainable
parameters. The pure-effect branch now has a fixed log-amplitude cap and a
high-effect taper so it can target middle-effect undercoverage without
overinflating the highest pure-effect bin. The combined branch now has a
separate fixed log-amplitude cap plus a support-excess activation gate, so it
is less likely to act like broad in-domain effect inflation.

The serialized metadata kind is now
`constrained_context_gated_effect_quantile_scale`, with the fixed caps and
shape-gate constants recorded in
`support.ood_uncertainty.curve.effect_shift_head`.

Focused validation:

```text
python -m py_compile pyhmsc/neural/conditional_calibration.py \
  examples/run_neural_hmsc_benchmark.py \
  examples/run_neural_hmsc_conditional_calibration.py

pytest tests/test_neural_hmsc_conditional_calibration.py -q
17 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
40 passed

git diff --check
passed
```

Benchmark metadata smoke run:

```text
/private/tmp/neural_hmsc_constrained_effect_head_smoke
```

The smoke manifest confirmed the constrained head metadata, including
`pure_log_cap`, `combined_log_cap`, the pure high-effect taper, and the combined
support gate. The run was intentionally tiny and emitted NumPy empty-slice
warnings, so it is only a metadata/application shape check.

The next roadmap step should run the production-like local sanity workflow for
the constrained effect-shift head and compare it against the unconstrained
effect-head and previous final-aware runs.

## Constrained Effect-Shift Head Local Sanity

The production-like local sanity workflow was rerun for the constrained
effect-shift head:

```text
/private/tmp/neural_hmsc_v8_constrained_effect_head_local_sanity
```

Held-out SBC comparison:

| Domain | Final-aware coverage 95 | Unconstrained head | Constrained head | Constrained vs final-aware |
| --- | ---: | ---: | ---: | ---: |
| in distribution | 0.9301 | 0.9311 | 0.9306 | +0.0004 |
| covariate shift | 0.9143 | 0.9199 | 0.9197 | +0.0054 |
| effect-size shift | 0.8256 | 0.8656 | 0.8621 | +0.0365 |
| combined shift | 0.8036 | 0.8333 | 0.8293 | +0.0257 |

The constrained head kept most, but not all, of the OOD improvement from the
unconstrained head. It did not reach the `0.90` OOD floor.

OOD calibration diagnostics:

| Regime | Final-aware diagnostic coverage | Unconstrained head | Constrained head | Key constrained-head effect-bin coverage |
| --- | ---: | ---: | ---: | --- |
| effect-size shift | 0.8494 | 0.8761 | 0.8733 | q2 `0.8433`, q3 `0.8356`, q4 `0.9711` |
| combined shift | 0.8172 | 0.8400 | 0.8411 | q2 `0.8222`, q3 `0.8756`, q4 `0.8444` |

The constraint modestly reduced extra-inflation pressure relative to the
unconstrained head, but not enough to restore the final-aware gate profile:

| Diagnostic | Final-aware | Unconstrained head | Constrained head |
| --- | ---: | ---: | ---: |
| in-domain extra inflation over 1.05 loss | 0.8885 | 2.9035 | 2.7131 |
| max group extra-cap loss | 0.4010 | 0.6523 | 0.6623 |
| in-domain mean group loss | 0.0004 | 0.0264 | 0.0219 |
| in-domain max group loss | 0.0043 | 0.1413 | 0.1602 |

Rare-validation scale still passed independently: selected shrinkage `1.0`,
validation overall coverage `0.9001`, and in-domain rare-scale guard coverage
`0.9506`.

In-domain SBC overall still passed, but design-stratum held-out coverage showed
pressure in intermediate/high design strata: intermediate `0.9133` and high
`0.9138`. This is consistent with the elevated extra-inflation penalties.

Conclusion: fixed amplitude caps and simple shape gates are not sufficient.
They reduce the unconstrained head slightly but mostly trade away OOD gains
without solving the in-domain extra-inflation problem.

Do not submit this candidate to LUMI.

The next roadmap step should move away from a globally applied learned
effect-shift head. A better next implementation is a two-stage post-fit
selection or shrinkage step that accepts head offsets only when OOD
effect-quantile coverage improves enough and in-domain extra-inflation/gate
penalties remain below explicit thresholds.

## Post-Fit Effect-Head Selection

Implemented a post-fit selector for the effect-shift head. The OOD objective
still fits the head, but after fitting the calibrator evaluates a shrinkage grid
for the head amplitudes and accepts a nonzero head only if explicit gates pass.

Selection thresholds:

| Threshold | Value |
| --- | ---: |
| minimum mean OOD coverage gain | 0.0125 |
| minimum worst-domain OOD coverage gain | 0.0050 |
| maximum in-domain extra inflation over 1.05 loss | 1.35 |
| maximum in-domain group extra-cap loss | 0.48 |
| maximum in-domain mean group loss | 0.0125 |
| maximum in-domain max group loss | 0.075 |

The selector records its decision under
`ood_objective.final_multiplier_diagnostics.effect_shift_head_selection`,
including the baseline, candidate shrinkage grid, selected shrinkage, thresholds,
OOD coverage gains, and in-domain gate metrics.

Focused validation:

```text
python -m py_compile pyhmsc/neural/conditional_calibration.py \
  examples/run_neural_hmsc_benchmark.py \
  examples/run_neural_hmsc_conditional_calibration.py

pytest tests/test_neural_hmsc_conditional_calibration.py -q
17 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
40 passed

git diff --check
passed
```

Benchmark metadata smoke run:

```text
/private/tmp/neural_hmsc_head_selection_smoke
```

The smoke manifest confirmed the selector metadata and candidate grid. Because
the smoke run is intentionally tiny, it selected shrinkage `0.0`; that only
verifies the selection machinery and metadata shape, not calibration quality.

The next roadmap step should run the production-like local sanity workflow with
post-fit head selection enabled and compare it against the final-aware,
unconstrained-head, and constrained-head runs.

## Post-Fit Head Selection Local Sanity

The production-like local sanity workflow was rerun with post-fit head
selection enabled:

```text
/private/tmp/neural_hmsc_v8_head_selection_local_sanity
```

The selector rejected nonzero head amplitudes and selected shrinkage `0.0`.
The production-like selection decision was:

| Metric | Selected value |
| --- | ---: |
| selected shrinkage | 0.0 |
| mean OOD coverage gain | 0.0 |
| worst-domain OOD coverage gain | 0.0 |
| effect-size diagnostic coverage | 0.8294 |
| combined diagnostic coverage | 0.8056 |
| in-domain extra inflation over 1.05 loss | 1.8772 |
| max group extra-cap loss | 0.5724 |

Held-out SBC comparison:

| Domain | Final-aware | Unconstrained head | Constrained head | Post-fit selection |
| --- | ---: | ---: | ---: | ---: |
| in distribution | 0.9301 | 0.9311 | 0.9306 | 0.9294 |
| covariate shift | 0.9143 | 0.9199 | 0.9197 | 0.9157 |
| effect-size shift | 0.8256 | 0.8656 | 0.8621 | 0.8481 |
| combined shift | 0.8036 | 0.8333 | 0.8293 | 0.8168 |

The selector successfully prevented the globally applied head from being used,
but the resulting candidate is still not qualified. It recovers some OOD
coverage relative to the original final-aware run, but loses most of the
effect-head gains and remains below the `0.90` OOD floor. In-domain overall SBC
still passes, but the selector baseline itself shows nontrivial gate pressure:
`extra_inflation_over_1_05_loss` `1.8772`, max group extra-cap loss `0.5724`,
and max group loss `0.1267`.

Rare-validation still passed independently with selected shrinkage `1.0`,
validation overall coverage `0.9001`, and rare-scale in-domain guard coverage
`0.9506`.

Do not submit this candidate to LUMI.

The next roadmap step should stop tuning the effect-shift head family and
revisit the base OOD objective decomposition. The evidence now indicates that
the learned core OOD inflation, rare-validation scale, and effect-head
correction are coupled too strongly; the next implementation should separate
domain-specific OOD calibration heads or fit independent pure-effect and
combined-shift calibration heads with their own held-out validation gates.

## Independent Effect-Head Selection

Implemented independent post-fit selection for the pure-effect and
combined-shift effect-head amplitudes. This replaces the previous shared
shrinkage decision over both branches.

Selection now proceeds in two branch-specific stages:

| Branch | Validation target | Minimum coverage gain |
| --- | --- | ---: |
| pure effect | `effect_size_shift` | 0.0125 |
| combined shift | `combined_shift` | 0.0125 |

Both branch decisions also require the same in-domain gate deltas to remain
below explicit thresholds:

| Threshold | Value |
| --- | ---: |
| maximum extra inflation over 1.05 loss increase | 0.25 |
| maximum group extra-cap loss increase | 0.08 |
| maximum mean group loss increase | 0.01 |
| maximum max group loss increase | 0.04 |

The selector records its decision under
`ood_objective.final_multiplier_diagnostics.effect_shift_head_selection` with
kind `post_fit_independent_effect_shift_head_selection`. The selected record
contains `pure_shrinkage`, `combined_shrinkage`, `pure_effect_accepted`,
`combined_shift_accepted`, domain coverage gains, and in-domain gate deltas.

Focused validation:

```text
python -m py_compile pyhmsc/neural/conditional_calibration.py \
  examples/run_neural_hmsc_benchmark.py \
  examples/run_neural_hmsc_conditional_calibration.py

pytest tests/test_neural_hmsc_conditional_calibration.py -q
17 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
40 passed
```

Benchmark metadata smoke run:

```text
/private/tmp/neural_hmsc_independent_heads_smoke
```

The smoke manifest confirmed the independent selector metadata. In this tiny
run the selected pure-effect shrinkage was nonzero, while combined-shift
shrinkage remained zero because the combined branch did not improve its target
domain enough. The smoke run verifies metadata and control flow only; it is not
a calibration-quality result.

The next roadmap step is to run the production-like local sanity workflow with
independent pure-effect/combined-shift head selection enabled and compare
selected branch shrinkage, branch acceptance, held-out OOD coverage,
effect-quantile diagnostics, and in-domain/rare-validation gates against the
final-aware, unconstrained-head, constrained-head, and shared post-fit-selection
runs.

## Independent Head Selection Local Sanity

The production-like local sanity workflow was rerun with independent
pure-effect and combined-shift post-fit selection:

```text
/private/tmp/neural_hmsc_v8_independent_head_selection_local_sanity
```

The selector accepted the pure-effect branch and rejected the combined-shift
branch:

| Field | Value |
| --- | ---: |
| pure-effect accepted | true |
| pure-effect shrinkage | 0.5 |
| combined-shift accepted | false |
| combined-shift shrinkage | 0.0 |
| effect-size validation coverage gain | 0.0167 |
| combined-shift validation coverage gain | 0.0117 |
| in-domain extra-inflation loss delta | 0.1726 |
| in-domain max group extra-cap delta | 0.0149 |

Held-out SBC comparison:

| Domain | Final-aware | Unconstrained head | Constrained head | Shared selection | Independent selection |
| --- | ---: | ---: | ---: | ---: | ---: |
| in distribution | 0.9301 | 0.9311 | 0.9306 | 0.9294 | 0.9294 |
| covariate shift | 0.9143 | 0.9199 | 0.9197 | 0.9157 | 0.9161 |
| effect-size shift | 0.8256 | 0.8656 | 0.8621 | 0.8481 | 0.8507 |
| combined shift | 0.8036 | 0.8333 | 0.8293 | 0.8168 | 0.8175 |

Effect-quantile diagnostic coverage:

| Run | Effect q2 | Effect q3 | Effect q4 | Combined q2 | Combined q3 | Combined q4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unconstrained head | 0.8400 | 0.8444 | 0.9800 | 0.8156 | 0.8689 | 0.8600 |
| constrained head | 0.8433 | 0.8356 | 0.9711 | 0.8222 | 0.8756 | 0.8444 |
| shared selection | 0.8478 | 0.8044 | 0.9600 | 0.8222 | 0.8511 | 0.8244 |
| independent selection | 0.8467 | 0.8133 | 0.9622 | 0.8200 | 0.8511 | 0.8244 |

In-domain and rare-validation gates:

| Metric | Independent selection |
| --- | ---: |
| in-domain overall coverage | 0.9294 |
| in-domain rare rank-mean error | 0.0419 |
| in-domain rare coverage | 0.9614 |
| in-domain intermediate-design coverage | 0.9117 |
| in-domain high-design coverage | 0.9179 |
| in-domain rank variance | 0.0825 |
| rare-validation selected shrinkage | 1.0 |
| rare-validation overall coverage | 0.9001 |
| rare-scale in-domain guard coverage | 0.9506 |

The independent selector is mechanically valid and slightly improves held-out
OOD coverage relative to shared post-fit selection, but it is still not
qualified. Effect-size coverage remains `0.8507` and combined-shift coverage
remains `0.8175`, both well below the `0.90` OOD floor. The rejected combined
branch also confirms that the current combined-shift head cannot clear its own
branch-specific held-out gate.

Do not submit this candidate to LUMI.

The next roadmap step should stop revising post-fit effect-head selection and
redesign the combined-shift calibration path itself. A plausible next
implementation is a domain-specific combined-shift scale objective/head trained
on combined-shift validation batches with a direct held-out combined-coverage
gate, while freezing or carrying forward the accepted pure-effect branch only
if it continues to pass the in-domain gate.

## Combined-Shift Scale Head

Implemented a domain-specific combined-shift scale head. This is separate from
post-fit effect-head selection: it applies a bounded log-scale multiplier after
the learned OOD curve and rare-validation scale, using a joint activation over
support excess and effect-size signal.

Serialized metadata:

```text
ood_objective.combined_shift_scale
ood_objective.final_multiplier_diagnostics.combined_shift_scale_selection
```

The selector evaluates a log-amplitude grid and accepts a nonzero head only
when all gates pass:

| Gate | Value |
| --- | ---: |
| held-out combined-shift coverage floor | 0.90 |
| minimum held-out combined-shift coverage gain | 0.005 |
| maximum extra inflation over 1.05 loss increase | 0.40 |
| maximum group extra-cap loss increase | 0.12 |
| maximum mean group loss increase | 0.01 |
| maximum max group loss increase | 0.04 |

Focused validation:

```text
python -m py_compile pyhmsc/neural/conditional_calibration.py \
  examples/run_neural_hmsc_benchmark.py \
  examples/run_neural_hmsc_conditional_calibration.py

pytest tests/test_neural_hmsc_conditional_calibration.py -q
18 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
41 passed
```

Benchmark metadata smoke run:

```text
/private/tmp/neural_hmsc_combined_shift_scale_smoke
```

The smoke manifest confirmed the serialized combined-shift scale block,
selection diagnostics, and per-domain `combined_shift_scale_multiplier`
summaries. Because this tiny smoke already had combined-shift validation
coverage above `0.90` and no positive coverage gain, the stricter gate selected
`log_amplitude = 0.0`. That is the intended behavior for an unnecessary
post-scale head.

The next roadmap step is to run the production-like local sanity workflow with
the combined-shift scale head enabled and compare selected log amplitude,
held-out combined-shift coverage, effect-size side effects, effect-quantile
diagnostics, and in-domain/rare-validation gates against the independent
selector run before considering any LUMI comparison.

## Combined-Shift Scale Head Local Sanity

The production-like local sanity workflow was rerun with the combined-shift
scale head enabled:

```text
/private/tmp/neural_hmsc_v8_combined_shift_scale_local_sanity
```

The combined-shift scale selector rejected every nonzero candidate and selected
`log_amplitude = 0.0`. The result is therefore identical to the independent
selector run on held-out SBC:

| Domain | Independent selection | Combined-shift scale |
| --- | ---: | ---: |
| in distribution | 0.9294 | 0.9294 |
| covariate shift | 0.9161 | 0.9161 |
| effect-size shift | 0.8507 | 0.8507 |
| combined shift | 0.8175 | 0.8175 |

Selection diagnostics:

| Metric | Value |
| --- | ---: |
| selected log amplitude | 0.0 |
| selected multiplier | 1.0 |
| selected accepted | false |
| baseline diagnostic effect-size coverage | 0.8672 |
| baseline diagnostic combined-shift coverage | 0.8289 |
| selected combined-shift coverage gain | 0.0 |
| in-domain extra-inflation loss delta | 0.0 |
| in-domain max group extra-cap delta | 0.0 |

Candidate grid behavior:

| Candidate | Combined coverage | Coverage gain | Extra-inflation delta | Max group-loss delta | Gate result |
| ---: | ---: | ---: | ---: | ---: | --- |
| log amp 0.1493 | 0.8378 | 0.0089 | 0.2793 | 0.1133 | reject |
| log amp 0.2986 | 0.8461 | 0.0172 | 0.5975 | 0.2613 | reject |
| log amp 0.4479 | 0.8561 | 0.0272 | 0.9587 | 0.4398 | reject |
| log amp 1.4931 | 0.9117 | 0.0828 | 4.6902 | 2.7321 | reject |
| log amp 1.7918 | 0.9222 | 0.0933 | 6.1431 | 3.6744 | reject |

The head can raise diagnostic combined-shift coverage above the `0.90` floor,
but only by violating in-domain gate limits. The smallest candidate already
exceeds the `max_group_loss` delta threshold, and larger candidates rapidly
increase extra-inflation and group-loss penalties. In-domain and rare-validation
status remain unchanged:

| Metric | Value |
| --- | ---: |
| in-domain overall coverage | 0.9294 |
| in-domain rare rank-mean error | 0.0419 |
| in-domain rare coverage | 0.9614 |
| in-domain intermediate-design coverage | 0.9117 |
| in-domain high-design coverage | 0.9179 |
| rare-validation selected shrinkage | 1.0 |
| rare-validation overall coverage | 0.9001 |
| rare-scale in-domain guard coverage | 0.9506 |

Do not submit this candidate to LUMI.

The next roadmap step should replace the globally applied combined-shift scale
shape with a more selective combined-shift correction. The evidence points to a
need for context separation inside the combined regime, such as
effect-bin-specific combined-shift multipliers, coefficient/design-stratum
caps, or a low-design/low-community combined-shift gate that can improve the
combined OOD tail without inflating in-domain rank/group-loss strata.

## Selective Combined-Shift Scale Shape

Implemented a more selective combined-shift scale activation. The serialized
`combined_shift_scale` head still uses a single selected log amplitude, but the
amplitude is now multiplied by four gates before application:

| Gate | Role |
| --- | --- |
| support-excess gate | activates outside calibration support |
| effect-size gate | focuses on shifted coefficient magnitude |
| low-design-information gate | suppresses high-information in-domain strata |
| low-community-occupancy gate | focuses on low-community combined-shift contexts |

The new metadata fields are:

```text
combined_shift_scale.activation.low_design_center
combined_shift_scale.activation.low_design_width
combined_shift_scale.activation.low_community_center
combined_shift_scale.activation.low_community_width
```

Focused validation:

```text
python -m py_compile pyhmsc/neural/conditional_calibration.py \
  examples/run_neural_hmsc_benchmark.py \
  examples/run_neural_hmsc_conditional_calibration.py

pytest tests/test_neural_hmsc_conditional_calibration.py -q
18 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
41 passed
```

Benchmark metadata smoke run:

```text
/private/tmp/neural_hmsc_selective_combined_shift_scale_smoke
```

The smoke manifest confirmed the selective activation metadata in both
`ood_objective.combined_shift_scale.activation` and
`ood_objective.final_multiplier_diagnostics.combined_shift_scale_selection`.
The tiny smoke selected zero amplitude because combined-shift validation
coverage already exceeded `0.90` and no positive coverage gain was available.

The next roadmap step is to run the production-like local sanity workflow with
the selective combined-shift scale enabled and compare selected log amplitude,
combined-shift coverage, in-domain group-loss/extra-inflation deltas,
effect-size side effects, and rare-validation gates against the previous
globally shaped combined-shift scale run.

## Selective Combined-Shift Scale Local Sanity

The production-like local sanity workflow was rerun with the selective
combined-shift scale enabled:

```text
/private/tmp/neural_hmsc_v8_selective_combined_shift_scale_local_sanity
```

The selector again selected `log_amplitude = 0.0`, so held-out SBC remained
identical to the globally shaped combined-shift scale and independent selector
runs:

| Domain | Global combined scale | Selective combined scale |
| --- | ---: | ---: |
| in distribution | 0.9294 | 0.9294 |
| covariate shift | 0.9161 | 0.9161 |
| effect-size shift | 0.8507 | 0.8507 |
| combined shift | 0.8175 | 0.8175 |

The selective activation did reduce in-domain gate pressure for nonzero
candidates, but it also reduced combined-shift coverage gain too much:

| Candidate | Global combined coverage | Global max group-loss delta | Selective combined coverage | Selective max group-loss delta |
| ---: | ---: | ---: | ---: | ---: |
| log amp 0.1493 | 0.8378 | 0.1133 | 0.8300 | 0.0333 |
| log amp 0.2986 | 0.8461 | 0.2613 | 0.8311 | 0.0707 |
| log amp 0.4479 | 0.8561 | 0.4398 | 0.8339 | 0.1120 |
| log amp 1.4931 | 0.9117 | 2.7321 | 0.8489 | 0.4955 |
| log amp 1.7918 | 0.9222 | 3.6744 | 0.8561 | 0.6311 |

Selection diagnostics:

| Metric | Value |
| --- | ---: |
| selected log amplitude | 0.0 |
| selected accepted | false |
| baseline diagnostic combined-shift coverage | 0.8289 |
| best selective candidate combined-shift coverage | 0.8561 |
| best selective candidate combined-shift gain | 0.0272 |
| best selective candidate max group-loss delta | 0.6311 |
| best selective candidate extra-inflation delta | 1.2254 |

In-domain and rare-validation gates remain unchanged:

| Metric | Value |
| --- | ---: |
| in-domain overall coverage | 0.9294 |
| in-domain rare rank-mean error | 0.0419 |
| in-domain rare coverage | 0.9614 |
| in-domain intermediate-design coverage | 0.9117 |
| in-domain high-design coverage | 0.9179 |
| rare-validation selected shrinkage | 1.0 |
| rare-validation overall coverage | 0.9001 |
| rare-scale in-domain guard coverage | 0.9506 |

Do not submit this candidate to LUMI.

The next roadmap step should strengthen the selective combined-shift correction
without returning to global in-domain inflation. The most plausible next
implementation is an effect-bin-specific or quantile-targeted combined-shift
head, where high-effect combined-shift coefficients can receive stronger
inflation while low/mid-effect and high-design in-domain strata remain capped.

## Effect-Bin Combined-Shift Scale Head

Implemented an effect-bin-specific combined-shift scale selector. The
`combined_shift_scale` metadata now records:

```text
combined_shift_scale.effect_bin_edges
combined_shift_scale.effect_bin_log_amplitudes
combined_shift_scale.effect_bin_multipliers
```

The selector keeps the selective support/effect/low-design/low-community
activation, but it now evaluates multiple correction shapes: scalar selective,
high-effect only, mid/high-effect, ranked-effect, and all-effect-bin patterns.
Each candidate is still accepted only if it clears the combined-shift coverage
floor and the existing in-domain gate deltas.

Focused validation:

```text
pytest tests/test_neural_hmsc_conditional_calibration.py -q
18 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
41 passed
```

Benchmark metadata smoke run:

```text
/private/tmp/neural_hmsc_effect_bin_combined_shift_scale_smoke
```

The smoke manifest confirmed 60 effect-bin candidates and serialized the
selected per-bin fields:

| Field | Value |
| --- | --- |
| selection kind | `effect_bin_specific_combined_shift_scale_selection` |
| candidate count | `60` |
| selected accepted | `false` |
| selected log amplitude | `0.0` |
| selected effect-bin log amplitudes | `[0.0, 0.0, 0.0]` |
| selected effect-bin multipliers | `[1.0, 1.0, 1.0]` |
| smoke effect-size-shift coverage | `0.75` |
| smoke combined-shift coverage | `0.9722` |

Because the tiny smoke already had high combined-shift coverage and poor
effect-size-shift coverage, it did not select a nonzero combined-shift branch.
This is only a metadata and plumbing check, not a qualification result.

The next roadmap step is to run the production-like local sanity workflow with
the effect-bin-specific combined-shift scale enabled and compare selected
pattern, per-bin amplitudes, held-out OOD coverage, effect-quantile coverage,
final multiplier diagnostics, in-domain gate deltas, and rare-validation gates
against the selective scalar combined-shift scale run.

## Effect-Bin Combined-Shift Scale Local Sanity

The first effect-bin production-like run was started without rare calibration
and rare validation batches, so it was not comparable to the selective scalar
run. The corrected production-like workflow was rerun with the same rare
calibration and validation settings as the selective scalar run:

```text
/private/tmp/neural_hmsc_v8_effect_bin_combined_shift_scale_local_sanity_rare32
```

Both runs use `rare_calibration_datasets = 32`,
`rare_validation_datasets = 32`, `sbc_datasets = 32`, and `sbc_draws = 128`.

Held-out 95% coefficient coverage is unchanged because the effect-bin selector
also selected zero amplitude:

| Domain | Selective scalar | Effect-bin selector |
| --- | ---: | ---: |
| in distribution | 0.9294 | 0.9294 |
| covariate shift | 0.9161 | 0.9161 |
| effect-size shift | 0.8507 | 0.8507 |
| combined shift | 0.8175 | 0.8175 |

Selection summary:

| Metric | Selective scalar | Effect-bin selector |
| --- | ---: | ---: |
| selector kind | `domain_specific_combined_shift_scale_selection` | `effect_bin_specific_combined_shift_scale_selection` |
| candidates evaluated | 12 | 60 |
| selected accepted | false | false |
| selected log amplitude | 0.0 | 0.0 |
| selected effect-bin log amplitudes | n/a | `[0.0, 0.0, 0.0]` |
| selector effect-size coverage | 0.8672 | 0.8672 |
| selector combined-shift coverage | 0.8289 | 0.8289 |
| combined-shift coverage shortfall | 0.0711 | 0.0711 |

Best candidate by pattern in the effect-bin selector:

| Pattern | Combined coverage | Effect-size coverage | Combined gain | Max group-loss delta | Extra-inflation delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scalar_selective` | 0.8561 | 0.8939 | 0.0272 | 0.6311 | 1.2254 |
| `all_effect_bins` | 0.8561 | 0.8939 | 0.0272 | 0.6311 | 1.2254 |
| `ranked_effect` | 0.8450 | 0.8828 | 0.0161 | 0.2701 | 1.0880 |
| `mid_high_effect` | 0.8428 | 0.8811 | 0.0139 | 0.1764 | 1.0617 |
| `high_effect` | 0.8389 | 0.8772 | 0.0100 | 0.1465 | 1.0013 |

The effect-bin shapes did reduce in-domain group-loss pressure as the
activation became more selective, but the coverage gain fell at the same time.
The only patterns that reached the best combined-shift coverage were the scalar
or all-bin patterns, and both reproduced the same in-domain gate failure as the
previous scalar correction.

Effect-quantile diagnostics also remain unchanged at the selected zero
amplitude:

| Domain | Quantile | Coverage | Final multiplier mean |
| --- | --- | ---: | ---: |
| effect-size shift | q2 | 0.8467 | 0.8012 |
| effect-size shift | q3 | 0.8133 | 0.8895 |
| effect-size shift | q4 | 0.9622 | 3.3631 |
| combined shift | q2 | 0.8200 | 6.1127 |
| combined shift | q3 | 0.8511 | 6.5542 |
| combined shift | q4 | 0.8244 | 4.5583 |

In-domain and rare-validation gates also match the selective scalar run:

| Metric | Effect-bin selector |
| --- | ---: |
| in-domain rare 95% coverage | 0.9614 |
| in-domain rare rank-mean error | 0.0419 |
| in-domain intermediate-design 95% coverage | 0.9117 |
| in-domain high-design 95% coverage | 0.9179 |
| rare-validation selected shrinkage | 1.0 |
| rare-validation overall coverage | 0.9001 |
| rare-validation rare coverage | 0.9363 |
| rare-validation rare rank error | 0.0236 |
| rare-scale in-domain guard coverage | 0.9506 |

Do not submit this candidate to LUMI.

The next roadmap step should stop tuning effect-bin amplitudes and implement a
more domain-discriminative combined-shift activation. The evidence here is that
effect-bin selectivity trades away the OOD coverage gain before the in-domain
gate clears. A better next implementation is a combined-shift domain/context
gate, or a low-community-by-support-excess classifier-style gate, trained and
validated on combined-shift batches with explicit in-domain overlap penalties.

## Context-Gated Combined-Shift Activation

Implemented a more domain-discriminative combined-shift activation. The
`combined_shift_scale` metadata now includes a classifier-style context gate:

```text
combined_shift_scale.context_gate.kind
combined_shift_scale.context_gate.strength
combined_shift_scale.context_gate.intercept
combined_shift_scale.context_gate.support_weight
combined_shift_scale.context_gate.effect_weight
combined_shift_scale.context_gate.low_design_weight
combined_shift_scale.context_gate.low_community_weight
combined_shift_scale.context_gate.support_low_community_interaction_weight
```

The gate scores support excess, effect signal, low design information, low
community occupancy, and a support-by-low-community interaction. Missing
context-gate metadata defaults to strength `0.0`, so older combined-shift
artifacts keep their original behavior.

Selection now evaluates three context variants across the existing scalar and
effect-bin correction shapes:

| Context pattern | Strength | Intercept |
| --- | ---: | ---: |
| `legacy_product` | 0.0 | 0.0 |
| `context_moderate` | 1.0 | 1.0 |
| `context_strict` | 1.0 | 2.0 |

Every candidate records `in_domain_context_gate` diagnostics and must satisfy
explicit in-domain overlap limits:

```text
max_in_domain_context_gate_mean = 0.74
max_in_domain_context_gate_active_fraction = 0.70
```

Focused validation:

```text
pytest tests/test_neural_hmsc_conditional_calibration.py -q
18 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
41 passed
```

Benchmark metadata smoke run:

```text
/private/tmp/neural_hmsc_context_gated_combined_shift_smoke
```

The smoke manifest confirmed:

| Field | Value |
| --- | --- |
| selector kind | `context_gated_combined_shift_scale_selection` |
| candidates evaluated | 180 |
| context patterns | `legacy_product`, `context_moderate`, `context_strict` |
| selected accepted | false |
| selected context strength | 0.0 |
| selected context intercept | 0.0 |
| smoke effect-size-shift coverage | 0.75 |
| smoke combined-shift coverage | 0.9722 |

The tiny smoke is only a metadata and plumbing check. Because it already had
high combined-shift coverage and poor effect-size-shift coverage, it selected
zero amplitude.

The next roadmap step is to run the production-like local sanity workflow with
the context-gated combined-shift selector enabled, using the same rare
calibration and validation settings as the previous selective scalar and
effect-bin local sanity runs.

## Context-Gated Combined-Shift Local Sanity

The production-like local sanity workflow was rerun with the context-gated
combined-shift selector enabled:

```text
/private/tmp/neural_hmsc_v8_context_gated_combined_shift_local_sanity_rare32
```

The run uses the same comparison settings as the selective scalar and effect-bin
local sanity runs:

```text
rare_calibration_datasets = 32
rare_validation_datasets = 32
sbc_datasets = 32
sbc_draws = 128
```

Held-out 95% coefficient coverage is unchanged because the context-gated
selector also selected zero amplitude:

| Domain | Selective scalar | Effect-bin selector | Context-gated selector |
| --- | ---: | ---: | ---: |
| in distribution | 0.9294 | 0.9294 | 0.9294 |
| covariate shift | 0.9161 | 0.9161 | 0.9161 |
| effect-size shift | 0.8507 | 0.8507 | 0.8507 |
| combined shift | 0.8175 | 0.8175 | 0.8175 |

Selection summary:

| Metric | Context-gated selector |
| --- | ---: |
| selector kind | `context_gated_combined_shift_scale_selection` |
| candidates evaluated | 180 |
| selected accepted | false |
| selected log amplitude | 0.0 |
| selected context strength | 0.0 |
| selected context intercept | 0.0 |
| selector effect-size coverage | 0.8672 |
| selector combined-shift coverage | 0.8289 |
| combined-shift coverage shortfall | 0.0711 |

Best candidate by context pattern:

| Context pattern | Best shape | Combined coverage | Effect-size coverage | Combined gain | Gate mean | Active >0.8 | Max group-loss delta | Extra-inflation delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `legacy_product` | `scalar_selective` | 0.8561 | 0.8939 | 0.0272 | 1.0000 | 1.0000 | 0.6311 | 1.2254 |
| `context_moderate` | `scalar_selective` | 0.8361 | 0.8711 | 0.0072 | 0.1896 | 0.0061 | 0.1483 | 0.5525 |
| `context_strict` | `scalar_selective` | 0.8328 | 0.8683 | 0.0039 | 0.0900 | 0.0033 | 0.0767 | 0.3601 |

The context gate reduced in-domain overlap sharply, but the OOD coverage gain
fell at the same time. The moderate context gate still failed the max group-loss
and extra-inflation deltas, while the strict context gate reduced those deltas
further but missed the minimum combined-shift gain and remained far below the
`0.90` combined-shift floor.

Effect-quantile diagnostics are unchanged at the selected zero amplitude:

| Domain | Quantile | Coverage | Final multiplier mean |
| --- | --- | ---: | ---: |
| effect-size shift | q2 | 0.8467 | 0.8012 |
| effect-size shift | q3 | 0.8133 | 0.8895 |
| effect-size shift | q4 | 0.9622 | 3.3631 |
| combined shift | q2 | 0.8200 | 6.1127 |
| combined shift | q3 | 0.8511 | 6.5542 |
| combined shift | q4 | 0.8244 | 4.5583 |

In-domain and rare-validation gates also match the previous runs:

| Metric | Context-gated selector |
| --- | ---: |
| in-domain rare 95% coverage | 0.9614 |
| in-domain rare rank-mean error | 0.0419 |
| in-domain intermediate-design 95% coverage | 0.9117 |
| in-domain high-design 95% coverage | 0.9179 |
| rare-validation selected shrinkage | 1.0 |
| rare-validation overall coverage | 0.9001 |
| rare-validation rare coverage | 0.9363 |
| rare-validation rare rank error | 0.0236 |
| rare-scale in-domain guard coverage | 0.9506 |

Do not submit this candidate to LUMI.

The next roadmap step should stop adding post-hoc combined-shift scale gates.
The local evidence now shows the same tradeoff across scalar, effect-bin, and
context-gated variants: candidates that raise combined-shift coverage enough
overlap in-domain strata too strongly, while selective gates that protect
in-domain strata remove the OOD gain. The next implementation should move the
combined-shift signal earlier into the learned OOD objective, for example by
adding a direct combined-shift coverage term with a learned context classifier
and explicit in-domain overlap regularization during OOD-objective fitting,
rather than selecting another post-scale multiplier.

## Combined-Shift-Aware OOD Objective

Implemented the next combined-shift revision inside the
final-multiplier-aware learned OOD objective instead of adding another
post-scale selector.

The serialized metadata now records:

```text
ood_objective.combined_shift_training_objective.kind
ood_objective.combined_shift_training_objective.coverage_weight
ood_objective.combined_shift_training_objective.effect_quantile_weight
ood_objective.combined_shift_training_objective.context_weight
ood_objective.combined_shift_training_objective.in_domain_overlap_weight
ood_objective.combined_shift_training_objective.in_domain_overlap_mean_cap
ood_objective.combined_shift_training_objective.in_domain_overlap_active_fraction_cap
```

The implementation keeps the OOD parameter vector stable. It reuses the learned
combined branch as a support/effect/low-design/low-community context classifier
and adds three final-aware OOD losses for `combined_shift` batches:

| Term | Role |
| --- | --- |
| direct combined coverage | pushes smooth combined-shift coverage toward the `0.90` floor |
| combined effect-quantile coverage | targets under-covered effect-signal quantiles inside combined shift |
| context-weighted coverage | focuses pressure where the learned combined context is active |

The in-domain gate now also penalizes:

| Term | Role |
| --- | --- |
| combined-context overlap mean | discourages broad combined-context activation on in-domain coefficients |
| combined-context active fraction | discourages many strongly active in-domain coefficients |
| context-weighted extra inflation | caps extra learned inflation where the combined context overlaps in-domain strata |

Final multiplier diagnostics now report `learned_combined_shift_context` by OOD
domain.

Focused validation:

```text
pytest tests/test_neural_hmsc_conditional_calibration.py -q
18 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
41 passed
```

Benchmark metadata smoke run:

```text
/private/tmp/neural_hmsc_combined_objective_smoke
```

The smoke manifest confirmed:

| Field | Value |
| --- | --- |
| training objective kind | `final_multiplier_aware_combined_shift_coverage` |
| coverage weight | 4.0 |
| effect-quantile weight | 2.5 |
| context weight | 2.0 |
| in-domain overlap weight | 0.2 |
| smoke effect-size-shift learned context mean | 0.0266 |
| smoke combined-shift learned context mean | 0.0647 |

The tiny smoke is only a metadata and plumbing check. The next roadmap step is
to rerun the production-like local sanity workflow with this combined-shift
aware OOD objective and compare it against the context-gated post-scale result.

## Combined-Shift-Aware OOD Objective Local Sanity

The production-like local sanity workflow was rerun with the combined-shift
aware OOD objective enabled:

```text
/private/tmp/neural_hmsc_v8_combined_objective_local_sanity_rare32
```

The run uses the same comparison settings as the previous local sanity runs:

```text
rare_calibration_datasets = 32
rare_validation_datasets = 32
sbc_datasets = 32
sbc_draws = 128
```

Held-out 95% coefficient coverage worsened for the OOD regimes:

| Domain | Context-gated post-scale | Combined-aware OOD objective |
| --- | ---: | ---: |
| in distribution | 0.9294 | 0.9308 |
| covariate shift | 0.9161 | 0.9157 |
| effect-size shift | 0.8507 | 0.8303 |
| combined shift | 0.8175 | 0.8060 |

The new objective did reduce in-domain gate penalties:

| Metric | Context-gated post-scale | Combined-aware OOD objective |
| --- | ---: | ---: |
| OOD objective loss | 13.0370 | 22.5912 |
| OOD rank loss | 5.9762 | 6.4780 |
| in-domain gate loss | 0.8911 | 1.7087 |
| diagnostic mean group loss | 0.0106 | 0.0005 |
| diagnostic max group loss | 0.1038 | 0.0052 |
| diagnostic extra-inflation loss | 2.0497 | 1.1963 |
| diagnostic max group extra-inflation loss | 0.5872 | 0.3991 |

However, the OOD gains moved in the wrong direction:

| Domain | Context-gated selector coverage | Combined-aware selector coverage | Coverage shortfall |
| --- | ---: | ---: | ---: |
| effect-size shift | 0.8672 | 0.8511 | 0.0489 |
| combined shift | 0.8289 | 0.8178 | 0.0822 |

Effect-quantile diagnostics also degraded:

| Domain | Quantile | Context-gated coverage | Combined-aware coverage | Combined-aware final multiplier mean |
| --- | --- | ---: | ---: | ---: |
| effect-size shift | q2 | 0.8467 | 0.8556 | 0.8077 |
| effect-size shift | q3 | 0.8133 | 0.7911 | 0.8355 |
| effect-size shift | q4 | 0.9622 | 0.9022 | 2.4447 |
| combined shift | q2 | 0.8200 | 0.8233 | 5.8962 |
| combined shift | q3 | 0.8511 | 0.8467 | 6.4795 |
| combined shift | q4 | 0.8244 | 0.7778 | 4.1532 |

The learned combined-shift context activated more on combined shift than on
pure effect-size shift, but not enough to recover coverage:

| Domain | Learned context mean | q95 | max |
| --- | ---: | ---: | ---: |
| effect-size shift | 0.0751 | 0.1414 | 0.3816 |
| combined shift | 0.1019 | 0.3580 | 0.5524 |

In-domain and rare-validation gates remained acceptable:

| Metric | Combined-aware OOD objective |
| --- | ---: |
| in-domain rare 95% coverage | 0.9614 |
| in-domain rare rank-mean error | 0.0397 |
| in-domain intermediate-design 95% coverage | 0.9146 |
| in-domain high-design 95% coverage | 0.9233 |
| rare-validation selected shrinkage | 1.0 |
| rare-validation overall coverage | 0.9001 |
| rare-validation rare coverage | 0.9281 |
| rare-validation rare rank error | 0.0194 |
| rare-scale in-domain guard coverage | 0.9508 |

Do not submit this candidate to LUMI.

The next roadmap step should rebalance or stage the combined-shift-aware OOD
objective instead of adding another gate. The local result suggests the
in-domain overlap penalty is too strong relative to the OOD coverage terms, so
the next implementation should use a two-stage or constrained schedule: first
fit the combined branch to recover combined-shift/effect-quantile coverage,
then apply an in-domain overlap constraint or shrinkage step that preserves the
coverage gain.

## Staged Combined-Shift-Aware OOD Objective

Implemented a staged version of the combined-shift-aware OOD objective. The
final-aware OOD refit now uses a `coverage_warmup_then_overlap_ramp` schedule:

| Stage | Behavior |
| --- | --- |
| coverage warmup | boosts direct combined-shift, effect-quantile, and context-weighted coverage terms |
| coverage warmup | down-weights the in-domain gate to allow the combined branch to recover OOD coverage |
| overlap ramp | ramps the in-domain gate and combined-context overlap penalty back to full strength |

The post-fit pure-effect/combined-shift shrinkage selector still runs after the
staged refit, so any OOD gain must survive the existing in-domain gate checks
before application.

The serialized metadata now includes:

```text
ood_objective.combined_shift_training_objective.schedule.kind
ood_objective.combined_shift_training_objective.schedule.warmup_fraction
ood_objective.combined_shift_training_objective.schedule.warmup_coverage_boost
ood_objective.combined_shift_training_objective.schedule.warmup_gate_fraction
```

Focused validation:

```text
pytest tests/test_neural_hmsc_conditional_calibration.py -q
18 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
41 passed
```

Benchmark metadata smoke run:

```text
/private/tmp/neural_hmsc_staged_combined_objective_smoke
```

The smoke manifest confirmed:

| Field | Value |
| --- | --- |
| schedule kind | `coverage_warmup_then_overlap_ramp` |
| warmup fraction | 0.55 |
| warmup coverage boost | 1.75 |
| warmup gate fraction | 0.35 |
| smoke effect-size-shift learned context mean | 0.0266 |
| smoke combined-shift learned context mean | 0.0646 |

The tiny smoke is only a metadata and plumbing check. The next roadmap step is
to rerun the production-like local sanity workflow with the staged
combined-shift-aware OOD objective and compare it against the unstaged
combined-aware result.

## Staged Combined-Shift-Aware OOD Objective Local Sanity

The production-like local sanity workflow was rerun with the staged
combined-shift-aware OOD objective enabled:

```text
/private/tmp/neural_hmsc_v8_staged_combined_objective_local_sanity_rare32
```

The run uses the same comparison settings as the prior local sanity runs:

```text
rare_calibration_datasets = 32
rare_validation_datasets = 32
sbc_datasets = 32
sbc_draws = 128
```

Held-out 95% coefficient coverage did not recover:

| Domain | Context-gated post-scale | Unstaged combined objective | Staged combined objective |
| --- | ---: | ---: | ---: |
| in distribution | 0.9294 | 0.9308 | 0.9308 |
| covariate shift | 0.9161 | 0.9157 | 0.9157 |
| effect-size shift | 0.8507 | 0.8303 | 0.8290 |
| combined shift | 0.8175 | 0.8060 | 0.8056 |

The staged schedule slightly reduced in-domain inflation penalties relative to
the unstaged combined objective, but it did not improve OOD coverage:

| Metric | Unstaged combined objective | Staged combined objective |
| --- | ---: | ---: |
| OOD objective loss | 22.5912 | 23.5975 |
| OOD rank loss | 6.4780 | 6.4873 |
| in-domain gate loss | 1.7087 | 1.6601 |
| diagnostic mean group loss | 0.0005 | 0.0003 |
| diagnostic max group loss | 0.0052 | 0.0031 |
| diagnostic extra-inflation loss | 1.1963 | 1.1597 |
| diagnostic max group extra-inflation loss | 0.3991 | 0.3920 |

Selector diagnostics stayed below the context-gated post-scale baseline:

| Domain | Context-gated selector | Unstaged combined objective | Staged combined objective |
| --- | ---: | ---: | ---: |
| effect-size shift | 0.8672 | 0.8511 | 0.8511 |
| combined shift | 0.8289 | 0.8178 | 0.8178 |

Effect-quantile diagnostics also stayed degraded:

| Domain | Quantile | Context-gated coverage | Staged coverage | Staged final multiplier mean |
| --- | --- | ---: | ---: | ---: |
| effect-size shift | q2 | 0.8467 | 0.8556 | 0.8070 |
| effect-size shift | q3 | 0.8133 | 0.7911 | 0.8331 |
| effect-size shift | q4 | 0.9622 | 0.9022 | 2.4105 |
| combined shift | q2 | 0.8200 | 0.8233 | 5.8906 |
| combined shift | q3 | 0.8511 | 0.8467 | 6.4737 |
| combined shift | q4 | 0.8244 | 0.7778 | 4.1332 |

The learned combined context remained selective but too weak to recover
combined-shift coverage:

| Domain | Unstaged context mean | Staged context mean | Staged q95 | Staged max |
| --- | ---: | ---: | ---: | ---: |
| effect-size shift | 0.0751 | 0.0746 | 0.1414 | 0.3815 |
| combined shift | 0.1019 | 0.1014 | 0.3579 | 0.5524 |

In-domain and rare-validation gates remained acceptable:

| Metric | Staged combined objective |
| --- | ---: |
| in-domain rare 95% coverage | 0.9614 |
| in-domain rare rank-mean error | 0.0395 |
| in-domain intermediate-design 95% coverage | 0.9146 |
| in-domain high-design 95% coverage | 0.9233 |
| rare-validation selected shrinkage | 1.0 |
| rare-validation overall coverage | 0.9001 |
| rare-validation rare coverage | 0.9281 |
| rare-validation rare rank error | 0.0194 |
| rare-scale in-domain guard coverage | 0.9508 |

Do not submit this candidate to LUMI.

The next roadmap step should stop this combined-shift objective family. The
local evidence now shows that post-scale gates, direct combined-shift objective
terms, and staged objective scheduling all fail to recover combined-shift OOD
coverage under the current feature/context representation. The next
implementation should add a new representation or data split for combined
shift, such as a domain-adversarial or mixture-of-experts OOD head with
separate pure-effect and combined-shift experts and a held-out expert-selection
gate.

## Domain-Expert OOD Selector Implementation

Implemented the first replacement for the failed combined-shift objective
family: a held-out domain-expert OOD selector. The final-multiplier-aware OOD
path now trains two candidate experts after the shared final-aware refit:

| Expert | Training split | Selection target |
| --- | --- | --- |
| `pure_effect` | `effect_size_shift` OOD calibration batches | effect-size-shift coverage gain |
| `combined_shift` | `combined_shift` OOD calibration batches | combined-shift coverage gain |

Selection uses explicit in-domain gate-delta thresholds. The selected candidate
must improve its target held-out OOD coverage and keep extra in-domain
inflation, grouped rank loss, and grouped coverage penalties below the same
gate-delta family used by the post-fit effect-head shrinkage selector. If no
candidate passes, the selector preserves the baseline OOD parameters. This
keeps the existing serialized OOD parameter format stable while adding a
domain-specific training/evaluation split.

The serialized metadata now includes:

```text
ood_objective.final_multiplier_diagnostics.domain_expert_selection.kind
ood_objective.final_multiplier_diagnostics.domain_expert_selection.split_modes
ood_objective.final_multiplier_diagnostics.domain_expert_selection.baseline
ood_objective.final_multiplier_diagnostics.domain_expert_selection.selected
ood_objective.final_multiplier_diagnostics.domain_expert_selection.candidates
```

Focused validation:

```text
pytest tests/test_neural_hmsc_conditional_calibration.py -q
18 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
41 passed
```

Benchmark metadata smoke run:

```text
/private/tmp/neural_hmsc_domain_expert_smoke
```

The tiny smoke confirmed `kind =
heldout_domain_expert_ood_selection`, split modes
`within_batch_axis0` for both experts, and two candidate records. Both
candidates were rejected because neither improved held-out target coverage in
the tiny setup:

| Expert | Target gain | Accepted |
| --- | ---: | --- |
| pure-effect | 0.0000 | false |
| combined-shift | 0.0000 | false |

This is only a metadata and plumbing check. The next roadmap step is to run the
production-like local sanity workflow with the domain-expert selector enabled
and compare it against the staged combined-objective and context-gated
post-scale baselines before considering LUMI.

## Domain-Expert OOD Selector Local Sanity

The production-like local sanity workflow was rerun with the held-out
domain-expert OOD selector enabled:

```text
/private/tmp/neural_hmsc_v8_domain_expert_local_sanity_rare32
```

The run used the same comparison settings as the staged combined-objective run:

```text
rare_calibration_datasets = 32
rare_validation_datasets = 32
sbc_datasets = 32
sbc_draws = 128
```

The selector did not change the final applied calibration. Both expert
candidates improved their target held-out OOD domains inside the selector, but
both violated in-domain gate-delta limits, so the selected expert was
`baseline` with reason `no_candidate_passed_selection_gate`.

Overall 95% coefficient coverage therefore matched the staged
combined-objective result and remained below the OOD floor:

| Domain | Context-gated post-scale | Staged combined objective | Domain-expert selector |
| --- | ---: | ---: | ---: |
| in distribution | 0.9294 | 0.9308 | 0.9308 |
| covariate shift | 0.9161 | 0.9157 | 0.9157 |
| effect-size shift | 0.8507 | 0.8290 | 0.8290 |
| combined shift | 0.8175 | 0.8056 | 0.8056 |

Expert-selection diagnostics show that the candidates learned useful OOD
directions but were too aggressive for the in-domain constraints:

| Candidate | Split mode | Target gain | Effect-size coverage | Combined coverage | Accepted |
| --- | --- | ---: | ---: | ---: | --- |
| pure-effect | `within_batch_axis0` | 0.0489 | 0.8967 | 0.8622 | false |
| combined-shift | `within_batch_axis0` | 0.0578 | 0.9033 | 0.8878 | false |

The rejection was driven by in-domain gate deltas far beyond the selector
thresholds:

| Candidate | Mean group loss delta | Max group loss delta | Extra-inflation loss delta | Max group extra-cap delta |
| --- | ---: | ---: | ---: | ---: |
| pure-effect | 0.1184 | 0.6717 | 3.1203 | 0.3974 |
| combined-shift | 0.2112 | 1.1772 | 4.4948 | 0.6782 |

Final OOD diagnostics remained unchanged from the staged objective:

| Domain | Diagnostic coverage | Shortfall | Final multiplier mean | Final multiplier q95 | Learned combined-context mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| effect-size shift | 0.8511 | 0.0489 | 1.2144 | 3.2841 | 0.0746 |
| combined shift | 0.8178 | 0.0822 | 5.5970 | 19.2144 | 0.1014 |

Effect-quantile diagnostics still identify the same weak strata:

| Domain | Quantile | Coverage | Final multiplier mean |
| --- | --- | ---: | ---: |
| effect-size shift | q2 | 0.8556 | 0.8070 |
| effect-size shift | q3 | 0.7911 | 0.8331 |
| effect-size shift | q4 | 0.9022 | 2.4105 |
| combined shift | q2 | 0.8233 | 5.8906 |
| combined shift | q3 | 0.8467 | 6.4737 |
| combined shift | q4 | 0.7778 | 4.1332 |

Rare-validation gates remained acceptable:

| Metric | Domain-expert selector |
| --- | ---: |
| rare-validation selected shrinkage | 1.0 |
| rare-validation overall coverage | 0.9001 |
| rare-validation rare coverage | 0.9281 |
| rare-validation rare rank error | 0.0194 |
| rare-scale in-domain guard coverage | 0.9508 |

Do not submit this candidate to LUMI. The next roadmap step should keep the
domain-expert split, but add constrained or shrinkage-aware expert selection so
the useful OOD direction can be partially accepted only when in-domain gate
deltas remain within explicit limits. A practical implementation would evaluate
a shrinkage grid between the baseline and each trained expert, or add a
trust-region/overlap penalty during expert fitting, then rerun this same local
sanity workflow before any LUMI comparison.

## Shrinkage-Aware Domain-Expert Selector Local Sanity

Implemented baseline-to-expert shrinkage selection inside the domain-expert
OOD selector. Each trained expert is now evaluated on a fixed shrinkage grid:

```text
0.0, 0.125, 0.25, 0.5, 0.75, 1.0
```

The selector accepts the best shrinkage only when the target held-out OOD
coverage gain clears the domain-specific minimum and the in-domain gate deltas
remain below explicit limits. The metadata now records `shrinkage_grid`,
`selected_shrinkage`, and per-shrinkage diagnostics for each expert under
`ood_objective.final_multiplier_diagnostics.domain_expert_selection`.

Validation:

```text
pytest tests/test_neural_hmsc_conditional_calibration.py -q
18 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
41 passed
```

The production-like local sanity workflow was rerun with the shrinkage-aware
selector:

```text
/private/tmp/neural_hmsc_v8_domain_expert_shrinkage_local_sanity_rare32
```

The run used the same comparison settings as the prior rare32 sanity runs:

```text
rare_calibration_datasets = 32
rare_validation_datasets = 32
sbc_datasets = 32
sbc_draws = 128
```

No shrinkage point qualified. The final selected expert remained `baseline`
with `selected_shrinkage = 0.0`, so final 95% coverage was unchanged from the
non-shrinkage domain-expert run:

| Domain | Context-gated post-scale | Staged combined objective | Shrinkage-aware domain expert |
| --- | ---: | ---: | ---: |
| in distribution | 0.9294 | 0.9308 | 0.9308 |
| covariate shift | 0.9161 | 0.9157 | 0.9157 |
| effect-size shift | 0.8507 | 0.8290 | 0.8290 |
| combined shift | 0.8175 | 0.8056 | 0.8056 |

Pure-effect expert shrinkage diagnostics:

| Shrinkage | Target gain | Effect-size coverage | Combined coverage | Extra-inflation delta | Accepted |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.000 | 0.0000 | 0.8478 | 0.8300 | 0.0000 | false |
| 0.125 | 0.0067 | 0.8544 | 0.8322 | 0.2868 | false |
| 0.250 | 0.0144 | 0.8622 | 0.8367 | 0.6077 | false |
| 0.500 | 0.0322 | 0.8800 | 0.8422 | 1.3544 | false |
| 0.750 | 0.0467 | 0.8944 | 0.8522 | 2.2019 | false |
| 1.000 | 0.0489 | 0.8967 | 0.8622 | 3.1203 | false |

Combined-shift expert shrinkage diagnostics:

| Shrinkage | Target gain | Effect-size coverage | Combined coverage | Extra-inflation delta | Accepted |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.000 | 0.0000 | 0.8478 | 0.8300 | 0.0000 | false |
| 0.125 | 0.0044 | 0.8567 | 0.8344 | 0.3992 | false |
| 0.250 | 0.0111 | 0.8667 | 0.8411 | 0.8596 | false |
| 0.500 | 0.0233 | 0.8911 | 0.8533 | 1.9441 | false |
| 0.750 | 0.0400 | 0.8967 | 0.8700 | 3.1494 | false |
| 1.000 | 0.0578 | 0.9033 | 0.8878 | 4.4948 | false |

The pure-effect branch nearly identifies the tradeoff: at shrinkage `0.125`,
the gate is only slightly too inflated but the target gain is still below the
minimum; by shrinkage `0.25`, the target gain clears the minimum but the
in-domain inflation penalty is already too large. This means post-fit linear
shrinkage alone is too blunt.

Rare-validation gates remained acceptable:

| Metric | Shrinkage-aware domain expert |
| --- | ---: |
| rare-validation selected shrinkage | 1.0 |
| rare-validation overall coverage | 0.9001 |
| rare-validation rare coverage | 0.9281 |
| rare-validation rare rank error | 0.0194 |
| rare-scale in-domain guard coverage | 0.9508 |

Do not submit this candidate to LUMI. The next roadmap step should keep the
domain-expert data split, but move the constraint into expert fitting itself:
add a trust-region or overlap-regularized expert objective that penalizes
in-domain final-multiplier drift while training the expert, then rerun the same
production-like local sanity workflow.

## Trust-Region Domain-Expert Objective Local Sanity

Implemented an in-domain log-inflation trust-region penalty inside domain
expert fitting. Expert refits now penalize deviation from the pre-expert
baseline OOD log-inflation on in-domain coefficients:

```text
trust-region kind = in_domain_log_inflation_trust_region
weight = 3.0
log_tolerance = log(1.08)
scale = log(1.25)
```

The post-fit shrinkage grid remains active after the constrained expert fit.
Metadata records the trust-region settings under
`ood_objective.final_multiplier_diagnostics.domain_expert_selection`.

Validation:

```text
pytest tests/test_neural_hmsc_conditional_calibration.py -q
18 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
41 passed
```

The production-like local sanity workflow was rerun with the trust-region
expert objective:

```text
/private/tmp/neural_hmsc_v8_domain_expert_trust_region_local_sanity_rare32
```

The run used the same rare32 settings as the prior domain-expert checks.

The trust region reduced expert aggressiveness but still did not produce an
accepted candidate. Final selected expert remained `baseline`, so final 95%
coverage was unchanged:

| Domain | Context-gated post-scale | Staged combined objective | Trust-region domain expert |
| --- | ---: | ---: | ---: |
| in distribution | 0.9294 | 0.9308 | 0.9308 |
| covariate shift | 0.9161 | 0.9157 | 0.9157 |
| effect-size shift | 0.8507 | 0.8290 | 0.8290 |
| combined shift | 0.8175 | 0.8056 | 0.8056 |

Compared with post-fit shrinkage alone, the full pure-effect expert became much
less aggressive:

| Pure-effect full expert | Target gain | Effect-size coverage | Combined coverage | Extra-inflation delta | Max group delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| shrinkage-only fit | 0.0489 | 0.8967 | 0.8622 | 3.1203 | 0.6717 |
| trust-region fit | 0.0311 | 0.8789 | 0.8389 | 0.6216 | 0.1897 |

But the acceptance tradeoff remains unresolved. Smaller trust-region shrinkages
were safe but too weak; stronger shrinkages improved OOD coverage but violated
the extra-inflation gate:

| Candidate | Shrinkage | Target gain | OOD target coverage | Extra-inflation delta | Accepted |
| --- | ---: | ---: | ---: | ---: | --- |
| pure-effect | 0.125 | 0.0033 | 0.8511 | 0.0759 | false |
| pure-effect | 0.250 | 0.0056 | 0.8533 | 0.1542 | false |
| pure-effect | 0.500 | 0.0189 | 0.8667 | 0.3195 | false |
| combined-shift | 0.250 | 0.0067 | 0.8367 | 0.3743 | false |
| combined-shift | 0.500 | 0.0133 | 0.8433 | 0.7905 | false |

Rare-validation gates remained acceptable:

| Metric | Trust-region domain expert |
| --- | ---: |
| rare-validation selected shrinkage | 1.0 |
| rare-validation overall coverage | 0.9001 |
| rare-validation rare coverage | 0.9281 |
| rare-validation rare rank error | 0.0194 |
| rare-scale in-domain guard coverage | 0.9508 |

Do not submit this candidate to LUMI. The next roadmap step should tune or
redesign the constrained expert objective locally before another production-like
run. The immediate implementation direction is a small trust-region sweep over
stronger weights/tighter tolerances, or a domain-localized trust region that
penalizes only in-domain overlap contexts while allowing more effect-specific
OOD movement.

## Trust-Region Expert Sweep Tuning Check

Implemented a compact trust-region sweep inside domain-expert fitting. Each
pure-effect and combined-shift expert is now trained under three in-domain
log-inflation trust-region settings, and each trained expert still goes through
the post-fit shrinkage grid:

| Setting | Weight | Log tolerance |
| --- | ---: | ---: |
| `moderate_w3_tol108` | 3.0 | 0.0770 |
| `strong_w6_tol106` | 6.0 | 0.0583 |
| `tight_w10_tol104` | 10.0 | 0.0392 |

The tuning run was intentionally smaller than the rare32 production-like
workflow:

```text
/private/tmp/neural_hmsc_v8_domain_expert_trust_sweep_tuning
```

Validation:

```text
pytest tests/test_neural_hmsc_conditional_calibration.py -q
18 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
41 passed
```

The compact tuning run selected the combined-shift expert at shrinkage `1.0`.
This is not a qualification result because the smaller run has different
dimensions and fewer rare/OOD/SBC batches, but it verifies that the trust sweep
can produce accepted candidates and that tighter trust settings reduce
in-domain extra-inflation deltas.

| Candidate | Trust setting | Selected shrinkage | Target gain | Extra-inflation delta | Accepted |
| --- | --- | ---: | ---: | ---: | --- |
| pure-effect | `moderate_w3_tol108` | 0.5 | 0.0037 | 0.0334 | false |
| pure-effect | `strong_w6_tol106` | 0.5 | 0.0037 | 0.0298 | false |
| pure-effect | `tight_w10_tol104` | 0.5 | 0.0037 | 0.0307 | false |
| combined-shift | `moderate_w3_tol108` | 1.0 | 0.0148 | 0.1098 | true |
| combined-shift | `strong_w6_tol106` | 1.0 | 0.0148 | 0.1019 | true |
| combined-shift | `tight_w10_tol104` | 1.0 | 0.0148 | 0.0931 | true |

The compact final SBC coverage remained below OOD floors, so this run is only a
selector-tuning check:

| Domain | Coverage 95 |
| --- | ---: |
| in distribution | 0.9231 |
| covariate shift | 0.9231 |
| effect-size shift | 0.7704 |
| combined shift | 0.7185 |

The next roadmap step is to rerun the production-like rare32 local sanity
workflow with the trust-region sweep enabled. If the sweep still fails under
rare32 settings, the next implementation should move from a global trust
region to a domain-localized overlap penalty.

## Trust-Region Expert Sweep Rare32 Local Sanity

The production-like rare32 local sanity workflow was rerun with the
trust-region sweep enabled:

```text
/private/tmp/neural_hmsc_v8_domain_expert_trust_sweep_local_sanity_rare32
```

The run used the same rare32 settings as the prior production-like checks:

```text
rare_calibration_datasets = 32
rare_validation_datasets = 32
sbc_datasets = 32
sbc_draws = 128
```

The selector accepted one candidate internally: the pure-effect expert trained
with `tight_w10_tol104` at shrinkage `1.0`. This candidate cleared the explicit
held-out selector gate by the minimum margin, but it did not improve the final
production-like SBC result.

| Candidate | Trust setting | Selected shrinkage | Target gain | Effect-size selector coverage | Combined selector coverage | Extra-inflation delta | Accepted |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| pure-effect | `moderate_w3_tol108` | 1.0 | 0.0311 | 0.8789 | 0.8389 | 0.6216 | false |
| pure-effect | `strong_w6_tol106` | 1.0 | 0.0278 | 0.8756 | 0.8378 | 0.5832 | false |
| pure-effect | `tight_w10_tol104` | 1.0 | 0.0100 | 0.8578 | 0.8311 | 0.2377 | true |
| combined-shift | `moderate_w3_tol108` | 1.0 | 0.0211 | 0.8933 | 0.8511 | 1.7572 | false |
| combined-shift | `strong_w6_tol106` | 1.0 | 0.0133 | 0.8789 | 0.8433 | 0.9445 | false |
| combined-shift | `tight_w10_tol104` | 1.0 | 0.0078 | 0.8767 | 0.8378 | 0.7283 | false |

Final 95% coefficient coverage did not qualify and did not improve relative to
the staged/global-trust results:

| Domain | Staged combined objective | Trust-region sweep rare32 |
| --- | ---: | ---: |
| in distribution | 0.9308 | 0.9306 |
| covariate shift | 0.9157 | 0.9157 |
| effect-size shift | 0.8290 | 0.8282 |
| combined shift | 0.8056 | 0.8056 |

Final OOD diagnostics remained below floor:

| Domain | Diagnostic coverage | Shortfall | Final multiplier mean | Final multiplier q95 | Learned combined-context mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| effect-size shift | 0.8511 | 0.0489 | 1.1981 | 3.1036 | 0.0818 |
| combined shift | 0.8161 | 0.0839 | 5.5518 | 19.2144 | 0.1091 |

Effect-quantile diagnostics remain weak in the same strata:

| Domain | Quantile | Coverage | Final multiplier mean |
| --- | --- | ---: | ---: |
| effect-size shift | q2 | 0.8556 | 0.8037 |
| effect-size shift | q3 | 0.7911 | 0.8363 |
| effect-size shift | q4 | 0.9022 | 2.3489 |
| combined shift | q2 | 0.8222 | 5.8530 |
| combined shift | q3 | 0.8467 | 6.4366 |
| combined shift | q4 | 0.7733 | 4.0644 |

Rare-validation gates remained acceptable:

| Metric | Trust-region sweep rare32 |
| --- | ---: |
| rare-validation selected shrinkage | 1.0 |
| rare-validation overall coverage | 0.9001 |
| rare-validation rare coverage | 0.9281 |
| rare-validation rare rank error | 0.0194 |
| rare-scale in-domain guard coverage | 0.9508 |

Do not submit this candidate to LUMI. The rare32 result shows that a global
trust-region sweep can create an internally accepted boundary candidate, but
the accepted candidate is too weak to transfer into final OOD coverage. The
next roadmap step should replace the global trust region with a domain-localized
overlap penalty and strengthen the expert-selection gate so boundary candidates
must produce a practical held-out gain and no OOD-domain degradation before
application.

## Domain-Localized Overlap Penalty Tuning Check

Replaced the global trust-region expert constraint with a domain-localized
overlap penalty. Expert fitting now penalizes in-domain OOD log-inflation drift
only in contexts that resemble the target OOD expert domain:

| Expert | Localized in-domain overlap context |
| --- | --- |
| pure-effect | high effect signal with support-close coefficients |
| combined-shift | support-excess, non-low effect, low-design, low-community coefficients |

The expert-selection gate was also strengthened:

| Gate | Value |
| --- | ---: |
| minimum effect-size target gain | 0.0200 |
| minimum combined-shift target gain | 0.0200 |
| maximum non-target OOD coverage loss | 0.0000 |
| maximum localized overlap excess loss | 0.1200 |

The localized-overlap fitting grid is:

| Setting | Weight | Log tolerance |
| --- | ---: | ---: |
| `localized_w4_tol106` | 4.0 | 0.0583 |
| `localized_w8_tol104` | 8.0 | 0.0392 |
| `localized_w12_tol103` | 12.0 | 0.0296 |

Validation:

```text
pytest tests/test_neural_hmsc_conditional_calibration.py -q
18 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
41 passed
```

The compact tuning check was:

```text
/private/tmp/neural_hmsc_v8_domain_expert_overlap_tuning
```

No candidate passed the strengthened gate. The previous boundary-style
combined-shift candidate remained non-degrading and overlap-controlled, but its
target gain was only `0.0148`, below the new practical-gain floor:

| Candidate | Penalty setting | Selected shrinkage | Target gain | Non-target OK | Overlap excess loss | Accepted |
| --- | --- | ---: | ---: | --- | ---: | --- |
| pure-effect | `localized_w4_tol106` | 0.5 | 0.0037 | true | 0.0000 | false |
| pure-effect | `localized_w8_tol104` | 0.5 | 0.0037 | true | 0.0000 | false |
| pure-effect | `localized_w12_tol103` | 0.5 | 0.0037 | true | 0.0001 | false |
| combined-shift | `localized_w4_tol106` | 1.0 | 0.0148 | true | 0.0238 | false |
| combined-shift | `localized_w8_tol104` | 1.0 | 0.0148 | true | 0.0283 | false |
| combined-shift | `localized_w12_tol103` | 1.0 | 0.0148 | true | 0.0323 | false |

Compact final SBC coverage remained below OOD floors, so this was only a
selector behavior check:

| Domain | Coverage 95 |
| --- | ---: |
| in distribution | 0.9231 |
| covariate shift | 0.9222 |
| effect-size shift | 0.7713 |
| combined shift | 0.7139 |

Do not run the production-like rare32 workflow yet. The strengthened gate now
rejects weak boundary candidates as intended, but the expert objective does not
produce a practical held-out gain. The next roadmap step should improve the
domain-expert objective itself, likely by increasing target-domain coverage
pressure or adding effect-quantile-specific expert losses while preserving the
localized overlap penalty and strengthened selection gate.

## Target-Pressure Domain-Expert Tuning Check

Added expert-only target-domain coverage and effect-quantile coverage pressure
to the domain-expert fitting objective while preserving the localized overlap
penalty and strengthened no-degradation selector gate.

| Profile | Localized overlap weight | Target coverage weight | Effect-quantile weight |
| --- | ---: | ---: | ---: |
| `localized_w4_tol106` | 4.0 | 4.0 | 3.0 |
| `localized_w8_tol104` | 8.0 | 6.0 | 4.0 |
| `localized_w12_tol103` | 12.0 | 8.0 | 5.0 |

Validation:

```text
pytest tests/test_neural_hmsc_conditional_calibration.py -q
18 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
41 passed
```

The compact tuning check was:

```text
/private/tmp/neural_hmsc_v8_domain_expert_target_pressure_tuning
```

The stronger objective did not change selector behavior. No candidate passed
the strengthened gate; the best combined-shift target gain remained `0.0148`,
below the practical-gain floor of `0.0200`.

| Candidate | Profile | Target gain | Non-target OK | Overlap excess loss | Accepted |
| --- | --- | ---: | --- | ---: | --- |
| pure-effect | `localized_w4_tol106` | 0.0037 | true | 0.0000 | false |
| pure-effect | `localized_w8_tol104` | 0.0037 | true | 0.0000 | false |
| pure-effect | `localized_w12_tol103` | 0.0037 | true | 0.0002 | false |
| combined-shift | `localized_w4_tol106` | 0.0148 | true | 0.0242 | false |
| combined-shift | `localized_w8_tol104` | 0.0148 | true | 0.0295 | false |
| combined-shift | `localized_w12_tol103` | 0.0148 | true | 0.0343 | false |

Compact final SBC coverage remained unchanged from the localized-overlap tuning
check:

| Domain | Coverage 95 |
| --- | ---: |
| in distribution | 0.9231 |
| covariate shift | 0.9222 |
| effect-size shift | 0.7713 |
| combined shift | 0.7139 |

Do not run rare32 yet. The target-pressure objective appears saturated under
the current expert parameterization. The next roadmap step should change the
domain-expert representation rather than adding more scalar weights: add
effect-bin-specific expert amplitudes or a target-domain-specific slope/cap
parameterization that can move the weak OOD effect quantiles without broad
in-domain inflation.

## Effect-Bin Domain-Expert Head Compact Tuning Check

Changed the domain-expert representation from one scalar amplitude per expert
to a 14-parameter effect-shift head. The new head keeps the previous
pure-effect and combined-shift context gates, and adds three localized
effect-bin log-amplitudes for each expert. Metadata now records
`parameter_count`, `effect_bin_centers`, `effect_bin_width`,
`pure_effect_bin_log_amplitudes`, and
`combined_effect_bin_log_amplitudes`. Legacy 8-parameter effect-shift heads
remain loadable.

Validation:

```text
python -m py_compile pyhmsc/neural/conditional_calibration.py

pytest tests/test_neural_hmsc_conditional_calibration.py -q
18 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
41 passed
```

The compact tuning check was:

```text
/private/tmp/neural_hmsc_v8_domain_expert_bin_head_tuning
```

The learned metadata head was:

| Field | Value |
| --- | ---: |
| parameter count | 14 |
| pure bin log amplitudes | `[0.0836, 0.0726, 0.0775]` |
| combined bin log amplitudes | `[0.0551, 0.0440, 0.0437]` |

The strengthened held-out selector accepted the combined-shift expert:

| Candidate | Profile | Selected shrinkage | Target gain | Non-target OK | Overlap excess loss | Accepted |
| --- | --- | ---: | ---: | --- | ---: | --- |
| pure-effect | `localized_w4_tol106` | 1.0 | 0.0074 | true | 0.0000 | false |
| pure-effect | `localized_w8_tol104` | 1.0 | 0.0074 | true | 0.0000 | false |
| pure-effect | `localized_w12_tol103` | 1.0 | 0.0074 | true | 0.0000 | false |
| combined-shift | `localized_w4_tol106` | 1.0 | 0.0222 | true | 0.0000 | true |
| combined-shift | `localized_w8_tol104` | 1.0 | 0.0222 | true | 0.0000 | true |
| combined-shift | `localized_w12_tol103` | 1.0 | 0.0222 | true | 0.0000 | true |

Selector baseline versus selected record:

| Metric | Baseline | Selected |
| --- | ---: | ---: |
| effect-size held-out coverage | 0.7222 | 0.7259 |
| combined-shift held-out coverage | 0.7148 | 0.7370 |
| mean held-out OOD coverage | 0.7185 | 0.7315 |
| worst held-out OOD coverage | 0.7148 | 0.7259 |

Compact final SBC coverage remained below production floors, so this is still
only a representation/selector check:

| Domain | Coverage 95 |
| --- | ---: |
| covariate shift | 0.9352 |
| effect-size shift | 0.7370 |
| combined shift | 0.7685 |

The next roadmap step is a production-like rare32 local sanity workflow with
the effect-bin-specific domain-expert head enabled, using the same rare
calibration/validation settings as prior rare32 runs. Compare selected expert
branch, bin amplitudes, held-out OOD coverage, effect-quantile diagnostics,
final multiplier quantiles, in-domain gate deltas, and rare-validation gates
against the previous target-pressure and localized-overlap compact baselines.
Do not submit a LUMI comparison unless the rare32 local gates hold.

## Effect-Bin Domain-Expert Head Rare32 Local Sanity

The production-like rare32 local sanity workflow was rerun with the
effect-bin-specific domain-expert head enabled:

```text
/private/tmp/neural_hmsc_v8_domain_expert_bin_head_local_sanity_rare32
```

The run used the same rare32 settings as the prior production-like checks:

```text
rare_calibration_datasets = 32
rare_validation_datasets = 32
sbc_datasets = 32
sbc_draws = 128
```

The compact selector result did not transfer to the production-like rare32
gate. The final selected expert remained `baseline` with
`selected_shrinkage = 0.0` and reason
`no_candidate_passed_selection_gate`. Final serialized effect-bin head
amplitudes were:

| Field | Value |
| --- | ---: |
| parameter count | 14 |
| pure bin log amplitudes | `[0.0686, 0.0067, 0.0028]` |
| combined bin log amplitudes | `[0.0000, 0.0000, 0.0000]` |

Selector comparison:

| Run | Selected expert | Accepted | Mean held-out OOD | Worst held-out OOD |
| --- | --- | --- | ---: | ---: |
| localized-overlap compact | baseline | false | 0.7167 | 0.7111 |
| target-pressure compact | baseline | false | 0.7167 | 0.7111 |
| effect-bin compact | combined-shift | true | 0.7315 | 0.7259 |
| effect-bin rare32 | baseline | false | 0.8378 | 0.8289 |

Production-like final SBC coverage:

| Domain | Prior shrinkage rare32 | Trust-sweep rare32 | Effect-bin rare32 |
| --- | ---: | ---: | ---: |
| covariate shift | 0.9344 | 0.9344 | 0.9350 |
| effect-size shift | 0.8511 | 0.8511 | 0.8517 |
| combined shift | 0.8178 | 0.8161 | 0.8183 |

Candidate selector details:

| Candidate | Profile | Target gain | Non-target OK | Overlap excess loss | Extra-inflation delta | Max group-loss delta | Accepted |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| pure-effect | `localized_w4_tol106` | 0.0256 | false | 0.0064 | 0.5805 | 0.1212 | false |
| pure-effect | `localized_w8_tol104` | 0.0256 | true | 0.0227 | 0.5492 | 0.1185 | false |
| pure-effect | `localized_w12_tol103` | 0.0267 | true | 0.0373 | 0.5440 | 0.1235 | false |
| combined-shift | `localized_w4_tol106` | 0.0400 | true | 1.2964 | 3.1190 | 0.8863 | false |
| combined-shift | `localized_w8_tol104` | 0.0267 | true | 0.6636 | 2.1783 | 0.6294 | false |
| combined-shift | `localized_w12_tol103` | 0.0211 | true | 0.4376 | 1.7546 | 0.5063 | false |

The combined-shift candidates could move coverage, but only by violating the
localized overlap and in-domain extra-inflation controls. The best diagnostic
combined-shift profile reached held-out selector coverage `0.8689` for
combined shift and `0.9011` for effect-size shift, but its localized overlap
excess loss was `1.2964`, far above the `0.1200` gate. Pure-effect candidates
met the practical target-gain threshold but exceeded in-domain group-loss and
extra-inflation gates.

Effect-quantile coverage for the selected baseline:

| Domain | q2 | q3 | q4 |
| --- | ---: | ---: | ---: |
| effect-size shift | 0.8578 | 0.7889 | 0.9022 |
| combined shift | 0.8233 | 0.8489 | 0.7778 |

Final multiplier summaries for the selected baseline:

| Domain | Mean | Median | q75 | q95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| covariate shift | 7.0569 | 1.3139 | 18.7940 | 19.2402 | 19.2446 |
| effect-size shift | 1.2160 | 0.8104 | 1.1150 | 3.3083 | 10.1676 |
| combined shift | 5.5884 | 1.4661 | 8.0622 | 19.1960 | 19.2441 |

Rare-validation gates remained acceptable and matched the prior production-like
rare32 behavior:

| Metric | Effect-bin rare32 |
| --- | ---: |
| rare-validation selected shrinkage | 1.0 |
| rare-validation overall coverage | 0.9001 |
| rare-validation rare coverage | 0.9285 |
| rare-validation rare rank error | 0.0194 |
| rare-validation intermediate-design coverage | 0.9001 |
| rare-validation high-design coverage | 0.9000 |
| rare-scale in-domain overall coverage | 0.9508 |
| rare-scale in-domain rare rank error | 0.0270 |

Do not submit this candidate to LUMI. The production-like rare32 run shows that
the effect-bin representation can produce target-domain gains, but the gains
are not separable from in-domain extra inflation under the current context
gates. The next roadmap step should make the bin-head movement more
domain-local and gate-compatible before another rare32 run. A plausible
implementation is a support/design/community-conditioned bin-amplitude cap or
selection step: allow effect-bin amplitudes only in OOD-like contexts that have
low in-domain overlap, and add direct penalties on per-bin in-domain
extra-inflation and group-loss deltas during expert fitting.

## Context-Capped Effect-Bin Expert Compact Tuning Check

Implemented the first domain-local effect-bin constraint. The serialized
14-parameter head shape is unchanged, but effect-bin amplitudes are now applied
through fixed context caps:

- pure-effect amplitudes are capped by support-excess or low-design context
- combined-shift amplitudes are capped by support-excess, low-design, and, when
  `Y` is available, low-community context
- expert fitting adds per-effect-bin in-domain penalties for extra log
  inflation, rank-mean drift, and low coverage in the bin's active context

Validation:

```text
python -m py_compile pyhmsc/neural/conditional_calibration.py

pytest tests/test_neural_hmsc_conditional_calibration.py -q
18 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
41 passed
```

The compact tuning check was:

```text
/private/tmp/neural_hmsc_v8_domain_expert_context_capped_bin_tuning
```

The context caps made the selector gate-compatible, but too conservative. No
candidate passed the strengthened gate because target gains fell below the
practical-gain floor:

| Candidate | Profile | Target gain | Non-target OK | Overlap excess loss | Extra-inflation delta | Accepted |
| --- | --- | ---: | --- | ---: | ---: | --- |
| pure-effect | `localized_w4_tol106` | 0.0037 | true | 0.0000 | 0.0276 | false |
| pure-effect | `localized_w8_tol104` | 0.0037 | true | 0.0000 | 0.0276 | false |
| pure-effect | `localized_w12_tol103` | 0.0037 | true | 0.0000 | 0.0276 | false |
| combined-shift | `localized_w4_tol106` | 0.0111 | true | 0.0271 | 0.0989 | false |
| combined-shift | `localized_w8_tol104` | 0.0111 | true | 0.0315 | 0.0985 | false |
| combined-shift | `localized_w12_tol103` | 0.0111 | true | 0.0356 | 0.0981 | false |

Selector comparison:

| Run | Selected expert | Accepted | Mean held-out OOD | Worst held-out OOD |
| --- | --- | --- | ---: | ---: |
| target-pressure compact | baseline | false | 0.7167 | 0.7111 |
| effect-bin compact | combined-shift | true | 0.7315 | 0.7259 |
| context-capped effect-bin compact | baseline | false | 0.7185 | 0.7148 |

Compact final SBC coverage was not qualified and remained below OOD floors:

| Domain | Coverage 95 |
| --- | ---: |
| covariate shift | 0.9389 |
| effect-size shift | 0.7426 |
| combined shift | 0.7667 |

Do not run rare32 yet. The context-capped bin head confirms the right failure
axis: the cap/penalty controls in-domain leakage, but suppresses target-domain
movement too much. The next roadmap step should tune or redesign the cap shape
locally, likely by using softer community/design caps, target-domain-specific
cap floors, or a two-stage fit that first learns target-domain bin movement and
then projects it through the per-bin in-domain gate.

## Soft-Floor Effect-Bin Cap Compact Tuning Check

Implemented the first cap-shape tuning pass by adding fixed target-domain
floors to the context caps while keeping the serialized 14-parameter head
unchanged:

- pure-effect context floor: `0.20`
- combined-shift context floor: `0.35`
- per-effect-bin in-domain penalties remain active

Validation:

```text
python -m py_compile pyhmsc/neural/conditional_calibration.py

pytest tests/test_neural_hmsc_conditional_calibration.py -q
18 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
41 passed
```

The compact tuning check was:

```text
/private/tmp/neural_hmsc_v8_domain_expert_soft_floor_bin_tuning
```

The soft floors increased learned combined-bin amplitudes relative to the hard
cap, but did not improve selected held-out coverage. No candidate passed:

| Candidate | Profile | Target gain | Non-target OK | Overlap excess loss | Extra-inflation delta | Accepted |
| --- | --- | ---: | --- | ---: | ---: | --- |
| pure-effect | `localized_w4_tol106` | 0.0037 | true | 0.0000 | 0.0279 | false |
| pure-effect | `localized_w8_tol104` | 0.0037 | true | 0.0000 | 0.0279 | false |
| pure-effect | `localized_w12_tol103` | 0.0037 | true | 0.0000 | 0.0279 | false |
| combined-shift | `localized_w4_tol106` | 0.0111 | true | 0.0274 | 0.1028 | false |
| combined-shift | `localized_w8_tol104` | 0.0111 | true | 0.0321 | 0.1028 | false |
| combined-shift | `localized_w12_tol103` | 0.0111 | true | 0.0364 | 0.1027 | false |

Selector comparison:

| Run | Selected expert | Accepted | Mean held-out OOD | Worst held-out OOD |
| --- | --- | --- | ---: | ---: |
| effect-bin compact | combined-shift | true | 0.7315 | 0.7259 |
| hard context-capped compact | baseline | false | 0.7185 | 0.7148 |
| soft-floor context-capped compact | baseline | false | 0.7185 | 0.7148 |

Do not run rare32 yet. Softer context floors alone do not solve the tradeoff:
the per-bin gate remains conservative enough that target-domain movement stays
below the practical-gain floor. The next roadmap step should implement a
two-stage target-then-projection expert fit: first learn target-domain
effect-bin movement without the full in-domain bin gate, then project or shrink
the learned bin amplitudes through explicit per-bin in-domain gate constraints.

## Two-Stage Target-Then-Projection Expert Implementation

Implemented the next selector profile without changing the serialized
14-parameter effect-shift head shape. The profile is named
`two_stage_target_w6_projection` and is recorded with kind
`two_stage_target_then_projection`.

Stage 1 fits the target-domain expert with:

- moderate target-domain coverage pressure
- moderate effect-quantile pressure
- reduced in-domain gate weight
- relaxed localized-overlap tolerance
- no per-bin in-domain penalty during the fit

Stage 2 projects the learned expert before selector acceptance. The projection
crosses the standard shrinkage grid with branch-specific effect-head cap factors
`(0.25, 0.5, 0.75, 1.0)`. The cap shrinks only the active branch's scalar
effect-head amplitude and three effect-bin amplitudes back toward the baseline
parameter vector; all candidates still must pass the existing non-target,
overlap, and in-domain gate constraints.

Validation:

```text
python -m py_compile pyhmsc/neural/conditional_calibration.py

pytest tests/test_neural_hmsc_conditional_calibration.py -q
18 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
41 passed
```

The next step is a compact local tuning check for this two-stage profile, using
the same rare calibration/validation settings as the hard-cap and soft-floor
context-cap runs. Do not run rare32 until the compact check shows practical
target-domain gain with controlled overlap and in-domain gate deltas.

## Two-Stage Projection Compact Tuning Check

Ran the compact tuning workflow:

```text
/private/tmp/neural_hmsc_v8_domain_expert_two_stage_projection_tuning
```

The two-stage projection profile did not qualify. The selector again kept the
baseline expert:

| Run | Selected expert | Accepted | Mean held-out OOD | Worst held-out OOD | Effect-size shift | Combined shift |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| hard context-capped compact | baseline | false | 0.7185 | 0.7148 | 0.7222 | 0.7148 |
| soft-floor context-capped compact | baseline | false | 0.7185 | 0.7148 | 0.7222 | 0.7148 |
| two-stage projection compact | baseline | false | 0.7185 | 0.7148 | 0.7222 | 0.7148 |

Best two-stage candidate diagnostics:

| Candidate | Selected shrinkage | Projection cap | Target gain | Non-target OK | Overlap excess loss | Extra-inflation delta | Max group extra-cap delta | Accepted |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| pure-effect | 0.25 | 0.75 | 0.0037 | true | 0.0000 | 0.0265 | 0.0313 | false |
| combined-shift | 1.00 | 0.25 | 0.0111 | true | 0.0245 | 0.1014 | 0.0889 | false |

The projection step did reduce some leakage relative to the soft-floor
single-stage profile, especially for combined-shift overlap loss
(`0.0245` versus `0.0274` at the best comparable candidate). It did not create
additional held-out OOD coverage gain. The pure-effect branch stayed far below
the `0.0200` practical-gain floor, and the combined-shift branch also stayed
below that floor while still slightly exceeding the max group extra-cap delta
gate (`0.0889` versus threshold `0.0800`).

Do not run rare32. The next roadmap step should stop using scalar/bin-amplitude
projection as the main control mechanism. The target movement appears coverage
quantized under this compact design: projection can reduce in-domain leakage,
but it cannot move enough held-out coefficients across the interval boundary.
The next implementation should change the target signal itself, likely by
fitting the expert on a larger or explicitly harder target-domain compact pool,
or by replacing interval-coverage pressure with a margin-aware loss that
targets near-miss OOD coefficients before applying the existing projection gate.

## Margin-Aware Target-Signal Compact Tuning Check

Implemented the target-signal variant by adding a margin-aware expert loss to
the two-stage target-then-projection profile. The loss is active only for the
profile with kind `two_stage_target_then_projection`; existing single-stage
localized profiles keep `margin_weight = 0.0`. The margin term upweights
target-domain coefficients that are outside but close to the baseline nominal
interval, so optimization pressure focuses on near-miss coefficients that could
cross the interval boundary after modest scale movement. Projection and
held-out selector gates are unchanged.

Validation:

```text
python -m py_compile pyhmsc/neural/conditional_calibration.py

pytest tests/test_neural_hmsc_conditional_calibration.py -q
18 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
41 passed
```

Ran the compact tuning workflow:

```text
/private/tmp/neural_hmsc_v8_domain_expert_margin_projection_tuning
```

The margin-aware profile did not qualify and did not materially change the
compact outcome:

| Run | Selected expert | Accepted | Mean held-out OOD | Worst held-out OOD | Effect-size shift | Combined shift |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| two-stage projection compact | baseline | false | 0.7185 | 0.7148 | 0.7222 | 0.7148 |
| margin-aware projection compact | baseline | false | 0.7185 | 0.7148 | 0.7222 | 0.7148 |

Best margin-aware two-stage candidate diagnostics:

| Candidate | Selected shrinkage | Projection cap | Target gain | Non-target OK | Overlap excess loss | Extra-inflation delta | Max group extra-cap delta | Accepted |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| pure-effect | 0.25 | 0.75 | 0.0037 | true | 0.0000 | 0.0265 | 0.0313 | false |
| combined-shift | 1.00 | 0.25 | 0.0111 | true | 0.0246 | 0.1015 | 0.0890 | false |

This is effectively unchanged from the previous two-stage projection run. The
near-miss margin term did not create enough held-out coverage movement under
the compact pool. Do not run rare32. The next roadmap step should change the
compact calibration data rather than add another scalar target-loss term:
construct a larger or explicitly harder target-domain OOD calibration pool,
preferably enriched for near-boundary misses and separated by pure-effect versus
combined-shift regimes, then rerun the same projection selector.

## Hard Target-Pool Projection Compact Tuning Check

Implemented a hard target-domain OOD calibration pool in the benchmark runner.
The new CLI options are:

- `--conditional-calibration-ood-hard-target-multiplier`
- `--conditional-calibration-ood-hard-target-candidate-multiplier`

When the hard-target multiplier is greater than one, `effect_size_shift` and
`combined_shift` calibration datasets are drawn from a larger candidate pool,
scored by near-boundary coefficient misses under the current posterior, and the
hardest subset is retained. Non-target OOD regimes keep the original sampling
path. Existing defaults preserve prior behavior.

Validation:

```text
python -m py_compile examples/run_neural_hmsc_benchmark.py

pytest tests/test_neural_hmsc_lumi_workflow.py -q
5 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
42 passed
```

Ran the compact hard-pool workflow:

```text
/private/tmp/neural_hmsc_v8_domain_expert_hard_pool_projection_tuning
```

Settings:

```text
--conditional-calibration-ood-datasets 4
--conditional-calibration-ood-hard-target-multiplier 3
--conditional-calibration-ood-hard-target-candidate-multiplier 2
```

The selector still kept `baseline`, but the hard pool moved target gains much
closer to the practical-gain floor:

| Run | Selected expert | Accepted | Mean held-out OOD | Worst held-out OOD | Effect-size shift | Combined shift |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| margin-aware projection compact | baseline | false | 0.7185 | 0.7148 | 0.7222 | 0.7148 |
| hard-pool projection compact | baseline | false | 0.7568 | 0.7420 | 0.7716 | 0.7420 |

Best hard-pool two-stage candidate diagnostics:

| Candidate | Training observations | Selected shrinkage | Projection cap | Target gain | Non-target OK | Overlap excess loss | Extra-inflation delta | Max group extra-cap delta | Accepted |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| pure-effect | 810 | 1.00 | 1.00 | 0.0185 | true | 0.0003 | 0.1600 | 0.0912 | false |
| combined-shift | 810 | 1.00 | 0.25 | 0.0160 | true | 0.0752 | 0.1667 | 0.0913 | false |

Final compact SBC coverage after applying the selected baseline expert:

| Domain | Coverage 95 |
| --- | ---: |
| covariate shift | 0.9389 |
| effect-size shift | 0.7642 |
| combined shift | 0.7630 |

Do not run rare32 yet. The hard pool is the first recent change that materially
increases held-out target-domain gains, but it still misses the `0.0200`
practical-gain floor and breaches the max group extra-cap delta gate. The next
roadmap step should refine the hard-pool/projection interaction: either
increase target-pool hardness modestly while adding a stricter projection cap
for in-domain extra-cap loss, or split hard target batches into independent
train/evaluation batches so accepted gains are not dominated by one within-batch
split.

## Split Hard Target-Pool Compact Checks

Implemented independent hard target-pool batch grouping in the benchmark
runner. When hard target-pool selection is enabled for `effect_size_shift` or
`combined_shift`, the selected hard datasets are now emitted as two calibration
batches instead of one combined batch. The existing domain-expert selector then
uses `alternating_batches` instead of `within_batch_axis0`, giving separate
batch-level train/evaluation folds for target experts.

Validation:

```text
python -m py_compile examples/run_neural_hmsc_benchmark.py

pytest tests/test_neural_hmsc_lumi_workflow.py -q
6 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
43 passed
```

Ran split hard-pool compact checks:

```text
/private/tmp/neural_hmsc_v8_domain_expert_hard_pool_split_projection_tuning
/private/tmp/neural_hmsc_v8_domain_expert_hard_pool_split_x4_projection_tuning
```

Selector comparison:

| Run | Hard multiplier | Split mode | Selected expert | Mean held-out OOD | Worst held-out OOD | Effect-size shift | Combined shift |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| hard-pool compact | 3 | within-batch | baseline | 0.7568 | 0.7420 | 0.7716 | 0.7420 |
| split hard-pool compact | 3 | alternating batches | baseline | 0.7630 | 0.7469 | 0.7790 | 0.7469 |
| split hard-pool compact | 4 | alternating batches | baseline | 0.7537 | 0.7343 | 0.7731 | 0.7343 |

Best split-candidate diagnostics:

| Run | Candidate | Target gain | Projection cap | Overlap excess loss | Extra-inflation delta | Max group extra-cap delta | Accepted |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| split x3 | pure-effect | 0.0111 | 0.75 | 0.0005 | 0.1507 | 0.0075 | false |
| split x3 | combined-shift | 0.0173 | 0.75 | 0.0993 | 0.2063 | 0.0112 | false |
| split x4 | pure-effect | 0.0130 | 0.75 | 0.0004 | 0.1525 | 0.0756 | false |
| split x4 | combined-shift | 0.0130 | 1.00 | 0.0604 | 0.1542 | 0.0760 | false |

The independent split fixed the extra-cap gate problem: max group extra-cap
deltas fell from about `0.091` in the unsplit hard-pool run to `0.0075` to
`0.0112` for split x3. However, target gains no longer reached the same level:
pure-effect fell from `0.0185` to `0.0111`, while combined-shift rose slightly
from `0.0160` to `0.0173` but still missed the `0.0200` floor. Increasing
hardness to x4 did not help; it lowered combined-shift held-out coverage and
target gains.

Do not run rare32. The next roadmap step should make hard-pool selection
gate-aware rather than simply harder or split differently. The likely direction
is to score candidate target-domain datasets by near-boundary misses while
penalizing high in-domain-overlap contexts, or to build separate hard pools for
train and validation with matched near-boundary difficulty so target gain does
not collapse under independent evaluation.

## Gate-Aware Hard Target-Pool Compact Check

Implemented gate-aware hard target-pool scoring. The hard-pool selector still
scores target-domain OOD candidates by near-boundary coefficient misses, but
now subtracts a regime-specific overlap proxy for `effect_size_shift` and
`combined_shift`. The proxy uses posterior effect magnitude, covariate-support
shift, and community occupancy to demote candidates that are likely to drive
in-domain overlap or extra-inflation gate failures. Default behavior remains
unchanged for non-target regimes and for runs without hard target-pool
selection.

Validation:

```text
python -m py_compile examples/run_neural_hmsc_benchmark.py

pytest tests/test_neural_hmsc_lumi_workflow.py -q
7 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
44 passed
```

Ran the compact gate-aware hard-pool workflow:

```text
/private/tmp/neural_hmsc_v8_domain_expert_gate_aware_hard_pool_tuning
```

Selector comparison:

| Run | Selected expert | Mean held-out OOD | Worst held-out OOD | Effect-size shift | Combined shift |
| --- | --- | ---: | ---: | ---: | ---: |
| split hard-pool x3 | baseline | 0.7630 | 0.7469 | 0.7790 | 0.7469 |
| gate-aware hard-pool x3 | baseline | 0.7710 | 0.7642 | 0.7778 | 0.7642 |

Best gate-aware candidate diagnostics:

| Candidate | Target gain | Non-target gain | Projection cap | Overlap excess loss | Extra-inflation delta | Max group extra-cap delta | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pure-effect | 0.0148 | 0.0198 | 1.00 | 0.0008 | 0.1891 | 0.0205 | false |
| combined-shift | 0.0173 | 0.0136 | 0.25 | 0.0895 | 0.1843 | 0.0196 | false |

Gate-aware scoring improved the selected baseline held-out OOD diagnostics,
especially combined-shift coverage (`0.7642` versus `0.7469` in split x3), and
kept max group extra-cap deltas well below the `0.0800` gate. It still did not
qualify because neither target gain reached the `0.0200` practical-gain floor.

Do not run rare32. The next roadmap step should build matched train/validation
hard pools rather than changing a scalar score again. The target selector needs
training batches and evaluation batches with similar near-boundary difficulty;
otherwise target gain either collapses under independent evaluation or remains
below the practical-gain floor despite better gate control.

## Matched Hard Target-Pool Compact Check

Implemented score-balanced hard target-pool grouping. When hard target-pool
selection is enabled, the runner now balances the two target-domain calibration
batches by the same gate-aware near-boundary score used for candidate
selection, instead of using a simple alternating split. This creates matched
train/evaluation hard pools with nearly equal aggregate hard-pool difficulty.

Validation:

```text
python -m py_compile examples/run_neural_hmsc_benchmark.py

pytest tests/test_neural_hmsc_lumi_workflow.py -q
8 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
45 passed
```

Ran the compact matched hard-pool workflow:

```text
/private/tmp/neural_hmsc_v8_domain_expert_matched_hard_pool_tuning
```

Selector comparison:

| Run | Selected expert | Mean held-out OOD | Worst held-out OOD | Effect-size shift | Combined shift |
| --- | --- | ---: | ---: | ---: | ---: |
| gate-aware hard-pool x3 | baseline | 0.7710 | 0.7642 | 0.7778 | 0.7642 |
| matched hard-pool x3 | baseline | 0.7611 | 0.7519 | 0.7704 | 0.7519 |

Best matched-pool candidate diagnostics:

| Candidate | Target gain | Non-target gain | Projection cap | Overlap excess loss | Extra-inflation delta | Max group extra-cap delta | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pure-effect | 0.0148 | 0.0173 | 1.00 | 0.0007 | 0.1746 | 0.0095 | false |
| combined-shift | 0.0148 | 0.0148 | 0.75 | 0.1164 | 0.2046 | 0.0100 | false |

Matched grouping controlled the max group extra-cap gate, but it worsened
held-out OOD coverage and reduced combined-shift target gain relative to the
gate-aware x3 split. Do not run rare32. The next roadmap step should instrument
the hard-pool selection path rather than change another grouping heuristic:
record selected candidate score distributions, train/evaluation score summaries,
near-boundary miss summaries, and overlap-proxy summaries by regime so the
training/evaluation mismatch can be diagnosed directly.

## Hard-Pool Instrumentation Check

Implemented hard-pool selection diagnostics in the benchmark record. When target
hard-pool selection is enabled, `benchmark_record.json` now records, by target
regime:

- candidate and selected score distributions,
- raw near-boundary score summaries before overlap penalty,
- overlap-proxy summaries,
- miss-rate, near-boundary miss-rate, excess-miss, and absolute-z summaries,
- matched train/evaluation group summaries and aggregate score delta.

Validation:

```text
python -m py_compile examples/run_neural_hmsc_benchmark.py

pytest tests/test_neural_hmsc_lumi_workflow.py -q
9 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
46 passed
```

Ran the compact instrumentation workflow:

```text
/private/tmp/neural_hmsc_v8_hard_pool_instrumentation_check
```

The selector again kept `baseline` with mean held-out OOD coverage `0.7611`,
worst OOD-domain coverage `0.7519`, effect-size-shift coverage `0.7704`, and
combined-shift coverage `0.7519`.

Hard-pool diagnostic summary:

| Regime | Candidates | Kept | Candidate score mean | Selected score mean | Selected overlap mean | Matched group score delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| effect-size shift | 24 | 12 | -0.0599 | -0.0485 | 0.2815 | 0.0605 |
| combined shift | 24 | 12 | -0.0463 | -0.0293 | 0.2458 | 0.1345 |

The instrumentation confirms that the matched hard pools are not simply
imbalanced; selected pools still carry substantial overlap penalties, and the
combined-shift train/evaluation score delta remains larger than the
effect-size-shift delta. Do not run rare32. The next roadmap step is to inspect
the new diagnostic arrays in detail and redesign hard-pool construction around
two explicit constraints: enough raw near-boundary misses and low target-domain
overlap, with separate train/evaluation matching for each regime.

## Constrained Hard-Pool Compact Check

Implemented constrained hard-pool construction for target OOD regimes. The
selector now separates raw near-boundary difficulty from target-domain overlap:
it first requires candidates to meet a raw near-boundary score threshold and a
low-overlap threshold, relaxes overlap before relaxing raw difficulty if the
eligible pool is too small, ranks eligible candidates by raw difficulty and low
overlap, and then forms train/evaluation batches with a matcher that balances
raw difficulty, overlap, and final score within each target regime.

Validation:

```text
python -m py_compile examples/run_neural_hmsc_benchmark.py

pytest tests/test_neural_hmsc_lumi_workflow.py -q
11 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
48 passed
```

Ran the compact constrained hard-pool workflow:

```text
/private/tmp/neural_hmsc_v8_constrained_hard_pool_check
```

Selector comparison:

| Run | Selected expert | Mean held-out OOD | Worst held-out OOD | Effect-size shift | Combined shift |
| --- | --- | ---: | ---: | ---: | ---: |
| instrumentation baseline | baseline | 0.7611 | 0.7519 | 0.7704 | 0.7519 |
| constrained hard-pool | baseline | 0.7525 | 0.7457 | 0.7593 | 0.7457 |

Constrained hard-pool diagnostics:

| Regime | Eligible | Fallback | Raw threshold q | Overlap threshold q | Selected raw mean | Selected overlap mean | Selected score mean | Matched group score delta |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| effect-size shift | 12 | false | 0.45 | 0.95 | 0.0501 | 0.2822 | -0.0487 | 0.0132 |
| combined shift | 12 | false | 0.45 | 0.95 | 0.0592 | 0.2606 | -0.0320 | 0.0107 |

The new matcher fixed the train/evaluation imbalance: score deltas fell from
`0.0605` to `0.0132` for effect-size shift and from `0.1345` to `0.0107` for
combined shift. The selection constraint itself did not solve the target-pool
quality problem. Both regimes had to relax the overlap threshold to the `0.95`
quantile to keep enough raw near-boundary misses, which means the current
candidate pool does not contain enough low-overlap hard misses. Do not run
rare32. The next roadmap step is to change candidate-pool generation, not
selection: generate or oversample target-domain candidates in low-overlap
contexts first, then apply the constrained hard-pool selector.

## Low-Overlap Candidate-Pool Compact Check

Implemented low-overlap target candidate-pool generation. Target regimes now
generate a wider seed window, prefilter that generated pool into a low-overlap
candidate pool while retaining a raw near-boundary miss floor, and then apply
the constrained hard-pool selector and regime-specific train/evaluation
matching.

Validation:

```text
python -m py_compile examples/run_neural_hmsc_benchmark.py

pytest tests/test_neural_hmsc_lumi_workflow.py -q
12 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
49 passed
```

Ran the compact low-overlap candidate-pool workflow:

```text
/private/tmp/neural_hmsc_v8_low_overlap_candidate_pool_check
```

Selector comparison:

| Run | Selected expert | Mean held-out OOD | Worst held-out OOD | Effect-size shift | Combined shift | Extra-inflation loss | Max extra-cap loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constrained hard-pool | baseline | 0.7525 | 0.7457 | 0.7593 | 0.7457 | 0.1898 | 0.1768 |
| low-overlap candidate pool | baseline | 0.7562 | 0.7519 | 0.7605 | 0.7519 | 0.1339 | 0.1060 |

Candidate-pool diagnostics:

| Regime | Generated | Candidate pool | Generated overlap mean | Pool overlap mean | Selected raw mean | Selected overlap mean | Selected score mean | Matched group score delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| effect-size shift | 72 | 24 | 0.2818 | 0.2761 | 0.0564 | 0.2751 | -0.0399 | 0.0009 |
| combined shift | 72 | 24 | 0.2683 | 0.2377 | 0.0638 | 0.2351 | -0.0184 | 0.0140 |

Low-overlap generation improved the candidate pool, especially for
combined-shift overlap and final score, and further reduced in-domain
extra-inflation penalties. It still did not produce an accepted expert, and
held-out OOD remains below the earlier instrumentation baseline (`0.7611`
mean OOD). Do not run rare32. The next roadmap step is to change the simulated
target-domain candidate distribution itself, not the seed-window prefilter:
add an explicit low-overlap target candidate regime or context-controlled OOD
simulator variant that creates more low-overlap hard misses before selection.

## Context-Controlled Low-Overlap Candidate Check

Implemented a simulator-level low-overlap target candidate context. The default
OOD simulator remains unchanged. For hard target calibration candidate
generation only, `effect_size_shift` now uses shifted covariate support to
reduce pure-effect overlap context, while `combined_shift` uses a positive
intercept context to reduce low-community overlap. The benchmark runner records
the `low_overlap` candidate context in hard-pool diagnostics.

Validation:

```text
python -m py_compile pyhmsc/neural/simulator.py \
  examples/run_neural_hmsc_benchmark.py

pytest tests/test_neural_hmsc_simulator.py \
  tests/test_neural_hmsc_lumi_workflow.py -q
21 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py \
  tests/test_neural_hmsc_simulator.py -q
58 passed
```

Ran the compact context-controlled candidate workflow:

```text
/private/tmp/neural_hmsc_v8_context_controlled_low_overlap_check
```

Selector comparison:

| Run | Selected expert | Mean held-out OOD | Worst held-out OOD | Effect-size shift | Combined shift | Extra-inflation loss | Max extra-cap loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low-overlap seed prefilter | baseline | 0.7562 | 0.7519 | 0.7605 | 0.7519 | 0.1339 | 0.1060 |
| context-controlled candidate | baseline | 0.7969 | 0.7580 | 0.8358 | 0.7580 | 0.2852 | 0.1809 |

Candidate-pool diagnostics:

| Regime | Generated overlap mean | Pool overlap mean | Selected raw mean | Selected overlap mean | Selected score mean | Matched group score delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| effect-size shift | 0.1308 | 0.1050 | 0.0696 | 0.0997 | 0.0347 | 0.0079 |
| combined shift | 0.2311 | 0.1970 | 0.0673 | 0.1958 | -0.0012 | 0.0055 |

The simulator-level candidate distribution solved the main pool-quality issue:
effect-size candidate scores are now positive, overlap is much lower, and
target held-out OOD coverage improved sharply. The selector still kept
`baseline` because no expert passed the gate. The best pure-effect candidates
had target gain `0.0173`, below the `0.0200` floor, and added about `0.3090`
extra-inflation loss. The best combined-shift candidates had target gain
`0.0136` and added about `0.2643` extra-inflation loss.

Do not run rare32. The next roadmap step is to keep the context-controlled
candidate distribution, but redesign expert fitting/acceptance for it: reduce
in-domain extra inflation and improve combined-shift target gain, likely with
a smaller expert amplitude schedule or a combined-shift-specific target loss
with a strict extra-inflation projection before selection.

## Gate-Compatible Expert Projection Check

Implemented a finer expert shrinkage schedule, strict branch-specific projection
caps, a combined-shift-specific target-loss profile, and gate-compatible
projection selection. Candidate records now prefer the best gate-compatible
projection row when no row clears the full target-gain floor, instead of
reporting the highest-gain high-inflation row.

Validation:

```text
python -m py_compile pyhmsc/neural/conditional_calibration.py

pytest tests/test_neural_hmsc_conditional_calibration.py -q
18 passed

pytest tests/test_neural_hmsc_public_api.py \
  tests/test_neural_hmsc_conditional_calibration.py \
  tests/test_neural_hmsc_lumi_workflow.py \
  tests/test_neural_hmsc_simulator.py -q
58 passed
```

Ran the compact gate-compatible projection workflow:

```text
/private/tmp/neural_hmsc_v8_gate_compatible_projection_check
```

Selector comparison:

| Run | Selected expert | Mean held-out OOD | Worst held-out OOD | Effect-size shift | Combined shift |
| --- | --- | ---: | ---: | ---: | ---: |
| context-controlled candidate | baseline | 0.7969 | 0.7580 | 0.8358 | 0.7580 |
| gate-compatible projection | baseline | 0.7969 | 0.7580 | 0.8358 | 0.7580 |

Best gate-compatible candidate diagnostics:

| Candidate | Target gain | Selection rule | Shrinkage | Projection cap | Extra-inflation delta | Max extra-cap delta | Mean OOD | Worst OOD |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pure-effect | 0.0136 | best_gate_compatible | 0.75 | 1.00 | 0.2369 | 0.0177 | 0.8093 | 0.7691 |
| combined-shift | 0.0099 | best_gate_compatible | 0.75 | 0.0625 | 0.1837 | 0.0137 | 0.8080 | 0.7679 |

The stricter projection path reduced in-domain extra-inflation deltas below the
explicit `0.2500` gate, but target gains fell below the practical-gain floor.
This means post-fit projection is now risk-controlling the experts, while the
remaining bottleneck is the fitted combined-shift signal under that constraint.
Do not run rare32. The next roadmap step is to move the extra-inflation
constraint into expert fitting itself instead of relying on post-fit projection:
fit a combined-shift expert with an in-objective gate-compatible amplitude
penalty or target-domain curriculum that can recover combined-shift gain while
staying below the extra-inflation gate.
