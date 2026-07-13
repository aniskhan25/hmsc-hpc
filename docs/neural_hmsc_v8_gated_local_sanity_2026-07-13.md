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
