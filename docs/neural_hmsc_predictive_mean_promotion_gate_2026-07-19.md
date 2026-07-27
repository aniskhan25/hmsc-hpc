# Predictive Mean Promotion Gate

Date: 2026-07-19

Purpose: replace prose-only promotion judgment for predictive-mean competitors
with an executable cross-dataset no-degradation gate.

## Implementation

Added:

- `pyhmsc/neural/predictive_selection.py`
- `examples/evaluate_neural_hmsc_predictive_promotion.py`
- `tests/test_neural_hmsc_predictive_selection.py`

The gate compares a predictive-mean candidate against the promoted
scale-only `external_monotone` predictive path on held-out real-data metrics.
By default:

- baseline model: `neural_predictive_only_calibrated`
- candidate model: `neural_predictive_mean_calibrated`
- maximum Brier ratio: `1.0`
- maximum log-loss ratio: `1.0`

Every dataset must pass. A candidate that improves one real dataset but
degrades another is rejected for default promotion. Optional simulated summary
rows can also be supplied so simulated proper-score improvement remains a
necessary but insufficient precondition.

The gate is predictive-only. It does not change coefficient posterior samples,
coefficient calibration, SBC/OOD gates, rare-validation gates, or Python-only
HMSC parity semantics.

## Validation Command

```bash
python3 examples/evaluate_neural_hmsc_predictive_promotion.py \
  --dataset whittaker=/private/tmp/neural_response_mean_realdata_20006616_20006620/whittaker/whittaker_heldout_metrics.csv \
  --dataset big_spatial=/private/tmp/neural_response_mean_realdata_20006616_20006620/big_spatial/big_spatial_transfer_heldout_metrics.csv \
  --output /private/tmp/neural_response_mean_realdata_20006616_20006620/promotion_gate
```

The command wrote:

- `predictive_mean_promotion_gate.json`
- `predictive_mean_promotion_gate_datasets.csv`
- `predictive_mean_promotion_gate.md`

It exited with status `1`, as expected, because the response-mean candidate
failed the no-degradation gate on Whittaker.

## Gate Result

| dataset | passed | Brier ratio | log-loss ratio |
| --- | --- | ---: | ---: |
| `whittaker` | no | `1.0030` | `1.0005` |
| `big_spatial` | yes | `0.9962` | `0.9944` |

Failure reasons:

- `whittaker`: Brier ratio `1.0030` exceeds `1.0`
- `whittaker`: log-loss ratio `1.0005` exceeds `1.0`

## Decision

The executable gate confirms the previous decision: keep
`probit_response_affine` as an experimental predictive-only competitor, but do
not promote it as default.

The next implementation direction should target a domain-conditional or
selector-based predictive-mean competitor that can retain the Big Spatial gain
without Whittaker degradation. The promotion check should remain frozen: any
new candidate must pass this cross-dataset no-degradation gate before default
promotion is reconsidered.
