# Response-Scale Predictive Mean Real-Data Transfer Validation

Date: 2026-07-19

Purpose: validate `external_monotone + probit_response_affine` against the
promoted `external_monotone` real-data path on Whittaker and Big Spatial, while
keeping Python-only HMSC parity metrics attached and keeping the response-mean
layer explicitly predictive-only.

## Implementation

The Whittaker real-data runner now exposes:

- `--predictive-mean-calibration {none,probit_response_affine}`
- `--predictive-mean-calibration-validation-datasets`
- `--predictive-mean-calibration-max-brier-ratio`
- `--predictive-mean-calibration-max-log-loss-ratio`
- `--predictive-mean-calibration-min-improvement`

When enabled, the runner fits the response-scale mean correction on simulated
calibration batches and gates it on independent simulated validation batches.
The coefficient posterior remains unchanged. The scale-only predictive artifact
is written to `neural_predictive_distribution_scale_only.h5`, while
`neural_predictive_distribution.h5` becomes the final predictive-only artifact
with `predictive_mean_calibration` metadata.

The Big Spatial transfer runner now reads `predictive_mean_calibration` metadata
from the frozen Whittaker `neural_predictive_distribution.h5` and applies that
same transform without refitting weights or calibration on target data.

## Local Checks

Completed:

- `python3 -m py_compile examples/run_neural_hmsc_whittaker.py examples/run_neural_hmsc_big_spatial_transfer.py pyhmsc/neural/mean_calibration.py`
- `bash -n docs/lumi_neural_hmsc_whittaker_sbatch.sh docs/lumi_neural_hmsc_big_spatial_transfer_sbatch.sh`
- `python3 examples/run_neural_hmsc_whittaker.py --help`
- `python3 examples/run_neural_hmsc_big_spatial_transfer.py --help`
- Tiny Whittaker response-mean smoke with `test_sites = 2`
- Tiny Whittaker transfer-shape smoke with `test_sites = 12`
- Tiny Big Spatial transfer smoke using the transfer-shape Whittaker artifact
- `pytest tests/test_neural_hmsc_mean_calibration.py tests/test_neural_hmsc_predictive_scores.py -q`
- `git diff --check`

The Big Spatial smoke used a temporary `/private/tmp` source-acceptance override
only to exercise the transfer path from an intentionally underpowered local
Whittaker smoke. No repository files or LUMI validation gates were changed.

## LUMI Submission

Whittaker job:

- Job: `20006616`
- State at submission check: running on `dev-g`
- Run root:
  `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_whittaker_response_mean_realdata_20260719`
- Whittaker parity metrics:
  `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/whittaker_r_python_parity_scaled_20260718_082539/whittaker_r_python_parity_metrics.json`
- Predictive mean calibration: `probit_response_affine`
- Predictive mean validation datasets: `128`
- Predictive mean minimum improvement: `0.0001`

Big Spatial job:

- Job: `20006620`
- Dependency: `afterok:20006616`
- State at submission check: pending on dependency
- Run root:
  `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_big_spatial_response_mean_realdata_20260719`
- Frozen Whittaker source:
  `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_whittaker_response_mean_realdata_20260719`
- Big Spatial parity metrics:
  `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/direct_r_python_big_spatial_full_parity_20260719/big_spatial_plants_validation_model_spatial_full/direct_r_python_parity_metrics.json`

## Completed Results

Both jobs completed successfully:

- Whittaker job `20006616`: `COMPLETED`, elapsed `00:10:07`, exit `0:0`
- Big Spatial job `20006620`: `COMPLETED`, elapsed `00:01:35`, exit `0:0`

Local result copy:
`/private/tmp/neural_response_mean_realdata_20006616_20006620`

The Whittaker response selector chose a non-identity transform:

- method: `probit_response_affine`
- selected: `true`
- slope: `1.0250`
- intercept: `0.0250`
- validation Brier ratio: `0.9997`
- validation log-loss ratio: `0.9993`

The same frozen transform was carried into Big Spatial without refitting.

## Whittaker Outcome

Acceptance passed:

- coefficient SBC acceptance: `true`
- held-out predictive acceptance: `true`
- combined qualification: `true`
- reference parity qualified: `true`
- final predictive model: `neural_predictive_mean_calibrated`

Held-out metrics:

| model | Brier | log loss | macro AUC | prevalence MAE | richness MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `neural_uncalibrated` | `0.076894` | `0.274523` | `0.538601` | `0.073989` | `3.426325` |
| `neural_predictive_only_calibrated` | `0.077161` | `0.273998` | `0.549285` | `0.077720` | `3.716901` |
| `neural_predictive_mean_calibrated` | `0.077395` | `0.274133` | `0.552126` | `0.077568` | `3.670441` |
| `qualified_python_mcmc_fixed` | `0.073985` | `0.262233` | `0.546126` | `0.069312` | `3.372503` |

Relative to scale-only predictive calibration, the response-mean layer slightly
worsened Whittaker Brier and log-loss:

- Brier ratio vs scale-only: `1.0030`
- log-loss ratio vs scale-only: `1.0005`

It improved Whittaker macro AUC, prevalence MAE, richness MAE, and rare-species
metrics, but the main proper-score gate did not improve.

Relative to the qualified Python MCMC comparator, response-mean calibrated
prediction remained worse on core proper scores:

- Brier ratio vs MCMC: `1.0461`
- log-loss ratio vs MCMC: `1.0454`

## Big Spatial Outcome

Acceptance passed:

- inherited source SBC acceptance: `true`
- inherited source qualification: `true`
- target predictive acceptance: `true`
- frozen predictive transfer acceptance: `true`
- reference parity qualified: `true`
- final predictive model: `neural_predictive_mean_calibrated`

Held-out metrics:

| model | Brier | log loss | macro AUC | prevalence MAE | richness MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `neural_uncalibrated` | `0.051242` | `0.203917` | `0.640447` | `0.051664` | `4.566353` |
| `neural_predictive_only_calibrated` | `0.052811` | `0.211215` | `0.632069` | `0.059852` | `5.148121` |
| `neural_predictive_mean_calibrated` | `0.052608` | `0.210028` | `0.635241` | `0.058232` | `5.057118` |
| `qualified_python_mcmc_fixed` | `0.047467` | `0.191435` | `0.646242` | `0.039737` | `3.535748` |

Relative to scale-only predictive calibration, the response-mean layer improved
Big Spatial transfer:

- Brier ratio vs scale-only: `0.9962`
- log-loss ratio vs scale-only: `0.9944`
- macro AUC ratio vs scale-only: `1.0050`
- prevalence MAE ratio vs scale-only: `0.9729`
- richness MAE ratio vs scale-only: `0.9823`

Relative to the qualified Python MCMC comparator, response-mean calibrated
prediction remained worse on core proper scores:

- Brier ratio vs MCMC: `1.1083`
- log-loss ratio vs MCMC: `1.0971`

## Decision

Do not promote `probit_response_affine` into the default predictive path.

It remains a valid experimental predictive-only competitor: both real-data
acceptance gates passed, parity metrics were attached, and the Big Spatial
transfer metrics improved relative to scale-only predictive calibration.
However, the Whittaker proper scores worsened slightly relative to scale-only,
and the qualified Python MCMC comparator remains better on Brier and log loss
for both real datasets.

The result suggests the response-affine idea is too weak and too global for
real-data promotion. Future predictive-mean work should use stricter
cross-dataset selection or domain-conditional response calibration, with
promotion requiring improvement or no-degradation on both Whittaker and Big
Spatial proper scores.

## Next Action

Keep the implemented response-mean path available as experimental. The next
roadmap step should return to predictive-mean competitor development with a
cross-dataset no-degradation gate: any candidate must improve simulated
proper-score evaluation and pass Whittaker plus Big Spatial real-data
no-degradation checks against the promoted `external_monotone` scale-only
predictive path before a default-promotion discussion.
