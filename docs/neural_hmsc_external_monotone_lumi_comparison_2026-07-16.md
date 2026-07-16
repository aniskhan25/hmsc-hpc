# Neural-HMSC External Monotone Five-Seed LUMI Comparison

Date: 2026-07-16

Job: `19940765`

Run root:
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_external_monotone_fixed_eval_20260716`

Local downloaded summary:
`/private/tmp/neural_hmsc_external_monotone_lumi_19940765`

## Setup

The LUMI job compared `scalar`, `default`, and `external_monotone` on five
fixed probit seeds: `20260716`, `20260717`, `20260718`, `20260719`, and
`20260720`.

Each seed used:

- fresh scalar checkpoint,
- frozen scalar checkpoint reused by `default` and `external_monotone`,
- identical independent SBC/OOD evaluation rows within the seed,
- `n_sites = 32`,
- `n_species = 45`,
- `train_datasets = 8`,
- `calibration_datasets = 8`,
- `rare_calibration_datasets = 8`,
- `rare_validation_datasets = 8`,
- `sbc_datasets = 8`,
- `sbc_draws = 64`,
- OOD regimes `covariate_shift`, `effect_size_shift`, and `combined_shift`.

The job completed successfully:

- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:06:08`
- Workflow wall time: `317` seconds

## Aggregate Result

| run | accepted seeds | in-domain | rare | mean OOD | worst OOD | effect-size | combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scalar | 0/5 | 0.9139 +/- 0.0051 | 0.8260 +/- 0.0407 | 0.7249 +/- 0.0079 | 0.6063 +/- 0.0077 | 0.7030 +/- 0.0078 | 0.6063 +/- 0.0077 |
| default | 5/5 | 0.9226 +/- 0.0074 | 0.8551 +/- 0.0286 | 0.8241 +/- 0.0090 | 0.7356 +/- 0.0204 | 0.7356 +/- 0.0204 | 0.8244 +/- 0.0144 |
| external_monotone | 5/5 | 0.9404 +/- 0.0072 | 0.8741 +/- 0.0200 | 0.8596 +/- 0.0080 | 0.8120 +/- 0.0196 | 0.8120 +/- 0.0196 | 0.8472 +/- 0.0122 |

Against scalar, `external_monotone` improved:

- mean OOD by `+0.1347 +/- 0.0155`,
- combined shift by `+0.2409 +/- 0.0140`.

Against `default`, `external_monotone` improved:

- mean OOD by `+0.0355`,
- worst OOD by `+0.0765`,
- effect-size shift by `+0.0765`,
- combined shift by `+0.0228`,
- rare-prevalence coverage by `+0.0190`,
- with in-domain coverage still inside the acceptance window.

## External-Monotone Selection

The external wrapper selected nonzero offsets for all five seeds:

| seed | selected | shrinkage | log offsets | multipliers |
| --- | --- | ---: | --- | --- |
| 20260716 | external_monotone | 1.0 | `[0.0, 0.0, 0.6931]` | `[1.0, 1.0, 2.0]` |
| 20260717 | external_monotone | 1.0 | `[0.0, 0.0, 0.6931]` | `[1.0, 1.0, 2.0]` |
| 20260718 | external_monotone | 1.0 | `[0.0, 0.0, 0.6931]` | `[1.0, 1.0, 2.0]` |
| 20260719 | external_monotone | 1.0 | `[0.0, 0.0, 0.6931]` | `[1.0, 1.0, 2.0]` |
| 20260720 | external_monotone | 1.0 | `[0.0, 0.0, 0.6931]` | `[1.0, 1.0, 2.0]` |

## Per-Seed Summary

| seed | run | accepted | mean OOD | worst OOD | effect-size | combined | in-domain | rare |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260716 | scalar | false | 0.7148 | 0.5972 | 0.6954 | 0.5972 | 0.9056 | 0.7619 |
| 20260716 | default | true | 0.8392 | 0.7704 | 0.7704 | 0.8407 | 0.9306 | 0.8095 |
| 20260716 | external_monotone | true | 0.8741 | 0.8435 | 0.8435 | 0.8602 | 0.9491 | 0.8571 |
| 20260717 | scalar | false | 0.7160 | 0.5991 | 0.6954 | 0.5991 | 0.9130 | 0.8000 |
| 20260717 | default | true | 0.8293 | 0.7333 | 0.7333 | 0.8259 | 0.9111 | 0.8444 |
| 20260717 | external_monotone | true | 0.8627 | 0.8083 | 0.8083 | 0.8454 | 0.9278 | 0.8444 |
| 20260718 | scalar | false | 0.7302 | 0.6065 | 0.7037 | 0.6065 | 0.9139 | 0.8627 |
| 20260718 | default | true | 0.8145 | 0.7222 | 0.7222 | 0.8019 | 0.9194 | 0.8627 |
| 20260718 | external_monotone | true | 0.8540 | 0.8056 | 0.8056 | 0.8287 | 0.9407 | 0.8824 |
| 20260719 | scalar | false | 0.7340 | 0.6102 | 0.7167 | 0.6102 | 0.9157 | 0.8718 |
| 20260719 | default | true | 0.8173 | 0.7417 | 0.7417 | 0.8157 | 0.9213 | 0.8974 |
| 20260719 | external_monotone | true | 0.8534 | 0.8194 | 0.8194 | 0.8407 | 0.9389 | 0.8974 |
| 20260720 | scalar | false | 0.7296 | 0.6185 | 0.7037 | 0.6185 | 0.9213 | 0.8333 |
| 20260720 | default | true | 0.8204 | 0.7102 | 0.7102 | 0.8380 | 0.9306 | 0.8611 |
| 20260720 | external_monotone | true | 0.8540 | 0.7833 | 0.7833 | 0.8611 | 0.9454 | 0.8889 |

## Conclusion

The five-seed LUMI fixed-evaluation comparison qualifies
`external_monotone` as a stronger competitor than both `scalar` and `default`
under the current compact fixed-evaluation gate. The result is not just an
internal calibration-batch improvement: the final selected calibration improved
fixed independent OOD rows in all five seeds and preserved in-domain/rare
acceptance.

Promotion decision: do not promote from the compact gate alone. The result is
strong, but the evaluation is still compact (`sbc_datasets = 8`,
`sbc_draws = 64`, `train_datasets = 8`). The next step is one production-shape
LUMI confirmation with larger independent SBC/OOD evaluation counts, preferably
`sbc_datasets >= 32` and `sbc_draws >= 256`, using the same fixed-evaluation
discipline and shared seeds. Promote `external_monotone` only if it still beats
`default` on mean OOD, worst OOD, effect-size shift, and combined shift while
preserving in-domain and rare-prevalence acceptance.
