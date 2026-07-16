# Neural-HMSC External Monotone Production-Shape Confirmation

Date: 2026-07-16

LUMI job: `19942240`

Remote run root:
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_external_monotone_production_confirm_20260716`

Downloaded local summary:
`/private/tmp/neural_hmsc_external_monotone_production_confirm_19942240`

## Configuration

The run compared `scalar`, `default`, and `external_monotone` with the same
five seeds used in the compact gate: `20260716` through `20260720`.

- shape: `40` sites, `75` species
- training datasets: `8`
- calibration datasets: `8`
- rare calibration datasets: `8`
- rare validation datasets: `8`
- SBC datasets: `32`
- SBC draws: `256`
- SBC bins: `10`
- OOD regimes: `covariate_shift`, `effect_size_shift`, `combined_shift`
- external monotone datasets: `4`
- external monotone maximum multiplier: `2`
- partition: `dev-g`
- elapsed time: `00:09:46`
- exit code: `0:0`

## Aggregate Results

| run | accepted seeds | in-domain | rare | mean OOD | worst OOD | effect-size shift | combined shift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scalar | 0/5 | 0.9414 | 0.9365 | 0.7889 | 0.6973 | 0.7594 | 0.6973 |
| default | 4/5 | 0.9409 | 0.9437 | 0.9086 | 0.7955 | 0.7955 | 0.9627 |
| external_monotone | 5/5 | 0.9442 | 0.9520 | 0.9203 | 0.8214 | 0.8214 | 0.9684 |

Against `default`, `external_monotone` changed the five-seed means by:

- mean OOD: `+0.0117`
- worst OOD: `+0.0258`
- effect-size shift: `+0.0258`
- combined shift: `+0.0057`
- in-domain: `+0.0033`
- rare prevalence: `+0.0083`

## Seed-Level Notes

`external_monotone` passed the fixed-evaluation gate for all five seeds. The
wrapper matched `default` exactly on seeds `20260717`, `20260718`, and
`20260719`, and improved seeds `20260716` and `20260720`. Seed `20260720` is
important because `default` failed the fixed-evaluation acceptance gate there,
while `external_monotone` passed.

The production-shape result is therefore not a broad per-seed movement across
all seeds. It is a conservative fallback: it leaves already-qualified default
solutions unchanged and applies a nonzero correction on the seeds where the
held-out external gate selects it.

## Decision

`external_monotone` qualifies for promotion as the default compact competitor.
It beats `default` on mean OOD, worst OOD, effect-size shift, and combined
shift in the production-shape confirmation while preserving in-domain and rare
validation acceptance. The combined-shift margin is small, so promotion should
be documented as a conservative gated improvement rather than as evidence that
the combined-shift problem is fully solved.

Next step: promote `external_monotone` in the benchmark workflow default path
and keep `default` available as the legacy conditional baseline for direct
comparisons.

Promotion implementation: `docs/lumi_neural_hmsc_benchmark_sbatch.sh` now
defaults `COEFFICIENT_CALIBRATION` to `external_monotone` and exposes the
external monotone calibration controls. The legacy conditional baseline remains
available by setting `COEFFICIENT_CALIBRATION=conditional`.
