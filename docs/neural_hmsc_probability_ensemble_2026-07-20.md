# Frozen Probability Deep-Ensemble Comparison

Date: 2026-07-20

LUMI job: `20032201`

Frozen source run:
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_source_transfer_realdata_sensitivity_20260720`

Evaluation root:
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_probability_ensemble_20260720`

Local artifacts:
`/private/tmp/neural_hmsc_probability_ensemble_20260720`

## Protocol

The competitor averages posterior predictive probabilities across frozen neural
members. It does not average coefficients, calibration parameters, or final
metrics. Every comparison uses identical member sets for the scale-only
baseline and affine candidate.

Evaluated subsets:

- full ensemble: seeds `20260721`, `20260722`, `20260723`;
- leave out `20260721`;
- leave out `20260722`;
- leave out `20260723`.

All four subsets were evaluated on Whittaker and Big Spatial. Target outcomes
were unavailable until scale-only and affine probabilities had been generated
for every member. Existing R/Python parity and dataset-acceptance provenance
passed for all inputs.

Promotion required Brier, log-loss, RMSE, and richness-MAE no degradation for
all eight dataset/subset rows, plus strict Brier and log-loss improvement for
the full Big Spatial ensemble.

## Results

| dataset | ensemble | Brier ratio | log-loss ratio | RMSE ratio | richness-MAE ratio | passed |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Whittaker | full | `1.0000` | `1.0000` | `1.0000` | `1.0000` | yes |
| Whittaker | leave out `20260721` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | yes |
| Whittaker | leave out `20260722` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | yes |
| Whittaker | leave out `20260723` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | yes |
| Big Spatial | full | `0.9975` | `0.9949` | `0.9988` | `0.9857` | yes |
| Big Spatial | leave out `20260721` | `0.9992` | `0.9973` | `0.9996` | `0.9931` | yes |
| Big Spatial | leave out `20260722` | `0.9994` | `0.9971` | `0.9997` | `0.9927` | yes |
| Big Spatial | leave out `20260723` | `0.9947` | `0.9908` | `0.9974` | `0.9713` | yes |

Full Big Spatial ensemble:

- Brier: `0.051345` to `0.051218`;
- log loss: `0.206504` to `0.205443`;
- predictive RMSE: `0.226595` to `0.226314`;
- prevalence MAE: `0.056358` to `0.055063`;
- richness MAE: `5.0020` to `4.9303`.

The job completed in `35` seconds (`32` seconds measured inside the workflow).
The final decision was `probability_ensemble_promotion_candidate`.

## Interpretation

Probability averaging resolves the seed-specific affine instability that the
generic and target-shaped simulation selectors could not detect. Even the two
leave-one-out ensembles containing harmful seed `20260723` remain
nondegrading. This supports a stable neural competitor relative to the frozen
scale-only ensemble baseline.

The evidence does not yet establish superiority to qualified Python MCMC, and
the current evaluator is a research workflow rather than a reusable prediction
API. Do not relabel coefficient-posterior calibration or SBC semantics: this is
still a predictive-only ensemble.

## Next Step

Implement a reusable probability-ensemble deployment artifact and API. The
artifact should store ordered member paths/hashes, seeds, calibration roles,
species/formula compatibility, and provenance; its `predict_mean` operation
must average member response probabilities. Integrate it into the real-data
reporting path and run one clean Whittaker/Big Spatial requalification against
both the matched scale-only ensemble and qualified Python MCMC before changing
the default predictive deployment policy.
