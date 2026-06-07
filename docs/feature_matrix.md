# Feature Matrix

| Feature | Status |
| --- | --- |
| No-R Python-native compile/sample path | Supported |
| Fixed effects | Supported |
| Gaussian/Poisson/Probit responses | Supported |
| Traits with formula | Supported |
| Phylogenetic covariance matrix | Supported |
| Newick tree parsing | Optional `phylo` extra |
| iid random intercepts | Supported |
| iid random-intercept real-data validation | Supported; Whittaker plant iid site-level run validated on LUMI |
| Full spatial random intercepts | Supported |
| Random slopes | Compile/load support; sampler guarded |
| GPP/NNGP spatial effects | Not yet |
| HDF5 posterior output | Supported |
| Zarr posterior output | Optional extra |
| Posterior metadata preservation | Supported for native HDF5/Zarr output |
| HDF5 posterior merge | Supported |
| Posterior predictive checks | Supported |
| Known random-effect posterior predictive checks | Supported when the fit is loaded with its model config |
| Site richness posterior predictive checks | Supported |
| Covariate gradient summaries | Supported for richness and response-weighted traits |
| Trait-effect posterior summaries | Supported for Gamma mean/interval tables |
| Species association summaries | Supported from random-level `Lambda` samples |
| Random-effect posterior summaries | Supported for `Eta` and random-intercept `Lambda`; random-slope `Lambda` requires `x_index` |
| Random-effect diagnostics | Supported for nested `Eta`/`Lambda` arrays |
| LUMI Slurm array workflow | Compile/sample-array/merge templates included |
| Chain status and retry helpers | Supported |
| Slow simulation recovery tests | Supported for fixed effects; smoke tests for traits/random/spatial |
| Deterministic spatial validation simulator | Supported via `simulate_spatial_effect_data` |
| Simulated spatial validation project | Added with fixed, iid, and full-spatial configs |
| Simulated spatial validation analyzer | Supported; fixed/iid/spatial LUMI validation completed |
| Big spatial real-data validation project | Added with fixed, iid, and full-spatial configs |
| Big spatial real-data analyzer | Supported; fixed/iid/spatial LUMI validation completed |
| Legacy TensorFlow updater tests | Current fixtures cover observed-response masks |
| CLI compile/sample/summarize/predict/validate | Supported |
| CLI init validation | Supported |
| No-R example smoke runner | Supported |
| ArviZ diagnostics | Optional extra |

## Validated Real-Data Runs

The Whittaker plant validation project in
`examples/projects/whittaker_plants_hmsc_book` has been run on LUMI without R.

| Model | Run | Result |
| --- | --- | --- |
| Fixed-effect probit with traits and phylogeny | `whittaker_ppc_step8` | Completed on LUMI; species occupancy PPC covered `75 / 75`, site richness PPC covered `40 / 52` |
| iid site-level random intercept probit with traits and phylogeny | `whittaker_iid_long` | Completed on LUMI; species occupancy PPC covered `75 / 75`, site richness PPC covered `52 / 52` |

The longer iid run used 2 chains, 3000 saved samples, 1000 transient
iterations, and thin 10. It preserved the expected ecological pattern:
most species respond negatively to TMG, richness decreases along TMG, and
community-weighted CN increases along TMG. Gamma diagnostics were clean
(`max R-hat = 1.00136`, `min ESS = 468.10`). Beta diagnostics were near-clean
with 2 R-hat flags and 2 ESS flags out of 150 coefficients.

## Validated Simulated Runs

The deterministic spatial validation project in
`examples/projects/simulated_spatial_validation` has been run on LUMI without R.

| Model | Run | Result |
| --- | --- | --- |
| Fixed-effect probit | `spatial_validation_full_codex` | Completed on LUMI; beta signs recovered `4 / 4`, species PPC covered `5 / 5`, site richness PPC covered `36 / 36` |
| iid random-intercept probit | `spatial_validation_full_codex` | Completed on LUMI; beta signs recovered `4 / 4`, species PPC covered `5 / 5`, site richness PPC covered `36 / 36`, Eta/truth correlation `0.675717` |
| full spatial random-intercept probit | `spatial_validation_full_codex` | Completed on LUMI; beta signs recovered `4 / 4`, species PPC covered `5 / 5`, site richness PPC covered `36 / 36`, Eta/truth correlation `0.868592` |

The run used 2 chains, 1000 saved samples, 500 transient iterations, and thin
10. The full Slurm job completed in 7 minutes 15 seconds on `dev-g` with
TensorFlow 2.16 and an MI250X GPU.

## Validated Spatial Real-Data Runs

The compact big-spatial plant validation project in
`examples/projects/big_spatial_plants_validation` has been run on LUMI without
R.

| Model | Run | Result |
| --- | --- | --- |
| Fixed-effect probit | `big_spatial_real_validation_codex` | Completed on LUMI; species PPC covered `40 / 40`, site richness PPC covered `309 / 400`, neighbor residual correlation `0.427027` |
| iid site-level random-intercept probit | `big_spatial_real_validation_codex` | Completed on LUMI; species PPC covered `40 / 40`, site richness PPC covered `400 / 400`, neighbor residual correlation `0.299458` |
| full spatial site-level random-intercept probit | `big_spatial_real_validation_codex` | Completed on LUMI; species PPC covered `40 / 40`, site richness PPC covered `400 / 400`, neighbor residual correlation `-0.291249` |

The run used 2 chains, 1000 saved samples, 500 transient iterations, and thin
10. The full Slurm job completed in 9 minutes 43 seconds on `dev-g` with
TensorFlow 2.16 and an MI250X GPU.

## Main Pending Features

| Feature | Next work |
| --- | --- |
| Longer spatial diagnostics | Run optional longer/4-chain spatial validation with nested `Eta`/`Lambda` diagnostics |
| Random slopes | Harden TensorFlow updater path, then enable strict validation |
| GPP/NNGP spatial effects | Add native compiler and loader support for approximate spatial effects |
| R parity checks | Compare selected Python-native models against equivalent R Hmsc outputs as one-time validation |
