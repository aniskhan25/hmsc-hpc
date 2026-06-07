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
| LUMI Slurm array workflow | Compile/sample-array/merge templates included |
| Chain status and retry helpers | Supported |
| Slow simulation recovery tests | Supported for fixed effects; smoke tests for traits/random/spatial |
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

## Main Pending Features

| Feature | Next work |
| --- | --- |
| Species association summaries | Derive association/correlation summaries from sampled `Lambda` |
| Random-effect posterior summaries | Add user-facing `Eta`/`Lambda` summary tables and diagnostics |
| Spatial validation | Run real or simulation validation for full spatial random intercepts |
| Random slopes | Harden TensorFlow updater path, then enable strict validation |
| GPP/NNGP spatial effects | Add native compiler and loader support for approximate spatial effects |
| R parity checks | Compare selected Python-native models against equivalent R Hmsc outputs as one-time validation |
