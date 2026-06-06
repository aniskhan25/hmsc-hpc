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
| Full spatial random intercepts | Supported |
| Random slopes | Compile/load support; sampler guarded |
| GPP/NNGP spatial effects | Not yet |
| HDF5 posterior output | Supported |
| Zarr posterior output | Optional extra |
| Posterior metadata preservation | Supported for native HDF5/Zarr output |
| HDF5 posterior merge | Supported |
| Posterior predictive checks | Supported |
| Covariate gradient summaries | Supported for richness and response-weighted traits |
| LUMI Slurm array workflow | Compile/sample-array/merge templates included |
| Chain status and retry helpers | Supported |
| Slow simulation recovery tests | Supported for fixed effects; smoke tests for traits/random/spatial |
| Legacy TensorFlow updater tests | Current fixtures cover observed-response masks |
| CLI compile/sample/summarize/predict/validate | Supported |
| CLI init validation | Supported |
| No-R example smoke runner | Supported |
| ArviZ diagnostics | Optional extra |
