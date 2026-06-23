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
| iid random-intercept real-data validation | Supported for environment-only random levels; Whittaker plant held-out iid marginal run validated on LUMI |
| Full spatial random intercepts | Supported |
| iid random slopes | Supported for native TensorFlow sampling |
| Full spatial random slopes | Supported for native TensorFlow sampling; deterministic and strong-signal LUMI validation completed |
| GPP spatial random slopes | Supported for native TensorFlow sampling; deterministic and strong-signal LUMI validation completed |
| NNGP spatial random slopes | Supported for native TensorFlow sampling; deterministic and strong-signal LUMI validation completed; Eta recovery remains weaker than full/GPP |
| GPP spatial effects | Supported for native TensorFlow sampling with deterministic knot selection |
| NNGP spatial effects | Supported for native TensorFlow sampling; LUMI runtime/PPC validation completed, aligned latent Eta recovery validated |
| HDF5 posterior output | Supported |
| Zarr posterior output | Optional extra |
| Posterior metadata preservation | Supported for native HDF5/Zarr output |
| HDF5 posterior merge | Supported |
| Posterior storage inspection | Supported for HDF5 and optional Zarr via `pyhmsc storage-info`; reports dataset shapes, chunks, byte size, chain/draw counts, and metadata presence |
| Posterior predictive checks | Supported |
| Known random-effect posterior predictive checks | Supported when the fit is loaded with its model config; prediction accepts separate `study_design` and `coords` frames |
| Random-effect/spatial-aware prediction | Supported for known iid/spatial groups, random slopes, nearest-neighbor projection, and conditional full-spatial/GPP/NNGP Eta sampling at unseen coordinates |
| Spatial held-out prediction simulator | Supported with deterministic 80/20 train/test split and distinct test coordinates |
| Spatial held-out prediction validation | Completed on LUMI for full, GPP, and NNGP conditional prediction; correlations 0.928/0.882/0.926, RMSE 0.505/0.626/0.510, and coverage 0.975/1.000/1.000 |
| Replicated spatial hold-out validation | Completed across three LUMI seeds; full/GPP/NNGP conditional coverage 0.947/0.939/0.944 and RMSE 0.624/0.753/0.635 |
| NNGP ordering sensitivity | Completed for canonical, reverse, and deterministic random orderings; largest absolute per-seed RMSE delta was 0.0191 |
| Real-data spatial hold-out benchmark | Completed on LUMI for the big-spatial plant data with a deterministic blocked 319/81 train/test split; GPP had the best Brier score (0.0691) and macro AUC (0.7322), but differences were small and the short-run Beta diagnostics were not publication-grade |
| Spatial runtime and memory benchmark | Completed on LUMI; sampler-only times were 5s fixed, 32s full, 16s GPP, and 613s NNGP, with peak process RSS of 1.9-2.3 GB |
| Site richness posterior predictive checks | Supported |
| Covariate gradient summaries | Supported for richness and response-weighted traits |
| Trait-effect posterior summaries | Supported for Gamma mean/interval tables |
| Trait/phylogeny-aware real-data hold-out | Completed on LUMI for the Whittaker plant data with 40 training and 12 TMG-stratified held-out sites; fixed trait/phylogeny Brier 0.0742, iid marginal Brier 0.0734, fixed `TMG x CN` Gamma mean 0.182 with 95% CI 0.047-0.318 |
| Trait/phylogeny plus random levels | Guarded as not sampler-ready in Python-native validation until the upstream `hmsc` updater path is fixed |
| Species association summaries | Supported from random-level `Lambda` samples |
| Species association diagnostics | Supported for identifiable `Lambda.T @ Lambda` association samples |
| Random-effect posterior summaries | Supported for `Eta` and random-intercept `Lambda`; random-slope `Lambda` requires `x_index` |
| Random-effect diagnostics | Supported for nested `Eta`/`Lambda` arrays |
| Latent-factor alignment | Optional post-hoc alignment for raw `Eta`/`Lambda` summaries and diagnostics |
| LUMI Slurm array workflow | Compile/sample-array/merge templates included |
| Chain status and retry helpers | Supported |
| Slow simulation recovery tests | Supported for fixed effects; smoke tests for traits/random/spatial |
| Deterministic spatial validation simulator | Supported via `simulate_spatial_effect_data` |
| Deterministic spatial Eta validation simulator | Supported via `simulate_spatial_eta_effect_data` |
| Simulated spatial validation project | Added with fixed, iid, and full-spatial configs |
| Simulated spatial validation analyzer | Supported; fixed/iid/spatial LUMI validation completed |
| Simulated spatial Eta validation project | Added with full/GPP/NNGP neighbor-count configs; LUMI validation completed after resume |
| Simulated spatial Eta validation analyzer | Supported; reports Eta recovery versus NNGP neighbor count |
| Simulated multi-factor NNGP Eta validation project | Added with `nf=2`; LUMI sampler validation completed |
| Simulated multi-factor NNGP Eta validation analyzer | Supported; matches adaptive extra factors to known truth |
| Simulated random-slope/GPP validation project | Supported; fixed/random-slope/full-spatial/GPP LUMI validation completed |
| Big spatial real-data validation project | Added with fixed, iid, and full-spatial configs |
| Big spatial real-data analyzer | Supported; fixed/iid/spatial LUMI validation completed |
| Legacy TensorFlow updater tests | Current fixtures cover observed-response masks |
| CLI compile/sample/summarize/predict/validate | Supported; `predict` accepts `--model-config`, `--study-design`, `--coords`, and random-effect options; LUMI end-to-end prediction validation completed |
| CLI init validation | Supported |
| Optional R parity checks | Supported via `examples/run_r_parity_checks.py` for fixed, trait/phylogeny, and environment-only iid random-intercept model-construction parity |
| Targeted longer validation planner | Supported via `examples/plan_long_validation.py`; recommends longer/4-chain follow-up only for diagnostics that fail configured R-hat/ESS thresholds |
| Targeted LUMI long-validation workflow | Supported via `docs/lumi_targeted_long_validation_sbatch.sh` with `associations`, `beta`, `latent`, and `all` profiles |
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

The strong-signal spatial random-slope validation project in
`examples/projects/simulated_spatial_random_slope_strong_validation` has also
been run on LUMI without R.

| Model | Run | Result |
| --- | --- | --- |
| full spatial random-slope normal | `spatial_random_slope_strong_validation_real` | Completed on LUMI; beta signs recovered `6 / 6`, species PPC covered `6 / 6`, site richness PPC covered `81 / 81`, Eta/truth correlation `0.962924`, Lambda slope/truth correlation `0.999992` |
| GPP spatial random-slope normal | `spatial_random_slope_strong_validation_real` | Completed on LUMI; beta signs recovered `6 / 6`, species PPC covered `6 / 6`, site richness PPC covered `81 / 81`, Eta/truth correlation `0.895142`, Lambda slope/truth correlation `0.999573` |
| NNGP spatial random-slope normal | `spatial_random_slope_strong_validation_real` | Completed on LUMI; beta signs recovered `6 / 6`, species PPC covered `6 / 6`, site richness PPC covered `81 / 81`, Eta/truth correlation `0.674563`, Lambda slope/truth correlation `0.999988` |

The run used 2 chains, 2000 saved samples, 1000 transient iterations, and thin
10. The Slurm job completed in 20 minutes 41 seconds on `dev-g` with TensorFlow
2.16 and an MI250X GPU. This confirms spatial random-slope Lambda recovery under
strong signal for full, GPP, and NNGP samplers; raw NNGP Eta summaries can be
sensitive to latent-factor sign switching and should be interpreted with
alignment.

Focused spatial Eta validation run `spatial_eta_validation_real` completed on
LUMI after one resume. The first `dev-g` job timed out after full spatial, GPP,
NNGP-5, and NNGP-10; the resume job completed NNGP-20 and generated the report.

| Model | Run | Result |
| --- | --- | --- |
| full spatial random-intercept normal | `spatial_eta_validation_real` | Completed on LUMI; beta signs recovered `6 / 6`, species PPC covered `6 / 6`, site richness PPC covered `100 / 100`, raw/aligned Eta truth correlation `0.988845` / `0.986014`, Lambda/truth correlation `0.999982` |
| GPP spatial random-intercept normal | `spatial_eta_validation_real` | Completed on LUMI; beta signs recovered `6 / 6`, species PPC covered `6 / 6`, site richness PPC covered `100 / 100`, raw/aligned Eta truth correlation `0.894420` / `0.984491`, Lambda/truth correlation `0.999186` |
| NNGP-5 spatial random-intercept normal | `spatial_eta_validation_real` | Completed on LUMI; beta signs recovered `5 / 6`, species PPC covered `6 / 6`, site richness PPC covered `100 / 100`, raw/aligned Eta truth correlation `0.162178` / `0.926203`, Lambda/truth correlation `0.998229` |
| NNGP-10 spatial random-intercept normal | `spatial_eta_validation_real` | Completed on LUMI; beta signs recovered `5 / 6`, species PPC covered `6 / 6`, site richness PPC covered `100 / 100`, raw/aligned Eta truth correlation `0.101626` / `0.926601`, Lambda/truth correlation `0.999068` |
| NNGP-20 spatial random-intercept normal | `spatial_eta_validation_real` | Completed on LUMI; beta signs recovered `5 / 6`, species PPC covered `6 / 6`, site richness PPC covered `100 / 100`, raw/aligned Eta truth correlation `0.199853` / `0.935971`, Lambda/truth correlation `0.944280` |

This run shows that raw NNGP Eta means were dominated by latent-factor sign
switching, while aligned Eta summaries recovered the simulated latent spatial
signal well. Use aligned Eta summaries for latent recovery diagnostics.

Multi-factor NNGP Eta validation run `spatial_multifactor_eta_validation_real`
exercised the `nf > 1` NNGP Eta updater path on LUMI. Job `19276714` completed
sampling in 6 minutes 8 seconds; after fixing the analyzer to account for
adaptive extra factors, the report was regenerated from the completed posterior.

| Model | Run | Result |
| --- | --- | --- |
| NNGP spatial random-intercept normal, `nf=2` | `spatial_multifactor_eta_validation_real` | Completed on LUMI; beta signs recovered `8 / 8`, species PPC covered `8 / 8`, site richness PPC covered `64 / 64`, raw/aligned Eta mean truth correlation `0.220537` / `0.856748`, raw/aligned Lambda mean truth correlation `0.776261` / `0.916068`, association truth correlation `0.981125` |

This validates the multi-factor NNGP path after the Eta prior precision ordering
fix. Interpret raw factor means cautiously; aligned factor summaries and
association summaries are the useful diagnostics.

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

Longer diagnostic run `big_spatial_long_diag_242e08a` completed on LUMI with
2 chains, 2000 saved samples, 1000 transient iterations, and thin 10. It
preserved the predictive pattern: species PPC covered `40 / 40` for all
models, site richness PPC improved from `311 / 400` fixed to `400 / 400` for
iid and spatial random effects, and nearest-neighbor residual correlation
declined from `0.427262` fixed and `0.292859` iid to `-0.290284` full spatial.
The new nested diagnostics were emitted successfully. Fixed-effect `Beta`
diagnostics were clean (`max R-hat = 1.0035`, `min ESS = 560.6`), but latent
`Eta`/`Lambda` diagnostics were not yet clean for iid or full-spatial models.
Use latent random-effect summaries as exploratory until longer or 4-chain runs
improve those diagnostics.

Four-chain spatial-only run `big_spatial_4chain_diag_codex` completed on LUMI
in 20 minutes 47 seconds. Raw `Eta`/`Lambda` diagnostics remained poor, but
identifiable association diagnostics on `Lambda.T @ Lambda` improved
substantially (`max R-hat = 1.0493`, `median R-hat = 1.0080`, `min ESS =
189.8`, `median ESS = 700.2`). Association diagnostics are now the preferred
diagnostic target for residual species association inference, although this run
still had R-hat flags (`320 / 780`) and should be extended before
publication-grade association estimates.

Longer four-chain spatial-only association run
`big_spatial_4chain_assoc_long_codex` completed on LUMI in 24 minutes 58
seconds with 2500 saved samples, 1000 transient iterations, and thin 10.
Identifiable association diagnostics improved again (`max R-hat = 1.0247`,
`median R-hat = 1.0052`, `min ESS = 242.3`, `median ESS = 782.0`; R-hat flags
`153 / 780`, ESS flags `72 / 780`). Post-hoc latent-factor alignment reduced
raw `Eta`/`Lambda` R-hat pathologies, but aligned latent factors still had low
ESS, so `Associations` remains the preferred diagnostic target for residual
species association inference.

The deterministic random-slope/GPP validation project in
`examples/projects/simulated_new_features_validation` has been run on LUMI
without R.

| Model | Run | Result |
| --- | --- | --- |
| Fixed-effect probit | `new_features_validation_fixed2_codex` | Completed on LUMI; beta signs recovered `4 / 4`, species PPC covered `5 / 5`, site richness PPC covered `46 / 48` |
| iid random-slope probit | `new_features_validation_fixed2_codex` | Completed on LUMI; beta signs recovered `4 / 4`, species PPC covered `5 / 5`, site richness PPC covered `48 / 48`, Eta/truth correlation `0.434882`, slope Lambda/truth correlation `0.561890` |
| full spatial random-intercept probit | `new_features_validation_fixed2_codex` | Completed on LUMI; beta signs recovered `4 / 4`, species PPC covered `5 / 5`, site richness PPC covered `36 / 36`, Eta/truth correlation `0.827719`, Lambda/truth correlation `0.921528` |
| GPP spatial random-intercept probit | `new_features_validation_fixed2_codex` | Completed on LUMI; beta signs recovered `3 / 4`, species PPC covered `5 / 5`, site richness PPC covered `36 / 36`, Eta/truth correlation `0.819624`, Lambda/truth correlation `0.929552` |
| NNGP spatial random-intercept probit | `new_features_nngp_validation_codex` | Completed on LUMI; beta signs recovered `4 / 4`, species PPC covered `5 / 5`, site richness PPC covered `36 / 36`, Eta/truth correlation `0.030639`, Lambda/truth correlation `0.228298` |

The run used 2 chains, 1000 saved samples, 500 transient iterations, and thin
10. It completed in 8 minutes 50 seconds on `dev-g` with TensorFlow 2.16 and
an MI250X GPU. The GPP result closely matched full-spatial latent recovery but
missed one coefficient sign, so this validation supports runtime compatibility
and qualitative behavior rather than strict coefficient recovery.

Follow-up run `new_features_nngp_validation_codex` added NNGP to the same
workflow. It completed in 13 minutes 5 seconds on `dev-g` and validated NNGP
runtime compatibility and PPC behavior, but latent recovery was weak on the
small 36-site dataset. Later replicated hold-out and aligned Eta validations
provide the stronger evidence for NNGP interpretation; raw NNGP factor means
remain provisional because of latent-factor non-identifiability.

## Remaining Work

| Area | Next work |
| --- | --- |
| Upstream sampler compatibility | Report or fix the original `hmsc` `updateBetaLambda` path for trait/phylogeny-structured models with random levels, then remove the Python-native validation guard |
| R parity checks | Run `examples/run_r_parity_checks.py` on an R-enabled machine and archive the report as one-time validation evidence |
| Publication-grade association inference | Use `examples/plan_long_validation.py` first; run `docs/lumi_targeted_long_validation_sbatch.sh` only for flagged association diagnostics |
| NNGP performance | Report the observed NNGP Eta-update runtime bottleneck upstream; avoid wrapper-side optimization until the sampler issue is understood |
| Storage release qualification | Use implemented `pyhmsc storage-info` and nested `chain-status --expected-draws`; run larger Zarr/merge stress tests only before a release |
