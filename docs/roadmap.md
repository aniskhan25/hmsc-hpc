# Python API Roadmap

## Milestone 1: Python wrapper over R initialization

Implemented initial package skeleton:

- `pyhmsc.HmscModel`
- R bridge that writes CSV inputs and generates `make_init.R`
- runner for `python -m hmsc.run_gibbs_sampler`
- `pyhmsc.HmscFit` with `beta_mean()`, `beta_ci()`, `summary("Beta")`, and
  fixed-effect Poisson prediction
- example: `examples/simple_birds_r_bridge.py`

This milestone still requires R plus the R packages `Hmsc` and `jsonify`.

## Milestone 2: Native fixed-effect path

- Use `docs/hmsc_hpc_input_schema.md` as the working schema reference.
- Use `pyhmsc compile` / `HmscModel.compile()` to create the Python-native
  `init.json` + `init_arrays.h5` artifact for fixed-effect models.
- `hmsc.run_gibbs_sampler --input run/init.json --output run/posterior.h5`
  now loads the Python-native fixed-effect artifact directly.
- Validate the path with pure-Python schema tests and sampler smoke tests for
  Gaussian, Poisson, and Probit models.

## Milestone 3: Python-native fixed-effect initializer

- Harden fixed-effect Gaussian, Poisson, probit, and Bernoulli models with
  simulation-based validation.
- Compare posterior summaries against known simulated coefficients and posterior
  predictive checks.
- Add JSON/HDF5 posterior storage for larger fixed-effect runs, then Zarr.

## Milestone 4: Native traits, phylogeny, and iid random effects

Implemented:

- species traits with Patsy formula expansion
- `Gamma` posterior summaries and diagnostics
- phylogenetic covariance matrix input
- optional Newick parsing through the `phylo` extra
- iid random intercept compilation, loading, sampling, and posterior export
- iid random-slope compilation, loading, sampling, and posterior export
- full spatial, GPP spatial, and NNGP spatial random-slope compilation,
  loading, sampling, and posterior export; locally smoke-tested
- full spatial, GPP spatial, and NNGP spatial random-intercept compilation, loading,
  sampling, and posterior export
- known random-effect posterior predictive checks
- nested `Eta` and `Lambda` R-hat/ESS diagnostics

Validated real-data target:

- Whittaker plant probit model with TMG, species CN traits, phylogenetic
  covariance, and iid site-level random intercepts
- LUMI run `whittaker_iid_long`: 2 chains, 3000 saved samples, 1000 transient
  iterations, thin 10
- PPC improved from fixed-effect site richness coverage `40 / 52` to iid
  random-effect coverage `52 / 52`
- ecological signal remained stable: richness decreases along TMG and
  community-weighted CN increases along TMG

Validation hardening now available:

- use `examples/plan_long_validation.py` to decide whether a result actually
  needs a longer or 4-chain follow-up before publication-grade inference
- use `docs/lumi_targeted_long_validation_sbatch.sh` for focused LUMI reruns
  of only the flagged diagnostic targets

## Later milestones
- Trait-related posterior summaries beyond the core `Beta`/`Gamma` samples
- More robust simulation recovery tests with longer optional `slow` runs

## Implemented After Milestone 4

The following infrastructure has been added after the initial native
traits/phylogeny/random-effect milestone:

- species association summaries and diagnostics from identifiable
  `Lambda.T @ Lambda` association samples
- `Eta` and `Lambda` posterior summaries, diagnostics, and optional post-hoc
  latent-factor alignment
- one-time optional R model-construction parity checks through
  `examples/run_r_parity_checks.py`; the default report is archived in
  `docs/r_parity_checks_2026-06-23.md`
- targeted long-validation planning through `examples/plan_long_validation.py`
  and LUMI profile script `docs/lumi_targeted_long_validation_sbatch.sh`
- posterior storage inspection through `pyhmsc storage-info`
- nested chain-status validation through `chain-status --expected-draws`
- storage release qualification through
  `examples/run_storage_release_qualification.py`
- optional Zarr release qualification archived in
  `docs/storage_zarr_release_qualification_2026-06-24.md`
- release-polish cleanup through `.gitignore` rules for `.DS_Store`, root
  `/output/`, and root `/run_*/`

Species association summaries are available from sampled random-level `Lambda`.
They can be returned as mean association matrices, credible interval matrices,
pairwise tables with sign probabilities, or diagnostics on identifiable
`Lambda.T @ Lambda` association samples.

Python API:

```python
assoc = fit.species_associations(level=0)
assoc_ci = fit.species_association_ci(level=0)
assoc_table = fit.species_association_summary(level=0)
assoc_diag = fit.diagnostics("Associations", level=0)
```

CLI:

```bash
python -m pyhmsc associations run/posterior.h5 --output run/species_associations.csv
python -m pyhmsc associations run/posterior.h5 --matrix --output run/species_association_matrix.csv
python -m pyhmsc diagnostics run/posterior.h5 --param Associations --output run/association_diagnostics.txt
```

Random-effect posterior summaries are available for `Eta` and `Lambda`. Use
`align=True` or `--align-factors` for post-hoc latent-factor sign/permutation
alignment when inspecting raw latent factors.

```python
fit.eta_summary(level=0)
fit.lambda_summary(level=0)
fit.lambda_summary(level=0, align=True)
```

CLI:

```bash
python -m pyhmsc summarize run/posterior.h5 --param Eta --random-level 0
python -m pyhmsc summarize run/posterior.h5 --param Lambda --random-level 0
python -m pyhmsc summarize run/posterior.h5 --param Lambda --random-level 0 --align-factors
```

## Full Spatial Random-Intercept Validation

The fixed vs iid vs full-spatial validation workflow has been implemented and
run on LUMI without R.

Validated LUMI run `spatial_validation_full_codex`:

- 2 chains, 1000 saved samples, 500 transient iterations, thin 10
- completed in 7 minutes 15 seconds on `dev-g`
- TensorFlow 2.16 with an MI250X GPU
- fixed, iid, and full-spatial models all recovered nonzero beta signs `4 / 4`
- fixed, iid, and full-spatial PPC coverage was `5 / 5` species and `36 / 36`
  site richness
- Eta/truth correlation was `0.675717` for iid and `0.868592` for full spatial

## Real-Data Spatial Validation

The real-data spatial validation project is now available as:

```text
examples/projects/big_spatial_plants_validation/
  model_fixed.yaml
  model_iid.yaml
  model_spatial_full.yaml
  data/
    Y_presence.csv
    X.csv
    study_design.csv
    taxonomy.csv
```

It uses 400 sites and the 40 most prevalent species in that subset from the
existing `examples/big_spatial` plant community data. The corresponding LUMI
script is:

```bash
RUN_NAME=big_spatial_real_validation sbatch docs/lumi_big_spatial_validation_sbatch.sh
```

The analyzer is:

```bash
python examples/analyze_big_spatial_plants.py \
  --fixed-posterior run/fixed/posterior.h5 \
  --iid-posterior run/iid/posterior.h5 \
  --spatial-posterior run/spatial/posterior.h5
```

It reports species and site richness PPC summaries plus nearest-neighbor
residual correlation.

Validated LUMI run `big_spatial_real_validation_codex`:

- 2 chains, 1000 saved samples, 500 transient iterations, thin 10
- completed in 9 minutes 43 seconds on `dev-g`
- TensorFlow 2.16 with an MI250X GPU
- species PPC coverage was `40 / 40` for fixed, iid, and full spatial models
- site richness PPC coverage improved from `309 / 400` fixed to `400 / 400`
  for iid and full spatial models
- nearest-neighbor residual correlation declined from `0.427027` fixed and
  `0.299458` iid to `-0.291249` full spatial

Longer LUMI diagnostic run `big_spatial_long_diag_242e08a`:

- 2 chains, 2000 saved samples, 1000 transient iterations, thin 10
- completed in 17 minutes 7 seconds on `dev-g`
- species PPC coverage stayed at `40 / 40` for all models
- site richness PPC improved from `311 / 400` fixed to `400 / 400` for iid and
  full spatial models
- nearest-neighbor residual correlation declined from `0.427262` fixed and
  `0.292859` iid to `-0.290284` full spatial
- nested `Eta`/`Lambda` diagnostics were emitted successfully, but latent
  random-effect convergence was not yet clean; fixed-effect `Beta` diagnostics
  were clean only for the fixed model

Four-chain spatial-only run `big_spatial_4chain_diag_codex`:

- 4 chains, 2000 saved samples, 1000 transient iterations, thin 10
- completed in 20 minutes 47 seconds on `dev-g`
- raw `Eta`/`Lambda` diagnostics remained poor, consistent with latent-factor
  sign/permutation non-identifiability
- identifiable association diagnostics on `Lambda.T @ Lambda` were much better
  (`max R-hat = 1.0493`, `median R-hat = 1.0080`, `min ESS = 189.8`,
  `median ESS = 700.2`), but still flagged `320 / 780` R-hats and `103 / 780`
  ESS values

Longer four-chain spatial-only association run
`big_spatial_4chain_assoc_long_codex`:

- 4 chains, 2500 saved samples, 1000 transient iterations, thin 10
- completed in 24 minutes 58 seconds on `dev-g`
- identifiable association diagnostics improved to `max R-hat = 1.0247`,
  `median R-hat = 1.0052`, `min ESS = 242.3`, and `median ESS = 782.0`
- association flags decreased to R-hat `153 / 780` and ESS `72 / 780`
- post-hoc aligned latent diagnostics were much better than raw `Eta`/`Lambda`
  diagnostics, but aligned factors still had low ESS

## Completed New-Feature Validation

The current implementation is validated for predictive behavior, iid random
slopes, full spatial random intercepts, and GPP spatial random intercepts.
Association diagnostics are the preferred identifiable target for residual
species association inference.

Validated LUMI run `new_features_validation_fixed2_codex`:

- 2 chains, 1000 saved samples, 500 transient iterations, thin 10
- completed in 8 minutes 50 seconds on `dev-g`
- iid random-slope model recovered beta signs `4 / 4`, species PPC `5 / 5`,
  and site richness PPC `48 / 48`
- full spatial and GPP models both covered species PPC `5 / 5` and site
  richness PPC `36 / 36`
- GPP latent recovery was close to full spatial
  (`Eta/truth = 0.819624`, `Lambda/truth = 0.929552`)
- GPP recovered `3 / 4` nonzero beta signs, so the current validation supports
  runtime compatibility and qualitative behavior, not strict coefficient
  recovery

NNGP spatial random-intercept support is implemented and locally smoke-tested.
LUMI run `new_features_nngp_validation_codex` completed the first deterministic
NNGP validation against full spatial and GPP fits:

- 2 chains, 1000 saved samples, 500 transient iterations, thin 10
- completed in 13 minutes 5 seconds on `dev-g`
- NNGP recovered beta signs `4 / 4`, species PPC `5 / 5`, and site richness
  PPC `36 / 36`
- NNGP latent recovery was weak on the small 36-site dataset
  (`Eta/truth = 0.030639`, `Lambda/truth = 0.228298`)

The deterministic full/GPP/NNGP spatial random-slope validation workflow is now
implemented and completed on LUMI. Run `spatial_random_slope_validation_cli_fixed`
completed in two stages after resuming the GPP/NNGP models from an existing full
spatial posterior:

- full spatial, GPP, and NNGP all recovered beta signs `4 / 4`
- all three covered species PPC `5 / 5` and site richness PPC `49 / 49`
- Eta/truth recovery was `0.831875` full spatial, `0.766687` GPP, and
  `0.107724` NNGP
- Lambda intercept/truth recovery was strong: `0.886723` full spatial,
  `0.921705` GPP, and `0.978135` NNGP
- Lambda slope/truth recovery was weak to moderate: `0.022262` full spatial,
  `0.384912` GPP, and `0.331418` NNGP

The sbatch workflow is resumable with `SKIP_EXISTING=1` and can target selected
models with `MODELS="spatial_gpp spatial_nngp"`, which avoids rerunning
completed posteriors after a post-processing or stale-code failure.

A stronger validation project is now available to test whether weak Lambda slope
recovery is signal-limited rather than a sampler-path issue:

```text
examples/projects/simulated_spatial_random_slope_strong_validation/
  model_spatial_full.yaml
  model_spatial_gpp.yaml
  model_spatial_nngp.yaml
  data/
    Y.csv
    X.csv
    study_design.csv
    truth_beta.csv
    truth_eta.csv
    truth_lambda.csv
```

It uses `n_sites=81`, `n_species=6`, `distr="normal"`,
`lambda_slope_scale=1.8`, and `noise_sd=0.05`. Run it on LUMI with:

```bash
RUN_NAME=spatial_random_slope_strong_validation \
PROJECT_DIR="${PWD}/examples/projects/simulated_spatial_random_slope_strong_validation" \
SAMPLES=2000 TRANSIENT=1000 THIN=10 \
sbatch docs/lumi_spatial_random_slope_validation_sbatch.sh
```

Validated LUMI run `spatial_random_slope_strong_validation_real` completed on
`dev-g` as job `19272750` in 20 minutes 41 seconds with 2 chains, 2000 saved
samples, 1000 transient iterations, and thin 10. Results:

- full spatial, GPP, and NNGP all recovered beta signs `6 / 6`
- all three covered species PPC `6 / 6` and site richness PPC `81 / 81`
- Eta/truth recovery was `0.962924` full spatial, `0.895142` GPP, and
  `0.674563` NNGP
- Lambda intercept/truth recovery was effectively exact: `0.999993` full
  spatial, `0.999142` GPP, and `0.999993` NNGP
- Lambda slope/truth recovery was strong: `0.999992` full spatial, `0.999573`
  GPP, and `0.999988` NNGP

This confirms that the weak Lambda slope recovery in the baseline validation was
signal-limited rather than an obvious sampler-path failure. The remaining
scientific caveat is lower NNGP Eta recovery relative to full spatial and GPP,
which is tracked as an interpretation caveat rather than an active blocker.

The focused spatial Eta validation workflow is implemented as:

```text
examples/projects/simulated_spatial_eta_validation/
  model_spatial_full.yaml
  model_spatial_gpp.yaml
  model_spatial_nngp_5.yaml
  model_spatial_nngp_10.yaml
  model_spatial_nngp_20.yaml
  data/
    Y.csv
    X.csv
    study_design.csv
    truth_beta.csv
    truth_eta.csv
    truth_lambda.csv
```

It uses `n_sites=100`, `n_species=6`, `distr="normal"`,
`spatial_range=0.24`, `spatial_sd=1.6`, `lambda_scale=1.2`, and
`noise_sd=0.06`. The analyzer reports raw and aligned Eta/truth correlation,
scaled Eta RMSE, Lambda/truth correlation, beta sign recovery, and PPC summaries
for full spatial, GPP, and NNGP neighbor counts 5, 10, and 20. Run it on LUMI
with:

```bash
RUN_NAME=spatial_eta_validation \
SAMPLES=1500 TRANSIENT=750 THIN=10 \
sbatch docs/lumi_spatial_eta_validation_sbatch.sh
```

The full five-model Eta validation can exceed one 30-minute `dev-g` allocation.
The sbatch script is resumable: reuse the same `RUN_NAME` with a narrowed
`MODELS` value, such as `MODELS=spatial_nngp_20`, to finish missing models and
then generate the combined report.

Validated LUMI run `spatial_eta_validation_real` completed after one resume. The
first job, `19273473`, timed out at 30 minutes after completing full spatial,
GPP, NNGP-5, and NNGP-10. Resume job `19275812` completed NNGP-20 in 9 minutes
27 seconds and generated the combined report. Results:

- full spatial Eta/truth recovery was strong: `0.988845`
- GPP Eta/truth recovery was good but lower: `0.894420`
- raw NNGP Eta/truth recovery was weak at all tested neighbor counts:
  `0.162178` for 5 neighbors, `0.101626` for 10, and `0.199853` for 20
- post-hoc aligned NNGP Eta/truth recovery was good: `0.926203` for 5
  neighbors, `0.926601` for 10, and `0.935971` for 20
- aligned full spatial and GPP Eta/truth recovery were `0.986014` and
  `0.984491`
- all models covered species PPC `6 / 6` and site richness PPC `100 / 100`
- Lambda/truth recovery was strong for full spatial, GPP, NNGP-5, and NNGP-10;
  NNGP-20 was lower but still high at `0.944280`

This shows the weak raw NNGP Eta metric was mostly a latent-factor sign-switching
summary issue, not a failed NNGP spatial approximation. Direct raw `Eta` means
should be treated as non-identifiable; aligned Eta summaries are the appropriate
diagnostic target for latent recovery.

The NNGP Eta updater also has a multi-factor validation workflow for the
`nf > 1` path:

```text
examples/projects/simulated_spatial_multifactor_eta_validation/
  model_spatial_nngp.yaml
  data/
    Y.csv
    X.csv
    study_design.csv
    truth_beta.csv
    truth_eta.csv
    truth_lambda.csv
```

It uses `n_sites=64`, `n_species=8`, `nf=2`, `distr="normal"`,
`n_neighbors=10`, and known two-factor spatial Eta/Lambda truth. LUMI job
`19276714` completed the NNGP sampler in 6 minutes 8 seconds. The batch step
initially failed during analysis because the adaptive factor sampler expanded to
four factors and the first analyzer assumed exactly two; after fixing the
analyzer to match the best estimated factors to truth, the report was
regenerated from the completed posterior. Results:

- beta signs recovered `8 / 8`
- species PPC covered `8 / 8`; site richness PPC covered `64 / 64`
- raw Eta mean/truth correlation was weak: `0.220537`
- aligned Eta mean/truth correlation was good: `0.856748`
- raw Lambda mean/truth correlation was `0.776261`; aligned was `0.916068`
- identifiable association recovery was strong: `0.981125`

This directly exercises the multi-factor NNGP Eta precision-order fix. The
remaining caveat is that aligned factor summaries are required; raw factor means
remain non-identifiable.

The deterministic simulator for the focused spatial Eta validation is available
as:

```python
from pyhmsc import simulate_spatial_eta_effect_data

Y, X, study_design, truth = simulate_spatial_eta_effect_data(seed=121)
```

The deterministic simulator for the original spatial random-intercept
validation is available as:

```python
from pyhmsc import simulate_spatial_effect_data

Y, X, study_design, truth = simulate_spatial_effect_data(seed=1)
```

It returns one environmental covariate, site coordinates, an iid plot column,
and truth tables for beta coefficients, site effects, species loadings, and the
linear predictor.

The corresponding example project is:

```text
examples/projects/simulated_spatial_validation/
  model_fixed.yaml
  model_iid.yaml
  model_spatial_full.yaml
  data/
    Y.csv
    X.csv
    study_design.csv
    truth_beta.csv
    truth_site_effect.csv
    truth_lambda.csv
```

The comparison analyzer is:

```bash
python examples/analyze_spatial_validation.py \
  --fixed-posterior run/fixed/posterior.h5 \
  --iid-posterior run/iid/posterior.h5 \
  --spatial-posterior run/spatial/posterior.h5
```

It reports beta sign recovery, species and site richness PPC summaries,
nearest-neighbor residual correlation, and Eta-to-truth correlation for random
effect models.

The deterministic simulator for the spatial random-slope validation is:

```python
from pyhmsc import simulate_spatial_random_slope_effect_data

Y, X, study_design, truth = simulate_spatial_random_slope_effect_data(seed=41)
```

The corresponding example project is:

```text
examples/projects/simulated_spatial_random_slope_validation/
  model_spatial_full.yaml
  model_spatial_gpp.yaml
  model_spatial_nngp.yaml
  data/
    Y.csv
    X.csv
    study_design.csv
    truth_beta.csv
    truth_eta.csv
    truth_lambda.csv
```

The LUMI job script is:

```bash
RUN_NAME=spatial_random_slope_validation sbatch docs/lumi_spatial_random_slope_validation_sbatch.sh
```

The analyzer is:

```bash
python examples/analyze_spatial_random_slope_validation.py \
  --spatial-full-posterior run/spatial_full/posterior.h5 \
  --spatial-gpp-posterior run/spatial_gpp/posterior.h5 \
  --spatial-nngp-posterior run/spatial_nngp/posterior.h5
```

## Conditional Held-Out Spatial Prediction

Full spatial random levels support joint conditional sampling of latent `Eta`
at unseen coordinates for every posterior chain, draw, and factor. Prediction
uses the posterior `Alpha` range-grid index and the same exponential covariance
as the sampler. Known groups retain their sampled `Eta` values.

LUMI job `19381199` resumed `spatial_holdout_validation_real`, reused the
existing full-spatial posterior, and completed in 19 seconds. Over 20 held-out
sites and 6 species, conditional prediction achieved correlation `0.927548`,
RMSE `0.504639`, MAE `0.384153`, 95% interval coverage `0.975000`, and mean
interval width `2.397476`. The nearest-unit full-spatial baseline had
correlation `0.818042`, RMSE `0.777394`, coverage `0.250000`, and mean interval
width `0.439665`.

This validation completes the full-spatial conditional prediction milestone.
Conditional GPP prediction uses the modified predictive-process covariance,
including its knot projection and diagonal residual variance. Conditional NNGP
prediction extends the directed neighbor graph in deterministic group order and
samples each held-out `Eta` from its nearest previous training or held-out
neighbors.

LUMI job `19387238` completed both held-out comparisons in 48 seconds. GPP
conditional prediction achieved correlation `0.882362`, RMSE `0.626497`, and
coverage `1.000000`; NNGP conditional prediction achieved correlation
`0.925759`, RMSE `0.510099`, and coverage `1.000000`. This completes conditional
held-out prediction support and deterministic validation for all three spatial
methods.

## Replicated Spatial Hold-Out Validation

A manifest-driven replicated workflow covers three default simulation seeds
and NNGP ordering sensitivity. Canonical ordering fits fixed, full-spatial, GPP,
and NNGP models. Reverse and deterministic random orderings fit additional NNGP
models while preserving observations, covariates, coordinates, truth, row
order, and train/test membership.

The default manifest contains 18 logical tasks. Each GPU array element runs one
seed's six tasks sequentially. Because the project association permits two
submitted `dev-g` jobs, the launcher supports staged `SEED_ARRAY` waves and an
optional dependent CPU analysis job. The analyzer writes raw per-task metrics,
across-seed means and standard deviations, coverage bias and replicate bounds,
and per-seed NNGP deltas from canonical ordering.

LUMI arrays `19388069` and `19388409` completed all tasks, and analysis job
`19388410` completed in 18 seconds. Conditional full-spatial, GPP, and canonical
NNGP coverage averaged `0.947222`, `0.938889`, and `0.944444`, demonstrating
that the original single-run `1.000` coverage was not persistent. Mean RMSE was
`0.624202`, `0.753345`, and `0.634689`.

Reverse and deterministic random NNGP orderings had mean RMSE `0.628314` and
`0.630734`, compared with `0.634689` canonical. The largest absolute per-seed
ordering delta was `0.019124` RMSE and `0.016154` correlation. This completes
the replicated calibration and ordering-sensitivity milestone.

## Real-Data Spatial Hold-Out and Resource Benchmark

The real-data spatial hold-out benchmark is implemented for the existing
400-site, 40-species big-spatial plant project. A deterministic spatial block
split uses 319 sites for fitting and 81 sites for evaluation. Fixed,
full-spatial, GPP, and NNGP probit models share the same observations,
covariates, and split. Conditional spatial prediction is used at every held-out
coordinate.

The workflow consists of:

```text
examples/generate_big_spatial_holdout_validation.py
examples/analyze_big_spatial_holdout_validation.py
docs/lumi_big_spatial_holdout_validation_sbatch.sh
```

In addition to Brier score, log loss, macro AUC, prevalence MAE, and richness
MAE, the LUMI workflow records elapsed sampler time, peak RSS, compiled model
size, and posterior size for each model. The completed LUMI run provides the
first real-data performance and resource comparison for all four methods.

LUMI job `19435459` completed the benchmark in 14 minutes 32 seconds using two
chains, 250 saved draws, 250 transient iterations, and thin 5. GPP had the
lowest Brier score (`0.069072`), lowest log loss (`0.243408`), and highest macro
AUC (`0.732161`). Fixed, full-spatial, and NNGP Brier scores were `0.070486`,
`0.072041`, and `0.074632`. These differences are modest and should be treated
as provisional out-of-sample evidence.

Sampler-only runtime was 5.1 seconds fixed, 31.6 seconds full spatial, 15.7
seconds GPP, and 613.1 seconds NNGP. Peak process RSS was between 1.9 and 2.3
GB. The benchmark therefore identifies NNGP Eta updating as a major scaling
bottleneck in the current upstream sampler implementation. This should be
reported upstream rather than optimized opportunistically in the Python-native
wrapper.

Short-run Beta convergence was not publication-grade: spatial median ESS was
77-83, 88-111 of 200 coefficients exceeded R-hat 1.01, and 174-184 were below
ESS 200. The run validates finite end-to-end held-out prediction and provides a
resource profile, but does not establish stable parameter inference.

The follow-up trait/phylogeny held-out validation is complete and documented
below. The combined trait/phylogeny plus random-level sampler path remains an
upstream compatibility issue.

## Trait and Phylogeny Hold-Out Validation

The Whittaker plant hold-out workflow is implemented with 40 training sites and
12 deterministic TMG-stratified test sites. Site selection preserves at least
one training occurrence for every one of the 75 species. The fixed probit model
retains the CN trait formula and phylogenetic covariance matrix. The iid
site-level probit model is environment-only so this validation does not depend
on the currently failing traits + latent random-effect path in the original
`hmsc` updater code.

The Python prediction layer now samples a new standard-normal iid latent effect
for each unseen group and posterior draw during marginal prediction. Repeated
rows belonging to the same new group share the same draw, and `rng_seed`
provides reproducibility. This replaces the previous approximation based on
the mean fitted Eta across training groups.

Python-native validation now guards trait/phylogeny-structured models with
random levels as not sampler-ready. The guard triggers from `validate-init
--strict`, `pyhmsc sample`, and direct `HmscModel.sample(init="python-native")`
before TensorFlow starts. Remove it only after the upstream `hmsc`
`updateBetaLambda` path supports that combined model.

The validation workflow consists of:

```text
examples/generate_whittaker_holdout_validation.py
examples/analyze_whittaker_holdout_validation.py
docs/lumi_whittaker_holdout_validation_sbatch.sh
```

LUMI job `19468166` completed `whittaker_holdout_validation_real_v2` in
`00:04:07` on `dev-g`. The fixed trait/phylogeny model produced Brier score
`0.0742`, log loss `0.2648`, macro AUC `0.5518`, and positive `TMG x CN` Gamma
mean `0.182` with 95% CI `0.047-0.318`. The environment-only iid marginal model
produced Brier score `0.0734`, log loss `0.2607`, and macro AUC `0.5495`. Both
models recovered the expected negative held-out richness slope and positive
community-weighted CN slope.

## Current Remaining Roadmap

The Python-only path now covers fixed effects, traits, phylogeny, iid random
effects, spatial full/GPP/NNGP effects, random slopes, held-out prediction,
diagnostics, LUMI workflows, recovery/retry helpers, archived optional R parity
evidence, targeted long-validation planning, and posterior storage inspection.
Remaining work is mostly external upstream coordination and optional
publication/release follow-up:

- submit the prepared upstream reports in `docs/upstream_issue_reports.md` once
  GitHub authentication is available: `updateBetaLambda` trait/phylogeny plus
  random-level failure, NNGP Eta runtime bottleneck, and multi-factor NNGP Eta
  ordering fix
- remove the Python-native sampler-readiness guard for trait/phylogeny plus
  random levels only after upstream sampler support exists
- run targeted longer/4-chain validation only when
  `examples/plan_long_validation.py` flags diagnostics that matter for the
  intended inference target
- rerun storage release qualification only when storage dependencies or output
  formats change; HDF5 and optional Zarr results are archived in
  `docs/storage_release_qualification_2026-06-24.md` and
  `docs/storage_zarr_release_qualification_2026-06-24.md`
