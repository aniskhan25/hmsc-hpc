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

Remaining hardening:

- run occasional longer or 4-chain validation jobs before publication-grade
  inference

## Later milestones
- Trait-related posterior summaries beyond the core `Beta`/`Gamma` samples
- Harden optional Zarr posterior output on large runs
- More robust simulation recovery tests with longer optional `slow` runs

## Implemented After Milestone 4

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

## Recommended Next Implementation Target

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
completed posteriors after a post-processing or stale-code failure. The next
practical target is improving or stress-testing latent slope recovery if stronger
random-slope loading recovery is needed. A larger or replicated NNGP validation
would also help separate expected approximation behavior from small-dataset
instability.

The deterministic simulator for this validation is available as:

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
