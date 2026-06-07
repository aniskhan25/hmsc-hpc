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
- known random-effect posterior predictive checks

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

- add diagnostics helpers for nested `Eta` and `Lambda` arrays
- run occasional longer or 4-chain validation jobs before publication-grade
  inference

## Later milestones

- GPP and NNGP spatial random levels
- Random slopes
- Trait-related posterior summaries beyond the core `Beta`/`Gamma` samples
- Harden optional Zarr posterior output on large runs
- More robust simulation recovery tests with longer optional `slow` runs

## Implemented After Milestone 4

Species association summaries are available from sampled random-level `Lambda`.
They can be returned as mean association matrices, credible interval matrices,
or pairwise tables with sign probabilities.

Python API:

```python
assoc = fit.species_associations(level=0)
assoc_ci = fit.species_association_ci(level=0)
assoc_table = fit.species_association_summary(level=0)
```

CLI:

```bash
python -m pyhmsc associations run/posterior.h5 --output run/species_associations.csv
python -m pyhmsc associations run/posterior.h5 --matrix --output run/species_association_matrix.csv
```

Random-effect posterior summaries are available for `Eta` and `Lambda`.

```python
fit.eta_summary(level=0)
fit.lambda_summary(level=0)
```

CLI:

```bash
python -m pyhmsc summarize run/posterior.h5 --param Eta --random-level 0
python -m pyhmsc summarize run/posterior.h5 --param Lambda --random-level 0
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

## Recommended Next Implementation Target

Add nested random-effect diagnostics for `Eta` and `Lambda`, then run optional
longer or 4-chain spatial validation jobs before publication-grade inference.

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
