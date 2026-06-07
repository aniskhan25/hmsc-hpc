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

- add richer user-facing summaries for `Eta` and `Lambda`
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

## Recommended Next Implementation Target

Add random-effect summaries:

```python
fit.eta_summary(level=0)
fit.lambda_summary(level=0)
```

Then validate full spatial random intercepts with the same fixed vs random vs
PPC workflow used for the Whittaker iid validation.
