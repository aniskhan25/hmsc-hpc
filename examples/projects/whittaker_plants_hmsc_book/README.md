# Whittaker Plants HMSC Book Example

This is a real-data validation project derived from the official HMSC book
supporting files for Section 6.7, the plant example from:

Ovaskainen, O. and Abrego, N. 2020. *Joint Species Distribution Modelling -
With Applications in R*. Cambridge University Press.

The official HMSC page provides the source archive:

https://www.helsinki.fi/en/researchgroups/statistical-ecology/software/hmsc

The original dataset is from:

Miller JED, Damschen EI, Ives AR (2018) Functional traits and community
composition: a comparison among community-weighted means, weighted
correlations, and multilevel models. Methods in Ecology and Evolution.
https://doi.org/10.1111/2041-210X.13119

Dryad data package:

Miller JED, Damschen EI, Ives AR (2018) Data from: Functional traits and
community composition: a comparison among community-weighted means, weighted
correlations, and multilevel models. Dryad Digital Repository.
https://doi.org/10.5061/dryad.7gj0s3b

## Derived Files

The official long-format file `whittaker revisit data.csv` was transformed into:

- `data/Y_presence.csv`: 52 sites x 75 plant species, binary occurrence matrix.
- `data/Y_abundance.csv`: 52 sites x 75 plant species, abundance matrix.
- `data/X.csv`: site-level topographic moisture gradient, named `TMG`.
- `data/traits.csv`: species-level leaf carbon-to-nitrogen ratio, named `CN`.
- `data/taxonomy.csv`: species taxonomy from the official archive.
- `data/phylo_cov.csv`: taxonomy-derived covariance proxy.

The first Python-native model uses the presence-absence/probit formulation:

```bash
python -m pyhmsc compile examples/projects/whittaker_plants_hmsc_book/model.yaml --output run_whittaker
python -m pyhmsc validate-init run_whittaker/init.json --strict
python -m pyhmsc sample run_whittaker/init.json --output run_whittaker/posterior.h5 --samples 1000 --transient 500 --thin 10
```

After sampling, generate the book-comparison report:

```bash
python examples/analyze_whittaker_plants.py \
  --posterior run_whittaker/posterior.h5 \
  --project examples/projects/whittaker_plants_hmsc_book \
  --output run_whittaker/whittaker_report.txt
```

## Literature Checks

The original HMSC Section 6.7 scripts state these expected qualitative results:

- Many species respond negatively to `TMG`.
- A typical species responds negatively to `TMG`.
- Species with higher `CN` respond less negatively, or positively, to `TMG`.
- Species richness decreases along the `TMG` gradient.
- Community-weighted mean `CN` increases along the `TMG` gradient.

Those are the comparison targets for confirming the Python-only HMSC path against
published/example HMSC analyses.

## Held-Out-Site Validation

Generate a deterministic 40-site training and 12-site test project that keeps
every species represented in training while spanning the full TMG range:

```bash
python examples/generate_whittaker_holdout_validation.py
```

The hold-out workflow compares a fixed trait/phylogeny model with an
environment-only iid site-level model. The iid model uses marginal prediction
for genuinely unseen site effects. The traits + latent random-effect sampler
combination is intentionally not exercised here because that path currently
fails inside the original `hmsc` updater code:

```bash
RUN_NAME=whittaker_holdout_validation_real \
  sbatch docs/lumi_whittaker_holdout_validation_sbatch.sh
```

The report includes Brier score, log loss, macro AUC, occupancy and richness
errors, observed and predicted TMG richness/CN slopes, the fixed-model
posterior `TMG x CN` Gamma effect, Beta diagnostics, runtime, and memory.

Validated LUMI run `whittaker_holdout_validation_real_v2` completed as job
`19468166` in `00:04:07`. The fixed trait/phylogeny model had Brier score
`0.0742`, log loss `0.2648`, macro AUC `0.5518`, and positive `TMG x CN` Gamma
mean `0.182` with 95% CI `0.047-0.318`. The environment-only iid marginal model
had Brier score `0.0734`, log loss `0.2607`, and macro AUC `0.5495`.
