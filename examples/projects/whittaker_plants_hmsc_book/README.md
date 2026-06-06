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
