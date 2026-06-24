# Simulated Poisson Recovery

This project is a deterministic no-R validation dataset for the Python-native
HMSC path. It has 60 sites, 4 species, and 3 covariates with known Poisson
log-link coefficients in `data/truth_beta.csv`.

Expected dominant signs:

| Covariate | Warbler | Owl | Newt | Beetle |
| --- | --- | --- | --- | --- |
| forest_cover | positive | negative | weak positive | weak negative |
| elevation_scaled | negative | positive | negative | weak positive |
| wetland | positive | weak negative | positive | negative |

The dataset is intentionally modest. It is stronger than the tiny smoke
examples, but still small enough for quick LUMI runs.
