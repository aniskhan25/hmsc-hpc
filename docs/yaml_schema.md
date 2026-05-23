# YAML Schema

Minimal:

```yaml
response: data/Y.csv
covariates: data/X.csv
formula:
  X: "~ forest_cover + elevation"
distribution: poisson
chains: 4
```

Optional fields:

```yaml
traits: data/traits.csv
trait_formula: "~ body_size + forest_specialist"
phylo_cov: data/phylo_cov.csv
study_design: data/study_design.csv
random_levels:
  plot:
    column: plot
    type: spatial_full
    coords: [xcoord, ycoord]
```

Supported random-level types are `iid` and `spatial_full`.
