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
# or
phylo_tree: data/tree.nwk
study_design: data/study_design.csv
random_levels:
  plot:
    column: plot
    type: spatial_full
    coords: [xcoord, ycoord]
  site:
    column: site
    type: iid
    x_formula: "~ elevation"
```

Supported random-level types are `iid` and `spatial_full`.
Random-slope `x_formula` is compiled and loaded but native TensorFlow sampling is
currently guarded until the random-slope updater path is hardened.

Before sampling a compiled Python-native model, run:

```bash
python -m pyhmsc validate-init run/init.json --strict
```

This validates the no-R compiled model boundary and reports whether the artifact
uses only features supported by the existing Python sampler.
