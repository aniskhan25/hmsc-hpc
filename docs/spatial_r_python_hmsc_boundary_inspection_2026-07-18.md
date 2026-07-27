# R/Python Spatial HMSC Boundary Inspection

This note records the spatial random-level boundary inspection needed before
extending direct Python-native/R-created HMSC parity to spatial fixtures.

## LUMI Runs

Inspection workflow:

```text
docs/lumi_spatial_boundary_inspection_sbatch.sh
examples/inspect_r_spatial_boundary.py
```

The workflow creates an R/Hmsc `HmscRandomLevel`, exports the original
HMSC-HPC initialization boundary, loads it through the Python RDS importer, and
compares R boundary arrays against Python-native compiled arrays.

Runs:

- `19985338`: failed because the inspection harness looked for `dataParList`
  inside `hM`; the RDS boundary stores it at the top-level import object.
- `19987994`: completed. Full and GPP distance arrays matched exactly; NNGP
  matched neighbor count but R exports ragged neighbor structures while
  Python-native compilation stores padded dense tensors.
- `19995051`: completed after the native compiler was corrected to use the
  R/Hmsc default spatial `alphapw` grid.

## Boundary Semantics Found

R/Hmsc spatial random levels should be constructed from a random-level-indexed
`sData` matrix, ordered by the factor levels in `studyDesign`.

```r
studyDesign <- studyDesign[, c("plot"), drop = FALSE]
studyDesign[["plot"]] <- factor(studyDesign[["plot"]])
sData <- sData[levels(studyDesign[["plot"]]), , drop = FALSE]
```

Spatial method mapping:

```r
HmscRandomLevel(sData = sData, sMethod = "Full")
HmscRandomLevel(sData = sData, sMethod = "GPP", sKnot = as.matrix(sKnot))
HmscRandomLevel(sData = sData, sMethod = "NNGP", nNeighbours = n)
```

`dataParList` contains the spatial arrays consumed by HMSC-HPC:

- Full: `distMat`
- GPP: `nKnots`, `distMat12`, `distMat22`
- NNGP: ragged `indices` and `distList`

The corrected inspection run `19995051` showed exact equality for:

| fixture | arrays | result |
| --- | --- | --- |
| `model_spatial_full.yaml` | `Y`, `X`, `T`, `Pi`, `alphapw`, `distMat` | passed |
| `model_spatial_gpp.yaml` | `Y`, `X`, `T`, `Pi`, `alphapw`, `nKnots`, `distMat12`, `distMat22` | passed |
| `model_spatial_nngp.yaml` | `Y`, `X`, `T`, `Pi`, `alphapw`, `nNeighbours` | passed |

For NNGP, the semantic difference is representation shape: R stores a ragged
per-site neighbor list, while Python-native stores padded arrays
`RandomLevel_*_nngp_indices` and `RandomLevel_*_nngp_distances`.

## Alpha Prior Correction

The completed inspection exposed a sampler-relevant mismatch: Python-native
spatial compilation used a one-point compact alpha support by default, while
R/Hmsc uses a 101-point default grid when `setDefault` spatial priors are in
effect.

R/Hmsc default:

```text
alphaN = 100
alphapw = cbind(enclosingRectDiag * (0:alphaN) / alphaN,
                c(0.5, rep(0.5 / alphaN, alphaN)))
```

Python-native compilation now mirrors this default using the enclosing
coordinate-rectangle diagonal. Explicit `alphapw` is still honored, and the
existing `alpha` shorthand remains available for native-only compact support
experiments.

## Next Step

Direct R/Python spatial parity fixtures were run with the corrected alpha-grid
semantics:

```text
examples/projects/simulated_spatial_validation/model_spatial_full.yaml
examples/projects/simulated_spatial_holdout_validation/model_spatial_gpp.yaml
examples/projects/simulated_spatial_holdout_validation/model_spatial_nngp.yaml
```

The first direct spatial run `19995138` failed after the Full R-boundary sampler
because random-level HDF5 export assumed fixed latent-factor shapes. The retry
`19995352` padded variable random-level factor shapes and passed the Full
fixture, but GPP produced NaN predictions because the legacy RDS import path
lacked the GPP jitter/clipping already used by the native loader.

After stabilizing the RDS GPP import path, LUMI job `19999784` passed the GPP
and NNGP fixtures.

| fixture | LUMI job | parity | boundary | prediction MAE delta | association corr |
| --- | --- | --- | --- | --- | --- |
| Full | `19995352` | passed | exact | `-0.077115` | `0.949683` |
| GPP | `19999784` | passed | exact | `0.002227` | `0.999998` |
| NNGP | `19999784` | passed | exact | `0.001771` | `0.999999` |

The direct parity claim remains split:

- boundary equality for scaled data, `Pi`, and spatial data arrays;
- predictive non-degradation as the acceptance gate;
- posterior correlations reported diagnostically unless fixture mixing is
  strong enough to justify strict posterior gates.
