# Hmsc-HPC Input Schema

## Compatibility Format: JSON Inside RDS

Hmsc-HPC currently reads an `.rds` file containing a single JSON string. The
JSON is produced in R with:

```r
init_obj <- sampleMcmc(m, engine = "HPC", ...)
saveRDS(jsonify::to_json(init_obj), file = "init_file.rds")
```

This path is compatibility-only. Python-native workflows use JSON+HDF5 and do
not require R or `pyreadr`; install the `rds` extra only if old RDS files must be
read or written.

The Python loader is `hmsc.run_gibbs_sampler.load_params()`. It reads the JSON
with `hmsc.utils.export_rds_utils.load_model_from_rds()` and then consumes these
top-level keys:

| Key | Shape / type | Meaning | Required? | Loader |
| --- | --- | --- | --- | --- |
| `hM` | object | Serialized Hmsc model definition and priors | yes | `load_model_dims`, `load_model_data`, `load_prior_hyperparams`, `load_random_level_hyperparams` |
| `initParList` | list length `nChains` | Chain-specific initial parameter values | yes | `init_params` |
| `dataParList` | object | Precomputed random-level data, especially spatial matrices | yes for random levels; may be mostly unused for fixed effects | `load_random_level_hyperparams` |
| `nChains` | length-1 numeric vector | Total initialized chains | yes | `load_params` |

## `hM` Fields

| Key | Shape / type | Meaning | Required? |
| --- | --- | --- | --- |
| `YScaled` | `ny x ns` numeric matrix | Response matrix used by sampler; missing observations are `NaN` | yes |
| `XScaled` | `ny x nc` matrix or species-indexed object of matrices | Environmental design matrix | yes |
| `TrScaled` | `ns x nt` numeric matrix | Species trait design matrix | yes, even if only intercept traits are used by R |
| `C` | `ns x ns` matrix or empty/null | Phylogenetic covariance matrix | no |
| `Pi` | integer matrix/vector, R 1-based | Trait prior structure/indexing; converted to Python 0-based | yes |
| `distr` | length `ns` integer vector | Species distributions as R/Hmsc numeric codes | yes |
| `ny` | length-1 integer | Number of sampling units/sites | yes |
| `ns` | length-1 integer | Number of species/responses | yes |
| `nc` | length-1 integer | Number of fixed-effect covariates | yes |
| `nt` | length-1 integer | Number of trait covariates | yes |
| `nr` | length-1 integer | Number of random levels | yes |
| `np` | integer vector length `nr` | Number of units per random level | yes when `nr > 0`; empty otherwise |
| `ncsel` | length-1 integer | Number of variable-selection groups | yes |
| `ncRRR` | length-1 integer | Number of reduced-rank regression covariates | yes |
| `ncNRRR` | length-1 integer | Number of non-ordinated RRR covariates | yes |
| `ncORRR` | length-1 integer | Number of ordinated RRR covariates | yes |
| `nuRRR` | length-1 numeric | RRR shrinkage prior parameter | yes |
| `XSelect` | list length `ncsel` | Variable-selection group metadata | required when `ncsel > 0` |
| `XRRRScaled` | `ny x ncRRR` matrix or empty | RRR design matrix | no; loader substitutes `ny x 0` |
| `mGamma` | `nt x nc` numeric matrix | Prior mean for `Gamma` | yes |
| `UGamma` | `nt x nt` numeric matrix | Prior covariance for `Gamma`; inverted by loader | yes |
| `f0` | scalar/vector | Inverse-Wishart prior degrees parameter | yes |
| `V0` | matrix/scalar | Inverse-Wishart prior scale parameter | yes |
| `rhopw` | numeric array | Phylogeny/correlation grid prior values | yes |
| `aSigma` | numeric vector | Residual variance prior shape | yes |
| `bSigma` | numeric vector | Residual variance prior rate/scale | yes |
| `a1RRR`, `b1RRR`, `a2RRR`, `b2RRR` | scalars | RRR shrinkage hyperparameters | yes |
| `rL` | named object length `nr` | Random-level definitions | required when `nr > 0` |

## `hM$XSelect` Fields

| Key | Shape / type | Meaning | Required? |
| --- | --- | --- | --- |
| `covGroup` | integer vector, R 1-based | Fixed covariate indices controlled by selection group | yes |
| `spGroup` | integer vector, R 1-based | Species group indices for selection state lookup | yes |
| `q` | numeric vector/matrix | Variable-selection prior probabilities | yes |

## `hM$rL` Random-Level Fields

For each random level name in `hM$rL`:

| Key | Shape / type | Meaning | Required? |
| --- | --- | --- | --- |
| `nu`, `a1`, `b1`, `a2`, `b2` | length-1 numeric vectors | Latent-factor shrinkage hyperparameters | yes |
| `nfMin`, `nfMax` | length-1 integer vectors | Minimum and maximum number of latent factors | yes |
| `sDim` | integer or `"Inf"` | Spatial dimension indicator | yes |
| `xDim` | integer | Random slope covariate dimension | yes |
| `spatialMethod` | string | `Full`, `GPP`, or `NNGP` | required when `sDim > 0` |
| `alphapw` | matrix | Spatial range prior grid | required when `sDim > 0` |
| `xMat` | matrix | Random-level covariate matrix | required when `xDim > 0` |

## `dataParList` Fields

| Key | Shape / type | Meaning | Required? |
| --- | --- | --- | --- |
| `rLPar` | list length `nr` | Per-random-level precomputed spatial data | required for spatial random levels |
| `rLPar[[r]]$distMat` | `np[r] x np[r]` flattened matrix | Full spatial distance matrix | required for `spatialMethod == "Full"` |
| `rLPar[[r]]$nKnots` | length-1 integer | GPP knot count | required for `GPP` |
| `rLPar[[r]]$distMat12` | `np[r] x nKnots` flattened matrix | GPP site-to-knot distances | required for `GPP` |
| `rLPar[[r]]$distMat22` | `nKnots x nKnots` flattened matrix | GPP knot-to-knot distances | required for `GPP` |
| `rLPar[[r]]$indices` | list | NNGP neighbor indices, R 1-based | required for `NNGP` |
| `rLPar[[r]]$distList` | list | NNGP neighbor distances | required for `NNGP` |
| `Qg`, `iQg`, `RQg` | matrices or grids | Phylogenetic precomputations | currently not used by `run_gibbs_sampler` |

## `initParList` Chain Fields

Each `initParList[[chain]]` entry must contain:

| Key | Shape / type | Meaning | Required? |
| --- | --- | --- | --- |
| `Beta` | `nc x ns` matrix | Initial fixed-effect coefficients | yes |
| `Gamma` | `nt x nc` matrix | Initial trait-response coefficients | yes |
| `V` | `nc x nc` matrix | Initial covariance for `Beta`; inverted to `iV` | yes |
| `rho` | integer vector, R 1-based | Initial rho-grid indices; converted to Python 0-based | yes |
| `sigma` | length `ns` vector | Initial residual standard deviations/variances as expected by Hmsc | yes |
| `Eta` | list length `nr` | Random-level latent variables | yes, empty when `nr == 0` |
| `Lambda` | list length `nr` | Species loadings or random-slope loadings | yes, empty when `nr == 0` |
| `Psi` | list length `nr` | Loading shrinkage parameters | yes, empty when `nr == 0` |
| `Delta` | list length `nr` | Loading shrinkage parameters | yes, empty when `nr == 0` |
| `Alpha` | list length `nr` | Spatial alpha-grid indices, R 1-based | yes, empty when `nr == 0` |
| `BetaSel` | list length `ncsel` | Boolean variable-selection states | yes, empty when `ncsel == 0` |
| `wRRR` | matrix | Initial RRR weights | required when `ncRRR > 0` |
| `PsiRRR` | matrix | RRR shrinkage parameters | required when `ncRRR > 0` |
| `DeltaRRR` | vector/matrix | RRR shrinkage parameters | required when `ncRRR > 0` |

## Notes For Python-Native Initialization

- R indices in `Pi`, `rho`, `Alpha`, `XSelect$covGroup`, and `XSelect$spGroup`
  are converted to Python zero-based indices by the loader.
- The loader assumes `hM$distr` already uses Hmsc's internal integer
  distribution codes, not strings like `"poisson"`.
- `XScaled`, `YScaled`, and `TrScaled` are R/Hmsc-transformed values. A
  Python-native initializer must reproduce Hmsc's scaling and default intercept
  conventions before it can write compatible inputs.
- For fixed-effect-only models, `nr`, `ncsel`, and `ncRRR` can be zero, but the
  corresponding list-like fields still need to be present in the R-exported JSON.

## Python-Native Format Target: JSON + HDF5

The Python-native path should not make `.rds` the main exchange format. Raw
data files and initialized model files have different jobs:

| Layer | Recommended formats | Purpose |
| --- | --- | --- |
| Raw response/covariate data | CSV, TSV, Parquet, NPY, NPZ | User-owned source data |
| User model config | YAML or JSON | Formula, distribution, chains, model options |
| Compiled model metadata | `init.json` | Sampler-ready schema, dimensions, names, priors, array references |
| Compiled model arrays | `init_arrays.h5` | Numeric matrices and chain initial values |
| Posterior output | Zarr or HDF5 later; RDS compatibility now | Large posterior draws |

The first Python-native compiler writes:

```text
run_001/
  init.json
  init_arrays.h5
```

Example `init.json`:

```json
{
  "schema_version": "0.1",
  "model_type": "hmsc",
  "format": "pyhmsc-json-hdf5",
  "distribution": "poisson",
  "formula": {
    "X": "~ forest_cover + elevation"
  },
  "dimensions": {
    "n_sites": 5,
    "n_species": 3,
    "n_covariates": 3,
    "n_chains": 4
  },
  "arrays": {
    "Y": "init_arrays.h5:/Y",
    "X": "init_arrays.h5:/X",
    "Beta_init": "init_arrays.h5:/Beta_init"
  }
}
```

The current Python-native sampler loader supports fixed effects plus initial
support for traits, phylogenetic covariance matrices, iid random intercepts, and
full spatial random intercepts. It maps the compiled artifact into the existing
sampler internals:

| Compiled field | Internal sampler object |
| --- | --- |
| `dimensions` | `modelDims` |
| `Y`, `X` | `modelData["Y"]`, `modelData["X"]`, `initParList[*]["Xeff"]` |
| `distribution` | `modelData["distr"]` integer matrix |
| `T` | `modelData["T"]` trait design matrix |
| `C` | `modelData["C"]`, `modelData["eC"]`, `modelData["VC"]` |
| `Pi`, random-level init arrays | `modelData["Pi"]`, `rLHyperparams`, `initParList[*]["Eta"]`, `Lambda`, `Psi`, `Delta`, `AlphaInd` |
| `Beta_init` | `initParList[*]["Beta"]` |
| `priors.Beta` | default `Gamma`, `V`, and sigma prior hyperparameters |

This is a functional fixed-effect Python-native path. Scientific validation
should be done with pure-Python simulation recovery and posterior predictive
tests so routine development does not depend on R.

Supported native feature scope:

- Traits: user-provided species trait table plus `trait_formula`
- Phylogeny: user-provided species covariance matrix
- Random effects: iid random intercepts
- Spatial effects: full spatial random intercepts from per-level coordinates

Not yet supported:

- Newick tree parsing
- GPP/NNGP spatial approximations
- Random slopes
- Trait/phylogeny/spatial model diagnostics beyond core posterior samples
