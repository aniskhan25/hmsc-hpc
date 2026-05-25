# Hmsc-HPC Output Schema

The compatibility sampler writes an `.rds` file containing a single JSON string
through `hmsc.utils.export_rds_utils.save_chains_postList_to_rds()`. When the
output path ends in `.json`, the sampler writes the same chain/sample structure
directly as JSON for Python-native smoke runs.

Top-level output:

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `0`, `1`, ... | object | One object per returned chain, keyed from zero |
| `time` | number | Elapsed sampler time in seconds |

Each chain object is keyed by sample index (`0`, `1`, ...). Each sample contains:

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `Beta` | `nc x ns` matrix | Fixed-effect coefficient draw |
| `BetaSel` | object/list length `ncsel` | Variable-selection draw |
| `Gamma` | `nt x nc` matrix | Trait coefficient draw |
| `iV` | `nc x nc` matrix | Inverse covariance draw |
| `rhoInd` | integer vector, R 1-based in output | Rho-grid index draw |
| `sigma` | length `ns` vector | Sigma draw |
| `Lambda` | object/list length `nr` | Random-level loading draws |
| `Psi` | object/list length `nr` | Loading shrinkage draws |
| `Delta` | object/list length `nr` | Loading shrinkage draws |
| `Eta` | object/list length `nr`, or `null` | Random-effect latent variable draws |
| `Alpha` | object/list length `nr` | Spatial alpha-grid indices, R 1-based in output |
| `wRRR` | matrix or `null` | RRR weight draw |
| `PsiRRR` | matrix or `null` | RRR shrinkage draw |
| `DeltaRRR` | vector/matrix or `null` | RRR shrinkage draw |

`pyhmsc.HmscFit` consumes `Beta` for summaries and predictions and now exposes
core summaries for `Gamma`, `sigma`, `Eta`, `Lambda`, and `rhoInd` where present.
HDF5 posterior output stores dense core arrays plus nested random-level groups:

```text
Beta
Gamma
iV
rhoInd
sigma
random_levels/0/Eta
random_levels/0/Lambda
random_levels/0/Psi
random_levels/0/Delta
random_levels/0/Alpha
```

For Python-native runs sampled from `init.json`, HDF5 and Zarr posterior output
also preserve the compiled model metadata under the `pyhmsc_metadata` root
attribute. `pyhmsc.HmscFit.from_file()` reads this metadata so posterior-only
summaries and predictions can recover species names, covariate names, formula,
distribution, and random-level metadata without requiring the original
`HmscModel` object.

HDF5 posterior shards from separate chain jobs can be merged with:

```bash
python -m pyhmsc chain-status run/chains --expected-chains 0 1
python -m pyhmsc merge run/chains/posterior_chain_*.h5 \
  --expected-chains 0 1 \
  --output run/posterior.h5
```

Merge concatenates every dataset along the first dimension, validates matching
dataset shapes after the chain axis, and preserves `pyhmsc_metadata`.

JSON/RDS compatibility output preserves the larger nested chain/sample
structure.

Optional Zarr output is available when the `zarr` extra is installed:

```bash
python -m pyhmsc sample run/init.json --output run/posterior.zarr
```
