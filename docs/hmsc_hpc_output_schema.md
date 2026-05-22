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

`pyhmsc.HmscFit` currently consumes `Beta` for fixed-effect summaries and
predictions. The remaining fields are preserved in `fit.posterior` for later
diagnostics and feature expansion.
