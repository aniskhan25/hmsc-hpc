# Big Spatial Plant Validation

Python-native real-data spatial validation project derived from the existing
`examples/big_spatial` plant community dataset.

The project uses a compact validation subset:

- sites: `site_part <= 3` (400 sites)
- species: 40 most prevalent `taxa_used_tree` species within those sites
- response: presence/absence, probit
- covariates: standardized `Hillshading270_40`, `HA_All_rivers_normalised`,
  `Thorium_mosaic_GWR2`, and `Max_temp_smooth`
- spatial coordinates: min/max scaled coordinates from `xy.csv`

Models:

- `model_fixed.yaml`: fixed effects only
- `model_iid.yaml`: iid site-level random intercept, `nf: 2`
- `model_spatial_full.yaml`: full spatial site-level random intercept, `nf: 2`

Validation target:

- fixed, iid, and full spatial models compile and sample without R
- iid/spatial random effects improve posterior predictive fit over fixed effects
- full spatial random effects reduce nearest-neighbor residual spatial
  autocorrelation relative to fixed and iid models

Run on LUMI with:

```bash
RUN_NAME=big_spatial_real_validation sbatch docs/lumi_big_spatial_validation_sbatch.sh
```

Validated LUMI run:

- run name: `big_spatial_real_validation_codex`
- environment: TensorFlow 2.16 on `dev-g` with an MI250X GPU
- sampler settings: 2 chains, 1000 saved samples, 500 transient iterations,
  thin 10
- elapsed Slurm time: 9 minutes 43 seconds

Result summary:

| Model | Species PPC | Site richness PPC | Species MAE | Site richness MAE | Neighbor residual corr |
| --- | --- | --- | --- | --- | --- |
| fixed | `40 / 40` | `309 / 400` | `0.002737` | `2.825639` | `0.427027` |
| iid | `40 / 40` | `400 / 400` | `0.001568` | `0.541675` | `0.299458` |
| full spatial | `40 / 40` | `400 / 400` | `0.001109` | `0.765010` | `-0.291249` |

## Longer Diagnostics Run

A longer diagnostics run was completed on LUMI as
`big_spatial_long_diag_242e08a`.

- environment: TensorFlow 2.16 on `dev-g` with an MI250X GPU
- sampler settings: 2 chains, 2000 saved samples, 1000 transient iterations,
  thin 10
- elapsed Slurm time: 17 minutes 7 seconds
- diagnostics emitted: `beta_diagnostics.txt`, `eta_diagnostics.txt`, and
  `lambda_diagnostics.txt` for the random-effect models

Predictive validation remained consistent with the shorter run:

| Model | Species PPC | Site richness PPC | Species MAE | Site richness MAE | Neighbor residual corr |
| --- | --- | --- | --- | --- | --- |
| fixed | `40 / 40` | `311 / 400` | `0.002860` | `2.824321` | `0.427262` |
| iid | `40 / 40` | `400 / 400` | `0.001654` | `0.542507` | `0.292859` |
| full spatial | `40 / 40` | `400 / 400` | `0.001270` | `0.746754` | `-0.290284` |

Convergence diagnostics were clean for fixed-effect `Beta`, mixed for
random-effect model `Beta`, and not yet clean for latent `Eta`/`Lambda`
parameters:

| Model | Parameter | Max R-hat | Median R-hat | Min ESS | Median ESS | Flags |
| --- | --- | --- | --- | --- | --- | --- |
| fixed | Beta | `1.0035` | `0.9999` | `560.6` | `3352.9` | R-hat `0 / 200`, ESS `0 / 200` |
| iid | Beta | `1.0373` | `1.0004` | `116.9` | `1104.6` | R-hat `14 / 200`, ESS `44 / 200` |
| iid | Eta | `8.5843` | `1.0222` | `9.7` | `246.7` | R-hat `901 / 1600`, ESS `954 / 1600` |
| iid | Lambda | `7.0356` | `1.0668` | `9.3` | `51.0` | R-hat `102 / 160`, ESS `159 / 160` |
| full spatial | Beta | `1.1397` | `1.0009` | `27.9` | `698.4` | R-hat `42 / 200`, ESS `58 / 200` |
| full spatial | Eta | `4.5547` | `1.0493` | `8.2` | `28.9` | R-hat `1161 / 1600`, ESS `1590 / 1600` |
| full spatial | Lambda | `4.8482` | `1.0608` | `7.9` | `18.3` | R-hat `111 / 160`, ESS `160 / 160` |

Interpretation: the spatial model is behaving as expected for posterior
predictive checks and spatial residual structure, but latent random-effect
posterior summaries should be treated as exploratory until a longer or
4-chain validation run improves `Eta`/`Lambda` diagnostics.
