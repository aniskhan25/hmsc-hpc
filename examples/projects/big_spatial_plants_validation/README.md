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
