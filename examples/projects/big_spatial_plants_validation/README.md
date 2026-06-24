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

For a true out-of-sample comparison, generate the deterministic spatially
blocked 319-site training and 81-site test project and run fixed, full-spatial,
GPP, and NNGP models:

```bash
python examples/generate_big_spatial_holdout_validation.py
RUN_NAME=big_spatial_holdout_validation_real \
  sbatch docs/lumi_big_spatial_holdout_validation_sbatch.sh
```

The hold-out analyzer reports Brier score, log loss, macro AUC, prevalence and
richness MAE, plus elapsed time, peak RSS, compiled size, and posterior size.
The spatial models use conditional prediction at the unseen coordinates.

Validated hold-out run `big_spatial_holdout_validation_short` completed on
LUMI as job `19435459` with two chains, 250 saved draws, 250 transient
iterations, and thin 5. GPP had the best Brier score (`0.069072`) and macro AUC
(`0.732161`). Full-spatial and NNGP did not improve Brier score over fixed in
this split. The differences were small, and short-run Beta convergence was not
clean, so use this result as an end-to-end predictive validation rather than a
publication-grade model comparison.

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
4-chain validation run improves `Eta`/`Lambda` diagnostics. Prefer
`Associations` diagnostics for identifiable residual association summaries,
because raw `Eta`/`Lambda` draws are sensitive to latent-factor sign and
permutation switching.

To run the spatial model only with a chain override, use:

```bash
RUN_NAME=big_spatial_4chain_diag \
CHAINS=4 SAMPLES=2000 TRANSIENT=1000 THIN=10 \
sbatch docs/lumi_big_spatial_4chain_diagnostics_sbatch.sh
```

This script compiles `model_spatial_full.yaml` with `pyhmsc compile --chains`
so the YAML file does not need to be edited for diagnostic runs.

The 4-chain spatial-only run `big_spatial_4chain_diag_codex` completed on
LUMI in 20 minutes 47 seconds with 2000 saved samples, 1000 transient
iterations, and thin 10. Raw latent diagnostics remained poor (`Eta` median
R-hat `1.6912`, `Lambda` median R-hat `1.9494`), but identifiable residual
association diagnostics were substantially better:

| Parameter | Max R-hat | Median R-hat | Min ESS | Median ESS | Flags |
| --- | --- | --- | --- | --- | --- |
| Associations | `1.0493` | `1.0080` | `189.8` | `700.2` | R-hat `320 / 780`, ESS `103 / 780` |

Interpretation: diagnostics on `Lambda.T @ Lambda` support the view that raw
latent loading diagnostics are strongly affected by latent-factor
non-identifiability. Association diagnostics are the preferred convergence
summary for residual species association inference, but the 4-chain run still
has enough R-hat flags to warrant longer sampling before publication-grade
association estimates.

The longer 4-chain spatial-only association run
`big_spatial_4chain_assoc_long_codex` completed on LUMI in 24 minutes 58
seconds with 2500 saved samples, 1000 transient iterations, and thin 10. It
also ran post-hoc latent-factor alignment diagnostics on the completed
posterior.

| Parameter | Max R-hat | Median R-hat | Min ESS | Median ESS | Flags |
| --- | --- | --- | --- | --- | --- |
| Beta | `1.0684` | `1.0012` | `65.4` | `1709.8` | R-hat `22 / 200`, ESS `40 / 200` |
| Associations | `1.0247` | `1.0052` | `242.3` | `782.0` | R-hat `153 / 780`, ESS `72 / 780` |
| Eta, aligned | `1.0731` | `1.0064` | `41.0` | `208.3` | R-hat `591 / 1600`, ESS `1438 / 1600` |
| Lambda, aligned | `1.0738` | `1.0125` | `38.9` | `180.5` | R-hat `92 / 160`, ESS `130 / 160` |

Interpretation: extending the run improved identifiable association
diagnostics substantially (`max R-hat` decreased from `1.0493` to `1.0247`,
and R-hat flags decreased from `320 / 780` to `153 / 780`). Post-hoc alignment
makes raw `Eta`/`Lambda` diagnostics much more interpretable than unaligned
draws, but the latent factors still have low ESS. For residual species
association inference, use the identifiable `Associations` diagnostics rather
than raw latent-factor diagnostics.
