# Simulated Random-Slope and GPP Validation

Deterministic no-R validation project for the native TensorFlow paths added
after fixed, iid, and full-spatial support.

The project has two independent subprojects:

- `random_slope`: iid random-slope probit model generated with
  `simulate_random_slope_effect_data(n_groups=12, sites_per_group=4, n_species=5, seed=31)`.
- `spatial_gpp`: spatial probit model generated with
  `simulate_spatial_effect_data(n_sites=36, n_species=5, seed=42)`.

Validation targets:

- iid random-slope model samples without R and improves known random-effect PPC
  relative to fixed effects.
- GPP spatial random intercept samples without R and has behavior comparable to
  the full spatial random intercept model on the same deterministic dataset.

Run on LUMI with:

```bash
RUN_NAME=new_features_validation sbatch docs/lumi_new_features_validation_sbatch.sh
```

Validated LUMI run `new_features_validation_fixed2_codex`:

- 2 chains, 1000 saved samples, 500 transient iterations, thin 10
- completed in 8 minutes 50 seconds on `dev-g`
- TensorFlow 2.16 with an MI250X GPU
- fixed and iid random-slope models both recovered beta signs `4 / 4`
- random-slope known random-effect PPC covered `5 / 5` species and `48 / 48`
  site richness values
- random-slope latent recovery was positive but moderate: Eta/truth
  correlation `0.434882`, slope Lambda/truth correlation `0.561890`
- full spatial and GPP spatial models both covered `5 / 5` species and
  `36 / 36` site richness values
- full spatial Eta/truth and Lambda/truth correlations were `0.827719` and
  `0.921528`
- GPP Eta/truth and Lambda/truth correlations were `0.819624` and `0.929552`

The GPP run recovered `3 / 4` nonzero beta signs while matching full-spatial
latent recovery closely. Treat this project as a runtime and qualitative
behavior validation, not as a publication-grade coefficient recovery benchmark.
