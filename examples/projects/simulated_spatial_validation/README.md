# Simulated Spatial Validation

Deterministic spatial validation dataset generated with
`pyhmsc.simulate_spatial_effect_data(n_sites=36, n_species=5, seed=21)`.

The project is intended for comparing three Python-native no-R models:

- `model_fixed.yaml`: fixed effects only
- `model_iid.yaml`: iid plot-level random intercept
- `model_spatial_full.yaml`: full spatial plot-level random intercept

The response is simulated from a probit model with one environmental covariate
(`env`), one latent spatial site effect, and species-specific loadings.

Truth files:

- `data/truth_beta.csv`: intercept and environmental coefficients
- `data/truth_site_effect.csv`: latent spatial site effect
- `data/truth_lambda.csv`: species loadings for the shared spatial factor

Expected validation pattern:

- fixed-effect model should miss spatial site structure
- iid random intercept should improve site-level PPC
- full spatial random intercept should recover smoother site effects and reduce
  residual spatial structure

Validated LUMI run:

- run name: `spatial_validation_full_codex`
- environment: TensorFlow 2.16 on `dev-g` with an MI250X GPU
- sampler settings: 2 chains, 1000 saved samples, 500 transient iterations,
  thin 10
- elapsed Slurm time: 7 minutes 15 seconds

Result summary:

| Model | Beta signs | Species PPC | Site richness PPC | Eta/truth correlation |
| --- | --- | --- | --- | --- |
| fixed | `4 / 4` | `5 / 5` | `36 / 36` | n/a |
| iid | `4 / 4` | `5 / 5` | `36 / 36` | `0.675717` |
| full spatial | `4 / 4` | `5 / 5` | `36 / 36` | `0.868592` |
