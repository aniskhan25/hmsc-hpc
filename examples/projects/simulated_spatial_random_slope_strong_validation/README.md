# Simulated Strong Spatial Random-Slope Validation

This project is a higher-signal companion to
`simulated_spatial_random_slope_validation`. It uses
`simulate_spatial_random_slope_effect_data` with more sites, a normal response,
lower observation noise, and stronger random-slope loadings:

- `n_sites=81`
- `n_species=6`
- `spatial_sd=1.4`
- `lambda_intercept_scale=1.1`
- `lambda_slope_scale=1.8`
- `noise_sd=0.05`
- `distr="normal"`
- `seed=91`

Use this project when testing whether weak Lambda slope recovery in the baseline
scenario is caused by low signal rather than the sampler path itself.
