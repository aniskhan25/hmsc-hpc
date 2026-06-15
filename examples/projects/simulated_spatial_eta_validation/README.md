# Simulated Spatial Eta Validation

This project is a focused validation dataset for latent spatial Eta recovery. It isolates one spatial random-intercept latent factor with known species loadings, uses a normal response, and compares full spatial, GPP, and NNGP approximations.

- `n_sites=100`
- `n_species=6`
- `spatial_range=0.24`
- `spatial_sd=1.6`
- `lambda_scale=1.2`
- `noise_sd=0.06`
- `distr="normal"`
- `seed=121`

Use this project to check whether NNGP Eta recovery improves or stabilizes as the neighbor count increases, while PPC and Lambda recovery remain acceptable.
