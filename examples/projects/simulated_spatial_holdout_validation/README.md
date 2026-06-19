# Simulated Spatial Hold-Out Validation

This project validates prediction at sites excluded from model fitting. A
deterministic 100-site normal-response simulation is split into 80 training
sites and 20 spatially interspersed test sites. Test coordinates are distinct
from all training coordinates.

The workflow compares:

- fixed-effects-only prediction,
- full spatial nearest-unit prediction,
- GPP spatial nearest-unit prediction,
- NNGP spatial nearest-unit prediction.

Nearest-unit prediction is the currently implemented baseline. It reuses the
posterior random effect of the closest sampled unit; it is not conditional GP
or NNGP interpolation. The report compares held-out predictions against the
known test linear predictor using correlation, RMSE, MAE, and posterior interval
coverage.

Regenerate project data with:

```bash
python examples/generate_spatial_holdout_validation.py
```

Run on LUMI with:

```bash
RUN_NAME=spatial_holdout_validation \
  sbatch docs/lumi_spatial_holdout_validation_sbatch.sh
```
