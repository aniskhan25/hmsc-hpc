# Neural-HMSC Whittaker Real-Data Validation

This validation applies the experimental fixed-effect Neural-HMSC posterior to
the real Whittaker plant community dataset. It uses the repository's
deterministic 40-site training and 12-site held-out split with 75 species.

## Scope

The matched model is:

```text
presence ~ TMG
distribution: probit
```

Traits, phylogeny, and latent site effects are intentionally excluded because
the current public neural checkpoint does not support them. The MCMC reference
uses the identical fixed-effect model.

The shape-matched simulation corpus uses the observed training TMG design, a
rare-species intercept mixture, and zero-centered Normal TMG slopes. The LUMI
run used 512 training datasets, 128 calibration datasets, 128 SBC datasets,
120 epochs, 512 SBC draws, 4 x 1000 neural draws, and 2 x 1000 MCMC draws.

## Run

```text
LUMI job: 19609057
state: COMPLETED
elapsed: 00:03:21
MaxRSS: 3735036K
run root: /scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_whittaker_real_62e6b3c
```

## Held-Out Results

| Model | Brier | Log loss | Macro AUC | Prevalence MAE | Richness MAE | Richness slope |
|---|---:|---:|---:|---:|---:|---:|
| Neural uncalibrated | 0.0924 | 0.3274 | 0.5854 | 0.1209 | 5.455 | -2.164 |
| Neural calibrated | 0.1272 | 0.4234 | 0.5904 | 0.2224 | 14.723 | -0.896 |
| MCMC fixed | 0.0740 | 0.2622 | 0.5433 | 0.0694 | 3.378 | -3.569 |

Observed held-out richness slope was -1.513. The uncalibrated neural model
ranked species occurrences better than MCMC by macro AUC, but MCMC retained
better probability, prevalence, and richness accuracy.

## Posterior and Calibration Results

- Neural/MCMC Beta posterior-mean correlation: 0.844 uncalibrated.
- Neural/MCMC Beta mean RMSE: 0.579 uncalibrated.
- Neural/MCMC 95% interval overlap: 0.406 uncalibrated.
- Shape-matched uncalibrated SBC coverage: 51.9% at nominal 95%.
- Simulation-fitted scale multiplier: 5.241.
- Shape-matched calibrated SBC coverage: 95.5%.
- Both SBC rank histograms remained strongly nonuniform. Calibration changed
  the rank variance from 0.168 to 0.051 versus a uniform expectation near
  0.083, moving from underdispersion to overdispersion.
- Applying the simulation scale to the real posterior worsened Brier score,
  log loss, prevalence MAE, richness MAE, posterior-SD agreement, and interval
  overlap.

## Runtime

```text
neural training: 102.476 seconds
neural real-data inference: 0.078 seconds
MCMC sampling: 48.873 seconds
inference-only speedup: 624.3x
```

For one dataset, neural training costs more than the MCMC reference. The speed
advantage is meaningful only when a validated checkpoint is reused across many
datasets from the same domain.

## Conclusion

The current model is useful as a fast ranking-oriented predictor, but the real
data do not support treating its calibrated samples as an HMSC-equivalent Beta
posterior. A single scalar simulation calibration is insufficient for rare
species and does not transfer reliably to real held-out predictions. The next
model revision should use prevalence-aware or species-conditional calibration,
add a probit-aware encoder anchor, and require real held-out predictive metrics
alongside SBC before accepting a checkpoint.
