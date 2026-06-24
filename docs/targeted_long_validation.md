# Targeted Longer Validation

Use longer or 4-chain validation only when a diagnostic makes it necessary.
Routine smoke tests and predictive hold-out checks should stay short so LUMI
GPU time is not spent on models that already answer the current question.

## Planning Step

Inspect one or more completed posteriors:

```bash
python examples/plan_long_validation.py run/posterior.h5 \
  --include-latent \
  --output run/targeted_long_validation_plan.txt \
  --csv-output run/targeted_long_validation_plan.csv
```

The planner checks `Beta`, `Gamma`, and `Associations` by default. With
`--include-latent`, it also checks aligned `Eta` and `Lambda` diagnostics. A
longer run is recommended only when R-hat or ESS flags remain under the chosen
thresholds.

## LUMI Targeted Run

Run only the model and diagnostic profile that needs extra evidence:

```bash
RUN_NAME=big_spatial_assoc_long \
DIAGNOSTIC_PROFILE=associations \
CHAINS=4 SAMPLES=2500 TRANSIENT=1000 THIN=10 \
sbatch docs/lumi_targeted_long_validation_sbatch.sh
```

Available profiles:

- `associations`: residual species association diagnostics and summaries
- `beta`: fixed-effect summaries and diagnostics
- `latent`: raw and aligned `Eta`/`Lambda` summaries and diagnostics
- `all`: all of the above

Set `SKIP_SAMPLE=1` and `POSTERIOR=/path/to/posterior.h5` to rerun diagnostics
against an existing posterior without resampling.

## Interpretation

Use `Associations` diagnostics for residual species association inference,
because raw latent factors are non-identifiable up to sign and rotation. Aligned
`Eta` and `Lambda` diagnostics are useful for debugging latent recovery, but
they should not replace identifiable association diagnostics when the scientific
target is residual species association.
