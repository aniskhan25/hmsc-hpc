# Generative Neural-HMSC IID Probit v1 Implementation

Date: 2026-07-27

Branch: `feature/generative-neural-hmsc`

Protocol: `generative_neural_hmsc_iid_probit_v1`

Status: implementation and preregistered tests complete; disposable and
production seeds remain unopened.

## Implemented Components

### Structural simulator and tensor contract

`pyhmsc/neural/generative_iid.py` implements:

- the exact alpha, Beta, Eta, Lambda, and log(tau) prior;
- Bernoulli-probit response generation;
- prior-conditional loading-strength and prevalence strata;
- normal and right-skewed standardized covariates;
- independent named RNG streams;
- outcome-blind 20% cell masking with observed/hidden row and column support;
- padded variable site/species batches and masks;
- exact truth tensors for R, A, and C.

The old iid residual-SVD simulator/model remains separate and cannot load as
this candidate.

### Exchangeable joint neural posterior

The same module implements:

- three alternating site/species bipartite message-passing rounds;
- shared 64-unit GELU message/update networks;
- no learned site/species IDs or positional encodings;
- shared Beta, Eta, and Lambda local posterior heads;
- a pooled alpha/log(tau) global head;
- one rank-16 diagonal-plus-low-rank Normal over the complete padded state;
- masked Woodbury log probability and determinant-lemma log determinant;
- posterior sampling and invariant R/A/C summaries;
- deterministic Eta/Lambda gauge fixing for compatibility only.

The candidate receives only X, Y, response, site, and species masks. Its
artifact dependency inventory is empty.

### Generative objective and training

The implementation includes:

- observed-cell probit likelihood;
- exact hierarchical prior density;
- eight-sample importance-weighted variational loss;
- frozen AdamW/cosine-decay schedule;
- epochs 1-20 KL warm-up;
- gradient clipping and non-finite aborts;
- deterministic context shuffling.

Simulation truths do not enter training.

### Exact-model MCMC reference

`pyhmsc/neural/generative_iid_mcmc.py` implements:

- the same target log density in float64;
- non-neural prior-scale chain initialization;
- TensorFlow Probability NUTS with dual-averaging adaptation;
- fixed protocol-derived Beta/R/C projections;
- non-gauge registered diagnostics;
- split R-hat and bulk-ESS summaries.

The runner was executed on a tiny ordinary-seed fixture. This verifies the
sampler path only and is not qualification evidence.

### Immutable artifact

`pyhmsc/neural/generative_iid_artifact.py` implements:

- structural checkpoint creation, save, load, and validation;
- exact architecture and scope validation;
- weights hash and size validation;
- exact checkpoint file-set validation;
- frozen preregistration, seed-audit, and review hashes;
- explicit null calibration;
- rejection of nonempty dependency inventories;
- rejection of legacy model-family manifests and modified weights.

### Sealed harness

`examples/run_generative_neural_hmsc_iid_v1.py` implements:

- document-hash validation before seed access;
- a no-seed `check-seal` mode;
- an exact 18-cell 591M/592M disposable factorial;
- explicit confirmation through
  `OPEN_GENERATIVE_IID_DISPOSABLE_SMOKE`;
- no production or reserved mode;
- disposable-only checkpoint, report, and freeze manifests.

The harness currently reports:

```json
{
  "sealed": true,
  "production_seed_ranges_opened": false
}
```

## Test Evidence

The new focused and compatibility suite passed:

```text
52 passed in 33.50s
```

It included:

- all generative iid v1 tests, including the explicit slow NUTS execution;
- legacy iid latent-factor tests;
- fixed-shape public Neural-HMSC API tests;
- variable-shape public API tests.

The ordinary focused suite separately passed:

```text
15 passed, 1 slow test skipped in 16.30s
```

The explicit slow NUTS test separately passed:

```text
1 passed in 13.02s
```

Additional checks passed:

- Python bytecode compilation for every new module, harness, and test;
- public lazy imports from `pyhmsc.neural`;
- `git diff --check`;
- frozen preregistration, seed-audit, and design-review hashes.

No 501M-515M, 591M, or 592M simulation dataset was generated.

## Claim Boundary

This completes implementation, not qualification. The model has not yet shown:

- calibrated Beta or R uncertainty;
- invariant joint calibration;
- association recovery;
- predictive improvement over the no-latent ablation;
- agreement with production exact-model MCMC;
- agreement with qualified Python HMSC-HPC;
- Whittaker transfer;
- production runtime compliance.

Those questions remain governed by the frozen gates.

## Next Step

Explicitly authorize and run only the 591M-592M disposable smoke using:

```text
OPEN_GENERATIVE_IID_DISPOSABLE_SMOKE=GENERATE_591M_592M_DISPOSABLE_ONLY
```

After completion, validate the smoke freeze, checkpoint hashes, finite
optimization, exact-target check, and seal booleans. Keep 501M-515M unopened.
