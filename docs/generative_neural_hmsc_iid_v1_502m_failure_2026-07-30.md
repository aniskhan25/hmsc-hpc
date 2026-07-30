# Generative Neural-HMSC iid v1: 502M qualification failure

Date: 2026-07-30

## Decision

The first production generative iid candidate failed fixed validation.

The immutable decision is:

`stop_before_reserved_evaluation`

Blocks 503M-505M, the Python-HMSC real-data replay, and promotion remain
blocked. Blocks 511M-515M remain sealed. This is a valid negative capability
result, not an infrastructure failure.

## Execution and artifact validation

Finalizer job `20461616` completed on LUMI `dev-g` in 10:16 with exit code
`0:0` and peak RSS 2,688,924 KiB.

Frozen output hashes:

- `freeze.json`:
  `6e5fe5afd296f4fd53f0a0bb08887982f99ee12ce5e43782bd33f33555360e45`;
- `fixed_validation_report.json`:
  `8e0a1047160a01a323193b328b496f71f45e6caf53da7476d529add0e095268a`;
- `context_metrics.json.gz`:
  `c0ef2bb3208f16b623933d53a82fd086ef03967baa9dc7b22d5ce91a74eab930`;
- `read_only_recovery_validation.json`:
  `98bee30f95f1a604084be3d36c0fdf314e5ea8b1650fd2e7423283a1b5c20d28`.

Standalone reconstruction validation used validator SHA-256
`66b19923b9b69774ff695cd4aae21bdae3db4ef553d662188b2ba8a67f7ce850`.
Its evidence SHA-256 is
`5313f5f68139bba72b0df8e4189b22216c58422acfe83feaa5f5d158178a6b2c`.
It confirmed:

- 36 exact seed owners;
- 36 reconstructed exact-MCMC files;
- 108 reconstructed Python-HMSC files;
- all 144 reconstructed files are the frozen shard hardlinks;
- shard-binding SHA-256
  `3cd102044677a48147475d8f951f108439c5306990e77cf228bba7730081fa45`;
- immutable `neural_hmsc_v0_1` content SHA-256
  `affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8`;
- no partial-attempt reuse and no opened later block.

The compressed metrics were downloaded and the frozen gate evaluator was
rerun locally. All 65 gate booleans matched the final report exactly.

## Qualification result

Twenty-six gates passed and 39 failed.

Posterior calibration failed materially:

| Diagnostic | Observed | Gate |
| --- | ---: | ---: |
| Beta 95% coverage | 0.6740 | 0.925-0.975 |
| R 95% coverage | 0.3062 | 0.90-0.98 |
| alpha 95% coverage | 0.3333 | 0.90-0.99 |
| log(tau) 95% coverage | 0.0247 | 0.90-0.99 |
| Beta rank mean | 0.3725 | 0.46-0.54 |
| Beta rank variance | 0.1279 | 0.060-0.108 |
| R rank variance | 0.2068 | 0.060-0.108 |
| R/exact interval-width ratio | 0.2203 | 0.75-1.35 |

Invariant projection coverage was 0.6557 for Beta and 0.2637 for R. The C
projection gates passed, but the candidate did not recover useful association
direction:

| Diagnostic | Observed | Gate |
| --- | ---: | ---: |
| association truth correlation, median | -0.0015 | >= 0.65 |
| association truth correlation, p10 | -0.0571 | >= 0.25 |
| candidate/Python association correlation | -0.0044 | >= 0.70 |
| candidate/exact association RMSE ratio | 1.1856 | <= 1.15 |
| random-effect RMSE ratio vs ablation | 1.0002 | <= 0.85 |

Prediction also failed comparator and posterior-predictive gates:

| Diagnostic | Observed | Gate |
| --- | ---: | ---: |
| masked Brier ratio vs exact | 1.1484 | <= 1.10 |
| masked log-loss ratio vs exact | 1.1610 | <= 1.10 |
| masked Brier ratio vs Python HMSC | 1.2110 | <= 1.10 |
| masked log-loss ratio vs Python HMSC | 1.2167 | <= 1.10 |
| new-site Brier ratio vs exact | 1.0476 | <= 1.03 |
| new-site log-loss ratio vs exact | 1.0348 | <= 1.03 |
| medium/strong Brier ratio vs ablation | 0.9968 | <= 0.98 |
| medium/strong log-loss ratio vs ablation | 1.0037 | <= 0.99 |
| site-richness 90% coverage | 0.7505 | 0.84-0.96 |

Species-prevalence coverage passed in aggregate at 0.8708, but registered
strata ranged from 0.1667 to 1.0 and failed. Site-richness strata ranged from
0.2865 to 1.0. Matched immutable-v0.1 coverage and proper-score gates failed.

The candidate passed finite-output, exact-MCMC diagnostics, padding, memory,
training-time, and speed gates. Maximum-shape inference took 0.139 seconds,
training used 11.03 GPU-hours, and the reported inference speedup over exact
MCMC was 58,521x. These operational results cannot qualify an inaccurate
posterior. Species permutation invariance also failed: maximum delta was
`3.968e-4` against the `2e-5` limit.

Exact MCMC evidence was healthy: maximum split R-hat was 1.0234, minimum bulk
ESS was 252.44, and one of 36 contexts used the preregistered continuation.

## Failure interpretation

The candidate behaves close to its no-latent ablation and does not recover
latent association structure. Its random-effect and loading-scale posteriors
are severely underdispersed, while Beta and alpha are biased. This pattern is
consistent with the preregistered primary risk: one low-rank joint Gaussian
over raw, rotationally non-identifiable Eta/Lambda coordinates is not an
adequate amortized representation of the structural posterior.

This cannot be repaired by post-hoc scaling, stratum-specific calibration,
threshold changes, MCMC teachers, or target routing. Those changes are
prohibited by the frozen stop rule and would not address the association and
latent-path collapse.

## Next step

Conduct one bounded, no-seed representation decision. Either:

1. preregister the single permitted redesign using a genuinely different
   posterior representation and encoder, while retaining the generative model,
   objective class, comparators, metrics, thresholds, and real-data boundary;
   or
2. close the iid generative family and retain Python MCMC as the qualified
   structural path.

Do not open 511M-515M until a redesign is concrete, technically defensible,
fully preregistered, and hash-frozen.
