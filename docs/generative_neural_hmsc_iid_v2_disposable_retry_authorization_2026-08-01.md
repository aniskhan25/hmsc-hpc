# Generative Neural-HMSC IID v2 Disposable Retry Authorization

Date: 2026-08-01

Protocol: `generative_neural_hmsc_iid_probit_v2_orbit`

## Decision

Separately authorize one retry execution and independent deterministic replay
of only:

- disposable training `593000001-593000018`; and
- disposable masked validation `594000001-594000018`.

This decision follows failed LUMI job `20518403`, which stopped during the
token-free import preflight. Independent inventory established that no 593M or
594M simulation seed and no later seed was opened by that attempt.

## Frozen Boundary

The candidate source remains exactly:

```text
source_commit = 940d73d6de6e032797e4d695bd9799a74ef0b943
archive_sha256 = 76911182c1d34bcd4c979f70b1340af126ddd89baafdc821c4024cc6f846a43a
archive_bytes = 3787568
```

The corrected scheduler is:

```text
path = docs/lumi_generative_neural_hmsc_iid_v2_disposable_retry_sbatch.sh
sha256 = ff618d92c4d4f616507aaa31e2f434cb2cdaa9b2d985bcc0e7e567bc6735cdb7
```

The only correction is LUMI container-safe source resolution using a relative
`PYTHONPATH` and module-mode harness launch. It does not alter the candidate,
simulator, posterior representation, objective, refinement, training schedule,
artifact contract, gates, thresholds, or seed roles. The corrected token-free
LUMI preflight passed without simulation generation or output creation.

The exact opening token is:

```text
OPEN_GENERATIVE_IID_V2_593M_594M_DISPOSABLE_SMOKE=\
GENERATE_593M_594M_DISPOSABLE_ONLY
```

No other `OPEN_GENERATIVE_IID*` environment variable may be present.

## Execution Boundary

The retry is fixed to LUMI `dev-g`, one GPU, seven CPUs, 60 GB memory, and a
three-hour limit. It must use the existing verified isolated source tree and a
fresh output root:

```text
source = /scratch/project_462000131/anisrahm/hmsc-hpc-sources/generative_iid_v2_940d73d
run = /scratch/project_462000131/anisrahm/hmsc-hpc-runs/generative_iid_v2_disposable_retry1_940d73d_20260801
```

The scheduler must run token-free preflight before exporting the opening token,
then run the two-epoch disposable smoke and independent validator. The opening
token returns to unset after this one scheduler execution.

## Acceptance Evidence

Scheduler completion alone is insufficient. The downloaded run must establish:

1. all 18 training and 18 validation corpus fingerprints by regeneration;
2. exact freeze inventory path, size, and SHA-256 agreement;
3. checkpoint schema, architecture, source provenance, and weight hash;
4. absence of calibration and external candidate dependencies;
5. finite loss, IWELBO, gradient norm, invariant outputs, and optimizer movement;
6. exactly four validation refinement steps;
7. deterministic fixed-validation loss and IWELBO replay;
8. exact-target replay for the first validation community; and
9. false `production_511m_opened`, `fixed_validation_512m_opened`, and
   `reserved_513m_515m_opened` flags in every relevant artifact.

## Prohibitions

This retry does not open 511M-515M, production training, fixed validation,
reserved evaluation, or real data. No model, simulator, loss, schedule, gate,
threshold, or seed role may change. A scheduler or candidate failure must be
recorded; it does not authorize another retry.
