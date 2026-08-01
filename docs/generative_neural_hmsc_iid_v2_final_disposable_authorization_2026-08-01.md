# Generative Neural-HMSC IID v2 Final Disposable Authorization

Date: 2026-08-01

Protocol: `generative_neural_hmsc_iid_probit_v2_orbit`

Decision: `authorize_one_final_repaired_disposable_verification`

## Rationale

Authorize one final repaired disposable verification. The accepted change is
limited to mathematically equivalent symmetric float64 CPU factorization of the
rank-16 Woodbury system. The representation, posterior density, IWAE objective,
refinement, architecture, optimizer schedule, simulator, gates, thresholds,
and seed roles remain unchanged.

The decision follows:

- complete local repaired suite: `30/30` tests passed;
- no-ledger LUMI repair validation job `20519352` with finite optimization and
  zero failed-Cholesky warnings; and
- token-free/no-seed LUMI preflight job `20520889`, whose JSON was
  byte-identical to local output and kept every opening flag false.

This decision authorizes execution only. It does not claim a disposable pass.

## Frozen Candidate Boundary

```text
source_commit = cca9e97518e77c5ca958dfdc3bee753997ed7ac5
archive_sha256 = bb343bcef927455b5ffedb0483015f75f3da053176d58c1f032b3fece7790eb1
archive_bytes = 3802838
model_sha256 = 87828857ee1718a8825a1a15e7af99abe49a86ee4d179f6cbce6591162aa71bc
harness_sha256 = b09e3e1eb743fe62a509876a284323e5bb151ce043cac7b60dba8a9a35f9300e
numerical_review_sha256 = e3b708a09b0c920676e592759f44f7457cc75decff66138b6b29c509254a6192
preflight_sha256 = a9f3c3f0f535f31217da279f9907f8c1d0fcf11001a7337ffbb4a4fdade9fe6f
```

The one-shot scheduler is:

```text
path = docs/lumi_generative_neural_hmsc_iid_v2_disposable_final_sbatch.sh
sha256 = 9ca1238d7e88560e58b0e92727c821933e90ec704d1ea69e61a86c4aef31066c
```

It is fixed to LUMI `dev-g`, one GPU, seven CPUs, 60 GB memory, and a
three-hour limit. It verifies the repaired source, harness, and numerical-review
hashes before running a token-free preflight.

## Authorized Opening

Authorize one scheduler execution plus the deterministic independent replay
performed by that scheduler for only:

- disposable training `593000001-593000018`; and
- disposable masked validation `594000001-594000018`.

The exact confirmation is:

```text
OPEN_GENERATIVE_IID_V2_593M_594M_DISPOSABLE_SMOKE=\
GENERATE_593M_594M_DISPOSABLE_ONLY
```

No other nonempty `OPEN_GENERATIVE_IID*` environment variable may be present.
The scheduler receives the confirmation as a non-opening shell variable,
performs preflight first, then exports only the exact opening variable.

The verified isolated source and fresh output roots are:

```text
source = /scratch/project_462000131/anisrahm/hmsc-hpc-sources/generative_iid_v2_repair_cca9e97
run = /scratch/project_462000131/anisrahm/hmsc-hpc-runs/generative_iid_v2_disposable_final_cca9e97_20260801
```

The run root and its preflight sidecar must both be absent before submission.
The decision-time LUMI check confirmed both absent and found no active
generative-v2 scheduler job.

## Required Acceptance Evidence

Scheduler exit status alone is insufficient. Independent download and replay
must establish:

1. all 18 training and 18 validation corpus fingerprints;
2. exact freeze inventory path, size, and SHA-256 agreement;
3. checkpoint schema, architecture, source provenance, content hash, and weight
   hash;
4. repaired model, harness, and numerical-review hashes in provenance;
5. no calibration or external candidate dependencies;
6. finite training loss, IWELBO, gradient norm, validation loss, invariant
   outputs, and nonzero optimizer movement;
7. exactly four validation refinement steps;
8. deterministic validation loss and IWELBO replay;
9. exact-target replay for the first validation community;
10. zero failed-Cholesky warnings in the scheduler logs; and
11. false `production_511m_opened`, `fixed_validation_512m_opened`, and
    `reserved_513m_515m_opened` flags in every relevant artifact.

## Terminal Boundary

This is the final disposable execution for generative iid v2. Any scheduler,
numerical, artifact, checkpoint, replay, exact-target, provenance, warning, or
seal failure closes this candidate before production. It does not authorize an
automatic retry, repair, tuning pass, or representation change.

Passing every disposable requirement does not open 511M-515M. It permits only
a separate production-authorization decision. Production training, fixed
validation, reserved evaluation, real data, and every 511M-515M seed remain
sealed under this authorization.
