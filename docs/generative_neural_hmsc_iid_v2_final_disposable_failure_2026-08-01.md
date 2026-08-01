# Generative Neural-HMSC IID v2 Final Disposable Failure

Date: 2026-08-01

Protocol: `generative_neural_hmsc_iid_probit_v2_orbit`

Decision: `close_generative_iid_v2_before_production_nonfinite_gradient`

## Execution

- LUMI job: `20521366`
- partition: `dev-g`
- state: `FAILED`
- elapsed: `00:02:12`
- exit code: `1:0`
- maximum resident memory: `2603944K`
- source commit: `cca9e97518e77c5ca958dfdc3bee753997ed7ac5`
- source archive SHA-256:
  `bb343bcef927455b5ffedb0483015f75f3da053176d58c1f032b3fece7790eb1`
- terminal scheduler SHA-256:
  `9ca1238d7e88560e58b0e92727c821933e90ec704d1ea69e61a86c4aef31066c`
- final authorization SHA-256:
  `a3e051a916798a41ea53cdb3d63bcf6f3685a342986003bef01d1284c477d858`

The scheduler was submitted exactly once with an explicit environment export
list and the authorized `593M-594M` confirmation. It first completed the
token-free preflight, then opened only disposable training
`593000001-593000018` and masked validation `594000001-594000018`.

## Downloaded Evidence

| Artifact | SHA-256 |
|---|---|
| token-free preflight | `a9f3c3f0f535f31217da279f9907f8c1d0fcf11001a7337ffbb4a4fdade9fe6f` |
| corpus manifest | `6c2dd30e0a2a90b37cff16201d27756a07bd02ef2c51e3de6120eb2ee5447a8b` |
| scheduler stdout | `10e062b977de53698b38b9463dad8345fd0a75276bd3961f6cf592d237d2c54f` |
| scheduler stderr | `cd4b216d685ae59e778f38d8040909ded57c40182b8abf21febb376f5569f90a` |

The token-free preflight validates all 11 source inventory records against the
exact source archive. It reports no simulation or output generation and keeps
every disposable and later-seed opening flag false.

Independent local regeneration from the exact archived simulator matched all
18 training and 18 validation records, including all 36
`dataset_sha256` fingerprints, with zero mismatches. The regenerated manifest
is exactly equal to the downloaded manifest.

## Acceptance Results

| Requirement | Result | Evidence |
|---|---|---|
| 36 corpus fingerprints | PASS | `36/36`; exact manifest equality |
| freeze inventory and hashes | FAIL | no `freeze.json` was produced |
| checkpoint schema and weight hashes | FAIL | no checkpoint was produced |
| finite optimization | FAIL | frozen gradient guard raised `FloatingPointError` |
| deterministic validation replay | FAIL | training stopped before replay |
| exact-target replay | FAIL | training stopped before exact-target evaluation |
| zero Cholesky warnings | PASS | zero case-insensitive `Cholesky` log records |
| source provenance | PARTIAL | source/archive/preflight hashes pass; checkpoint provenance is absent |
| later-seed seals | PASS | every 511M-515M opening flag is false |

The only file in the partial run root is `corpus_manifest.json`. There is no
checkpoint, weights file, disposable smoke report, freeze inventory, or
post-freeze validation record.

## Terminal Numerical Failure

TensorFlow initialized the MI250X and entered the frozen two-epoch training
path. Training then stopped at the unchanged finite-gradient guard:

```text
FloatingPointError: non-finite v2 gradient
```

The repaired rank-16 Woodbury factorization emitted zero failed-Cholesky
warnings, so the prior backend-specific Cholesky symptom was removed. The
end-to-end v2 objective still does not provide finite gradients on the frozen
disposable corpus. The candidate therefore fails independently of corpus
reproducibility and before an immutable model artifact exists.

## Closure Decision

This was the terminal disposable execution authorized for generative iid v2.
The acceptance rule was conjunctive: any scheduler, numerical, artifact,
checkpoint, replay, exact-target, provenance, warning, or seal failure closes
v2 before production. Finite optimization and all downstream artifact/replay
requirements failed.

Decision: close generative iid v2. Do not open 511M-515M, do not retry this
representation, and do not report it as a neural HMSC capability. The branch
retains the negative evidence and implementation scaffolding only.

The next roadmap step is a no-seed branch-level closure audit. It must state
the qualified neural capabilities, failed generative families, retained Python
MCMC reference path, and whether any branch components are worth merging or
archiving. It must not introduce another calibration or generative model.
