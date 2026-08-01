# Generative Neural-HMSC IID v2 Disposable Retry Failure

Date: 2026-08-01

Protocol: `generative_neural_hmsc_iid_probit_v2_orbit`

Decision: `stop_before_511m_numerical_failure`

## Execution

- LUMI job: `20518775`
- partition: `dev-g`
- state: `FAILED`
- elapsed: `00:02:06`
- exit code: `1:0`
- source commit: `940d73d6de6e032797e4d695bd9799a74ef0b943`
- source archive SHA-256:
  `76911182c1d34bcd4c979f70b1340af126ddd89baafdc821c4024cc6f846a43a`
- corrected scheduler SHA-256:
  `ff618d92c4d4f616507aaa31e2f434cb2cdaa9b2d985bcc0e7e567bc6735cdb7`
- retry authorization SHA-256:
  `257cba945a3d7a40697190812021a69d01be5e7a831d8f2fca0e69018ef4770f`

The scheduler used an explicit environment export list and the exact disposable
confirmation. The token-free preflight passed before the disposable token was
exported.

## Independent Evidence

Downloaded artifact hashes:

| Artifact | SHA-256 |
|---|---|
| token-free preflight | `86d5a67dab05a5d89232dcca09f8cbb34694c155838dabfbc2ba49905ff3b3bb` |
| corpus manifest | `6c2dd30e0a2a90b37cff16201d27756a07bd02ef2c51e3de6120eb2ee5447a8b` |
| scheduler stdout | `78c5abef6058493b3cd28a8ba3a4afe52ed19b1bd9da529df1f258082f697d8a` |
| scheduler stderr | `88b3a432454905c0252187a773095e6948f3bc12e8c7c3c6c25928dd2016682e` |

The preflight reports:

- `simulation_generation_called = false`;
- `output_created = false`;
- `production_511m_opened = false`;
- `fixed_validation_512m_opened = false`; and
- `reserved_513m_515m_opened = false`.

The disposable corpus manifest contains exactly 18 training records over
`593000001-593000018` and 18 masked-validation records over
`594000001-594000018`. Independent local regeneration from the frozen
simulator matched all 36 `dataset_sha256` fingerprints with zero mismatches.
The manifest sets all 511M-515M opening flags to false.

## Numerical Failure

Training emitted repeated failed batched Cholesky decompositions on the MI250X,
followed by:

```text
FloatingPointError: non-finite v2 gradient
```

The exception was raised by the frozen finite-gradient check in
`train_generative_iid_orbit_model`. The partial run contains only
`corpus_manifest.json`. It contains no:

- checkpoint manifest or weights;
- disposable smoke report;
- freeze inventory;
- post-freeze validation;
- finite optimization summary;
- validation loss or IWELBO replay; or
- exact-target replay.

Therefore the operational smoke and artifact acceptance conjunction fails.
Scheduler correction succeeded, but the candidate did not demonstrate finite
GPU optimization.

## Boundary Decision

Do not open 511M-515M. Do not treat the generated corpus as a qualified
checkpoint or disposable pass. The retry authorization is consumed and does
not permit another execution.

The only defensible next decision is a bounded no-seed numerical review. It may
repair a proven backend-specific implementation defect only if the frozen
posterior mathematics, representation, objective, refinement, schedule, gates,
thresholds, and seed roles remain unchanged. If finite gradients cannot be
restored under those constraints using ordinary non-ledger fixtures, close the
generative iid v2 family. Any later disposable execution requires a new
explicit authorization; production seeds remain sealed meanwhile.
