# Generative Neural-HMSC IID v2 Disposable Authorization

Date: 2026-08-01

Protocol: `generative_neural_hmsc_iid_probit_v2_orbit`

## Authorization

Authorize one execution and independent replay of only:

- disposable training `593000001-593000018`; and
- disposable masked validation `594000001-594000018`.

The authorized source is exactly:

`940d73d6de6e032797e4d695bd9799a74ef0b943`

The exact opening token is:

```text
OPEN_GENERATIVE_IID_V2_593M_594M_DISPOSABLE_SMOKE=\
GENERATE_593M_594M_DISPOSABLE_ONLY
```

No other `OPEN_GENERATIVE_IID*` variable may be present.

## Execution Boundary

The source is packaged from the authorized commit into an isolated archive,
not run from the dirty shared LUMI checkout:

```text
archive_sha256 = 76911182c1d34bcd4c979f70b1340af126ddd89baafdc821c4024cc6f846a43a
archive_bytes = 3787568
```

The LUMI scheduler wrapper is:

`docs/lumi_generative_neural_hmsc_iid_v2_disposable_sbatch.sh`

SHA-256:

`7a0bf9ecf89a5e1896ba254d24e916978cfe8e76caf0898a7dffef1df679cf07`

It is fixed to:

- partition `dev-g`;
- one GPU, seven CPUs, 60 GB memory, and a three-hour limit;
- model seed `511900001`;
- exactly two smoke epochs;
- isolated source host attestation for the authorized commit;
- a token-free preflight before opening the disposable token;
- disposable execution followed immediately by independent validation; and
- refusal if the output root already exists.

The authorized roots are:

```text
source = /scratch/project_462000131/anisrahm/hmsc-hpc-sources/generative_iid_v2_940d73d
run = /scratch/project_462000131/anisrahm/hmsc-hpc-runs/generative_iid_v2_disposable_940d73d_20260801
```

## Required Validation

Completion is not accepted from scheduler exit status alone. The downloaded
run must independently establish:

1. exact `freeze.json` inventory and artifact byte/hash agreement;
2. all 18 training and 18 validation corpus fingerprints;
3. v2 checkpoint manifest, weight hash, schema, architecture, source, and
   absence of calibration or external candidate dependencies;
4. finite training loss, IWELBO, gradient norm, validation loss, and invariant
   outputs;
5. exactly four validation refinement steps;
6. deterministic validation loss/IWELBO replay;
7. exact-target replay for the first validation community;
8. nonzero optimizer movement from seeded initialization; and
9. `production_511m_opened = false`,
   `fixed_validation_512m_opened = false`, and
   `reserved_513m_515m_opened = false` in every relevant artifact.

## Prohibitions

This authorization does not open 511M-515M, real data, production training,
fixed validation, or reserved evaluation. Disposable results may identify
implementation, numerical, artifact, or scheduler defects only. They may not
tune architecture, posterior family, refinement, loss, schedule, gates, or
thresholds.

After this one execution, the token returns to unset. Any retry requires a new
explicit authorization record after the failed attempt is inspected.
