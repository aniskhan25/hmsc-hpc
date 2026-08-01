# Generative Neural-HMSC IID v2 Repaired Preflight

Date: 2026-08-01

Protocol: `generative_neural_hmsc_iid_probit_v2_orbit`

Decision: `repaired_boundary_preflight_passed_no_seed_opened`

## Frozen Boundary

The repaired harness boundary is the clean commit:

```text
cca9e97518e77c5ca958dfdc3bee753997ed7ac5
```

It was packaged as an isolated Git archive:

```text
archive_sha256 = bb343bcef927455b5ffedb0483015f75f3da053176d58c1f032b3fece7790eb1
archive_bytes = 3802838
```

The dedicated token-free scheduler is
`docs/lumi_generative_neural_hmsc_iid_v2_repaired_preflight_sbatch.sh`,
SHA-256
`be897437cc97961cfcd8de91b4ddd30cf091154e0eeb2ce1d34aa855a20a2f04`.
It contains no confirmation token and rejects every nonempty
`OPEN_GENERATIVE_IID*` environment variable.

## Local Qualification

The complete local v2 implementation, harness, authorization, numerical, slow
maximum-shape, and mixed-shape training suite passed `30/30` tests. The clean
commit's local token-free preflight returned
`generative_iid_v2_disposable_preflight_sealed`.

## LUMI Preflight

- job: `20520889`
- partition: `dev-g`
- state: `COMPLETED`
- elapsed: `00:00:28`
- exit code: `0:0`
- output SHA-256:
  `a9f3c3f0f535f31217da279f9907f8c1d0fcf11001a7337ffbb4a4fdade9fe6f`
- stdout SHA-256:
  `014bb3583ce903b3706a872515a0f4f851d45e702a62dc37c3ed526057d397d2`
- stderr SHA-256:
  `2d2a618696d1ea6b6281894b1d8e6469dc8b4f0eedb3a5cfdbd8610ae6ce61c1`

The downloaded LUMI JSON is byte-identical to the local preflight JSON. Its 11
source records were independently checked against the clean local commit for
path, byte count, and SHA-256. Important inventory entries are:

```text
87828857ee1718a8825a1a15e7af99abe49a86ee4d179f6cbce6591162aa71bc  pyhmsc/neural/generative_iid_v2.py
b09e3e1eb743fe62a509876a284323e5bb151ce043cac7b60dba8a9a35f9300e  examples/run_generative_neural_hmsc_iid_v2.py
e3b708a09b0c920676e592759f44f7457cc75decff66138b6b29c509254a6192  docs/generative_neural_hmsc_iid_v2_numerical_review_2026-08-01.md
```

The preflight independently reports:

- `simulation_generation_called = false`;
- `output_created = false`;
- `disposable_seed_ranges_opened = false`;
- `production_511m_opened = false`;
- `fixed_validation_512m_opened = false`; and
- `reserved_513m_515m_opened = false`.

## Decision Boundary

The repaired source inventory and token-free/no-seed preflight pass. This does
not authorize disposable simulation, training, validation, production, or
reserved evaluation. The prior failed disposable runs are not converted into a
pass. Blocks 593M-594M and 511M-515M remain sealed after this preflight.

The next action must be a separate bounded authorization decision about one
final 593M-594M disposable verification pinned to the clean repaired commit.
No 511M-515M opening may be considered until a complete disposable artifact
and independent replay pass every frozen requirement.
