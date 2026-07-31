# Generative Neural-HMSC IID v2 Disposable Harness Review

Date: 2026-07-31

Protocol: `generative_neural_hmsc_iid_probit_v2_orbit`

## Scope

This review covers the sealed 593M-594M disposable harness implementation. It
does not authorize or execute the smoke. No 593M-594M, 511M-512M, or
513M-515M simulation dataset was generated.

## Frozen Disposable Contract

The harness fixes:

- training seeds `593000001-593000018`;
- masked validation seeds `594000001-594000018`;
- the same 18 unique shape/covariate/loading/prevalence cells used by the v1
  disposable factorial;
- model seed `511900001`;
- exactly two disposable smoke epochs;
- the frozen v2 eight-draw objective and four-step refinement;
- one exact confirmation:
  `OPEN_GENERATIVE_IID_V2_593M_594M_DISPOSABLE_SMOKE=GENERATE_593M_594M_DISPOSABLE_ONLY`.

There is no command, function, or seed owner for production 511M, fixed
validation 512M, or reserved 513M-515M execution.

## Fail-Closed Boundaries

`examples/run_generative_neural_hmsc_iid_v2.py` provides only:

1. `check-seal`, which refuses any generative opening token;
2. `preflight`, which also refuses every opening token and requires a clean,
   full, pinned commit;
3. `disposable-smoke`, which requires the exact disposable token and rejects
   every other generative opening token; and
4. `validate-disposable`, which retains the same exact-token and clean-source
   boundary because it regenerates the already-opened disposable corpus.

Both executable paths validate all frozen v2 document hashes and the clean
source commit before output creation or simulation generation.

The token-free preflight calls no simulation function and creates no output.
It reports all seed-open booleans as false.

## Artifact and Replay Contract

An authorized run will:

- fingerprint all 18 training and 18 validation datasets from tensor bytes;
- train and save only the v2 candidate checkpoint;
- record finite optimization, validation, invariant, refinement, and
  exact-target checks;
- freeze the corpus manifest, report, checkpoint manifest, and weight hashes
  with exact byte counts;
- record that disposable seeds opened while 511M-515M remained sealed; and
- require independent replay of all 36 corpus fingerprints, validation IWELBO,
  exact target, checkpoint hashes, and optimizer movement.

Unknown output artifacts, missing files, changed hashes, changed seed roles,
non-finite diagnostics, absent refinement, source mismatch, and unchanged model
weights all fail validation.

Disposable metrics have no promotion thresholds and cannot change the
architecture, posterior, objective, schedule, gates, or later qualification
criteria.

## Verification

The implementation and unchanged v2 feasibility suite passed together:

```text
pytest -q --runslow \
  tests/test_neural_hmsc_generative_iid_v2.py \
  tests/test_neural_hmsc_generative_iid_v2_harness.py
19 passed
```

The token-free seal command returned:

```text
status = generative_iid_v2_disposable_sealed
disposable_seed_ranges_opened = false
production_511m_opened = false
fixed_validation_512m_opened = false
reserved_513m_515m_opened = false
confirmation_present = false
```

The reviewed source hashes are:

```text
209743be767459721ae809d59bba30efb77f4394a52e8934e9b06ed869a4b7fa  examples/run_generative_neural_hmsc_iid_v2.py
1e6ff9c14dde510f58e66a4b7967cf933357d4b79d162149fe2b1a271a3e5a5c  tests/test_neural_hmsc_generative_iid_v2_harness.py
```

## Decision

The sealed harness is ready for a clean-commit, token-free preflight. The
disposable smoke remains unauthorized.

After this implementation is committed, run:

```text
python examples/run_generative_neural_hmsc_iid_v2.py \
  --mode preflight \
  --expected-source-commit <full-clean-commit>
```

Only a complete preflight pass may support a later, separate authorization
decision. This review does not set the confirmation token.
