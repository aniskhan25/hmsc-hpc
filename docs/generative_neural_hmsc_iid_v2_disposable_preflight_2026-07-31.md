# Generative Neural-HMSC IID v2 Disposable Preflight

Date: 2026-07-31

Protocol: `generative_neural_hmsc_iid_probit_v2_orbit`

## Result

The token-free, no-seed disposable preflight passed against clean harness
commit:

`ba48fc93c53447cf4277f9c15946bb95f00d332e`

Command:

```text
python examples/run_generative_neural_hmsc_iid_v2.py \
  --mode preflight \
  --expected-source-commit ba48fc93c53447cf4277f9c15946bb95f00d332e
```

The process exited zero and reported:

```text
status = generative_iid_v2_disposable_preflight_sealed
factorial_cell_count = 18
factorial_unique_cell_count = 18
smoke_epochs = 2
model_seed = 511900001
simulation_generation_called = false
output_created = false
disposable_seed_ranges_opened = false
production_511m_opened = false
fixed_validation_512m_opened = false
reserved_513m_515m_opened = false
authorization_required = true
```

No `OPEN_GENERATIVE_IID*` variable was present. The confirmation token was not
set by the preflight.

## Source Inventory

The clean preflight independently reported:

```text
a7885c9123ac4e52beb1ed366fd5c09857f132789e21cac540be6c96663b8d52  pyhmsc/neural/generative_iid.py
558e40a6e98639899588f56c42f7595b81c9e05c34467a87ec5502eca794ee7c  pyhmsc/neural/generative_iid_mcmc.py
155292c9f8edc027ae64f1b1a0046998927308be24889662dab1b90bd48a4cbb  pyhmsc/neural/generative_iid_v2.py
2b950e83c6166352bbe7cc0e0e9baee709a6b157f016f18966539634ff2dac6d  pyhmsc/neural/generative_iid_v2_artifact.py
4a3125dea1733e4b386b3634a2c7f293bec828b37b6f18e628d612b04671468d  pyhmsc/neural/__init__.py
209743be767459721ae809d59bba30efb77f4394a52e8934e9b06ed869a4b7fa  examples/run_generative_neural_hmsc_iid_v2.py
a2eaee0441833167f707f7cb9ae6b1162ba4e118ee3dfc1a245983cc9ada24c2  docs/generative_neural_hmsc_iid_v2_orbit_preregistration_2026-07-31.md
9a463943508651e74855701cdbd9870961efd3fd3c07a444674da36a67d49344  docs/generative_neural_hmsc_iid_v2_seed_reaudit_2026-07-31.json.md
13041f6368eeaa64d4eae4446782c99c7a0b8af2a13bb13be9a69bec040df7ea  docs/generative_neural_hmsc_iid_v2_representation_decision_2026-07-31.md
0d54f04ea5ec5c654df73594b7ff6614157152ec87bdfc3ecfd09c2401550cab  docs/generative_neural_hmsc_iid_v2_implementation_2026-07-31.md
```

Every frozen document hash matched. The source worktree was clean and the full
commit matched the requested commit.

## Interpretation

This establishes that the reviewed harness is executable from a clean source
state without crossing a seed or output boundary. It does not establish model
optimization, statistical recovery, calibration, MCMC agreement, or
qualification.

The disposable smoke remains separately gated. The 593M-594M blocks and every
511M-515M block remain unopened.

## Next Step

Conduct a separate authorization decision for only the 593M-594M disposable
smoke. If authorized, the execution must pin a clean reviewed commit and use
only:

```text
OPEN_GENERATIVE_IID_V2_593M_594M_DISPOSABLE_SMOKE=\
GENERATE_593M_594M_DISPOSABLE_ONLY
```

After completion, independently validate `freeze.json`, all 36 corpus
fingerprints, checkpoint manifest and weights, optimization diagnostics,
validation IWELBO, exact-target replay, optimizer movement, source provenance,
and all 511M-515M seal booleans. Disposable results may expose implementation
or scheduler defects only and may not tune the frozen candidate.
