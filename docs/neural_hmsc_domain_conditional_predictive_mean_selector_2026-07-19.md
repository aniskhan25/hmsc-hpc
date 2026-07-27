# Domain-Conditional Predictive Mean Selector

Date: 2026-07-19

Purpose: keep the useful Big Spatial transfer movement from
`probit_response_affine` while falling back to the promoted scale-only
predictive path on Whittaker-like source contexts.

## Implementation

Added selector helpers in `pyhmsc/neural/mean_calibration.py`:

- `domain_conditional_predictive_mean_selector_metadata`
- `select_predictive_mean_calibration_for_context`

The selector stores the response-affine candidate in predictive-only metadata
but applies it only for transfer-like contexts. Current context policy:

- source-like contexts: `whittaker`, `whittaker_source`, `source_like`
- transfer-like contexts: `big_spatial`, `big_spatial_transfer`, `transfer_like`
- unknown contexts: identity fallback

Whittaker runner behavior with
`--predictive-mean-selection-policy domain_conditional`:

- fit the response-affine candidate exactly as before
- write the scale-only predictive artifact
- write `neural_predictive_distribution.h5` as the final selector artifact
- for Whittaker/source-like context, final predictive samples are identical to
  scale-only samples
- store `predictive_mean_selector` and
  `predictive_mean_selector_decision` metadata so transfer workflows can use
  the candidate later

Big Spatial transfer behavior:

- read `predictive_mean_selector` from the frozen Whittaker predictive artifact
- select context `big_spatial_transfer`
- apply the response-affine candidate only if the selector marks that context
  active
- otherwise fall back to scale-only

The coefficient posterior, coefficient calibration, SBC/OOD gates,
rare-validation gates, and Python-only/R-boundary parity semantics are
unchanged.

## Local Smoke

Whittaker smoke:

```text
/private/tmp/neural_whittaker_domain_selector_smoke2
```

Selector decision:

```json
{
  "context": "whittaker",
  "method": "domain_conditional_context_selector",
  "action": "identity",
  "selected": false,
  "reason": "source_like_context_uses_identity",
  "context_family": "source_like"
}
```

Whittaker scale-only and selector-final metrics were identical:

| model | Brier | log loss |
| --- | ---: | ---: |
| `neural_predictive_only_calibrated` | `0.076054` | `0.273493` |
| `neural_predictive_mean_calibrated` | `0.076054` | `0.273493` |

Big Spatial transfer smoke:

```text
/private/tmp/neural_big_spatial_domain_selector_smoke
```

Selector decision:

```json
{
  "context": "big_spatial_transfer",
  "method": "domain_conditional_context_selector",
  "action": "apply_candidate",
  "selected": true,
  "reason": "active_transfer_like_context",
  "context_family": "transfer_like",
  "candidate_method": "probit_response_affine",
  "candidate_slope": 1.25,
  "candidate_intercept": -0.05
}
```

Big Spatial improved relative to scale-only in the smoke:

| model | Brier | log loss |
| --- | ---: | ---: |
| `neural_predictive_only_calibrated` | `0.049501` | `0.196157` |
| `neural_predictive_mean_calibrated` | `0.048095` | `0.187991` |

The smoke promotion gate passed:

| dataset | passed | Brier ratio | log-loss ratio |
| --- | --- | ---: | ---: |
| `whittaker` | yes | `1.0000` | `1.0000` |
| `big_spatial` | yes | `0.9716` | `0.9584` |

## Decision

The domain-conditional selector works mechanically and satisfies the local
cross-dataset no-degradation gate. This is not yet production evidence because
the smoke used tiny local training/SBC/MCMC settings and a temporary
`/private/tmp` source-acceptance override only to exercise the dependent Big
Spatial transfer path.

Next step: run the production-shape Whittaker plus dependent Big Spatial LUMI
validation with `PREDICTIVE_MEAN_SELECTION_POLICY=domain_conditional`, the same
parity metrics as the previous real-data run, and the frozen cross-dataset
promotion gate as the acceptance decision.

## Production-Shape Submission

Submitted on 2026-07-19 after syncing the selector implementation, updated
real-data runners, promotion-gate CLI, and sbatch wrappers to LUMI.

Remote verification used the TensorFlow venv:

```text
/scratch/project_462000131/anisrahm/venvs/hmsc_tf_env/bin/python3
```

Checks passed:

- `bash -n docs/lumi_neural_hmsc_whittaker_sbatch.sh docs/lumi_neural_hmsc_big_spatial_transfer_sbatch.sh`
- `py_compile` for the selector module, promotion gate CLI, and both real-data runners
- selector imports under the TensorFlow venv

Whittaker job:

- Job: `20008206`
- State at submission check: running on `dev-g`
- Run root:
  `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_whittaker_domain_selector_realdata_20260719`
- Predictive mean calibration: `probit_response_affine`
- Predictive mean selection policy: `domain_conditional`
- Predictive mean validation datasets: `128`
- Predictive mean minimum improvement: `0.0001`
- Whittaker parity metrics:
  `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/whittaker_r_python_parity_scaled_20260718_082539/whittaker_r_python_parity_metrics.json`

Big Spatial job:

- Job: `20008208`
- Dependency: `afterok:20008206`
- State at submission check: pending on dependency
- Run root:
  `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_big_spatial_domain_selector_realdata_20260719`
- Frozen Whittaker source:
  `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_whittaker_domain_selector_realdata_20260719`
- Big Spatial parity metrics:
  `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/direct_r_python_big_spatial_full_parity_20260719/big_spatial_plants_validation_model_spatial_full/direct_r_python_parity_metrics.json`

Next action: monitor jobs `20008206` and `20008208`; after completion,
download held-out metrics, acceptance JSON, reports, metadata, and wall-time
files, then run `examples/evaluate_neural_hmsc_predictive_promotion.py` on the
Whittaker and Big Spatial held-out metric CSVs.

## Production-Shape Result

Both production-shape LUMI jobs completed successfully:

- Whittaker job `20008206`: `COMPLETED`, elapsed `00:10:28`, exit `0:0`
- Big Spatial job `20008208`: `COMPLETED`, elapsed `00:01:50`, exit `0:0`

Local result copy:
`/private/tmp/neural_domain_selector_realdata_20008206_20008208`

Whittaker selector decision:

```json
{
  "context": "whittaker",
  "method": "domain_conditional_context_selector",
  "action": "identity",
  "selected": false,
  "reason": "source_like_context_uses_identity",
  "context_family": "source_like"
}
```

Big Spatial selector decision:

```json
{
  "context": "big_spatial_transfer",
  "method": "domain_conditional_context_selector",
  "action": "apply_candidate",
  "selected": true,
  "reason": "active_transfer_like_context",
  "context_family": "transfer_like",
  "candidate_method": "probit_response_affine",
  "candidate_slope": 1.025,
  "candidate_intercept": 0.025
}
```

Acceptance:

- Whittaker: coefficient SBC, held-out predictive, combined qualification, and
  reference parity gates passed.
- Big Spatial: inherited source SBC, inherited source qualification, target
  predictive, frozen transfer, and reference parity gates passed.

Held-out predictive comparison against scale-only:

| dataset | Brier ratio | log-loss ratio | macro AUC ratio | prevalence MAE ratio | richness MAE ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `whittaker` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| `big_spatial` | `0.9962` | `0.9944` | `1.0050` | `0.9729` | `0.9823` |

The frozen cross-dataset promotion gate passed:

```text
/private/tmp/neural_domain_selector_realdata_20008206_20008208/promotion_gate/predictive_mean_promotion_gate.md
```

Gate rows:

| dataset | passed | Brier ratio | log-loss ratio |
| --- | --- | ---: | ---: |
| `whittaker` | yes | `1.0000` | `1.0000` |
| `big_spatial` | yes | `0.9962` | `0.9944` |

Qualified Python MCMC remains stronger on core proper scores for both real
datasets. The selector is therefore a predictive-transfer improvement over the
promoted scale-only neural path, not evidence of neural superiority over the
qualified Python-only HMSC comparator.

## Production Decision

The domain-conditional selector qualifies under the frozen real-data
cross-dataset no-degradation gate. It should be considered a promotable
predictive-only improvement over scale-only `external_monotone` for the current
two-dataset real-data gate.

Because this is a context rule rather than a newly trained neural posterior,
the remaining decision is policy rather than mechanics: either promote the
selector as the default predictive-mean deployment policy now, or run a bounded
three-seed real-data sensitivity confirmation first, using the same frozen
gate and parity attachments.

## Promotion Policy Decision

Decision: run a bounded three-seed real-data sensitivity confirmation before
changing the default predictive-mean deployment policy.

Rationale:

- The selector passed the production-shape two-dataset gate, but only for one
  seed.
- The earlier response-affine evidence was mixed across datasets, so the
  context rule should not be promoted from a single paired real-data run.
- The selector is cheap enough to evaluate on a bounded three-seed workflow.
- The default sbatch environment remains `PREDICTIVE_MEAN_SELECTION_POLICY=apply_selected`
  unless the confirmation run explicitly requests `domain_conditional`.

Confirmation scope:

- Seeds: use the same bounded real-data sensitivity pattern as the promoted
  `external_monotone` confirmation.
- Whittaker: attach Whittaker R/Python parity metrics and run
  `PREDICTIVE_MEAN_CALIBRATION=probit_response_affine`,
  `PREDICTIVE_MEAN_SELECTION_POLICY=domain_conditional`,
  `PREDICTIVE_MEAN_CALIBRATION_VALIDATION_DATASETS=128`, and
  `PREDICTIVE_MEAN_CALIBRATION_MIN_IMPROVEMENT=0.0001`.
- Big Spatial: run dependent frozen transfer for each Whittaker seed with Big
  Spatial R/Python parity metrics attached.
- Acceptance: every seed pair must pass existing Whittaker/Big Spatial gates
  and the frozen cross-dataset no-degradation gate.

If the bounded confirmation passes, promote the domain-conditional selector
policy, not raw `probit_response_affine`, as the default predictive-mean
deployment policy.

## Bounded Three-Seed Sensitivity Submission

Submitted on 2026-07-19 after updating the existing real-data sensitivity
harness to run the domain-conditional selector and write per-seed promotion
gate artifacts.

Remote verification passed under the TensorFlow venv:

- `bash -n docs/lumi_neural_hmsc_realdata_sensitivity_sbatch.sh`
- `py_compile` for the aggregation, promotion-gate, Whittaker, Big Spatial,
  selector, and predictive-selection modules
- focused pytest:
  `tests/test_neural_hmsc_realdata_sensitivity.py`
  `tests/test_neural_hmsc_predictive_selection.py`

LUMI job:

- Job: `20010991`
- Initial state: running on `dev-g`
- Run root:
  `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_domain_selector_realdata_sensitivity_20260719`
- Seeds: `20260721 20260722 20260723`
- Predictive mean calibration: `probit_response_affine`
- Predictive mean selection policy: `domain_conditional`
- Predictive mean validation datasets: `128`
- Cross-dataset promotion gate thresholds: Brier ratio `<= 1.0`, log-loss
  ratio `<= 1.0`

Expected outputs include per-seed Whittaker and Big Spatial held-out metrics,
acceptance JSON, selector decisions, `promotion_gate/` JSON/CSV/Markdown, and
`realdata_sensitivity.{csv,json,md}` at the run root.

Next action: monitor job `20010991`; after completion, download and inspect the
aggregate report, then make the final promotion decision for
`domain_conditional`.

## Bounded Three-Seed Sensitivity Result

Job `20010991` completed successfully:

- State: `COMPLETED`
- Elapsed: `00:34:09`
- Exit code: `0:0`
- Local result copy:
  `/private/tmp/neural_domain_selector_realdata_sensitivity_20010991/neural_domain_selector_realdata_sensitivity_20260719`
- Run-root wall time: `2000` seconds

Aggregate files inspected:

- `realdata_sensitivity.csv`
- `realdata_sensitivity.json`
- `realdata_sensitivity.md`
- per-seed `promotion_gate/predictive_mean_promotion_gate.{json,csv,md}`

Aggregate decision from the frozen report:

```text
Decision: inspect_seed_level_diagnostics
Completed rows: 6 / 6
Paired pass count: 3
Paired promotion-gate pass count: 2
Paired MCMC advantage count: 3
```

All Whittaker and Big Spatial dataset acceptance gates passed. The failure is
limited to the stricter cross-dataset predictive-mean no-degradation gate:

| seed | Whittaker selector | Big Spatial selector | promotion gate | Big Spatial Brier ratio | Big Spatial log-loss ratio |
| ---: | --- | --- | --- | ---: | ---: |
| `20260721` | identity | apply candidate | pass | `0.996873` | `0.994787` |
| `20260722` | identity | apply candidate | pass | `0.997322` | `0.995501` |
| `20260723` | identity | apply candidate | fail | `1.004801` | `1.004174` |

Seed `20260723` failed because the candidate worsened Big Spatial Brier and
log-loss relative to the promoted scale-only baseline:

```text
big_spatial: Brier ratio 1.0048 exceeds 1
big_spatial: log-loss ratio 1.00417 exceeds 1
```

The source-side response-affine fit selected a nonzero candidate in every seed:

| seed | slope | intercept | validation Brier gain | validation log-loss gain |
| ---: | ---: | ---: | ---: | ---: |
| `20260721` | `1.025` | `0.025` | `0.0000085` | `0.0001888` |
| `20260722` | `1.025` | `0.025` | `0.0000238` | `0.0002487` |
| `20260723` | `1.025` | `0.050` | `0.0000877` | `0.0003143` |

This shows the source validation gain is not sufficient as a transfer
promotion signal. The selector is mechanically useful and often improves Big
Spatial, but it does not satisfy the frozen every-seed no-degradation rule.

Promotion decision: do not promote `domain_conditional` as the default
predictive-mean deployment policy. Keep the promoted scale-only
`external_monotone` path as default, keep `domain_conditional` experimental,
and require a stronger transfer-stability criterion before revisiting default
promotion.

Next roadmap step: add a conservative transfer-stability guard for predictive
mean calibration, using source validation gain margin and candidate amplitude
constraints, then rerun the bounded three-seed real-data sensitivity check only
if local/fixed simulated gates still hold. The guard should prefer identity
unless evidence for response-affine movement is materially above the current
near-zero validation improvements.

## Transfer-Stability Guard Implementation

Implemented a conservative second-stage guard for
`domain_conditional` selector metadata. The response-affine candidate may still
be fitted and selected by source validation, but transfer-like contexts now
apply it only if all transfer-stability checks pass:

- validation Brier gain is at least `0.0001`
- validation log-loss gain is at least `0.0005`
- absolute slope movement from `1.0` is at most `0.05`
- absolute intercept movement is at most `0.025`

The guard is stored under `transfer_stability_guard` in selector metadata with
thresholds, measured gains, candidate movement, pass/fail state, and explicit
failure reasons. Transfer decisions now return
`candidate_failed_transfer_stability_guard` when a transfer-like context is
blocked by the guard.

The Whittaker runner exposes configurable guard flags:

- `--predictive-mean-transfer-min-brier-gain`
- `--predictive-mean-transfer-min-log-loss-gain`
- `--predictive-mean-transfer-max-slope-delta`
- `--predictive-mean-transfer-max-abs-intercept`

The Whittaker and bounded real-data sensitivity LUMI sbatch wrappers expose
matching environment variables and print the active guard settings.

Focused validation passed:

- `python3 -m py_compile pyhmsc/neural/mean_calibration.py examples/run_neural_hmsc_whittaker.py`
- `bash -n docs/lumi_neural_hmsc_whittaker_sbatch.sh docs/lumi_neural_hmsc_realdata_sensitivity_sbatch.sh`
- `python3 -m pytest tests/test_neural_hmsc_mean_calibration.py -q`
- `python3 -m pytest tests/test_neural_hmsc_realdata_sensitivity.py tests/test_neural_hmsc_predictive_selection.py -q`

Next action: run a local or fixed-simulated guard sanity check confirming that
the previous near-zero-gain candidates are withheld from transfer deployment
without changing coefficient-posterior calibration, Whittaker source identity,
or scale-only predictive outputs.

## Local Transfer-Stability Guard Sanity

Ran a local fixed replay against the downloaded three-seed LUMI sensitivity
artifacts:

```text
/private/tmp/neural_domain_selector_realdata_sensitivity_20010991/neural_domain_selector_realdata_sensitivity_20260719
```

The replay reconstructed the new selector metadata from each seed's stored
Whittaker `predictive_mean_calibration_metadata`, then evaluated selector
decisions for `whittaker` and `big_spatial_transfer` without rewriting any
posterior artifacts or rerunning calibration.

Guard defaults:

| threshold | value |
| --- | ---: |
| minimum validation Brier gain | `0.0001` |
| minimum validation log-loss gain | `0.0005` |
| maximum slope delta | `0.05` |
| maximum absolute intercept | `0.025` |

Replay result:

| seed | source action | transfer action | Brier gain | log-loss gain | intercept | guarded transfer ratios |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| `20260721` | identity | identity | `0.0000246` | `0.0002135` | `0.025` | Brier `1.0000`, log loss `1.0000` |
| `20260722` | identity | identity | `0.0000295` | `0.0002264` | `0.025` | Brier `1.0000`, log loss `1.0000` |
| `20260723` | identity | identity | `0.0000417` | `0.0002031` | `0.050` | Brier `1.0000`, log loss `1.0000` |

All three prior response-affine candidates had been selected by the source
validation fit, but all three are now withheld from transfer deployment because
their validation gains are below the practical margins. Seed `20260723` is also
blocked by the intercept cap. The guarded identity fallback would avoid the
previous Big Spatial degradation in seed `20260723`.

The sanity check also verified that a strong synthetic transfer candidate still
applies when it satisfies the guard: slope `1.025`, intercept `0.020`, Brier
gain `0.0010`, log-loss gain `0.0010`, action `apply_candidate`.

Conclusion: the guard fixes the observed no-degradation failure mode
mechanically and preserves Whittaker source identity. It also reveals that the
previous real-data candidates are too weak to justify transfer movement under
the new criterion, so any next real-data promotion check must distinguish
"safe identity fallback" from an actual predictive improvement.

## Tightened Promotion Reporting

Updated the bounded real-data sensitivity aggregator so guarded-selector
promotion reporting separates:

- `genuine_transfer_improvement`: transfer-like context applied the candidate
  and both Brier/log-loss ratios are below `1.0`
- `safe_identity_fallback`: transfer-like context used identity and preserved
  Brier/log-loss ratios at or below `1.0`
- `applied_degradation`: transfer-like context applied the candidate and
  degraded Brier or log loss
- non-transfer source rows: `not_transfer_context`

The aggregate promotion decision now requires:

- all paired dataset acceptance gates pass
- all paired no-degradation promotion gates pass
- at least two Big Spatial transfer seeds show genuine transfer improvement

This prevents a guarded selector from qualifying merely by behaving identically
to the promoted scale-only default.

Focused validation passed:

- `python3 -m py_compile examples/aggregate_neural_hmsc_realdata_sensitivity.py`
- `bash -n docs/lumi_neural_hmsc_realdata_sensitivity_sbatch.sh`
- `python3 -m pytest tests/test_neural_hmsc_realdata_sensitivity.py tests/test_neural_hmsc_predictive_selection.py -q`

Replayed the tightened aggregator on the downloaded unguarded sensitivity run:

```text
python3 examples/aggregate_neural_hmsc_realdata_sensitivity.py \
  --run-root /private/tmp/neural_domain_selector_realdata_sensitivity_20010991/neural_domain_selector_realdata_sensitivity_20260719 \
  --seeds 20260721 20260722 20260723 \
  --output-prefix /private/tmp/neural_domain_selector_realdata_sensitivity_20010991/tightened_reporting_check \
  --strict
```

The tightened replay correctly reported:

```text
Decision: inspect_seed_level_no_degradation
Paired pass count: 3
Paired promotion-gate pass count: 2
Paired genuine transfer-improvement count: 2
Paired safe identity-fallback count: 0
```

Per-seed transfer outcomes for the unguarded run:

| seed | transfer outcome | Brier ratio | log-loss ratio |
| ---: | --- | ---: | ---: |
| `20260721` | `genuine_transfer_improvement` | `0.996873` | `0.994787` |
| `20260722` | `genuine_transfer_improvement` | `0.997322` | `0.995501` |
| `20260723` | `applied_degradation` | `1.004801` | `1.004174` |

The unit tests also cover the guarded identity-only case: if all three Big
Spatial seeds pass no-degradation only through identity fallback, the aggregate
decision is `safe_identity_fallback_not_promotable`.

## Guarded Bounded Sensitivity Submission

Submitted the guarded bounded three-seed real-data sensitivity run on
2026-07-20 after syncing the guarded selector, tightened aggregator, Whittaker
runner, Big Spatial runner, promotion gate CLI, sbatch wrappers, and focused
tests to LUMI.

Local validation passed:

- `python3 -m py_compile pyhmsc/neural/mean_calibration.py examples/run_neural_hmsc_whittaker.py examples/aggregate_neural_hmsc_realdata_sensitivity.py examples/evaluate_neural_hmsc_predictive_promotion.py`
- `bash -n docs/lumi_neural_hmsc_realdata_sensitivity_sbatch.sh docs/lumi_neural_hmsc_whittaker_sbatch.sh`
- `python3 -m pytest tests/test_neural_hmsc_mean_calibration.py tests/test_neural_hmsc_realdata_sensitivity.py tests/test_neural_hmsc_predictive_selection.py -q`
- `git diff --check`

Remote LUMI validation passed under the TensorFlow venv:

- `bash -n` for the two sbatch wrappers
- `py_compile` for selector, predictive-selection, runners, aggregator, and
  promotion gate CLI
- focused pytest for mean calibration, real-data sensitivity aggregation, and
  predictive selection

LUMI job:

- Job: `20020582`
- Initial state: running on `dev-g`
- Run root:
  `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_domain_selector_guarded_realdata_sensitivity_20260720`
- Seeds: `20260721 20260722 20260723`
- Predictive mean calibration: `probit_response_affine`
- Predictive mean selection policy: `domain_conditional`
- Transfer-stability guard: default thresholds
- Tightened reporting: enabled through
  `examples/aggregate_neural_hmsc_realdata_sensitivity.py`

Next action: monitor job `20020582`; when it completes, download the run root
and inspect `realdata_sensitivity.{csv,json,md}` plus per-seed
`promotion_gate/` outputs. The final decision must distinguish genuine
transfer improvement from safe identity fallback.

## Guarded Bounded Sensitivity Result

Job `20020582` completed successfully:

- State: `COMPLETED`
- Elapsed: `00:31:41`
- Exit code: `0:0`
- Run-root wall time: `1880` seconds
- Local result copy:
  `/private/tmp/neural_domain_selector_guarded_realdata_sensitivity_20020582/neural_domain_selector_guarded_realdata_sensitivity_20260720`

Inspected:

- `realdata_sensitivity.csv`
- `realdata_sensitivity.json`
- `realdata_sensitivity.md`
- per-seed `promotion_gate/predictive_mean_promotion_gate.{json,csv,md}`
- per-seed Big Spatial selector metadata and guard failure reasons

Aggregate decision:

```text
Decision: safe_identity_fallback_not_promotable
Completed rows: 6 / 6
Paired pass count: 3
Paired promotion-gate pass count: 3
Paired genuine transfer-improvement count: 0
Paired safe identity-fallback count: 3
Paired MCMC advantage count: 3
```

Per-seed Big Spatial transfer outcome:

| seed | selector action | transfer outcome | promotion gate | Brier ratio | log-loss ratio |
| ---: | --- | --- | --- | ---: | ---: |
| `20260721` | identity | `safe_identity_fallback` | pass | `1.0000` | `1.0000` |
| `20260722` | identity | `safe_identity_fallback` | pass | `1.0000` | `1.0000` |
| `20260723` | identity | `safe_identity_fallback` | pass | `1.0000` | `1.0000` |

Guard failure reasons:

| seed | Brier gain | log-loss gain | intercept | failure reasons |
| ---: | ---: | ---: | ---: | --- |
| `20260721` | `0.0000246` | `0.0002135` | `0.025` | Brier gain below margin; log-loss gain below margin |
| `20260722` | `0.0000295` | `0.0002264` | `0.025` | Brier gain below margin; log-loss gain below margin |
| `20260723` | `0.0000417` | `0.0002031` | `0.050` | Brier gain below margin; log-loss gain below margin; intercept above cap |

Interpretation:

- The guarded selector now satisfies the no-degradation gate.
- It does so entirely by falling back to the current promoted scale-only
  default.
- There is no nonzero transfer improvement under the guarded policy.
- Therefore `domain_conditional` is not promotable as a default
  predictive-mean deployment policy.

Decision: keep `external_monotone` scale-only as the default predictive path.
Keep `domain_conditional` and `probit_response_affine` experimental. Do not
rerun this guarded selector family on real data without a new candidate that
can produce a genuine transfer-improvement signal under the tightened gate.

Next roadmap step: redesign the predictive-mean candidate itself using
transfer-aware or multi-domain validation rather than tightening source-only
affine selection. The candidate should be evaluated locally/fixed-sim first and
must show both no-degradation and nonzero transfer improvement before another
real-data LUMI sensitivity run.

## Transfer-Aware Predictive-Mean Candidate

Implemented a new predictive-only candidate:

```text
probit_transfer_response_affine
```

This stops tuning the source-only `probit_response_affine` selector and moves
selection into a multi-domain response-scale validation problem. The new fitter
chooses a bounded affine movement using source calibration plus transfer-like
validation batches, then selects the candidate only when:

- source-like validation does not degrade Brier or log loss
- transfer-like validation does not degrade Brier or log loss
- transfer-like validation improves Brier plus log loss by a practical margin

Implementation points:

- `BetaResponseCalibrationBatch` stores transfer-like response validation
  batches.
- `fit_beta_transfer_response_mean_calibration` fits the new candidate.
- `BetaPredictiveMeanCalibration` metadata now optionally records
  `transfer_response_validation` separately from source `response_validation`.
- The existing transfer-stability guard uses transfer-validation gains when
  available, falling back to source-validation gains for older artifacts.
- `examples/run_neural_hmsc_benchmark.py` exposes
  `--predictive-mean-calibration probit_transfer_response_affine` for compact
  and production-shape simulated fixed-effect benchmarks.

Local validation passed:

- `python3 -m py_compile pyhmsc/neural/mean_calibration.py pyhmsc/neural/__init__.py examples/run_neural_hmsc_benchmark.py`
- `python3 -m pytest tests/test_neural_hmsc_mean_calibration.py -q`
- `python3 examples/run_neural_hmsc_benchmark.py --help`

Tiny end-to-end smoke:

```text
/private/tmp/neural_transfer_mean_candidate_smoke
```

Smoke command:

```text
python3 examples/run_neural_hmsc_benchmark.py \
  --output /private/tmp/neural_transfer_mean_candidate_smoke \
  --suite probit \
  --n-sites 12 \
  --n-species 2 \
  --train-datasets 4 \
  --calibration-datasets 3 \
  --predictive-mean-calibration probit_transfer_response_affine \
  --predictive-mean-calibration-validation-datasets 3 \
  --predictive-mean-calibration-min-improvement 0.0001 \
  --epochs 2 \
  --batch-size 2 \
  --neural-chains 1 \
  --neural-draws 8 \
  --sbc-datasets 0 \
  --ood-regimes effect_size_shift combined_shift
```

Smoke metadata confirmed the new method stayed predictive-only and recorded
separate source/transfer response scores:

| field | value |
| --- | --- |
| method | `probit_transfer_response_affine` |
| selected | `true` |
| slope | `1.15` |
| intercept | `0.025` |
| source Brier/log-loss ratios | `0.9736 / 0.9714` |
| transfer Brier/log-loss ratios | `0.9214 / 0.9306` |
| transfer labels | `effect_size_shift`, `combined_shift` |

Next action: run a compact fixed-evaluation comparison of promoted
`external_monotone` scale-only versus
`external_monotone + probit_transfer_response_affine` on shared simulated
SBC/OOD seeds. Do not submit real-data LUMI sensitivity unless the compact
comparison shows both no-degradation and genuine transfer-improvement signals.

## Transfer-Aware Compact Fixed-Evaluation Comparison

Run date: 2026-07-20

Purpose: compare the promoted `external_monotone` scale-only path against
`external_monotone + probit_transfer_response_affine` on shared simulated
probit seeds before any real-data or five-seed LUMI submission.

Baseline run:

```text
/private/tmp/neural_mean_fixed_eval_20260719/external_monotone
```

Transfer-aware candidate run:

```text
/private/tmp/neural_transfer_mean_fixed_eval_20260720/external_monotone_transfer_response
```

Comparison outputs:

```text
/private/tmp/neural_transfer_mean_fixed_eval_20260720/comparison
/private/tmp/neural_transfer_mean_fixed_eval_20260720/predictive_scores
```

The fixed-evaluation comparison used identical coefficient/SBC/OOD rows. As
intended, the predictive-mean layer did not change coefficient-posterior
calibration:

| run | fixed gate | in-domain coverage | mean OOD coverage | effect shift | combined shift | rare coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `external_monotone` | fail | `0.8750` | `0.7731` | `0.7222` | `0.7361` | `1.0000` |
| `external_monotone_transfer_response` | fail | `0.8750` | `0.7731` | `0.7222` | `0.7361` | `1.0000` |

Both arms failed the compact fixed-evaluation acceptance gate because
in-domain coverage was below the `0.9000` minimum. This is a compact-bundle
qualification issue shared by both arms, not an effect of the predictive-only
candidate.

The transfer-aware predictive metadata selected a bounded response-affine
movement:

| field | value |
| --- | --- |
| method | `probit_transfer_response_affine` |
| selected | `true` |
| slope | `1.15` |
| intercept | `0.000` |
| source validation Brier/log-loss ratios | `0.9924 / 0.9894` |
| transfer validation Brier/log-loss ratios | `0.9874 / 0.9807` |
| transfer validation observations | `288` |
| transfer labels | `covariate_shift`, `effect_size_shift`, `combined_shift` |

Held-out predictive-score comparison versus scale-only:

| run | Brier ratio | log-loss ratio | predictive RMSE ratio | prevalence MAE ratio | richness MAE ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `external_monotone` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| `external_monotone_transfer_response` | `0.9849` | `0.9784` | `0.9924` | `0.7302` | `1.0194` |

Interpretation:

- The new transfer-aware response-affine candidate produced a genuine
  predictive-only improvement on this compact simulated probit comparison.
- The improvement was visible in Brier, log loss, predictive RMSE, and
  prevalence MAE.
- Richness MAE worsened slightly, so a larger confirmation should keep richness
  as a no-large-degradation metric.
- Coefficient/SBC/OOD rows were unchanged, which preserves the separation
  between coefficient calibration and predictive-only response movement.
- Because the compact fixed-evaluation bundle failed the in-domain acceptance
  gate for both arms, this run is not sufficient for promotion or LUMI
  real-data sensitivity.

Decision: keep `probit_transfer_response_affine` experimental but worth a
larger simulated confirmation. Do not submit real-data LUMI sensitivity from
this compact result alone.

Next roadmap step: run a larger fixed-evaluation simulated confirmation of
`external_monotone + probit_transfer_response_affine` against the promoted
scale-only `external_monotone` path, using enough shared SBC/OOD rows to
re-qualify the in-domain gate and retaining predictive-score no-degradation
checks for Brier, log loss, predictive RMSE, prevalence MAE, and richness MAE.

## Transfer-Aware Larger Fixed-Evaluation Confirmation

Run date: 2026-07-20

Run root:

```text
/private/tmp/neural_transfer_mean_larger_eval_20260720
```

Purpose: repeat the compact fixed-evaluation comparison with the same frozen
checkpoint and seed schedule, but increase fixed SBC/OOD evaluation from
`8 x 64` to `24 x 128`.

Baseline:

```text
/private/tmp/neural_transfer_mean_larger_eval_20260720/external_monotone
```

Candidate:

```text
/private/tmp/neural_transfer_mean_larger_eval_20260720/external_monotone_transfer_response
```

Comparison outputs:

```text
/private/tmp/neural_transfer_mean_larger_eval_20260720/comparison
/private/tmp/neural_transfer_mean_larger_eval_20260720/predictive_scores
```

The transfer-aware response calibrator selected the same bounded
predictive-only movement as the compact run:

| field | value |
| --- | --- |
| method | `probit_transfer_response_affine` |
| selected | `true` |
| slope | `1.15` |
| intercept | `0.000` |
| source validation Brier/log-loss ratios | `0.9924 / 0.9894` |
| transfer validation Brier/log-loss ratios | `0.9874 / 0.9807` |
| transfer validation observations | `288` |
| transfer labels | `covariate_shift`, `effect_size_shift`, `combined_shift` |

Fixed coefficient/SBC/OOD rows remained identical between scale-only and the
transfer-aware predictive candidate:

| run | fixed gate | in-domain coverage | mean OOD coverage | worst OOD coverage | effect shift | combined shift | rare coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `external_monotone` | fail | `0.8519` | `0.7701` | `0.6898` | `0.6898` | `0.7130` | `1.0000` |
| `external_monotone_transfer_response` | fail | `0.8519` | `0.7701` | `0.6898` | `0.6898` | `0.7130` | `1.0000` |

Held-out predictive-score comparison versus scale-only:

| run | Brier ratio | log-loss ratio | predictive RMSE ratio | prevalence MAE ratio | richness MAE ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `external_monotone` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| `external_monotone_transfer_response` | `0.9841` | `0.9771` | `0.9920` | `0.7290` | `1.0200` |

Interpretation:

- The larger local confirmation reproduced the compact predictive-only gain in
  Brier, log loss, predictive RMSE, and prevalence MAE.
- Richness MAE again worsened by about `2%`, so richness should remain an
  explicit no-large-degradation guard.
- Coefficient/SBC/OOD rows were unchanged, confirming that the response-affine
  layer is not contaminating coefficient-posterior calibration.
- Increasing the local fixed-evaluation rows did not rescue the compact
  coefficient gate; in-domain coverage moved from `0.8750` to `0.8519` for
  both arms. This points to the frozen compact checkpoint/coefficient
  calibration setup as underqualified, not to the predictive-only candidate.

Decision: do not submit real-data sensitivity or promote the candidate from
this local checkpoint. The candidate remains promising on predictive-only
proper scores, but the decision must be made on a qualified coefficient
baseline.

Next roadmap step: evaluate `probit_transfer_response_affine` on the previously
qualified production-shape `external_monotone` fixed-evaluation baseline, using
the same frozen baseline rows and predictive-score comparison discipline. This
is the direct analogue of the earlier `probit_response_affine` production-shape
evaluation and avoids blocking the predictive-mean decision on an
underqualified compact checkpoint.
