# Generative Neural-HMSC IID v2 Disposable Attempt 20518403

Date: 2026-08-01

Status: failed before disposable simulation generation.

## Submitted Boundary

- LUMI job: `20518403`
- partition: `dev-g`
- source commit: `940d73d6de6e032797e4d695bd9799a74ef0b943`
- source archive SHA-256:
  `76911182c1d34bcd4c979f70b1340af126ddd89baafdc821c4024cc6f846a43a`
- scheduler SHA-256:
  `7a0bf9ecf89a5e1896ba254d24e916978cfe8e76caf0898a7dffef1df679cf07`
- exact confirmation: `GENERATE_593M_594M_DISPOSABLE_ONLY`

## Failure

The job exited `1:0` after 37 seconds during the token-free preflight import:

```text
ModuleNotFoundError: No module named 'pyhmsc.neural'
```

The isolated source archive contains `pyhmsc/neural`. The scheduler exported
the host absolute source path as `PYTHONPATH`, but the TensorFlow Singularity
container exposes the same directory under `/pfs/lustrep4/scratch`. Python
therefore resolved another installed `pyhmsc` package instead of the pinned
source tree.

## Independent Inventory

- authorized run root: absent
- preflight file: zero bytes, SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- stdout SHA-256:
  `c319a353e2e066d3d5b49f38671694a47fbc3658e0d1270d2f496b71686d8198`
- stderr SHA-256:
  `a258a6135a93a31fe43867969067f1f38e3a7f12f6e05632b549e84c96d4a3d4`
- corpus manifests: absent
- checkpoints and weights: absent
- freeze and post-freeze validation artifacts: absent

No 593M or 594M seed was opened. The 511M-515M blocks remain sealed.

## Bounded Correction

The corrected scheduler uses `PYTHONPATH=.` after changing to the isolated
source root and launches the harness as the repository module
`examples.run_generative_neural_hmsc_iid_v2`. This avoids script-mode path
preemption by `examples/`. A full token-free LUMI harness preflight then passed
with `simulation_generation_called = false`, `output_created = false`, and all
production/evaluation seal booleans false. The correction changes only
container path resolution and launcher mode; it does not change the source
archive, representation, objective, schedule, gates, thresholds, seed roles,
or artifact contract.

The corrected, unsubmitted scheduler is
`docs/lumi_generative_neural_hmsc_iid_v2_disposable_retry_sbatch.sh`, SHA-256
`ff618d92c4d4f616507aaa31e2f434cb2cdaa9b2d985bcc0e7e567bc6735cdb7`.

The original one-shot authorization is consumed. The corrected scheduler may
not be submitted until its hash is frozen and a separate retry authorization
record explicitly opens the same disposable blocks.
