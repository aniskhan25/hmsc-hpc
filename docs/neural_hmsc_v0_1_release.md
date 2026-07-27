# Neural-HMSC v0.1 Release

Date: 2026-07-21

## Immutable Identifier

The qualified Neural-HMSC release is frozen under:

`neural_hmsc_v0_1`

The durable LUMI registry entry is:

`/scratch/project_462000131/anisrahm/hmsc-hpc-deployments/neural_hmsc_v0_1`

The local validation mirror used during the freeze is:

`/private/tmp/neural_hmsc_releases/neural_hmsc_v0_1`

The release is created atomically, and attempting to freeze the same identifier
again is rejected. Its 36-file inventory includes all three calibrated probit
checkpoints, both predictive policies for Whittaker and Big Spatial, the
original predictive baseline manifests and evidence, the release audit, and the
support matrix. No model was fitted and no calibration was selected during the
freeze.

Before publication, all 36 staged files were rehashed on LUMI and matched the
content inventory. The staging directory was then renamed atomically to the
stable identifier only after confirming that destination did not exist.

| Record | SHA-256 |
| --- | --- |
| Release content inventory | `affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8` |
| `release.json` | `31ee489898e3657b97919803c0e850dc20494ef9118e9b963fe4a20365822e98` |
| Complete checkpoint package manifest | `d2daa81ec841390df59324a208216ffa0032ac514e6c679649d98815490bdbc7` |
| Release audit JSON | `9b6392b4ca45f91c3bd4979656712a6ce4e0d1a1573eb753f8cc65eb6e4a188f` |
| Release audit Markdown | `5d9638cf823667d9e417ed9dee02251ae9f9b161fdcac4125c005eb6fdeb6f9f` |

The release occupies approximately 58 MB. Predictive manifests are relocated
to release-local member paths without changing member bytes or semantic
provenance. The original qualified manifests are retained unchanged under
`predictive/source_baseline`.

## Public API

Load the release by identifier, then select a calibrated coefficient checkpoint
or a predictive-only ensemble:

```python
from pyhmsc import load_neural_hmsc_release

release = load_neural_hmsc_release(
    "/private/tmp/neural_hmsc_releases",
    release_id="neural_hmsc_v0_1",
)

engine = release.load_checkpoint(seed=20260721)
ensemble = release.load_predictive_ensemble(dataset="whittaker")
scale_only = release.load_predictive_ensemble(
    dataset="whittaker",
    policy="scale_only",
)
```

`engine.infer(...)` applies the packaged `external_monotone` coefficient scale
calibration. The ensemble is explicitly predictive-only; its default policy is
`affine_branch`, and `scale_only` remains the explicit fallback.

Run the complete compiled-artifact, posterior, `HmscFit`, and ensemble example:

```bash
python examples/run_neural_hmsc_v0_1_release.py \
  --registry-root /private/tmp/neural_hmsc_releases \
  --output /private/tmp/neural_hmsc_v0_1_public_example
```

The verified example emitted a `1 x 32 x 2 x 75` calibrated `Beta` sample array
and a `40 x 75` three-member Whittaker probability prediction.

## Freeze Reproduction

```bash
python examples/package_neural_hmsc_v0_1_checkpoints.py \
  --source-members-root /private/tmp/neural_predictive_affine_v1_members \
  --sensitivity-root /private/tmp/neural_hmsc_source_transfer_realdata_sensitivity_20260720 \
  --output-root /private/tmp/neural_hmsc_v0_1_packaged_members_complete

python examples/freeze_neural_hmsc_v0_1_release.py \
  --registry-root /private/tmp/neural_hmsc_releases \
  --packaged-members-root /private/tmp/neural_hmsc_v0_1_packaged_members_complete \
  --predictive-baseline-root /private/tmp/neural_predictive_affine_v1/neural_predictive_affine_v1 \
  --audit-root /private/tmp/neural_hmsc_v0_1_release_audit_complete_20260721
```

## Qualified Scope

v0.1 qualifies a bounded, accelerated, fixed-shape fixed-effect probit `Beta`
approximation. It does not claim joint-posterior equivalence, full HMSC
structural equivalence, or predictive superiority over MCMC. Qualified Python
MCMC remains the statistical reference. Normal and Poisson are implemented but
not release-qualified.

Checkpoint training may cost more than one compact MCMC fit. The operational
benefit assumes repeated inference over compatible datasets, where the audit
measured more than 200x inference-only speedup.

## Next Step

Proceed to variable-shape fixed-effect inference under Milestone 52. Preserve
the complete v0.1 release as the fixed-shape regression baseline and do not
change its calibration or deployment manifests in place.
