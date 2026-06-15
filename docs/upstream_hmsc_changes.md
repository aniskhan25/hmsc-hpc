# Upstream hmsc-hpc Change Register

This document tracks changes made inside the original `hmsc/` package while
building the Python-native workflow. The intent is to keep these changes visible:
anything under `hmsc/` should be treated as an upstream `hmsc-hpc` patch or a
clearly documented compatibility extension, not as ordinary `pyhmsc` wrapper
code.

The Python-native layer lives mostly in `pyhmsc/`, `examples/`, `docs/`, and
top-level tests. Those changes are expected for this project. Changes to
`hmsc/` require extra scrutiny because they modify the sampler, loaders,
exporters, or original test suite.

## Current policy

- Avoid further edits to `hmsc/` unless a sampler bug or missing native-format
  bridge makes it unavoidable.
- Record every `hmsc/` edit here with the reason, affected behavior, validation,
  and whether it should be upstreamed.
- Prefer putting Python-only model compilation, validation, and analysis code in
  `pyhmsc/`.
- Keep `.rds` compatibility working unless a change is intentionally scoped to
  the new Python-native JSON/HDF5 path.
- When a core sampler fix is found, prepare an upstream issue or pull request
  with a minimal reproducer and tests.

## Changes from the Python-native implementation period

These commits were made during the Python-native HMSC implementation and touch
the original `hmsc/` tree.

| Commit | Files under `hmsc/` | Category | Upstream status |
| --- | --- | --- | --- |
| `b80c2b0` Validate multi-factor NNGP Eta recovery | `hmsc/updaters/updateEta.py`, `hmsc/test/updateEta_test.py` | Core sampler bug fix and tests | Should be upstreamed |
| `f268ae9` Support NNGP spatial random slopes | `hmsc/run_gibbs_sampler.py` | Native runner support | Audit before upstreaming |
| `7625036` Add native NNGP spatial support | `hmsc/utils/export_native_utils.py` | Native JSON/HDF5 loader support | Extension; upstream optional |
| `aad33f3` Validate random slope and GPP native paths | `hmsc/updaters/updateLambdaPriors.py`, `hmsc/test/updateLambdaPriors_test.py` | Core updater compatibility/test | Should be reviewed upstream |
| `9568764` Support native random slopes and GPP spatial effects | `hmsc/gibbs_sampler.py`, `hmsc/run_gibbs_sampler.py`, `hmsc/updaters/updateBetaLambda.py`, `hmsc/updaters/updateEta.py`, `hmsc/updaters/updateLambdaPriors.py`, `hmsc/updaters/updateNf.py`, `hmsc/updaters/updateSigma.py`, `hmsc/utils/export_native_utils.py` | Core sampler plus native-format support | Needs detailed upstream patch split |
| `cb22cd9` Add posterior gradient helpers | `hmsc/test/test_update_z.py`, `hmsc/test/updateBetaLambda_test.py`, `hmsc/test/updateSigma_test.py`, `hmsc/test/updateZ_test.py` | Original test-suite updates | Low risk; still upstream-visible |
| `71280a2` Preserve native metadata in posteriors | `hmsc/run_gibbs_sampler.py`, `hmsc/utils/export_hdf5_utils.py`, `hmsc/utils/export_zarr_utils.py` | Native output metadata | Extension; upstream optional |
| `bc6b33d` Harden native workflow validation | `hmsc/run_gibbs_sampler.py` | Native input validation | Extension; upstream optional |
| `0548118` Add Newick and prediction mode support | `hmsc/utils/export_native_utils.py`, `hmsc/utils/export_zarr_utils.py` | Native loader/output support | Extension; upstream optional |
| `cce79e3` Add validation CLI and richer posterior tooling | `hmsc/run_gibbs_sampler.py`, `hmsc/utils/export_hdf5_utils.py`, `hmsc/utils/export_zarr_utils.py` | Native output and CLI support | Extension; upstream optional |
| `edf4ecd` Add native traits phylogeny and spatial support | `hmsc/utils/export_native_utils.py` | Native loader support | Extension; upstream optional |
| `05407fb` Add Python-native HMSC workflow | `hmsc/run_gibbs_sampler.py`, `hmsc/utils/export_hdf5_utils.py`, `hmsc/utils/export_json_utils.py`, `hmsc/utils/export_native_utils.py`, `hmsc/utils/export_rds_utils.py` | Initial native loader/output bridge | Extension; upstream optional |

## Highest-priority upstream candidate

### Multi-factor NNGP Eta prior precision

Commit: `b80c2b0`

Affected file: `hmsc/updaters/updateEta.py`

The multi-factor NNGP Eta prior precision construction was refactored into a
helper and corrected for vector ordering. The previous construction assembled
the spatial precision in factor-major order, while the likelihood and `mu0`
vector used unit-major ordering. The fix maps sparse precision indices as:

```python
row = iW.row * nf + h
col = iW.col * nf + h
```

This is not just Python-native plumbing. It is a sampler correctness fix for
multi-factor NNGP spatial random effects and should be reported upstream with:

- a minimal two-factor NNGP reproducer,
- the added `hmsc/test/updateEta_test.py` coverage,
- the LUMI validation result from the multi-factor Eta recovery run,
- confirmation that single-factor behavior is unchanged.

Validation already run locally for this patch included the focused test suite
and the multi-factor NNGP validation workflow. The final validation report for
the LUMI run showed strong aligned Eta/Lambda recovery and near-perfect
association recovery.

## Changes that should be split before upstreaming

Commit `9568764` is too broad for an upstream pull request as-is. It combines:

- native-format model loading,
- random-slope support,
- GPP spatial handling,
- updater shape compatibility,
- sampler state handling,
- command-line runner changes.

Before proposing upstream changes, split this into smaller patches:

1. Native input/export bridge only.
2. Random-slope model-state support.
3. GPP spatial model-state support.
4. Updater shape fixes required independent of native input.
5. Tests for each isolated behavior.

This separation matters because upstream maintainers may want core sampler
fixes but not the Python-native JSON/HDF5 API in the same pull request.

## Older `hmsc/` history visible in this branch

The branch also contains older commits touching `hmsc/` that predate the most
recent Python-native validation cycle. They should not automatically be treated
as part of the current implementation, but they are still relevant if we prepare
a clean upstream patch series.

| Commit | Files under `hmsc/` | Summary |
| --- | --- | --- |
| `07f0a0b` | `hmsc/utils/export_rds_utils.py` | BetaSel posterior RDS export shape adjustment |
| `54a0753` | `hmsc/run_gibbs_sampler.py` | Output text typo fix |
| `d76b88e` | `hmsc/run_gibbs_sampler.py` | Startup print flushing |
| `2db465a` | `hmsc/run_gibbs_sampler.py` | Eager execution flag |
| `3847559` | `hmsc/updaters/updateBetaLambda.py`, `hmsc/updaters/updateSigma.py`, `hmsc/updaters/updateZ.py`, `hmsc/utils/import_utils.py` | Faster special-case updater computation |
| `ff4b03d` | `hmsc/utils/import_utils.py` | Spatial import warning handling |
| `1c3d579` | `hmsc/utils/import_utils.py` | NumPy `np.Inf` compatibility fix |
| `c3f7c5b` | `hmsc/updaters/updateAlpha.py`, `hmsc/utils/import_utils.py` | Spatial import fix for `sDim=Inf` |
| `a43bb3a` | `hmsc/test/test_update_z.py` | NaN masking test fix |
| `bcd3f78` | `hmsc/test/test_update_z.py` | Restored deleted test |
| `cdc45c3` | `hmsc/updaters/updateLambdaPriors.py` | Psi updater shape fix |
| `c162dfd` | `hmsc/test/updateEta_test.py` | Eta test modification |
| `91fdb16` | `hmsc/test/updateEta_test.py` | Eta test compatibility fix |
| `fb4e10e` | `hmsc/utils/import_utils.py` | GPP spatial import fix |
| `3dfb931` | `hmsc/test/updatewRRRPriors_test.py` | wRRR prior test amendment |
| `91c15ba` | `hmsc/test/updateLambdaPriors_test.py` | Lambda prior test amendment |
| `297d9ce` | `hmsc/test/updatewRRRPriors_test.py` | wRRR prior test amendment |
| `0cad6b2` | `hmsc/gibbs_sampler.py`, `hmsc/run_gibbs_sampler.py`, `hmsc/updaters/updateBetaEta.py`, `hmsc/updaters/updateBetaLambda.py`, `hmsc/updaters/updateHMC.py`, `hmsc/updaters/updateZ.py`, `hmsc/utils/export_rds_utils.py`, `hmsc/utils/import_utils.py` | BetaEta/HMC updater and related fixes |
| `e144b46` | `hmsc/examples/demo_gibbs_sampler_models.R`, `hmsc/examples/run_gibbs_sampler.py`, `hmsc/gibbs_sampler.py`, `hmsc/updaters/updateBetaEta.py`, `hmsc/updaters/updateHMC.py` | BetaEta finalization and HMC frequency control |

## Recommended next actions

1. Stop making opportunistic edits in `hmsc/` during Python-native feature work.
2. Create a clean upstream patch branch containing only `hmsc/` changes.
3. Split broad commits, especially `9568764`, into focused upstream-reviewable
   patches.
4. Open an upstream issue or pull request first for the multi-factor NNGP Eta
   ordering fix, because it is the clearest correctness issue.
5. Keep the Python-native feature branch carrying only `pyhmsc/`, examples,
   docs, sbatch scripts, and tests when possible.

