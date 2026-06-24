# Storage Release Qualification: 2026-06-24

This release-qualification run validates the posterior storage paths without
running MCMC. It uses deterministic synthetic posterior arrays with nested
random-level datasets to exercise HDF5 storage inspection, chain shard status,
HDF5 merge behavior, and nested truncation detection.

## Command

```bash
python examples/run_storage_release_qualification.py \
  --output /private/tmp/hmsc_storage_release_qualification_default
```

## Environment

- Host: local macOS workstation
- Python: 3.12.0
- Output directory:
  `/private/tmp/hmsc_storage_release_qualification_default`
- Zarr: not installed locally, so optional Zarr inspection was skipped

## Synthetic Dimensions

- chains: `4`
- draws: `500`
- covariates: `12`
- species: `30`
- sites: `200`
- latent factors: `3`

## Results

All required HDF5 release-qualification checks passed.

| Check | Result | Notes |
| --- | --- | --- |
| `hdf5_storage_info` | passed | Metadata present; 4 chains; 500 draws; 4 datasets; `17,280,000` dense array bytes |
| `nested_chain_status` | passed | All chain shards `0..3` passed nested draw-count inspection |
| `hdf5_merge` | passed | Merged posterior preserved 4 chains, 500 draws, metadata, and 4 datasets |
| `nested_truncation_detection` | passed | Intentional `random_levels/0/Eta` truncation was detected with strict chain-status |
| `zarr_storage_info` | passed/skipped | Optional path skipped because `zarr` is not installed |

The truncation check produced the expected strict failure:

```text
random_levels/0/Eta expected 500 draws, found 499
```

## Focused Test Run

```bash
pytest tests/test_storage_release_qualification.py \
  tests/test_storage_info.py \
  tests/test_merge_hdf5.py
```

Result:

```text
10 passed, 1 skipped
```

The skipped test was the optional Zarr storage test, skipped because `zarr` is
not installed in the local environment.

## Interpretation

The default HDF5 posterior release path is qualified for:

- storage inspection through `pyhmsc storage-info`
- nested `Eta`/`Lambda` dataset discovery
- chain shard completeness checks with `chain-status --expected-draws`
- detection of truncated nested random-level datasets
- HDF5 shard merge compatibility

Optional Zarr storage remains supported by code paths and tests when the `zarr`
extra is installed, but it was not exercised in this local qualification run.
