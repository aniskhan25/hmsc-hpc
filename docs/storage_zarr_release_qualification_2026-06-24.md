# Zarr Storage Release Qualification: 2026-06-24

This run validates the optional Zarr posterior storage path in an isolated
temporary environment. It does not modify the system Python environment.

## Environment

- Host: local macOS workstation
- Python: 3.12.0
- Temporary venv:
  `/private/tmp/hmsc_zarr_qualification_env_zarr2`
- Zarr: `2.18.7`
- NumPy: existing project-compatible `1.26.4`
- Output directory:
  `/private/tmp/hmsc_zarr_release_qualification`

The first attempt with latest `zarr` pulled NumPy 2.x into a temporary venv and
caused a Pandas binary compatibility failure. The successful run used
`zarr<3`, which exercises the optional Zarr path while preserving the existing
NumPy/Pandas stack.

## Command

```bash
python -m venv --system-site-packages /private/tmp/hmsc_zarr_qualification_env_zarr2
/private/tmp/hmsc_zarr_qualification_env_zarr2/bin/python \
  -m pip install --upgrade-strategy only-if-needed 'zarr<3'
/private/tmp/hmsc_zarr_qualification_env_zarr2/bin/python \
  examples/run_storage_release_qualification.py \
  --output /private/tmp/hmsc_zarr_release_qualification
```

## Synthetic Dimensions

- chains: `4`
- draws: `500`
- covariates: `12`
- species: `30`
- sites: `200`
- latent factors: `3`

## Results

All HDF5 and Zarr release-qualification checks passed.

| Check | Result | Notes |
| --- | --- | --- |
| `hdf5_storage_info` | passed | Metadata present; 4 chains; 500 draws; 4 datasets; `17,280,000` dense array bytes |
| `nested_chain_status` | passed | All chain shards `0..3` passed nested draw-count inspection |
| `hdf5_merge` | passed | Merged posterior preserved 4 chains, 500 draws, metadata, and 4 datasets |
| `nested_truncation_detection` | passed | Intentional `random_levels/0/Eta` truncation was detected |
| `zarr_storage_info` | passed | Metadata present; 4 chains; 500 draws; 4 datasets; `17,280,000` dense array bytes |

The strict truncation check produced the expected failure:

```text
random_levels/0/Eta expected 500 draws, found 499
```

## Focused Zarr Test Run

```bash
/private/tmp/hmsc_zarr_qualification_env_zarr2/bin/python \
  -m pytest tests/test_storage_release_qualification.py \
  tests/test_storage_info.py \
  tests/test_zarr_optional.py
```

Result:

```text
6 passed
```

## Interpretation

The optional Zarr storage path is release-qualified for:

- synthetic posterior store creation
- `pyhmsc storage-info` Zarr inspection
- native metadata detection
- nested random-level dataset discovery
- compatibility with Zarr v2 writer APIs

Zarr v3 can be revisited later, but release-scope optional Zarr qualification is
covered with `zarr 2.18.7`.
