# Optional Storage Hardening

HDF5 remains the default posterior format. Zarr is optional and is only used
when the `zarr` package is installed.

Inspect a posterior before merge, resume, or archival:

```bash
python -m pyhmsc storage-info run/posterior.h5
```

For automation:

```bash
python -m pyhmsc storage-info run/posterior.h5 --json \
  > run/posterior_storage.json
```

The report includes:

- storage format
- whether native posterior metadata is present
- chain and draw counts inferred from `Beta`
- every dataset path, shape, dtype, chunk shape, and byte size
- total dense array byte size

Use `chain-status --expected-draws` before merging array jobs:

```bash
python -m pyhmsc chain-status run/chains \
  --expected-chains 0 1 2 3 \
  --expected-draws 1000 \
  --strict
```

This now checks nested random-level datasets as well as `Beta`, so truncated
`Eta`, `Lambda`, or association-relevant samples are caught before merge.

Zarr stores can be inspected the same way:

```bash
python -m pyhmsc storage-info run/posterior.zarr
```

If `zarr` is not installed, HDF5 workflows are unaffected.

## Release Qualification

Before a release, run the synthetic storage qualification workflow:

```bash
python examples/run_storage_release_qualification.py \
  --output run/storage_release_qualification
```

This creates deterministic synthetic posterior stores and checks:

- larger HDF5 posterior storage inspection
- nested `Eta` and `Lambda` random-level datasets
- per-chain HDF5 shard completeness with `chain-status --expected-draws`
- HDF5 shard merge compatibility
- intentional nested `Eta` truncation detection
- optional Zarr store inspection when `zarr` is installed

Archived qualification result:

- [Storage release qualification: 2026-06-24](storage_release_qualification_2026-06-24.md)

For constrained environments, reduce dimensions explicitly:

```bash
python examples/run_storage_release_qualification.py \
  --output run/storage_release_qualification_smoke \
  --chains 2 \
  --draws 50 \
  --covariates 4 \
  --species 8 \
  --sites 20 \
  --factors 2 \
  --skip-zarr
```
