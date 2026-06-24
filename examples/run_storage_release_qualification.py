"""Run synthetic storage release-qualification checks.

This script does not run MCMC. It creates deterministic HDF5 posterior files
with nested random-level arrays large enough to exercise release storage paths,
then validates storage inspection, chain shard checks, and HDF5 merge behavior.
If zarr is installed, it also creates and inspects an equivalent Zarr store.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.merge import inspect_chain_directory, merge_hdf5_posteriors
from pyhmsc.storage import inspect_posterior_storage


@dataclass(frozen=True)
class QualificationCheck:
    name: str
    passed: bool
    details: dict[str, Any]


def run_release_qualification(
    output: Path,
    chains: int,
    draws: int,
    covariates: int,
    species: int,
    sites: int,
    factors: int,
    include_zarr: bool,
) -> list[QualificationCheck]:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    rng = np.random.default_rng(20260624)
    metadata = _metadata(chains, draws, covariates, species, sites, factors)

    posterior = output / "posterior.h5"
    _write_hdf5_posterior(posterior, metadata, rng, chains, draws, covariates, species, sites, factors)

    chain_dir = output / "chains"
    chain_dir.mkdir()
    for chain in range(chains):
        _write_hdf5_posterior(
            chain_dir / f"posterior_chain_{chain}.h5",
            metadata,
            rng,
            1,
            draws,
            covariates,
            species,
            sites,
            factors,
            chain_id=chain,
        )

    checks = [
        _check_storage_info(posterior, "hdf5", chains, draws),
        _check_chain_status(chain_dir, chains, draws),
        _check_merge(chain_dir, output / "merged.h5", chains, draws),
        _check_truncated_nested_chain(output / "truncated_chain", metadata, rng, draws, covariates, species, sites, factors),
    ]

    if include_zarr:
        checks.append(
            _check_zarr(output / "posterior.zarr", metadata, rng, chains, draws, covariates, species, sites, factors)
        )
    else:
        checks.append(QualificationCheck("zarr_storage_info", True, {"skipped": True}))

    _write_report(output / "storage_release_qualification.txt", checks)
    (output / "storage_release_qualification.json").write_text(
        json.dumps([asdict(check) for check in checks], indent=2),
        encoding="utf-8",
    )
    return checks


def _metadata(
    chains: int,
    draws: int,
    covariates: int,
    species: int,
    sites: int,
    factors: int,
) -> dict[str, Any]:
    return {
        "schema_version": "release-qualification",
        "dimensions": {
            "n_chains": chains,
            "n_draws": draws,
            "n_covariates": covariates,
            "n_species": species,
            "n_sites": sites,
            "n_factors": factors,
        },
        "names": {
            "covariates": [f"x{i}" for i in range(covariates)],
            "species": [f"sp{i}" for i in range(species)],
        },
        "random_levels": [{"name": "site", "levels": [f"site_{i}" for i in range(sites)]}],
    }


def _write_hdf5_posterior(
    path: Path,
    metadata: dict[str, Any],
    rng: np.random.Generator,
    chains: int,
    draws: int,
    covariates: int,
    species: int,
    sites: int,
    factors: int,
    chain_id: int | None = None,
    truncate_eta: bool = False,
) -> None:
    import h5py

    eta_draws = draws - 1 if truncate_eta else draws
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["pyhmsc_metadata"] = json.dumps(metadata)
        handle.attrs["nChains"] = chains
        handle.attrs["time"] = float(10.0 + (chain_id or 0))
        handle.create_dataset(
            "Beta",
            data=rng.normal(size=(chains, draws, covariates, species)),
            chunks=(1, min(draws, 100), covariates, species),
        )
        handle.create_dataset(
            "sigma",
            data=np.abs(rng.normal(size=(chains, draws, species))) + 0.1,
            chunks=(1, min(draws, 100), species),
        )
        level = handle.create_group("random_levels").create_group("0")
        level.create_dataset(
            "Eta",
            data=rng.normal(size=(chains, eta_draws, sites, factors)),
            chunks=(1, min(eta_draws, 100), sites, factors),
        )
        level.create_dataset(
            "Lambda",
            data=rng.normal(size=(chains, draws, species, factors)),
            chunks=(1, min(draws, 100), species, factors),
        )


def _check_storage_info(path: Path, expected_format: str, chains: int, draws: int) -> QualificationCheck:
    info = inspect_posterior_storage(path)
    names = {dataset.name for dataset in info.datasets}
    required = {"Beta", "sigma", "random_levels/0/Eta", "random_levels/0/Lambda"}
    passed = (
        info.format == expected_format
        and info.metadata_present
        and info.n_chains == chains
        and info.n_draws == draws
        and required.issubset(names)
        and info.total_nbytes > 0
    )
    return QualificationCheck(
        f"{expected_format}_storage_info",
        passed,
        {
            "path": str(path),
            "format": info.format,
            "metadata_present": info.metadata_present,
            "n_chains": info.n_chains,
            "n_draws": info.n_draws,
            "dataset_count": len(info.datasets),
            "total_nbytes": info.total_nbytes,
            "missing": sorted(required.difference(names)),
        },
    )


def _check_chain_status(chain_dir: Path, chains: int, draws: int) -> QualificationCheck:
    statuses = inspect_chain_directory(chain_dir, expected_chains=list(range(chains)), expected_draws=draws)
    failed = [status for status in statuses if status.status != "passed"]
    return QualificationCheck(
        "nested_chain_status",
        not failed,
        {
            "chain_dir": str(chain_dir),
            "statuses": [
                {
                    "chain": status.chain,
                    "path": str(status.path),
                    "status": status.status,
                    "message": status.message,
                }
                for status in statuses
            ],
        },
    )


def _check_merge(chain_dir: Path, output: Path, chains: int, draws: int) -> QualificationCheck:
    inputs = [chain_dir / f"posterior_chain_{chain}.h5" for chain in range(chains)]
    merge_hdf5_posteriors(inputs, output, expected_chains=list(range(chains)))
    info = inspect_posterior_storage(output)
    passed = info.n_chains == chains and info.n_draws == draws and info.metadata_present
    return QualificationCheck(
        "hdf5_merge",
        passed,
        {
            "path": str(output),
            "n_chains": info.n_chains,
            "n_draws": info.n_draws,
            "dataset_count": len(info.datasets),
            "total_nbytes": info.total_nbytes,
        },
    )


def _check_truncated_nested_chain(
    directory: Path,
    metadata: dict[str, Any],
    rng: np.random.Generator,
    draws: int,
    covariates: int,
    species: int,
    sites: int,
    factors: int,
) -> QualificationCheck:
    directory.mkdir()
    _write_hdf5_posterior(
        directory / "posterior_chain_0.h5",
        metadata,
        rng,
        1,
        draws,
        covariates,
        species,
        sites,
        factors,
        truncate_eta=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyhmsc",
            "chain-status",
            str(directory),
            "--expected-chains",
            "0",
            "--expected-draws",
            str(draws),
            "--strict",
        ],
        text=True,
        capture_output=True,
    )
    passed = result.returncode != 0 and "random_levels/0/Eta expected" in result.stdout
    return QualificationCheck(
        "nested_truncation_detection",
        passed,
        {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        },
    )


def _check_zarr(
    path: Path,
    metadata: dict[str, Any],
    rng: np.random.Generator,
    chains: int,
    draws: int,
    covariates: int,
    species: int,
    sites: int,
    factors: int,
) -> QualificationCheck:
    try:
        import zarr  # type: ignore
    except ImportError as exc:
        return QualificationCheck("zarr_storage_info", True, {"skipped": True, "reason": str(exc)})

    root = zarr.open_group(str(path), mode="w")
    root.attrs["pyhmsc_metadata"] = metadata
    root.create_array("Beta", data=rng.normal(size=(chains, draws, covariates, species)), overwrite=True)
    root.create_array("sigma", data=np.abs(rng.normal(size=(chains, draws, species))) + 0.1, overwrite=True)
    level = root.create_group("random_levels").create_group("0")
    level.create_array("Eta", data=rng.normal(size=(chains, draws, sites, factors)), overwrite=True)
    level.create_array("Lambda", data=rng.normal(size=(chains, draws, species, factors)), overwrite=True)
    return _check_storage_info(path, "zarr", chains, draws)


def _write_report(path: Path, checks: list[QualificationCheck]) -> None:
    lines = ["# Storage Release Qualification", ""]
    for check in checks:
        status = "passed" if check.passed else "failed"
        lines.append(f"- {check.name}: {status} {check.details}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("run/storage_release_qualification"))
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--covariates", type=int, default=12)
    parser.add_argument("--species", type=int, default=30)
    parser.add_argument("--sites", type=int, default=200)
    parser.add_argument("--factors", type=int, default=3)
    parser.add_argument("--skip-zarr", action="store_true", help="skip optional Zarr store generation")
    args = parser.parse_args()

    checks = run_release_qualification(
        output=args.output,
        chains=args.chains,
        draws=args.draws,
        covariates=args.covariates,
        species=args.species,
        sites=args.sites,
        factors=args.factors,
        include_zarr=not args.skip_zarr,
    )
    for check in checks:
        status = "passed" if check.passed else "failed"
        print(f"{check.name}: {status} {check.details}")
    if not all(check.passed for check in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
