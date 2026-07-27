"""Manifest-backed response-probability ensembles for Neural-HMSC artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from pyhmsc.posterior import HmscFit


ENSEMBLE_KIND = "pyhmsc_predictive_probability_ensemble"
ENSEMBLE_SCHEMA_VERSION = 1
ENSEMBLE_AGGREGATION = "arithmetic_mean_response_probability"
DEFAULT_PREDICTIVE_MEAN_POLICY = "affine_branch"
PREDICTIVE_MEAN_POLICIES = (DEFAULT_PREDICTIVE_MEAN_POLICY, "scale_only")
PARITY_PROVENANCE_FILES = (
    "acceptance",
    "run_metadata",
    "parity_metrics",
    "mcmc_reference",
)


@dataclass(frozen=True)
class PredictiveEnsembleMember:
    """One immutable predictive posterior member."""

    path: Path
    sha256: str
    seed: int
    calibration_role: str
    provenance: dict[str, Any]


class PredictiveProbabilityEnsemble:
    """Average response-scale probabilities from compatible HmscFit members."""

    def __init__(
        self,
        members: Sequence[PredictiveEnsembleMember],
        fits: Sequence[HmscFit],
        *,
        compatibility: dict[str, Any],
        provenance: dict[str, Any] | None = None,
    ) -> None:
        if len(members) < 2:
            raise ValueError("a predictive ensemble requires at least two members")
        if len(members) != len(fits):
            raise ValueError("ensemble members and fits must have equal length")
        seeds = [int(member.seed) for member in members]
        if len(set(seeds)) != len(seeds):
            raise ValueError("ensemble member seeds must be unique")
        roles = {str(member.calibration_role) for member in members}
        if len(roles) != 1:
            raise ValueError("ensemble members must share one calibration role")
        self.members = tuple(members)
        self._fits = tuple(fits)
        self.compatibility = dict(compatibility)
        self.provenance = {} if provenance is None else dict(provenance)

    @classmethod
    def create(
        cls,
        member_paths: Sequence[str | Path],
        *,
        seeds: Sequence[int],
        calibration_role: str,
        member_provenance: Sequence[dict[str, Any]] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> "PredictiveProbabilityEnsemble":
        """Create and validate an ensemble from ordered predictive HDF5 files."""
        paths = [Path(path).expanduser().resolve() for path in member_paths]
        if len(paths) != len(seeds):
            raise ValueError("member_paths and seeds must have equal length")
        if member_provenance is None:
            provenance_rows = [{} for _ in paths]
        else:
            provenance_rows = [dict(value) for value in member_provenance]
            if len(provenance_rows) != len(paths):
                raise ValueError(
                    "member_provenance and member_paths must have equal length"
                )
        fits = []
        signatures = []
        members = []
        for path, seed, member_metadata in zip(paths, seeds, provenance_rows):
            if not path.is_file():
                raise FileNotFoundError(f"ensemble member does not exist: {path}")
            fit = HmscFit.from_file(path)
            signature = _predictive_fit_signature(fit, path=path)
            fits.append(fit)
            signatures.append(signature)
            _validate_member_parity_provenance(member_metadata, verify_hashes=True)
            members.append(
                PredictiveEnsembleMember(
                    path=path,
                    sha256=file_sha256(path),
                    seed=int(seed),
                    calibration_role=str(calibration_role),
                    provenance=member_metadata,
                )
            )
        compatibility = _validate_signatures(signatures)
        return cls(
            members,
            fits,
            compatibility=compatibility,
            provenance=provenance,
        )

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        *,
        verify_hashes: bool = True,
    ) -> "PredictiveProbabilityEnsemble":
        """Load an ensemble manifest and verify member integrity by default."""
        manifest_path = Path(path).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("kind") != ENSEMBLE_KIND:
            raise ValueError("unsupported predictive ensemble manifest kind")
        if int(payload.get("schema_version", -1)) != ENSEMBLE_SCHEMA_VERSION:
            raise ValueError("unsupported predictive ensemble schema version")
        if payload.get("aggregation") != ENSEMBLE_AGGREGATION:
            raise ValueError("unsupported predictive ensemble aggregation")
        member_rows = payload.get("members")
        if not isinstance(member_rows, list) or len(member_rows) < 2:
            raise ValueError("predictive ensemble manifest requires members")

        members = []
        fits = []
        signatures = []
        for row in member_rows:
            if not isinstance(row, dict):
                raise ValueError("predictive ensemble member metadata must be an object")
            member_path = Path(str(row["path"])).expanduser()
            if not member_path.is_absolute():
                member_path = manifest_path.parent / member_path
            member_path = member_path.resolve()
            expected_hash = str(row["sha256"])
            if verify_hashes:
                observed_hash = file_sha256(member_path)
                if observed_hash != expected_hash:
                    raise ValueError(
                        f"ensemble member hash mismatch for {member_path}: "
                        f"expected {expected_hash}, got {observed_hash}"
                    )
            fit = HmscFit.from_file(member_path)
            member_provenance = dict(row.get("provenance", {}))
            _validate_member_parity_provenance(
                member_provenance,
                verify_hashes=verify_hashes,
            )
            signatures.append(_predictive_fit_signature(fit, path=member_path))
            fits.append(fit)
            members.append(
                PredictiveEnsembleMember(
                    path=member_path,
                    sha256=expected_hash,
                    seed=int(row["seed"]),
                    calibration_role=str(row["calibration_role"]),
                    provenance=member_provenance,
                )
            )
        observed_compatibility = _validate_signatures(signatures)
        expected_compatibility = payload.get("compatibility")
        if observed_compatibility != expected_compatibility:
            raise ValueError("ensemble compatibility metadata does not match members")
        return cls(
            members,
            fits,
            compatibility=observed_compatibility,
            provenance=dict(payload.get("provenance", {})),
        )

    @property
    def calibration_role(self) -> str:
        return self.members[0].calibration_role

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(member.seed for member in self.members)

    @property
    def parity_provenance_qualified(self) -> bool:
        return bool(
            self.members
            and all(
                member.provenance.get("reference_parity_qualified", False)
                and member.provenance.get("dataset_acceptance_passed", False)
                and _member_has_complete_parity_provenance(member.provenance)
                for member in self.members
            )
        )

    @property
    def qualified_mcmc_reference(self) -> dict[str, Any]:
        """Return validated Python-MCMC reference provenance."""
        reference = self.provenance.get("qualified_python_mcmc_reference")
        if not isinstance(reference, dict):
            raise ValueError("ensemble lacks qualified Python MCMC reference")
        if reference.get("kind") != "qualified_python_mcmc_probability_reference":
            raise ValueError("ensemble has unsupported Python MCMC reference kind")
        rows = reference.get("ordered_members")
        if not isinstance(rows, list) or len(rows) != len(self.members):
            raise ValueError("ensemble Python MCMC reference member count differs")
        for member, row in zip(self.members, rows):
            if not isinstance(row, dict) or int(row.get("seed", -1)) != member.seed:
                raise ValueError("ensemble Python MCMC reference seed order differs")
            if row.get("path") != member.provenance.get("mcmc_reference_path"):
                raise ValueError("ensemble Python MCMC reference path differs")
            if row.get("sha256") != member.provenance.get("mcmc_reference_sha256"):
                raise ValueError("ensemble Python MCMC reference hash differs")
        return dict(reference)

    def predict_mean(
        self,
        X_new: Any,
        *,
        random_effects: str = "none",
        unseen_groups: str = "error",
        study_design: Any | None = None,
        coords: Any | None = None,
        spatial_prediction: str = "nearest",
        rng_seed: int | None = None,
    ) -> pd.DataFrame:
        """Return the arithmetic mean of member response probabilities."""
        predictions = []
        reference: pd.DataFrame | None = None
        for fit in self._fits:
            prediction = fit.predict_mean(
                X_new,
                response=True,
                random_effects=random_effects,
                unseen_groups=unseen_groups,
                study_design=study_design,
                coords=coords,
                spatial_prediction=spatial_prediction,
                rng_seed=rng_seed,
            )
            if reference is None:
                reference = prediction
            elif not prediction.index.equals(reference.index):
                raise ValueError("ensemble member prediction index mismatch")
            elif not prediction.columns.equals(reference.columns):
                raise ValueError("ensemble member prediction species mismatch")
            predictions.append(prediction.to_numpy(dtype=float))
        if reference is None:
            raise ValueError("predictive ensemble contains no members")
        values = np.mean(np.stack(predictions), axis=0)
        return pd.DataFrame(
            values,
            index=reference.index,
            columns=reference.columns,
        )

    def subset(self, seeds: Sequence[int]) -> "PredictiveProbabilityEnsemble":
        """Return an ordered member subset while preserving compatibility."""
        requested = [int(seed) for seed in seeds]
        if len(requested) < 2:
            raise ValueError("an ensemble subset requires at least two seeds")
        if len(set(requested)) != len(requested):
            raise ValueError("ensemble subset seeds must be unique")
        positions = {member.seed: index for index, member in enumerate(self.members)}
        missing = [seed for seed in requested if seed not in positions]
        if missing:
            raise ValueError(f"ensemble subset contains unknown seeds: {missing}")
        indices = [positions[seed] for seed in requested]
        provenance = {
            **self.provenance,
            "parent_seeds": list(self.seeds),
            "subset_seeds": requested,
        }
        return PredictiveProbabilityEnsemble(
            [self.members[index] for index in indices],
            [self._fits[index] for index in indices],
            compatibility=self.compatibility,
            provenance=provenance,
        )

    def to_manifest(
        self,
        *,
        manifest_dir: str | Path | None = None,
        relative_paths: bool = False,
    ) -> dict[str, Any]:
        base = None if manifest_dir is None else Path(manifest_dir).resolve()
        member_rows = []
        for member in self.members:
            stored_path = str(member.path)
            if relative_paths:
                if base is None:
                    raise ValueError("manifest_dir is required for relative paths")
                stored_path = os.path.relpath(member.path, base)
            member_rows.append(
                {
                    "path": stored_path,
                    "sha256": member.sha256,
                    "seed": int(member.seed),
                    "calibration_role": member.calibration_role,
                    "provenance": member.provenance,
                }
            )
        return {
            "schema_version": ENSEMBLE_SCHEMA_VERSION,
            "kind": ENSEMBLE_KIND,
            "artifact_role": "predictive_only_probability_ensemble",
            "aggregation": ENSEMBLE_AGGREGATION,
            "ordered_members": True,
            "calibration_role": self.calibration_role,
            "compatibility": self.compatibility,
            "members": member_rows,
            "provenance": self.provenance,
        }

    def save(self, path: str | Path, *, relative_paths: bool = False) -> Path:
        """Write a JSON ensemble manifest."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_manifest(
            manifest_dir=output.parent,
            relative_paths=relative_paths,
        )
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predictive_mean_ensemble(
    manifest_dir: str | Path,
    *,
    dataset: str,
    policy: str = DEFAULT_PREDICTIVE_MEAN_POLICY,
    require_qualified_reference: bool = True,
) -> PredictiveProbabilityEnsemble:
    """Load a deployment ensemble using the promoted neural policy by default."""
    if policy not in PREDICTIVE_MEAN_POLICIES:
        raise ValueError(
            f"unsupported predictive-mean policy {policy!r}; "
            f"expected one of {PREDICTIVE_MEAN_POLICIES}"
        )
    dataset_name = str(dataset)
    if not dataset_name or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for character in dataset_name
    ):
        raise ValueError("dataset must contain only lowercase letters, digits, or '_'")
    path = Path(manifest_dir) / f"{dataset_name}_{policy}_ensemble.json"
    ensemble = PredictiveProbabilityEnsemble.from_manifest(path)
    if ensemble.calibration_role != policy:
        raise ValueError(
            f"deployment manifest role {ensemble.calibration_role!r} does not "
            f"match requested policy {policy!r}"
        )
    if ensemble.provenance.get("dataset") != dataset_name:
        raise ValueError("deployment manifest dataset provenance differs")
    if ensemble.provenance.get("response_semantics") != "predictive_only":
        raise ValueError("deployment manifest is not explicitly predictive-only")
    if ensemble.provenance.get("selection_outcomes_used") is not False:
        raise ValueError("deployment manifest does not exclude selection outcomes")
    if not ensemble.parity_provenance_qualified:
        raise ValueError("deployment manifest parity provenance is not qualified")
    if require_qualified_reference:
        ensemble.qualified_mcmc_reference
    return ensemble


def _predictive_fit_signature(fit: HmscFit, *, path: Path) -> dict[str, Any]:
    metadata = fit.metadata
    if not isinstance(metadata, dict):
        raise ValueError(f"predictive ensemble member lacks metadata: {path}")
    if metadata.get("artifact_role") != "predictive_only":
        raise ValueError(f"ensemble member is not predictive-only: {path}")
    names = metadata.get("names")
    formula = metadata.get("formula")
    inference = metadata.get("inference")
    if not isinstance(names, dict) or not isinstance(formula, dict):
        raise ValueError(f"ensemble member lacks names/formula metadata: {path}")
    if not isinstance(inference, dict):
        raise ValueError(f"ensemble member lacks inference metadata: {path}")
    species = names.get("species")
    covariates = names.get("covariates")
    if not isinstance(species, list) or not species:
        raise ValueError(f"ensemble member lacks species names: {path}")
    if not isinstance(covariates, list) or not covariates:
        raise ValueError(f"ensemble member lacks covariate names: {path}")
    return {
        "distribution": str(metadata.get("distribution")),
        "formula_X": str(formula.get("X")),
        "covariates": [str(value) for value in covariates],
        "species": [str(value) for value in species],
        "parameter": str(inference.get("parameter")),
        "artifact_role": str(metadata.get("artifact_role")),
    }


def _validate_signatures(signatures: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not signatures:
        raise ValueError("predictive ensemble requires compatibility signatures")
    expected = signatures[0]
    for index, signature in enumerate(signatures[1:], start=1):
        if signature != expected:
            raise ValueError(
                f"ensemble member {index} is incompatible with member 0"
            )
    return dict(expected)


def _member_has_complete_parity_provenance(provenance: dict[str, Any]) -> bool:
    return all(
        provenance.get(f"{name}_path") and provenance.get(f"{name}_sha256")
        for name in PARITY_PROVENANCE_FILES
    )


def _validate_member_parity_provenance(
    provenance: dict[str, Any],
    *,
    verify_hashes: bool,
) -> None:
    claims_qualification = bool(
        provenance.get("reference_parity_qualified", False)
        or provenance.get("dataset_acceptance_passed", False)
    )
    if not claims_qualification:
        return
    if not (
        provenance.get("reference_parity_qualified", False)
        and provenance.get("dataset_acceptance_passed", False)
    ):
        raise ValueError("ensemble member has incomplete parity qualification flags")
    if not _member_has_complete_parity_provenance(provenance):
        raise ValueError("ensemble member has incomplete parity provenance files")
    if not verify_hashes:
        return
    for name in PARITY_PROVENANCE_FILES:
        path = Path(str(provenance[f"{name}_path"])).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"parity provenance file does not exist: {path}")
        expected = str(provenance[f"{name}_sha256"])
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(
                f"parity provenance hash mismatch for {path}: "
                f"expected {expected}, got {observed}"
            )
