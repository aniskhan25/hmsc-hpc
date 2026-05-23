"""Python posterior reader and convenience summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyhmsc.formulas import build_design_matrix


class HmscFit:
    def __init__(self, posterior: dict[str, Any], model: Any | None = None) -> None:
        self.posterior = posterior
        self.model = model
        self.init_file: Path | None = None
        self.output_file: Path | None = None
        self.workdir: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path, model: Any | None = None) -> "HmscFit":
        path = Path(path)
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() in {".h5", ".hdf5"}:
            data = _read_hdf5_posterior(path)
        elif path.suffix.lower() == ".zarr":
            data = _read_zarr_posterior(path)
        else:
            try:
                import pyreadr  # type: ignore
            except ImportError as exc:
                raise RuntimeError("Install pyreadr to read Hmsc-HPC RDS output files") from exc
            raw = pyreadr.read_r(str(path))
            data = json.loads(raw[None][None][0])
        return cls(data, model=model)

    def _samples(self, param: str) -> np.ndarray:
        if "__arrays__" in self.posterior:
            arrays = self.posterior["__arrays__"]
            if param not in arrays:
                raise ValueError(f"Posterior does not contain {param!r}")
            return arrays[param]
        chains = []
        for chain_key in sorted(
            [key for key in self.posterior.keys() if str(key).isdigit()],
            key=lambda value: int(value),
        ):
            chain = self.posterior[chain_key]
            draws = []
            for draw_key in sorted(chain.keys(), key=lambda value: int(value)):
                draws.append(np.asarray(chain[draw_key][param], dtype=float))
            chains.append(np.stack(draws, axis=0))
        if not chains:
            raise ValueError("Posterior contains no chains")
        return np.stack(chains, axis=0)

    def beta_samples(self) -> np.ndarray:
        """Return Beta samples with shape chains x draws x covariates x species."""
        return self._samples("Beta")

    def gamma_samples(self) -> np.ndarray:
        return self._samples("Gamma")

    def sigma_samples(self) -> np.ndarray:
        return self._samples("sigma")

    def eta_samples(self, level: int = 0) -> np.ndarray:
        return self._random_level_samples("Eta", level)

    def lambda_samples(self, level: int = 0) -> np.ndarray:
        return self._random_level_samples("Lambda", level)

    def rho_samples(self) -> np.ndarray:
        return self._samples("rhoInd")

    def beta_mean(self) -> pd.DataFrame:
        beta = self.beta_samples().mean(axis=(0, 1))
        return self._beta_frame(beta)

    def gamma_mean(self) -> pd.DataFrame:
        gamma = self.gamma_samples().mean(axis=(0, 1))
        return pd.DataFrame(gamma)

    def sigma_mean(self) -> pd.Series:
        sigma = self.sigma_samples().mean(axis=(0, 1))
        return pd.Series(sigma, index=self._species_names(len(sigma)), name="sigma")

    def eta_mean(self, level: int = 0) -> pd.DataFrame:
        eta = self.eta_samples(level).mean(axis=(0, 1))
        return pd.DataFrame(eta)

    def lambda_mean(self, level: int = 0) -> pd.DataFrame:
        values = self.lambda_samples(level).mean(axis=(0, 1))
        return pd.DataFrame(values, columns=self._species_names(values.shape[-1]))

    def rho_mean(self) -> pd.Series:
        rho = self.rho_samples().mean(axis=(0, 1))
        return pd.Series(rho, index=[f"rho_{idx}" for idx in range(len(rho))], name="rho")

    def beta_ci(self, level: float = 0.95) -> dict[str, pd.DataFrame]:
        if not 0 < level < 1:
            raise ValueError("level must be between 0 and 1")
        beta = self.beta_samples()
        lo = np.quantile(beta, (1 - level) / 2, axis=(0, 1))
        hi = np.quantile(beta, 1 - (1 - level) / 2, axis=(0, 1))
        return {"lower": self._beta_frame(lo), "upper": self._beta_frame(hi)}

    def gamma_ci(self, level: float = 0.95) -> dict[str, pd.DataFrame]:
        return _ci_frames(self.gamma_samples(), level)

    def sigma_ci(self, level: float = 0.95) -> dict[str, pd.Series]:
        lo, hi = _ci_arrays(self.sigma_samples(), level)
        names = self._species_names(len(lo))
        return {"lower": pd.Series(lo, index=names), "upper": pd.Series(hi, index=names)}

    def predict_samples(
        self,
        X_new: Any,
        response: bool = True,
        random_effects: str = "none",
        unseen_groups: str = "error",
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("predict_samples requires the HmscModel used to create the fit")
        X_new = X_new if isinstance(X_new, pd.DataFrame) else pd.DataFrame(X_new)
        design = build_design_matrix(self.model.x_formula, X_new)
        beta_frame = self.beta_mean()
        missing = [column for column in beta_frame.index if column not in design.columns]
        missing_non_intercept = [column for column in missing if column != "Intercept"]
        if missing_non_intercept:
            raise ValueError(f"Prediction data is missing covariates: {missing_non_intercept}")
        for column in missing:
            design[column] = 1.0
        design = design.loc[:, beta_frame.index].to_numpy(dtype=float)
        linear = np.einsum("nk,cdks->cdns", design, self.beta_samples())
        if random_effects not in {"none", "known", "marginal"}:
            raise ValueError("random_effects must be 'none', 'known', or 'marginal'")
        if random_effects == "known":
            linear = linear + self._known_random_effect_prediction(X_new, unseen_groups=unseen_groups)
        elif random_effects == "marginal":
            linear = linear + self._marginal_random_effect_prediction(linear.shape[2])
        if response and self.model.distr.lower() == "poisson":
            linear = np.exp(linear)
        return linear

    def predict_mean(
        self,
        X_new: Any,
        response: bool = True,
        random_effects: str = "none",
        unseen_groups: str = "error",
    ) -> pd.DataFrame:
        samples = self.predict_samples(
            X_new,
            response=response,
            random_effects=random_effects,
            unseen_groups=unseen_groups,
        )
        values = samples.mean(axis=(0, 1))
        return pd.DataFrame(
            values,
            index=(X_new.index if isinstance(X_new, pd.DataFrame) else None),
            columns=self.beta_mean().columns,
        )

    def predict_ci(
        self,
        X_new: Any,
        level: float = 0.95,
        response: bool = True,
        random_effects: str = "none",
        unseen_groups: str = "error",
    ) -> dict[str, pd.DataFrame]:
        samples = self.predict_samples(
            X_new,
            response=response,
            random_effects=random_effects,
            unseen_groups=unseen_groups,
        )
        lo = np.quantile(samples, (1 - level) / 2, axis=(0, 1))
        hi = np.quantile(samples, 1 - (1 - level) / 2, axis=(0, 1))
        index = X_new.index if isinstance(X_new, pd.DataFrame) else None
        cols = self.beta_mean().columns
        return {"lower": pd.DataFrame(lo, index=index, columns=cols), "upper": pd.DataFrame(hi, index=index, columns=cols)}

    def to_arviz(self) -> Any:
        try:
            import arviz as az  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install arviz to use diagnostics") from exc
        posterior = {"Beta": self.beta_samples()}
        for param in ["Gamma", "sigma"]:
            try:
                posterior[param] = self._samples(param)
            except ValueError:
                pass
        return az.from_dict(posterior=posterior)

    def rhat(self, param: str = "Beta") -> Any:
        return self.to_arviz().posterior[param].to_numpy() if False else _arviz_stat(self, param, "rhat")

    def ess(self, param: str = "Beta") -> Any:
        return _arviz_stat(self, param, "ess")

    def traceplot(self, param: str = "Beta") -> Any:
        try:
            import arviz as az  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install arviz to use diagnostics") from exc
        return az.plot_trace(self.to_arviz(), var_names=[param])

    def summary(self, param: str = "Beta") -> pd.DataFrame:
        if param != "Beta":
            samples = self._samples(param)
            return pd.DataFrame(
                {
                    "mean": [float(samples.mean())],
                    "sd": [float(samples.std(ddof=1))],
                },
                index=[param],
            )
        mean = self.beta_mean()
        ci = self.beta_ci()
        rows = []
        for covariate in mean.index:
            for species in mean.columns:
                rows.append(
                    {
                        "covariate": covariate,
                        "species": species,
                        "mean": mean.loc[covariate, species],
                        "lower": ci["lower"].loc[covariate, species],
                        "upper": ci["upper"].loc[covariate, species],
                    }
                )
        return pd.DataFrame(rows)

    def predict(
        self,
        X_new: Any,
        response: bool = True,
        random_effects: str = "none",
        unseen_groups: str = "error",
        return_samples: bool = False,
    ) -> pd.DataFrame | np.ndarray:
        if return_samples:
            return self.predict_samples(
                X_new,
                response=response,
                random_effects=random_effects,
                unseen_groups=unseen_groups,
            )
        return self.predict_mean(
            X_new,
            response=response,
            random_effects=random_effects,
            unseen_groups=unseen_groups,
        )

    def _beta_frame(self, beta: np.ndarray) -> pd.DataFrame:
        covariates = None
        species = None
        if self.model is not None:
            covariates = getattr(self.model, "covariate_names", None)
            species = getattr(self.model, "species_names", None)
        covariates = _names_or_default(covariates, beta.shape[0], "covariate")
        species = _names_or_default(species, beta.shape[1], "species")
        return pd.DataFrame(beta, index=covariates, columns=species)

    def _random_level_samples(self, param: str, level: int) -> np.ndarray:
        if "__arrays__" in self.posterior:
            key = f"random_levels/{level}/{param}"
            arrays = self.posterior["__arrays__"]
            if key not in arrays:
                raise ValueError(f"Posterior does not contain {key!r}")
            return arrays[key]
        chains = []
        for chain_key in sorted(
            [key for key in self.posterior.keys() if str(key).isdigit()],
            key=lambda value: int(value),
        ):
            chain = self.posterior[chain_key]
            draws = []
            for draw_key in sorted(chain.keys(), key=lambda value: int(value)):
                draws.append(np.asarray(chain[draw_key][param][level], dtype=float))
            chains.append(np.stack(draws, axis=0))
        if not chains:
            raise ValueError("Posterior contains no chains")
        return np.stack(chains, axis=0)

    def _known_random_effect_prediction(self, X_new: pd.DataFrame, unseen_groups: str = "error") -> np.ndarray:
        if unseen_groups not in {"error", "zero", "sample", "nearest"}:
            raise ValueError("unseen_groups must be 'error', 'zero', 'sample', or 'nearest'")
        if self.model is None or not getattr(self.model, "random_levels", None):
            return 0
        total = 0
        for level_idx, (level_name, spec) in enumerate(self.model.random_levels.items()):
            column = spec.get("column", level_name)
            if column not in X_new:
                raise ValueError(f"Known random-effect prediction requires column {column!r}")
            if self.model.study_design is None:
                raise ValueError("Known random-effect prediction requires model.study_design")
            _, levels = pd.factorize(self.model.study_design[column], sort=True)
            mapping = {value: idx for idx, value in enumerate(levels)}
            codes = []
            zero_mask = []
            for row_idx, value in enumerate(X_new[column]):
                code = mapping.get(value)
                if code is None:
                    zero_mask.append(unseen_groups == "zero")
                    code = self._resolve_unseen_group(spec, X_new, row_idx, unseen_groups)
                else:
                    zero_mask.append(False)
                codes.append(code)
            eta = self.eta_samples(level_idx)[:, :, codes, :]
            lam = self.lambda_samples(level_idx)
            effect = np.einsum("cdnf,cdfs->cdns", eta, lam)
            if any(zero_mask):
                effect[:, :, zero_mask, :] = 0
            total = total + effect
        return total

    def _resolve_unseen_group(self, spec: dict[str, Any], X_new: pd.DataFrame, row_idx: int, mode: str) -> int:
        if mode == "error":
            raise ValueError("Prediction contains unknown random-effect group")
        if mode == "zero":
            return 0
        if mode == "sample":
            return row_idx % self.eta_samples(0).shape[2]
        if mode == "nearest":
            if spec.get("type") != "spatial_full":
                return row_idx % self.eta_samples(0).shape[2]
            coords = spec.get("coords", ["x", "y"])
            if self.model is None or self.model.study_design is None or any(col not in X_new for col in coords):
                raise ValueError("nearest unseen-group prediction requires coordinate columns")
            known = (
                self.model.study_design.groupby(spec.get("column", "plot"), sort=True)[coords]
                .mean()
                .to_numpy(dtype=float)
            )
            point = X_new.iloc[row_idx][coords].to_numpy(dtype=float)
            return int(np.argmin(np.sum((known - point) ** 2, axis=1)))
        raise ValueError(f"Unsupported unseen group mode {mode!r}")

    def _marginal_random_effect_prediction(self, n_new: int) -> np.ndarray:
        if self.model is None or not getattr(self.model, "random_levels", None):
            return 0
        total = 0
        for level_idx, _item in enumerate(self.model.random_levels.items()):
            eta = self.eta_samples(level_idx).mean(axis=2, keepdims=True)
            eta = np.repeat(eta, n_new, axis=2)
            lam = self.lambda_samples(level_idx)
            total = total + np.einsum("cdnf,cdfs->cdns", eta, lam)
        return total

    def _species_names(self, size: int) -> list[str]:
        if self.model is not None and getattr(self.model, "species_names", None):
            names = self.model.species_names
            if len(names) == size:
                return names
        return [f"species_{idx}" for idx in range(size)]


def _names_or_default(names: list[str] | None, size: int, prefix: str) -> list[str]:
    if names and len(names) == size:
        return names
    return [f"{prefix}_{idx}" for idx in range(size)]


def _read_hdf5_posterior(path: Path) -> dict[str, Any]:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install h5py to read HDF5 posterior files") from exc
    arrays = {}
    with h5py.File(path, "r") as handle:
        _read_hdf5_group(handle, arrays)
    return {"__arrays__": arrays}


def _read_hdf5_group(group: Any, arrays: dict[str, np.ndarray], prefix: str = "") -> None:
    for name, value in group.items():
        key = f"{prefix}/{name}" if prefix else name
        if hasattr(value, "keys"):
            _read_hdf5_group(value, arrays, key)
        else:
            arrays[key] = value[()]


def _read_zarr_posterior(path: Path) -> dict[str, Any]:
    try:
        import zarr  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install zarr to read Zarr posterior files") from exc
    arrays = {}
    root = zarr.open_group(str(path), mode="r")
    _read_zarr_group(root, arrays)
    return {"__arrays__": arrays}


def _read_zarr_group(group: Any, arrays: dict[str, np.ndarray], prefix: str = "") -> None:
    for name, value in group.items():
        key = f"{prefix}/{name}" if prefix else name
        if hasattr(value, "items"):
            _read_zarr_group(value, arrays, key)
        else:
            arrays[key] = value[:]


def _arviz_stat(fit: HmscFit, param: str, fn: str) -> Any:
    try:
        import arviz as az  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install arviz to use diagnostics") from exc
    data = fit.to_arviz()
    return getattr(az, fn)(data, var_names=[param])


def _ci_arrays(samples: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    return (
        np.quantile(samples, (1 - level) / 2, axis=(0, 1)),
        np.quantile(samples, 1 - (1 - level) / 2, axis=(0, 1)),
    )


def _ci_frames(samples: np.ndarray, level: float) -> dict[str, pd.DataFrame]:
    lo, hi = _ci_arrays(samples, level)
    return {"lower": pd.DataFrame(lo), "upper": pd.DataFrame(hi)}
