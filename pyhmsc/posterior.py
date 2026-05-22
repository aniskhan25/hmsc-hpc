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

    def beta_mean(self) -> pd.DataFrame:
        beta = self.beta_samples().mean(axis=(0, 1))
        return self._beta_frame(beta)

    def beta_ci(self, level: float = 0.95) -> dict[str, pd.DataFrame]:
        if not 0 < level < 1:
            raise ValueError("level must be between 0 and 1")
        beta = self.beta_samples()
        lo = np.quantile(beta, (1 - level) / 2, axis=(0, 1))
        hi = np.quantile(beta, 1 - (1 - level) / 2, axis=(0, 1))
        return {"lower": self._beta_frame(lo), "upper": self._beta_frame(hi)}

    def predict_samples(self, X_new: Any, response: bool = True) -> np.ndarray:
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
        if response and self.model.distr.lower() == "poisson":
            linear = np.exp(linear)
        return linear

    def predict_mean(self, X_new: Any, response: bool = True) -> pd.DataFrame:
        samples = self.predict_samples(X_new, response=response)
        values = samples.mean(axis=(0, 1))
        return pd.DataFrame(
            values,
            index=(X_new.index if isinstance(X_new, pd.DataFrame) else None),
            columns=self.beta_mean().columns,
        )

    def predict_ci(self, X_new: Any, level: float = 0.95, response: bool = True) -> dict[str, pd.DataFrame]:
        samples = self.predict_samples(X_new, response=response)
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
        return az.from_dict(posterior={"Beta": self.beta_samples()})

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

    def predict(self, X_new: Any, response: bool = True) -> pd.DataFrame:
        return self.predict_mean(X_new, response=response)

    def _beta_frame(self, beta: np.ndarray) -> pd.DataFrame:
        covariates = None
        species = None
        if self.model is not None:
            covariates = getattr(self.model, "covariate_names", None)
            species = getattr(self.model, "species_names", None)
        covariates = _names_or_default(covariates, beta.shape[0], "covariate")
        species = _names_or_default(species, beta.shape[1], "species")
        return pd.DataFrame(beta, index=covariates, columns=species)


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
        for name in handle.keys():
            arrays[name] = handle[name][()]
    return {"__arrays__": arrays}


def _arviz_stat(fit: HmscFit, param: str, fn: str) -> Any:
    try:
        import arviz as az  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install arviz to use diagnostics") from exc
    data = fit.to_arviz()
    return getattr(az, fn)(data, var_names=[param])
