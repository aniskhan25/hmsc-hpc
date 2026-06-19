"""Python posterior reader and convenience summaries."""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyhmsc.formulas import build_design_matrix


class HmscFit:
    def __init__(self, posterior: dict[str, Any], model: Any | None = None) -> None:
        self.posterior = posterior
        self.model = model
        self.metadata = posterior.get("__metadata__")
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

    def alpha_samples(self, level: int = 0) -> np.ndarray:
        """Return zero-based posterior spatial range-grid indices."""
        values = np.asarray(self._random_level_samples("Alpha", level), dtype=int)
        if values.ndim > 3 and values.shape[-1] == 1:
            values = np.squeeze(values, axis=-1)
        return values - 1

    def rho_samples(self) -> np.ndarray:
        return self._samples("rhoInd")

    def beta_mean(self) -> pd.DataFrame:
        beta = self.beta_samples().mean(axis=(0, 1))
        return self._beta_frame(beta)

    def gamma_mean(self) -> pd.DataFrame:
        gamma = self.gamma_samples().mean(axis=(0, 1))
        return self._gamma_frame(gamma)

    def sigma_mean(self) -> pd.Series:
        sigma = self.sigma_samples().mean(axis=(0, 1))
        return pd.Series(sigma, index=self._species_names(len(sigma)), name="sigma")

    def eta_mean(self, level: int = 0, align: bool = False) -> pd.DataFrame:
        eta = self._aligned_random_level_samples(level=level)[0] if align else self.eta_samples(level)
        eta = eta.mean(axis=(0, 1))
        return pd.DataFrame(eta, index=self._random_level_unit_names(level, eta.shape[0]), columns=_factor_names(eta.shape[1]))

    def lambda_mean(self, level: int = 0, align: bool = False) -> pd.DataFrame:
        values = self._aligned_random_level_samples(level=level)[1] if align else self.lambda_samples(level)
        values = values.mean(axis=(0, 1))
        if values.ndim != 2:
            raise ValueError("lambda_mean supports random intercept loadings only")
        return pd.DataFrame(values, index=_factor_names(values.shape[0]), columns=self._species_names(values.shape[-1]))

    def rho_mean(self) -> pd.Series:
        rho = self.rho_samples().mean(axis=(0, 1))
        return pd.Series(rho, index=[f"rho_{idx}" for idx in range(len(rho))], name="rho")

    def species_association_samples(
        self,
        level: int = 0,
        correlation: bool = True,
        x_index: int | None = None,
    ) -> np.ndarray:
        """Return residual species association samples from random-level loadings.

        The returned array has shape ``chains x draws x species x species``.
        For ordinary iid/spatial random intercepts, associations are computed as
        ``Lambda.T @ Lambda`` for each posterior draw. For random-slope loadings,
        pass ``x_index`` to select the random-slope covariate dimension.
        """
        loadings = self.lambda_samples(level)
        if loadings.ndim == 4:
            selected = loadings
        elif loadings.ndim == 5:
            if x_index is None:
                raise ValueError("x_index is required for random-slope Lambda samples")
            if not 0 <= x_index < loadings.shape[-1]:
                raise ValueError(f"x_index must be between 0 and {loadings.shape[-1] - 1}")
            selected = loadings[..., x_index]
        else:
            raise ValueError(
                "Lambda samples must have shape chains x draws x factors x species "
                "or chains x draws x factors x species x random_covariates"
            )
        covariance = np.einsum("cdfi,cdfj->cdij", selected, selected)
        if not correlation:
            return covariance
        diagonal = np.diagonal(covariance, axis1=-2, axis2=-1)
        scale = np.sqrt(np.maximum(diagonal[..., :, None] * diagonal[..., None, :], np.finfo(float).eps))
        correlation_samples = covariance / scale
        species_count = correlation_samples.shape[-1]
        diag_idx = np.arange(species_count)
        correlation_samples[..., diag_idx, diag_idx] = 1.0
        return correlation_samples

    def species_associations(
        self,
        level: int = 0,
        correlation: bool = True,
        x_index: int | None = None,
    ) -> pd.DataFrame:
        """Return mean residual species association matrix."""
        samples = self.species_association_samples(
            level=level,
            correlation=correlation,
            x_index=x_index,
        )
        mean = samples.mean(axis=(0, 1))
        species = self._species_names(mean.shape[0])
        return pd.DataFrame(mean, index=species, columns=species)

    def species_association_ci(
        self,
        level: int = 0,
        cred_level: float = 0.95,
        correlation: bool = True,
        x_index: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Return credible interval matrices for residual species associations."""
        if not 0 < cred_level < 1:
            raise ValueError("cred_level must be between 0 and 1")
        samples = self.species_association_samples(
            level=level,
            correlation=correlation,
            x_index=x_index,
        )
        alpha = (1 - cred_level) / 2
        lo = np.quantile(samples, alpha, axis=(0, 1))
        hi = np.quantile(samples, 1 - alpha, axis=(0, 1))
        species = self._species_names(lo.shape[0])
        return {
            "lower": pd.DataFrame(lo, index=species, columns=species),
            "upper": pd.DataFrame(hi, index=species, columns=species),
        }

    def species_association_summary(
        self,
        level: int = 0,
        cred_level: float = 0.95,
        correlation: bool = True,
        x_index: int | None = None,
        include_self: bool = False,
    ) -> pd.DataFrame:
        """Return pairwise residual species associations with intervals."""
        if not 0 < cred_level < 1:
            raise ValueError("cred_level must be between 0 and 1")
        samples = self.species_association_samples(
            level=level,
            correlation=correlation,
            x_index=x_index,
        )
        flattened = samples.reshape((-1,) + samples.shape[2:])
        mean = flattened.mean(axis=0)
        alpha = (1 - cred_level) / 2
        lo = np.quantile(flattened, alpha, axis=0)
        hi = np.quantile(flattened, 1 - alpha, axis=0)
        p_positive = (flattened > 0).mean(axis=0)
        p_negative = (flattened < 0).mean(axis=0)
        species = self._species_names(mean.shape[0])
        rows = []
        for left_idx, left in enumerate(species):
            start = left_idx if include_self else left_idx + 1
            for right_idx in range(start, len(species)):
                rows.append(
                    {
                        "species_1": left,
                        "species_2": species[right_idx],
                        "mean": float(mean[left_idx, right_idx]),
                        "lower": float(lo[left_idx, right_idx]),
                        "upper": float(hi[left_idx, right_idx]),
                        "p_positive": float(p_positive[left_idx, right_idx]),
                        "p_negative": float(p_negative[left_idx, right_idx]),
                    }
                )
        return pd.DataFrame(rows)

    def species_association_diagnostics(
        self,
        level: int = 0,
        correlation: bool = True,
        x_index: int | None = None,
        include_self: bool = False,
    ) -> pd.DataFrame:
        """Return diagnostics for identifiable residual species associations."""
        samples = self.species_association_samples(
            level=level,
            correlation=correlation,
            x_index=x_index,
        )
        mean = samples.mean(axis=(0, 1))
        sd = samples.reshape((-1,) + samples.shape[2:]).std(axis=0, ddof=1)
        rhat = _rhat_array(samples)
        ess = _ess_array(samples)
        species = self._species_names(mean.shape[0])
        rows = []
        for left_idx, left in enumerate(species):
            start = left_idx if include_self else left_idx + 1
            for right_idx in range(start, len(species)):
                row = {
                    "random_level": level,
                    "species_1": left,
                    "species_2": species[right_idx],
                    "mean": float(mean[left_idx, right_idx]),
                    "sd": float(sd[left_idx, right_idx]),
                    "rhat": float(rhat[left_idx, right_idx]),
                    "ess": float(ess[left_idx, right_idx]),
                }
                if x_index is not None:
                    row["x_index"] = x_index
                rows.append(row)
        return pd.DataFrame(rows)

    def species_association_diagnostics_overview(
        self,
        level: int = 0,
        correlation: bool = True,
        x_index: int | None = None,
        include_self: bool = False,
        rhat_threshold: float = 1.01,
        ess_threshold: float = 400.0,
    ) -> dict[str, Any]:
        diagnostics = self.species_association_diagnostics(
            level=level,
            correlation=correlation,
            x_index=x_index,
            include_self=include_self,
        )
        return _diagnostics_overview(
            diagnostics,
            param="Associations",
            rhat_threshold=rhat_threshold,
            ess_threshold=ess_threshold,
            extra={
                "random_level": int(level),
                "association": "correlation" if correlation else "covariance",
                **({"x_index": int(x_index)} if x_index is not None else {}),
            },
        )

    def beta_ci(self, level: float = 0.95) -> dict[str, pd.DataFrame]:
        if not 0 < level < 1:
            raise ValueError("level must be between 0 and 1")
        beta = self.beta_samples()
        lo = np.quantile(beta, (1 - level) / 2, axis=(0, 1))
        hi = np.quantile(beta, 1 - (1 - level) / 2, axis=(0, 1))
        return {"lower": self._beta_frame(lo), "upper": self._beta_frame(hi)}

    def gamma_ci(self, level: float = 0.95) -> dict[str, pd.DataFrame]:
        lo, hi = _ci_arrays(self.gamma_samples(), level)
        return {"lower": self._gamma_frame(lo), "upper": self._gamma_frame(hi)}

    def sigma_ci(self, level: float = 0.95) -> dict[str, pd.Series]:
        lo, hi = _ci_arrays(self.sigma_samples(), level)
        names = self._species_names(len(lo))
        return {"lower": pd.Series(lo, index=names), "upper": pd.Series(hi, index=names)}

    def eta_ci(self, level: int = 0, cred_level: float = 0.95, align: bool = False) -> dict[str, pd.DataFrame]:
        samples = self._aligned_random_level_samples(level=level)[0] if align else self.eta_samples(level)
        lo, hi = _ci_arrays(samples, cred_level)
        return {
            "lower": self._eta_frame(lo, level),
            "upper": self._eta_frame(hi, level),
        }

    def lambda_ci(
        self,
        level: int = 0,
        cred_level: float = 0.95,
        x_index: int | None = None,
        align: bool = False,
    ) -> dict[str, pd.DataFrame]:
        samples = (
            self._aligned_random_level_samples(level=level, x_index=x_index)[1]
            if align
            else self._lambda_samples_for_summary(level=level, x_index=x_index)
        )
        lo, hi = _ci_arrays(samples, cred_level)
        return {
            "lower": self._lambda_frame(lo),
            "upper": self._lambda_frame(hi),
        }

    def predict_samples(
        self,
        X_new: Any,
        response: bool = True,
        random_effects: str = "none",
        unseen_groups: str = "error",
        study_design: Any | None = None,
        coords: Any | None = None,
        include_random_effects: bool | None = None,
        spatial_prediction: str = "nearest",
        rng_seed: int | None = None,
    ) -> np.ndarray:
        if self.model is None:
            if self._x_formula() is None:
                raise ValueError(
                    "predict_samples requires the HmscModel used to create the fit "
                    "or embedded posterior metadata with formula.X"
                )
        random_effects = _resolve_random_effect_mode(random_effects, include_random_effects)
        X_new = _prediction_frame(X_new, study_design=study_design, coords=coords)
        design = build_design_matrix(self._x_formula(), X_new)
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
            linear = linear + self._known_random_effect_prediction(
                X_new,
                unseen_groups=unseen_groups,
                spatial_prediction=spatial_prediction,
                rng=np.random.default_rng(rng_seed),
            )
        elif random_effects == "marginal":
            linear = linear + self._marginal_random_effect_prediction(linear.shape[2])
        return _response_scale(linear, self._distribution()) if response else linear

    def predict_mean(
        self,
        X_new: Any,
        response: bool = True,
        random_effects: str = "none",
        unseen_groups: str = "error",
        study_design: Any | None = None,
        coords: Any | None = None,
        include_random_effects: bool | None = None,
        spatial_prediction: str = "nearest",
        rng_seed: int | None = None,
    ) -> pd.DataFrame:
        X_frame = _prediction_frame(X_new, study_design=study_design, coords=coords)
        samples = self.predict_samples(
            X_frame,
            response=response,
            random_effects=random_effects,
            unseen_groups=unseen_groups,
            include_random_effects=include_random_effects,
            spatial_prediction=spatial_prediction,
            rng_seed=rng_seed,
        )
        values = samples.mean(axis=(0, 1))
        return pd.DataFrame(
            values,
            index=X_frame.index,
            columns=self.beta_mean().columns,
        )

    def predict_ci(
        self,
        X_new: Any,
        level: float = 0.95,
        response: bool = True,
        random_effects: str = "none",
        unseen_groups: str = "error",
        study_design: Any | None = None,
        coords: Any | None = None,
        include_random_effects: bool | None = None,
        spatial_prediction: str = "nearest",
        rng_seed: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        X_frame = _prediction_frame(X_new, study_design=study_design, coords=coords)
        samples = self.predict_samples(
            X_frame,
            response=response,
            random_effects=random_effects,
            unseen_groups=unseen_groups,
            include_random_effects=include_random_effects,
            spatial_prediction=spatial_prediction,
            rng_seed=rng_seed,
        )
        lo = np.quantile(samples, (1 - level) / 2, axis=(0, 1))
        hi = np.quantile(samples, 1 - (1 - level) / 2, axis=(0, 1))
        cols = self.beta_mean().columns
        return {"lower": pd.DataFrame(lo, index=X_frame.index, columns=cols), "upper": pd.DataFrame(hi, index=X_frame.index, columns=cols)}

    def posterior_predictive(
        self,
        X_new: Any,
        random_effects: str = "none",
        unseen_groups: str = "error",
        study_design: Any | None = None,
        coords: Any | None = None,
        include_random_effects: bool | None = None,
        spatial_prediction: str = "nearest",
        rng_seed: int | None = None,
    ) -> np.ndarray:
        """Draw replicated responses from the posterior predictive distribution."""
        rng = np.random.default_rng(rng_seed)
        prediction_seed = None
        if rng_seed is not None:
            prediction_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        random_effects = _resolve_random_effect_mode(random_effects, include_random_effects)
        X_frame = _prediction_frame(X_new, study_design=study_design, coords=coords)
        distribution = self._distribution().lower()
        if distribution in {"normal", "gaussian"}:
            mean = self.predict_samples(
                X_frame,
                response=False,
                random_effects=random_effects,
                unseen_groups=unseen_groups,
                spatial_prediction=spatial_prediction,
                rng_seed=prediction_seed,
            )
            sigma = np.maximum(self.sigma_samples(), np.finfo(float).eps)
            return rng.normal(loc=mean, scale=sigma[:, :, None, :])
        if distribution == "poisson":
            rate = self.predict_samples(
                X_frame,
                response=True,
                random_effects=random_effects,
                unseen_groups=unseen_groups,
                spatial_prediction=spatial_prediction,
                rng_seed=prediction_seed,
            )
            return rng.poisson(np.clip(rate, 0.0, 1e12))
        if distribution in {"probit", "bernoulli", "binomial"}:
            probability = self.predict_samples(
                X_frame,
                response=True,
                random_effects=random_effects,
                unseen_groups=unseen_groups,
                spatial_prediction=spatial_prediction,
                rng_seed=prediction_seed,
            )
            return rng.binomial(1, probability)
        raise ValueError(f"Posterior predictive checks do not support distribution {distribution!r}")

    def posterior_predictive_summary(
        self,
        Y: Any,
        X: Any,
        level: float = 0.95,
        random_effects: str = "none",
        unseen_groups: str = "error",
        study_design: Any | None = None,
        coords: Any | None = None,
        include_random_effects: bool | None = None,
        spatial_prediction: str = "nearest",
        rng_seed: int | None = None,
    ) -> pd.DataFrame:
        """Summarize observed species means against replicated posterior means."""
        if not 0 < level < 1:
            raise ValueError("level must be between 0 and 1")
        y_frame = Y if isinstance(Y, pd.DataFrame) else pd.DataFrame(Y)
        samples = self.posterior_predictive(
            X,
            random_effects=random_effects,
            unseen_groups=unseen_groups,
            study_design=study_design,
            coords=coords,
            include_random_effects=include_random_effects,
            spatial_prediction=spatial_prediction,
            rng_seed=rng_seed,
        )
        species = self._species_names(samples.shape[-1])
        missing = [name for name in species if name not in y_frame.columns]
        if missing:
            raise ValueError(f"Observed Y is missing species columns: {missing}")
        observed = y_frame.loc[:, species].mean(axis=0).to_numpy(dtype=float)
        replicated_means = samples.mean(axis=2).reshape(-1, samples.shape[-1])
        predicted = replicated_means.mean(axis=0)
        lo = np.quantile(replicated_means, (1 - level) / 2, axis=0)
        hi = np.quantile(replicated_means, 1 - (1 - level) / 2, axis=0)
        return pd.DataFrame(
            {
                "species": species,
                "observed_mean": observed,
                "replicated_mean": predicted,
                "lower": lo,
                "upper": hi,
                "covered": (lo <= observed) & (observed <= hi),
            }
        )

    def site_richness_posterior_predictive_summary(
        self,
        Y: Any,
        X: Any,
        level: float = 0.95,
        random_effects: str = "none",
        unseen_groups: str = "error",
        study_design: Any | None = None,
        coords: Any | None = None,
        include_random_effects: bool | None = None,
        spatial_prediction: str = "nearest",
        rng_seed: int | None = None,
    ) -> pd.DataFrame:
        """Summarize observed site richness against replicated posterior richness."""
        if not 0 < level < 1:
            raise ValueError("level must be between 0 and 1")
        y_frame = Y if isinstance(Y, pd.DataFrame) else pd.DataFrame(Y)
        samples = self.posterior_predictive(
            X,
            random_effects=random_effects,
            unseen_groups=unseen_groups,
            study_design=study_design,
            coords=coords,
            include_random_effects=include_random_effects,
            spatial_prediction=spatial_prediction,
            rng_seed=rng_seed,
        )
        observed = y_frame.sum(axis=1).to_numpy(dtype=float)
        replicated_richness = samples.sum(axis=-1).reshape(-1, samples.shape[2])
        predicted = replicated_richness.mean(axis=0)
        lo = np.quantile(replicated_richness, (1 - level) / 2, axis=0)
        hi = np.quantile(replicated_richness, 1 - (1 - level) / 2, axis=0)
        return pd.DataFrame(
            {
                "site": [str(value) for value in y_frame.index],
                "observed_richness": observed,
                "replicated_richness": predicted,
                "lower": lo,
                "upper": hi,
                "covered": (lo <= observed) & (observed <= hi),
            }
        )

    def ppc_summary(
        self,
        Y: Any,
        X: Any,
        level: float = 0.95,
        random_effects: str = "none",
        unseen_groups: str = "error",
        study_design: Any | None = None,
        coords: Any | None = None,
        include_random_effects: bool | None = None,
        spatial_prediction: str = "nearest",
        rng_seed: int | None = None,
    ) -> pd.DataFrame:
        return self.posterior_predictive_summary(
            Y=Y,
            X=X,
            level=level,
            random_effects=random_effects,
            unseen_groups=unseen_groups,
            study_design=study_design,
            coords=coords,
            include_random_effects=include_random_effects,
            spatial_prediction=spatial_prediction,
            rng_seed=rng_seed,
        )

    def richness_ppc_summary(
        self,
        Y: Any,
        X: Any,
        level: float = 0.95,
        random_effects: str = "none",
        unseen_groups: str = "error",
        study_design: Any | None = None,
        coords: Any | None = None,
        include_random_effects: bool | None = None,
        spatial_prediction: str = "nearest",
        rng_seed: int | None = None,
    ) -> pd.DataFrame:
        return self.site_richness_posterior_predictive_summary(
            Y=Y,
            X=X,
            level=level,
            random_effects=random_effects,
            unseen_groups=unseen_groups,
            study_design=study_design,
            coords=coords,
            include_random_effects=include_random_effects,
            spatial_prediction=spatial_prediction,
            rng_seed=rng_seed,
        )

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

    def rhat(self, param: str = "Beta", level: int = 0, x_index: int | None = None) -> Any:
        samples = self._diagnostic_samples(param, level=level, x_index=x_index)
        return self._diagnostic_frame(param, _rhat_array(samples), "rhat", level=level, x_index=x_index)

    def ess(self, param: str = "Beta", level: int = 0, x_index: int | None = None) -> Any:
        samples = self._diagnostic_samples(param, level=level, x_index=x_index)
        return self._diagnostic_frame(param, _ess_array(samples), "ess", level=level, x_index=x_index)

    def diagnostics(
        self,
        param: str = "Beta",
        level: int = 0,
        x_index: int | None = None,
        correlation: bool = True,
        include_self: bool = False,
        align: bool = False,
    ) -> pd.DataFrame:
        if _is_association_param(param):
            return self.species_association_diagnostics(
                level=level,
                correlation=correlation,
                x_index=x_index,
                include_self=include_self,
            )
        samples = self._diagnostic_samples(param, level=level, x_index=x_index, align=align)
        mean = samples.mean(axis=(0, 1))
        sd = samples.reshape((-1,) + samples.shape[2:]).std(axis=0, ddof=1)
        rhat = _rhat_array(samples)
        ess = _ess_array(samples)
        labels = self._diagnostic_labels(param, mean.shape, level=level, x_index=x_index)
        rows = []
        for flat_idx, index in enumerate(np.ndindex(mean.shape)):
            row = dict(labels[flat_idx])
            row.update(
                {
                    "mean": float(mean[index]),
                    "sd": float(sd[index]),
                    "rhat": float(rhat[index]),
                    "ess": float(ess[index]),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def diagnostics_overview(
        self,
        param: str = "Beta",
        rhat_threshold: float = 1.01,
        ess_threshold: float = 400.0,
        level: int = 0,
        x_index: int | None = None,
        correlation: bool = True,
        include_self: bool = False,
        align: bool = False,
    ) -> dict[str, Any]:
        diagnostics = self.diagnostics(
            param,
            level=level,
            x_index=x_index,
            correlation=correlation,
            include_self=include_self,
            align=align,
        )
        overview = _diagnostics_overview(
            diagnostics,
            param="Associations" if _is_association_param(param) else param,
            rhat_threshold=rhat_threshold,
            ess_threshold=ess_threshold,
        )
        if param in {"Eta", "Lambda"}:
            overview["random_level"] = int(level)
            overview["aligned"] = bool(align)
        if param == "Lambda" and x_index is not None:
            overview["x_index"] = int(x_index)
        if _is_association_param(param):
            overview["random_level"] = int(level)
            overview["association"] = "correlation" if correlation else "covariance"
            if x_index is not None:
                overview["x_index"] = int(x_index)
        return overview

    def traceplot(self, param: str = "Beta") -> Any:
        try:
            import arviz as az  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install arviz to use diagnostics") from exc
        return az.plot_trace(self.to_arviz(), var_names=[param])

    def summary(self, param: str = "Beta", level: float = 0.95) -> pd.DataFrame:
        if param == "Beta":
            return self.beta_summary(level=level)
        if param == "Gamma":
            return self.gamma_summary(level=level)
        if param == "Eta":
            return self.eta_summary(cred_level=level)
        if param == "Lambda":
            return self.lambda_summary(cred_level=level)
        samples = self._samples(param)
        return pd.DataFrame(
            {
                "mean": [float(samples.mean())],
                "sd": [float(samples.std(ddof=1))],
            },
            index=[param],
        )

    def beta_summary(self, level: float = 0.95) -> pd.DataFrame:
        mean = self.beta_mean()
        ci = self.beta_ci(level=level)
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

    def gamma_summary(self, level: float = 0.95) -> pd.DataFrame:
        mean = self.gamma_mean()
        ci = self.gamma_ci(level=level)
        rows = []
        for covariate in mean.index:
            for trait in mean.columns:
                rows.append(
                    {
                        "covariate": covariate,
                        "trait": trait,
                        "mean": mean.loc[covariate, trait],
                        "lower": ci["lower"].loc[covariate, trait],
                        "upper": ci["upper"].loc[covariate, trait],
                    }
                )
        return pd.DataFrame(rows)

    def eta_summary(self, level: int = 0, cred_level: float = 0.95, align: bool = False) -> pd.DataFrame:
        mean = self.eta_mean(level=level, align=align)
        ci = self.eta_ci(level=level, cred_level=cred_level, align=align)
        rows = []
        for unit in mean.index:
            for factor in mean.columns:
                row = {
                    "random_level": level,
                    "unit": unit,
                    "factor": factor,
                    "mean": mean.loc[unit, factor],
                    "lower": ci["lower"].loc[unit, factor],
                    "upper": ci["upper"].loc[unit, factor],
                }
                if align:
                    row["aligned"] = True
                rows.append(row)
        return pd.DataFrame(rows)

    def lambda_summary(
        self,
        level: int = 0,
        cred_level: float = 0.95,
        x_index: int | None = None,
        align: bool = False,
    ) -> pd.DataFrame:
        samples = (
            self._aligned_random_level_samples(level=level, x_index=x_index)[1]
            if align
            else self._lambda_samples_for_summary(level=level, x_index=x_index)
        )
        mean = self._lambda_frame(samples.mean(axis=(0, 1)))
        ci = self.lambda_ci(level=level, cred_level=cred_level, x_index=x_index, align=align)
        rows = []
        for factor in mean.index:
            for species in mean.columns:
                row = {
                    "random_level": level,
                    "factor": factor,
                    "species": species,
                    "mean": mean.loc[factor, species],
                    "lower": ci["lower"].loc[factor, species],
                    "upper": ci["upper"].loc[factor, species],
                }
                if x_index is not None:
                    row["x_index"] = x_index
                if align:
                    row["aligned"] = True
                rows.append(row)
        return pd.DataFrame(rows)

    def predict(
        self,
        X_new: Any,
        response: bool = True,
        random_effects: str = "none",
        unseen_groups: str = "error",
        study_design: Any | None = None,
        coords: Any | None = None,
        include_random_effects: bool | None = None,
        spatial_prediction: str = "nearest",
        rng_seed: int | None = None,
        return_samples: bool = False,
    ) -> pd.DataFrame | np.ndarray:
        if return_samples:
            return self.predict_samples(
                X_new,
                response=response,
                random_effects=random_effects,
                unseen_groups=unseen_groups,
                study_design=study_design,
                coords=coords,
                include_random_effects=include_random_effects,
                spatial_prediction=spatial_prediction,
                rng_seed=rng_seed,
            )
        return self.predict_mean(
            X_new,
            response=response,
            random_effects=random_effects,
            unseen_groups=unseen_groups,
            study_design=study_design,
            coords=coords,
            include_random_effects=include_random_effects,
            spatial_prediction=spatial_prediction,
            rng_seed=rng_seed,
        )

    def gradient(
        self,
        variable: str,
        X_reference: Any,
        values: Any | None = None,
        n: int = 25,
    ) -> pd.DataFrame:
        """Construct a one-variable prediction gradient from reference covariates."""
        x_frame = X_reference if isinstance(X_reference, pd.DataFrame) else pd.DataFrame(X_reference)
        if variable not in x_frame.columns:
            raise ValueError(f"X_reference is missing gradient variable {variable!r}")
        if values is None:
            if not pd.api.types.is_numeric_dtype(x_frame[variable]):
                raise ValueError("values must be provided for non-numeric gradient variables")
            values = np.linspace(float(x_frame[variable].min()), float(x_frame[variable].max()), n)
        values = list(values)
        if not values:
            raise ValueError("gradient values must not be empty")
        row: dict[str, Any] = {}
        for column in x_frame.columns:
            series = x_frame[column].dropna()
            if series.empty:
                row[column] = np.nan
            elif pd.api.types.is_numeric_dtype(series):
                row[column] = float(series.mean())
            else:
                row[column] = series.mode().iloc[0]
        gradient = pd.DataFrame([row.copy() for _ in values])
        gradient[variable] = values
        return gradient

    def richness_gradient(
        self,
        variable: str,
        X_reference: Any,
        values: Any | None = None,
        n: int = 25,
        level: float = 0.95,
        random_effects: str = "none",
        unseen_groups: str = "error",
    ) -> pd.DataFrame:
        """Summarize expected species richness along a covariate gradient."""
        gradient = self.gradient(variable, X_reference, values=values, n=n)
        samples = self.predict_samples(
            gradient,
            response=True,
            random_effects=random_effects,
            unseen_groups=unseen_groups,
        ).sum(axis=-1)
        return _gradient_summary_frame(gradient[variable], samples, level)

    def trait_weighted_gradient(
        self,
        variable: str,
        traits: Any,
        trait: str,
        X_reference: Any,
        values: Any | None = None,
        n: int = 25,
        level: float = 0.95,
        random_effects: str = "none",
        unseen_groups: str = "error",
    ) -> pd.DataFrame:
        """Summarize a response-weighted trait mean along a covariate gradient."""
        gradient = self.gradient(variable, X_reference, values=values, n=n)
        trait_frame = traits if isinstance(traits, pd.DataFrame) else pd.DataFrame(traits)
        species = self.beta_mean().columns
        missing = [name for name in species if name not in trait_frame.index]
        if missing:
            raise ValueError(f"traits missing species rows: {missing}")
        if trait not in trait_frame.columns:
            raise ValueError(f"traits is missing column {trait!r}")
        weights = self.predict_samples(
            gradient,
            response=True,
            random_effects=random_effects,
            unseen_groups=unseen_groups,
        )
        trait_values = trait_frame.loc[species, trait].to_numpy(dtype=float)
        samples = (weights @ trait_values) / np.maximum(weights.sum(axis=-1), 1e-12)
        return _gradient_summary_frame(gradient[variable], samples, level)

    def _beta_frame(self, beta: np.ndarray) -> pd.DataFrame:
        covariates = None
        species = None
        if self.model is not None:
            covariates = getattr(self.model, "covariate_names", None)
            species = getattr(self.model, "species_names", None)
        if (
            covariates is None
            or len(covariates) != beta.shape[0]
            or species is None
            or len(species) != beta.shape[1]
        ):
            names = self._metadata_names()
            if covariates is None or len(covariates) != beta.shape[0]:
                covariates = names.get("covariates")
            if species is None or len(species) != beta.shape[1]:
                species = names.get("species")
        covariates = _names_or_default(covariates, beta.shape[0], "covariate")
        species = _names_or_default(species, beta.shape[1], "species")
        return pd.DataFrame(beta, index=covariates, columns=species)

    def _gamma_frame(self, gamma: np.ndarray) -> pd.DataFrame:
        covariates = None
        traits = None
        if self.model is not None:
            covariates = getattr(self.model, "covariate_names", None)
        names = self._metadata_names()
        if covariates is None or len(covariates) != gamma.shape[0]:
            covariates = names.get("covariates")
        if traits is None or len(traits) != gamma.shape[1]:
            traits = names.get("traits")
        covariates = _names_or_default(covariates, gamma.shape[0], "covariate")
        traits = _names_or_default(traits, gamma.shape[1], "trait")
        return pd.DataFrame(gamma, index=covariates, columns=traits)

    def _eta_frame(self, eta: np.ndarray, level: int) -> pd.DataFrame:
        return pd.DataFrame(
            eta,
            index=self._random_level_unit_names(level, eta.shape[0]),
            columns=_factor_names(eta.shape[1]),
        )

    def _lambda_frame(self, values: np.ndarray) -> pd.DataFrame:
        if values.ndim != 2:
            raise ValueError("Lambda summaries support one loading matrix at a time")
        return pd.DataFrame(
            values,
            index=_factor_names(values.shape[0]),
            columns=self._species_names(values.shape[1]),
        )

    def _lambda_samples_for_summary(self, level: int = 0, x_index: int | None = None) -> np.ndarray:
        samples = self.lambda_samples(level)
        if samples.ndim == 4:
            return samples
        if samples.ndim == 5:
            if x_index is None:
                raise ValueError("x_index is required for random-slope Lambda summaries")
            if not 0 <= x_index < samples.shape[-1]:
                raise ValueError(f"x_index must be between 0 and {samples.shape[-1] - 1}")
            return samples[..., x_index]
        raise ValueError(
            "Lambda samples must have shape chains x draws x factors x species "
            "or chains x draws x factors x species x random_covariates"
        )

    def _diagnostic_samples(
        self,
        param: str,
        level: int = 0,
        x_index: int | None = None,
        align: bool = False,
    ) -> np.ndarray:
        if param == "Eta":
            return self._aligned_random_level_samples(level=level, x_index=x_index)[0] if align else self.eta_samples(level)
        if param == "Lambda":
            return (
                self._aligned_random_level_samples(level=level, x_index=x_index)[1]
                if align
                else self._lambda_samples_for_summary(level=level, x_index=x_index)
            )
        return self._samples(param)

    def _aligned_random_level_samples(
        self,
        level: int = 0,
        x_index: int | None = None,
        max_iter: int = 2,
    ) -> tuple[np.ndarray, np.ndarray]:
        eta = self.eta_samples(level)
        if x_index is None and self.lambda_samples(level).ndim == 5:
            x_index = 0
        lam = self._lambda_samples_for_summary(level=level, x_index=x_index)
        if eta.ndim != 4:
            raise ValueError("Eta alignment requires samples with shape chains x draws x units x factors")
        if lam.ndim != 4:
            raise ValueError("Lambda alignment requires one loading matrix per draw")
        if eta.shape[:2] != lam.shape[:2] or eta.shape[-1] != lam.shape[2]:
            raise ValueError("Eta and Lambda samples have incompatible factor dimensions")
        return _align_eta_lambda_samples(eta, lam, max_iter=max_iter)

    def _diagnostic_frame(
        self,
        param: str,
        values: np.ndarray,
        value_name: str,
        level: int = 0,
        x_index: int | None = None,
    ) -> pd.DataFrame:
        labels = self._diagnostic_labels(param, values.shape, level=level, x_index=x_index)
        rows = []
        for flat_idx, index in enumerate(np.ndindex(values.shape)):
            row = dict(labels[flat_idx])
            row[value_name] = float(values[index])
            rows.append(row)
        return pd.DataFrame(rows)

    def _diagnostic_labels(
        self,
        param: str,
        shape: tuple[int, ...],
        level: int = 0,
        x_index: int | None = None,
    ) -> list[dict[str, Any]]:
        if param == "Beta" and len(shape) == 2:
            frame = self._beta_frame(np.zeros(shape))
            return [
                {"covariate": covariate, "species": species}
                for covariate in frame.index
                for species in frame.columns
            ]
        if param == "Gamma" and len(shape) == 2:
            frame = self._gamma_frame(np.zeros(shape))
            return [
                {"covariate": covariate, "trait": trait}
                for covariate in frame.index
                for trait in frame.columns
            ]
        if param == "Eta" and len(shape) == 2:
            frame = self._eta_frame(np.zeros(shape), level=level)
            return [
                {"random_level": level, "unit": unit, "factor": factor}
                for unit in frame.index
                for factor in frame.columns
            ]
        if param == "Lambda" and len(shape) == 2:
            frame = self._lambda_frame(np.zeros(shape))
            labels = [
                {"random_level": level, "factor": factor, "species": species}
                for factor in frame.index
                for species in frame.columns
            ]
            if x_index is not None:
                for label in labels:
                    label["x_index"] = x_index
            return labels
        return [
            {f"dim_{axis}": int(value) for axis, value in enumerate(index)}
            for index in np.ndindex(shape)
        ]

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

    def _random_level_unit_names(self, level: int, size: int) -> list[str]:
        if self.model is not None and getattr(self.model, "random_levels", None):
            try:
                level_name, spec = list(self.model.random_levels.items())[level]
            except IndexError:
                pass
            else:
                column = spec.get("column", level_name)
                if self.model.study_design is not None and column in self.model.study_design:
                    _, levels = pd.factorize(self.model.study_design[column], sort=True)
                    names = [str(value) for value in levels]
                    if len(names) == size:
                        return names
        metadata = self.metadata if isinstance(self.metadata, dict) else {}
        random_levels = metadata.get("random_levels", [])
        if isinstance(random_levels, list) and 0 <= level < len(random_levels):
            names = random_levels[level].get("levels")
            if isinstance(names, list) and len(names) == size:
                return [str(value) for value in names]
        return [f"unit_{idx}" for idx in range(size)]

    def _known_random_effect_prediction(
        self,
        X_new: pd.DataFrame,
        unseen_groups: str = "error",
        spatial_prediction: str = "nearest",
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if unseen_groups not in {"error", "zero", "sample", "nearest"}:
            raise ValueError("unseen_groups must be 'error', 'zero', 'sample', or 'nearest'")
        if spatial_prediction not in {"nearest", "conditional"}:
            raise ValueError("spatial_prediction must be 'nearest' or 'conditional'")
        if self.model is None or not getattr(self.model, "random_levels", None):
            return 0
        rng = rng or np.random.default_rng()
        total = 0
        for level_idx, (level_name, spec) in enumerate(self.model.random_levels.items()):
            spec = self._prediction_random_level_spec(level_idx, spec)
            column = spec.get("column", level_name)
            if column not in X_new:
                raise ValueError(f"Known random-effect prediction requires column {column!r}")
            if self.model.study_design is None:
                raise ValueError("Known random-effect prediction requires model.study_design")
            _, levels = pd.factorize(self.model.study_design[column], sort=True)
            mapping = {value: idx for idx, value in enumerate(levels)}
            raw_codes = np.asarray([mapping.get(value, -1) for value in X_new[column]], dtype=int)
            if spatial_prediction == "conditional" and np.any(raw_codes < 0):
                eta = self._conditional_spatial_eta_samples(level_idx, spec, X_new, raw_codes, rng)
                zero_mask = [False] * len(X_new)
            else:
                codes = []
                zero_mask = []
                for row_idx, code in enumerate(raw_codes):
                    if code < 0:
                        zero_mask.append(unseen_groups == "zero")
                        code = self._resolve_unseen_group(spec, X_new, row_idx, unseen_groups)
                    else:
                        zero_mask.append(False)
                    codes.append(code)
                eta = self.eta_samples(level_idx)[:, :, codes, :]
            lam = self.lambda_samples(level_idx)
            if lam.ndim == 5:
                x_mat = self._random_slope_design_for_groups(spec, X_new)
                effect = np.einsum("cdnf,nk,cdfsk->cdns", eta, x_mat, lam)
            else:
                effect = np.einsum("cdnf,cdfs->cdns", eta, lam)
            if any(zero_mask):
                effect[:, :, zero_mask, :] = 0
            total = total + effect
        return total

    def _conditional_spatial_eta_samples(
        self,
        level: int,
        spec: dict[str, Any],
        X_new: pd.DataFrame,
        known_codes: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if spec.get("type") != "spatial_full":
            raise NotImplementedError("conditional spatial prediction currently supports spatial_full only")
        if self.model is None or self.model.study_design is None:
            raise ValueError("conditional spatial prediction requires model.study_design")
        coord_columns = spec.get("coords", ["x", "y"])
        if any(column not in X_new for column in coord_columns):
            raise ValueError("conditional spatial prediction requires coordinate columns")
        level_name = spec.get("name", "plot")
        group_column = spec.get("column", level_name)
        training_coords = (
            self.model.study_design.groupby(group_column, sort=True)[coord_columns]
            .mean()
            .to_numpy(dtype=float)
        )
        eta_train = self.eta_samples(level)
        if eta_train.shape[2] != training_coords.shape[0]:
            raise ValueError("training coordinates do not match posterior Eta units")

        eta_new = np.empty((*eta_train.shape[:2], len(X_new), eta_train.shape[-1]), dtype=float)
        known_rows = np.flatnonzero(known_codes >= 0)
        if len(known_rows):
            eta_new[:, :, known_rows, :] = eta_train[:, :, known_codes[known_rows], :]

        unknown_rows = np.flatnonzero(known_codes < 0)
        if len(unknown_rows):
            unknown_frame = X_new.iloc[unknown_rows]
            unknown_codes, _unknown_levels = pd.factorize(unknown_frame[group_column], sort=True)
            new_coords = (
                unknown_frame.assign(__prediction_group=unknown_codes)
                .groupby("__prediction_group", sort=True)[coord_columns]
                .mean()
                .to_numpy(dtype=float)
            )
            conditional = self._sample_full_spatial_conditional_eta(
                level,
                spec,
                eta_train,
                training_coords,
                new_coords,
                rng,
            )
            eta_new[:, :, unknown_rows, :] = conditional[:, :, unknown_codes, :]
        return eta_new

    def _sample_full_spatial_conditional_eta(
        self,
        level: int,
        spec: dict[str, Any],
        eta_train: np.ndarray,
        training_coords: np.ndarray,
        new_coords: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        ranges = np.asarray(spec.get("alphapw", []), dtype=float)
        if ranges.size == 0:
            distance = _pairwise_distance_matrix(training_coords, training_coords)
            positive = distance[distance > 0]
            scale = float(spec.get("alpha", np.median(positive) if positive.size else 1.0))
            ranges = np.asarray([[scale, 1.0]], dtype=float)
        ranges = np.atleast_2d(ranges)
        try:
            alpha = self.alpha_samples(level)
        except ValueError:
            alpha = np.zeros((*eta_train.shape[:2], eta_train.shape[-1]), dtype=int)
        if alpha.shape != (*eta_train.shape[:2], eta_train.shape[-1]):
            raise ValueError("posterior Alpha and Eta factor dimensions do not match")

        train_distance = _pairwise_distance_matrix(training_coords, training_coords)
        cross_distance = _pairwise_distance_matrix(new_coords, training_coords)
        new_distance = _pairwise_distance_matrix(new_coords, new_coords)
        output = np.empty((*eta_train.shape[:2], len(new_coords), eta_train.shape[-1]), dtype=float)
        for factor in range(eta_train.shape[-1]):
            target = output[..., factor]
            for alpha_index in np.unique(alpha[..., factor]):
                if alpha_index < 0 or alpha_index >= len(ranges):
                    raise ValueError(f"posterior Alpha index {alpha_index} is outside alphapw")
                scale = float(ranges[alpha_index, 0])
                if scale <= 0:
                    scale = 1.0
                train_cov = np.exp(-train_distance / scale)
                train_cov = train_cov + np.eye(len(training_coords)) * 1e-8
                cross_cov = np.exp(-cross_distance / scale)
                new_cov = np.exp(-new_distance / scale)
                solved = np.linalg.solve(train_cov, cross_cov.T)
                projection = solved.T
                conditional_cov = new_cov - cross_cov @ solved
                conditional_root = _positive_semidefinite_root(conditional_cov)
                means = np.einsum("nt,cdt->cdn", projection, eta_train[..., factor])
                noise = np.einsum(
                    "ij,cdj->cdi",
                    conditional_root,
                    rng.normal(size=means.shape),
                )
                candidate = means + noise
                selected = alpha[..., factor] == alpha_index
                target[selected] = candidate[selected]
        return output

    def _prediction_random_level_spec(self, level: int, spec: dict[str, Any]) -> dict[str, Any]:
        merged = dict(spec)
        metadata = self.metadata if isinstance(self.metadata, dict) else {}
        levels = metadata.get("random_levels", [])
        if isinstance(levels, list) and 0 <= level < len(levels) and isinstance(levels[level], dict):
            merged.update(levels[level])
        return merged

    def _resolve_unseen_group(self, spec: dict[str, Any], X_new: pd.DataFrame, row_idx: int, mode: str) -> int:
        if mode == "error":
            raise ValueError("Prediction contains unknown random-effect group")
        if mode == "zero":
            return 0
        if mode == "sample":
            return row_idx % self.eta_samples(0).shape[2]
        if mode == "nearest":
            if spec.get("type") not in {"spatial_full", "spatial_gpp", "gpp", "spatial_nngp", "nngp"}:
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
        for level_idx, (_level_name, spec) in enumerate(self.model.random_levels.items()):
            eta = self.eta_samples(level_idx).mean(axis=2, keepdims=True)
            eta = np.repeat(eta, n_new, axis=2)
            lam = self.lambda_samples(level_idx)
            if lam.ndim == 5:
                x_mat = self._random_slope_design_for_marginal(spec, n_new)
                total = total + np.einsum("cdnf,nk,cdfsk->cdns", eta, x_mat, lam)
            else:
                total = total + np.einsum("cdnf,cdfs->cdns", eta, lam)
        return total

    def _random_slope_design_for_groups(self, spec: dict[str, Any], X_new: pd.DataFrame) -> np.ndarray:
        x_formula = spec.get("x_formula")
        if not x_formula:
            raise ValueError("Random-slope Lambda samples require random_levels.*.x_formula")
        if self.model is None or self.model.study_design is None:
            raise ValueError("Random-slope prediction requires model.study_design")
        reference = build_design_matrix(x_formula, self.model.study_design)
        matrix = build_design_matrix(x_formula, X_new)
        missing = [column for column in reference.columns if column not in matrix.columns]
        missing_non_intercept = [column for column in missing if column != "Intercept"]
        if missing_non_intercept:
            raise ValueError(f"Prediction data is missing random-slope covariates: {missing_non_intercept}")
        for column in missing:
            matrix[column] = 1.0
        return matrix.loc[:, reference.columns].to_numpy(dtype=float)

    def _random_slope_design_for_marginal(self, spec: dict[str, Any], n_new: int) -> np.ndarray:
        if self.model is None or self.model.study_design is None:
            raise ValueError("Random-slope prediction requires model.study_design")
        column = spec.get("column", "plot")
        matrix = build_design_matrix(spec.get("x_formula"), self.model.study_design)
        averaged = matrix.groupby(self.model.study_design[column], sort=True).mean().mean(axis=0).to_numpy(dtype=float)
        return np.repeat(averaged[None, :], n_new, axis=0)

    def _species_names(self, size: int) -> list[str]:
        if self.model is not None and getattr(self.model, "species_names", None):
            names = self.model.species_names
            if len(names) == size:
                return names
        names = self._metadata_names().get("species")
        if names and len(names) == size:
            return names
        return [f"species_{idx}" for idx in range(size)]

    def _metadata_names(self) -> dict[str, list[str]]:
        if not isinstance(self.metadata, dict):
            return {}
        names = self.metadata.get("names", {})
        if not isinstance(names, dict):
            return {}
        return {
            key: [str(value) for value in values]
            for key, values in names.items()
            if isinstance(values, list)
        }

    def _x_formula(self) -> str | None:
        if self.model is not None:
            return self.model.x_formula
        if isinstance(self.metadata, dict):
            formula = self.metadata.get("formula", {})
            if isinstance(formula, dict):
                return formula.get("X")
            if isinstance(formula, str):
                return formula
        return None

    def _distribution(self) -> str:
        if self.model is not None:
            return self.model.distr
        if isinstance(self.metadata, dict):
            return str(self.metadata.get("distribution", "normal"))
        return "normal"


def _names_or_default(names: list[str] | None, size: int, prefix: str) -> list[str]:
    if names and len(names) == size:
        return names
    return [f"{prefix}_{idx}" for idx in range(size)]


def _factor_names(size: int) -> list[str]:
    return [f"factor_{idx}" for idx in range(size)]


def _read_hdf5_posterior(path: Path) -> dict[str, Any]:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install h5py to read HDF5 posterior files") from exc
    arrays = {}
    metadata = None
    with h5py.File(path, "r") as handle:
        if "pyhmsc_metadata" in handle.attrs:
            metadata = json.loads(handle.attrs["pyhmsc_metadata"])
        _read_hdf5_group(handle, arrays)
    data: dict[str, Any] = {"__arrays__": arrays}
    if metadata is not None:
        data["__metadata__"] = metadata
    return data


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
    metadata = root.attrs.get("pyhmsc_metadata")
    _read_zarr_group(root, arrays)
    data: dict[str, Any] = {"__arrays__": arrays}
    if metadata is not None:
        data["__metadata__"] = metadata
    return data


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


def _is_association_param(param: str) -> bool:
    return param.lower() in {"association", "associations", "speciesassociation", "speciesassociations"}


def _align_eta_lambda_samples(
    eta: np.ndarray,
    lam: np.ndarray,
    max_iter: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Post-hoc align latent factors using Lambda as the loading reference.

    HMSC latent factors are invariant to factor permutation and sign flips. This
    routine orients each posterior draw to an iterative Lambda reference and
    applies the same transformation to Eta so Eta @ Lambda is unchanged.
    """
    reference = np.array(lam[0, 0], dtype=float, copy=True)
    aligned_eta = np.array(eta, dtype=float, copy=True)
    aligned_lam = np.array(lam, dtype=float, copy=True)
    for _ in range(max(1, max_iter)):
        for chain_idx in range(lam.shape[0]):
            for draw_idx in range(lam.shape[1]):
                order, signs = _factor_alignment(lam[chain_idx, draw_idx], reference)
                aligned_lam[chain_idx, draw_idx] = lam[chain_idx, draw_idx][order] * signs[:, None]
                aligned_eta[chain_idx, draw_idx] = eta[chain_idx, draw_idx][:, order] * signs[None, :]
        new_reference = aligned_lam.mean(axis=(0, 1))
        if np.allclose(new_reference, reference):
            break
        reference = new_reference
    return aligned_eta, aligned_lam


def _factor_alignment(loadings: np.ndarray, reference: np.ndarray) -> tuple[list[int], np.ndarray]:
    n_factors = loadings.shape[0]
    scores = _factor_alignment_scores(loadings, reference)
    if n_factors <= 8:
        best_order = max(permutations(range(n_factors)), key=lambda order: sum(scores[i, order[i]] for i in range(n_factors)))
    else:
        remaining = set(range(n_factors))
        order = []
        for ref_idx in range(n_factors):
            chosen = max(remaining, key=lambda factor_idx: scores[ref_idx, factor_idx])
            order.append(chosen)
            remaining.remove(chosen)
        best_order = tuple(order)
    signs = []
    for ref_idx, factor_idx in enumerate(best_order):
        dot = float(np.dot(reference[ref_idx], loadings[factor_idx]))
        signs.append(1.0 if dot >= 0 else -1.0)
    return list(best_order), np.asarray(signs, dtype=float)


def _factor_alignment_scores(loadings: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref_norm = np.linalg.norm(reference, axis=1)
    load_norm = np.linalg.norm(loadings, axis=1)
    denom = np.maximum(ref_norm[:, None] * load_norm[None, :], np.finfo(float).eps)
    return np.abs(reference @ loadings.T) / denom


def _diagnostics_overview(
    diagnostics: pd.DataFrame,
    param: str,
    rhat_threshold: float,
    ess_threshold: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rhat = diagnostics["rhat"].replace([np.inf, -np.inf], np.nan)
    ess = diagnostics["ess"].replace([np.inf, -np.inf], np.nan)
    overview = {
        "param": param,
        "n_parameters": int(len(diagnostics)),
        "rhat_max": float(rhat.max(skipna=True)),
        "rhat_median": float(rhat.median(skipna=True)),
        "ess_min": float(ess.min(skipna=True)),
        "ess_median": float(ess.median(skipna=True)),
        "rhat_threshold": float(rhat_threshold),
        "ess_threshold": float(ess_threshold),
        "n_rhat_flagged": int((rhat > rhat_threshold).sum()),
        "n_ess_flagged": int((ess < ess_threshold).sum()),
    }
    if extra:
        overview.update(extra)
    return overview


def _rhat_array(samples: np.ndarray) -> np.ndarray:
    if samples.ndim < 3:
        raise ValueError("diagnostics require samples with shape chains x draws x ...")
    n_chains, n_draws = samples.shape[:2]
    if n_chains < 2 or n_draws < 2:
        return np.full(samples.shape[2:], np.nan, dtype=float)
    chain_means = samples.mean(axis=1)
    chain_vars = samples.var(axis=1, ddof=1)
    within = chain_vars.mean(axis=0)
    between = n_draws * chain_means.var(axis=0, ddof=1)
    var_hat = ((n_draws - 1) / n_draws) * within + between / n_draws
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / within)
    rhat = np.where((within == 0) & (between == 0), 1.0, rhat)
    return rhat


def _ess_array(samples: np.ndarray) -> np.ndarray:
    if samples.ndim < 3:
        raise ValueError("diagnostics require samples with shape chains x draws x ...")
    n_chains, n_draws = samples.shape[:2]
    if n_chains < 1 or n_draws < 2:
        return np.full(samples.shape[2:], np.nan, dtype=float)
    flat = samples.reshape(n_chains, n_draws, -1)
    ess = np.empty(flat.shape[-1], dtype=float)
    for idx in range(flat.shape[-1]):
        ess[idx] = _ess_one(flat[:, :, idx])
    return ess.reshape(samples.shape[2:])


def _ess_one(values: np.ndarray) -> float:
    n_chains, n_draws = values.shape
    chain_vars = values.var(axis=1, ddof=1)
    variance = float(chain_vars.mean())
    if variance <= 0 or not np.isfinite(variance):
        return float(n_chains * n_draws)
    rho_sum = 0.0
    previous_pair = np.inf
    centered = values - values.mean(axis=1, keepdims=True)
    for lag in range(1, n_draws):
        autocov = np.mean(centered[:, :-lag] * centered[:, lag:])
        rho = float(autocov / variance)
        if lag % 2 == 0:
            pair = previous_pair + rho
            if pair < 0:
                break
            rho_sum += pair
            previous_pair = np.inf
        else:
            previous_pair = rho
    estimate = n_chains * n_draws / max(1.0 + 2.0 * rho_sum, np.finfo(float).eps)
    return float(min(max(estimate, 0.0), n_chains * n_draws))


def _gradient_summary_frame(values: pd.Series, samples: np.ndarray, level: float) -> pd.DataFrame:
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    lo = np.quantile(samples, (1 - level) / 2, axis=(0, 1))
    hi = np.quantile(samples, 1 - (1 - level) / 2, axis=(0, 1))
    mean = samples.mean(axis=(0, 1))
    return pd.DataFrame(
        {
            values.name or "gradient": values.to_numpy(),
            "mean": mean,
            "lower": lo,
            "upper": hi,
        }
    )


def _prediction_frame(
    X_new: Any,
    study_design: Any | None = None,
    coords: Any | None = None,
) -> pd.DataFrame:
    frame = X_new.copy() if isinstance(X_new, pd.DataFrame) else pd.DataFrame(X_new)
    for extra_name, extra in [("study_design", study_design), ("coords", coords)]:
        if extra is None:
            continue
        extra_frame = extra.copy() if isinstance(extra, pd.DataFrame) else pd.DataFrame(extra)
        if len(extra_frame) != len(frame):
            raise ValueError(f"{extra_name} must have the same number of rows as X_new")
        extra_frame.index = frame.index
        for column in extra_frame.columns:
            if column not in frame:
                frame[column] = extra_frame[column]
    return frame


def _resolve_random_effect_mode(random_effects: str, include_random_effects: bool | None) -> str:
    if include_random_effects is None:
        return random_effects
    if include_random_effects:
        return "known" if random_effects == "none" else random_effects
    return "none"


def _pairwise_distance_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def _positive_semidefinite_root(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues))


def _response_scale(values: np.ndarray, distribution: str) -> np.ndarray:
    key = distribution.lower()
    if key == "poisson":
        return np.exp(values)
    if key == "probit":
        return np.clip(_normal_cdf(values), 0.0, 1.0)
    if key in {"bernoulli", "binomial"}:
        return np.clip(1.0 / (1.0 + np.exp(-values)), 0.0, 1.0)
    return values


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import ndtr  # type: ignore

        return ndtr(values)
    except ImportError:
        import math

        return np.vectorize(lambda value: 0.5 * (1.0 + math.erf(value / math.sqrt(2.0))))(values)
