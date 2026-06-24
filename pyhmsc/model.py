"""Public model object for the Python Hmsc-HPC wrapper."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Any

import pandas as pd

from pyhmsc.compiler import CompiledModel, compile_hmsc_model
from pyhmsc.formulas import covariate_names_from_formula, normalize_formula
from pyhmsc.posterior import HmscFit
from pyhmsc.r_bridge import make_init_with_r
from pyhmsc.runner import run_gibbs_sampler
from pyhmsc.validation import validate_compiled_native_model


class HmscModel:
    """Python-facing HMSC model definition.

    Phase 1 supports the fixed-effect user experience and delegates official
    initialization to R's Hmsc package through :mod:`pyhmsc.r_bridge`.
    """

    def __init__(
        self,
        Y: Any,
        X: Any,
        x_formula: str,
        distr: str = "poisson",
        traits: Any | None = None,
        trait_formula: str | None = None,
        study_design: Any | None = None,
        random_levels: dict[str, Any] | None = None,
        phylo_cov: Any | None = None,
        phylo_tree: str | Path | None = None,
    ) -> None:
        self.Y = self._as_frame(Y, "Y")
        self.X = self._as_frame(X, "X")
        if len(self.Y) != len(self.X):
            raise ValueError("Y and X must have the same number of rows")
        self.x_formula = normalize_formula(x_formula)
        self.distr = distr
        self.traits = None if traits is None else self._as_frame(traits, "traits")
        self.trait_formula = normalize_formula(trait_formula) if trait_formula else None
        self.study_design = (
            None if study_design is None else self._as_frame(study_design, "study_design")
        )
        self.random_levels = random_levels
        self.phylo_cov = None if phylo_cov is None else self._as_frame(phylo_cov, "phylo_cov")
        self.phylo_tree = phylo_tree
        self.species_names = list(self.Y.columns)
        self.covariate_names = covariate_names_from_formula(self.x_formula, self.X)

    @staticmethod
    def _as_frame(value: Any, name: str) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            frame = value.copy()
        else:
            frame = pd.DataFrame(value)
        if frame.empty:
            raise ValueError(f"{name} must not be empty")
        return frame

    def compile(
        self,
        output: str | Path,
        chains: int = 4,
        init: str = "python-native",
    ) -> CompiledModel:
        """Compile raw model data to the Python-native JSON+HDF5 artifact."""
        if init != "python-native":
            raise ValueError("compile currently supports init='python-native' only")
        return compile_hmsc_model(
            Y=self.Y,
            X=self.X,
            formula=self.x_formula,
            distr=self.distr,
            chains=chains,
            output=output,
            study_design=self.study_design,
            random_levels=self.random_levels,
            traits=self.traits,
            trait_formula=self.trait_formula,
            phylo_cov=self.phylo_cov,
            phylo_tree=self.phylo_tree,
        )

    def sample(
        self,
        samples: int,
        transient: int,
        thin: int,
        chains: int = 4,
        backend: str = "hmsc-hpc",
        init: str = "r-bridge",
        verbose: int = 100,
        workdir: str | Path | None = None,
        keep_workdir: bool = False,
        python: str | None = None,
        rscript: str = "Rscript",
        **runner_kwargs: Any,
    ) -> HmscFit:
        """Initialize, run Hmsc-HPC, and return a Python fit object."""
        if backend != "hmsc-hpc":
            raise ValueError("Only backend='hmsc-hpc' is currently supported")
        if init not in {"r-bridge", "python-native"}:
            raise ValueError("init must be 'r-bridge' or 'python-native'")
        if samples <= 0 or thin <= 0 or chains <= 0 or transient < 0:
            raise ValueError("samples, thin, and chains must be positive; transient must be >= 0")
        if init == "python-native" and self.random_levels:
            unsupported = [
                name for name, spec in self.random_levels.items()
                if spec.get("x_formula")
                and spec.get("type", "iid") not in {"iid", "spatial_full", "spatial_gpp", "gpp", "spatial_nngp", "nngp"}
            ]
            if unsupported:
                raise NotImplementedError(
                    "Native TensorFlow sampling for random slopes currently supports "
                    f"iid and spatial random levels: {unsupported}"
                )
        if workdir is not None:
            run_dir = Path(workdir)
            run_dir.mkdir(parents=True, exist_ok=True)
            return self._sample_in_dir(
                run_dir, samples, transient, thin, chains, verbose, python, rscript, init, runner_kwargs
            )

        if keep_workdir:
            run_dir = Path(mkdtemp(prefix="pyhmsc-"))
            return self._sample_in_dir(
                run_dir, samples, transient, thin, chains, verbose, python, rscript, init, runner_kwargs
            )

        with TemporaryDirectory(prefix="pyhmsc-") as tmp:
            run_dir = Path(tmp)
            fit = self._sample_in_dir(
                run_dir, samples, transient, thin, chains, verbose, python, rscript, init, runner_kwargs
            )
            return fit

    def _sample_in_dir(
        self,
        run_dir: Path,
        samples: int,
        transient: int,
        thin: int,
        chains: int,
        verbose: int,
        python: str | None,
        rscript: str,
        init: str,
        runner_kwargs: dict[str, Any],
    ) -> HmscFit:
        if init == "r-bridge":
            init_file = run_dir / "init_file.rds"
            post_file = run_dir / "post_file.rds"
            make_init_with_r(
                self,
                init_file=init_file,
                workdir=run_dir,
                samples=samples,
                transient=transient,
                thin=thin,
                chains=chains,
                verbose=verbose,
                rscript=rscript,
            )
        else:
            compiled = self.compile(run_dir, chains=chains)
            _raise_if_not_sampler_ready(compiled.init_json)
            init_file = compiled.init_json
            post_file = Path(runner_kwargs.pop("output_file", run_dir / "posterior.h5"))
        run_gibbs_sampler(
            init_file=init_file,
            output_file=post_file,
            samples=samples,
            transient=transient,
            thin=thin,
            verbose=verbose,
            python=python,
            **runner_kwargs,
        )
        fit = HmscFit.from_file(post_file, model=self)
        fit.init_file = init_file
        fit.output_file = post_file
        fit.workdir = run_dir
        return fit


def _raise_if_not_sampler_ready(init_json: Path) -> None:
    failed = [
        result
        for result in validate_compiled_native_model(init_json)
        if not result.passed
    ]
    if failed:
        details = "; ".join(f"{result.name}: {result.details}" for result in failed)
        raise NotImplementedError(f"Compiled native model is not sampler-ready: {details}")
