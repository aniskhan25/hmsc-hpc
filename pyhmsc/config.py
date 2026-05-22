"""YAML/JSON model configuration loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pyhmsc.data import read_table
from pyhmsc.model import HmscModel


def load_model_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install PyYAML to read YAML model config files") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Model config must be a mapping")
    return data


def model_from_config(path: str | Path) -> tuple[HmscModel, dict[str, Any]]:
    path = Path(path)
    config = load_model_config(path)
    required = ["response", "covariates", "formula", "distribution"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Model config missing required fields: {missing}")
    formula = config["formula"]
    if isinstance(formula, dict):
        formula = formula.get("X")
    if not formula:
        raise ValueError("Model config must provide formula.X or formula")
    base = path.parent
    model = HmscModel(
        Y=read_table(base / config["response"]),
        X=read_table(base / config["covariates"]),
        x_formula=formula,
        distr=config["distribution"],
        traits=read_table(base / config["traits"]) if config.get("traits") else None,
        trait_formula=config.get("trait_formula"),
        study_design=read_table(base / config["study_design"])
        if config.get("study_design")
        else None,
        random_levels=config.get("random_levels"),
        phylo_cov=read_table(base / config["phylo_cov"]) if config.get("phylo_cov") else None,
    )
    return model, config
