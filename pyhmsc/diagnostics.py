"""Optional diagnostics helpers."""

from __future__ import annotations

from typing import Any


def to_arviz(fit: Any) -> Any:
    """Convert a fit to ArviZ InferenceData when arviz is installed."""
    try:
        import arviz as az  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install arviz to use diagnostics helpers") from exc
    return az.from_dict(posterior={"Beta": fit.beta_samples()})
