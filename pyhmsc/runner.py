"""Subprocess runner for the existing Hmsc-HPC sampler entrypoint."""

from __future__ import annotations

import subprocess
import sys
import os
import tempfile
from pathlib import Path
from typing import Iterable


def run_gibbs_sampler(
    init_file: str | Path,
    output_file: str | Path,
    samples: int,
    transient: int,
    thin: int,
    verbose: int = 100,
    python: str | None = None,
    chains: Iterable[int] | None = None,
    rng_seed: int | None = None,
    fp: int | None = None,
    extra_args: Iterable[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    exe = python or sys.executable
    cmd = [
        exe,
        "-m",
        "hmsc.run_gibbs_sampler",
        "--input",
        str(init_file),
        "--output",
        str(output_file),
        "--samples",
        str(samples),
        "--transient",
        str(transient),
        "--thin",
        str(thin),
        "--verbose",
        str(verbose),
    ]
    if chains is not None:
        cmd.append("--chains")
        cmd.extend(str(chain) for chain in chains)
    if rng_seed is not None:
        cmd.extend(["--rngseed", str(rng_seed)])
    if fp is not None:
        cmd.extend(["--fp", str(fp)])
    if extra_args:
        cmd.extend(extra_args)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pyhmsc-mpl"))
    return subprocess.run(cmd, check=True, text=True, env=env)
