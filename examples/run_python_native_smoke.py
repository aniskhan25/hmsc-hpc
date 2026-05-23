"""Run Python-native pyhmsc example projects end to end.

The runner intentionally uses the public CLI so it exercises the same no-R
workflow documented for users:

    compile -> validate-init -> sample -> summarize
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {
    "fixed_poisson": ROOT / "examples" / "projects" / "fixed_poisson" / "model.yaml",
    "traits_phylogeny": ROOT / "examples" / "projects" / "traits_phylogeny" / "model.yaml",
    "iid_random_intercept": ROOT / "examples" / "projects" / "iid_random_intercept" / "model.yaml",
    "spatial_full": ROOT / "examples" / "projects" / "spatial_full" / "model.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run supported Python-native pyhmsc example projects."
    )
    parser.add_argument(
        "--project",
        choices=sorted(PROJECTS),
        action="append",
        help="project to run; repeat to run several; defaults to all supported projects",
    )
    parser.add_argument("--output-root", default="run_examples", help="directory for run outputs")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--transient", type=int, default=0)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--skip-sample",
        action="store_true",
        help="only compile and validate init artifacts; useful for fast checks",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove the output directory before running",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    selected = args.project or list(PROJECTS)
    for name in selected:
        run_project(
            name=name,
            config=PROJECTS[name],
            output_root=output_root,
            samples=args.samples,
            transient=args.transient,
            thin=args.thin,
            verbose=args.verbose,
            skip_sample=args.skip_sample,
        )


def run_project(
    name: str,
    config: Path,
    output_root: Path,
    samples: int,
    transient: int,
    thin: int,
    verbose: int,
    skip_sample: bool,
) -> None:
    run_dir = output_root / name
    posterior = run_dir / "posterior.h5"
    print(f"\n== {name} ==", flush=True)
    _run(["compile", str(config), "--output", str(run_dir)])
    _run(["validate-init", str(run_dir / "init.json"), "--strict"])
    if skip_sample:
        print("sample: skipped", flush=True)
        return
    _run(
        [
            "sample",
            str(run_dir / "init.json"),
            "--output",
            str(posterior),
            "--samples",
            str(samples),
            "--transient",
            str(transient),
            "--thin",
            str(thin),
            "--verbose",
            str(verbose),
        ]
    )
    _run(["summarize", str(posterior), "--param", "Beta"])


def _run(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "pyhmsc", *args]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
