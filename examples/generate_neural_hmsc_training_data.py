"""Generate fixed-effect Neural-HMSC benchmark corpora."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.neural.datasets import load_benchmark_config
from pyhmsc.neural.simulator import generate_fixed_effect_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="benchmark.yaml path")
    parser.add_argument("--output", required=True, help="output corpus directory")
    parser.add_argument(
        "--profile",
        default="smoke",
        help="corpus_sizes profile from the benchmark config, e.g. smoke or default",
    )
    parser.add_argument("--chains", type=int, help="override compiled model chain count")
    args = parser.parse_args()

    config = load_benchmark_config(args.config)
    manifest = generate_fixed_effect_corpus(
        config,
        Path(args.output),
        profile=args.profile,
        chains=args.chains,
    )
    total = sum(split["count"] for split in manifest["splits"].values())
    print(f"Generated {total} datasets in {args.output}")


if __name__ == "__main__":
    main()
