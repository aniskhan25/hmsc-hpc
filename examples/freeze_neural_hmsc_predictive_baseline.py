"""Freeze a qualified neural predictive ensemble as a versioned baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.neural.deployment import (
    PROMOTED_PREDICTIVE_BASELINE_ID,
    freeze_predictive_deployment_baseline,
    validate_predictive_deployment_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--requalification-root", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-id", default=PROMOTED_PREDICTIVE_BASELINE_ID
    )
    args = parser.parse_args()

    path = freeze_predictive_deployment_baseline(
        registry_root=args.registry_root,
        requalification_root=args.requalification_root,
        smoke_root=args.smoke_root,
        baseline_id=args.baseline_id,
    )
    result = validate_predictive_deployment_baseline(
        path,
        expected_baseline_id=args.baseline_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
