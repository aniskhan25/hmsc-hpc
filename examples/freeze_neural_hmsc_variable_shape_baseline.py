#!/usr/bin/env python3
"""Freeze the promoted variable-shape probit checkpoint and evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.neural.variable_inference import (  # noqa: E402
    VARIABLE_SHAPE_BASELINE_ID,
    freeze_variable_shape_baseline,
    validate_variable_shape_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--baseline-id", default=VARIABLE_SHAPE_BASELINE_ID)
    args = parser.parse_args()
    path = freeze_variable_shape_baseline(
        registry_root=args.registry_root,
        candidate_checkpoint=args.candidate_checkpoint,
        qualification_root=args.qualification_root,
        baseline_id=args.baseline_id,
    )
    payload = validate_variable_shape_baseline(
        path, expected_baseline_id=args.baseline_id
    )
    print(
        json.dumps(
            {
                "baseline": str(path),
                "baseline_id": payload["baseline_id"],
                "content_sha256": payload["content_sha256"],
                "inventory_files": len(payload["inventory"]),
                "status": payload["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
