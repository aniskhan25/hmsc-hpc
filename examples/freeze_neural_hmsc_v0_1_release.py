#!/usr/bin/env python3
"""Freeze and validate the complete qualified Neural-HMSC v0.1 release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhmsc.neural.release import (  # noqa: E402
    NEURAL_HMSC_RELEASE_ID,
    freeze_neural_hmsc_release,
    validate_neural_hmsc_release,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--packaged-members-root", type=Path, required=True)
    parser.add_argument("--predictive-baseline-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--release-id", default=NEURAL_HMSC_RELEASE_ID)
    args = parser.parse_args()

    path = freeze_neural_hmsc_release(
        registry_root=args.registry_root,
        packaged_members_root=args.packaged_members_root,
        predictive_baseline_root=args.predictive_baseline_root,
        audit_root=args.audit_root,
        release_id=args.release_id,
    )
    payload = validate_neural_hmsc_release(path, expected_release_id=args.release_id)
    print(
        json.dumps(
            {
                "release": str(path),
                "release_id": payload["release_id"],
                "release_status": payload["release_status"],
                "content_sha256": payload["content_sha256"],
                "inventory_files": len(payload["inventory"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
