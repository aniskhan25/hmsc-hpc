"""Run the Neural-HMSC benchmark with conditional coefficient calibration.

This entry point accepts the same arguments as ``run_neural_hmsc_benchmark.py``
and selects the structured conditional coefficient-scale head by default.
Predictive-only calibration remains on the existing scalar path.
"""

from __future__ import annotations

import sys

from run_neural_hmsc_benchmark import main as benchmark_main


def main() -> None:
    if "--coefficient-calibration" not in sys.argv:
        sys.argv.extend(["--coefficient-calibration", "conditional"])
    benchmark_main()


if __name__ == "__main__":
    main()
