from pathlib import Path

import pandas as pd

from examples.replay_neural_hmsc_target_context_gate import summarize_replay


def test_target_context_replay_promotion_requires_stability_and_two_gains():
    frame = pd.DataFrame(
        [
            {
                "seed": 1,
                "target_gate_passed": True,
                "no_degradation_passed": True,
                "genuine_big_spatial_improvement": True,
            },
            {
                "seed": 2,
                "target_gate_passed": True,
                "no_degradation_passed": True,
                "genuine_big_spatial_improvement": True,
            },
            {
                "seed": 3,
                "target_gate_passed": False,
                "no_degradation_passed": True,
                "genuine_big_spatial_improvement": False,
            },
        ]
    )

    summary = summarize_replay(frame)

    assert summary["decision"] == "target_context_gate_promotion_candidate"
    assert summary["no_degradation_pass_count"] == 3
    assert summary["genuine_big_spatial_improvement_count"] == 2
    assert not summary["target_response_used_for_selection"]


def test_target_context_replay_opens_heldout_metrics_after_selection():
    text = Path("examples/replay_neural_hmsc_target_context_gate.py").read_text(
        encoding="utf-8"
    )

    selection = text.index("active, decision = select_predictive_mean_calibration")
    heldout = text.index("heldout = pd.read_csv")

    assert selection < heldout
