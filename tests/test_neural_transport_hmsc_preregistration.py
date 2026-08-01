import json
from pathlib import Path


ACCEPTANCE = Path(
    "docs/neural_transport_hmsc_milestone_78_acceptance_2026-08-01.md"
)
PREREGISTRATION = Path(
    "docs/neural_transport_hmsc_iid_probit_v0_1_"
    "preregistration_2026-08-01.md"
)
SEED_AUDIT = Path(
    "docs/neural_transport_hmsc_iid_probit_v0_1_"
    "seed_audit_2026-08-01.json.md"
)


def test_milestone_78_acceptance_is_bounded_to_preregistration():
    record = ACCEPTANCE.read_text(encoding="utf-8")
    assert "accept_exact_corrected_transport_direction" in record
    assert "253e7802642192b0d72427b461bf9fc9cc30fa99" in record
    assert "feature/neural-transport-hmsc" in record
    assert "may not" in record
    assert "accepted posterior" in record
    assert "does not authorize model code" in record
    assert "Proceed to the Milestone 79" in record


def test_seed_audit_is_fresh_role_separated_and_fully_sealed():
    audit = json.loads(SEED_AUDIT.read_text(encoding="utf-8"))
    assert audit["protocol"] == "neural_transport_hmsc_iid_probit_v0_1"
    assert audit["decision"] == "reserve_fresh_blocks_without_opening"
    assert audit["simulation_generation_called"] is False
    assert audit["mcmc_generation_called"] is False
    assert audit["scheduler_submission_performed"] is False
    assert audit["artifact_output_created"] is False
    assert audit["all_reserved_blocks_opened"] is False
    assert audit["freshness_check"]["all_absent_before_this_audit"] is True

    occupied = set()
    for record in audit["blocks"].values():
        assert record["opened"] is False
        values = (
            range(record["range"][0], record["range"][1] + 1)
            if "range" in record
            else record["values"]
        )
        values = set(values)
        assert len(values) == record["count"]
        assert occupied.isdisjoint(values)
        occupied.update(values)

    assert audit["retired_and_forbidden"]["generative_iid_v2"] == [
        511000001,
        515000324,
    ]


def test_preregistration_freezes_model_transport_and_controls():
    record = PREREGISTRATION.read_text(encoding="utf-8")
    assert "preregister_exact_corrected_transport_before_implementation" in record
    assert "| sites | 40 |" in record
    assert "| species | 12 |" in record
    assert "exactly 2" in record
    assert "one iid site-level random intercept" in record
    assert "two-stream DeepSets encoder" in record
    assert "deterministic rank-two SVD" in record
    assert "log_target_hmsc" not in record
    assert "exact log absolute determinant" in record
    assert "native_gibbs" in record
    assert "identity_hmc_gibbs" in record
    assert "neural_warmstart_gibbs" in record
    assert "neural_affine_hmc_gibbs" in record


def test_preregistration_puts_exactness_before_efficiency():
    record = PREREGISTRATION.read_text(encoding="utf-8")
    parity = record.index("### Posterior-parity gates")
    efficiency = record.index("### Efficiency gates")
    assert parity < efficiency
    assert "candidate median time to the frozen convergence target" in record
    assert "candidate median ESS/second" in record
    assert "Posterior parity without efficiency improvement" in record
    assert "does not permit threshold tuning" in record


def test_preregistration_keeps_every_seed_and_implementation_sealed():
    record = PREREGISTRATION.read_text(encoding="utf-8")
    assert "does not authorize" in record
    assert "model implementation" in record
    assert "Do not generate simulation or MCMC corpora" in record
    assert "No seed opens automatically after implementation" in record
    assert "No post-hoc calibration" in record
    roadmap = Path("docs/neural_hmsc_implementation_roadmap.md").read_text(
        encoding="utf-8"
    )
    assert "Retired" in roadmap
    assert "511M-515M remains forbidden" in roadmap
