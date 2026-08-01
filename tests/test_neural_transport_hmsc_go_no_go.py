from pathlib import Path


DECISION = Path("docs/neural_transport_hmsc_go_no_go_2026-08-01.md")
ROADMAP = Path("docs/neural_hmsc_implementation_roadmap.md")


def test_transport_decision_rejects_a_third_direct_posterior():
    decision = DECISION.read_text(encoding="utf-8")
    assert (
        "go_to_preregistration_for_exact_corrected_neural_transport" in decision
    )
    assert "Do not build a third standalone amortized joint posterior" in decision
    assert "may not" in decision
    assert "define the accepted posterior" in decision
    assert "Third raw-state/orbit amortized posterior" in decision
    assert "Rejected" in decision


def test_transport_decision_preserves_the_hmsc_target_and_fallback():
    decision = DECISION.read_text(encoding="utf-8")
    assert "log_target_hmsc(T(z)) + log_abs_det_jacobian_T(z)" in decision
    assert "TFP HMC retains its accept/reject correction" in decision
    assert "ordinary-Gibbs fallback" in decision
    assert "output remains MCMC" in decision
    assert "Failure to improve efficiency closes the transport candidate" in decision


def test_transport_first_scope_and_controls_are_bounded():
    decision = DECISION.read_text(encoding="utf-8")
    assert "40 sites and 12 species" in decision
    assert "exactly two latent factors" in decision
    assert "one iid site-level random intercept" in decision
    assert "warm-start-only network and identity transport are mandatory" in decision
    assert "at least 25% lower median time" in decision
    assert "at least 1.25x median effective samples per second" in decision


def test_roadmap_requires_preregistration_before_implementation():
    roadmap = ROADMAP.read_text(encoding="utf-8")
    assert "79. Preregister Neural-Transport HMSC on a new branch" in roadmap
    assert "Do not implement the model or generate simulations" in roadmap
    assert "80. Implement the ordinary-fixture transport kernel" in roadmap
    assert "Status: blocked by Milestone 79" in roadmap
    assert (
        "81. Run fresh disposable exactness and efficiency qualification" in roadmap
    )
    assert "82. Run fixed validation and bounded real-data confirmation" in roadmap
