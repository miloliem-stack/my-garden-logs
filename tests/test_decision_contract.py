import inspect

from src.research import decision_contract as contract
from src.research.decision_contract import (
    DecisionInput,
    ExpectedGrowthSnapshot,
    HMMPolicyState,
    ProbabilitySnapshot,
    QuoteSnapshot,
    SafetyVetoSnapshot,
    TauPolicySnapshot,
    evaluate_replay_decision,
)


def _input(
    *,
    p_yes=0.60,
    p_no=0.40,
    q_yes=0.50,
    q_no=0.50,
    edge_threshold_yes=0.03,
    edge_threshold_no=0.03,
    allow_new_entries=True,
    hmm=None,
    safety=None,
    expected=None,
):
    return DecisionInput(
        timestamp="2026-04-28T12:00:00Z",
        market_id="M1",
        probability=ProbabilitySnapshot(p_yes=p_yes, p_no=p_no, engine_name="gaussian_vol"),
        quote=QuoteSnapshot(q_yes=q_yes, q_no=q_no, yes_bid=q_yes - 0.01, yes_ask=q_yes + 0.01, no_bid=q_no - 0.01, no_ask=q_no + 0.01),
        tau_policy=TauPolicySnapshot(
            tau_minutes=20,
            tau_bucket="mid",
            edge_threshold_yes=edge_threshold_yes,
            edge_threshold_no=edge_threshold_no,
            allow_new_entries=allow_new_entries,
        ),
        hmm_policy_state=hmm or HMMPolicyState(
            map_state=1,
            posterior_confidence=0.85,
            next_same_state_confidence=0.80,
            persistence_count=3,
            policy_state="state_1_confident",
        ),
        safety_veto=safety or SafetyVetoSnapshot(
            tail_veto=False,
            polarization_veto=False,
            reversal_veto=False,
            quote_quality_pass=True,
        ),
        expected_growth=expected or ExpectedGrowthSnapshot(
            expected_log_growth=0.02,
            conservative_expected_log_growth=0.01,
            passes=True,
        ),
    )


def test_positive_yes_edge_passes_when_all_gates_pass():
    out = evaluate_replay_decision(_input(p_yes=0.60, p_no=0.40, q_yes=0.50, q_no=0.50))

    assert out.allowed is True
    assert out.action == "buy_yes"
    assert out.chosen_side == "YES"
    assert out.reason == "ok"


def test_positive_no_edge_passes_when_all_gates_pass():
    out = evaluate_replay_decision(_input(p_yes=0.35, p_no=0.65, q_yes=0.50, q_no=0.50))

    assert out.allowed is True
    assert out.action == "buy_no"
    assert out.chosen_side == "NO"


def test_no_edge_blocks():
    out = evaluate_replay_decision(_input(p_yes=0.51, p_no=0.49, q_yes=0.50, q_no=0.50))

    assert out.allowed is False
    assert out.action == "abstain"
    assert out.reason == "no_edge_above_threshold"
    assert "no_edge_above_threshold" in out.blocking_reasons


def test_expected_growth_nonpositive_blocks():
    out = evaluate_replay_decision(
        _input(expected=ExpectedGrowthSnapshot(expected_log_growth=0.02, conservative_expected_log_growth=0.0, passes=False))
    )

    assert out.allowed is False
    assert out.reason == "expected_growth_veto"
    assert out.blocking_reasons == ["expected_growth_veto"]


def test_tail_and_polarization_vetoes_block():
    out = evaluate_replay_decision(
        _input(safety=SafetyVetoSnapshot(tail_veto=True, polarization_veto=True, reversal_veto=False, quote_quality_pass=True))
    )

    assert out.allowed is False
    assert "tail_veto" in out.blocking_reasons
    assert "polarization_veto" in out.blocking_reasons


def test_reversal_veto_blocks():
    out = evaluate_replay_decision(
        _input(safety=SafetyVetoSnapshot(tail_veto=False, polarization_veto=False, reversal_veto=True, quote_quality_pass=True))
    )

    assert out.allowed is False
    assert out.reason == "reversal_veto"


def test_transition_uncertain_hmm_state_blocks():
    out = evaluate_replay_decision(
        _input(hmm=HMMPolicyState(map_state=1, posterior_confidence=0.90, next_same_state_confidence=0.80, persistence_count=4, policy_state="transition_uncertain"))
    )

    assert out.allowed is False
    assert out.reason == "hmm_transition_uncertain"


def test_low_posterior_confidence_blocks():
    out = evaluate_replay_decision(
        _input(hmm=HMMPolicyState(map_state=1, posterior_confidence=0.69, next_same_state_confidence=0.80, persistence_count=4, policy_state="state_1_confident"))
    )

    assert out.allowed is False
    assert out.reason == "hmm_low_posterior_confidence"


def test_low_next_same_state_confidence_blocks():
    out = evaluate_replay_decision(
        _input(hmm=HMMPolicyState(map_state=1, posterior_confidence=0.80, next_same_state_confidence=0.64, persistence_count=4, policy_state="state_1_confident"))
    )

    assert out.allowed is False
    assert out.reason == "hmm_low_next_same_state_confidence"


def test_insufficient_persistence_blocks():
    out = evaluate_replay_decision(
        _input(hmm=HMMPolicyState(map_state=1, posterior_confidence=0.80, next_same_state_confidence=0.80, persistence_count=1, policy_state="state_1_confident"))
    )

    assert out.allowed is False
    assert out.reason == "hmm_insufficient_persistence"


def test_p_yes_is_not_modified():
    decision_input = _input(p_yes=0.62, p_no=0.38, q_yes=0.50, q_no=0.50)
    before = decision_input.probability.p_yes

    out = evaluate_replay_decision(decision_input)

    assert decision_input.probability.p_yes == before
    assert out.diagnostics["p_yes"] == before


def test_output_includes_all_blocking_reasons_when_several_gates_fail():
    out = evaluate_replay_decision(
        _input(
            p_yes=0.51,
            p_no=0.49,
            q_yes=0.50,
            q_no=0.50,
            allow_new_entries=False,
            safety=SafetyVetoSnapshot(tail_veto=True, polarization_veto=True, reversal_veto=True, quote_quality_pass=False),
            expected=ExpectedGrowthSnapshot(expected_log_growth=-0.01, conservative_expected_log_growth=-0.02, passes=False),
            hmm=HMMPolicyState(map_state=2, posterior_confidence=0.60, next_same_state_confidence=0.50, persistence_count=0, policy_state="transition_uncertain"),
        )
    )

    assert out.action == "abstain"
    assert out.reason == "no_edge_above_threshold"
    assert out.blocking_reasons == [
        "no_edge_above_threshold",
        "new_entries_disabled",
        "expected_growth_veto",
        "tail_veto",
        "polarization_veto",
        "reversal_veto",
        "quote_quality_veto",
        "hmm_transition_uncertain",
        "hmm_low_posterior_confidence",
        "hmm_low_next_same_state_confidence",
        "hmm_insufficient_persistence",
    ]


def test_contract_does_not_import_operational_modules():
    source = inspect.getsource(contract)

    forbidden = [
        "storage",
        "execution",
        "wallet_state",
        "redeemer",
        "polymarket_client",
        "place_marketable",
    ]
    for token in forbidden:
        assert token not in source
