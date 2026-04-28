"""Offline replay-first decision contract.

This module is research-only. It does not place orders, query operational
state, mutate inventory, or modify probability-engine outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Action = Literal["buy_yes", "buy_no", "abstain"]
Side = Literal["YES", "NO"]

MIN_HMM_POSTERIOR_CONFIDENCE = 0.70
MIN_HMM_NEXT_SAME_STATE_CONFIDENCE = 0.65
MIN_HMM_PERSISTENCE_COUNT = 2


@dataclass(frozen=True)
class ProbabilitySnapshot:
    p_yes: float
    p_no: float
    engine_name: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuoteSnapshot:
    q_yes: float | None = None
    q_no: float | None = None
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None
    spread_yes: float | None = None
    spread_no: float | None = None
    quote_age_sec: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TauPolicySnapshot:
    tau_minutes: float | None
    tau_bucket: str
    edge_threshold_yes: float
    edge_threshold_no: float
    allow_new_entries: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HMMPolicyState:
    map_state: int | None = None
    posterior_confidence: float | None = None
    next_same_state_confidence: float | None = None
    persistence_count: int | None = None
    policy_state: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SafetyVetoSnapshot:
    tail_veto: bool | None = None
    polarization_veto: bool | None = None
    reversal_veto: bool | None = None
    quote_quality_pass: bool | None = None
    tail_fields: dict[str, Any] = field(default_factory=dict)
    polarization_fields: dict[str, Any] = field(default_factory=dict)
    reversal_fields: dict[str, Any] = field(default_factory=dict)
    quote_quality_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExpectedGrowthSnapshot:
    expected_log_growth: float | None = None
    conservative_expected_log_growth: float | None = None
    passes: bool | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionInput:
    timestamp: str
    market_id: str
    probability: ProbabilitySnapshot
    quote: QuoteSnapshot
    tau_policy: TauPolicySnapshot
    hmm_policy_state: HMMPolicyState | None = None
    safety_veto: SafetyVetoSnapshot | None = None
    expected_growth: ExpectedGrowthSnapshot | None = None
    side_context: Side | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionOutput:
    action: Action
    allowed: bool
    chosen_side: Side | None
    reason: str
    blocking_reasons: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _finite_probability(value: float) -> bool:
    return 0.0 <= float(value) <= 1.0


def _edge_candidates(input_data: DecisionInput) -> tuple[list[dict[str, Any]], dict[str, float | None], list[str]]:
    probability = input_data.probability
    quote = input_data.quote
    tau_policy = input_data.tau_policy
    diagnostics: dict[str, float | None] = {
        "edge_yes": None,
        "edge_no": None,
        "q_yes": quote.q_yes,
        "q_no": quote.q_no,
        "p_yes": probability.p_yes,
        "p_no": probability.p_no,
    }
    blockers: list[str] = []
    candidates: list[dict[str, Any]] = []

    if not (_finite_probability(probability.p_yes) and _finite_probability(probability.p_no)):
        return candidates, diagnostics, ["invalid_probability"]

    if quote.q_yes is None or quote.q_no is None:
        return candidates, diagnostics, ["missing_quote"]

    q_yes = float(quote.q_yes)
    q_no = float(quote.q_no)
    if q_yes <= 0.0 or q_no <= 0.0:
        return candidates, diagnostics, ["missing_quote"]

    edge_yes = float(probability.p_yes) - q_yes
    edge_no = float(probability.p_no) - q_no
    diagnostics["edge_yes"] = edge_yes
    diagnostics["edge_no"] = edge_no

    if edge_yes >= float(tau_policy.edge_threshold_yes):
        candidates.append({"side": "YES", "action": "buy_yes", "edge": edge_yes, "threshold": tau_policy.edge_threshold_yes})
    if edge_no >= float(tau_policy.edge_threshold_no):
        candidates.append({"side": "NO", "action": "buy_no", "edge": edge_no, "threshold": tau_policy.edge_threshold_no})
    if not candidates:
        blockers.append("no_edge_above_threshold")

    candidates.sort(key=lambda item: (float(item["edge"]), item["side"] == "YES"), reverse=True)
    return candidates, diagnostics, blockers


def _policy_blockers(input_data: DecisionInput) -> list[str]:
    blockers: list[str] = []
    tau_policy = input_data.tau_policy
    safety = input_data.safety_veto
    expected = input_data.expected_growth
    hmm = input_data.hmm_policy_state

    if not tau_policy.allow_new_entries:
        blockers.append("new_entries_disabled")

    if expected is not None:
        conservative = expected.conservative_expected_log_growth
        if conservative is not None and float(conservative) <= 0.0:
            blockers.append("expected_growth_veto")
        if expected.passes is False:
            blockers.append("expected_growth_veto")

    if safety is not None:
        if safety.tail_veto is True:
            blockers.append("tail_veto")
        if safety.polarization_veto is True:
            blockers.append("polarization_veto")
        if safety.reversal_veto is True:
            blockers.append("reversal_veto")
        if safety.quote_quality_pass is False:
            blockers.append("quote_quality_veto")

    if hmm is not None:
        if hmm.policy_state == "transition_uncertain":
            blockers.append("hmm_transition_uncertain")
        if hmm.posterior_confidence is not None and float(hmm.posterior_confidence) < MIN_HMM_POSTERIOR_CONFIDENCE:
            blockers.append("hmm_low_posterior_confidence")
        if hmm.next_same_state_confidence is not None and float(hmm.next_same_state_confidence) < MIN_HMM_NEXT_SAME_STATE_CONFIDENCE:
            blockers.append("hmm_low_next_same_state_confidence")
        if hmm.persistence_count is not None and int(hmm.persistence_count) < MIN_HMM_PERSISTENCE_COUNT:
            blockers.append("hmm_insufficient_persistence")

    return blockers


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def evaluate_replay_decision(input_data: DecisionInput) -> DecisionOutput:
    """Evaluate an offline replay decision from explicit policy inputs."""

    candidates, edge_diagnostics, edge_blockers = _edge_candidates(input_data)
    blocking_reasons = _dedupe(edge_blockers + _policy_blockers(input_data))
    chosen = candidates[0] if candidates else None
    allowed = chosen is not None and not blocking_reasons
    action: Action = "abstain" if not allowed else chosen["action"]
    chosen_side: Side | None = None if not allowed else chosen["side"]
    reason = "ok" if allowed else (blocking_reasons[0] if blocking_reasons else "not_allowed")

    diagnostics: dict[str, Any] = {
        **edge_diagnostics,
        "candidate_edges": candidates,
        "tau_bucket": input_data.tau_policy.tau_bucket,
        "engine_name": input_data.probability.engine_name,
        "hmm_policy_state": None if input_data.hmm_policy_state is None else input_data.hmm_policy_state.policy_state,
    }
    diagnostics.update(input_data.diagnostics)

    return DecisionOutput(
        action=action,
        allowed=allowed,
        chosen_side=chosen_side,
        reason=reason,
        blocking_reasons=blocking_reasons,
        diagnostics=diagnostics,
    )
