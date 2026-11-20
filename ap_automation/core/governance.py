"""
Governance gate - the single place where specialist-agent output is turned into a
routing decision.

Used by BOTH the imperative pipeline (core/pipeline.py) and the LangGraph nodes
(graph/nodes.py), so the two can no longer drift apart. Previously the VAT flag
strings compared here ("reverse_charge", "blocked_entertainment") never matched
what the VAT agent emitted ("REVERSE_CHARGE", "BLOCKED"), so VAT governance never
fired, and `review_required` was computed but never read.

Principles:
  * The gate can only make a decision STRICTER than the rule gate, never looser.
  * Fail closed: an agent that errors is itself a reason for human review.
"""
from __future__ import annotations

from typing import Any

from ap_automation.core.models import RoutingDecision

VAT_FLAG_TREATMENTS = ("BLOCKED", "REVERSE_CHARGE", "OUTSIDE_SCOPE")


def extract_signals(orchestration_result: Any) -> dict[str, Any]:
    """Pull the governance-relevant signals out of an AgentOrchestrationResult."""
    r = orchestration_result
    cost_centre_mismatch = False
    entity_valid: bool | None = None
    vat_treatment: str | None = None
    vat_review_required = False
    inferred_cc: str | None = None
    agent_errors: list[str] = []

    for res in getattr(r, "results", []):
        if not res.success:
            agent_errors.append(res.agent_name)

    if r.cost_centre and r.cost_centre.success:
        cost_centre_mismatch = bool(r.cost_centre.data.get("cost_centre_mismatch", False))
        inferred_cc = r.cost_centre.data.get("inferred_cost_centre") or None
    if r.entity and r.entity.success:
        entity_valid = r.entity.data.get("entity_valid")
    if r.vat and r.vat.success:
        vat_treatment = r.vat.data.get("vat_treatment")
        vat_review_required = bool(r.vat.data.get("review_required", False))

    return {
        "cost_centre_mismatch": cost_centre_mismatch,
        "entity_valid": entity_valid,
        "vat_treatment": vat_treatment,
        "vat_review_required": vat_review_required,
        "inferred_cost_centre": inferred_cc,
        "agent_errors": agent_errors,
    }


def agent_flags(
    *,
    cost_centre_mismatch: bool = False,
    entity_valid: bool | None = None,
    vat_treatment: str | None = None,
    vat_review_required: bool = False,
    agent_errors: list[str] | None = None,
    **_ignored: Any,
) -> list[str]:
    flags: list[str] = []
    if cost_centre_mismatch:
        flags.append("cost_centre_mismatch")
    if entity_valid is False:
        flags.append("entity_validation_failed")
    treatment = (vat_treatment or "").upper()
    if treatment in VAT_FLAG_TREATMENTS:
        flags.append(f"vat_{treatment.lower()}")
    elif treatment == "NOT_ASSESSED":
        flags.append("vat_not_assessed")
    elif vat_review_required:
        flags.append("vat_review_required")
    for name in agent_errors or []:
        flags.append(f"agent_error:{name}")
    return flags


def apply_gate(
    rule_decision: RoutingDecision | str,
    rule_reason: str,
    flags: list[str],
) -> tuple[RoutingDecision, str]:
    """Combine the rule-gate decision with agent flags. Never loosens a decision."""
    rule = RoutingDecision(rule_decision)
    if rule == RoutingDecision.HUMAN_REVIEW:
        if flags:
            return rule, f"{rule_reason}; agent flags: {', '.join(flags)}"
        return rule, rule_reason
    if flags:
        return RoutingDecision.HUMAN_REVIEW, f"Agent flags raised: {', '.join(flags)}"
    return RoutingDecision.AUTO_POST, rule_reason


def never_loosen(base: RoutingDecision | str, proposed: str | None) -> RoutingDecision:
    """Accept a proposed routing (e.g. from the LLM gate) only if it is at least as strict."""
    base_dec = RoutingDecision(base)
    if base_dec == RoutingDecision.HUMAN_REVIEW:
        return RoutingDecision.HUMAN_REVIEW
    try:
        return RoutingDecision(proposed) if proposed else base_dec
    except ValueError:
        return base_dec
