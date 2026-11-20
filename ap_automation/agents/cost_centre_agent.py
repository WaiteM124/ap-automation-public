"""
Cost Centre Specialist Agent.

Validates the SUBMITTED cost centre against the one the chart of accounts expects
for the predicted GL, and recommends a cost centre when none was submitted.
Deterministic rules - no LLM.

Previously this agent compared the evidence-derived cost centre to the GL default,
so the cost centre on the invoice itself was never checked.

Consumes vat_agent context:
  - BLOCKED: entertainment never maps to a departmental cost centre (CC100).
  - REVERSE_CHARGE: restricted to Facilities (CC300) or Professional Services (CC400).
"""
from __future__ import annotations

from ap_automation.agents.base import A2AAgent, AgentResult
from ap_automation.core.chart_of_accounts import get_account

_BLOCKED_CC = "CC100"
_RC_ALLOWED = ("CC300", "CC400")


class CostCentreAgent(A2AAgent):
    @property
    def name(self) -> str:
        return "cost_centre_agent"

    @property
    def description(self) -> str:
        return (
            "Checks the submitted cost centre against the chart of accounts for the "
            "predicted GL, narrowed by VAT treatment, and recommends one when missing."
        )

    def input_fields(self) -> list[str]:
        return ["predicted_gl", "submitted_cost_centre", "predicted_cost_centre", "vat_treatment"]

    def output_fields(self) -> list[str]:
        return ["inferred_cost_centre", "cost_centre_mismatch", "cost_centre_note"]

    def can_handle(self, context: dict) -> bool:
        return get_account(context.get("predicted_gl", "") or "") is not None

    def run(self, context: dict) -> AgentResult:
        try:
            gl = context.get("predicted_gl", "") or ""
            submitted = (context.get("submitted_cost_centre") or "").strip().upper()
            evidence_cc = context.get("predicted_cost_centre", "") or ""
            vat_treatment = (context.get("vat_treatment") or "").upper()

            account = get_account(gl)
            expected = account.default_cost_centre if account else ""
            note = ""

            if vat_treatment == "BLOCKED":
                expected = _BLOCKED_CC
                note = f"VAT BLOCKED - cost centre forced to {_BLOCKED_CC} (no input VAT recovery)."
            elif vat_treatment == "REVERSE_CHARGE" and expected not in _RC_ALLOWED:
                expected = _RC_ALLOWED[0]
                note = f"VAT REVERSE_CHARGE - cost centre restricted to {'/'.join(_RC_ALLOWED)}; using {expected}."

            # A chart account with no default cost centre (a client chart that does not define one) sets no
            # expectation, so there is nothing to mismatch against.
            mismatch = bool(submitted) and bool(expected) and submitted != expected.upper()

            if not submitted:
                reasoning = f"No cost centre submitted - recommending {expected} for GL {gl}."
            elif mismatch:
                reasoning = (
                    f"Cost centre mismatch: submitted {submitted}, GL {gl} expects {expected}. "
                    "Review required."
                )
            else:
                reasoning = f"Cost centre {submitted} is consistent with GL {gl}."
            if note:
                reasoning += f" {note}"

            return AgentResult(
                agent_name=self.name,
                success=True,
                confidence=0.85,
                reasoning=reasoning,
                data={
                    "inferred_cost_centre": expected,
                    "provided_cost_centre": submitted,
                    "evidence_cost_centre": evidence_cc,
                    "cost_centre_mismatch": mismatch,
                    "cost_centre_note": reasoning,
                    "vat_treatment_applied": vat_treatment or "none",
                },
            )
        except Exception as e:
            return AgentResult(agent_name=self.name, success=False, error=str(e))
