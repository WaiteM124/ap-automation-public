"""
VAT Specialist Agent.

Determines UK VAT treatment for an invoice line from the chart of accounts, the
description, and the entity the invoice is posted on. Deterministic rules - no LLM.

Treatments: STANDARD (20%), REDUCED (5%), ZERO, EXEMPT, OUTSIDE_SCOPE,
REVERSE_CHARGE (domestic CIS), BLOCKED (input VAT not recoverable), and
NOT_ASSESSED (the entity is outside the jurisdictions these rules cover).

`review_required` is part of the governance contract: the gate sends any invoice
with review_required=True to a human (see core/governance.py).
"""
from __future__ import annotations

import re

from ap_automation.agents.base import A2AAgent, AgentResult
from ap_automation.core.chart_of_accounts import get_account

HIGH_VALUE_THRESHOLD = 25_000.0
SUPPORTED_ENTITY_PREFIXES: tuple[str, ...] = ("UK",)

# (pattern, treatment, note). Word-boundary regexes so that e.g. "cis" does not
# match inside "precision" or "decision".
DESCRIPTION_SIGNALS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\b(entertainment|hospitality|client dinner|client lunch|team lunch|christmas party)\b"),
     "BLOCKED", "Entertainment keyword detected - input VAT blocked"),
    (re.compile(r"\b(construction|cis|building works?|scaffolding|groundworks?)\b"),
     "REVERSE_CHARGE", "Construction keyword detected - domestic reverse charge"),
    (re.compile(r"\binsurance\b"), "EXEMPT", "Insurance keyword - exempt"),
    (re.compile(r"\b(postage|royal mail|stamps)\b"), "EXEMPT", "Postage - exempt"),
    (re.compile(r"\b(salary|salaries|wages|payroll|paye)\b"),
     "OUTSIDE_SCOPE", "Payroll keyword - outside scope of VAT"),
]

# Possible overseas / zero-rated supply: we do not guess, we ask for review.
OVERSEAS_SIGNAL = re.compile(r"\b(export|overseas|international|intra-?eu)\b")

_RATES = {
    "STANDARD": "20%", "REDUCED": "5%", "ZERO": "0%", "EXEMPT": "N/A",
    "OUTSIDE_SCOPE": "N/A", "REVERSE_CHARGE": "20% (RC)",
    "BLOCKED": "20% (blocked)", "NOT_ASSESSED": "N/A",
}

_FLAG_TREATMENTS = ("BLOCKED", "REVERSE_CHARGE", "OUTSIDE_SCOPE")


class VATAgent(A2AAgent):
    @property
    def name(self) -> str:
        return "vat_agent"

    @property
    def description(self) -> str:
        return (
            "Determines UK VAT treatment from the chart of accounts, the description and "
            "the posting entity. Blocks entertainment, flags reverse charge and payroll, "
            "refuses to assess entities outside UK VAT, and flags high-value or "
            "unrecognised accounts for review."
        )

    def input_fields(self) -> list[str]:
        return ["predicted_gl", "description", "amount", "currency", "entity_id", "entity_valid"]

    def output_fields(self) -> list[str]:
        return ["vat_treatment", "vat_rate", "vat_flag", "review_required", "vat_note"]

    def can_handle(self, context: dict) -> bool:
        return True

    def run(self, context: dict) -> AgentResult:
        try:
            gl = context.get("predicted_gl", "") or ""
            description = (context.get("description", "") or "").lower()
            amount = float(context.get("amount", 0.0) or 0.0)
            entity_id = (context.get("entity_id", "") or "").upper()
            entity_valid = context.get("entity_valid")

            if not entity_id.startswith(SUPPORTED_ENTITY_PREFIXES):
                return self._result(
                    gl, "NOT_ASSESSED", False, True,
                    f"Entity {entity_id or 'unknown'} is outside the jurisdictions covered by "
                    "the UK VAT rules - VAT treatment must be set by a reviewer.",
                    known_gl=get_account(gl) is not None,
                )

            treatment, note, review = self._classify(gl, description)

            if OVERSEAS_SIGNAL.search(description):
                review = True
                note += " Description suggests an overseas or export supply - confirm VAT treatment."

            if entity_valid is False:
                review = True
                note += " Entity validation failed - VAT treatment unconfirmed until entity resolved."

            if amount >= HIGH_VALUE_THRESHOLD and treatment == "STANDARD":
                review = True
                note += f" High-value invoice ({amount:,.2f}) - VAT review recommended."

            return self._result(
                gl, treatment, treatment in _FLAG_TREATMENTS, review, note,
                known_gl=get_account(gl) is not None,
            )
        except Exception as e:  # never raise - the orchestrator treats failure as a review flag
            return AgentResult(agent_name=self.name, success=False, error=str(e))

    def _result(self, gl: str, treatment: str, flag: bool, review: bool, note: str,
                known_gl: bool) -> AgentResult:
        return AgentResult(
            agent_name=self.name,
            success=True,
            confidence=0.90 if known_gl else 0.50,
            reasoning=note,
            data={
                "vat_treatment": treatment,
                "vat_rate": _RATES.get(treatment, "N/A"),
                "vat_flag": flag,
                "review_required": review,
                "vat_note": note,
                "gl_code": gl,
            },
        )

    @staticmethod
    def _classify(gl: str, description: str) -> tuple[str, str, bool]:
        for pattern, treatment, note in DESCRIPTION_SIGNALS:
            if pattern.search(description):
                return treatment, note, treatment in ("BLOCKED", "REVERSE_CHARGE")
        account = get_account(gl)
        if account:
            return account.vat_treatment, account.vat_note, False
        return (
            "STANDARD",
            f"GL {gl or '(none)'} is not in the chart of accounts - defaulting to standard "
            "rate; manual review required.",
            True,
        )
