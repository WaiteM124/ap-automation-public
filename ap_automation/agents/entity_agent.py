"""
Entity Specialist Agent.

Validates that an invoice is posted on a legal entity the vendor is allowed to
invoice, and flags entity-sensitive accounts (capex). Deterministic - no LLM.

VENDOR_ENTITIES mocks tenant / MDM configuration. Keys are the canonical vendor
IDs produced by core/mdm.py (data/vendor_map.csv). Previously most keys here
were vendor IDs that did not exist in the vendor map, so the check rarely ran.
"""
from __future__ import annotations

from ap_automation.agents.base import A2AAgent, AgentResult
from ap_automation.core.chart_of_accounts import get_account

_UK = frozenset({"UK001", "UK002"})
_UK_AND_IE = frozenset({"UK001", "UK002", "IE001"})

VENDOR_ENTITIES: dict[str, frozenset[str]] = {
    "MERIDIAN_FACILITIES": _UK,
    "TECHSOURCE_IT": _UK_AND_IE,
    "GLOBAL_OFFICE": _UK_AND_IE,
    "HORIZON_CONSULT": _UK,
    "SWIFT_TRAVEL": _UK,
    "POWERLINE_UTILS": _UK,
    "BRANDWORKS_MKT": _UK,
    "LEARN_FORWARD": _UK,
    "APEX_EQUIPMENT": _UK,
    "CLEANZONE_FAC": _UK,
    "GREENPATH_UTL": _UK,
    "DATABRIDGE_IT": _UK,
    "SUMMIT_TRAIN": _UK,
    "NOVACORP_SVCS": _UK,
}


class EntityAgent(A2AAgent):
    @property
    def name(self) -> str:
        return "entity_agent"

    @property
    def description(self) -> str:
        return (
            "Validates that the invoice is posted on a legal entity the vendor may invoice, "
            "and flags entity-sensitive (capex) accounts."
        )

    def input_fields(self) -> list[str]:
        return ["vendor_id", "entity_id", "predicted_gl"]

    def output_fields(self) -> list[str]:
        return ["entity_valid", "expected_entity", "entity_sensitive", "entity_note"]

    def can_handle(self, context: dict) -> bool:
        account = get_account(context.get("predicted_gl", "") or "")
        return context.get("vendor_id", "") in VENDOR_ENTITIES or bool(account and account.entity_sensitive)

    def run(self, context: dict) -> AgentResult:
        try:
            vendor_id = context.get("vendor_id", "") or ""
            entity_id = (context.get("entity_id", "") or "").upper()
            gl = context.get("predicted_gl", "") or ""
            account = get_account(gl)
            entity_sensitive = bool(account and account.entity_sensitive)

            allowed = VENDOR_ENTITIES.get(vendor_id)
            entity_valid: bool | None
            if allowed is None:
                entity_valid = None
                reasoning = "Vendor has no entity mapping - entity cannot be validated."
            elif not entity_id:
                entity_valid = False
                reasoning = f"Entity missing - {vendor_id} may invoice {', '.join(sorted(allowed))}."
            elif entity_id in allowed:
                entity_valid = True
                reasoning = f"Entity {entity_id} is permitted for {vendor_id}."
            else:
                entity_valid = False
                reasoning = (
                    f"Entity mismatch: invoice on {entity_id}, {vendor_id} may invoice "
                    f"{', '.join(sorted(allowed))}. Review required."
                )

            if entity_sensitive:
                reasoning += f" GL {gl} is entity-sensitive (capex) - confirm the asset-owning entity."

            return AgentResult(
                agent_name=self.name,
                success=True,
                confidence=0.90 if allowed else 0.60,
                reasoning=reasoning,
                data={
                    "entity_valid": entity_valid,
                    "provided_entity": entity_id,
                    "expected_entity": ", ".join(sorted(allowed)) if allowed else "unknown",
                    "entity_sensitive": entity_sensitive,
                    "entity_note": reasoning,
                },
            )
        except Exception as e:
            return AgentResult(agent_name=self.name, success=False, error=str(e))
