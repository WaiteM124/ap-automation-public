from ap_automation.agents.orchestrator import AgentOrchestrator

orch = AgentOrchestrator()
print("Agent cards:")
for card in orch.agent_cards():
    print("  -", card["name"], ":", card["description"][:60])

# --- Case 1: UK entity, standard facilities invoice ---
print("\n--- Case 1: UK entity, standard facilities ---")
context = {
    "predicted_gl": "6300",
    "predicted_cost_centre": "FACILITIES",
    "description": "Office cleaning monthly",
    "amount": 450.0,
    "currency": "GBP",
    "vendor_id": "MERIDIAN_FACILITIES",
    "entity_id": "UK001",
    "routing_decision": "auto_post",
    "weighted_confidence": 0.85,
}
result = orch.run(context)
print("Agents called:", result.to_dict()["agents_called"])
print("VAT treatment:", result.vat.data["vat_treatment"])
print("Cost centre:", result.cost_centre.data["inferred_cost_centre"])
print("Entity valid:", result.entity.data["entity_valid"])
print("All successful:", result.all_successful)

# --- Case 2: Overseas entity - VAT agent should override to ZERO ---
print("\n--- Case 2: Overseas entity (EU001) - expects VAT=ZERO ---")
context2 = {
    "predicted_gl": "6500",
    "predicted_cost_centre": "PROFESSIONAL",
    "description": "Consultancy services Q3",
    "amount": 8500.0,
    "currency": "GBP",
    "vendor_id": "VERTEX_CONSULTING",
    "entity_id": "EU001",
    "routing_decision": "auto_post",
    "weighted_confidence": 0.80,
}
result2 = orch.run(context2)
print("Agents called:", result2.to_dict()["agents_called"])
entity_r = result2.entity
print("Entity valid:", entity_r.data["entity_valid"] if entity_r else "N/A")
print("Expected entity:", entity_r.data["expected_entity"] if entity_r else "N/A")
print("VAT treatment:", result2.vat.data["vat_treatment"] if result2.vat else "N/A")
print("VAT note:", result2.vat.data["vat_note"] if result2.vat else "N/A")
print("Cost centre:", result2.cost_centre.data["inferred_cost_centre"] if result2.cost_centre else "N/A")

# --- Case 3: Entertainment - VAT=BLOCKED should force cost centre to GENERAL ---
print("\n--- Case 3: Entertainment - expects VAT=BLOCKED, CC=GENERAL ---")
context3 = {
    "predicted_gl": "6800",
    "predicted_cost_centre": "MARKETING",
    "description": "Client dinner at The Ivy",
    "amount": 1200.0,
    "currency": "GBP",
    "vendor_id": "APEX_EQUIPMENT",
    "entity_id": "UK001",
    "routing_decision": "auto_post",
    "weighted_confidence": 0.75,
}
result3 = orch.run(context3)
print("Agents called:", result3.to_dict()["agents_called"])
print("VAT treatment:", result3.vat.data["vat_treatment"] if result3.vat else "N/A")
print("Cost centre mismatch:", result3.cost_centre.data["cost_centre_mismatch"] if result3.cost_centre else "N/A")
print("Inferred cost centre:", result3.cost_centre.data["inferred_cost_centre"] if result3.cost_centre else "N/A")
print("Cost centre note:", result3.cost_centre.data["cost_centre_note"] if result3.cost_centre else "N/A")
print("All successful:", result3.all_successful)
