"""
A2A Agent Orchestrator.
Runs specialist agents sequentially with context passing:
  1. entity_agent  - resolves canonical entity; enriches context
  2. vat_agent     - receives entity result; enriches context with VAT treatment
  3. cost_centre_agent - receives VAT treatment; narrows cost centre search

Each agent result is merged into the running context before the next agent runs.
Each agent call is traced as a child span in Langfuse under the parent trace.
"""
from __future__ import annotations

import time
from typing import Any

from ap_automation.agents.base import A2AAgent, AgentResult
from ap_automation.agents.cost_centre_agent import CostCentreAgent
from ap_automation.agents.entity_agent import EntityAgent
from ap_automation.agents.vat_agent import VATAgent


class AgentOrchestrationResult:
    """Merged result from all agent calls."""

    def __init__(self, results: list[AgentResult]):
        self.results = results
        self.by_agent = {r.agent_name: r for r in results}

    @property
    def vat(self) -> AgentResult | None:
        return self.by_agent.get("vat_agent")

    @property
    def cost_centre(self) -> AgentResult | None:
        return self.by_agent.get("cost_centre_agent")

    @property
    def entity(self) -> AgentResult | None:
        return self.by_agent.get("entity_agent")

    @property
    def all_successful(self) -> bool:
        return all(r.success for r in self.results)

    def to_dict(self) -> dict:
        return {
            "agents_called": len(self.results),
            "all_successful": self.all_successful,
            "results": [r.to_dict() for r in self.results],
        }


# Ordered pipeline - each agent sees the enriched context from all prior agents
_AGENT_PIPELINE: list[A2AAgent] = [
    EntityAgent(),
    VATAgent(),
    CostCentreAgent(),
]

# Keys each agent exposes that downstream agents consume
_AGENT_CONTEXT_KEYS: dict[str, list[str]] = {
    "entity_agent": ["entity_valid", "expected_entity", "entity_sensitive"],
    "vat_agent": ["vat_treatment", "vat_rate", "vat_flag", "review_required"],
    "cost_centre_agent": [],
}


class AgentOrchestrator:
    """
    Runs registered A2A agents sequentially, passing enriched context forward.
    Safe to use alongside the existing pipeline - never raises,
    never modifies GL prediction or routing decision.
    """

    def __init__(self):
        self._pipeline: list[A2AAgent] = list(_AGENT_PIPELINE)

    def register(self, agent: A2AAgent) -> None:
        self._pipeline.append(agent)

    def agent_cards(self) -> list[dict]:
        return [a.agent_card() for a in self._pipeline]

    def run(
        self,
        context: dict[str, Any],
        langfuse_client=None,
        trace_id: str | None = None,
        parent_observation_id: str | None = None,
    ) -> AgentOrchestrationResult:
        """
        Run agents sequentially. Each agent's output keys are merged into
        the context dict before the next agent executes.
        Langfuse child spans logged after each agent completes.
        """
        running_context = dict(context)
        results: list[AgentResult] = []

        for agent in self._pipeline:
            if not agent.can_handle(running_context):
                continue

            result = self._call_agent(agent, running_context)
            results.append(result)

            # Enrich context with this agent's outputs for downstream agents
            if result.success and result.data:
                passthrough_keys = _AGENT_CONTEXT_KEYS.get(agent.name, [])
                for key in passthrough_keys:
                    if key in result.data:
                        running_context[key] = result.data[key]

            # Log Langfuse child span immediately after each agent
            if langfuse_client:
                try:
                    with langfuse_client.start_as_current_observation(
                        as_type="span",
                        name=f"a2a.{result.agent_name}",
                        input={k: str(v)[:200] for k, v in running_context.items()},
                        metadata={
                            "agent": result.agent_name,
                            "success": result.success,
                            "latency_ms": result.latency_ms,
                        },
                    ):
                        langfuse_client.update_current_span(
                            output=result.to_dict()
                        )
                except Exception:
                    pass

        return AgentOrchestrationResult(results)

    @staticmethod
    def _call_agent(agent: A2AAgent, context: dict[str, Any]) -> AgentResult:
        t0 = time.time()
        try:
            result = agent.run(context)
            result.latency_ms = int((time.time() - t0) * 1000)
            return result
        except Exception as e:
            return AgentResult(
                agent_name=agent.name,
                success=False,
                error=str(e),
                latency_ms=int((time.time() - t0) * 1000),
            )
