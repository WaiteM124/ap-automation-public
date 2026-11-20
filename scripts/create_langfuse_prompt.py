"""
Creates the LLM gate prompt in Langfuse Prompt Management.
Run once: python scripts/create_langfuse_prompt.py
"""
import os
from dotenv import load_dotenv

load_dotenv()

from langfuse import Langfuse

client = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

PROMPT_NAME = "llm-gate-conflict-analysis"

PROMPT_TEXT = """You are a senior AP controller and AI governance officer reviewing an automated invoice coding decision.

Your job is to identify CONFLICTS in the decision - places where agents disagree with each other or where an agent flag contradicts the RAG evidence - and provide structured rationale for each conflict.

INVOICE:
- Description: {{description}}
- Vendor: {{vendor}}
- Amount: {{currency}} {{amount}}
- Entity: {{entity_id}}

RAG EVIDENCE SUMMARY:
- Lines retrieved: {{evidence_total_lines}}
- GL agreement: {{evidence_gl_agreement_pct}}% of evidence lines coded to {{predicted_gl}}
- Top similarity score: {{evidence_top_similarity}}
- Weighted confidence: {{weighted_confidence}}

RULE GATE DECISION:
- Routing: {{final_routing}}
- Reason: {{gate_reason}}
- Flags raised: {{agent_flags}}

AGENT RESULTS (sequential chain: entity -> vat -> cost_centre):
{{agent_results_json}}

INSTRUCTIONS:
1. Identify every conflict present. A conflict is one of:
   a) An agent flag that contradicts strong RAG evidence (e.g. cost centre mismatch flagged but GL evidence is 100% consistent)
   b) Two agents whose results are logically inconsistent with each other
   c) An agent result that appears incorrect given the invoice context
   d) A case where the rule gate decision seems disproportionate given the evidence

2. For each conflict, explain what the rule gate decided and whether you agree.

3. You MAY escalate auto_post to human_review if you find a genuine risk the rules missed.
   You MUST NOT reverse human_review to auto_post - governance rules are absolute.

4. If there are no conflicts, say so clearly in the summary.

5. Return ONLY valid JSON in this exact format - no preamble, no markdown fences:

{
  "routing": "auto_post" or "human_review",
  "escalated": true or false,
  "reasoning_confidence": 0.0 to 1.0,
  "conflicts": [
    {
      "conflict_id": "c1",
      "agents_involved": ["agent_name"] or ["agent_name", "rag_evidence"],
      "description": "What the conflict is - one sentence, specific to this invoice",
      "challenge": "What a reasonable reviewer would question about this flag or result",
      "resolution": "What should happen and why - reference governance rules if applicable",
      "severity": "high", "medium" or "low"
    }
  ],
  "summary": "2-3 sentence plain English overall rationale. Be specific: name the GL, amounts, agents, flags."
}"""

prompt = client.create_prompt(
    name=PROMPT_NAME,
    prompt=PROMPT_TEXT,
    labels=["production"],
    config={
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "temperature": 0.1,
    },
)

print(f"Created prompt: {prompt.name} (version {prompt.version})")
print(f"Labels: {prompt.labels}")
