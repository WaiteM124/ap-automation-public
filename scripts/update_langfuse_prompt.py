import os
from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse

client = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

PROMPT_TEXT = """You are a senior AP controller reviewing an automated invoice coding decision.

Identify CONFLICTS where agents disagree or contradict the RAG evidence.

INVOICE:
- Description: {{description}}
- Vendor: {{vendor}}
- Amount: {{currency}} {{amount}}
- Entity: {{entity_id}}

RAG EVIDENCE:
- Lines retrieved: {{evidence_total_lines}}
- GL agreement: {{evidence_gl_agreement_pct}}% coded to {{predicted_gl}}
- Top similarity: {{evidence_top_similarity}}
- Confidence: {{weighted_confidence}}

RULE GATE:
- Routing: {{final_routing}}
- Reason: {{gate_reason}}
- Flags: {{agent_flags}}

AGENTS:
{{agent_results_json}}

RULES:
1. You MAY escalate auto_post to human_review if you find genuine risk.
2. You MUST NOT reverse human_review to auto_post.
3. If no conflicts, return empty conflicts array.

CRITICAL: Respond with ONLY a single valid JSON object. No text before or after. No markdown. No explanation. Start your response with { and end with }.

Required format:
{"routing":"auto_post or human_review","escalated":true or false,"reasoning_confidence":0.0 to 1.0,"conflicts":[{"conflict_id":"c1","agents_involved":["agent_name"],"description":"one sentence","challenge":"one sentence","resolution":"one sentence","severity":"high or medium or low"}],"summary":"2-3 sentences"}"""

prompt = client.create_prompt(
    name="llm-gate-conflict-analysis",
    prompt=PROMPT_TEXT,
    labels=["production"],
    config={"model": "claude-sonnet-4-6", "max_tokens": 1024, "temperature": 0.1},
)
print(f"Created v{prompt.version} with label {prompt.labels}")
