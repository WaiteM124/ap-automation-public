# ADR-001: Fail-closed routing gate

**Status:** Accepted  
**Date:** 2026-11-24  

---

## Context

AP automation chooses between `auto_post` and `human_review`. Several failure modes can look like “not enough signal” rather than a hard error: retrieval outages, specialist-agent exceptions, thin or missing vendor history, entity / VAT uncertainty (including non-UK entities), and optional LLM gate failures.

A permissive default (guess through, or let a model relax a restrictive decision) would post invoices the system cannot justify. [ARCHITECTURE.md](../../ARCHITECTURE.md) states that unknown, missing, stale, conflicting or unavailable input cannot authorise posting, and that a restrictive state cannot be downgraded by an agent, model, frontend or retry.

## Decision

The system **fails closed**:

- Retrieval exceptions in `pipeline._predict` yield empty evidence and a review route (never a guessed auto-post).
- Specialist-agent failures are surfaced as flags (`governance.agent_flags`); `governance.apply_gate` sends flagged invoices to `human_review`.
- Sparse evidence (`len(evidence) < MIN_EVIDENCE_LINES`), uncertain MDM identity, weak agreement, or low confidence route to review via `confidence.route`.
- Non-UK / unsupported entities are not auto-posted: the VAT agent returns `NOT_ASSESSED` with `review_required`, and entity mismatches set `entity_valid=False`.
- The optional LLM gate may only **tighten** via `governance.never_loosen` (escalate `auto_post` → `human_review`); it cannot reverse a review decision.

`final_routing` is the governed decision; `AutoPostNotPermitted` blocks force-posting a rejected prediction.

## Rationale

Financial write-backs need a conservative default. Fail-open would maximise automation rate at the cost of silent misposts. Encoding fail-closed in one governance path (`apply_gate` + `never_loosen` + retrieval catch) keeps the imperative pipeline and LangGraph nodes aligned.

## Consequences

- Cold-start and hard-edge (unknown vendor) invoices stay in review until human-confirmed history exists.
- Eval claims require cold-start / hard-edge automation at 0% and anomaly catch-rate 100% on the synthetic set.
- Changing fail-closed behaviour or allowing the LLM to loosen routing requires a new ADR and an offline eval gate run.
