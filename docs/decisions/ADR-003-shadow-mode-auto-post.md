# ADR-003: Shadow-mode auto-post before real ERP writes

**Status:** Proposed  
**Date:** 2026-11-24  

---

## Context

[ARCHITECTURE.md](../../ARCHITECTURE.md) defines a decision contract with states including `AUTO_POST`, and states that **auto-post is shadow only initially** — measured without financial side effects. Release 1 scope is recommendation plus human-confirmed write-back; payment execution and production auto-post are excluded. The roadmap phase “Shadow → Limited live” requires a separate control and change approval before real ERP writes.

Today the POC can post a governed `auto_post` to a **simulated** ledger, or (when explicitly enabled) to SAP / Xero adapters that default **off**. There is not yet a first-class shadow mode that records would-be auto-posts with full audit/metrics while guaranteeing zero external financial side effects under production-like config.

## Decision (proposed)

Before enabling real ERP writes for governed `auto_post` in any shared or production environment:

1. Run auto-post in **shadow mode**: compute and record `final_routing == auto_post` with the same gate, audit provenance, and eval metrics, but **do not** call SAP/Xero (or any external ledger write).
2. Keep `SAP_POSTING_ENABLED`, `XERO_POSTING_ENABLED`, and `SAP_ALLOW_NON_SANDBOX` false in committed config until a control owner signs off after shadow metrics meet agreed exit criteria.
3. Promote from shadow → limited live only with an owner, measurable exit evidence, and an update to this ADR (`Proposed` → `Accepted` with the promotion criteria attached).

## Rationale

Shadowing separates “the gate would have posted” from “money moved”. That matches the platform rule that models provide evidence and policy owns decisions, while humans (and explicit control changes) keep authority over financial actions.

## Consequences

- **Not implemented yet** in application code; this ADR documents the planned control from ARCHITECTURE.md so agents and contributors do not treat live ERP auto-post as the default path.
- Implementing shadow mode is application work (out of scope for docs/config-only PRs) and should land with tests, eval hooks, and posting-flag safeguards.
- Until then, treat optional SAP/Xero adapters as opt-in lab features, off by default, never enabled in committed deploy config.
