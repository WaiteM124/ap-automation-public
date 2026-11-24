# Architecture decision records

Short records of decisions that affect routing, learning, posting, or the governed platform boundary.

## Format

Each ADR is a Markdown file: `ADR-NNN-short-slug.md`.

| Field | Meaning |
|---|---|
| **Status** | `Proposed` · `Accepted` · `Superseded` · `Deprecated` |
| **Date** | ISO date the decision was recorded |
| **Context** | Problem and constraints (what the code or roadmap requires) |
| **Decision** | What we chose |
| **Rationale** | Why (alternatives considered when useful) |
| **Consequences** | Follow-ons, risks, and what must not change without a new ADR |

## Index

| ADR | Status | Title |
|---|---|---|
| [ADR-001](ADR-001-fail-closed-gate.md) | Accepted | Fail-closed routing gate |
| [ADR-002](ADR-002-human-only-learning.md) | Accepted | Human-only corpus learning |
| [ADR-003](ADR-003-shadow-mode-auto-post.md) | Proposed | Shadow-mode auto-post before real ERP writes |

Threshold changes (`MIN_EVIDENCE_LINES`, agreement / confidence floors, etc.) and any move to put an LLM inside specialist agents also need an ADR here, plus an offline eval run.
