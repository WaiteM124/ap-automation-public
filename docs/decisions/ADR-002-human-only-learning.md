# ADR-002: Human-only corpus learning

**Status:** Accepted  
**Date:** 2026-11-24  

---

## Context

After a posting, the pipeline can index the confirmed line into the vector corpus so future invoices for that vendor retrieve better evidence. If auto-posted lines were also indexed, the corpus would largely agree with the model’s own previous guesses: agreement ratios and automation rate would rise without new human information, and errors could reinforce themselves.

Retrieval already treats evidence as vendor-scoped and excludes points whose source is `auto_post` (`core/retrieval.py`).

## Decision

**Only human-confirmed or human-corrected lines are learned:**

- `pipeline.write_back` returns `False` immediately when `outcome.source == ConfirmationSource.AUTO_POST`.
- `post_outcome` is the single path that confirms → ERP → audit → optional write-back; learning still goes through `write_back`, so auto-posts never enter the corpus.
- Retrieval ignores `source=auto_post` payloads even if such a point were present.

## Rationale

Automation rate should improve from human write-back (and seeded history), not from self-training on unverified auto-posts. The offline eval’s claim “The system never learns from its own auto-posts” (`lines_learned_from_auto_posts == 0`) encodes this.

## Consequences

- New vendors need a short run of human confirms before auto-post (observed: first auto-post after four reviews on the synthetic set).
- Bypass attempts that somehow index auto-posts would violate both write-back and retrieval filters — do not remove either check.
- Changing this policy (e.g. limited auto-post learning under shadow metrics) needs a new ADR, threshold/eval updates, and explicit product approval.
