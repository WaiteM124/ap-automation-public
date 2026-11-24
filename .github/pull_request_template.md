## Summary

<!-- What changed and why. State assumptions and out-of-scope items. -->

## Checklist

- [ ] **AP rules respected** — fail-closed; LLM gate only tightens (`never_loosen`); posting only via `post_outcome` / `AutoPostNotPermitted`; human-only learning; vendor-scoped evidence; posting flags false in committed config; no secrets; no `CORS_ORIGINS=*` in deploy config; audit writes kept; specialist agents stay deterministic unless an ADR says otherwise ([AGENTS.md](../AGENTS.md))
- [ ] **Tests added or updated** — gate / posting / learning cases where relevant; existing tests not weakened or deleted to pass
- [ ] **Tests and lint passing** — `ruff check` and `pytest` (CI green)
- [ ] **Eval run if routing or thresholds changed** — `python scripts/check_eval_gate.py` (or rely on CI eval gate); numbers labelled **synthetic**
- [ ] **No secrets** — no API keys, credentials, private keys, or real task-definition values in code or this PR
- [ ] **Docs updated** — README / ADRs / `AGENTS.md` / `CONTRIBUTING.md` updated if behaviour or workflow changed

## Test plan

<!-- Commands run locally, or "rely on CI". -->
