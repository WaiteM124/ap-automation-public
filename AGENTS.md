# AGENTS.md — AP Automation contributor & agent guide

Single source of truth for humans and AI agents (including Cursor) working in this repository.

**Product:** AP Automation — self-learning non-PO invoice GL coding (POC). Predicts GL from the company's coding history, auto-posts only when the governed gate is sure, sends the rest to a human, and **learns only from human confirmations**.

**Owner:** Miles Waite (AI systems architect). Portfolio / showcase copy of a private POC. Synthetic invoice data and public-data replay results only — no live credentials or client ledgers.

**Platform context:** This POC is the first proving ground for a governed enterprise agentic AI platform: models provide evidence, policy owns decisions, humans keep authority over financial actions. See [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Repository map

| Path | Role |
|---|---|
| `main.py` | FastAPI app (single HTTP entry point) |
| `ap_automation/core/` | Pipeline, governance, posting, confidence gate, hybrid retrieval, chart of accounts, SAP/Xero adapters, readiness, synthetic dataset |
| `ap_automation/agents/` | Deterministic specialist agents (entity, VAT, cost centre) + orchestrator |
| `ap_automation/graph/` | LangGraph version of the same pipeline (incl. human-in-the-loop interrupt) |
| `ap_automation/ui/` | Streamlit app |
| `ap_automation/api/` | Mock ERP helper |
| `scripts/` | `evaluate_poc.py` (start here), `check_eval_gate.py`, demos, public-data replay, Langfuse helpers |
| `tests/` | Offline pytest suite |
| `docs/` | Concise docs, synthetic eval baseline (`evaluation.json`), ADRs under `docs/decisions/` |
| `deploy/` | Redacted ECS/Fargate task-definition template (placeholders + Secrets Manager refs) |
| `onboarding/` | Chart-of-accounts and invoice-history CSV templates |
| `.github/workflows/ci.yml` | Lint, pytest, offline eval gate |

Stack: **Python 3.12+**, FastAPI, Pydantic v2, Qdrant (or in-memory/offline lexical), pytest, ruff. Offline path (`pytest`, `scripts/evaluate_poc.py`) needs **no API keys**.

---

## Non-negotiable AP rules

Verify these against the code before changing routing, posting, learning, or deploy config. Phrase and enforce them accurately — do not invent behaviour that is not in the repo.

1. **Fail closed.** Any error, retrieval outage (`pipeline._predict` catch → review), specialist-agent exception (flagged via `governance.agent_flags`), missing / thin evidence (`MIN_EVIDENCE_LINES`), or non-UK entity (VAT `NOT_ASSESSED` / entity mismatch) routes to `human_review`, never `auto_post`.
2. **The LLM gate can only tighten.** Keep `governance.never_loosen`. Model output may escalate `auto_post` → `human_review`; it must never relax routing the other way. Enforced in `pipeline._predict` and `graph/nodes.py`.
3. **All posting goes through `core/posting.py::post_outcome`.** `AutoPostNotPermitted` stays enforced in `pipeline.confirm` when `source` is `AUTO_POST` but `final_routing` is not `auto_post` (or the GL would change). No direct ERP writes that bypass this path.
4. **Only human-confirmed lines are learned.** `pipeline.write_back` returns `False` for `ConfirmationSource.AUTO_POST`. Auto-posts are never written back to the corpus; retrieval also ignores points whose source is `auto_post`.
5. **Evidence is vendor-scoped.** Retrieval is restricted to the resolved vendor; there is no cross-vendor fallback. Auto-post requires at least `MIN_EVIDENCE_LINES` (default 3) for that vendor. Threshold changes need an eval run and a decision record under `docs/decisions/`.
6. **ERP posting flags default to false** (`SAP_POSTING_ENABLED`, `XERO_POSTING_ENABLED`, `SAP_ALLOW_NON_SANDBOX` in `core/config.py` / `.env.example`). Never enable them in committed config (`deploy/`, `render.yaml`, checked-in task defs).
7. **Never commit secrets or task definitions with real values.** Use the redacted `deploy/ecs-task-definition.example.json` plus Secrets Manager. Do not commit `task-definition.json` / `service.json` (gitignored). **No `CORS_ORIGINS=*`** in shared or deploy config — use an explicit allow-list.
8. **Every posting decision is audited.** `post_outcome` writes the audit trail (full provenance). Do not remove or bypass those audit writes.
9. **Specialist agents are deterministic rules** (entity, VAT, cost centre — no LLM). Adding an LLM to them is an architecture decision for `docs/decisions/`.
10. **Eval numbers are synthetic.** Label them as synthetic in docs and PRs. Baseline: `docs/evaluation.json` from offline `scripts/evaluate_poc.py` (seed 42, lexical embedder).

---

## Engineering standards

- **Typed code.** Type hints on public functions and methods. Pydantic models (and typed contracts) at API, persistence, and ERP / model boundaries.
- **Tests as the spec.** Prefer known-answer cases for the gate, posting enforcement, and learning rules. Cover happy path, fail-closed paths, and `AutoPostNotPermitted`.
- **Green before PR.** Lint, tests, and the offline eval gate must pass. CI must stay green.
- **Small, focused PRs.** One concern per PR; clear title and description; state assumptions and out-of-scope items.
- **No secrets in git.** Use `.env` (gitignored) or a secret manager. Prefer `.env.example` for documentation (names only, empty or placeholder values).
- **Pin versions for production-bound work.** Prefer pinned dependency and model versions (`embedding_model`, etc.) when hardening for deploy; do not silently bump models in a feature PR. CI pins **ruff**.
- **Structured logging.** Prefer structured / tagged logs with correlation or prediction ids when available. Do not log secrets (API keys, SAP/Xero credentials, tokens).
- **AI outputs are evaluated, never trusted blindly.** The LLM gate is escalation-only; routing and maths stay independently testable. Do not treat model text as authorisation to post.
- **Docs track behaviour.** Update README / ADRs / this file when behaviour, contracts, thresholds, or run instructions change. Do not invent features or eval scores without measured evidence.

### Definition of done

- [ ] AP rules above respected (especially fail-closed, never-loosen, `post_outcome`, human-only learning, vendor-scoped evidence)
- [ ] Types and contracts in place at boundaries
- [ ] Tests added or updated; existing tests not weakened or deleted to pass
- [ ] Lint, tests, and eval gate pass (`ruff check`, `pytest`, `python scripts/check_eval_gate.py`)
- [ ] No secrets introduced; posting flags remain false in committed config; no `CORS_ORIGINS=*` in deploy config
- [ ] Docs updated if behaviour or workflow changed; synthetic eval numbers labelled as such

---

## How to run, test, lint, and evaluate

### Setup

```bash
pip install -e ".[dev]"          # or: pip install -r requirements.txt && pip install 'ruff==0.17.6' pytest
cp .env.example .env             # fill locally — never commit .env
```

Optional pre-commit (ruff + gitleaks):

```bash
pip install pre-commit
pre-commit install
```

### Run

```bash
# Offline demo API (no cloud credentials)
python scripts/run_demo.py

# Live stack (needs Qdrant + OPENAI_API_KEY)
docker run -p 6333:6333 qdrant/qdrant
python -m uvicorn main:app --reload --port 8000
streamlit run ap_automation/ui/app.py
```

### Test and lint

```bash
ruff check main.py ap_automation tests scripts/run_demo.py scripts/evaluate_poc.py scripts/check_eval_gate.py
pytest -q
```

### Evaluate (synthetic, offline)

```bash
python scripts/evaluate_poc.py              # seed 42, lexical embedder, in-memory Qdrant — no API keys
python scripts/check_eval_gate.py           # same run + compare to docs/evaluation.json (CI gate)
```

Numbers are **synthetic**. See README “Measured results” and `docs/evaluation.json`. Live embeddings: `python scripts/evaluate_poc.py --live-embeddings` (needs `OPENAI_API_KEY`; not used in CI).

---

## Workflow for agents

1. **Read before writing.** Inspect the real layout, `ARCHITECTURE.md`, ADRs, and neighbouring tests. Prefer matching existing patterns over inventing new ones.
2. **Plan large changes.** For multi-file or cross-cutting work, outline the approach, files touched, and risks before coding. Keep scope to the assigned task.
3. **Do not weaken tests or eval thresholds** solely to make CI pass. Fix the code or add a justified, documented change (and an ADR when thresholds move).
4. **State assumptions in the PR.** Call out env expectations, concurrent work, and follow-ups.
5. **Docs/config-only vs product code.** If the task says docs/config/CI only, do not change application behaviour.
6. **Cursor rules.** Follow `.cursor/rules/*.mdc`; they summarise this file for scoped globs. When in conflict, **this file wins**.

---

## Related docs

- [CONTRIBUTING.md](CONTRIBUTING.md) — short human setup and PR conventions
- [README.md](README.md) — product overview and quick start
- [ARCHITECTURE.md](ARCHITECTURE.md) — governed platform target architecture
- [docs/decisions/](docs/decisions/) — ADRs (fail-closed gate, human-only learning, shadow-mode auto-post)
- [docs/evaluation.json](docs/evaluation.json) — synthetic offline eval baseline (seed 42)
