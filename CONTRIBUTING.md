# Contributing

Short human guide. Agents and detailed product rules: see [AGENTS.md](AGENTS.md).

## Setup

**Prerequisites:** Python 3.12+, pip. Optional: Docker (Qdrant), OpenAI / Anthropic keys for live embeddings or the LLM gate.

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill locally — never commit .env
```

Default offline path (`pytest`, `scripts/evaluate_poc.py`, `scripts/check_eval_gate.py`) needs **no API keys**.

### Pre-commit (optional but recommended)

```bash
pip install pre-commit
pre-commit install    # ruff + gitleaks on commit
```

## Running

```bash
python scripts/run_demo.py                                    # offline demo API
python -m uvicorn main:app --reload --port 8000               # API (live stack needs Qdrant + keys)
streamlit run ap_automation/ui/app.py                         # UI
```

## Tests, lint, and evals

```bash
ruff check main.py ap_automation tests scripts/run_demo.py scripts/evaluate_poc.py scripts/check_eval_gate.py
pytest -q
python scripts/evaluate_poc.py        # synthetic offline eval (seed 42)
python scripts/check_eval_gate.py     # same + compare to docs/evaluation.json (CI eval gate)
```

Eval numbers are **synthetic**. Label them as such in docs and PRs. Live embeddings (`--live-embeddings`) are optional and not part of CI.

## Branch and PR conventions

- Branch from `main` with a short, descriptive name.
- Keep PRs **small and focused** on one concern.
- Fill in the pull request template checklist (AP rules, tests, eval if needed, no secrets, docs).
- State assumptions, concurrent work, and follow-ups in the description.
- Do not commit secrets; use env vars or a secret manager. Keep ERP posting flags false in committed config; no `CORS_ORIGINS=*` in deploy config.
- Do not delete or weaken tests or eval thresholds solely to make CI pass.

## Definition of done

- AP rules in `AGENTS.md` respected (fail-closed, never-loosen, `post_outcome`, human-only learning, vendor-scoped evidence)
- Tests added/updated and passing; lint clean; eval gate green when routing or thresholds change (and on CI)
- No secrets in the change
- Docs updated when behaviour or workflow changes; synthetic metrics labelled
- CI green on the PR
