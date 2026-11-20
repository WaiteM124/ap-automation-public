# Data readiness

A client uploads a CSV of already-coded invoices and gets a scored report: how ready that history is for the live system, a recommended mode, and a short list of actions. Nothing is written. The score and verdict are computed by deterministic checks; the narrative is templated from the numbers. There is no LLM, no agent framework, and no ERP call.

The file format is the same as `POST /corpus/import` (see that endpoint). Required columns: `vendor_name`, `description`, `gl` (or `account_code` / `gl_account`). Optional: `amount`, `currency`, `entity_id`, `cost_centre`, `date`, `vat` / `tax_code`. Caps are the same: 5 MB and 5,000 rows. Rows that fail import validation are counted as invalid. A GL that is not in the **active** chart is reported (and used in the other checks), not treated as a fatal file error.

## What is checked

| Check | Weight | What it measures |
| --- | --- | --- |
| Vendor consistency | 10 | How many raw vendor-name spellings collapse to the same normalised name (case, punctuation, legal suffixes). Matching is exact-only after that normalisation. |
| GL coding consistency | 16 | Per vendor, how often near-identical descriptions are coded to different GLs. |
| Chart coverage | 14 | Share of the active chart that appears in the file, and share of rows whose GL is not in the chart. |
| Description quality | 10 | Generic or very short text (`Invoice`, `Services`, `Payment`, under 3 characters/words, mostly digits). |
| Volume sufficiency | 14 | Vendors with at least `min_n` lines (the live gate; shipped default is 3), and the share of lines that belong to those vendors. |
| Cost-centre consistency | 6 | Per GL, how consistent the cost centre is. **Not assessed** if the column is absent. |
| VAT completeness | 5 | Share of rows with a VAT/tax code. **Not assessed** if the column is absent (imported history is otherwise `UNASSESSED`). |
| Recency | 8 | Share of lines in the last 12 months, and the age of the newest line. **Not assessed** if there is no date column. |
| Replay estimate | 17 | Seed the first ~60% of lines (by date, else file order) into a throwaway in-process store; replay the remaining ~40% through the live decision path (retrieval + rule gate + governed routing + configured scalar gate) with the offline hashing embedder. Capped at 2,000 replayed lines. Always computed for `standard`; a second replay (the selected profile, or `tuned` when the selection is `standard`) is included so the two figures can be compared. At most two replays per request. |

Checks that are not assessed are dropped from the weighted score and the remaining weights are re-normalised.

Replay reports: lines replayed, automation rate (share that would auto-post), auto-post accuracy (share of auto-posts whose predicted GL equals the client's GL), wrong auto-posts, and overall GL accuracy. Below ~200 usable lines or ~10 replayable lines it returns `insufficient data for a replay estimate` instead of a number. Review-routed lines are written back **only** to the throwaway store so later lines see the same learning the live loop would; auto-posts are never learned from. The live corpus, vendor registry, chart, database and audit trail are not touched.

## Scoring and verdicts

Overall score is the weighted average of assessed category scores (0–100). Recommended mode:

- **auto-post ready** — replay is assessed, automation ≥ 30%, auto-post accuracy ≥ 97%, and no hard fails.
- **suggest-only** — usable history but below that bar, or data-quality warnings.
- **not ready** — too few lines/vendors, more than half the rows invalid or off-chart, a poor replay (auto-post accuracy below 80%), or an overall score below 45.

Hard fails: volume fail; >50% rows invalid; >50% rows off-chart; replay accuracy below 80%. A replay that is merely below the auto-post bar is suggest-only.

**These thresholds are provisional** and should be recalibrated against real client data. The JSON always includes `thresholds_provisional: true`.

## Gate profiles

The replay can be scored under more than one gate. Pass `gate_profile` as a query parameter or form field (`standard`, `tuned`, or `tuned_context`; default `standard`). An unknown value is rejected (422).

| Profile | What it is |
| --- | --- |
| `standard` | Today's behaviour: the live/default gate (all extra `gate_*` knobs off unless you set them as server-wide environment variables). The recommended mode is based on this profile unless you select another. |
| `tuned` | Gate-only settings chosen on public City of Los Angeles payments data to target about 99% auto-post accuracy **without extra columns**. Exact values (from `docs/public_replay_tuned_live.json` → `shadow_mode_sweep.gate.tuned_99`): `top_k=20`, `min_agreement_ratio=1.0`, shipped `min_evidence_lines=3` / `default_confidence_threshold=0.5`, and every extra `gate_*` knob off. On the LA test window that was 34.9% automated at 98.84% accuracy. |
| `tuned_context` | The `combined/tuned_99.5` settings from the same file: `tuned` plus subset retrieval (`context_signals=subset`, `context_min_lines=5`, `top_k=30`), recency window 120 days, diversity 0.5 per vendor code, and `gate_required_context=dept+fund`. On the LA test window that was 71.5% automated at 99.47% accuracy. **Only used when the file actually has department and fund columns** (`entity_id` / `entity` / `department_name` **and** `cost_centre` / `cost_center` / `fund`). Otherwise the response says `not available: file has no context columns` and falls back to `tuned` — never silently. |

Those LA headlines are not a promise for a typical client CSV. The 71.5% / 99.47% figure used LA-specific department and fund fields; without them the gate-only LA result was about 34.9% automated at 98.84%. Tuned profiles need recalibration on each client's data.

The verdict (`mode`) is based on the **selected** profile's replay (after any fallback). The response always includes a `replays` list (each entry has `profile`, `lines`, automation rate, auto-post accuracy, wrong auto-posts) and a plain-English `replay_note`.

**Live gate settings are not changed.** They are server-wide environment variables. This feature only overrides them inside the isolated replay and restores them afterwards, including after an exception.

## Call the endpoint

```bash
curl -s -X POST "$AP_API_URL/readiness?gate_profile=standard" \
  -H "X-API-Key: $AP_API_KEY" \
  -F "file=@history.csv"
```

Response: `score`, `mode`, `mode_reason`, `gate_profile`, `gate_profile_used`, `context_available`, `categories`, `replay` (the selected profile, or an insufficient-data note), `replays` (standard plus the comparison profile), `replay_note`, `actions`, `summary`, `rows` (total / invalid with reasons / capped), `weights`, `weights_used`, `thresholds_provisional`.

Findings may include at most five short examples; they never dump raw invoice text.

Typical duration: a couple of seconds for a small file; a pair of 2,000-line replays typically finishes in well under two minutes (runs in a threadpool).
