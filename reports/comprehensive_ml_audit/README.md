# Comprehensive ML Audit — Status

This directory holds the evidence-based technical audit of the cheese
shelf-life prediction project's ML development history. Built
incrementally; this file tracks what's real and complete versus pending.

## Phase status

| Phase | Status | Output |
|---|---|---|
| Repository/git scope check | Done | `audit_limitations.md` (2 confirmed structural gaps) |
| Master experiment results table | Done (314 rows) | `master_experiment_results.csv` |
| Legacy v1/v2/v3 pipeline history | Partial | Found in `docs/MODEL_VERSIONS.md` + `model_versions/*/`; 1 contradiction logged, `docs/WORK_SUMMARY_v2.md`/`GENERALIZATION_ANALYSIS.md`/`EXPERIMENT_v2_IN_PROGRESS.md` not yet read |
| Generation-rule reconstruction | Blocked (see limitations doc) | Source code not in this repo; will be inferred statistically from data instead, not yet started |
| Dataset version inventory + stats | Not started | |
| Per-model deep-dive chapters (RF/LightGBM/XGBoost/EBM/others) | Not started | |
| Six-specialist breakdown | Partial | Real numbers already in `master_experiment_results.csv`; narrative chapter not written |
| External literature validation audit + independence check | Not started | Row-level data exists (`reports/v7_external_test_predictions.csv`, `reports/external_test_predictions.csv`, `data/raw/CHEESE_100_EXTERNAL_LITERATURE_TEST_CASES.xlsx`, `data/raw/CHEESE_319_REAL_EXTERNAL_STRESS_TEST_CASES.xlsx`) |
| Censoring/safety-endpoint audit | Not started | |
| Feature/leakage audit | Not started | |
| Hyperparameter/model-selection reconstruction | Not started | |
| Full 80-page narrative report (PDF/MD/DOCX) | Not started | |
| `generation_rules_inventory.csv` | Not started | |
| `model_configurations_history.csv` | Not started | |
| `external_validation_results.csv` | Not started | |
| `source_evidence_index.csv` | Not started | |

## What's real right now

- `master_experiment_results.csv`: 314 rows, every one traceable to a
  specific result file already in this repo (production context-holdout
  metrics for 6 specialists x 4 models x 2 dataset versions, the full
  overfitting-diagnosis suite, and the full 17-model leave-rule-out sweep).
- `audit_limitations.md`: two structural gaps (no pre-V6 git history, no
  generation-rule source code in this repo) and one confirmed numeric
  contradiction between `docs/MODEL_VERSIONS.md` and its own
  `metrics.json`, found by direct comparison, not assumed.

## Why this is being built incrementally

An 80-page, fully evidence-traced report covering 30 chapters and 14
appendices is a genuinely large undertaking if done without inventing
content — which was an explicit, correct requirement of the audit
request. Each phase above is being completed for real before moving to
the next, so the final report is built on verified evidence rather than
assembled and checked for evidence afterward.
