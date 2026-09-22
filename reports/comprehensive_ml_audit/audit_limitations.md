# Audit Limitations and Unresolved Questions

Living document. Every entry records something the audit could **not**
establish from this repository, and why — per the audit's own rule against
inventing content to fill a gap.

## Confirmed gaps

### 1. No pre-V6 commit history
`git log cheese-shelf-life-v7 --oneline` (142 commits total) starts at
`2ea8d67 "Cheese Shelf-Life Studio: full V6/V7 specialist pipeline + Next.js
frontend"` — a single bulk commit that already contains the full V6/V7
system. If V1-V5 dataset/model iterations existed (and file names in
`data/raw/` — `CHEESE_SHELF_LIFE_REVISED_READY_TO_TRAIN.xlsx`,
`CORRECTED_GENERALIZED`, `RECALIBRATED_V3`, `TARGETED_V4` — strongly imply
they did), their development history is not in this repository's git log.
Any reconstruction of that lineage has to come from file timestamps,
`docs/*.md`, and `model_versions/` snapshots, not commit-by-commit diffs.

### 2. Synthetic data-generation source code is not in this repository
Grepped every `.py` file (excluding `.venv`/`node_modules`) for code that
**assigns** `source_rule_id` (as opposed to reading it for grouping/
splitting). Result: none. Every script that touches `source_rule_id`
(`train_specialists.py`, `model_service.py`, `backend/main.py`, the
leave-rule-out experiment scripts, etc.) consumes it as an existing column
in the already-generated CSVs. The actual generation logic — parameter
distributions, sampling ranges, the mathematical transformation from
inputs to `shelf_life_days`, censoring mechanism, noise model — is not
implemented anywhere in this repository. Per the project's own README
(written in an earlier session, `README.md`), that pipeline "lives in a
companion repository, not here."

**Consequence:** Part III of the audit request (reconstruct every
generation rule's implemented mathematical formula from source code)
cannot be performed. What *can* be done instead, and will be, is an
empirical/statistical reverse-engineering of rule *behavior* from the
resulting data (per-rule target distributions, per-rule feature ranges,
per-rule sample sizes) — clearly labeled as inferred-from-data, not
recovered-from-code.

## Confirmed contradictions between sources

### v1 XGBoost test R²: 0.932 (doc) vs 0.954 (raw metrics.json)
`docs/MODEL_VERSIONS.md` line 18 reports, in its "Model Performance (on
Synthetic Test Set)" table: XGBoost Val R²=0.925, **Test R²=0.932**, Test
RMSE=14.03. The raw `model_versions/v1_synthetic_only_2026-08-13_101320/metrics.json`
for the same snapshot (same `created_at_utc`, same dataset path, same
`n_total=30500`) reports **test_r2=0.9540**, test_rmse=18.627 for xgboost.
Neither the R² nor the RMSE matches between the two sources for the same
run. Possible explanations (not yet verified): the markdown table was
transcribed from an earlier/different training run before being
overwritten by a rerun that regenerated `metrics.json` without updating
the doc; a metric-computation change between when the doc was written and
when metrics.json was last regenerated; simple transcription error. This
is preserved as an open discrepancy, not resolved in either source's
favor.

### The sim-to-real generalization gap was flagged from the project's first model version, not discovered recently
`docs/MODEL_VERSIONS.md` (lines 24-38) documents that the **v1** model
(2026-08-13, the earliest recoverable snapshot, pre-dating the V6/V7
specialist architecture by roughly a week) was evaluated against the same
100-case external literature test used later, and the results were
already flagged as a "problem identified": XGBoost external MAE ~50-100
days (a 2-3x overprediction), Fresh Mozzarella true=4d/pred=5.9d (+48%),
Provolone true=45d/pred=126d (+180%) and true=65d/pred=140d (+115%), and
a documented "generalization gap: ~80+ R² points between synthetic and
real test sets." The doc lists four root causes the project team already
suspected at that point: synthetic data not matching real cheese physics,
sparse real data requiring median imputation, the model learning
synthetic correlations rather than causal structure, and the
generalization gap itself. A formal accept/reject decision checklist
(lines 112-129) was already in place at v1 to gate future dataset/model
changes on whether they improved the external-literature MAE, not just
the synthetic test R². This matters for the audit's overfitting chapter:
the sim-to-real gap identified by the 2026-09 leave-rule-out investigation
was not a new discovery — it is the same problem the project had already
identified and was actively tracking a full month earlier, under a
different (pre-specialist, single-dataset) architecture.

## Open questions (pending the archaeology pass)

- Whether `docs/MODEL_VERSIONS.md`, `docs/GENERALIZATION_ANALYSIS.md`,
  `docs/EXPERIMENT_v2_IN_PROGRESS.md`, `docs/WORK_SUMMARY_v2.md` document
  the V1-V5 → V6 → V7 transition well enough to substitute for missing
  commit history. (Archaeology pass in progress.)
- Whether the three `model_versions/` snapshots
  (`v1_synthetic_only_2026-08-13`, `v1_restored_baseline_2026-08-18`,
  `v3_recalibrated_2026-08-19`) contain saved metrics comparable to the
  current V6/V7 numbers, or only model binaries. (Archaeology pass in
  progress.)
- Whether the pre-V7 classification/ingredient-ranking backups
  (`artifacts_classification_pre_v7_backup/`,
  `artifacts_ingredient_ranking_pre_v7_backup/`) have a directly comparable
  metrics schema to the current `artifacts_classification/` /
  `artifacts_ingredient_ranking/`. (Archaeology pass in progress.)
- Independence of the 21-case external literature test
  (`data/raw/CHEESE_100_EXTERNAL_LITERATURE_TEST_CASES.xlsx` /
  `reports/v7_external_test_predictions.csv`) from the synthetic
  generation rules has not yet been audited — this requires checking
  whether any external test case's source publication also appears as a
  cited source in a `source_rule_id` (not yet checked; the rule IDs
  themselves don't carry publication metadata directly retrievable
  without the generation code, per gap #2 above).

## Status
This file is updated as each phase of the audit proceeds. See
`reports/comprehensive_ml_audit/README.md` for the current phase status.
