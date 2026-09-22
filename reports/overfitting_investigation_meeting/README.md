# Overfitting Investigation — Meeting Report Status

This directory holds the evidence-based report for the two-hour professor
meeting on the overfitting/generalization question. Built incrementally,
same evidence rules as `reports/comprehensive_ml_audit/`: nothing is
written that isn't traceable to a file already in this repo, and gaps are
stated as gaps rather than filled in.

This project reuses, rather than redoes, verified work already completed
earlier in the same investigation: `docs/OVERFITTING_DIAGNOSIS_2026-09.md`,
`ShelfLife_Overfitting_Report/main.tex` (the earlier 6-page leave-rule-out
report), `reports/comprehensive_ml_audit/master_experiment_results.csv`
(314 rows), and `reports/comprehensive_ml_audit/audit_limitations.md`. New
work happens where the meeting-prompt's requirements go beyond what those
already cover (Part 2 dataset archaeology, Part 9 row-level RF fold
reconstruction, Part 15 external-validation independence).

## Phase status

| Phase | Status | Output |
|---|---|---|
| Part 2.1 — reconstruct the "~400 original examples" claim | Done — **negative result** | See `part2_dataset_origin_finding.md` |
| Part 2.2-2.4 — why synthetic data was needed, defensible answer | Not started | Depends on 2.1 being final |
| Part 3 — generation-rule source-code reconstruction | Blocked (confirmed earlier) | `reports/comprehensive_ml_audit/audit_limitations.md` gap #2; statistical inference from data not yet started |
| Part 4 — original evaluation reconstruction | Reusable | Already in `docs/OVERFITTING_DIAGNOSIS_2026-09.md` + `master_experiment_results.csv`; needs re-framing into this report's narrative, not re-derivation |
| Part 5 — diagnostic experiments 1-4 | Reusable | Same source as above |
| Part 6-7 — shared-rule discovery + leave-rule-out method | Reusable | Same source; `ShelfLife_Overfitting_Report/main.tex` already has the diagrams/tables for this |
| Part 8-9 — Random Forest deep dive + per-fold row-level reconstruction | Not started | Requires rerunning `leave_rule_out_rf_xgb.py` with per-row prediction logging, not just summary metrics |
| Part 10-11 — LightGBM / XGBoost deep dives | Partial | Summary metrics exist; per-fold row-level reconstruction not done |
| Part 12 — EBM + simpler models | Partial | Summary metrics exist for simpler models; EBM leave-rule-out not run (different preprocessing pipeline, noted in earlier work) |
| Part 13 — cross-model comparison + heatmaps | Not started | |
| Part 14 — remaining limitations of leave-rule-out | Reusable | Already written in `docs/OVERFITTING_DIAGNOSIS_2026-09.md` |
| Part 15 — external validation independence audit | Not started | Row-level files exist (`data/raw/CHEESE_100_EXTERNAL_LITERATURE_TEST_CASES.xlsx`, `reports/v7_external_test_predictions.csv`) |
| Part 16 — proposed enhanced pipeline | Not started | |
| Full LaTeX report (60-100 pages) | Not started | |
| `rf_fold_predictions.csv` | Not started | |
| `fold_diagnostic_summary.csv` | Not started | |
| `meeting_questions_and_answers.md` | Not started | |
| `evidence_index.md` | Started | |

## Why this is being built incrementally

Same reasoning as `reports/comprehensive_ml_audit/README.md`: a genuinely
evidence-traced 60-100 page report with row-level fold reconstruction is
real, multi-session work if nothing in it is invented. Each phase is
completed for real, with its evidence checked, before the next one starts.
