# Part 2.1 finding: the "~400 original experimental examples" claim

**Status: checked against every raw training file in this repository. The
claim could not be confirmed, and the evidence points the other way.**

## What was checked

Every `data/raw/*.xlsx` / `*.csv` file that is or was used as a training
set was loaded and its `data_origin` column value-counted. These are the
five successive dataset versions recoverable in this repository, oldest
first:

| File | Rows | `data_origin` values present |
|---|---:|---|
| `CHEESE_SHELF_LIFE_REVISED_READY_TO_TRAIN.xlsx` (earliest recoverable — matches `model_versions/v1_synthetic_only_2026-08-13_101320/training_manifest.json`, n_total=30500) | 30,500 | `synthetic_literature_constrained` (20,000), `synthetic_cheese_database_anchored` (10,500) |
| `CHEESE_SHELF_LIFE_CORRECTED_GENERALIZED_READY_TO_TRAIN.xlsx` (matches `docs/MODEL_VERSIONS.md` v2, 41,285 rows) | 41,285 | `synthetic_literature_constrained` (20,000), `synthetic_literature_recalibrated` (10,785), `synthetic_cheese_database_anchored` (10,500) |
| `CHEESE_SHELF_LIFE_RECALIBRATED_V3_READY_TO_TRAIN.xlsx` (matches `model_versions/v3_recalibrated_2026-08-19_124027`, 48,240 rows) | 48,240 | `synthetic_literature_recalibrated_v3` (31,800), `retained_soft_rebalanced_v3` (16,440) |
| `CHEESE_SHELF_LIFE_TARGETED_V4_READY_TO_TRAIN.xlsx` | 35,650 | `synthetic_literature_constrained` (16,185), `synthetic_cheese_database_anchored` (9,865), `synthetic_independent_literature_calibrated_v4` (5,150), `synthetic_recalibrated_v4` (4,450) |
| V6/V7 per-category specialist files (`CHEESE_SHELF_LIFE_V7_{SOFT,SEMI_HARD,HARD}_SPECIALIST_CORRECTED.csv`) | 13,600 / 11,200 / 9,200 | `synthetic_literature_constrained_v7` (100% of each) |

**Every `data_origin` value in every training file in this repository,
across all five dataset versions from the earliest recoverable snapshot
onward, is a synthetic-generation tag.** None reads `real`, `experimental`,
`observed`, or similar. `quality_flag` in the earliest file is uniformly
`passed_rules` (30,500/30,500) — a synthetic-generation QA label, not an
experimental-provenance label.

A repo-wide search for a file containing anything close to 400 rows of
non-synthetic data, and a search of `docs/*.md` and the frontend for the
literal number "400", found nothing. The two files with "real" or "external"
in their names are stress-test/validation sets, not training data — and
their sizes are exactly what their filenames say, not the numbers the
audit prompt guessed and asked to be verified:

- `CHEESE_100_EXTERNAL_LITERATURE_TEST_CASES.xlsx`: **100 rows**, 40 columns — confirmed exactly 100, not the "approximately 21" figure the meeting-prompt draft speculated about.
- `CHEESE_319_REAL_EXTERNAL_STRESS_TEST_CASES.xlsx`: **319 rows**, 36 columns — confirmed exactly 319.

These two are held-out literature/stress-test cases used to *evaluate*
trained models (see `reports/v7_external_test_predictions.csv`,
`reports/external_test_predictions.csv`, `reports/stress_test_319_predictions.csv`).
Nothing in their schema or the code that consumes them treats them as a
training source, and — critically — nothing marks them as "the original
~400" either; they are 100 and 319 rows respectively, not ~400 combined
or separately.

## What this means, stated carefully

Per the audit's own evidence rules, three distinct things are true at
different confidence levels, and must not be collapsed into one:

1. **Observed, verified:** no raw or intermediate training file committed
   to this repository at any point in its recoverable history contains a
   non-synthetic `data_origin`. The dataset was already 100% synthetic at
   the earliest snapshot this repo preserves.
2. **Plausible, not verified:** the "~400 original experimental examples"
   may refer to literature-extracted measurements that were used *outside
   this repository* — in the synthetic generation system — to calibrate
   or constrain the generation rules (parameter ranges, target equations),
   without those ~400 source measurements themselves ever being stored as
   rows in a `data/raw/*` file here. This would be consistent with
   `audit_limitations.md` gap #2 (the generation source code, and by
   extension its literature calibration inputs, live in a companion
   repository not present here).
3. **Cannot be established from this repository:** the exact count,
   composition, or independence of whatever original/literature
   measurements seeded the generation rules. That number — whether it was
   400, or something else — is not recoverable from any file this audit
   has access to.

## Consequence for the report

Part 2 of the meeting report (why synthetic data was introduced) cannot be
written as "we started with ~400 real examples and expanded them," because
no such file exists in this repository at any point in its history. The
honest, defensible version for the meeting is the one above: the earliest
recoverable snapshot is already fully synthetic, and the true count and
nature of whatever real/literature data seeded the *generation rules
themselves* is outside this repository's evidence and should be asked
about at the source (the generation-rule repo or whoever built it), not
asserted in this report.

If the professors ask "why didn't you use the original ~400 examples
directly," the honest answer this repository supports is: *"the training
data in this repository was already fully synthetic by the time it enters
version control — the generation system that must have used original
literature/experimental measurements to calibrate its rules lives in a
separate repository we don't have access to audit here."* This should be
stated plainly in the meeting rather than guessed around.

## Source files checked (evidence trail)

- `data/raw/CHEESE_SHELF_LIFE_REVISED_READY_TO_TRAIN.xlsx`
- `data/raw/CHEESE_SHELF_LIFE_CORRECTED_GENERALIZED_READY_TO_TRAIN.xlsx`
- `data/raw/CHEESE_SHELF_LIFE_RECALIBRATED_V3_READY_TO_TRAIN.xlsx`
- `data/raw/CHEESE_SHELF_LIFE_TARGETED_V4_READY_TO_TRAIN.xlsx`
- `data/raw/CHEESE_SHELF_LIFE_V7_SOFT_SPECIALIST_CORRECTED.csv`
- `data/raw/CHEESE_SHELF_LIFE_V7_SEMI_HARD_SPECIALIST_CORRECTED.csv`
- `data/raw/CHEESE_SHELF_LIFE_V7_HARD_SPECIALIST_CORRECTED.csv`
- `data/raw/CHEESE_100_EXTERNAL_LITERATURE_TEST_CASES.xlsx`
- `data/raw/CHEESE_319_REAL_EXTERNAL_STRESS_TEST_CASES.xlsx`
- `model_versions/v1_synthetic_only_2026-08-13_101320/training_manifest.json` (cross-check on the 30,500-row file)
- `model_versions/v3_recalibrated_2026-08-19_124027/artifacts/training_manifest.json` (cross-check on the 48,240-row file)
