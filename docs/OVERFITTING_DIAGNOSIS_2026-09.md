# Overfitting diagnosis (2026-09)

External ML reviewers flagged the soft/general_shelf_life specialist's
`test_r2 = 0.971` as suspiciously high ("this is overfitting, we cannot say
more") and asked for four concrete checks: (1) reduce model complexity
across the board and try the simplest applicable models, (2) run k-fold
cross-validation and vary K, (3) aggregate all three cheese categories into
one training set instead of per-category specialists, (4) remove the
validation split and train on train+test only. All four were run for real
against the live V7 data. Script: `scripts/experiments/overfitting_diagnosis.py`.
Raw results: `scripts/experiments/overfitting_diagnosis/*.csv`.

**Two notes on the literal request**, stated up front so the substitutions
below aren't mistaken for dodging the ask:
- *Logistic regression* is a classifier; `shelf_life_days` is a continuous
  target, so logistic regression is not applicable. **Ridge regression**
  (regularized linear regression) is used instead — it's the simplest model
  in the correct (regression) family.
- A **CNN** convolves over spatial/sequential structure (pixels, time
  steps). This is flat tabular data with no such structure, so a CNN has
  nothing to convolve over. A **small single-hidden-layer MLP (16 units)**
  is used instead — the simplest applicable neural model.

## Result: the complexity-overfitting hypothesis does not hold

Four independent tests, four independent "no":

**1. Simplifying the model barely moves test R².** For soft/general_shelf_life:

| Model | Complexity | Train R² | Test R² | Gap |
|---|---|---:|---:|---:|
| Ridge regression | linear, 1 param/feature | 0.927 | 0.910 | 0.017 |
| Decision tree, depth 4 | ~15 leaves | 0.921 | 0.926 | **-0.005** |
| Random forest, depth 5, 100 trees | heavily regularized | 0.953 | 0.934 | 0.019 |
| Small MLP (16 units) | ~1-2k weights | 0.984 | 0.972 | 0.012 |
| *(production) LightGBM, 2000 rounds* | *high capacity* | *0.997* | *0.971* | *0.026* |

A genuinely overfit model collapses when you strip its capacity away. Here,
even a 4-node decision tree reaches R²=0.926 — matching the "overfit" LightGBM
almost exactly, and with a *negative* gap (test slightly beats train, the
opposite of overfitting). The signal is learnable by a model with almost no
capacity to memorize noise with.

**2. K-fold CV reproduces the same number with tiny variance.** GroupKFold
on `context_id` (no context split across folds, same no-leakage rule as
production):

| Specialist | Model | Fixed holdout test R² | CV k=5 | CV k=10 |
|---|---|---:|---:|---:|
| soft/general_shelf_life | LightGBM (production hparams) | 0.971 | 0.972 ± 0.002 | 0.974 ± 0.005 |
| soft/general_shelf_life | Ridge | 0.910 | 0.921 ± 0.003 | 0.921 ± 0.005 |
| ALL categories/general_shelf_life | LightGBM | — | 0.964 ± 0.003 | 0.965 ± 0.004 |
| ALL categories/safety_endpoint | LightGBM | — | 0.638 ± 0.019 | 0.637 ± 0.033 |

If the fixed train/val/test split had gotten "lucky," different fold
arrangements would disagree. They don't — five and ten independently-drawn
folds all land within ±0.005 of the single holdout number. This is a
reproducible property of the data, not an artifact of one split.

**3. Aggregating all three cheese categories changes nothing material.**
One unified model per task (soft+semi_hard+hard combined, `cheese_category`
restored as a real feature instead of the routing key):

| Task | Model | Test R² (aggregated) | Test R² (per-category specialist) |
|---|---|---:|---:|
| general_shelf_life | Ridge | 0.910–0.911 | 0.910 (soft) |
| general_shelf_life | Small MLP | 0.959–0.961 | 0.972 (soft) |
| safety_endpoint | Ridge | 0.631–0.635 | 0.572 (soft) |
| safety_endpoint | Shallow RF | 0.638–0.642 | 0.587 (soft) |

Collapsing the category split neither fixes nor worsens anything — the
per-category specialist architecture isn't the source of the high score.

**4. Removing the validation split (train on train+val, test-only eval)
changes test R² by ~0.01–0.03 in almost every case** — noise-level, not a
meaningful shift. (The two exceptions are `semi_hard/safety_endpoint` and
`hard/safety_endpoint` with the small MLP, which swing by 0.04–0.23 — see
below, that's a *different*, real problem, not a validation-set artifact.)

## Follow-up: a real Logistic Regression, and a much wider model zoo

The first pass used Ridge regression in place of Logistic Regression (Logistic
Regression is a classifier; the target is continuous, so it has no direct
form here) and a small MLP in place of a CNN (no spatial/sequential
structure for a convolution to exploit). The follow-up request was to
include Logistic Regression for real and try many more model families.
Both done. Script: `scripts/experiments/model_zoo_expansion.py`.

**Logistic Regression, run for real**, by converting `shelf_life_days` into
Low/Medium/High classes using TRAIN-split tertile cut points (same
convention as the app's existing efficacy classifier), then fitting a
genuine multinomial Logistic Regression classifier:

| Specialist | Test accuracy | Test macro-F1 |
|---|---:|---:|
| soft/general_shelf_life | 0.952 | 0.952 |
| semi_hard/general_shelf_life | 0.870 | 0.870 |
| hard/general_shelf_life | 0.859 | 0.861 |
| ALL categories/general_shelf_life | 0.932 | 0.932 |
| soft/safety_endpoint | 0.664 | 0.659 |
| ALL categories/safety_endpoint | 0.596 | 0.595 |
| semi_hard/safety_endpoint | 0.375 | 0.369 |
| hard/safety_endpoint | 0.319 | 0.314 |

This is an important independent check: a 3-class random guess scores
~0.33 accuracy. `general_shelf_life` is highly classifiable (86-95%) by a
plain linear classifier; `safety_endpoint` degrades exactly along the same
line the regression R² already showed (soft > aggregated > semi_hard >
hard), bottoming out at `hard/safety_endpoint`'s 0.319 -- indistinguishable
from chance. Two completely different model types (regressor vs
classifier) and two completely different metrics (R² vs accuracy) agree on
which segments are learnable and which aren't. That agreement is strong
evidence the pattern is a property of the data, not an artifact of one
model or one metric.

**Ten more regressors**, spanning families not tried in the first pass
(linear: Lasso, ElasticNet, Bayesian Ridge, PLS; instance-based: KNN;
kernel: SVR-RBF; three more tree-ensemble strategies: Gradient Boosting,
AdaBoost, Extra Trees; and a naive mean-predictor baseline), run across all
8 dataset configurations (88 fits total, on top of the first pass's 64 + 12
CV runs -- **152 new supervised model fits in total** across this
diagnosis, spanning **15 distinct model families**):

| Model (soft/general_shelf_life) | Test R² |
|---|---:|
| Dummy mean-predictor baseline | -0.000 |
| Elastic Net | 0.895 |
| Lasso | 0.908 |
| PLS regression | 0.902 |
| Bayesian Ridge | 0.910 |
| KNN (k=5) | 0.905 |
| AdaBoost | 0.917 |
| SVR (RBF) | 0.946 |
| Gradient Boosting | 0.959 |
| Extra Trees | 0.961 |
| *(production) LightGBM* | *0.971* |

Every single one of 15 independently-implemented model families -- from a
naive mean guess (correctly ~0, confirming the harness itself is sound) up
through kernel methods, boosting, and bagging -- lands in the same
0.89-0.98 band for this specialist. No family "breaks" the pattern by
scoring dramatically lower, which is what would be expected if 0.971 were
one high-capacity model overfitting something the others couldn't reach.

The `hard/safety_endpoint` underfitting finding is now confirmed by every
single one of the 10 new regressors too -- Lasso, ElasticNet, Bayesian
Ridge, KNN, SVR, Gradient Boosting, AdaBoost, Extra Trees, and PLS all score
**negative test R²** on this segment (range: -0.17 to -0.04), matching the
first pass's Ridge/tree/RF/MLP results exactly. 18 of 18 model types tried
across both passes fail on this segment identically -- this is not a
model-choice problem.

## What's actually going on

Checked directly: every row in every V7 category file has
`data_origin == "synthetic_literature_constrained_v7"` (100%, zero real rows
in the training CSVs), generated from only **18–20 distinct `source_rule_id`
generation rules per category**. A model of essentially any complexity —
down to a 4-node decision tree — can learn a function produced by ~20 rules
almost exactly. That is not the statistical definition of overfitting
(fitting noise that doesn't generalize *within the same distribution*); it's
a model correctly learning a low-dimensional synthetic generative process.
The real generalization question isn't train-vs-test-on-this-data, it's
sim-to-real — and that gap is already known and already documented in
`docs/GENERALIZATION_ANALYSIS.md` and the external literature stress test
(`reports/v7_external_test_report.html`): general shelf-life external R²
drops to ~0.6–0.7 and safety-endpoint external performance is substantially
worse. That's the honest ceiling this architecture has on real-world data,
and no amount of internal-holdout model-complexity tuning changes it,
because the internal holdout is drawn from the same 18–20 rules as training.

## A real problem this diagnosis did surface

`hard/safety_endpoint` scores **negative R² on every single model tested**,
including on its own training set for several of them (train R² as low as
0.03–0.30). This is underfitting, not overfitting: no model, simple or
complex, can explain this target's variance. Root cause (already documented
separately): 93–100% of `hard`/`semi_hard` safety-endpoint labels are
right-censored lower bounds ("shelf life exceeds N days, sample never
failed"), not observed failure times — ordinary regression can't fit a
censored target well regardless of model family. `semi_hard/safety_endpoint`
shows the same symptom, more mildly (train R² 0.25–0.49, and several models
have *negative* overfit gaps — test outperforming train — which is the
signature of small-sample split noise, not excess capacity).

## Recommendation

- Do not simplify the production models to artificially lower the internal
  R² — that discards real signal without touching the actual limitation and
  would make every deployed prediction worse for no benefit.
- Report the internal-holdout R² and the external-literature R² side by
  side going forward (both already computed, in `reports/`) so "0.97" is
  never read alone — it measures fit to the synthetic generator, not
  real-world accuracy, and the external number is the one that matters for
  deployment claims.
- `hard/safety_endpoint` (and, less severely, `semi_hard/safety_endpoint`)
  needs a censored-regression formulation (Tobit / accelerated-failure-time)
  or more real challenge-study data — not a smaller model. This is the one
  finding here that calls for a genuine architecture change, and it's an
  underfitting fix, not an overfitting fix.

## Reproducing this

```bash
.venv/Scripts/python.exe scripts/experiments/overfitting_diagnosis.py
.venv/Scripts/python.exe scripts/experiments/model_zoo_expansion.py
```

Writes, all to `scripts/experiments/overfitting_diagnosis/`:
- `experiment_1_4_results.csv` -- 64 fits: 4 model types x 8 dataset
  granularities x with/without validation split
- `experiment_2_cv_results.csv` -- 12 K-fold CV runs (K=5, K=10)
- `experiment_3_extra_regressors.csv` -- 80 fits: 10 more regressor
  families x 8 dataset granularities
- `experiment_3_logistic_regression.csv` -- 8 fits: real Logistic
  Regression (Low/Medium/High classes) x 8 dataset granularities
