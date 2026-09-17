#!/usr/bin/env python
"""
Overfitting diagnosis requested by external ML reviewers, run against the
live V7 data (data/raw/CHEESE_SHELF_LIFE_V7_*_SPECIALIST_CORRECTED.csv).

Reviewer asks, addressed literally below:
  1. Reduce model complexity class across the board and try the simplest
     models (linear/ridge regression, a shallow decision tree, a small
     neural net) instead of only RF/LightGBM/XGBoost/EBM.
     NOTE: "logistic regression" and "CNNs" don't apply here -- the target
     (shelf_life_days) is continuous, not a class label, so logistic
     regression has no meaning; and there is no spatial/sequential
     structure (no images, no time series) for a CNN's convolution to
     exploit on this tabular data. The literal, correct substitutes used
     instead: Ridge regression (regularized linear regression) for
     "logistic regression", and a small single-hidden-layer MLP for "CNN"
     -- both are the simplest members of their respective model families
     that are actually applicable to this problem.
  2. Change K in cross-validation (the production pipeline does not use
     k-fold CV at all -- a single fixed train/validation/test split). Both
     K=5 and K=10 GroupKFold (grouped by context_id, same no-leakage rule
     as production) are run here and compared against the fixed holdout.
  3. Aggregate the three cheese-category datasets into one combined
     training set per task (no more separate soft/semi_hard/hard
     specialists) and retrain.
  4. Remove the validation split entirely -- train on train+validation
     combined, evaluate only on test.

Every one of these is run for real, on real data, with real (not
fabricated) resulting metrics written to
scripts/experiments/overfitting_diagnosis/results.csv and printed as a
summary at the end.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/experiments/ -> repo root
import sys
sys.path.insert(0, str(ROOT))

from train_specialists import (  # noqa: E402
    DATA_VERSION_CONFIG, EXCLUDED_COLUMNS_V7, detect_schema, regression_metrics,
)
from model_service import make_context_splits  # noqa: E402

TARGET = "shelf_life_days"
CATEGORIES = ["soft", "semi_hard", "hard"]
TASKS = ["general_shelf_life", "safety_endpoint"]
SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "overfitting_diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# cheese_category / routing_category are excluded in the production V7 list
# because they're CONSTANT within any one specialist file (they equal the
# routing key itself). Once we aggregate categories together they carry real
# signal, so they must come back in as an actual feature.
AGGREGATE_EXCLUDE = [c for c in EXCLUDED_COLUMNS_V7 if c not in ("cheese_category", "routing_category")]


def load_category_df(category: str) -> pd.DataFrame:
    filename = DATA_VERSION_CONFIG["v7"]["csv_pattern"].format(CAT=category.upper())
    return pd.read_csv(ROOT / "data" / "raw" / filename)


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Scaled preprocessing (unlike the tree-only pipeline in
    train_specialists.py) -- Ridge and the MLP both need standardized
    numeric inputs to be meaningful/stable."""
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric_pipe, numeric_cols), ("cat", categorical_pipe, categorical_cols)])


def make_simple_models() -> dict[str, object]:
    return {
        "ridge_regression": Ridge(alpha=1.0, random_state=SEED),
        "shallow_decision_tree": DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=SEED),
        "shallow_random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=SEED, n_jobs=-1
        ),
        "small_mlp": MLPRegressor(
            hidden_layer_sizes=(16,), alpha=1e-2, max_iter=2000, early_stopping=True,
            n_iter_no_change=15, random_state=SEED,
        ),
    }


def fit_eval(model, pre, train_df, val_df, test_df, numeric_cols, categorical_cols, use_validation: bool):
    """use_validation=False merges validation rows into the training set
    (trial #4: no validation split, train+test only)."""
    fit_df = pd.concat([train_df, val_df], ignore_index=True) if not use_validation else train_df
    cols = numeric_cols + categorical_cols
    X_fit = pre.fit_transform(fit_df[cols])
    X_val = pre.transform(val_df[cols])
    X_test = pre.transform(test_df[cols])
    y_fit = fit_df[TARGET].to_numpy(dtype=float)
    y_val = val_df[TARGET].to_numpy(dtype=float)
    y_test = test_df[TARGET].to_numpy(dtype=float)

    t0 = time.time()
    model.fit(X_fit, y_fit)
    dur = time.time() - t0

    pred_fit = model.predict(X_fit)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    m_fit = regression_metrics(y_fit, pred_fit)
    m_val = regression_metrics(y_val, pred_val)
    m_test = regression_metrics(y_test, pred_test)
    return {
        "train_r2": m_fit["r2"], "train_rmse": m_fit["rmse"], "train_mae": m_fit["mae"],
        "val_r2": m_val["r2"], "val_rmse": m_val["rmse"],
        "test_r2": m_test["r2"], "test_rmse": m_test["rmse"], "test_mae": m_test["mae"],
        "overfit_gap_train_minus_test_r2": m_fit["r2"] - m_test["r2"],
        "n_fit": len(y_fit), "n_val": len(y_val), "n_test": len(y_test),
        "training_duration_sec": dur,
    }


def run_experiment_1_and_4() -> pd.DataFrame:
    """Reduced-complexity model zoo x {per-category specialist, aggregated
    across categories} x {with validation split, without (train+val merged)}."""
    rows = []

    # --- per-category specialists (6 combos), same routing as production ---
    for category in CATEGORIES:
        df = load_category_df(category)
        for task in TASKS:
            task_df = df[df["model_task"] == task].reset_index(drop=True)
            splits = make_context_splits(task_df, seed=SEED)
            train_df, val_df, test_df = splits["train"], splits["validation"], splits["test"]
            dynamic_exclude = [c for c in task_df.columns if c != TARGET and c not in EXCLUDED_COLUMNS_V7 and task_df[c].isna().all()]
            exclude = EXCLUDED_COLUMNS_V7 + dynamic_exclude
            schema = detect_schema(train_df, TARGET, exclude=exclude)
            numeric_cols = schema["numeric"] + schema["binary"]
            categorical_cols = schema["categorical"]

            for model_name, model_factory in make_simple_models().items():
                for use_val in (True, False):
                    pre = build_preprocessor(numeric_cols, categorical_cols)
                    model = make_simple_models()[model_name]
                    res = fit_eval(model, pre, train_df, val_df, test_df, numeric_cols, categorical_cols, use_val)
                    rows.append({
                        "granularity": "per_category_specialist", "category": category, "task": task,
                        "model": model_name, "validation_split_used": use_val, **res,
                    })
                    print(f"[specialist] {category}/{task} {model_name} val_used={use_val}: "
                          f"train_r2={res['train_r2']:.3f} test_r2={res['test_r2']:.3f} gap={res['overfit_gap_train_minus_test_r2']:.3f}")

    # --- aggregated across categories (2 combos: one dataset per task) ---
    all_cat_dfs = {c: load_category_df(c) for c in CATEGORIES}
    for task in TASKS:
        task_df = pd.concat(
            [df[df["model_task"] == task] for df in all_cat_dfs.values()], ignore_index=True
        )
        splits = make_context_splits(task_df, seed=SEED)
        train_df, val_df, test_df = splits["train"], splits["validation"], splits["test"]
        dynamic_exclude = [c for c in task_df.columns if c != TARGET and c not in AGGREGATE_EXCLUDE and task_df[c].isna().all()]
        exclude = AGGREGATE_EXCLUDE + dynamic_exclude
        schema = detect_schema(train_df, TARGET, exclude=exclude)
        numeric_cols = schema["numeric"] + schema["binary"]
        categorical_cols = schema["categorical"]
        assert "cheese_category" in categorical_cols, "aggregation must keep cheese_category as a real feature"

        for model_name, _ in make_simple_models().items():
            for use_val in (True, False):
                pre = build_preprocessor(numeric_cols, categorical_cols)
                model = make_simple_models()[model_name]
                res = fit_eval(model, pre, train_df, val_df, test_df, numeric_cols, categorical_cols, use_val)
                rows.append({
                    "granularity": "aggregated_all_categories", "category": "ALL", "task": task,
                    "model": model_name, "validation_split_used": use_val, **res,
                })
                print(f"[aggregated] ALL/{task} {model_name} val_used={use_val}: "
                      f"train_r2={res['train_r2']:.3f} test_r2={res['test_r2']:.3f} gap={res['overfit_gap_train_minus_test_r2']:.3f}")

    return pd.DataFrame(rows)


def run_experiment_2_cv_k() -> pd.DataFrame:
    """K-fold CV (K=5, K=10), GroupKFold on context_id so no context straddles
    a fold boundary -- same no-leakage principle as the production split.
    Run on the aggregated per-task datasets (the natural place to ask 'what
    does K-fold say') plus the specific specialist the reviewers called out
    (soft/general_shelf_life, test_r2=0.971) with both Ridge (simple) and the
    production-hyperparameter LightGBM (to see whether the flagged number
    moves under CV)."""
    import lightgbm as lgb
    rows = []

    def cv_once(df, exclude, k, model_name, model_factory, group_col="context_id"):
        dynamic_exclude = [c for c in df.columns if c != TARGET and c not in exclude and df[c].isna().all()]
        schema = detect_schema(df, TARGET, exclude=exclude + dynamic_exclude)
        numeric_cols = schema["numeric"] + schema["binary"]
        categorical_cols = schema["categorical"]
        cols = numeric_cols + categorical_cols
        y = df[TARGET].to_numpy(dtype=float)
        groups = df[group_col].to_numpy()
        gkf = GroupKFold(n_splits=k)
        fold_r2, fold_rmse = [], []
        for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(df[cols], y, groups=groups)):
            pre = build_preprocessor(numeric_cols, categorical_cols)
            X_tr = pre.fit_transform(df.iloc[tr_idx][cols])
            X_te = pre.transform(df.iloc[te_idx][cols])
            model = model_factory()
            model.fit(X_tr, y[tr_idx])
            pred = model.predict(X_te)
            m = regression_metrics(y[te_idx], pred)
            fold_r2.append(m["r2"])
            fold_rmse.append(m["rmse"])
        return {
            "k": k, "model": model_name, "fold_r2_mean": float(np.mean(fold_r2)), "fold_r2_std": float(np.std(fold_r2)),
            "fold_rmse_mean": float(np.mean(fold_rmse)), "fold_rmse_std": float(np.std(fold_rmse)),
            "fold_r2_values": [round(v, 4) for v in fold_r2],
        }

    def lgbm_factory():
        return lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
            subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbose=-1,
        )

    def ridge_factory():
        return Ridge(alpha=1.0, random_state=SEED)

    # aggregated per-task datasets
    all_cat_dfs = {c: load_category_df(c) for c in CATEGORIES}
    for task in TASKS:
        task_df = pd.concat([df[df["model_task"] == task] for df in all_cat_dfs.values()], ignore_index=True).reset_index(drop=True)
        for k in (5, 10):
            for model_name, factory in (("ridge_regression", ridge_factory), ("lightgbm_production_hparams", lgbm_factory)):
                res = cv_once(task_df, AGGREGATE_EXCLUDE, k, model_name, factory)
                res.update({"granularity": "aggregated_all_categories", "category": "ALL", "task": task})
                rows.append(res)
                print(f"[CV] ALL/{task} k={k} {model_name}: r2_mean={res['fold_r2_mean']:.3f} +/- {res['fold_r2_std']:.3f}")

    # the specific flagged specialist: soft / general_shelf_life
    flagged_df = load_category_df("soft")
    flagged_df = flagged_df[flagged_df["model_task"] == "general_shelf_life"].reset_index(drop=True)
    for k in (5, 10):
        for model_name, factory in (("ridge_regression", ridge_factory), ("lightgbm_production_hparams", lgbm_factory)):
            res = cv_once(flagged_df, EXCLUDED_COLUMNS_V7, k, model_name, factory)
            res.update({"granularity": "per_category_specialist", "category": "soft", "task": "general_shelf_life"})
            rows.append(res)
            print(f"[CV] soft/general_shelf_life k={k} {model_name}: r2_mean={res['fold_r2_mean']:.3f} +/- {res['fold_r2_std']:.3f}")

    return pd.DataFrame(rows)


def main():
    t_start = time.time()
    print("=" * 78)
    print("EXPERIMENT 1 + 4: reduced-complexity models x granularity x validation-split")
    print("=" * 78)
    df1 = run_experiment_1_and_4()
    df1.to_csv(OUT_DIR / "experiment_1_4_results.csv", index=False)
    print(f"\n  -> {len(df1)} model fits written to {OUT_DIR / 'experiment_1_4_results.csv'}")

    print("\n" + "=" * 78)
    print("EXPERIMENT 2: K-fold cross-validation (K=5, K=10), GroupKFold on context_id")
    print("=" * 78)
    df2 = run_experiment_2_cv_k()
    df2.to_csv(OUT_DIR / "experiment_2_cv_results.csv", index=False)
    print(f"\n  -> {len(df2)} CV runs written to {OUT_DIR / 'experiment_2_cv_results.csv'}")

    print(f"\nTotal duration: {time.time() - t_start:.1f}s")
    print(f"Total NEW models trained (experiment 1+4 fits, excluding CV folds): {len(df1)}")


if __name__ == "__main__":
    main()
