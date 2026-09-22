#!/usr/bin/env python
"""
Leave-one-rule-out CV for the three production tree ensembles (Random
Forest, LightGBM, XGBoost) across all six V7 specialists, saving the
out-of-fold prediction for every individual row rather than only the
per-fold summary metrics. Needed for the fold-level diagnostic chapters
of the meeting report (actual-vs-predicted, residuals, error by product /
preservative, per-rule feature-distribution comparisons).

Outputs, into reports/overfitting_investigation_meeting/:
  rf_fold_predictions.csv     row-level OOF predictions, all 3 models
  fold_diagnostic_summary.csv per-fold metrics, all 3 models

Usage: .venv/Scripts/python.exe scripts/experiments/leave_rule_out_row_level.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_specialists import EXCLUDED_COLUMNS_V7, DATA_VERSION_CONFIG, detect_schema, regression_metrics  # noqa: E402
from overfitting_diagnosis import build_preprocessor  # noqa: E402

TARGET = "shelf_life_days"
CATEGORIES = ["soft", "semi_hard", "hard"]
TASKS = ["general_shelf_life", "safety_endpoint"]
SEED = 42
OUT_DIR = ROOT / "reports" / "overfitting_investigation_meeting"

CONTEXT_COLS = [
    "row_id", "context_id", "base_cheese_name", "food_matrix", "physical_form",
    "treatment_type", "primary_ingredient_name", "primary_ingredient_family",
    "primary_concentration", "primary_concentration_unit",
    "storage_temperature_c", "packaging_type", "matrix_ph", "matrix_water_activity",
    "matrix_salt_pct", "matrix_moisture_pct",
    "indicator_group", "indicator_type", "indicator_threshold",
    "is_control", "is_challenge_test", "target_is_lower_bound", "evidence_class",
]


def load_category_df(category: str) -> pd.DataFrame:
    filename = DATA_VERSION_CONFIG["v7"]["csv_pattern"].format(CAT=category.upper())
    return pd.read_csv(ROOT / "data" / "raw" / filename)


def rf_factory():
    return RandomForestRegressor(n_estimators=400, max_depth=None, min_samples_leaf=2,
                                 random_state=SEED, n_jobs=-1)


def lgbm_factory():
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                             subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                             n_jobs=-1, verbose=-1)


def xgb_factory():
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6,
                            subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                            n_jobs=-1, verbosity=0)


MODELS = [
    ("random_forest", rf_factory),
    ("lightgbm", lgbm_factory),
    ("xgboost", xgb_factory),
]


def main():
    t_start = time.time()
    pred_rows: list[pd.DataFrame] = []
    fold_rows: list[dict] = []

    for category in CATEGORIES:
        df = load_category_df(category)
        for task in TASKS:
            task_df = df[df["model_task"] == task].reset_index(drop=True)
            label = f"{category}/{task}"
            n_rules = task_df["source_rule_id"].nunique()
            print(f"\n{label}  (n={len(task_df)}, n_rules={n_rules})", flush=True)

            dynamic_exclude = [c for c in task_df.columns
                               if c != TARGET and c not in EXCLUDED_COLUMNS_V7
                               and task_df[c].isna().all()]
            schema = detect_schema(task_df, TARGET, exclude=EXCLUDED_COLUMNS_V7 + dynamic_exclude)
            numeric_cols = schema["numeric"] + schema["binary"]
            categorical_cols = schema["categorical"]
            cols = numeric_cols + categorical_cols
            y = task_df[TARGET].to_numpy(dtype=float)
            rules = task_df["source_rule_id"].to_numpy()
            ctx_cols = [c for c in CONTEXT_COLS if c in task_df.columns]

            gkf = GroupKFold(n_splits=n_rules)
            for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(task_df[cols], y, groups=rules)):
                held_out = sorted(set(rules[te_idx]))
                held_out_rule = held_out[0] if len(held_out) == 1 else "|MULTI|".join(held_out)
                pre = build_preprocessor(numeric_cols, categorical_cols)
                X_tr = pre.fit_transform(task_df.iloc[tr_idx][cols])
                X_te = pre.transform(task_df.iloc[te_idx][cols])
                y_tr, y_te = y[tr_idx], y[te_idx]

                base = task_df.iloc[te_idx][ctx_cols].copy()
                base.insert(0, "held_out_rule", held_out_rule)
                base.insert(0, "fold", fold_idx)
                base.insert(0, "task", task)
                base.insert(0, "category", category)
                base["actual_shelf_life_days"] = y_te
                base["n_train_rows"] = len(tr_idx)
                base["n_test_rows"] = len(te_idx)

                for model_name, factory in MODELS:
                    t0 = time.time()
                    model = factory()
                    model.fit(X_tr, y_tr)
                    pred = model.predict(X_te)
                    m = regression_metrics(y_te, pred)

                    block = base.copy()
                    block.insert(3, "model", model_name)
                    block["predicted_shelf_life_days"] = pred
                    block["signed_error_days"] = pred - y_te
                    block["abs_error_days"] = np.abs(pred - y_te)
                    pred_rows.append(block)

                    fold_rows.append({
                        "category": category, "task": task, "model": model_name,
                        "fold": fold_idx, "held_out_rule": held_out_rule,
                        "n_train_rows": len(tr_idx), "n_test_rows": len(te_idx),
                        "n_train_rules": int(pd.Series(rules[tr_idx]).nunique()),
                        "r2": m["r2"], "mae": m["mae"], "rmse": m["rmse"],
                        "y_test_mean": float(np.mean(y_te)), "y_test_std": float(np.std(y_te)),
                        "y_test_min": float(np.min(y_te)), "y_test_max": float(np.max(y_te)),
                        "y_train_mean": float(np.mean(y_tr)), "y_train_std": float(np.std(y_tr)),
                        "pred_mean": float(np.mean(pred)), "pred_std": float(np.std(pred)),
                        "mean_signed_error": float(np.mean(pred - y_te)),
                        "duration_sec": time.time() - t0,
                    })
                    print(f"  fold {fold_idx:2d} {held_out_rule[:44]:44s} {model_name:14s} "
                          f"n_te={len(te_idx):5d} r2={m['r2']:8.3f} mae={m['mae']:7.2f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds = pd.concat(pred_rows, ignore_index=True)
    preds.to_csv(OUT_DIR / "rf_fold_predictions.csv", index=False)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUT_DIR / "fold_diagnostic_summary.csv", index=False)
    print(f"\nTotal duration: {time.time() - t_start:.1f}s")
    print(f"Written: rf_fold_predictions.csv ({len(preds)} rows)")
    print(f"Written: fold_diagnostic_summary.csv ({len(folds)} rows)")


if __name__ == "__main__":
    main()
