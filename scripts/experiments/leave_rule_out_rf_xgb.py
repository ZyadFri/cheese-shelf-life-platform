#!/usr/bin/env python
"""
Fills the one gap in experiment_5_leave_rule_out_full_zoo.csv: Random
Forest and XGBoost (production-hyperparameter versions) under the same
leave-one-rule-out GroupKFold protocol as everything else, so the
"high-capacity ensemble" class has real leave-rule-out numbers for all
three tree-ensemble members (EBM excluded -- it uses a different
preprocessing pipeline, FrameImputer, not the shared scaled
ColumnTransformer everything else here uses; its context-holdout number is
reported instead, clearly labeled as such).

Usage: .venv/Scripts/python.exe scripts/experiments/leave_rule_out_rf_xgb.py
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
OUT_DIR = Path(__file__).resolve().parent / "overfitting_diagnosis"


def load_category_df(category: str) -> pd.DataFrame:
    filename = DATA_VERSION_CONFIG["v7"]["csv_pattern"].format(CAT=category.upper())
    return pd.read_csv(ROOT / "data" / "raw" / filename)


def rf_factory():
    return RandomForestRegressor(n_estimators=400, max_depth=None, min_samples_leaf=2, random_state=SEED, n_jobs=-1)


def xgb_factory():
    import xgboost as xgb
    return xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=6, subsample=0.9, colsample_bytree=0.9,
        random_state=SEED, n_jobs=-1, verbosity=0,
    )


def run_loro(task_df: pd.DataFrame, exclude: list[str], model_name: str, factory) -> dict:
    dynamic_exclude = [c for c in task_df.columns if c != TARGET and c not in exclude and task_df[c].isna().all()]
    schema = detect_schema(task_df, TARGET, exclude=exclude + dynamic_exclude)
    numeric_cols = schema["numeric"] + schema["binary"]
    categorical_cols = schema["categorical"]
    cols = numeric_cols + categorical_cols
    y = task_df[TARGET].to_numpy(dtype=float)
    rules = task_df["source_rule_id"].to_numpy()
    n_rules = task_df["source_rule_id"].nunique()

    gkf = GroupKFold(n_splits=n_rules)
    r2_values = []
    for tr_idx, te_idx in gkf.split(task_df[cols], y, groups=rules):
        pre = build_preprocessor(numeric_cols, categorical_cols)
        X_tr = pre.fit_transform(task_df.iloc[tr_idx][cols])
        X_te = pre.transform(task_df.iloc[te_idx][cols])
        model = factory()
        model.fit(X_tr, y[tr_idx])
        pred = model.predict(X_te)
        m = regression_metrics(y[te_idx], pred)
        r2_values.append(m["r2"])

    return {
        "model": model_name, "n_rules": n_rules,
        "mean_r2": float(np.mean(r2_values)), "median_r2": float(np.median(r2_values)),
        "std_r2": float(np.std(r2_values)), "min_r2": float(np.min(r2_values)), "max_r2": float(np.max(r2_values)),
    }


def main():
    t_start = time.time()
    all_results = []
    for category in CATEGORIES:
        df = load_category_df(category)
        for task in TASKS:
            task_df = df[df["model_task"] == task].reset_index(drop=True)
            label = f"{category}/{task}"
            print(f"{label}  (n={len(task_df)}, n_rules={task_df['source_rule_id'].nunique()})")
            for model_name, factory in (("random_forest_production_hparams", rf_factory), ("xgboost_production_hparams", xgb_factory)):
                t0 = time.time()
                res = run_loro(task_df, EXCLUDED_COLUMNS_V7, model_name, factory)
                res.update({"category": category, "task": task, "duration_sec": time.time() - t0})
                all_results.append(res)
                print(f"  {model_name:32s} mean_r2={res['mean_r2']:7.3f}  median={res['median_r2']:7.3f}  "
                      f"std={res['std_r2']:.3f}  range=[{res['min_r2']:.2f}, {res['max_r2']:.2f}]  ({res['duration_sec']:.1f}s)")

    out_df = pd.DataFrame(all_results)
    out_df.to_csv(OUT_DIR / "experiment_6_leave_rule_out_rf_xgb.csv", index=False)
    print(f"\nTotal duration: {time.time() - t_start:.1f}s")
    print(f"Written: experiment_6_leave_rule_out_rf_xgb.csv ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
