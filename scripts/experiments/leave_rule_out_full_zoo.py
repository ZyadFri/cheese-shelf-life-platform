#!/usr/bin/env python
"""
Leave-rule-out CV (see leave_rule_out_cv.py for the method and rationale)
run across ALL 15 model families from the earlier diagnosis, not just
Ridge + LightGBM. The first leave-rule-out pass showed the honest
generalization number is materially lower than the context-holdout number
for the two models it covered; this extends that same harder test to every
model family so the full picture -- not just the two extremes -- is real
and on the record.

Usage: .venv/Scripts/python.exe scripts/experiments/leave_rule_out_full_zoo.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_specialists import EXCLUDED_COLUMNS_V7, DATA_VERSION_CONFIG, detect_schema, regression_metrics  # noqa: E402
from overfitting_diagnosis import build_preprocessor, make_simple_models  # noqa: E402
from model_zoo_expansion import make_extra_regressors, SVR_MAX_ROWS  # noqa: E402

TARGET = "shelf_life_days"
CATEGORIES = ["soft", "semi_hard", "hard"]
TASKS = ["general_shelf_life", "safety_endpoint"]
SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "overfitting_diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_category_df(category: str) -> pd.DataFrame:
    filename = DATA_VERSION_CONFIG["v7"]["csv_pattern"].format(CAT=category.upper())
    return pd.read_csv(ROOT / "data" / "raw" / filename)


def lgbm_factory():
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
        subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbose=-1,
    )


def all_model_factories() -> dict:
    factories = {}
    for name, model in make_simple_models().items():
        factories[name] = (lambda n=name: make_simple_models()[n])
    for name, model in make_extra_regressors().items():
        factories[name] = (lambda n=name: make_extra_regressors()[n])
    factories["lightgbm_production_hparams"] = lgbm_factory
    return factories


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
    per_rule = []
    for tr_idx, te_idx in gkf.split(task_df[cols], y, groups=rules):
        held_out_rule = task_df.iloc[te_idx]["source_rule_id"].iat[0]
        pre = build_preprocessor(numeric_cols, categorical_cols)
        X_tr_full = pre.fit_transform(task_df.iloc[tr_idx][cols])
        y_tr_full = y[tr_idx]
        if model_name == "svr_rbf" and len(y_tr_full) > SVR_MAX_ROWS:
            rng = np.random.default_rng(SEED)
            sub = rng.choice(len(y_tr_full), size=SVR_MAX_ROWS, replace=False)
            X_tr, y_tr = X_tr_full[sub], y_tr_full[sub]
        else:
            X_tr, y_tr = X_tr_full, y_tr_full
        X_te = pre.transform(task_df.iloc[te_idx][cols])
        model = factory()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        m = regression_metrics(y[te_idx], pred)
        per_rule.append({"held_out_rule": held_out_rule, "n_test_rows": len(te_idx), "r2": m["r2"], "mae": m["mae"]})

    r2_values = [r["r2"] for r in per_rule]
    return {
        "model": model_name, "n_rules": n_rules,
        "mean_r2": float(np.mean(r2_values)), "median_r2": float(np.median(r2_values)),
        "std_r2": float(np.std(r2_values)), "min_r2": float(np.min(r2_values)), "max_r2": float(np.max(r2_values)),
    }


def main():
    t_start = time.time()
    factories = all_model_factories()
    print(f"Model families: {list(factories.keys())} (n={len(factories)})\n")

    all_results = []
    for category in CATEGORIES:
        df = load_category_df(category)
        for task in TASKS:
            task_df = df[df["model_task"] == task].reset_index(drop=True)
            label = f"{category}/{task}"
            print(f"{'=' * 78}\n{label}  (n={len(task_df)}, n_rules={task_df['source_rule_id'].nunique()})\n{'=' * 78}")
            for model_name, factory in factories.items():
                t0 = time.time()
                res = run_loro(task_df, EXCLUDED_COLUMNS_V7, model_name, factory)
                res.update({"category": category, "task": task, "duration_sec": time.time() - t0})
                all_results.append(res)
                print(f"  {model_name:28s} mean_r2={res['mean_r2']:7.3f}  median={res['median_r2']:7.3f}  "
                      f"std={res['std_r2']:.3f}  range=[{res['min_r2']:.2f}, {res['max_r2']:.2f}]  ({res['duration_sec']:.1f}s)")

    out_df = pd.DataFrame(all_results)
    out_df.to_csv(OUT_DIR / "experiment_5_leave_rule_out_full_zoo.csv", index=False)
    print(f"\n{'=' * 78}\nTotal duration: {time.time() - t_start:.1f}s")
    print(f"Written: experiment_5_leave_rule_out_full_zoo.csv ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
