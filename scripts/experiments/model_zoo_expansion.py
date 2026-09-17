#!/usr/bin/env python
"""
Expanded model zoo, run on top of overfitting_diagnosis.py, in response to
the explicit follow-up request to (a) actually include Logistic Regression
and (b) try many more model families, not just the 4 from the first pass.

Logistic Regression is a classifier -- it cannot predict a continuous
shelf_life_days value. To use it for real (not just explain why it doesn't
apply and skip it), the target is discretized into Low/Medium/High classes
by TRAIN-split tertile cut points (same convention as the app's existing
efficacy classifier), and Logistic Regression is evaluated as a classifier
(accuracy, macro-F1) -- clearly a different task than the regressors below,
reported separately.

Ten additional regressors, spanning distinct model families (linear,
kernel, instance-based, boosting, bagging, dimensionality-reduction, and a
naive baseline):
  Lasso, ElasticNet, BayesianRidge          -- linear family, different
                                                regularization each
  KNeighborsRegressor                        -- instance-based, no
                                                parametric form at all
  SVR (RBF kernel)                           -- kernel method
  GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
                                              -- three more tree-ensemble
                                                strategies distinct from
                                                RF/LightGBM/XGBoost
  PLSRegression                              -- linear, dimensionality
                                                reduction based
  DummyRegressor (mean predictor)            -- naive floor: how much of
                                                the R^2 comes from "just
                                                guess the average"?

Run on the same 8 dataset granularities as overfitting_diagnosis.py (6
per-category specialists + 2 aggregated-across-categories), WITH the
validation split kept (that dimension was already answered honestly in the
first pass) -- this pass's purpose is breadth of model family, not
re-litigating the validation-split question.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_decomposition import PLSRegression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_specialists import EXCLUDED_COLUMNS_V7, detect_schema, regression_metrics  # noqa: E402
from model_service import make_context_splits  # noqa: E402
from overfitting_diagnosis import (  # noqa: E402
    CATEGORIES, TASKS, SEED, TARGET, AGGREGATE_EXCLUDE, load_category_df, build_preprocessor,
)

OUT_DIR = Path(__file__).resolve().parent / "overfitting_diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SVR_MAX_ROWS = 4000  # SVR is O(n^2)-O(n^3); subsample training rows above this for tractable runtime


class PLSRegressionFlat(PLSRegression):
    """PLSRegression.predict() returns shape (n, 1) -- flatten to match
    every other regressor's 1-D output so the shared eval code just works."""
    def predict(self, X, copy=True):
        return super().predict(X, copy=copy).ravel()


def make_extra_regressors() -> dict[str, object]:
    return {
        "lasso": Lasso(alpha=0.01, random_state=SEED, max_iter=5000),
        "elastic_net": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=5000),
        "bayesian_ridge": BayesianRidge(),
        "knn_regressor_k5": KNeighborsRegressor(n_neighbors=5),
        "svr_rbf": SVR(kernel="rbf", C=10.0, epsilon=0.5),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED),
        "adaboost": AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=4), n_estimators=100, random_state=SEED),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "pls_regression": PLSRegressionFlat(n_components=8),
        "dummy_mean_baseline": DummyRegressor(strategy="mean"),
    }


def fit_eval_regressor(model, pre, train_df, val_df, test_df, numeric_cols, categorical_cols, model_name: str):
    cols = numeric_cols + categorical_cols
    X_train = pre.fit_transform(train_df[cols])
    X_val = pre.transform(val_df[cols])
    X_test = pre.transform(test_df[cols])
    y_train = train_df[TARGET].to_numpy(dtype=float)
    y_val = val_df[TARGET].to_numpy(dtype=float)
    y_test = test_df[TARGET].to_numpy(dtype=float)

    if model_name == "svr_rbf" and len(y_train) > SVR_MAX_ROWS:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(y_train), size=SVR_MAX_ROWS, replace=False)
        X_fit, y_fit = X_train[idx], y_train[idx]
        subsampled = True
    else:
        X_fit, y_fit = X_train, y_train
        subsampled = False

    t0 = time.time()
    model.fit(X_fit, y_fit)
    dur = time.time() - t0

    m_train = regression_metrics(y_train, model.predict(X_train))
    m_val = regression_metrics(y_val, model.predict(X_val))
    m_test = regression_metrics(y_test, model.predict(X_test))
    return {
        "train_r2": m_train["r2"], "train_rmse": m_train["rmse"],
        "val_r2": m_val["r2"], "val_rmse": m_val["rmse"],
        "test_r2": m_test["r2"], "test_rmse": m_test["rmse"], "test_mae": m_test["mae"],
        "overfit_gap_train_minus_test_r2": m_train["r2"] - m_test["r2"],
        "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
        "svr_subsampled_to": SVR_MAX_ROWS if subsampled else None,
        "training_duration_sec": dur,
    }


def fit_eval_logistic(pre, train_df, val_df, test_df, numeric_cols, categorical_cols):
    """Discretize shelf_life_days into Low/Medium/High by TRAIN-split
    tertiles (cut points computed on train only, applied unchanged to
    val/test -- no leakage), then fit multinomial Logistic Regression as an
    actual classifier."""
    cols = numeric_cols + categorical_cols
    q1, q2 = train_df[TARGET].quantile([1 / 3, 2 / 3])

    def to_class(y):
        return pd.cut(y, bins=[-np.inf, q1, q2, np.inf], labels=["Low", "Medium", "High"])

    y_train_cls = to_class(train_df[TARGET])
    y_val_cls = to_class(val_df[TARGET])
    y_test_cls = to_class(test_df[TARGET])

    X_train = pre.fit_transform(train_df[cols])
    X_val = pre.transform(val_df[cols])
    X_test = pre.transform(test_df[cols])

    t0 = time.time()
    clf = LogisticRegression(max_iter=3000, random_state=SEED)
    clf.fit(X_train, y_train_cls)
    dur = time.time() - t0

    def scores(y_true, X):
        pred = clf.predict(X)
        return {
            "accuracy": float(accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro")),
        }

    return {
        "class_cutpoints": {"tertile_1": float(q1), "tertile_2": float(q2)},
        "train": scores(y_train_cls, X_train),
        "val": scores(y_val_cls, X_val),
        "test": scores(y_test_cls, X_test),
        "training_duration_sec": dur,
        "n_train": len(y_train_cls), "n_val": len(y_val_cls), "n_test": len(y_test_cls),
    }


def iter_granularities():
    """Yields (granularity, category_label, task, train_df, val_df, test_df,
    numeric_cols, categorical_cols) for the same 8 configs as
    overfitting_diagnosis.py's experiment 1."""
    for category in CATEGORIES:
        df = load_category_df(category)
        for task in TASKS:
            task_df = df[df["model_task"] == task].reset_index(drop=True)
            splits = make_context_splits(task_df, seed=SEED)
            train_df, val_df, test_df = splits["train"], splits["validation"], splits["test"]
            dynamic_exclude = [c for c in task_df.columns if c != TARGET and c not in EXCLUDED_COLUMNS_V7 and task_df[c].isna().all()]
            schema = detect_schema(train_df, TARGET, exclude=EXCLUDED_COLUMNS_V7 + dynamic_exclude)
            yield ("per_category_specialist", category, task, train_df, val_df, test_df,
                   schema["numeric"] + schema["binary"], schema["categorical"])

    all_cat_dfs = {c: load_category_df(c) for c in CATEGORIES}
    for task in TASKS:
        task_df = pd.concat([df[df["model_task"] == task] for df in all_cat_dfs.values()], ignore_index=True)
        splits = make_context_splits(task_df, seed=SEED)
        train_df, val_df, test_df = splits["train"], splits["validation"], splits["test"]
        dynamic_exclude = [c for c in task_df.columns if c != TARGET and c not in AGGREGATE_EXCLUDE and task_df[c].isna().all()]
        schema = detect_schema(train_df, TARGET, exclude=AGGREGATE_EXCLUDE + dynamic_exclude)
        yield ("aggregated_all_categories", "ALL", task, train_df, val_df, test_df,
               schema["numeric"] + schema["binary"], schema["categorical"])


def main():
    t_start = time.time()
    reg_rows, clf_rows = [], []

    for granularity, category, task, train_df, val_df, test_df, numeric_cols, categorical_cols in iter_granularities():
        label = f"{category}/{task}"
        for model_name, model_factory in make_extra_regressors().items():
            pre = build_preprocessor(numeric_cols, categorical_cols)
            model = make_extra_regressors()[model_name]
            res = fit_eval_regressor(model, pre, train_df, val_df, test_df, numeric_cols, categorical_cols, model_name)
            reg_rows.append({"granularity": granularity, "category": category, "task": task, "model": model_name, **res})
            print(f"[{label}] {model_name}: train_r2={res['train_r2']:.3f} test_r2={res['test_r2']:.3f} "
                  f"gap={res['overfit_gap_train_minus_test_r2']:.3f} ({res['training_duration_sec']:.1f}s)")

        pre = build_preprocessor(numeric_cols, categorical_cols)
        res = fit_eval_logistic(pre, train_df, val_df, test_df, numeric_cols, categorical_cols)
        clf_rows.append({"granularity": granularity, "category": category, "task": task, "model": "logistic_regression_tertile_classes", **res})
        print(f"[{label}] logistic_regression (Low/Med/High classes): "
              f"train_acc={res['train']['accuracy']:.3f} test_acc={res['test']['accuracy']:.3f} "
              f"test_macro_f1={res['test']['macro_f1']:.3f}")

    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(OUT_DIR / "experiment_3_extra_regressors.csv", index=False)
    clf_df = pd.json_normalize(clf_rows, sep="_")
    clf_df.to_csv(OUT_DIR / "experiment_3_logistic_regression.csv", index=False)

    print(f"\n-> {len(reg_df)} additional regressor fits written to experiment_3_extra_regressors.csv")
    print(f"-> {len(clf_df)} logistic regression fits written to experiment_3_logistic_regression.csv")
    print(f"Total duration: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
