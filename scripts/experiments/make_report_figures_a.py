#!/usr/bin/env python
"""
Figure generator (part A) for the overfitting-investigation meeting report.
Everything here is built from result files that already exist in the repo:
  reports/comprehensive_ml_audit/master_experiment_results.csv
  scripts/experiments/overfitting_diagnosis/experiment_*.csv
  data/raw/CHEESE_SHELF_LIFE_V7_*_SPECIALIST_CORRECTED.csv

No value is typed by hand; every number plotted is read from those files.
Output: reports/overfitting_investigation_meeting/figures/*.pdf

Usage: .venv/Scripts/python.exe scripts/experiments/make_report_figures_a.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent.parent
EXP = ROOT / "scripts" / "experiments" / "overfitting_diagnosis"
OUT = ROOT / "reports" / "overfitting_investigation_meeting" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BURGUNDY = "#7A1B2E"
BURG_MID = "#A84C5B"
BURG_LIGHT = "#C9939B"
PALE = "#F5E6E8"
GREEN = "#1E4D3B"
GREEN_MID = "#4C7A66"
GREEN_LIGHT = "#A8C4B7"
INK = "#1F1B1A"
GREY = "#8A8583"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.edgecolor": INK,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

MODEL_COLOR = {
    "random_forest": BURGUNDY,
    "lightgbm": GREEN,
    "xgboost": BURG_MID,
    "ebm": GREEN_MID,
}
SPEC_ORDER = [
    ("soft", "general_shelf_life"), ("semi_hard", "general_shelf_life"), ("hard", "general_shelf_life"),
    ("soft", "safety_endpoint"), ("semi_hard", "safety_endpoint"), ("hard", "safety_endpoint"),
]
SPEC_LABEL = {
    ("soft", "general_shelf_life"): "soft\ngeneral",
    ("semi_hard", "general_shelf_life"): "semi-hard\ngeneral",
    ("hard", "general_shelf_life"): "hard\ngeneral",
    ("soft", "safety_endpoint"): "soft\nsafety",
    ("semi_hard", "safety_endpoint"): "semi-hard\nsafety",
    ("hard", "safety_endpoint"): "hard\nsafety",
}


def tidy(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#DDD8D6", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


master = pd.read_csv(ROOT / "reports" / "comprehensive_ml_audit" / "master_experiment_results.csv")
prod = master[master.phase == "production_context_holdout"].copy()


# ---------------------------------------------------------------- fig 01
def fig01_production_v7():
    d = prod[prod.dataset_version == "v7"]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    models = ["random_forest", "lightgbm", "xgboost", "ebm"]
    w = 0.2
    x = np.arange(len(SPEC_ORDER))
    for i, mdl in enumerate(models):
        vals = [d[(d.cheese_category == c) & (d.prediction_task == t) & (d.model_family == mdl)].test_r2.iloc[0]
                for c, t in SPEC_ORDER]
        ax.bar(x + (i - 1.5) * w, vals, w, label=mdl.replace("_", " "), color=MODEL_COLOR[mdl], zorder=3)
    ax.axhline(0, color=INK, linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([SPEC_LABEL[s] for s in SPEC_ORDER])
    ax.set_ylabel(r"test $R^2$ (context holdout)")
    ax.set_title("V7 production specialists: original context-holdout test $R^2$")
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, -0.38))
    tidy(ax)
    save(fig, "fig01_production_v7_test_r2")


# ---------------------------------------------------------------- fig 02
def fig02_v6_vs_v7():
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    x = np.arange(len(SPEC_ORDER))
    for i, (ver, col) in enumerate([("v6", GREEN), ("v7", BURGUNDY)]):
        d = prod[(prod.dataset_version == ver) & (prod.model_family == "lightgbm")]
        vals = [d[(d.cheese_category == c) & (d.prediction_task == t)].test_r2.iloc[0] for c, t in SPEC_ORDER]
        ax.bar(x + (i - 0.5) * 0.36, vals, 0.36, label=f"{ver.upper()} dataset", color=col, zorder=3)
    ax.axhline(0, color=INK, linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([SPEC_LABEL[s] for s in SPEC_ORDER])
    ax.set_ylabel(r"test $R^2$")
    ax.set_title("Dataset version effect (LightGBM): V6 $\\rightarrow$ V7 under the same split protocol")
    ax.legend(ncol=2, loc="lower left")
    tidy(ax)
    save(fig, "fig02_v6_vs_v7_lightgbm")


# ---------------------------------------------------------------- fig 03
def fig03_train_test_gap():
    d = prod[prod.dataset_version == "v7"]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    x = np.arange(len(SPEC_ORDER))
    models = ["random_forest", "lightgbm", "xgboost", "ebm"]
    w = 0.2
    for i, mdl in enumerate(models):
        gaps = [d[(d.cheese_category == c) & (d.prediction_task == t) & (d.model_family == mdl)].train_r2.iloc[0]
                - d[(d.cheese_category == c) & (d.prediction_task == t) & (d.model_family == mdl)].test_r2.iloc[0]
                for c, t in SPEC_ORDER]
        ax.bar(x + (i - 1.5) * w, gaps, w, label=mdl.replace("_", " "), color=MODEL_COLOR[mdl], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([SPEC_LABEL[s] for s in SPEC_ORDER])
    ax.set_ylabel(r"train $R^2$ $-$ test $R^2$")
    ax.set_title("Optimism gap of the V7 production fits (train minus context-holdout test)")
    ax.legend(ncol=4, loc="upper left")
    tidy(ax)
    save(fig, "fig03_train_test_gap_v7")


# ---------------------------------------------------------------- fig 04/05/06
def load_specialists():
    frames = []
    for cat in ["SOFT", "SEMI_HARD", "HARD"]:
        df = pd.read_csv(ROOT / "data" / "raw" / f"CHEESE_SHELF_LIFE_V7_{cat}_SPECIALIST_CORRECTED.csv")
        df["cheese_category"] = cat.lower()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def fig04_dataset_composition(spec):
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    cats = ["soft", "semi_hard", "hard"]
    gen = [len(spec[(spec.cheese_category == c) & (spec.model_task == "general_shelf_life")]) for c in cats]
    saf = [len(spec[(spec.cheese_category == c) & (spec.model_task == "safety_endpoint")]) for c in cats]
    x = np.arange(len(cats))
    ax.bar(x, gen, 0.55, label="general shelf life", color=BURGUNDY, zorder=3)
    ax.bar(x, saf, 0.55, bottom=gen, label="safety endpoint", color=GREEN, zorder=3)
    for i, (g, s) in enumerate(zip(gen, saf)):
        ax.text(i, g / 2, f"{g:,}", ha="center", va="center", color="white", fontsize=7.5)
        ax.text(i, g + s / 2, f"{s:,}", ha="center", va="center", color="white", fontsize=7.5)
        ax.text(i, g + s + 250, f"{g+s:,} rows", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["soft", "semi-hard", "hard"])
    ax.set_ylabel("synthetic rows")
    ax.set_title("V7 training data: every row is synthetic, split across six specialists")
    ax.legend(ncol=2, loc="upper right")
    tidy(ax)
    save(fig, "fig04_dataset_composition")


def fig05_rule_counts(spec):
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    labels, nrules, nrows = [], [], []
    for c, t in SPEC_ORDER:
        sub = spec[(spec.cheese_category == c) & (spec.model_task == t)]
        labels.append(SPEC_LABEL[(c, t)].replace("\n", " "))
        nrules.append(sub.source_rule_id.nunique())
        nrows.append(len(sub))
    x = np.arange(len(labels))
    ax.bar(x, nrules, 0.55, color=BURGUNDY, zorder=3)
    for i, (r, n) in enumerate(zip(nrules, nrows)):
        ax.text(i, r + 0.15, f"{r} rules\n{n:,} rows", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("distinct source_rule_id")
    ax.set_ylim(0, max(nrules) * 1.35)
    ax.set_title("Effective information: thousands of rows, but only 7--11 generation rules per specialist")
    tidy(ax)
    save(fig, "fig05_rule_counts")


def fig06_rule_size_distribution(spec):
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.0))
    for ax, (c, t) in zip(axes.ravel(), SPEC_ORDER):
        sub = spec[(spec.cheese_category == c) & (spec.model_task == t)]
        vc = sub.source_rule_id.value_counts().sort_values(ascending=True)
        ax.barh(np.arange(len(vc)), vc.values, color=BURG_MID, zorder=3, height=0.7)
        ax.set_yticks(np.arange(len(vc)))
        ax.set_yticklabels([s.replace("V7|", "")[:26] for s in vc.index], fontsize=5.2)
        ax.set_xscale("log")
        ax.set_title(SPEC_LABEL[(c, t)].replace("\n", " "), fontsize=8)
        ax.tick_params(axis="x", labelsize=6)
        tidy(ax, grid_axis="x")
    fig.suptitle("Rows per generation rule (log scale): every specialist is dominated by one or two rules",
                 fontsize=9.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig06_rule_size_distribution")


def fig07_rule_concentration(spec):
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    for (c, t), col in zip(SPEC_ORDER, [BURGUNDY, BURG_MID, BURG_LIGHT, GREEN, GREEN_MID, GREEN_LIGHT]):
        sub = spec[(spec.cheese_category == c) & (spec.model_task == t)]
        vc = sub.source_rule_id.value_counts().sort_values(ascending=False)
        cum = np.cumsum(vc.values) / vc.values.sum()
        ax.plot(np.arange(1, len(cum) + 1), cum * 100, marker="o", ms=3, color=col, lw=1.3,
                label=SPEC_LABEL[(c, t)].replace("\n", " "))
    ax.axhline(90, color=GREY, ls="--", lw=0.7)
    ax.text(0.6, 91, "90% of rows", fontsize=6.5, color=GREY)
    ax.set_xlabel("number of largest generation rules included")
    ax.set_ylabel("cumulative share of rows (%)")
    ax.set_title("Concentration of the synthetic corpus in a handful of rules")
    ax.legend(ncol=3, fontsize=6.5)
    tidy(ax)
    save(fig, "fig07_rule_concentration")


# ---------------------------------------------------------------- fig 08-11 diagnostics
def fig08_complexity_classes():
    e14 = pd.read_csv(EXP / "experiment_1_4_results.csv")
    e3 = pd.read_csv(EXP / "experiment_3_extra_regressors.csv")
    d14 = e14[(e14.granularity == "per_category_specialist") & (e14.validation_split_used)]
    d3 = e3[e3.granularity == "per_category_specialist"]
    rows = pd.concat([d14[["category", "task", "model", "test_r2"]], d3[["category", "task", "model", "test_r2"]]])
    piv = rows.groupby("model").test_r2.mean().sort_values()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    colors = [BURGUNDY if v >= 0.8 else (BURG_MID if v >= 0.5 else GREY) for v in piv.values]
    ax.barh(np.arange(len(piv)), piv.values, color=colors, zorder=3, height=0.72)
    ax.set_yticks(np.arange(len(piv)))
    ax.set_yticklabels([m.replace("_", " ") for m in piv.index], fontsize=7)
    for i, v in enumerate(piv.values):
        ax.text(v + 0.012 if v > 0 else 0.012, i, f"{v:.3f}", va="center", fontsize=6.5, color=INK)
    ax.set_xlabel(r"mean test $R^2$ across the six specialists (context holdout)")
    ax.set_title("Model complexity does not explain the high scores")
    ax.set_xlim(min(0, piv.values.min() * 1.1), 1.06)
    tidy(ax, grid_axis="x")
    save(fig, "fig08_complexity_classes")


def fig09_validation_removal():
    e14 = pd.read_csv(EXP / "experiment_1_4_results.csv")
    d = e14[e14.granularity == "per_category_specialist"]
    piv = d.pivot_table(index=["category", "task", "model"], columns="validation_split_used", values="test_r2")
    piv = piv.dropna()
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.scatter(piv[True], piv[False], s=22, color=BURGUNDY, alpha=0.8, zorder=3, edgecolor="white", linewidth=0.4)
    lo = min(piv[True].min(), piv[False].min()) - 0.02
    ax.plot([lo, 1.0], [lo, 1.0], color=GREY, ls="--", lw=0.8)
    ax.set_xlabel(r"test $R^2$ with validation split (70/15/15)")
    ax.set_ylabel(r"test $R^2$ without validation split (85/15)")
    ax.set_title("Experiment 4: removing the validation set")
    delta = (piv[False] - piv[True]).mean()
    ax.text(0.03, 0.95, f"mean change: {delta:+.4f} $R^2$\nn = {len(piv)} model$\\times$specialist fits",
            transform=ax.transAxes, va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PALE, edgecolor="none"))
    tidy(ax, grid_axis="both")
    save(fig, "fig09_validation_removal")


def fig10_kfold():
    cv = pd.read_csv(EXP / "experiment_2_cv_results.csv")
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    labels, means, stds, cols = [], [], [], []
    for _, r in cv.iterrows():
        gran = "aggregated" if r.granularity.startswith("agg") else f"{r.category}"
        labels.append(f"{r.model.replace('_',' ')}\nK={r.k}, {gran}")
        means.append(r.fold_r2_mean)
        stds.append(r.fold_r2_std)
        cols.append(BURGUNDY if r.model == "ridge_regression" else GREEN)
    order = np.argsort(means)
    x = np.arange(len(labels))
    ax.bar(x, np.array(means)[order], 0.6, yerr=np.array(stds)[order], color=np.array(cols)[order],
           zorder=3, error_kw=dict(ecolor=INK, lw=0.7, capsize=2))
    ax.set_xticks(x)
    ax.set_xticklabels(np.array(labels)[order], fontsize=5.6)
    ax.set_ylabel(r"cross-validated $R^2$ (mean $\pm$ s.d.)")
    ax.set_ylim(0, 1.03)
    ax.set_title("Experiment 2: changing $K$ changes nothing, because the grouping did not change")
    tidy(ax)
    save(fig, "fig10_kfold_cv")


def fig11_granularity():
    e14 = pd.read_csv(EXP / "experiment_1_4_results.csv")
    d = e14[e14.validation_split_used]
    g = d.groupby(["granularity", "model"]).test_r2.mean().unstack()
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    gran_order = [g_ for g_ in ["per_category_specialist", "per_category_all_tasks",
                                "aggregated_all_categories", "aggregated_all"] if g_ in g.index]
    if not gran_order:
        gran_order = list(g.index)
    x = np.arange(len(gran_order))
    models = list(g.columns)
    w = 0.8 / len(models)
    palette = [BURGUNDY, BURG_MID, GREEN, GREEN_MID, BURG_LIGHT, GREEN_LIGHT]
    for i, mdl in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * w, g.loc[gran_order, mdl], w,
               label=mdl.replace("_", " "), color=palette[i % len(palette)], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in gran_order], fontsize=7)
    ax.set_ylabel(r"mean test $R^2$")
    ax.set_title("Experiment 3: aggregating categories into one training set")
    ax.legend(ncol=4, fontsize=6.5)
    ax.set_ylim(0, 1.05)
    tidy(ax)
    save(fig, "fig11_granularity")


def fig12_logistic():
    lg = pd.read_csv(EXP / "experiment_3_logistic_regression.csv")
    d = lg[lg.granularity == "per_category_specialist"]
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    labels = [SPEC_LABEL.get((r.category, r.task), f"{r.category} {r.task}").replace("\n", " ")
              for _, r in d.iterrows()]
    x = np.arange(len(d))
    ax.bar(x - 0.2, d.test_accuracy, 0.4, label="test accuracy", color=BURGUNDY, zorder=3)
    ax.bar(x + 0.2, d.test_macro_f1, 0.4, label="test macro $F_1$", color=GREEN, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Experiment 1b: logistic regression on tertile-binned shelf life")
    ax.legend(ncol=2, fontsize=7)
    tidy(ax)
    save(fig, "fig12_logistic_regression")


# ---------------------------------------------------------------- fig 13-17 leave-rule-out
def load_zoo():
    z5 = pd.read_csv(EXP / "experiment_5_leave_rule_out_full_zoo.csv")
    z6 = pd.read_csv(EXP / "experiment_6_leave_rule_out_rf_xgb.csv")
    return pd.concat([z5, z6], ignore_index=True)


def fig13_zoo_ranking(zoo):
    agg = zoo.groupby("model")[["mean_r2", "median_r2"]].mean().sort_values("median_r2")
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    y = np.arange(len(agg))
    ax.barh(y - 0.19, agg.mean_r2, 0.38, label=r"mean $R^2$ across folds", color=BURG_LIGHT, zorder=3)
    ax.barh(y + 0.19, agg.median_r2, 0.38, label=r"median $R^2$ across folds", color=BURGUNDY, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([m.replace("_", " ") for m in agg.index], fontsize=7)
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_xlabel(r"leave-rule-out $R^2$, averaged over the six specialists")
    ax.set_title("Leave-one-rule-out ranking of all 17 evaluated models")
    ax.legend(loc="lower left", fontsize=7)
    tidy(ax, grid_axis="x")
    save(fig, "fig13_zoo_leave_rule_out_ranking")


def fig14_holdout_vs_loro(zoo):
    e14 = pd.read_csv(EXP / "experiment_1_4_results.csv")
    e3 = pd.read_csv(EXP / "experiment_3_extra_regressors.csv")
    ho = pd.concat([
        e14[(e14.granularity == "per_category_specialist") & (e14.validation_split_used)][["category", "task", "model", "test_r2"]],
        e3[e3.granularity == "per_category_specialist"][["category", "task", "model", "test_r2"]],
    ])
    ho_m = ho.groupby("model").test_r2.mean()
    lo_m = zoo.groupby("model").median_r2.mean()
    common = sorted(set(ho_m.index) & set(lo_m.index), key=lambda m: -ho_m[m])
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for i, m in enumerate(common):
        ax.plot([0, 1], [ho_m[m], lo_m[m]], color=BURG_MID, lw=1.0, alpha=0.75, zorder=2)
        ax.scatter([0], [ho_m[m]], s=20, color=BURGUNDY, zorder=3)
        ax.scatter([1], [lo_m[m]], s=20, color=GREEN, zorder=3)
        ax.text(-0.03, ho_m[m], m.replace("_", " "), ha="right", va="center", fontsize=6)
        ax.text(1.03, lo_m[m], f"{lo_m[m]:.2f}", ha="left", va="center", fontsize=6)
    ax.set_xlim(-0.55, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["context holdout\n(original protocol)", "leave-rule-out\n(revised protocol)"], fontsize=7.5)
    ax.set_ylabel(r"$R^2$")
    ax.axhline(0, color=GREY, lw=0.7, ls="--")
    ax.set_title("What changes when the evaluation changes, not the model")
    tidy(ax, grid_axis="y")
    save(fig, "fig14_holdout_vs_loro_slope")


def fig15_loro_heatmap(zoo, value="mean_r2", name="fig15_loro_heatmap_mean", title=None):
    zoo = zoo.copy()
    zoo["spec"] = zoo.category + "/" + zoo.task
    piv = zoo.pivot_table(index="model", columns="spec", values=value)
    cols = [f"{c}/{t}" for c, t in SPEC_ORDER if f"{c}/{t}" in piv.columns]
    piv = piv[cols]
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    data = piv.values
    vmax = 1.0
    vmin = max(-2.0, np.nanmin(data))
    im = ax.imshow(np.clip(data, vmin, vmax), cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([c.replace("/", "\n") for c in piv.columns], fontsize=6.5)
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([m.replace("_", " ") for m in piv.index], fontsize=6.5)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                        color="white" if (v < vmin + 0.45 * (vmax - vmin)) else INK)
    ax.set_title(title or r"Leave-rule-out mean $R^2$ by model and specialist")
    fig.colorbar(im, ax=ax, shrink=0.6, label=r"$R^2$ (clipped at $-2$)")
    save(fig, name)


def fig17_mean_vs_median(zoo):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(zoo.median_r2, zoo.mean_r2, s=24, color=BURGUNDY, alpha=0.75,
               edgecolor="white", linewidth=0.4, zorder=3)
    lim_lo = min(zoo.mean_r2.min(), zoo.median_r2.min())
    ax.plot([lim_lo, 1], [lim_lo, 1], ls="--", color=GREY, lw=0.8)
    ax.set_xlabel(r"median fold $R^2$")
    ax.set_ylabel(r"mean fold $R^2$")
    ax.set_title("Mean vs. median across folds")
    ax.text(0.04, 0.05, "points below the diagonal:\nthe mean is dragged down by\na small number of tiny folds",
            transform=ax.transAxes, fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PALE, edgecolor="none"))
    tidy(ax, grid_axis="both")
    save(fig, "fig17_mean_vs_median")


# ---------------------------------------------------------------- diagrams
def fig18_split_diagram():
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.axis("off")
    ax.add_patch(Rectangle((0.02, 0.35), 0.17, 0.3, facecolor=BURGUNDY, edgecolor="none"))
    ax.text(0.105, 0.5, "generation\nrule $g$", ha="center", va="center", color="white", fontsize=8)
    ctx_y = [0.78, 0.55, 0.32, 0.09]
    for i, y in enumerate(ctx_y):
        ax.add_patch(Rectangle((0.32, y), 0.15, 0.14, facecolor=PALE, edgecolor=BURGUNDY, lw=0.8))
        ax.text(0.395, y + 0.07, f"context {i+1}", ha="center", va="center", fontsize=7, color=INK)
        ax.add_patch(FancyArrowPatch((0.19, 0.5), (0.32, y + 0.07), arrowstyle="->",
                                     mutation_scale=8, color=BURG_MID, lw=0.8))
    dest = [("train", GREEN, 0.78), ("train", GREEN, 0.55), ("validation", GREEN_MID, 0.32), ("test", BURGUNDY, 0.09)]
    for (lbl, col, y) in dest:
        ax.add_patch(Rectangle((0.63, y), 0.16, 0.14, facecolor=col, edgecolor="none"))
        ax.text(0.71, y + 0.07, lbl, ha="center", va="center", color="white", fontsize=7.5)
        ax.add_patch(FancyArrowPatch((0.47, y + 0.07), (0.63, y + 0.07), arrowstyle="->",
                                     mutation_scale=8, color=GREY, lw=0.8))
    ax.annotate("the test rows come from a rule\nthe model has already seen",
                xy=(0.79, 0.16), xytext=(0.84, 0.45), fontsize=7, color=BURGUNDY, ha="center",
                arrowprops=dict(arrowstyle="->", color=BURGUNDY, lw=0.8))
    ax.set_title("Original protocol: grouping by context_id, not by generation rule", fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "fig18_context_split_diagram")


def fig19_loro_diagram():
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ax.axis("off")
    rules = ["rule 1", "rule 2", "rule 3", "rule 4", "rule 5"]
    for fold in range(3):
        y = 0.72 - fold * 0.28
        ax.text(0.015, y + 0.06, f"fold {fold+1}", fontsize=7.5, color=INK, va="center")
        for i, r in enumerate(rules):
            held = (i == fold)
            col = BURGUNDY if held else GREEN
            ax.add_patch(Rectangle((0.13 + i * 0.165, y), 0.15, 0.12,
                                   facecolor=col, edgecolor="none"))
            ax.text(0.205 + i * 0.165, y + 0.06, r, ha="center", va="center",
                    color="white", fontsize=7)
    ax.add_patch(Rectangle((0.13, 0.03), 0.04, 0.05, facecolor=GREEN, edgecolor="none"))
    ax.text(0.185, 0.055, "trained on", fontsize=7, va="center")
    ax.add_patch(Rectangle((0.33, 0.03), 0.04, 0.05, facecolor=BURGUNDY, edgecolor="none"))
    ax.text(0.385, 0.055, "held out and predicted", fontsize=7, va="center")
    ax.set_title("Revised protocol: leave-one-generation-rule-out cross-validation",
                 fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.92)
    save(fig, "fig19_loro_diagram")


def main():
    fig01_production_v7()
    fig02_v6_vs_v7()
    fig03_train_test_gap()
    spec = load_specialists()
    fig04_dataset_composition(spec)
    fig05_rule_counts(spec)
    fig06_rule_size_distribution(spec)
    fig07_rule_concentration(spec)
    fig08_complexity_classes()
    fig09_validation_removal()
    fig10_kfold()
    fig11_granularity()
    fig12_logistic()
    zoo = load_zoo()
    fig13_zoo_ranking(zoo)
    fig14_holdout_vs_loro(zoo)
    fig15_loro_heatmap(zoo, "mean_r2", "fig15_loro_heatmap_mean",
                       r"Leave-rule-out mean $R^2$ by model and specialist")
    fig15_loro_heatmap(zoo, "median_r2", "fig16_loro_heatmap_median",
                       r"Leave-rule-out median $R^2$ by model and specialist")
    fig17_mean_vs_median(zoo)
    fig18_split_diagram()
    fig19_loro_diagram()


if __name__ == "__main__":
    main()
