#!/usr/bin/env python
"""
Figure + table generator (part B) for the overfitting-investigation meeting
report: everything that needs the row-level out-of-fold predictions
produced by leave_rule_out_row_level.py.

Reads:
  reports/overfitting_investigation_meeting/rf_fold_predictions.csv
  reports/overfitting_investigation_meeting/fold_diagnostic_summary.csv
  data/raw/CHEESE_SHELF_LIFE_V7_*_SPECIALIST_CORRECTED.csv

Writes figures/*.pdf and tables/*.tex.

Usage: .venv/Scripts/python.exe scripts/experiments/make_report_figures_b.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
BASE = ROOT / "reports" / "overfitting_investigation_meeting"
OUT = BASE / "figures"
TABLES = BASE / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

BURGUNDY = "#7A1B2E"; BURG_MID = "#A84C5B"; BURG_LIGHT = "#C9939B"
PALE = "#F5E6E8"; GREEN = "#1E4D3B"; GREEN_MID = "#4C7A66"; GREEN_LIGHT = "#A8C4B7"
INK = "#1F1B1A"; GREY = "#8A8583"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.edgecolor": INK, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

MODEL_COLOR = {"random_forest": BURGUNDY, "lightgbm": GREEN, "xgboost": BURG_MID}
SPEC_ORDER = [
    ("soft", "general_shelf_life"), ("semi_hard", "general_shelf_life"), ("hard", "general_shelf_life"),
    ("soft", "safety_endpoint"), ("semi_hard", "safety_endpoint"), ("hard", "safety_endpoint"),
]
SPEC_LABEL = {k: f"{k[0].replace('_','-')}\n{k[1].split('_')[0]}" for k in SPEC_ORDER}


def tidy(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#DDD8D6", linewidth=0.5, zorder=0); ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); plt.close(fig); print("wrote", name)


def esc(s):
    return (str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
            .replace("|", r"\textbar{}").replace("#", r"\#"))


def short(rule):
    return rule.replace("V7|", "")


def rule_tex(rule):
    body = (short(str(rule))
            .replace("_", r"\_\allowbreak{}")
            .replace("|", r"\textbar\allowbreak{}"))
    return r"{\scriptsize\ttfamily\raggedright " + body + r"\par}"


preds = pd.read_csv(BASE / "rf_fold_predictions.csv")
folds = pd.read_csv(BASE / "fold_diagnostic_summary.csv")
rf_folds = folds[folds.model == "random_forest"].copy()
rf_preds = preds[preds.model == "random_forest"].copy()


# ---------------------------------------------------------------- fold overview
def fig20_fold_r2_by_rule():
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.2))
    for ax, (c, t) in zip(axes.ravel(), SPEC_ORDER):
        d = rf_folds[(rf_folds.category == c) & (rf_folds.task == t)].sort_values("r2")
        cols = [BURGUNDY if v < 0 else (BURG_MID if v < 0.5 else GREEN) for v in d.r2]
        ax.barh(np.arange(len(d)), d.r2.clip(lower=-2), color=cols, height=0.7, zorder=3)
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels([short(r)[:24] for r in d.held_out_rule], fontsize=5.0)
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_title(f"{c.replace('_','-')} / {t.split('_')[0]}", fontsize=8)
        ax.tick_params(axis="x", labelsize=6)
        tidy(ax, grid_axis="x")
    fig.suptitle(r"Random Forest: leave-rule-out $R^2$ for every held-out rule (clipped at $-2$)",
                 fontsize=9.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig20_rf_fold_r2_by_rule")


def fig21_fold_size_vs_r2():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for mdl in ["random_forest", "lightgbm", "xgboost"]:
        d = folds[folds.model == mdl]
        ax.scatter(d.n_test_rows, d.r2.clip(lower=-4), s=24, alpha=0.75,
                   color=MODEL_COLOR[mdl], label=mdl.replace("_", " "),
                   edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_xscale("log")
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xlabel("rows in the held-out rule (log scale)")
    ax.set_ylabel(r"fold $R^2$ (clipped at $-4$)")
    ax.set_title("Small held-out rules produce the unstable scores")
    ax.legend(ncol=3, loc="lower right")
    tidy(ax, grid_axis="both")
    save(fig, "fig21_fold_size_vs_r2")


def fig22_mae_vs_r2():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    d = rf_folds
    sc = ax.scatter(d.mae, d.r2.clip(lower=-4), s=np.clip(d.n_test_rows / 8, 8, 160),
                    c=d.y_test_std, cmap="viridis", alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(0, color=INK, lw=0.7)
    for _, r in d.nsmallest(4, "r2").iterrows():
        ax.annotate(short(r.held_out_rule)[:22], (r.mae, max(r.r2, -4)), fontsize=5.5,
                    xytext=(4, 4), textcoords="offset points", color=BURGUNDY)
    ax.set_xlabel("fold MAE (days)")
    ax.set_ylabel(r"fold $R^2$")
    ax.set_title(r"A small error in days does not guarantee a high $R^2$")
    fig.colorbar(sc, ax=ax, shrink=0.8, label=r"s.d. of held-out target (days)")
    tidy(ax, grid_axis="both")
    save(fig, "fig22_mae_vs_r2")


def fig23_r2_variance_decomposition():
    d = rf_folds.copy()
    d["ss_tot_per_row"] = d.y_test_std ** 2
    d["ss_res_per_row"] = d.rmse ** 2
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    order = d.sort_values("r2").head(16)
    y = np.arange(len(order))
    ax.barh(y - 0.2, order.ss_tot_per_row, 0.4, label=r"target variance $s_y^2$ (the $R^2$ denominator)",
            color=GREEN_LIGHT, zorder=3)
    ax.barh(y + 0.2, order.ss_res_per_row, 0.4, label=r"mean squared error (the numerator)",
            color=BURGUNDY, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{short(r.held_out_rule)[:26]} (n={int(r.n_test_rows)})"
                        for _, r in order.iterrows()], fontsize=5.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"days$^2$ (log scale)")
    ax.set_title(r"Why $R^2$ goes negative: error variance exceeds target variance")
    ax.legend(loc="lower right", fontsize=6.5)
    tidy(ax, grid_axis="x")
    save(fig, "fig23_r2_variance_decomposition")


# ---------------------------------------------------------------- predictions
def fig24_actual_vs_pred(category, task, name, title):
    d = rf_preds[(rf_preds.category == category) & (rf_preds.task == task)]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    rules = d.held_out_rule.value_counts().index.tolist()
    palette = [BURGUNDY, GREEN, BURG_MID, GREEN_MID, BURG_LIGHT, GREEN_LIGHT, "#5A3E52",
               "#7D6B4F", "#3F5E72", "#8C5A3B", "#4A4458", "#6B7A3F"]
    for i, r in enumerate(rules):
        sub = d[d.held_out_rule == r]
        ax.scatter(sub.actual_shelf_life_days, sub.predicted_shelf_life_days, s=7, alpha=0.5,
                   color=palette[i % len(palette)], label=f"{short(r)[:22]} (n={len(sub)})",
                   edgecolor="none", zorder=3)
    lim = [0, max(d.actual_shelf_life_days.max(), d.predicted_shelf_life_days.max()) * 1.03]
    ax.plot(lim, lim, ls="--", color=INK, lw=0.8, zorder=4)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("actual shelf life (days)"); ax.set_ylabel("predicted shelf life (days)")
    ax.set_title(title, fontsize=8.5)
    ax.legend(fontsize=4.8, loc="upper left", markerscale=1.6)
    tidy(ax, grid_axis="both")
    save(fig, name)


def fig26_residual_hist():
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0))
    for ax, (c, t) in zip(axes.ravel(), SPEC_ORDER):
        d = rf_preds[(rf_preds.category == c) & (rf_preds.task == t)]
        ax.hist(d.signed_error_days, bins=50, color=BURG_MID, zorder=3)
        ax.axvline(0, color=INK, lw=0.7)
        ax.axvline(d.signed_error_days.mean(), color=GREEN, lw=1.0, ls="--")
        ax.set_title(f"{c.replace('_','-')} / {t.split('_')[0]}\n"
                     f"mean {d.signed_error_days.mean():+.1f} d, MAE {d.abs_error_days.mean():.1f} d",
                     fontsize=7)
        ax.tick_params(labelsize=6)
        tidy(ax)
    fig.suptitle("Random Forest out-of-fold residual distributions (prediction $-$ actual, days)",
                 fontsize=9.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig26_residual_histograms")


def fig27_residual_vs_actual():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for ax, (c, t) in zip(axes, [s for s in SPEC_ORDER if s[1] == "general_shelf_life"]):
        d = rf_preds[(rf_preds.category == c) & (rf_preds.task == t)]
        ax.scatter(d.actual_shelf_life_days, d.signed_error_days, s=3, alpha=0.25,
                   color=BURGUNDY, edgecolor="none", zorder=3)
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_title(f"{c.replace('_','-')} / general", fontsize=8)
        ax.set_xlabel("actual (days)", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("signed error (days)", fontsize=7)
        ax.tick_params(labelsize=6)
        tidy(ax, grid_axis="both")
    fig.suptitle("Residual structure against the true value", fontsize=9.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "fig27_residual_vs_actual")


def fig28_error_by_product():
    d = rf_preds.dropna(subset=["base_cheese_name"])
    g = d.groupby("base_cheese_name").agg(mae=("abs_error_days", "mean"),
                                          n=("abs_error_days", "size"),
                                          bias=("signed_error_days", "mean"))
    g = g[g.n >= 40].sort_values("mae", ascending=False).head(22)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    y = np.arange(len(g))
    ax.barh(y, g.mae, color=[BURGUNDY if b < 0 else GREEN for b in g.bias], height=0.72, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i[:30]} (n={int(n)})" for i, n in zip(g.index, g.n)], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("mean absolute out-of-fold error (days)")
    ax.set_title("Where the error lives: MAE by cheese product (Random Forest)")
    ax.text(0.98, 0.04, "burgundy = under-predicted on average\ngreen = over-predicted on average",
            transform=ax.transAxes, ha="right", fontsize=6,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=PALE, edgecolor="none"))
    tidy(ax, grid_axis="x")
    save(fig, "fig28_error_by_product")


def fig29_error_by_ingredient():
    d = rf_preds.copy()
    d["ing"] = d.primary_ingredient_name.fillna("(none / control)")
    g = d.groupby("ing").agg(mae=("abs_error_days", "mean"), n=("abs_error_days", "size"),
                             bias=("signed_error_days", "mean"))
    g = g[g.n >= 40].sort_values("mae", ascending=False).head(22)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    y = np.arange(len(g))
    ax.barh(y, g.mae, color=[BURGUNDY if b < 0 else GREEN for b in g.bias], height=0.72, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([f"{i[:30]} (n={int(n)})" for i, n in zip(g.index, g.n)], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("mean absolute out-of-fold error (days)")
    ax.set_title("MAE by preservative / treatment ingredient (Random Forest)")
    tidy(ax, grid_axis="x")
    save(fig, "fig29_error_by_ingredient")


def fig30_signed_error_by_rule():
    d = rf_folds.sort_values("mean_signed_error")
    fig, ax = plt.subplots(figsize=(6.4, 8.6))
    y = np.arange(len(d))
    ax.barh(y, d.mean_signed_error, color=[BURGUNDY if v < 0 else GREEN for v in d.mean_signed_error],
            height=0.72, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{short(r.held_out_rule)[:34]}  [{r.category[:4]}/{r.task[:3]}, n={int(r.n_test_rows)}]"
                        for _, r in d.iterrows()], fontsize=5.8)
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_xlabel("mean signed error (days); negative = under-prediction")
    ax.set_title("Directional bias on each held-out generation rule")
    tidy(ax, grid_axis="x")
    save(fig, "fig30_signed_error_by_rule")


# ---------------------------------------------------------------- worst folds
WORST = [
    ("hard", "general_shelf_life", "V7|CLOSTRIDIA|CLOSTRIDIA_GRANA|HARD_SEMI_REVIEW"),
    ("hard", "general_shelf_life", "V7|HARD_SEMI_REVIEW|SHRED_NAT"),
    ("soft", "general_shelf_life", "V7|HMMC_NAT|NATAMYCIN_REVIEW"),
    ("soft", "general_shelf_life", "V7|RICOTTA_MAP|NATAMYCIN_REVIEW"),
]


def fig31_worst_folds_scatter():
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
    for ax, (c, t, rule) in zip(axes, WORST):
        d = rf_preds[(rf_preds.category == c) & (rf_preds.task == t) & (rf_preds.held_out_rule == rule)]
        row = rf_folds[(rf_folds.category == c) & (rf_folds.task == t) &
                       (rf_folds.held_out_rule == rule)].iloc[0]
        ax.scatter(d.actual_shelf_life_days, d.predicted_shelf_life_days, s=12, alpha=0.7,
                   color=BURGUNDY, edgecolor="white", linewidth=0.3, zorder=3)
        lim = [min(d.actual_shelf_life_days.min(), d.predicted_shelf_life_days.min()) * 0.9,
               max(d.actual_shelf_life_days.max(), d.predicted_shelf_life_days.max()) * 1.05]
        ax.plot(lim, lim, ls="--", color=INK, lw=0.7)
        ax.set_title(f"{short(rule)[:20]}\n$R^2$={row.r2:.2f}, MAE={row.mae:.1f}d, n={int(row.n_test_rows)}",
                     fontsize=6.2)
        ax.tick_params(labelsize=5.5)
        ax.set_xlabel("actual (d)", fontsize=6)
        if ax is axes[0]:
            ax.set_ylabel("predicted (d)", fontsize=6)
        tidy(ax, grid_axis="both")
    fig.suptitle("The four weakest Random Forest folds", fontsize=9.5, fontweight="bold", y=1.10)
    fig.tight_layout()
    save(fig, "fig31_worst_folds_scatter")


def fig32_target_distributions():
    spec_frames = {}
    for cat in ["soft", "semi_hard", "hard"]:
        spec_frames[cat] = pd.read_csv(
            ROOT / "data" / "raw" / f"CHEESE_SHELF_LIFE_V7_{cat.upper()}_SPECIALIST_CORRECTED.csv")
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
    for ax, (c, t, rule) in zip(axes, WORST):
        df = spec_frames[c]
        sub = df[df.model_task == t]
        held = sub[sub.source_rule_id == rule].shelf_life_days
        train = sub[sub.source_rule_id != rule].shelf_life_days
        bins = np.linspace(0, max(train.max(), held.max()), 40)
        ax.hist(train, bins=bins, color=GREEN_LIGHT, label="training rules", density=True, zorder=3)
        ax.hist(held, bins=bins, color=BURGUNDY, alpha=0.85, label="held-out rule", density=True, zorder=4)
        ax.set_title(short(rule)[:20], fontsize=6.2)
        ax.tick_params(labelsize=5.5)
        ax.set_xlabel("shelf life (days)", fontsize=6)
        if ax is axes[0]:
            ax.set_ylabel("density", fontsize=6); ax.legend(fontsize=5)
        tidy(ax)
    fig.suptitle("Target distribution: training rules vs. the held-out rule",
                 fontsize=9.5, fontweight="bold", y=1.10)
    fig.tight_layout()
    save(fig, "fig32_target_distributions")


def fig33_feature_distributions():
    spec_frames = {c: pd.read_csv(ROOT / "data" / "raw" /
                                  f"CHEESE_SHELF_LIFE_V7_{c.upper()}_SPECIALIST_CORRECTED.csv")
                   for c in ["soft", "semi_hard", "hard"]}
    feats = ["storage_temperature_c", "matrix_ph", "matrix_water_activity", "primary_concentration"]
    c, t, rule = WORST[0]
    df = spec_frames[c]; sub = df[df.model_task == t]
    held = sub[sub.source_rule_id == rule]; train = sub[sub.source_rule_id != rule]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
    for ax, f in zip(axes, feats):
        if f not in sub.columns:
            continue
        a = train[f].dropna(); b = held[f].dropna()
        if len(b) == 0:
            ax.text(0.5, 0.5, "not populated\nfor this rule", ha="center", va="center", fontsize=6)
            ax.set_axis_off(); continue
        bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 25)
        ax.hist(a, bins=bins, color=GREEN_LIGHT, density=True, zorder=3, label="training rules")
        ax.hist(b, bins=bins, color=BURGUNDY, density=True, alpha=0.85, zorder=4, label="held out")
        ax.set_title(f.replace("_", " "), fontsize=6.2)
        ax.tick_params(labelsize=5.5)
        if ax is axes[0]:
            ax.legend(fontsize=5)
        tidy(ax)
    fig.suptitle(f"Feature coverage for the worst fold ({short(WORST[0][2])[:34]})",
                 fontsize=9, fontweight="bold", y=1.10)
    fig.tight_layout()
    save(fig, "fig33_feature_distributions_worst")


# ---------------------------------------------------------------- cross-model
def fig34_model_agreement():
    piv = folds.pivot_table(index=["category", "task", "held_out_rule"], columns="model", values="r2")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    pairs = [("random_forest", "lightgbm"), ("random_forest", "xgboost"), ("lightgbm", "xgboost")]
    for ax, (a, b) in zip(axes, pairs):
        ax.scatter(piv[a].clip(lower=-3), piv[b].clip(lower=-3), s=18, color=BURGUNDY,
                   alpha=0.75, edgecolor="white", linewidth=0.3, zorder=3)
        ax.plot([-3, 1], [-3, 1], ls="--", color=GREY, lw=0.8)
        ax.axhline(0, color=INK, lw=0.5); ax.axvline(0, color=INK, lw=0.5)
        ax.set_xlabel(a.replace("_", " "), fontsize=7); ax.set_ylabel(b.replace("_", " "), fontsize=7)
        r = np.corrcoef(piv[a].clip(lower=-3), piv[b].clip(lower=-3))[0, 1]
        ax.set_title(f"$r$ = {r:.3f}", fontsize=7.5)
        ax.tick_params(labelsize=6)
        tidy(ax, grid_axis="both")
    fig.suptitle(r"Do the three ensembles fail on the same rules? Per-fold $R^2$, clipped at $-3$",
                 fontsize=9, fontweight="bold", y=1.06)
    fig.tight_layout()
    save(fig, "fig34_model_agreement")


def fig35_rule_model_heatmap(value="r2", name="fig35_rule_model_heatmap_r2",
                             title=r"Per-rule $R^2$ by model", clip=(-2, 1), cmap="RdYlGn"):
    f = folds.copy()
    f["key"] = f.category.str[:4] + "/" + f.task.str[:3] + " " + f.held_out_rule.map(short)
    piv = f.pivot_table(index="key", columns="model", values=value)
    piv = piv.loc[piv.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(4.6, 8.0))
    data = np.clip(piv.values, clip[0], clip[1])
    im = ax.imshow(data, cmap=cmap, vmin=clip[0], vmax=clip[1], aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([c.replace("_", " ") for c in piv.columns], fontsize=6.5, rotation=20, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([k[:42] for k in piv.index], fontsize=4.8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=4.4,
                    color="white" if data[i, j] < clip[0] + 0.4 * (clip[1] - clip[0]) else INK)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.35)
    save(fig, name)


def fig37_error_ecdf():
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for mdl in ["random_forest", "lightgbm", "xgboost"]:
        e = np.sort(preds[preds.model == mdl].abs_error_days.values)
        ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=MODEL_COLOR[mdl], lw=1.4,
                label=f"{mdl.replace('_',' ')} (median {np.median(e):.2f} d)")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("absolute out-of-fold error (days, symlog scale)")
    ax.set_ylabel("cumulative share of predictions")
    ax.set_title("Out-of-fold error distribution, all 34\\,000 rows per model")
    ax.legend(loc="lower right")
    tidy(ax, grid_axis="both")
    save(fig, "fig37_error_ecdf")


def fig38_pooled_oof():
    rows = []
    for mdl in ["random_forest", "lightgbm", "xgboost"]:
        for c, t in SPEC_ORDER:
            d = preds[(preds.model == mdl) & (preds.category == c) & (preds.task == t)]
            y, p = d.actual_shelf_life_days.values, d.predicted_shelf_life_days.values
            ss_res = np.sum((y - p) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
            rows.append({"model": mdl, "category": c, "task": t,
                         "pooled_r2": 1 - ss_res / ss_tot,
                         "mae": np.mean(np.abs(y - p)),
                         "rmse": np.sqrt(np.mean((y - p) ** 2))})
    pooled = pd.DataFrame(rows)
    pooled.to_csv(BASE / "pooled_oof_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(SPEC_ORDER))
    for i, mdl in enumerate(["random_forest", "lightgbm", "xgboost"]):
        vals = [pooled[(pooled.model == mdl) & (pooled.category == c) & (pooled.task == t)].pooled_r2.iloc[0]
                for c, t in SPEC_ORDER]
        ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=mdl.replace("_", " "), color=MODEL_COLOR[mdl], zorder=3)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([SPEC_LABEL[s] for s in SPEC_ORDER], fontsize=7)
    ax.set_ylabel(r"pooled out-of-fold $R^2$")
    ax.set_title(r"Pooled out-of-fold $R^2$: all folds concatenated, then scored once")
    ax.legend(ncol=3, fontsize=7)
    tidy(ax)
    save(fig, "fig38_pooled_oof_r2")

    # table
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Specialist & Model & pooled OOF $R^2$ & MAE (d) & RMSE (d) \\", r"\midrule"]
    for c, t in SPEC_ORDER:
        for i, mdl in enumerate(["random_forest", "lightgbm", "xgboost"]):
            r = pooled[(pooled.model == mdl) & (pooled.category == c) & (pooled.task == t)].iloc[0]
            spec = esc(f"{c.replace('_','-')}/{t.replace('_',' ')}") if i == 0 else ""
            lines.append(f"{spec} & {esc(mdl)} & \\textbf{{{r.pooled_r2:.3f}}} & {r.mae:.2f} & {r.rmse:.2f} \\\\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "tab_pooled_oof.tex").write_text("\n".join(lines), encoding="utf-8")
    print("wrote tab_pooled_oof.tex")


# ---------------------------------------------------------------- tables
def tab_all_folds():
    for mdl in ["random_forest", "lightgbm", "xgboost"]:
        d = folds[folds.model == mdl]
        lines = [r"\begin{longtable}{p{4.1cm}rrrrrr}", r"\toprule",
                 r"Held-out rule & $n_{\text{test}}$ & $n_{\text{train}}$ & $R^2$ & MAE (d) & RMSE (d) & bias (d) \\",
                 r"\midrule", r"\endhead"]
        for c, t in SPEC_ORDER:
            sub = d[(d.category == c) & (d.task == t)].sort_values("r2", ascending=False)
            lines.append(r"\multicolumn{7}{l}{\textbf{" + esc(f"{c.replace('_','-')} / {t.replace('_',' ')}") + r"}} \\[1pt]")
            for _, r in sub.iterrows():
                bold = r"\textbf" if r.r2 < 0 else ""
                lines.append(
                    f"{rule_tex(r.held_out_rule)} & {int(r.n_test_rows):,} & "
                    f"{int(r.n_train_rows):,} & {bold}{{{r.r2:.3f}}} & {r.mae:.2f} & {r.rmse:.2f} & "
                    f"{r.mean_signed_error:+.2f} \\\\")
            lines.append(r"\addlinespace[3pt]")
        lines += [r"\bottomrule", r"\end{longtable}"]
        (TABLES / f"tab_folds_{mdl}.tex").write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote tab_folds_{mdl}.tex")


def tab_worst_fold_detail():
    spec_frames = {c: pd.read_csv(ROOT / "data" / "raw" /
                                  f"CHEESE_SHELF_LIFE_V7_{c.upper()}_SPECIALIST_CORRECTED.csv")
                   for c in ["soft", "semi_hard", "hard"]}
    lines = [r"\begin{tabular}{p{3.5cm}p{2.3cm}p{2.3cm}p{2.3cm}p{2.3cm}}", r"\toprule",
             r"& " + " & ".join(f"{{\\scriptsize \\texttt{{{esc(short(r)[:18])}}}}}" for _, _, r in WORST) + r" \\",
             r"\midrule"]
    fields = []
    for c, t, rule in WORST:
        df = spec_frames[c]; sub = df[df.model_task == t]
        held = sub[sub.source_rule_id == rule]
        train = sub[sub.source_rule_id != rule]
        row = rf_folds[(rf_folds.category == c) & (rf_folds.task == t) &
                       (rf_folds.held_out_rule == rule)].iloc[0]
        prods = sorted(set(str(x) for x in held.base_cheese_name.dropna().unique()))
        ings = sorted(set(str(x) for x in held.primary_ingredient_name.dropna().unique()))
        train_prods = set(str(x) for x in train.base_cheese_name.dropna().unique())
        train_ings = set(str(x) for x in train.primary_ingredient_name.dropna().unique())
        combo_in_train = 0
        for p in prods:
            for ing in ings:
                combo_in_train += len(train[(train.base_cheese_name == p) &
                                            (train.primary_ingredient_name == ing)])
        fields.append({
            "specialist": f"{c.replace('_','-')}/{t.split('_')[0]}",
            "rows": f"{len(held):,}",
            "contexts": f"{held.context_id.nunique()}",
            "products": "; ".join(p[:22] for p in prods[:3]) + (" ..." if len(prods) > 3 else ""),
            "ingredients": "; ".join(i[:20] for i in ings[:3]) + (" ..." if len(ings) > 3 else ""),
            "temp": f"{held.storage_temperature_c.min():.0f}--{held.storage_temperature_c.max():.0f}",
            "y_range": f"{held.shelf_life_days.min():.1f}--{held.shelf_life_days.max():.1f}",
            "y_mean": f"{held.shelf_life_days.mean():.1f}",
            "y_sd": f"{held.shelf_life_days.std():.2f}",
            "pred_mean": f"{row.pred_mean:.1f}",
            "r2": f"{row.r2:.3f}",
            "mae": f"{row.mae:.2f}",
            "bias": f"{row.mean_signed_error:+.2f}",
            "prod_seen": "yes" if any(p in train_prods for p in prods) else "no",
            "ing_seen": "yes" if any(i in train_ings for i in ings) else "no",
            "combo_rows": f"{combo_in_train:,}",
        })
    labels = [
        ("specialist", "Specialist"), ("rows", "Held-out rows"), ("contexts", "Contexts"),
        ("products", "Cheese products"), ("ingredients", "Ingredients"),
        ("temp", r"Storage temp ($^\circ$C)"), ("y_range", "Actual range (d)"),
        ("y_mean", r"Actual mean $\bar{y}$ (d)"), ("y_sd", r"Actual s.d. $s_y$ (d)"),
        ("pred_mean", "Predicted mean (d)"), ("r2", r"Fold $R^2$"), ("mae", "Fold MAE (d)"),
        ("bias", "Mean signed error (d)"),
        ("prod_seen", "Product seen in training?"), ("ing_seen", "Ingredient seen in training?"),
        ("combo_rows", "Product+ingredient rows in training"),
    ]
    for key, label in labels:
        vals = " & ".join(f"{{\\scriptsize {esc(f[key])}}}" for f in fields)
        lines.append(f"{{\\footnotesize {label}}} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "tab_worst_fold_detail.tex").write_text("\n".join(lines), encoding="utf-8")
    print("wrote tab_worst_fold_detail.tex")


def tab_largest_errors():
    lines = [r"\begin{tabular}{p{2.5cm}p{2.3cm}rrrrr}", r"\toprule",
             r"Held-out rule & Cheese & $T$ ($^\circ$C) & actual (d) & pred. (d) & error (d) \\", r"\midrule"]
    for c, t, rule in WORST[:2]:
        d = rf_preds[(rf_preds.category == c) & (rf_preds.task == t) &
                     (rf_preds.held_out_rule == rule)].nlargest(6, "abs_error_days")
        lines.append(r"\multicolumn{6}{l}{\scriptsize\bfseries\ttfamily " + esc(short(rule)) + r"} \\[1pt]")
        for _, r in d.iterrows():
            lines.append(f"& {{\\scriptsize {esc(str(r.base_cheese_name)[:22])}}} & "
                         f"{r.storage_temperature_c:.0f} & {r.actual_shelf_life_days:.1f} & "
                         f"{r.predicted_shelf_life_days:.1f} & {r.signed_error_days:+.1f} \\\\")
        lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "tab_largest_errors.tex").write_text("\n".join(lines), encoding="utf-8")
    print("wrote tab_largest_errors.tex")


def main():
    fig20_fold_r2_by_rule()
    fig21_fold_size_vs_r2()
    fig22_mae_vs_r2()
    fig23_r2_variance_decomposition()
    fig24_actual_vs_pred("soft", "general_shelf_life", "fig24_avp_soft_general",
                         "Random Forest out-of-fold predictions: soft / general")
    fig24_actual_vs_pred("hard", "general_shelf_life", "fig25_avp_hard_general",
                         "Random Forest out-of-fold predictions: hard / general")
    fig26_residual_hist()
    fig27_residual_vs_actual()
    fig28_error_by_product()
    fig29_error_by_ingredient()
    fig30_signed_error_by_rule()
    fig31_worst_folds_scatter()
    fig32_target_distributions()
    fig33_feature_distributions()
    fig34_model_agreement()
    fig35_rule_model_heatmap()
    fig35_rule_model_heatmap("mae", "fig36_rule_model_heatmap_mae",
                             "Per-rule MAE (days) by model", (0, 40), "RdYlGn_r")
    fig37_error_ecdf()
    fig38_pooled_oof()
    tab_all_folds()
    tab_worst_fold_detail()
    tab_largest_errors()


if __name__ == "__main__":
    main()
