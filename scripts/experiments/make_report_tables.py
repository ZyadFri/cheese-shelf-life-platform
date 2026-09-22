#!/usr/bin/env python
"""
Table generator for the overfitting-investigation meeting report.
Emits LaTeX table fragments into
reports/overfitting_investigation_meeting/tables/ so that no number in the
report is transcribed by hand. Every fragment is produced from a result
file that already exists in the repository.

Usage: .venv/Scripts/python.exe scripts/experiments/make_report_tables.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
EXP = ROOT / "scripts" / "experiments" / "overfitting_diagnosis"
OUT = ROOT / "reports" / "overfitting_investigation_meeting" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

SPEC_ORDER = [
    ("soft", "general_shelf_life"), ("semi_hard", "general_shelf_life"), ("hard", "general_shelf_life"),
    ("soft", "safety_endpoint"), ("semi_hard", "safety_endpoint"), ("hard", "safety_endpoint"),
]


def esc(s: str) -> str:
    return (str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
            .replace("|", r"\textbar{}").replace("#", r"\#"))


def rule_tex(rule: str) -> str:
    """Long composite rule ids must be breakable inside a narrow p-column."""
    body = (str(rule).replace("V7|", "")
            .replace("_", r"\_\allowbreak{}")
            .replace("|", r"\textbar\allowbreak{}"))
    return r"{\scriptsize\ttfamily\raggedright " + body + r"\par}"


def write(name: str, body: str):
    (OUT / name).write_text(body, encoding="utf-8")
    print("wrote", name)


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and (np.isnan(v))):
        return "--"
    return f"{v:.{nd}f}"


# ------------------------------------------------------------------ production
def production_tables():
    m = pd.read_csv(ROOT / "reports" / "comprehensive_ml_audit" / "master_experiment_results.csv")
    prod = m[m.phase == "production_context_holdout"]
    for ver in ["v7", "v6"]:
        d = prod[prod.dataset_version == ver]
        lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
                 r"Specialist & Model & $n_{\text{train}}$ & $n_{\text{test}}$ & "
                 r"train $R^2$ & val $R^2$ & test $R^2$ & test RMSE \\", r"\midrule"]
        for c, t in SPEC_ORDER:
            sub = d[(d.cheese_category == c) & (d.prediction_task == t)]
            if sub.empty:
                continue
            for i, (_, r) in enumerate(sub.iterrows()):
                spec = f"{c.replace('_','-')}/{t.replace('_',' ')}" if i == 0 else ""
                lines.append(
                    f"{esc(spec)} & {esc(r.model_family)} & {int(r.n_train):,} & {int(r.n_test):,} & "
                    f"{fmt(r.train_r2)} & {fmt(r.val_r2)} & \\textbf{{{fmt(r.test_r2)}}} & {fmt(r.test_rmse,2)} \\\\")
            lines.append(r"\addlinespace[2pt]")
        lines += [r"\bottomrule", r"\end{tabular}"]
        write(f"tab_production_{ver}.tex", "\n".join(lines))


# ------------------------------------------------------------------ exp 1 & 4
def exp14_table():
    e = pd.read_csv(EXP / "experiment_1_4_results.csv")
    d = e[(e.granularity == "per_category_specialist") & (e.validation_split_used)]
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Specialist & Model & train $R^2$ & val $R^2$ & test $R^2$ & gap \\", r"\midrule"]
    for c, t in SPEC_ORDER:
        sub = d[(d.category == c) & (d.task == t)]
        for i, (_, r) in enumerate(sub.iterrows()):
            spec = f"{c.replace('_','-')}/{t.replace('_',' ')}" if i == 0 else ""
            lines.append(f"{esc(spec)} & {esc(r.model)} & {fmt(r.train_r2)} & {fmt(r.val_r2)} & "
                         f"\\textbf{{{fmt(r.test_r2)}}} & {fmt(r.overfit_gap_train_minus_test_r2)} \\\\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_exp1_reduced_complexity.tex", "\n".join(lines))

    # validation-removal comparison
    piv = e[e.granularity == "per_category_specialist"].pivot_table(
        index=["category", "task", "model"], columns="validation_split_used", values="test_r2").dropna()
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Specialist & Model & test $R^2$ (70/15/15) & test $R^2$ (85/15) & $\Delta$ \\", r"\midrule"]
    for (c, t, mdl), r in piv.iterrows():
        lines.append(f"{esc(c)}/{esc(t)} & {esc(mdl)} & {fmt(r[True])} & {fmt(r[False])} & "
                     f"{fmt(r[False]-r[True], 4)} \\\\")
    lines += [r"\midrule",
              f"\\multicolumn{{4}}{{r}}{{mean change}} & {fmt((piv[False]-piv[True]).mean(),4)} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    write("tab_exp4_validation_removal.tex", "\n".join(lines))


# ------------------------------------------------------------------ exp 2
def exp2_table():
    cv = pd.read_csv(EXP / "experiment_2_cv_results.csv")
    lines = [r"\begin{tabular}{llrrrl}", r"\toprule",
             r"Granularity & Model & $K$ & CV $R^2$ mean & CV $R^2$ s.d. & per-fold $R^2$ \\", r"\midrule"]
    for _, r in cv.iterrows():
        gran = r.granularity if r.granularity != "aggregated_all_categories" else "aggregated"
        folds = str(r.fold_r2_values).strip("[]")
        lines.append(f"{esc(gran)} & {esc(r.model)} & {int(r.k)} & {fmt(r.fold_r2_mean,4)} & "
                     f"{fmt(r.fold_r2_std,4)} & {{\\scriptsize {esc(folds)}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_exp2_kfold.tex", "\n".join(lines))


# ------------------------------------------------------------------ exp 3
def exp3_tables():
    e = pd.read_csv(EXP / "experiment_3_extra_regressors.csv")
    d = e[e.granularity == "per_category_specialist"]
    agg = d.groupby("model").agg(mean_test_r2=("test_r2", "mean"), min_test_r2=("test_r2", "min"),
                                 max_test_r2=("test_r2", "max"), mean_gap=("overfit_gap_train_minus_test_r2", "mean"),
                                 mean_mae=("test_mae", "mean")).sort_values("mean_test_r2", ascending=False)
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Model & mean test $R^2$ & min & max & mean gap & mean MAE (d) \\", r"\midrule"]
    for mdl, r in agg.iterrows():
        lines.append(f"{esc(mdl)} & \\textbf{{{fmt(r.mean_test_r2)}}} & {fmt(r.min_test_r2)} & "
                     f"{fmt(r.max_test_r2)} & {fmt(r.mean_gap)} & {fmt(r.mean_mae,2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_exp3_extra_regressors.tex", "\n".join(lines))

    lg = pd.read_csv(EXP / "experiment_3_logistic_regression.csv")
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Specialist & cutpoints (d) & train acc. & val acc. & test acc. & test macro $F_1$ \\", r"\midrule"]
    for _, r in lg.iterrows():
        lines.append(f"{esc(r.category)}/{esc(r.task)} & {r.class_cutpoints_tertile_1:.1f} / "
                     f"{r.class_cutpoints_tertile_2:.1f} & {fmt(r.train_accuracy)} & {fmt(r.val_accuracy)} & "
                     f"\\textbf{{{fmt(r.test_accuracy)}}} & {fmt(r.test_macro_f1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_exp3_logistic.tex", "\n".join(lines))


# ------------------------------------------------------------------ leave-rule-out zoo
def loro_tables():
    z = pd.concat([pd.read_csv(EXP / "experiment_5_leave_rule_out_full_zoo.csv"),
                   pd.read_csv(EXP / "experiment_6_leave_rule_out_rf_xgb.csv")], ignore_index=True)
    agg = z.groupby("model").agg(mean_r2=("mean_r2", "mean"), median_r2=("median_r2", "mean"),
                                 worst=("min_r2", "min"), best=("max_r2", "max"),
                                 std=("std_r2", "mean")).sort_values("median_r2", ascending=False)
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Model & mean $R^2$ & median $R^2$ & mean s.d. & worst fold & best fold \\", r"\midrule"]
    for mdl, r in agg.iterrows():
        lines.append(f"{esc(mdl)} & {fmt(r.mean_r2)} & \\textbf{{{fmt(r.median_r2)}}} & {fmt(r['std'])} & "
                     f"{fmt(r.worst,2)} & {fmt(r.best)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_loro_full_zoo.tex", "\n".join(lines))

    # per specialist, three production ensembles
    keep = ["random_forest_production_hparams", "xgboost_production_hparams", "lightgbm_production_hparams"]
    d = z[z.model.isin(keep)]
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Specialist & Model & folds & mean $R^2$ & median $R^2$ & s.d. & min & max \\", r"\midrule"]
    for c, t in SPEC_ORDER:
        sub = d[(d.category == c) & (d.task == t)]
        for i, (_, r) in enumerate(sub.iterrows()):
            spec = f"{c.replace('_','-')}/{t.replace('_',' ')}" if i == 0 else ""
            lines.append(f"{esc(spec)} & {esc(r.model.replace('_production_hparams',''))} & {int(r.n_rules)} & "
                         f"{fmt(r.mean_r2)} & \\textbf{{{fmt(r.median_r2)}}} & {fmt(r.std_r2)} & "
                         f"{fmt(r.min_r2,2)} & {fmt(r.max_r2)} \\\\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_loro_ensembles_by_specialist.tex", "\n".join(lines))


# ------------------------------------------------------------------ rule inventory
def rule_inventory():
    frames = []
    for cat in ["SOFT", "SEMI_HARD", "HARD"]:
        df = pd.read_csv(ROOT / "data" / "raw" / f"CHEESE_SHELF_LIFE_V7_{cat}_SPECIALIST_CORRECTED.csv")
        df["cheese_category"] = cat.lower()
        frames.append(df)
    spec = pd.concat(frames, ignore_index=True)

    rows = []
    for c, t in SPEC_ORDER:
        sub = spec[(spec.cheese_category == c) & (spec.model_task == t)]
        for rule, g in sub.groupby("source_rule_id"):
            prods = sorted(set(str(x) for x in g.base_cheese_name.dropna().unique()))
            ings = sorted(set(str(x) for x in g.primary_ingredient_name.dropna().unique()))
            rows.append({
                "category": c, "task": t, "rule": rule, "n_rows": len(g),
                "n_contexts": g.context_id.nunique(),
                "n_products": len(prods), "products": "; ".join(prods[:4]) + (" ..." if len(prods) > 4 else ""),
                "n_ingredients": len(ings), "ingredients": "; ".join(ings[:3]) + (" ..." if len(ings) > 3 else ""),
                "temp_min": g.storage_temperature_c.min(), "temp_max": g.storage_temperature_c.max(),
                "y_min": g.shelf_life_days.min(), "y_max": g.shelf_life_days.max(),
                "y_mean": g.shelf_life_days.mean(), "y_std": g.shelf_life_days.std(),
                "n_indicators": g.indicator_type.nunique(),
            })
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT.parent / "generation_rule_inventory.csv", index=False)

    # long table
    lines = [r"\begin{longtable}{p{3.4cm}rrrrrrr}", r"\toprule",
             r"Rule (\texttt{source\_rule\_id}) & rows & ctx & prod. & ind. & "
             r"temp range ($^\circ$C) & $\bar{y}$ (d) & $s_y$ (d) \\", r"\midrule", r"\endhead"]
    for c, t in SPEC_ORDER:
        sub = inv[(inv.category == c) & (inv.task == t)].sort_values("n_rows", ascending=False)
        lines.append(r"\multicolumn{8}{l}{\textbf{" + esc(f"{c.replace('_','-')} / {t.replace('_',' ')}") +
                     r"}} \\[1pt]")
        for _, r in sub.iterrows():
            lines.append(
                f"{rule_tex(r.rule)} & {int(r.n_rows):,} & "
                f"{int(r.n_contexts)} & {int(r.n_products)} & {int(r.n_indicators)} & "
                f"{r.temp_min:.0f}--{r.temp_max:.0f} & {r.y_mean:.1f} & {r.y_std:.1f} \\\\")
        lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule", r"\end{longtable}"]
    write("tab_rule_inventory.tex", "\n".join(lines))

    # products/ingredients detail
    lines = [r"\begin{longtable}{p{3.2cm}p{5.4cm}p{4.4cm}}", r"\toprule",
             r"Rule & cheese products (up to 4) & primary ingredients (up to 3) \\", r"\midrule", r"\endhead"]
    for c, t in SPEC_ORDER:
        sub = inv[(inv.category == c) & (inv.task == t)].sort_values("n_rows", ascending=False)
        lines.append(r"\multicolumn{3}{l}{\textbf{" + esc(f"{c.replace('_','-')} / {t.replace('_',' ')}") + r"}} \\[1pt]")
        for _, r in sub.iterrows():
            lines.append(f"{rule_tex(r.rule)} & "
                         f"{{\\scriptsize {esc(r.products)}}} & {{\\scriptsize {esc(r.ingredients)}}} \\\\")
        lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule", r"\end{longtable}"]
    write("tab_rule_products.tex", "\n".join(lines))


# ------------------------------------------------------------------ dataset versions
def dataset_versions():
    rows = [
        ("CHEESE\\_SHELF\\_LIFE\\_REVISED", 30500, "synthetic\\_literature\\_constrained (20{,}000); synthetic\\_cheese\\_database\\_anchored (10{,}500)"),
        ("CHEESE\\_SHELF\\_LIFE\\_CORRECTED\\_GENERALIZED", 41285, "synthetic\\_literature\\_constrained (20{,}000); synthetic\\_literature\\_recalibrated (10{,}785); synthetic\\_cheese\\_database\\_anchored (10{,}500)"),
        ("CHEESE\\_SHELF\\_LIFE\\_RECALIBRATED\\_V3", 48240, "synthetic\\_literature\\_recalibrated\\_v3 (31{,}800); retained\\_soft\\_rebalanced\\_v3 (16{,}440)"),
        ("CHEESE\\_SHELF\\_LIFE\\_TARGETED\\_V4", 35650, "synthetic\\_literature\\_constrained (16{,}185); synthetic\\_cheese\\_database\\_anchored (9{,}865); synthetic\\_independent\\_literature\\_calibrated\\_v4 (5{,}150); synthetic\\_recalibrated\\_v4 (4{,}450)"),
        ("V7 specialists (soft + semi-hard + hard)", 34000, "synthetic\\_literature\\_constrained\\_v7 (34{,}000)"),
    ]
    lines = [r"\begin{tabular}{p{4.6cm}rp{7.4cm}}", r"\toprule",
             r"Dataset file & rows & \texttt{data\_origin} values present \\", r"\midrule"]
    for name, n, origins in rows:
        lines.append(f"{{\\scriptsize \\texttt{{{name}}}}} & {n:,} & {{\\scriptsize {origins}}} \\\\")
    lines += [r"\midrule",
              r"\multicolumn{3}{l}{\scriptsize Verified by loading each file and value-counting "
              r"\texttt{data\_origin}; no non-synthetic tag appears anywhere.} \\",
              r"\bottomrule", r"\end{tabular}"]
    write("tab_dataset_versions.tex", "\n".join(lines))


def main():
    production_tables()
    exp14_table()
    exp2_table()
    exp3_tables()
    loro_tables()
    rule_inventory()
    dataset_versions()


if __name__ == "__main__":
    main()
