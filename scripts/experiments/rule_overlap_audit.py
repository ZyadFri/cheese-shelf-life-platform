#!/usr/bin/env python
"""
Quantifies the central methodological finding: under the ORIGINAL production
split (group by context_id, 70/15/15), how many generation rules are shared
between train / validation / test, and what share of test rows come from a
rule the model already saw during training.

Reproduces the production split exactly by importing make_context_splits from
model_service.py with the production seed.

Outputs:
  reports/overfitting_investigation_meeting/rule_overlap_audit.csv
  reports/overfitting_investigation_meeting/tables/tab_rule_overlap.tex

Usage: .venv/Scripts/python.exe scripts/experiments/rule_overlap_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from model_service import make_context_splits, SPLIT_FRACTIONS  # noqa: E402

OUT = ROOT / "reports" / "overfitting_investigation_meeting"
TABLES = OUT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
SEED = 42

SPEC_ORDER = [
    ("soft", "general_shelf_life"), ("semi_hard", "general_shelf_life"), ("hard", "general_shelf_life"),
    ("soft", "safety_endpoint"), ("semi_hard", "safety_endpoint"), ("hard", "safety_endpoint"),
]


def main():
    rows = []
    for cat in ["soft", "semi_hard", "hard"]:
        df = pd.read_csv(ROOT / "data" / "raw" /
                         f"CHEESE_SHELF_LIFE_V7_{cat.upper()}_SPECIALIST_CORRECTED.csv")
        for task in ["general_shelf_life", "safety_endpoint"]:
            task_df = df[df.model_task == task].reset_index(drop=True)
            splits = make_context_splits(task_df, seed=SEED)
            tr, va, te = splits["train"], splits["validation"], splits["test"]
            rules_tr = set(tr.source_rule_id.unique())
            rules_va = set(va.source_rule_id.unique())
            rules_te = set(te.source_rule_id.unique())
            shared = rules_tr & rules_te
            test_rows_from_seen_rule = int(te.source_rule_id.isin(rules_tr).sum())
            # context-level check, for contrast
            ctx_overlap = len(set(tr.context_id) & set(te.context_id))
            rows.append({
                "category": cat, "task": task,
                "n_total": len(task_df), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                "n_rules_total": task_df.source_rule_id.nunique(),
                "n_rules_train": len(rules_tr), "n_rules_val": len(rules_va), "n_rules_test": len(rules_te),
                "n_rules_shared_train_test": len(shared),
                "pct_test_rules_seen_in_train": 100.0 * len(shared) / max(1, len(rules_te)),
                "n_test_rows_from_seen_rule": test_rows_from_seen_rule,
                "pct_test_rows_from_seen_rule": 100.0 * test_rows_from_seen_rule / max(1, len(te)),
                "n_contexts_shared_train_test": ctx_overlap,
                "n_contexts_total": task_df.context_id.nunique(),
                "rows_per_context_mean": len(task_df) / task_df.context_id.nunique(),
                "contexts_per_rule_mean": task_df.context_id.nunique() / task_df.source_rule_id.nunique(),
            })
            print(f"{cat}/{task}: rules train={len(rules_tr)} test={len(rules_te)} shared={len(shared)}  "
                  f"test rows from seen rule={test_rows_from_seen_rule}/{len(te)} "
                  f"({100.0*test_rows_from_seen_rule/max(1,len(te)):.1f}%)  ctx overlap={ctx_overlap}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "rule_overlap_audit.csv", index=False)

    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Specialist & rules & rules in & rules in & shared & test rows from & contexts shared \\",
             r" & total & train & test & rules & a seen rule & train/test \\", r"\midrule"]
    for c, t in SPEC_ORDER:
        r = out[(out.category == c) & (out.task == t)].iloc[0]
        spec = f"{c.replace('_','-')}/{t.replace('_',' ')}".replace("_", r"\_")
        lines.append(f"{spec} & {int(r.n_rules_total)} & {int(r.n_rules_train)} & {int(r.n_rules_test)} & "
                     f"{int(r.n_rules_shared_train_test)} & "
                     f"\\textbf{{{int(r.n_test_rows_from_seen_rule):,}}} / {int(r.n_test):,} "
                     f"({r.pct_test_rows_from_seen_rule:.1f}\\%) & {int(r.n_contexts_shared_train_test)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "tab_rule_overlap.tex").write_text("\n".join(lines), encoding="utf-8")
    print("\nWritten: rule_overlap_audit.csv, tab_rule_overlap.tex")
    print("SPLIT_FRACTIONS used:", SPLIT_FRACTIONS)


if __name__ == "__main__":
    main()
