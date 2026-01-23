# -*- coding: utf-8 -*-
"""
Re-run hard-sample analysis after excluding known-bad wells (e.g., run7 D09/D10).

Input:
  - data/metrics/step2_pred_ct_trajectory_long.csv  (or --long_path)

Outputs:
  - data/metrics/step2_sample_first_success_excl.csv
  - data/metrics/step2_hard_samples_excl.csv
  - data/metrics/step2_group_summary_excl.csv
  - data/metrics/step2_pred_ct_trajectory_summary_excl.csv
  - figures:
      - data/figures/first_success_cycle_hist_excl.png
      - data/figures/hard_samples_trajectories_excl.png
      - data/figures/error_by_true_ct_scatter_excl.png
      - data/figures/ct_error_vs_cycle_excl.png
      - data/figures/ct_accuracy_fraction_vs_cycle_excl.png
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_well_from_well_uid(well_uid: str) -> str:
    # well_uid ends with "__<WELL>" in your pipeline
    if not isinstance(well_uid, str):
        return ""
    if "__" in well_uid:
        return well_uid.split("__")[-1]
    return ""


def ct_bin(true_ct: float) -> str:
    if pd.isna(true_ct):
        return "NA"
    if true_ct <= 20:
        return "Ct<=20"
    if true_ct <= 30:
        return "20<Ct<=30"
    return "Ct>30"


def recompute_summary(df_long: pd.DataFrame) -> pd.DataFrame:
    # df_long columns: cutoff, well_uid, run_id, true_cq, pred_cq, abs_err, max_cycle
    def p90(x):
        return np.quantile(x, 0.90) if len(x) else np.nan

    g = df_long.groupby("cutoff", as_index=False)
    out = g.agg(
        n_curves=("well_uid", "nunique"),
        n_pred=("pred_cq", lambda s: int(s.notna().sum())),
        n_runs=("run_id", "nunique"),
        mae=("abs_err", "mean"),
        median_abs_err=("abs_err", "median"),
        p90_abs_err=("abs_err", p90),
        pct_within_0_5=("abs_err", lambda s: float((s <= 0.5).mean())),
        pct_within_1_0=("abs_err", lambda s: float((s <= 1.0).mean())),
    )
    return out.sort_values("cutoff")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long_path", default="data/metrics/step2_pred_ct_trajectory_long.csv")
    ap.add_argument("--out_dir_metrics", default="data/metrics")
    ap.add_argument("--out_dir_figs", default="data/figures")

    ap.add_argument("--early_cutoff", type=int, default=25)
    ap.add_argument("--tol", type=float, default=1.0)

    # exclusion options
    ap.add_argument(
        "--exclude_run_wells",
        action="append",
        default=[],
        help='Format: "<RUN_ID_REGEX>:D09,D10" (repeatable). Example: --exclude_run_wells "7__7_251209.*:D09,D10"',
    )
    ap.add_argument(
        "--exclude_well_uid",
        action="append",
        default=[],
        help='Exact well_uid to exclude (repeatable). Example: --exclude_well_uid "7__...__D09"',
    )

    args = ap.parse_args()

    ensure_dir(args.out_dir_metrics)
    ensure_dir(args.out_dir_figs)

    df = pd.read_csv(args.long_path)
    required = {"well_uid", "run_id", "true_cq", "pred_cq", "abs_err", "cutoff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in long file: {sorted(missing)}")

    df["well"] = df["well_uid"].map(parse_well_from_well_uid)

    n_before = df["well_uid"].nunique()

    # 1) exclude exact well_uid
    if args.exclude_well_uid:
        df = df[~df["well_uid"].isin(args.exclude_well_uid)].copy()

    # 2) exclude by run_id regex + wells list
    for spec in args.exclude_run_wells:
        if ":" not in spec:
            raise ValueError(f"Bad --exclude_run_wells spec: {spec}")
        run_pat, wells_csv = spec.split(":", 1)
        wells = [w.strip() for w in wells_csv.split(",") if w.strip()]
        run_re = re.compile(run_pat)
        mask_run = df["run_id"].astype(str).apply(lambda x: bool(run_re.search(x)))
        mask_well = df["well"].isin(wells)
        df = df[~(mask_run & mask_well)].copy()

    n_after = df["well_uid"].nunique()
    print(f"[FILTER] unique samples before={n_before} after={n_after} removed={n_before - n_after}")

    max_cutoff = int(df["cutoff"].max())
    print(f"[INFO] max_cutoff_in_file={max_cutoff} early_cutoff={args.early_cutoff} tol={args.tol}")

    # --- per-sample first success cutoff (<= early_cutoff) ---
    early_df = df[df["cutoff"] <= args.early_cutoff].copy()
    early_df = early_df.sort_values(["well_uid", "cutoff"])

    # find first cutoff where abs_err <= tol
    ok_mask = early_df["abs_err"] <= args.tol
    first_ok = (
        early_df[ok_mask]
        .groupby("well_uid", as_index=False)["cutoff"]
        .min()
        .rename(columns={"cutoff": "first_success_cutoff"})
    )

    # abs err at early cutoff and at max cutoff
    err_at_early = (
        df[df["cutoff"] == args.early_cutoff][["well_uid", "abs_err"]]
        .rename(columns={"abs_err": f"abs_err_at_{args.early_cutoff}"})
    )
    err_at_max = (
        df[df["cutoff"] == max_cutoff][["well_uid", "abs_err"]]
        .rename(columns={"abs_err": f"abs_err_at_{max_cutoff}"})
    )

    base = (
        df.groupby("well_uid", as_index=False)
        .agg(run_id=("run_id", "first"), true_ct=("true_cq", "first"))
    )

    out = base.merge(first_ok, on="well_uid", how="left")
    out = out.merge(err_at_early, on="well_uid", how="left")
    out = out.merge(err_at_max, on="well_uid", how="left")

    # group labeling
    def assign_group(row):
        fsc = row.get("first_success_cutoff", np.nan)
        err_max = row.get(f"abs_err_at_{max_cutoff}", np.nan)
        if not pd.isna(fsc) and fsc <= args.early_cutoff:
            return "EARLY_SUCCESS"
        if not pd.isna(err_max) and err_max <= args.tol:
            return "LATE_SUCCESS"
        return "INDETERMINATE"

    out["group"] = out.apply(assign_group, axis=1)
    out["ct_bin"] = out["true_ct"].apply(ct_bin)

    # save per-sample table
    p_first = os.path.join(args.out_dir_metrics, "step2_sample_first_success_excl.csv")
    out.to_csv(p_first, index=False)
    print(f"[SAVED] {p_first}")

    # group summary by ct_bin
    grp = (
        out.groupby(["ct_bin", "group"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    total_by_bin = out.groupby("ct_bin").size().rename("n_total").reset_index()
    grp = grp.merge(total_by_bin, on="ct_bin", how="left")
    grp["pct"] = (grp["n"] / grp["n_total"]) * 100.0

    p_grp = os.path.join(args.out_dir_metrics, "step2_group_summary_excl.csv")
    grp.to_csv(p_grp, index=False)
    print(f"[SAVED] {p_grp}")

    # hard samples list
    hard = out[out["group"] != "EARLY_SUCCESS"].copy()
    p_hard = os.path.join(args.out_dir_metrics, "step2_hard_samples_excl.csv")
    hard.to_csv(p_hard, index=False)
    print(f"[SAVED] {p_hard}")

    # recompute overall cutoff summary as well
    summ = recompute_summary(df)
    p_summ = os.path.join(args.out_dir_metrics, "step2_pred_ct_trajectory_summary_excl.csv")
    summ.to_csv(p_summ, index=False)
    print(f"[SAVED] {p_summ}")

    # ---- FIG 1: histogram of first success cutoff ----
    plt.figure()
    vals = out["first_success_cutoff"].dropna().astype(int)
    plt.hist(vals, bins=np.arange(1, args.early_cutoff + 2) - 0.5)
    plt.xlabel("First cutoff achieving |err| <= tol")
    plt.ylabel("Count")
    plt.title(f"First success cutoff (tol={args.tol}, early_cutoff={args.early_cutoff})")
    fig1 = os.path.join(args.out_dir_figs, "first_success_cycle_hist_excl.png")
    plt.tight_layout()
    plt.savefig(fig1, dpi=200)
    plt.close()
    print(f"[FIG] {fig1}")

    # ---- FIG 2: trajectories of hard samples (abs_err vs cutoff) ----
    plt.figure()
    hard_ids = set(hard["well_uid"].tolist())
    if len(hard_ids) == 0:
        plt.text(0.5, 0.5, "No hard samples after exclusion", ha="center", va="center")
    else:
        for wid in sorted(hard_ids):
            tmp = df[df["well_uid"] == wid].sort_values("cutoff")
            plt.plot(tmp["cutoff"], tmp["abs_err"], alpha=0.8)
        plt.axvline(args.early_cutoff, linestyle="--")
        plt.axhline(args.tol, linestyle="--")
    plt.xlabel("Cutoff (cycle)")
    plt.ylabel("Absolute error |predCt - trueCt|")
    plt.title("Hard sample trajectories (after exclusion)")
    fig2 = os.path.join(args.out_dir_figs, "hard_samples_trajectories_excl.png")
    plt.tight_layout()
    plt.savefig(fig2, dpi=200)
    plt.close()
    print(f"[FIG] {fig2}")

    # ---- FIG 3: scatter true Ct vs error at early_cutoff ----
    plt.figure()
    x = out["true_ct"].values
    y = out[f"abs_err_at_{args.early_cutoff}"].values
    plt.scatter(x, y, alpha=0.8)
    plt.axhline(args.tol, linestyle="--")
    plt.xlabel("True Ct")
    plt.ylabel(f"Abs error at cutoff={args.early_cutoff}")
    plt.title("Error vs True Ct (after exclusion)")
    fig3 = os.path.join(args.out_dir_figs, "error_by_true_ct_scatter_excl.png")
    plt.tight_layout()
    plt.savefig(fig3, dpi=200)
    plt.close()
    print(f"[FIG] {fig3}")

    # ---- FIG 4: overall MAE vs cutoff ----
    plt.figure()
    plt.plot(summ["cutoff"], summ["mae"], marker="o")
    plt.xlabel("Cutoff (cycle)")
    plt.ylabel("MAE")
    plt.title("MAE vs cutoff (after exclusion)")
    fig4 = os.path.join(args.out_dir_figs, "ct_error_vs_cycle_excl.png")
    plt.tight_layout()
    plt.savefig(fig4, dpi=200)
    plt.close()
    print(f"[FIG] {fig4}")

    # ---- FIG 5: accuracy fractions vs cutoff ----
    plt.figure()
    plt.plot(summ["cutoff"], summ["pct_within_0_5"] * 100.0, marker="o", label="|err|<=0.5")
    plt.plot(summ["cutoff"], summ["pct_within_1_0"] * 100.0, marker="s", label="|err|<=1.0")
    plt.axvline(args.early_cutoff, linestyle="--")
    plt.xlabel("Cutoff (cycle)")
    plt.ylabel("Percent (%)")
    plt.title("Accuracy fraction vs cutoff (after exclusion)")
    plt.legend()
    fig5 = os.path.join(args.out_dir_figs, "ct_accuracy_fraction_vs_cycle_excl.png")
    plt.tight_layout()
    plt.savefig(fig5, dpi=200)
    plt.close()
    print(f"[FIG] {fig5}")

    # print group counts
    counts = out["group"].value_counts(dropna=False)
    print("[GROUP] counts")
    print(counts.to_string())
    print("[DONE]")


if __name__ == "__main__":
    main()
