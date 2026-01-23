#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Early-cycle Ct evaluation from a long-format prediction CSV.

Assumptions:
- You already have a "long" CSV with per-(well_uid, cutoff) predictions.
- Columns needed (case-insensitive match):
    - well_uid  (unique id for a sample/well)
    - cutoff    (cycle cutoff integer)
    - pred_ct   (predicted Ct at that cutoff)
    - true_ct   (ground-truth Ct / Cq)
  Optional:
    - run_id    (if missing, will be inferred from well_uid)
    - well      (if missing, will be inferred from well_uid)

Outputs (default):
  data/metrics/step2_ct_error_vs_cycle_excl.csv
  data/metrics/step2_sample_first_success_excl.csv
  data/metrics/step2_group_summary_excl.csv
  data/metrics/step2_hard_samples_excl.csv
  data/metrics/step2_pred_ct_trajectory_summary_excl.csv
  data/figures/*.png
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Utilities
# -------------------------

def _norm_cols(df: pd.DataFrame) -> Dict[str, str]:
    """Map lower-case column name -> actual column name."""
    return {c.lower(): c for c in df.columns}

def _pick_col(cols_map: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in cols_map:
            return cols_map[k]
    return None

def infer_run_and_well(well_uid: str) -> Tuple[str, str]:
    """
    Try to infer run_id and well from well_uid.
    Expected patterns seen in your logs:
      run_id__D09
      ...__E05
    If not found, returns ("", "").
    """
    if not isinstance(well_uid, str):
        return "", ""
    # last "__<Well>" pattern
    m = re.search(r"__(?P<well>[A-H][0-1][0-9])$", well_uid)
    well = m.group("well") if m else ""
    run_id = well_uid[:-len(well)-2] if well else well_uid
    # Clean possible trailing separators
    run_id = run_id.rstrip("_")
    return run_id, well

def parse_exclude_run_wells(spec: str) -> Dict[str, set]:
    """
    Parse: "RUN_ID:D09,D10;RUN2:E01" -> {RUN_ID:{D09,D10}, RUN2:{E01}}
    Accept separators: ';' or '|' between run blocks.
    """
    out: Dict[str, set] = {}
    if not spec:
        return out
    blocks = re.split(r"[;|]+", spec.strip())
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if ":" not in b:
            # if only run provided, treat as exclude_runs use-case
            out.setdefault(b, set())
            continue
        run, wells = b.split(":", 1)
        run = run.strip()
        wells_set = set([w.strip().upper() for w in wells.split(",") if w.strip()])
        out.setdefault(run, set()).update(wells_set)
    return out

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

@dataclass
class EvalConfig:
    early_cutoff: int
    tol: float
    min_ct: float
    max_ct: float
    apply_ct_filter: bool
    hard_topk: int


# -------------------------
# Core evaluation
# -------------------------

def compute_metrics_by_cutoff(df_long: pd.DataFrame) -> pd.DataFrame:
    """Return per-cutoff metrics table."""
    g = df_long.groupby("cutoff", as_index=False)
    rows = []
    for cutoff, sdf in g:
        err = (sdf["pred_ct"] - sdf["true_ct"]).astype(float)
        abs_err = err.abs()
        rows.append({
            "cutoff": int(cutoff),
            "n_curves": int(sdf["well_uid"].nunique()),
            "n_pred": int(len(sdf)),
            "n_runs": int(sdf["run_id"].nunique()),
            "mae": float(abs_err.mean()) if len(abs_err) else np.nan,
            "median_abs_err": float(abs_err.median()) if len(abs_err) else np.nan,
            "p90_abs_err": float(np.quantile(abs_err, 0.90)) if len(abs_err) else np.nan,
            "pct_within_0_5": float((abs_err <= 0.5).mean()) if len(abs_err) else np.nan,
            "pct_within_1_0": float((abs_err <= 1.0).mean()) if len(abs_err) else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("cutoff").reset_index(drop=True)
    return out

def first_success_and_groups(df_long: pd.DataFrame, cfg: EvalConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-well: find first cutoff achieving |err| <= tol.
    Group rules:
      - EARLY_SUCCESS: first_success_cutoff exists and <= early_cutoff
      - LATE_SUCCESS: first_success_cutoff exists and > early_cutoff
      - INDETERMINATE: no cutoff achieves tol
    Also adds:
      - abs_err_at_<early_cutoff>
      - abs_err_at_<max_cutoff>
      - ct_bin
    """
    max_cutoff_in_file = int(df_long["cutoff"].max())

    # Prepare fast lookup tables
    df_long = df_long.copy()
    df_long["abs_err"] = (df_long["pred_ct"] - df_long["true_ct"]).abs()

    # ct bins (same style as your printouts)
    def ct_bin(v: float) -> str:
        if pd.isna(v):
            return "NA"
        if v <= 20:
            return "Ct<=20"
        if v <= 30:
            return "20<Ct<=30"
        return "Ct>30"

    per_well_rows = []
    for well_uid, sdf in df_long.groupby("well_uid"):
        sdf = sdf.sort_values("cutoff")
        true_ct = float(sdf["true_ct"].iloc[0])
        run_id = str(sdf["run_id"].iloc[0])
        well = str(sdf["well"].iloc[0])

        # first success
        ok = sdf[sdf["abs_err"] <= cfg.tol]
        first_success_cutoff = int(ok["cutoff"].iloc[0]) if len(ok) else np.nan

        # abs_err at early_cutoff / max_cutoff
        e_row = sdf[sdf["cutoff"] == cfg.early_cutoff]
        abs_err_at_early = float(e_row["abs_err"].iloc[0]) if len(e_row) else np.nan

        m_row = sdf[sdf["cutoff"] == max_cutoff_in_file]
        abs_err_at_max = float(m_row["abs_err"].iloc[0]) if len(m_row) else np.nan

        # group
        if pd.notna(first_success_cutoff):
            if first_success_cutoff <= cfg.early_cutoff:
                group = "EARLY_SUCCESS"
            else:
                group = "LATE_SUCCESS"
        else:
            group = "INDETERMINATE"

        per_well_rows.append({
            "well_uid": well_uid,
            "run_id": run_id,
            "well": well,
            "true_ct": true_ct,
            "first_success_cutoff": first_success_cutoff,
            f"abs_err_at_{cfg.early_cutoff}": abs_err_at_early,
            f"abs_err_at_{max_cutoff_in_file}": abs_err_at_max,
            "group": group,
            "ct_bin": ct_bin(true_ct),
        })

    per_well = pd.DataFrame(per_well_rows)

    # Group summary by ct_bin x group
    grp = (
        per_well
        .groupby(["ct_bin", "group"], as_index=False)
        .agg(n=("well_uid", "count"))
    )
    totals = per_well.groupby("ct_bin", as_index=False).agg(n_total=("well_uid", "count"))
    grp = grp.merge(totals, on="ct_bin", how="left")
    grp["pct"] = grp["n"] / grp["n_total"] * 100.0
    grp = grp.sort_values(["ct_bin", "group"]).reset_index(drop=True)

    return per_well, grp

def select_hard_samples(df_long: pd.DataFrame, per_well: pd.DataFrame, cfg: EvalConfig) -> pd.DataFrame:
    """
    Define 'hard' samples: those not meeting tol at early_cutoff (or missing early_cutoff),
    ranked by abs_err_at_early_cutoff descending.
    """
    col = f"abs_err_at_{cfg.early_cutoff}"
    hard = per_well.copy()
    hard = hard[(hard["group"] != "EARLY_SUCCESS") | (hard[col] > cfg.tol) | (hard[col].isna())]
    hard = hard.sort_values(col, ascending=False, na_position="last")
    if cfg.hard_topk and cfg.hard_topk > 0:
        hard = hard.head(cfg.hard_topk)
    return hard

def trajectories_summary(df_long: pd.DataFrame, per_well: pd.DataFrame, cfg: EvalConfig) -> pd.DataFrame:
    """
    A compact per-well trajectory summary:
      - err at early_cutoff
      - err at max cutoff
      - min abs_err within early window
      - cutoff where min abs_err occurs (within early window)
    """
    max_cutoff_in_file = int(df_long["cutoff"].max())
    df_long = df_long.copy()
    df_long["abs_err"] = (df_long["pred_ct"] - df_long["true_ct"]).abs()

    early = df_long[df_long["cutoff"] <= cfg.early_cutoff]
    idx = early.groupby("well_uid")["abs_err"].idxmin()
    best = early.loc[idx, ["well_uid", "cutoff", "abs_err"]].rename(columns={
        "cutoff": "best_cutoff_within_early",
        "abs_err": "best_abs_err_within_early",
    })

    out = per_well.merge(best, on="well_uid", how="left")
    # keep stable ordering
    return out.sort_values(["group", "ct_bin", "run_id", "well"]).reset_index(drop=True)

# -------------------------
# Plotting
# -------------------------

def plot_accuracy_fraction(metrics: pd.DataFrame, fig_path: str) -> None:
    x = metrics["cutoff"].astype(int).values
    y05 = metrics["pct_within_0_5"].values * 100.0
    y10 = metrics["pct_within_1_0"].values * 100.0

    plt.figure(figsize=(10, 7))
    plt.plot(x, y05, marker="o", label="|err|<=0.5")
    plt.plot(x, y10, marker="s", label="|err|<=1.0")
    plt.xlabel("Cutoff (cycle)")
    plt.ylabel("Percent (%)")
    plt.title("Accuracy fraction vs cutoff (after exclusion)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

def plot_mae(metrics: pd.DataFrame, fig_path: str) -> None:
    x = metrics["cutoff"].astype(int).values
    y = metrics["mae"].values

    plt.figure(figsize=(10, 7))
    plt.plot(x, y, marker="o")
    plt.xlabel("Cutoff (cycle)")
    plt.ylabel("MAE")
    plt.title("MAE vs cutoff (after exclusion)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

def plot_error_by_true_ct(df_long: pd.DataFrame, cutoff: int, fig_path: str, tol: float) -> None:
    sdf = df_long[df_long["cutoff"] == cutoff].copy()
    sdf["abs_err"] = (sdf["pred_ct"] - sdf["true_ct"]).abs()

    plt.figure(figsize=(10, 7))
    plt.scatter(sdf["true_ct"], sdf["abs_err"], alpha=0.8)
    plt.axhline(tol, linestyle="--")
    plt.xlabel("True Ct")
    plt.ylabel(f"Abs error at cutoff={cutoff}")
    plt.title("Error vs True Ct (after exclusion)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

def plot_first_success_hist(per_well: pd.DataFrame, early_cutoff: int, tol: float, fig_path: str) -> None:
    # Only those who succeed within early window
    x = per_well.loc[per_well["group"] == "EARLY_SUCCESS", "first_success_cutoff"].dropna().astype(int).values
    plt.figure(figsize=(10, 7))
    plt.hist(x, bins=min(early_cutoff, 30))
    plt.xlabel(f"First cutoff achieving |err| <= {tol}")
    plt.ylabel("Count")
    plt.title(f"First success cutoff (tol={tol}, early_cutoff={early_cutoff})")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

def plot_hard_trajectories(df_long: pd.DataFrame, hard: pd.DataFrame, early_cutoff: int, tol: float, fig_path: str) -> None:
    # Plot abs error trajectories for hard samples across cutoffs
    df_long = df_long.copy()
    df_long["abs_err"] = (df_long["pred_ct"] - df_long["true_ct"]).abs()

    plt.figure(figsize=(12, 8))
    for well_uid in hard["well_uid"].tolist():
        sdf = df_long[df_long["well_uid"] == well_uid].sort_values("cutoff")
        plt.plot(sdf["cutoff"], sdf["abs_err"])
    plt.axvline(early_cutoff, linestyle="--")
    plt.axhline(tol, linestyle="--")
    plt.xlabel("Cutoff (cycle)")
    plt.ylabel("|predCt - trueCt|")
    plt.title("Hard sample trajectories (after exclusion)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_long", type=str, default="outputs/preds_from_saved_models_long.csv",
                    help="Long-format predictions CSV (default: outputs/preds_from_saved_models_long.csv)")
    ap.add_argument("--early_cutoff", type=int, default=25)
    ap.add_argument("--tol", type=float, default=1.0)

    ap.add_argument("--exclude_run_wells", type=str, default="",
                    help='Example: "RUN_ID:D09,D10;RUN2:E01"')
    ap.add_argument("--exclude_runs", type=str, default="",
                    help='Comma-separated run_ids to exclude entirely (example: "run7,run8")')

    ap.add_argument("--min_ct", type=float, default=5.0, help="Min Ct to keep (default 5.0)")
    ap.add_argument("--max_ct", type=float, default=45.0, help="Max Ct to keep (default 45.0)")
    ap.add_argument("--no_ct_filter", action="store_true", help="Disable Ct range filter")

    ap.add_argument("--hard_topk", type=int, default=25, help="How many hard samples to plot/save (default 25)")

    ap.add_argument("--metrics_dir", type=str, default="data/metrics")
    ap.add_argument("--fig_dir", type=str, default="data/figures")

    args = ap.parse_args()

    cfg = EvalConfig(
        early_cutoff=int(args.early_cutoff),
        tol=float(args.tol),
        min_ct=float(args.min_ct),
        max_ct=float(args.max_ct),
        apply_ct_filter=not bool(args.no_ct_filter),
        hard_topk=int(args.hard_topk),
    )

    if not os.path.exists(args.preds_long):
        raise FileNotFoundError(f"preds_long not found: {args.preds_long}")

    df = pd.read_csv(args.preds_long)
    cols = _norm_cols(df)

    c_well_uid = _pick_col(cols, ["well_uid", "wellid", "uid"])
    c_cutoff = _pick_col(cols, ["cutoff", "cycle_cutoff", "k", "t"])
    c_pred = _pick_col(cols, ["pred_ct", "predicted_ct", "pred", "yhat", "predicted"])
    c_true = _pick_col(cols, ["true_ct", "true", "ct", "cq", "label", "y"])

    if not all([c_well_uid, c_cutoff, c_pred, c_true]):
        raise ValueError(
            "Missing required columns in preds_long.\n"
            f"Found columns: {list(df.columns)}\n"
            "Need (case-insensitive): well_uid, cutoff, pred_ct, true_ct (or ct/cq)."
        )

    # Standardize column names
    df = df.rename(columns={
        c_well_uid: "well_uid",
        c_cutoff: "cutoff",
        c_pred: "pred_ct",
        c_true: "true_ct",
    })

    # run_id / well
    c_run = _pick_col(cols, ["run_id", "run", "runid"])
    c_well = _pick_col(cols, ["well", "well_name", "wellid2"])
    if c_run and c_run in df.columns:
        df = df.rename(columns={c_run: "run_id"})
    if c_well and c_well in df.columns:
        df = df.rename(columns={c_well: "well"})

    if "run_id" not in df.columns or "well" not in df.columns:
        runs, wells = [], []
        for uid in df["well_uid"].tolist():
            r, w = infer_run_and_well(uid)
            runs.append(r)
            wells.append(w)
        if "run_id" not in df.columns:
            df["run_id"] = runs
        if "well" not in df.columns:
            df["well"] = wells

    # Basic type cleanup
    df["cutoff"] = pd.to_numeric(df["cutoff"], errors="coerce").astype("Int64")
    df["pred_ct"] = pd.to_numeric(df["pred_ct"], errors="coerce")
    df["true_ct"] = pd.to_numeric(df["true_ct"], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["well_uid", "cutoff", "pred_ct", "true_ct"]).copy()
    df["cutoff"] = df["cutoff"].astype(int)

    # Ct range filter
    before_uid = df["well_uid"].nunique()
    if cfg.apply_ct_filter:
        df = df[(df["true_ct"] >= cfg.min_ct) & (df["true_ct"] <= cfg.max_ct)].copy()

    # Exclusions: run+wells
    ex_rw = parse_exclude_run_wells(args.exclude_run_wells)
    if ex_rw:
        mask = np.ones(len(df), dtype=bool)
        for run_id, wells_set in ex_rw.items():
            if not wells_set:
                # if no wells provided, exclude nothing here
                continue
            m = ~((df["run_id"] == run_id) & (df["well"].str.upper().isin(wells_set)))
            mask &= m
        df = df[mask].copy()

    # Exclusions: runs
    ex_runs = [r.strip() for r in (args.exclude_runs or "").split(",") if r.strip()]
    if ex_runs:
        df = df[~df["run_id"].isin(ex_runs)].copy()

    after_uid = df["well_uid"].nunique()
    removed = before_uid - after_uid
    print(f"[FILTER] unique samples before={before_uid} after={after_uid} removed={removed}")

    max_cutoff_in_file = int(df["cutoff"].max())
    print(f"[INFO] max_cutoff_in_file={max_cutoff_in_file} early_cutoff={cfg.early_cutoff} tol={cfg.tol}")

    # Compute outputs
    ensure_dir(args.metrics_dir)
    ensure_dir(args.fig_dir)

    metrics = compute_metrics_by_cutoff(df)
    metrics_path = os.path.join(args.metrics_dir, "step2_ct_error_vs_cycle_excl.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"[SAVED] {metrics_path}")

    per_well, grp = first_success_and_groups(df, cfg)

    per_well_path = os.path.join(args.metrics_dir, "step2_sample_first_success_excl.csv")
    per_well.to_csv(per_well_path, index=False)
    print(f"[SAVED] {per_well_path}")

    grp_path = os.path.join(args.metrics_dir, "step2_group_summary_excl.csv")
    grp.to_csv(grp_path, index=False)
    print(f"[SAVED] {grp_path}")

    traj = trajectories_summary(df, per_well, cfg)
    traj_path = os.path.join(args.metrics_dir, "step2_pred_ct_trajectory_summary_excl.csv")
    traj.to_csv(traj_path, index=False)
    print(f"[SAVED] {traj_path}")

    hard = select_hard_samples(df, per_well, cfg)
    hard_path = os.path.join(args.metrics_dir, "step2_hard_samples_excl.csv")
    hard.to_csv(hard_path, index=False)
    print(f"[SAVED] {hard_path}")

    # Figures
    fig1 = os.path.join(args.fig_dir, "first_success_cycle_hist_excl.png")
    fig2 = os.path.join(args.fig_dir, "hard_samples_trajectories_excl.png")
    fig3 = os.path.join(args.fig_dir, "error_by_true_ct_scatter_excl.png")
    fig4 = os.path.join(args.fig_dir, "ct_error_vs_cycle_excl.png")
    fig5 = os.path.join(args.fig_dir, "ct_accuracy_fraction_vs_cycle_excl.png")

    plot_first_success_hist(per_well, cfg.early_cutoff, cfg.tol, fig1)
    print(f"[FIG] {fig1}")

    plot_hard_trajectories(df, hard, cfg.early_cutoff, cfg.tol, fig2)
    print(f"[FIG] {fig2}")

    plot_error_by_true_ct(df, cfg.early_cutoff, fig3, cfg.tol)
    print(f"[FIG] {fig3}")

    plot_mae(metrics, fig4)
    print(f"[FIG] {fig4}")

    plot_accuracy_fraction(metrics, fig5)
    print(f"[FIG] {fig5}")

    # Print group counts (same vibe as 네 로그)
    counts = per_well["group"].value_counts()
    print("[GROUP] counts")
    print(pd.DataFrame({"group": counts.index, "count": counts.values}).to_string(index=False))

    print("[DONE]")


if __name__ == "__main__":
    main()
