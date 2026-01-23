# core/step2_analyze_hard_samples.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/cphotonic/qpcr_v2")
LONG_CSV = ROOT / "data" / "metrics" / "step2_pred_ct_trajectory_long.csv"

OUT_METRICS = ROOT / "data" / "metrics"
OUT_FIG = ROOT / "data" / "figures"
OUT_METRICS.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

# ---- USER SETTINGS ----
EARLY_CUTOFF = 25          # early-stop boundary
TOL = 1.0                  # absolute error tolerance (cycles)
TOPK_HARD_PLOT = 12        # how many hard samples to plot trajectories for


def _guess_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Make the script robust to small column-name changes.
    We expect at least: well_uid, cutoff, y_true, y_pred.
    """
    cols = {c.lower(): c for c in df.columns}

    def pick(*cands: str) -> str:
        for k in cands:
            if k in cols:
                return cols[k]
        raise KeyError(f"Missing any of {cands}. Available={list(df.columns)}")

    col = {}
    col["well_uid"] = pick("well_uid", "wellid", "curve_id")
    col["cutoff"] = pick("cutoff", "cycle_cutoff", "cycle")
    col["y_true"] = pick("y_true", "true_ct", "true_cq", "cq", "ct")
    col["y_pred"] = pick("y_pred", "pred_ct", "pred_cq", "pred")
    # optional
    col["run_id"] = cols.get("run_id", None)
    return col


def main():
    df = pd.read_csv(LONG_CSV)
    col = _guess_columns(df)

    # Ensure types
    df[col["cutoff"]] = pd.to_numeric(df[col["cutoff"]], errors="coerce").astype(int)
    df[col["y_true"]] = pd.to_numeric(df[col["y_true"]], errors="coerce")
    df[col["y_pred"]] = pd.to_numeric(df[col["y_pred"]], errors="coerce")

    df = df.dropna(subset=[col["well_uid"], col["cutoff"], col["y_true"], col["y_pred"]]).copy()

    # abs error per row
    df["abs_err"] = (df[col["y_pred"]] - df[col["y_true"]]).abs()

    # ------------------------------------------------------------------
    # 1) First-success cycle per sample (well_uid)
    # ------------------------------------------------------------------
    # success means abs_err <= TOL
    success = df[df["abs_err"] <= TOL].copy()

    # earliest cutoff that succeeds
    first_success = (
        success.groupby(col["well_uid"])[col["cutoff"]]
        .min()
        .rename("first_success_cutoff")
        .reset_index()
    )

    # For samples that never succeed, first_success_cutoff = NaN
    all_true = df.groupby(col["well_uid"])[col["y_true"]].first().rename("true_ct").reset_index()
    out = all_true.merge(first_success, on=col["well_uid"], how="left")

    # also compute error at cutoff=EARLY_CUTOFF and at max cutoff (often 40)
    def _err_at(cut: int) -> pd.Series:
        sub = df[df[col["cutoff"]] == cut].set_index(col["well_uid"])["abs_err"]
        return sub.rename(f"abs_err_at_{cut}")

    out = out.merge(_err_at(EARLY_CUTOFF).reset_index(), on=col["well_uid"], how="left")
    # max cutoff per sample present in the file
    max_cut = df[col["cutoff"]].max()
    out = out.merge(_err_at(int(max_cut)).reset_index(), on=col["well_uid"], how="left")

    # run_id (if present)
    if col["run_id"] is not None:
        run_map = df.groupby(col["well_uid"])[col["run_id"]].first().rename("run_id").reset_index()
        out = run_map.merge(out, on=col["well_uid"], how="right")

    # ------------------------------------------------------------------
    # 2) Classify samples (EARLY / LATE / INDETERMINATE)
    # ------------------------------------------------------------------
    def classify(x):
        fs = x["first_success_cutoff"]
        if pd.isna(fs):
            return "INDETERMINATE"
        if fs <= EARLY_CUTOFF:
            return "EARLY_SUCCESS"
        return "LATE_SUCCESS"

    out["group"] = out.apply(classify, axis=1)

    # ------------------------------------------------------------------
    # 3) Summary numbers (overall and by true_ct bins)
    # ------------------------------------------------------------------
    n_total = len(out)
    grp_counts = out["group"].value_counts().to_dict()

    # bins by true Ct (tweak as you like)
    bins = [-np.inf, 20, 30, np.inf]
    labels = ["Ct<=20", "20<Ct<=30", "Ct>30"]
    out["ct_bin"] = pd.cut(out["true_ct"], bins=bins, labels=labels)

    group_summary = (
        out.groupby(["ct_bin", "group"])
        .size()
        .rename("n")
        .reset_index()
    )
    total_by_bin = out.groupby("ct_bin").size().rename("n_total").reset_index()
    group_summary = group_summary.merge(total_by_bin, on="ct_bin", how="left")
    group_summary["pct"] = group_summary["n"] / group_summary["n_total"] * 100.0

    # Save tables
    out_path = OUT_METRICS / "step2_sample_first_success.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    hard = out[out["group"] != "EARLY_SUCCESS"].copy()
    hard_path = OUT_METRICS / "step2_hard_samples.csv"
    hard.to_csv(hard_path, index=False, encoding="utf-8-sig")

    gs_path = OUT_METRICS / "step2_group_summary.csv"
    group_summary.to_csv(gs_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 4) Figures
    # ------------------------------------------------------------------
    # (A) histogram of first success cutoff
    plt.figure(figsize=(10, 5))
    fs_vals = out["first_success_cutoff"].dropna().astype(int)
    plt.hist(fs_vals, bins=range(1, int(max_cut) + 2), rwidth=0.9)
    plt.axvline(EARLY_CUTOFF, linestyle="--")
    plt.title("First success cutoff histogram")
    plt.xlabel("First cutoff where abs_err <= TOL")
    plt.ylabel("Number of samples")
    fig1 = OUT_FIG / "first_success_cycle_hist.png"
    plt.tight_layout()
    plt.savefig(fig1, dpi=200)
    plt.close()

    # (B) hard sample trajectories: abs_err vs cutoff
    # pick TOPK hardest by abs_err_at_EARLY_CUTOFF (or max error if missing)
    key_early = f"abs_err_at_{EARLY_CUTOFF}"
    sort_col = key_early if key_early in hard.columns else f"abs_err_at_{int(max_cut)}"
    hard2 = hard.copy()
    hard2[sort_col] = pd.to_numeric(hard2[sort_col], errors="coerce")
    hard2 = hard2.sort_values(sort_col, ascending=False).head(TOPK_HARD_PLOT)

    plt.figure(figsize=(10, 6))
    for _, row in hard2.iterrows():
        wid = row[col["well_uid"]]
        sub = df[df[col["well_uid"]] == wid].sort_values(col["cutoff"])
        plt.plot(sub[col["cutoff"]], sub["abs_err"], marker="o", linewidth=1, alpha=0.9)

    plt.axvline(EARLY_CUTOFF, linestyle="--")
    plt.axhline(TOL, linestyle="--")
    plt.title(f"Hard samples abs error trajectories (top {TOPK_HARD_PLOT})")
    plt.xlabel("Cycle cutoff")
    plt.ylabel("Abs error (cycles)")
    fig2 = OUT_FIG / "hard_samples_trajectories.png"
    plt.tight_layout()
    plt.savefig(fig2, dpi=200)
    plt.close()

    # (C) scatter: true Ct vs first_success_cutoff (late tendency)
    plt.figure(figsize=(8, 6))
    x = out["true_ct"].to_numpy()
    y = out["first_success_cutoff"].to_numpy()
    plt.scatter(x, y)
    plt.axhline(EARLY_CUTOFF, linestyle="--")
    plt.title("True Ct vs first success cutoff")
    plt.xlabel("True Ct")
    plt.ylabel("First cutoff where abs_err <= TOL")
    fig3 = OUT_FIG / "error_by_true_ct_scatter.png"
    plt.tight_layout()
    plt.savefig(fig3, dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # 5) Print concise report
    # ------------------------------------------------------------------
    early = grp_counts.get("EARLY_SUCCESS", 0)
    late = grp_counts.get("LATE_SUCCESS", 0)
    inde = grp_counts.get("INDETERMINATE", 0)

    print(f"[TOTAL] n_samples={n_total}  max_cutoff_in_file={int(max_cut)}")
    print(f"[RULE ] EARLY_CUTOFF={EARLY_CUTOFF}  TOL={TOL}")
    print(f"[GROUP] EARLY_SUCCESS={early} ({early/n_total*100:.1f}%)")
    print(f"[GROUP] LATE_SUCCESS ={late} ({late/n_total*100:.1f}%)")
    print(f"[GROUP] INDETERMINATE={inde} ({inde/n_total*100:.1f}%)")

    print(f"[SAVED] {out_path}")
    print(f"[SAVED] {hard_path}")
    print(f"[SAVED] {gs_path}")
    print(f"[FIG  ] {fig1}")
    print(f"[FIG  ] {fig2}")
    print(f"[FIG  ] {fig3}")


if __name__ == "__main__":
    main()
