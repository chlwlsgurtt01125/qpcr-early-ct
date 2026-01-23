# -*- coding: utf-8 -*-
"""
Run sanity check:
- Baseline-correct + normalize amplification curves
- Recompute Ct via threshold crossing
- Compare recomputed Ct vs provided Cq (label sanity)
- Produce plots + CSV summary

Usage:
  python core/step2_run_sanity_check.py \
    --amp data/raw/run3_amp.csv \
    --cq  data/raw/run3_cq.csv \
    --run_name run3

  python core/step2_run_sanity_check.py \
    --amp data/raw/run7_amp.csv \
    --cq  data/raw/run7_cq.csv \
    --run_name run7
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def norm_well_name(x: str) -> str:
    """
    Normalize well name variants:
      C3 -> C03
      D10 -> D10 (already ok)
      E5 -> E05
      D03 -> D03 (ok)
    Also strips spaces/underscores.
    """
    if x is None:
        return ""
    s = str(x).strip()
    s = s.replace("_", "").replace(" ", "")
    s = s.upper()

    # match like C3, C03, D10, E5, etc.
    m = re.match(r"^([A-H])0*([0-9]{1,2})$", s)
    if not m:
        return s
    row = m.group(1)
    col = int(m.group(2))
    return f"{row}{col:02d}"


def baseline_correct_and_normalize(y: np.ndarray, baseline_cycles: int = 5, eps: float = 1e-9):
    """
    Baseline = median of first baseline_cycles points.
    y_bc = y - baseline
    y_norm = y_bc / (max(y_bc) + eps)
    """
    y = np.asarray(y, dtype=float)
    base = np.median(y[:baseline_cycles])
    y_bc = y - base
    denom = np.max(y_bc) + eps
    y_norm = y_bc / denom
    return y_bc, y_norm, base


def ct_threshold_crossing(cycles: np.ndarray, y_bc: np.ndarray, frac: float = 0.2, min_span: float = 50.0):
    """
    Compute Ct as first cycle where y_bc >= threshold,
    threshold = frac * max(y_bc), but also requires max(y_bc) >= min_span.
    Returns (ct, threshold, max_y, status)
      status: OK | NO_AMP (too small span) | NO_CROSS (never crosses)
    """
    cycles = np.asarray(cycles, dtype=float)
    y_bc = np.asarray(y_bc, dtype=float)

    max_y = np.max(y_bc)
    if not np.isfinite(max_y):
        return np.nan, np.nan, max_y, "BAD"
    if max_y < min_span:
        thr = frac * max_y
        return np.nan, thr, max_y, "NO_AMP"

    thr = frac * max_y
    idx = np.where(y_bc >= thr)[0]
    if len(idx) == 0:
        return np.nan, thr, max_y, "NO_CROSS"

    i = int(idx[0])
    # linear interpolation for slightly nicer Ct
    if i == 0:
        return float(cycles[0]), thr, max_y, "OK"
    x0, x1 = cycles[i - 1], cycles[i]
    y0, y1 = y_bc[i - 1], y_bc[i]
    if y1 == y0:
        return float(cycles[i]), thr, max_y, "OK"
    ct = x0 + (thr - y0) * (x1 - x0) / (y1 - y0)
    return float(ct), thr, max_y, "OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", required=True, help="Amplification wide CSV: Cycle + well columns")
    ap.add_argument("--cq", required=True, help="Cq CSV: Well,Cq")
    ap.add_argument("--run_name", required=True, help="Run name for outputs")
    ap.add_argument("--baseline_cycles", type=int, default=5, help="Baseline cycles count (default=5)")
    ap.add_argument("--thr_frac", type=float, default=0.2, help="Threshold fraction of max(y_bc) (default=0.2)")
    ap.add_argument("--min_span", type=float, default=50.0, help="Minimum (max(y_bc)) to call amplified (default=50)")
    ap.add_argument("--preview_wells", type=int, default=8, help="How many wells to plot in preview (default=8)")
    args = ap.parse_args()

    out_metrics_dir = "data/metrics"
    out_fig_dir = "data/figures"
    ensure_dir(out_metrics_dir)
    ensure_dir(out_fig_dir)

    # --- load amplification
    amp = pd.read_csv(args.amp)
    if "Cycle" not in amp.columns:
        raise ValueError(f"'Cycle' column not found in {args.amp}. Columns={list(amp.columns)[:10]}...")

    cycles = amp["Cycle"].to_numpy()
    well_cols = [c for c in amp.columns if c != "Cycle"]

    # normalize col names
    col_map = {c: norm_well_name(c) for c in well_cols}
    # if duplicates after normalization, keep original but warn-like behavior by suffix
    rev = {}
    for orig, normed in col_map.items():
        if normed in rev:
            # make unique
            k = 2
            nn = f"{normed}_{k}"
            while nn in rev:
                k += 1
                nn = f"{normed}_{k}"
            col_map[orig] = nn
        rev[col_map[orig]] = orig

    amp2 = amp.rename(columns=col_map)

    # --- load cq
    cq = pd.read_csv(args.cq)
    if not {"Well", "Cq"}.issubset(set(cq.columns)):
        raise ValueError(f"Cq file must have columns Well,Cq. Got {list(cq.columns)}")
    cq["Well_norm"] = cq["Well"].apply(norm_well_name)
    cq["Cq"] = pd.to_numeric(cq["Cq"], errors="coerce")

    # --- per well compute
    rows = []
    for w in [c for c in amp2.columns if c != "Cycle"]:
        y = amp2[w].to_numpy(dtype=float)
        y_bc, y_norm, base = baseline_correct_and_normalize(y, baseline_cycles=args.baseline_cycles)
        ct_hat, thr, max_y, status = ct_threshold_crossing(
            cycles, y_bc, frac=args.thr_frac, min_span=args.min_span
        )

        cq_row = cq.loc[cq["Well_norm"] == w, "Cq"]
        cq_val = float(cq_row.iloc[0]) if len(cq_row) > 0 else np.nan

        rows.append(
            {
                "run": args.run_name,
                "well": w,
                "cq_provided": cq_val,
                "ct_recomputed": ct_hat,
                "delta_ct": (ct_hat - cq_val) if np.isfinite(ct_hat) and np.isfinite(cq_val) else np.nan,
                "status": status,
                "baseline_median": base,
                "max_y_bc": max_y,
                "threshold": thr,
            }
        )

    out = pd.DataFrame(rows)

    # flags
    out["flag_cq_suspicious"] = False
    # e.g., Cq < 5 or > 45 are suspicious (domain rule)
    out.loc[(out["cq_provided"].notna()) & ((out["cq_provided"] < 5) | (out["cq_provided"] > 45)), "flag_cq_suspicious"] = True

    out["flag_large_mismatch"] = False
    out.loc[(out["delta_ct"].abs() >= 5) & (out["delta_ct"].notna()), "flag_large_mismatch"] = True

    # --- save table
    out_path = os.path.join(out_metrics_dir, f"{args.run_name}_sanity_check.csv")
    out.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")
    print("[INFO] status counts:")
    print(out["status"].value_counts(dropna=False))
    print("[INFO] suspicious Cq count:", int(out["flag_cq_suspicious"].sum()))
    print("[INFO] large mismatch (|ΔCt|>=5) count:", int(out["flag_large_mismatch"].sum()))

    # --- plot 1: provided Cq vs recomputed Ct scatter
    fig = plt.figure(figsize=(8, 6))
    ok = out[(out["cq_provided"].notna()) & (out["ct_recomputed"].notna())]
    plt.scatter(ok["cq_provided"], ok["ct_recomputed"])
    plt.xlabel("Provided Cq")
    plt.ylabel("Recomputed Ct (threshold crossing)")
    plt.title(f"{args.run_name}: Provided Cq vs Recomputed Ct")
    # y=x line
    if len(ok) > 0:
        mn = float(np.nanmin([ok["cq_provided"].min(), ok["ct_recomputed"].min()]))
        mx = float(np.nanmax([ok["cq_provided"].max(), ok["ct_recomputed"].max()]))
        plt.plot([mn, mx], [mn, mx], linestyle="--")
    fig_path1 = os.path.join(out_fig_dir, f"{args.run_name}_cq_vs_ct_recomputed.png")
    plt.tight_layout()
    plt.savefig(fig_path1, dpi=200)
    plt.close(fig)
    print(f"[FIG] {fig_path1}")

    # --- plot 2: delta histogram
    fig = plt.figure(figsize=(8, 6))
    if len(ok) > 0:
        plt.hist(ok["delta_ct"].dropna().values, bins=30)
    plt.xlabel("Recomputed Ct - Provided Cq")
    plt.ylabel("Count")
    plt.title(f"{args.run_name}: ΔCt distribution")
    fig_path2 = os.path.join(out_fig_dir, f"{args.run_name}_delta_ct_hist.png")
    plt.tight_layout()
    plt.savefig(fig_path2, dpi=200)
    plt.close(fig)
    print(f"[FIG] {fig_path2}")

    # --- plot 3: preview normalized curves for a few wells
    # choose wells: suspicious first, then random
    preview = out.sort_values(
        by=["flag_large_mismatch", "flag_cq_suspicious", "cq_provided"],
        ascending=[False, False, True],
    )["well"].tolist()
    preview = preview[: max(1, args.preview_wells)]

    fig = plt.figure(figsize=(10, 6))
    for w in preview:
        y = amp2[w].to_numpy(dtype=float)
        y_bc, y_norm, _ = baseline_correct_and_normalize(y, baseline_cycles=args.baseline_cycles)
        plt.plot(cycles, y_norm, label=w)

    plt.xlabel("Cycle")
    plt.ylabel("Normalized (baseline-corrected)")
    plt.title(f"{args.run_name}: normalized curves (preview)")
    plt.legend(fontsize=8, ncol=2)
    fig_path3 = os.path.join(out_fig_dir, f"{args.run_name}_normalized_curve_preview.png")
    plt.tight_layout()
    plt.savefig(fig_path3, dpi=200)
    plt.close(fig)
    print(f"[FIG] {fig_path3}")

    # --- quick verdict print
    n = len(out)
    n_ok = int((out["status"] == "OK").sum())
    n_mis = int(out["flag_large_mismatch"].sum())
    print(f"[SUMMARY] wells={n}, OK_amp={n_ok}, large_mismatch(|ΔCt|>=5)={n_mis}")

    if n_mis >= max(2, int(0.3 * n)):
        print("[VERDICT] 라벨(Cq) 매핑/정합 문제 가능성이 큼 (불일치가 매우 많음).")
    else:
        print("[VERDICT] 라벨은 대체로 일관적일 가능성. 전처리/스케일 mismatch 쪽을 더 의심.")


if __name__ == "__main__":
    main()
