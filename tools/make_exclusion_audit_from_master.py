import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def guess_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def normalize_well_id(x: str) -> str:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    m = re.search(r'([A-P])\s*[-_ ]?\s*0*([0-9]{1,2})$', s) or re.search(r'([A-P])\s*[-_ ]?\s*0*([0-9]{1,2})', s)
    if m:
        row = m.group(1)
        col = int(m.group(2))
        if 1 <= col <= 24:
            return f"{row}{col:02d}"
    return s


def compute_qc_flags(grp, cycle_col, y_col, cutoff):
    g = grp.sort_values(cycle_col)
    early = g[g[cycle_col] <= cutoff]
    if early.empty:
        return pd.Series({"qc_flat": False, "qc_noisy": False, "qc_detail": ""})

    y = early[y_col].astype(float).values
    dyn = float(np.nanmax(y) - np.nanmin(y))
    mean = float(np.nanmean(y))
    std = float(np.nanstd(y))
    safe_mean = max(abs(mean), 1e-6)
    cv = std / safe_mean

    # NOTE: fluorescence scale에 따라 튜닝 가능 (일단 audit 목적의 휴리스틱)
    flat = dyn < 0.05
    noisy = cv > 0.5

    detail = f"dyn={dyn:.4g}, mean={mean:.4g}, cv={cv:.3g}"
    return pd.Series({"qc_flat": flat, "qc_noisy": noisy, "qc_detail": detail})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--df_long", required=True, help="Path to master_long (.parquet/.csv)")
    ap.add_argument("--cutoff", type=int, default=24)
    ap.add_argument("--out_dir", default="reports/exclusions")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_path = Path(args.df_long)
    if df_path.suffix.lower() == ".parquet":
        df_long = pd.read_parquet(df_path)
    else:
        df_long = pd.read_csv(df_path)

    run_col = guess_col(df_long, ["run_id", "run", "plate_id", "plate", "experiment_id"])
    well_col = guess_col(df_long, ["well_id", "well", "Well", "Well Position", "well_position"])
    cycle_col = guess_col(df_long, ["cycle", "Cycle", "cycle_num", "cycle_number"])
    y_col = guess_col(df_long, ["fluor", "fluorescence", "rfu", "RFU", "signal", "value"])
    cq_col = guess_col(df_long, ["Cq", "cq", "Ct", "ct", "true_ct"])

    if well_col is None or cycle_col is None or y_col is None:
        raise ValueError(f"Cannot find required columns. run={run_col}, well={well_col}, cycle={cycle_col}, y={y_col}")

    if cq_col is None:
        raise ValueError("Cannot find label column (Cq/Ct). Your master_long should include 'Cq'.")

    df_long = df_long.copy()
    df_long["_well_norm"] = df_long[well_col].map(normalize_well_id)
    df_long["_run_norm"] = df_long[run_col].astype(str).str.strip() if run_col else "NA"

    # Well universe
    wells_all = (
        df_long[["_run_norm", "_well_norm"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"_run_norm": "run_id", "_well_norm": "well_id"})
    )

    # Max cycle coverage
    max_cycle = (
        df_long.dropna(subset=["_well_norm"])
        .groupby(["_run_norm", "_well_norm"])[cycle_col]
        .max()
        .reset_index()
        .rename(columns={"_run_norm": "run_id", "_well_norm": "well_id", cycle_col: "max_cycle"})
    )
    wells_all = wells_all.merge(max_cycle, on=["run_id", "well_id"], how="left")

    # True Ct/Cq label from master_long
    labels_key = (
        df_long.dropna(subset=["_well_norm"])
        .groupby(["_run_norm", "_well_norm"])[cq_col]
        .apply(lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan)
        .reset_index()
        .rename(columns={"_run_norm": "run_id", "_well_norm": "well_id", cq_col: "true_ct"})
    )
    wells_all = wells_all.merge(labels_key, on=["run_id", "well_id"], how="left")

    # QC flags (optional) - disable by default for audit stability
    wells_all["qc_flat"] = False
    wells_all["qc_noisy"] = False
    wells_all["qc_detail"] = ""


    # Reasons (priority)
    cutoff = args.cutoff
    wells_all["reason"] = "INCLUDED"
    wells_all["detail"] = ""

    # Missing label (Cq missing)
    mask = wells_all["true_ct"].isna()
    wells_all.loc[mask, "reason"] = "EXCLUDE_NO_LABEL"
    wells_all.loc[mask, "detail"] = f"{cq_col} missing"

    # Cycle insufficient
    mask = wells_all["reason"].eq("INCLUDED") & (wells_all["max_cycle"] < cutoff)
    wells_all.loc[mask, "reason"] = "EXCLUDE_CYCLE_SHORT"
    wells_all.loc[mask, "detail"] = wells_all.loc[mask, "max_cycle"].apply(lambda x: f"max_cycle={x} < cutoff={cutoff}")

    # QC flat/noisy
    #mask = wells_all["reason"].eq("INCLUDED") & (wells_all["qc_flat"] == True)
    #wells_all.loc[mask, "reason"] = "EXCLUDE_QC_FLAT"
    #wells_all.loc[mask, "detail"] = wells_all.loc[mask, "qc_detail"].fillna("").map(lambda s: f"flat ({s})")

    #mask = wells_all["reason"].eq("INCLUDED") & (wells_all["qc_noisy"] == True)
    #wells_all.loc[mask, "reason"] = "EXCLUDE_QC_NOISY"
    #wells_all.loc[mask, "detail"] = wells_all.loc[mask, "qc_detail"].fillna("").map(lambda s: f"noisy ({s})")

    # Outputs
    total = len(wells_all)
    included = int((wells_all["reason"] == "INCLUDED").sum())
    excluded = total - included

    summary = (
        wells_all.assign(count=1)
        .groupby("reason")["count"]
        .sum()
        .reset_index()
        .sort_values("count", ascending=False)
    )
    summary["ratio"] = summary["count"] / total
    summary.insert(0, "cutoff", cutoff)
    summary.insert(1, "total_wells", total)
    summary.insert(2, "included", included)
    summary.insert(3, "excluded", excluded)

    detail = wells_all[wells_all["reason"] != "INCLUDED"].copy()
    detail.insert(0, "cutoff", cutoff)

    summary_path = out_dir / f"exclusion_summary_cutoff{cutoff}.csv"
    detail_path = out_dir / f"exclusions_cutoff{cutoff}.csv"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Wrote summary: {summary_path}")
    print(f"[OK] Wrote detail : {detail_path}")
    print(f"[INFO] total={total}, included={included}, excluded={excluded}")


if __name__ == "__main__":
    main()
