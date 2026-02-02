# tools/make_exclusion_audit.py
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


def guess_col(df, candidates):
    """Return first candidate column present in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    # try case-insensitive
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def normalize_well_id(x: str) -> str:
    """Normalize well id to A01..H12 style when possible."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()

    # common forms: A1, A01, A-01, A 01, well A01
    m = re.search(r'([A-P])\s*[-_ ]?\s*0*([0-9]{1,2})$', s)
    if not m:
        # try anywhere in string
        m = re.search(r'([A-P])\s*[-_ ]?\s*0*([0-9]{1,2})', s)
    if m:
        row = m.group(1)
        col = int(m.group(2))
        if 1 <= col <= 24:
            return f"{row}{col:02d}"
    return s


def is_control_like(text: str) -> bool:
    if pd.isna(text):
        return False
    s = str(text).strip().upper()
    # adjust keywords to your lab/company conventions
    keywords = ["NTC", "CONTROL", "CTRL", "STD", "STANDARD", "POS", "NEG", "BLANK"]
    return any(k in s for k in keywords)


def compute_qc_flags(grp, cycle_col, y_col, cutoff):
    """
    Simple QC heuristics:
    - flat/no_amp: very low dynamic range in early window
    - noisy: high CV in early window (after shifting to avoid near-zero mean issues)
    """
    g = grp.sort_values(cycle_col)
    early = g[g[cycle_col] <= cutoff]
    if early.empty:
        return pd.Series({"qc_flat": False, "qc_noisy": False, "qc_detail": ""})

    y = early[y_col].astype(float).values
    dyn = float(np.nanmax(y) - np.nanmin(y))
    mean = float(np.nanmean(y))
    std = float(np.nanstd(y))

    # heuristics (tune later)
    flat = dyn < 0.05  # depends on fluorescence scale; adjust if needed
    # stabilize mean to avoid blow-up when mean ~0
    safe_mean = max(abs(mean), 1e-6)
    cv = std / safe_mean
    noisy = cv > 0.5  # adjust if needed

    detail = f"dyn={dyn:.4g}, mean={mean:.4g}, cv={cv:.3g}"
    return pd.Series({"qc_flat": flat, "qc_noisy": noisy, "qc_detail": detail})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--df_long", required=True, help="Path to curve long file (.parquet/.csv)")
    ap.add_argument("--labels", required=True, help="Path to Ct labels file (.csv/.xlsx/.parquet)")
    ap.add_argument("--cutoff", type=int, default=24)
    ap.add_argument("--out_dir", default="reports/exclusions")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load df_long
    df_path = Path(args.df_long)
    if df_path.suffix.lower() == ".parquet":
        df_long = pd.read_parquet(df_path)
    else:
        df_long = pd.read_csv(df_path)

    # ---- Load labels
    lab_path = Path(args.labels)
    if lab_path.suffix.lower() in [".xlsx", ".xls"]:
        labels = pd.read_excel(lab_path)
    elif lab_path.suffix.lower() == ".parquet":
        labels = pd.read_parquet(lab_path)
    else:
        labels = pd.read_csv(lab_path)

    # ---- Guess columns
    run_col = guess_col(df_long, ["run_id", "run", "plate_id", "plate", "experiment_id"])
    well_col = guess_col(df_long, ["well_id", "well", "Well", "Well Position", "well_position"])
    cycle_col = guess_col(df_long, ["cycle", "Cycle", "cycle_num", "cycle_number"])
    y_col = guess_col(df_long, ["fluor", "fluorescence", "rfu", "RFU", "signal", "value"])

    if well_col is None or cycle_col is None or y_col is None:
        raise ValueError(f"Cannot find required columns in df_long. "
                         f"Found run={run_col}, well={well_col}, cycle={cycle_col}, y={y_col}. "
                         f"Please rename columns or adjust candidates.")

    # labels columns
    lab_run_col = guess_col(labels, ["run_id", "run", "plate_id", "plate", "experiment_id"])
    lab_well_col = guess_col(labels, ["well_id", "well", "Well", "Well Position", "well_position"])
    ct_col = guess_col(labels, ["true_ct", "ct", "Ct", "Cq", "cq", "true_cq"])

    if lab_well_col is None or ct_col is None:
        raise ValueError(f"Cannot find required columns in labels. "
                         f"Found run={lab_run_col}, well={lab_well_col}, ct={ct_col}. "
                         f"Please rename columns or adjust candidates.")

    # optional: control annotation column
    ann_col = guess_col(labels, ["sample", "sample_name", "target", "type", "comment", "notes"])

    # ---- Normalize keys
    df_long = df_long.copy()
    labels = labels.copy()

    df_long["_well_norm"] = df_long[well_col].map(normalize_well_id)
    labels["_well_norm"] = labels[lab_well_col].map(normalize_well_id)

    if run_col is None:
        df_long["_run_norm"] = "NA"
    else:
        df_long["_run_norm"] = df_long[run_col].astype(str).str.strip()

    if lab_run_col is None:
        labels["_run_norm"] = "NA"
    else:
        labels["_run_norm"] = labels[lab_run_col].astype(str).str.strip()

    # ---- Build well universe from df_long
    # A "well" is defined by (run, well)
    wells_all = (
        df_long[["_run_norm", "_well_norm"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"_run_norm": "run_id", "_well_norm": "well_id"})
    )

    # ---- Cycle coverage (<= cutoff)
    max_cycle = (
        df_long.dropna(subset=["_well_norm"])
        .groupby(["_run_norm", "_well_norm"])[cycle_col]
        .max()
        .reset_index()
        .rename(columns={"_run_norm": "run_id", "_well_norm": "well_id", cycle_col: "max_cycle"})
    )
    wells_all = wells_all.merge(max_cycle, on=["run_id", "well_id"], how="left")

    # ---- Label presence
    labels_key = labels[["_run_norm", "_well_norm", ct_col]].rename(
        columns={"_run_norm": "run_id", "_well_norm": "well_id", ct_col: "true_ct"}
    )
    wells_all = wells_all.merge(labels_key, on=["run_id", "well_id"], how="left")

    # ---- Control flag (if we have annotation)
    if ann_col is not None:
        ann_key = labels[["_run_norm", "_well_norm", ann_col]].rename(
            columns={"_run_norm": "run_id", "_well_norm": "well_id", ann_col: "annotation"}
        )
        wells_all = wells_all.merge(ann_key, on=["run_id", "well_id"], how="left")
        wells_all["is_control"] = wells_all["annotation"].map(is_control_like)
    else:
        wells_all["annotation"] = ""
        wells_all["is_control"] = False

    # ---- QC flags (optional but good)
    qc = (
        df_long.dropna(subset=["_well_norm"])
        .groupby(["_run_norm", "_well_norm"])
        .apply(lambda g: compute_qc_flags(g, cycle_col, y_col, args.cutoff))
        .reset_index()
        .rename(columns={"_run_norm": "run_id", "_well_norm": "well_id"})
    )
    wells_all = wells_all.merge(qc, on=["run_id", "well_id"], how="left")

    # ---- Determine exclusion reason with priority
    cutoff = args.cutoff
    wells_all["reason"] = "INCLUDED"
    wells_all["detail"] = ""

    # 1) Control/NTC exclusion (if flagged)
    mask = wells_all["is_control"] == True
    wells_all.loc[mask, "reason"] = "EXCLUDE_CONTROL"
    wells_all.loc[mask, "detail"] = "control/NTC/STD flagged"

    # 2) Missing label
    mask = wells_all["reason"].eq("INCLUDED") & wells_all["true_ct"].isna()
    wells_all.loc[mask, "reason"] = "EXCLUDE_NO_LABEL"
    wells_all.loc[mask, "detail"] = "true_ct missing (cannot compute error)"

    # 3) Cycle insufficient
    mask = wells_all["reason"].eq("INCLUDED") & (wells_all["max_cycle"] < cutoff)
    wells_all.loc[mask, "reason"] = "EXCLUDE_CYCLE_SHORT"
    wells_all.loc[mask, "detail"] = wells_all.loc[mask, "max_cycle"].apply(lambda x: f"max_cycle={x} < cutoff={cutoff}")

    # 4) QC flat/noisy (only if still included)
    mask = wells_all["reason"].eq("INCLUDED") & (wells_all["qc_flat"] == True)
    wells_all.loc[mask, "reason"] = "EXCLUDE_QC_FLAT"
    wells_all.loc[mask, "detail"] = wells_all.loc[mask, "qc_detail"].fillna("").map(lambda s: f"flat ({s})")

    mask = wells_all["reason"].eq("INCLUDED") & (wells_all["qc_noisy"] == True)
    wells_all.loc[mask, "reason"] = "EXCLUDE_QC_NOISY"
    wells_all.loc[mask, "detail"] = wells_all.loc[mask, "qc_detail"].fillna("").map(lambda s: f"noisy ({s})")

    # ---- Create outputs
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

    # detail list: only excluded
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
