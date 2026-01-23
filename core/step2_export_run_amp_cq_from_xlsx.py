# -*- coding: utf-8 -*-
"""
Export amp/cq CSVs from the original run Excel (.xlsx).

Expected:
- One sheet contains amplification table with first column like "Cycle" and remaining columns = wells (C3, D4, ...).
- Another sheet contains Cq table with columns like ["Well", "Cq"] (or similar).

Usage:
python core/step2_export_run_amp_cq_from_xlsx.py \
  --xlsx "data/raw/3번 데이터_3_250806_gist_RNA_MOPSS_14th.xlsx" \
  --run_name run3

python core/step2_export_run_amp_cq_from_xlsx.py \
  --xlsx "data/raw/7번데이터_7_251209_GIST_RNA_MOPSS_18th_UMI_RNA.xlsx" \
  --run_name run7
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


ROOT = Path("/home/cphotonic/qpcr_v2")
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _find_amp_sheet(xls: pd.ExcelFile) -> str:
    # Heuristic: sheet where first col contains "Cycle" (case-insensitive)
    for s in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=s, nrows=5)
        cols = [str(c).strip().lower() for c in df.columns]
        if len(cols) >= 2 and ("cycle" in cols[0]):
            return s
    # fallback: first sheet
    return xls.sheet_names[0]


def _find_cq_sheet(xls: pd.ExcelFile) -> str | None:
    # Heuristic: sheet that has columns containing 'cq' and 'well'
    for s in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=s, nrows=20)
        cols = [str(c).strip().lower() for c in df.columns]
        has_cq = any("cq" in c for c in cols)
        has_well = any("well" in c for c in cols) or any(c in ("pos", "position") for c in cols)
        if has_cq and has_well:
            return s
    return None


def _clean_amp(df: pd.DataFrame) -> pd.DataFrame:
    # normalize col names
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # find cycle column (first col or any col named Cycle)
    cycle_col = None
    for c in df.columns:
        if str(c).strip().lower() == "cycle":
            cycle_col = c
            break
    if cycle_col is None:
        cycle_col = df.columns[0]

    # drop empty rows
    df = df.dropna(how="all")
    # keep rows where cycle is numeric
    df[cycle_col] = pd.to_numeric(df[cycle_col], errors="coerce")
    df = df.dropna(subset=[cycle_col])
    df[cycle_col] = df[cycle_col].astype(int)

    # ensure numeric for well columns
    for c in df.columns:
        if c == cycle_col:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # sort by cycle
    df = df.sort_values(cycle_col).reset_index(drop=True)

    # rename cycle col to exact "Cycle"
    if cycle_col != "Cycle":
        df = df.rename(columns={cycle_col: "Cycle"})
    return df


def _clean_cq(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # find Well column
    well_col = None
    cq_col = None
    for c in df.columns:
        lc = c.lower()
        if lc == "well":
            well_col = c
        if "cq" in lc:
            cq_col = c
    if well_col is None:
        # try common alternatives
        for c in df.columns:
            if c.lower() in ("pos", "position"):
                well_col = c
                break
    if cq_col is None:
        raise ValueError(f"Cannot find Cq column in cq sheet columns={list(df.columns)}")
    if well_col is None:
        raise ValueError(f"Cannot find Well column in cq sheet columns={list(df.columns)}")

    out = df[[well_col, cq_col]].rename(columns={well_col: "Well", cq_col: "Cq"})
    out = out.dropna(how="any")
    out["Well"] = out["Well"].astype(str).str.strip()
    out["Cq"] = pd.to_numeric(out["Cq"], errors="coerce")
    out = out.dropna(subset=["Cq"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Path to original run .xlsx")
    ap.add_argument("--run_name", required=True, help="e.g., run3, run7")
    args = ap.parse_args()

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Not found: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)

    amp_sheet = _find_amp_sheet(xls)
    cq_sheet = _find_cq_sheet(xls)

    print(f"[INFO] xlsx={xlsx_path}")
    print(f"[INFO] sheets={xls.sheet_names}")
    print(f"[INFO] detected amp_sheet={amp_sheet}")
    print(f"[INFO] detected cq_sheet={cq_sheet}")

    amp_df = pd.read_excel(xls, sheet_name=amp_sheet)
    amp_df = _clean_amp(amp_df)

    if cq_sheet is None:
        raise ValueError(
            "Could not detect Cq sheet automatically. "
            "Open the xlsx and check which sheet contains Well/Cq table."
        )
    cq_df = pd.read_excel(xls, sheet_name=cq_sheet)
    cq_df = _clean_cq(cq_df)

    amp_out = RAW_DIR / f"{args.run_name}_amp.csv"
    cq_out = RAW_DIR / f"{args.run_name}_cq.csv"

    amp_df.to_csv(amp_out, index=False, encoding="utf-8-sig")
    cq_df.to_csv(cq_out, index=False, encoding="utf-8-sig")

    print(f"[SAVED] {amp_out}")
    print(f"[SAVED] {cq_out}")
    print(f"[AMP ] shape={amp_df.shape} cols(head)={list(amp_df.columns)[:8]}")
    print(f"[CQ  ] n={len(cq_df)} well examples={cq_df['Well'].head(5).tolist()}")


if __name__ == "__main__":
    main()
