# ingest/ingest_excel.py
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingest.well import normalize_well
from ingest.excel_detect import detect_sheets, pick_best

@dataclass
class IngestResult:
    run_id: str
    file_path: str
    n_amp_rows: int
    n_wells_amp: int
    n_cq: int
    n_merged: int
    warnings: list[str]

def _infer_run_id(path: Path) -> str:
    # Use folder name if present (data_1_xxx), else file stem
    # Sanitize into a stable id
    s = path.stem
    parent = path.parent.name
    base = parent if parent and parent != "raw" else s
    base = re.sub(r"\s+", "_", base.strip())
    base = re.sub(r"[^A-Za-z0-9_\-]+", "_", base)
    return base
    
def _find_header_row_for_amp(xls: pd.ExcelFile, sheet: str, max_rows: int = 80) -> int | None:
    """
    Find a row index that looks like an amplification header:
      contains 'Cycle' and at least 2 well-like labels (e.g., E5, E07, B03)
    Returns 0-indexed row number to use as header=...
    """
    probe = xls.parse(sheet, header=None, nrows=max_rows, dtype=str)
    probe = probe.fillna("")

    def looks_well(s: str) -> bool:
        return bool(re.match(r"^[A-Za-z]{1,2}\s*0*\d{1,2}$", str(s).strip()))

    for r in range(min(max_rows, probe.shape[0])):
        row = [str(x).strip() for x in probe.iloc[r, :].tolist()]
        has_cycle = any("cycle" in x.lower() for x in row)
        if not has_cycle:
            continue
        n_wells = sum(looks_well(x) for x in row)
        if n_wells >= 2:
            return r
    return None

def _read_amp_sheet(xls: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    # 1) normal read
    df = xls.parse(sheet)
    df.columns = [str(c).strip() for c in df.columns]

    # If we cannot find any 'cycle' column in current header, try to locate header row inside sheet.
    if not any("cycle" in str(c).lower() for c in df.columns):
        hdr = _find_header_row_for_amp(xls, sheet)
        if hdr is not None:
            df = xls.parse(sheet, header=hdr)
            df.columns = [str(c).strip() for c in df.columns]

    # --- helper ---
    def looks_well(c: str) -> bool:
        return bool(re.match(r"^[A-Za-z]{1,2}\s*0*\d{1,2}$", str(c).strip()))

    def looks_cycle(c: str) -> bool:
        s = str(c).strip()
        return bool(re.match(r"^\d+$", s)) and 1 <= int(s) <= 100

    # Try Layout A: cycle column + many well columns
    cycle_candidates = [c for c in df.columns if "cycle" in c.lower()]
    well_cols = [c for c in df.columns if looks_well(c)]
    if cycle_candidates and len(well_cols) >= 2:
        cycle_col = cycle_candidates[0]
        keep = [cycle_col] + well_cols
        dfA = df[keep].copy()
        dfA.rename(columns={cycle_col: "Cycle"}, inplace=True)

        long = dfA.melt(id_vars=["Cycle"], var_name="Well_raw", value_name="Fluor")
        long["Well"] = long["Well_raw"].map(normalize_well)
        long.drop(columns=["Well_raw"], inplace=True)
        long = long.dropna(subset=["Well"])
        long["Cycle"] = pd.to_numeric(long["Cycle"], errors="coerce").astype("Int64")
        long["Fluor"] = pd.to_numeric(long["Fluor"], errors="coerce")
        long = long.dropna(subset=["Cycle", "Fluor"])
        long["Cycle"] = long["Cycle"].astype(int)
        return long

    # Try Layout B: well column + many numeric cycle columns
    # find a well-like column OR a column whose values look like wells
    well_col = None
    for c in df.columns[:6]:
        if "well" in c.lower() or "sample" in c.lower() or "pos" in c.lower():
            well_col = c
            break

    if well_col is None:
        # value-based detection
        for c in df.columns[:6]:
            s = df[c].astype(str).head(50)
            if s.str.match(r"^[A-Za-z]{1,2}\s*0*\d{1,2}$").mean() > 0.6:
                well_col = c
                break

    cycle_cols = [c for c in df.columns if looks_cycle(c)]
    if well_col is not None and len(cycle_cols) >= 8:
        dfB = df[[well_col] + cycle_cols].copy()
        dfB.rename(columns={well_col: "Well_raw"}, inplace=True)
        dfB["Well"] = dfB["Well_raw"].map(normalize_well)
        dfB.drop(columns=["Well_raw"], inplace=True)
        dfB = dfB.dropna(subset=["Well"])

        long = dfB.melt(id_vars=["Well"], var_name="Cycle", value_name="Fluor")
        long["Cycle"] = pd.to_numeric(long["Cycle"], errors="coerce").astype("Int64")
        long["Fluor"] = pd.to_numeric(long["Fluor"], errors="coerce")
        long = long.dropna(subset=["Cycle", "Fluor"])
        long["Cycle"] = long["Cycle"].astype(int)
        return long

    raise ValueError(f"No recognized amplification table layout in sheet '{sheet}'")


def _read_cq_sheet(xls: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    df = xls.parse(sheet)
    df.columns = [str(c).strip() for c in df.columns]
    lc = {c.lower(): c for c in df.columns}

    # well column candidates
    well_col = None
    for key in ["well", "well position", "position", "sample", "sample name"]:
        for c in df.columns:
            if key in c.lower():
                well_col = c
                break
        if well_col:
            break

    if well_col is None:
        # fallback: find first column that looks like wells in values
        for c in df.columns[:6]:
            series = df[c].astype(str).head(50)
            if series.str.match(r"^[A-Za-z]{1,2}\s*0*\d{1,2}$").mean() > 0.6:
                well_col = c
                break

    if well_col is None:
        raise ValueError(f"Cannot find well column in cq sheet '{sheet}'")

    # cq column candidates
    cq_col = None
    for c in df.columns:
        n = c.lower().strip()
        if n == "cq" or n == "ct" or "cq" in n or "ct" in n:
            cq_col = c
            break
    if cq_col is None:
        raise ValueError(f"Cannot find Cq/Ct column in cq sheet '{sheet}'")

    out = df[[well_col, cq_col]].copy()
    out.rename(columns={well_col: "Well_raw", cq_col: "Cq"}, inplace=True)
    out["Well"] = out["Well_raw"].map(normalize_well)
    out.drop(columns=["Well_raw"], inplace=True)
    out["Cq"] = pd.to_numeric(out["Cq"], errors="coerce")
    out = out.dropna(subset=["Well", "Cq"])
    return out

def ingest_one_excel(file_path: Path, channel: str = "SYBR") -> tuple[pd.DataFrame, IngestResult]:
    warnings: list[str] = []
    run_id = _infer_run_id(file_path)

    xls = pd.ExcelFile(file_path)
    hits = detect_sheets(xls)
    amp_sheet = pick_best(hits, "amp")
    cq_sheet = pick_best(hits, "cq")

    if amp_sheet is None:
        # fallback: try every sheet until one parses as amplification table
        for sh in xls.sheet_names:
            try:
                _ = _read_amp_sheet(xls, sh)
                amp_sheet = sh
                warnings.append(
                    f"Amp sheet autodetect failed; fallback selected sheet='{sh}'"
                )
                break
            except Exception:
                continue

    if amp_sheet is None:
        raise RuntimeError(
            f"[{run_id}] Cannot detect amplification sheet in {file_path.name}"
        )


    if cq_sheet is None:
        warnings.append("Cq sheet not detected; will ingest amplification only (Cq missing).")

    amp_long = _read_amp_sheet(xls, amp_sheet)
    amp_long["run_id"] = run_id
    amp_long["channel"] = channel

    cq_df = None
    if cq_sheet is not None:
        try:
            cq_df = _read_cq_sheet(xls, cq_sheet)
        except Exception as e:
            warnings.append(f"Cq read failed: {e}. Proceeding without Cq.")
            cq_df = None

    if cq_df is not None:
        cq_df = cq_df.copy()
        cq_df["run_id"] = run_id
        merged = amp_long.merge(cq_df[["run_id", "Well", "Cq"]], on=["run_id", "Well"], how="left")
    else:
        merged = amp_long.copy()
        merged["Cq"] = np.nan
        
    merged["well_uid"] = merged["run_id"].astype(str) + "__" + merged["Well"].astype(str)

    # basic sanity
    n_wells_amp = merged["Well"].nunique()
    n_amp_rows = len(merged)
    n_cq = 0 if cq_df is None else len(cq_df)
    n_merged = merged["Cq"].notna().sum()

    res = IngestResult(
        run_id=run_id,
        file_path=str(file_path),
        n_amp_rows=n_amp_rows,
        n_wells_amp=n_wells_amp,
        n_cq=n_cq,
        n_merged=n_merged,
        warnings=warnings,
    )
    return merged, res

def find_excel_files(raw_dir: Path) -> list[Path]:
    exts = {".xlsx", ".xlsm", ".xls"}  # openpyxl supports xlsx/xlsm well
    files = []
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files
