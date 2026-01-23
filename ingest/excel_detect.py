# ingest/excel_detect.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

@dataclass
class SheetHit:
    sheet: str
    kind: str  # "amp" or "cq"
    score: int
    notes: str

def _clean_columns(cols: Iterable[Any]) -> list[str]:
    out = []
    for c in cols:
        s = str(c).strip()
        s = re.sub(r"\s+", " ", s)
        out.append(s)
    return out

def _is_cycle_column(name: str) -> bool:
    # Accept: "Cycle", "Cycle #" etc.
    n = name.lower()
    return "cycle" in n

def _looks_like_well(name: str) -> bool:
    # header like A01, B03 etc
    return bool(re.match(r"^[A-Za-z]{1,2}\s*0*\d{1,2}$", str(name).strip()))

def detect_sheets(xls: pd.ExcelFile) -> list[SheetHit]:
    """
    Heuristic detection:
      - amplification sheet: has a cycle column + many well-like columns
      - cq sheet: has well column + cq/cq value column (Cq/Ct)
    """
    hits: list[SheetHit] = []
    for sh in xls.sheet_names:
        try:
            df0 = xls.parse(sh, nrows=60)  # sample rows
        except Exception:
            continue

        cols = _clean_columns(df0.columns)
        well_cols = [c for c in cols if _looks_like_well(c)]
        has_cycle = any(_is_cycle_column(c) for c in cols)

                # amplification score (relaxed for small-well admin runs)
        amp_score = 0
        if has_cycle:
            amp_score += 5

        if len(well_cols) >= 8:
            amp_score += 5
        elif len(well_cols) >= 4:
            amp_score += 3
        elif len(well_cols) >= 2:
            amp_score += 2  # <-- NEW: allow 2~3 wells

        # accept as amp if cycle column exists AND at least 2 well-like columns exist
        if has_cycle and len(well_cols) >= 2:
            hits.append(SheetHit(
                sheet=sh,
                kind="amp",
                score=amp_score,
                notes=f"cycle={has_cycle}, wells={len(well_cols)}"
            ))


        # cq score: search for common column names in first 60 rows
        lc_cols = [c.lower() for c in cols]
        has_well_col = any("well" in c for c in lc_cols) or any("sample" in c for c in lc_cols) or any("pos" in c for c in lc_cols)
        has_cq_col = any(c in {"cq", "ct"} or "cq" in c or "ct" in c for c in lc_cols)

        cq_score = 0
        if has_well_col:
            cq_score += 4
        if has_cq_col:
            cq_score += 6

        if cq_score >= 8:
            hits.append(SheetHit(sheet=sh, kind="cq", score=cq_score, notes=f"well_col={has_well_col}, cq_col={has_cq_col}"))

    # prefer higher score first
    hits.sort(key=lambda h: (h.kind, -h.score))
    return hits

def pick_best(hits: list[SheetHit], kind: str) -> str | None:
    cand = [h for h in hits if h.kind == kind]
    if not cand:
        return None
    cand.sort(key=lambda h: -h.score)
    return cand[0].sheet
