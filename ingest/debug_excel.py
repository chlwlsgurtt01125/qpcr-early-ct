# ingest/debug_excel.py
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

WELL_COL_RE = re.compile(r"^[A-Za-z]{1,2}\s*0*\d{1,2}$")
CYCLE_HEADER_RE = re.compile(r"^\s*\d+\s*$")  # "1", "2", ... "40"

def looks_well_header(x: str) -> bool:
    return bool(WELL_COL_RE.match(str(x).strip()))

def looks_cycle_header(x: str) -> bool:
    return bool(CYCLE_HEADER_RE.match(str(x).strip()))

def scan_one(fp: Path):
    print("=" * 120)
    print("FILE:", fp)
    xls = pd.ExcelFile(fp)
    print("SHEETS:", xls.sheet_names)

    for sh in xls.sheet_names:
        print("-" * 120)
        print("SHEET:", sh)
        try:
            # 헤더/병합셀 대응 위해 2가지로 읽어봄
            df0 = xls.parse(sh, nrows=12)
            cols = [str(c).strip() for c in df0.columns]
            well_cols = sum(looks_well_header(c) for c in cols)
            cycle_cols = sum(looks_cycle_header(c) for c in cols)
            has_cycle_word = any("cycle" in c.lower() for c in cols)
            has_cq_word = any(("cq" in c.lower()) or ("ct" in c.lower()) for c in cols)
            print(f"  cols(n={len(cols)}): {cols[:12]}{' ...' if len(cols)>12 else ''}")
            print(f"  stats: well_cols={well_cols}, cycle_cols={cycle_cols}, has_cycle_word={has_cycle_word}, has_cq_word={has_cq_word}")
            print("  head:")
            print(df0.head(5).to_string(index=False))
        except Exception as e:
            print("  [READ FAIL]", e)

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m ingest.debug_excel <excel_path1> [excel_path2 ...]")
        raise SystemExit(1)
    for p in sys.argv[1:]:
        scan_one(Path(p))

if __name__ == "__main__":
    main()
