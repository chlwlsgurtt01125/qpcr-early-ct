# ingest/run_ingest_step1.py
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

from ingest.ingest_excel import find_excel_files, ingest_one_excel

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CANON_DIR = ROOT / "data" / "canonical"
AUDIT_DIR = ROOT / "data" / "audit"

def main():
    CANON_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    files = find_excel_files(RAW_DIR)
    if not files:
        raise SystemExit(f"No Excel files found under {RAW_DIR}")

    all_frames = []
    audit_rows = []

    for fp in files:
        try:
            df, res = ingest_one_excel(fp, channel="SYBR")
            all_frames.append(df)
            audit_rows.append({
                "run_id": res.run_id,
                "file": res.file_path,
                "n_amp_rows": res.n_amp_rows,
                "n_wells_amp": res.n_wells_amp,
                "n_cq_rows": res.n_cq,
                "n_rows_with_cq": res.n_merged,
                "warnings": " | ".join(res.warnings) if res.warnings else ""
            })
            print(f"[OK] {res.run_id}: amp_rows={res.n_amp_rows:,} wells={res.n_wells_amp:,} cq_rows={res.n_cq:,} rows_with_cq={res.n_merged:,}")
            if res.warnings:
                for w in res.warnings:
                    print(f"     [WARN] {w}")
        except Exception as e:
            audit_rows.append({
                "run_id": fp.stem,
                "file": str(fp),
                "n_amp_rows": 0,
                "n_wells_amp": 0,
                "n_cq_rows": 0,
                "n_rows_with_cq": 0,
                "warnings": f"FAILED: {e}"
            })
            print(f"[FAIL] {fp.name}: {e}")

    audit_df = pd.DataFrame(audit_rows)
    audit_path = AUDIT_DIR / "step1_audit.csv"
    audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")

    if all_frames:
        master = pd.concat(all_frames, ignore_index=True)
        # enforce types
        master["Cycle"] = master["Cycle"].astype(int)
        master["Fluor"] = pd.to_numeric(master["Fluor"], errors="coerce")
        master["Cq"] = pd.to_numeric(master["Cq"], errors="coerce")

        out_path = CANON_DIR / "master_long.parquet"
        master.to_parquet(out_path, index=False)
        print(f"\n[SAVED] {out_path} rows={len(master):,} runs={master['run_id'].nunique():,} wells={master['well_uid'].nunique():,}")
    else:
        print("\n[NO DATA] No runs ingested successfully. Check audit CSV for failures.")

    print(f"[AUDIT] {audit_path}")

if __name__ == "__main__":
    main()
