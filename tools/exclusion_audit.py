import re, os, sys
import pandas as pd

def norm_well(x):
    x = str(x).strip().upper()
    m = re.match(r"^([A-Z]+)(\d+)$", x)
    if not m:
        return x
    r, c = m.group(1), int(m.group(2))
    return f"{r}{c:02d}"

def pick_cols(df):
    cols = df.columns.tolist()
    cols_l = [c.lower() for c in cols]

    # 우선순위: 정확히 run_id / well_id
    run_col  = next((cols[i] for i,c in enumerate(cols_l) if c == "run_id"), None)
    well_col = next((cols[i] for i,c in enumerate(cols_l) if c == "well_id"), None)

    # 보조: run / plate / well 같은 패턴
    if run_col is None:
        run_col = next((cols[i] for i,c in enumerate(cols_l) if "run" in c or "plate" in c), None)
    if well_col is None:
        well_col = next((cols[i] for i,c in enumerate(cols_l) if "well" in c), None)

    return run_col, well_col

def main():
    if len(sys.argv) < 2:
        raise SystemExit("USAGE: python tools/exclusion_audit.py <LIST_FILE.xlsx|csv> [CUTOFF]\n"
                         "example: python tools/exclusion_audit.py reports/exclusions/384_list.xlsx 24")

    list_file = sys.argv[1]
    cutoff = int(sys.argv[2]) if len(sys.argv) >= 3 else 24

    if not os.path.exists(list_file):
        raise SystemExit(f"[ERR] list file not found: {list_file}")

    # 1) 리스트 로드
    if list_file.lower().endswith(".xlsx"):
        lst_raw = pd.read_excel(list_file)
    else:
        lst_raw = pd.read_csv(list_file)

    run_col, well_col = pick_cols(lst_raw)
    if run_col is None or well_col is None:
        raise SystemExit(f"[ERR] cannot find run/well columns in list.\n"
                         f"columns={lst_raw.columns.tolist()}")

    lst = lst_raw[[run_col, well_col]].rename(columns={run_col:"run_id", well_col:"well_id"}).copy()
    lst["run_id"] = lst["run_id"].astype(str).str.strip()
    lst["well_id"] = lst["well_id"].map(norm_well)

    # 중복 제거(384리스트가 '384개'라고 했으니 보통 유니크여야 함)
    lst = lst.drop_duplicates(["run_id","well_id"]).reset_index(drop=True)

    print(f"[OK] list loaded: {list_file}")
    print("[INFO] rows:", len(lst), "unique(run,well):", lst.shape[0])
    print("[INFO] unique run_id:", lst["run_id"].nunique())

    # 2) master_long 요약
    df = pd.read_parquet("data/canonical/master_long.parquet").copy()
    df["run_id"] = df["run_id"].astype(str).str.strip()
    df["well_id"] = df["Well"].map(norm_well)

    g = (df.groupby(["run_id","well_id"], as_index=False)
           .agg(
               max_cycle=("Cycle","max"),
               true_ct=("Cq", lambda s: pd.to_numeric(s, errors="coerce").dropna().iloc[0]
                        if pd.to_numeric(s, errors="coerce").dropna().shape[0] else float("nan"))
           ))

    g["has_enough_cycles"] = g["max_cycle"] >= cutoff
    g["has_label"] = pd.notna(g["true_ct"])

    # 3) merge + reason
    out = lst.merge(g, on=["run_id","well_id"], how="left")

    has_data = out["max_cycle"].notna()
    has_label = out["has_label"].fillna(False).astype(bool)
    has_enough = out["has_enough_cycles"].fillna(False).astype(bool)

    out["reason"] = "INCLUDED"
    out.loc[~has_data, "reason"] = "EXCLUDE_NO_DATA"
    out.loc[has_data & (~has_label), "reason"] = "EXCLUDE_NO_LABEL"
    out.loc[has_data & has_label & (~has_enough), "reason"] = "EXCLUDE_SHORT_TRACE"

    out["included"] = (out["reason"] == "INCLUDED")

    # 4) 요약
    included_n = int(out["included"].sum())
    total_n = int(len(out))
    excluded_n = total_n - included_n

    print("\n=== SUMMARY (from 384 list) ===")
    print(out["reason"].value_counts())
    print("total:", total_n, "included:", included_n, "excluded:", excluded_n)

    # 5) 저장
    os.makedirs("reports/exclusions", exist_ok=True)
    out_file = f"reports/exclusions/exclusion_audit_from_384_list_cutoff{cutoff}.xlsx"
    out.to_excel(out_file, index=False)
    print("[OK] wrote:", out_file)

if __name__ == "__main__":
    main()
