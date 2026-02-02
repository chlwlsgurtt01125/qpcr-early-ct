import re, os, sys
import pandas as pd

# -----------------
# 설정
# -----------------
DEFAULT_CUTOFF = 24

# NTC/Control 판정: (리스트 엑셀에 sample_name/target/name 같은 컬럼이 있으면 그 텍스트에서 탐지)
CONTROL_PATTERNS = [
    r"\bNTC\b", r"\bN\.?T\.?C\.?\b",
    r"\bCONTROL\b", r"\bCTRL\b",
    r"\bBLANK\b", r"\bH2O\b", r"\bWATER\b",
    r"\bNEG\b", r"\bNEGATIVE\b",
    r"\bPOS\b", r"\bPOSITIVE\b",
]

def norm_well(x):
    x = str(x).strip().upper()
    m = re.match(r"^([A-Z]+)\s*0*([0-9]+)$", x)
    if not m:
        return x
    r, c = m.group(1), int(m.group(2))
    return f"{r}{c:02d}"

def find_col(df, candidates):
    cols = df.columns.tolist()
    cols_l = [c.lower() for c in cols]
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols[cols_l.index(cand.lower())]
    return None

def pick_run_well_cols(df):
    run_col  = find_col(df, ["run_id","run","plate_id","plate","batch","runid"])
    well_col = find_col(df, ["well_id","well","wellname","well_name","wellposition","position"])
    # fallback
    if run_col is None:
        run_col = next((c for c in df.columns if "run" in c.lower() or "plate" in c.lower()), None)
    if well_col is None:
        well_col = next((c for c in df.columns if "well" in c.lower() or "pos" in c.lower()), None)
    return run_col, well_col

def detect_control_flags(lst_raw):
    # sample_name / target / name / sample 같은 컬럼에서 NTC/Control 패턴 탐지
    text_cols = []
    for c in lst_raw.columns:
        cl = c.lower()
        if any(k in cl for k in ["sample","name","target","assay","desc","type"]):
            text_cols.append(c)

    if not text_cols:
        return pd.Series([False]*len(lst_raw), index=lst_raw.index)

    pat = re.compile("|".join(CONTROL_PATTERNS), flags=re.IGNORECASE)
    joined = lst_raw[text_cols].astype(str).agg(" ".join, axis=1)
    return joined.map(lambda s: bool(pat.search(str(s))))

def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "USAGE: python tools/exclusion_audit_384.py <384_list.xlsx|csv> [CUTOFF]\n"
            "example: python tools/exclusion_audit_384.py reports/exclusions/384_list.xlsx 24"
        )

    list_file = sys.argv[1]
    cutoff = int(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_CUTOFF

    if not os.path.exists(list_file):
        raise SystemExit(f"[ERR] list file not found: {list_file}")

    # 1) 384 기준 리스트 로드
    if list_file.lower().endswith(".xlsx"):
        lst_raw = pd.read_excel(list_file)
    else:
        lst_raw = pd.read_csv(list_file)

    run_col, well_col = pick_run_well_cols(lst_raw)
    if run_col is None or well_col is None:
        raise SystemExit(f"[ERR] cannot find run/well columns in list.\ncolumns={lst_raw.columns.tolist()}")

    lst = lst_raw.copy()
    lst["run_id"]  = lst[run_col].astype(str).str.strip()
    lst["well_id"] = lst[well_col].map(norm_well)

    # (ii) NTC/Control 자동 탐지(없으면 False)
    lst["is_control"] = detect_control_flags(lst_raw)

    # 384 기준이니까 중복은 보통 없어야 함(혹시 있으면 유지하되 audit은 유니크 기준으로도 계산)
    lst_key = lst[["run_id","well_id"]].copy()

    # 2) canonical 요약 테이블 만들기
    df = pd.read_parquet("data/canonical/master_long.parquet").copy()
    df["run_id"]  = df["run_id"].astype(str).str.strip()
    df["well_id"] = df["Well"].map(norm_well)

    g = (df.groupby(["run_id","well_id"])
           .agg(max_cycle=("Cycle","max"),
                has_label=("Cq", lambda s: pd.to_numeric(s, errors="coerce").notna().any()))
           .reset_index())

    g["has_enough_cycles"] = g["max_cycle"] >= cutoff
    g["usable_signal"] = g["has_label"] & g["has_enough_cycles"]

    # 3) 384 기준 리스트 + canonical merge
    out = lst.merge(g, on=["run_id","well_id"], how="left")

    # boolean 정리
    out["in_canonical"] = out["max_cycle"].notna()  # 데이터가 아예 들어왔는지
    out["has_label"] = out["has_label"].fillna(False).astype(bool)
    out["has_enough_cycles"] = out["has_enough_cycles"].fillna(False).astype(bool)

    # 4) reason 매기기 (네가 말한 3그룹으로 깔끔하게)
    # 기본: OK
    out["usable"] = True
    out["reason"] = "OK"

    # (ii) control 우선 제외
    out.loc[out["is_control"] == True, ["usable","reason"]] = [False, "EXCLUDE_CONTROL_NTC"]

    # (i) 라벨 누락/매칭 실패
    # - canonical에 데이터는 있는데 라벨이 없는 경우
    mask_no_label = (out["is_control"] == False) & (out["in_canonical"] == True) & (out["has_label"] == False)
    out.loc[mask_no_label, ["usable","reason"]] = [False, "EXCLUDE_NO_LABEL_OR_MATCH_FAIL"]

    # (iii) cycle 부족/결측/QC 이슈(여기서는 최소조건: cutoff 미만)
    mask_short = (out["is_control"] == False) & (out["in_canonical"] == True) & (out["has_label"] == True) & (out["has_enough_cycles"] == False)
    out.loc[mask_short, ["usable","reason"]] = [False, "EXCLUDE_SHORT_TRACE_OR_QC"]

    # canonical에 데이터 자체가 없는 경우 (업로드/파싱/매칭 전 단계 이슈)
    mask_no_data = (out["in_canonical"] == False)
    out.loc[mask_no_data, ["usable","reason"]] = [False, "EXCLUDE_NOT_IN_CANONICAL"]

    # 5) 요약표 만들기
    total_n = len(out)
    usable_n = int(out["usable"].sum())
    excluded_n = total_n - usable_n

    reason_counts = (out["reason"].value_counts()
                       .rename_axis("reason")
                       .reset_index(name="count"))
    reason_counts["pct"] = (reason_counts["count"] / total_n * 100).round(1)

    totals = pd.DataFrame({
        "reason": ["__TOTAL__", "__USABLE_TRUE__", "__USABLE_FALSE__"],
        "count": [total_n, usable_n, excluded_n],
        "pct": [100.0, round(usable_n/total_n*100,1), round(excluded_n/total_n*100,1)]
    })
    summary = pd.concat([totals, reason_counts], ignore_index=True)

    # 6) 저장
    out_file = f"reports/exclusions/384_audit_cutoff{cutoff}.xlsx"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with pd.ExcelWriter(out_file, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        out.sort_values(["run_id","well_id"]).to_excel(w, sheet_name="well_level", index=False)
        out[out["usable"]].to_excel(w, sheet_name="usable_true", index=False)
        out[~out["usable"]].to_excel(w, sheet_name="usable_false", index=False)

    print("[OK] wrote:", out_file)
    print(summary)

if __name__ == "__main__":
    main()
