import re, os
import pandas as pd

CUTOFF = 24
OUT_FILE = "reports/exclusions/exclusion_report_384_cutoff24.xlsx"

def norm_well(x):
    x = str(x).strip().upper()
    m = re.match(r"^([A-Z]+)(\d+)$", x)
    if not m:
        return x
    return f"{m.group(1)}{int(m.group(2)):02d}"

def make_plate384_wells():
    rows = [chr(ord('A') + i) for i in range(16)]  # A~P
    cols = [f"{j:02d}" for j in range(1, 25)]      # 01~24
    wells = [f"{r}{c}" for r in rows for c in cols]
    return wells

def main():
    os.makedirs("reports/exclusions", exist_ok=True)

    # 1) canonical 요약 (run,well 단위)
    df = pd.read_parquet("data/canonical/master_long.parquet").copy()
    df["run_id"] = df["run_id"].astype(str).str.strip()
    df["well_id"] = df["Well"].map(norm_well)

    g = (df.groupby(["run_id","well_id"])
           .agg(
               max_cycle=("Cycle","max"),
               true_ct=("Cq", lambda s: pd.to_numeric(s, errors="coerce").dropna().iloc[0]
                        if pd.to_numeric(s, errors="coerce").dropna().shape[0] else float("nan")),
               has_label=("Cq", lambda s: pd.to_numeric(s, errors="coerce").notna().any())
           )
           .reset_index())

    g["has_enough_cycles"] = g["max_cycle"] >= CUTOFF
    g["usable"] = g["has_label"] & g["has_enough_cycles"]

    # 2) “384 모수”를 만들기 위해 run_id는 'BATCH_ALL'로 묶어서 plate map 생성
    #    (네가 진짜로 '384 list'를 가지고 있으면 여기 대신 그 리스트를 merge해야 함)
    plate = pd.DataFrame({"well_id": make_plate384_wells()})
    plate["run_id"] = "BATCH_ALL"

    # 3) canonical 전체를 'BATCH_ALL'로 접어 넣기 위해 well_id 단위로 presence만 사용
    #    -> 384 기준 요약이 필요할 때, 네가 말한 “총 384 중 274 사용” 형태를 일단 만들기 위함
    #    (단, 이건 multi-run을 하나로 합친 근사치라서 교수님께 보낼 때 문구로 설명 필요)
    gw = (g.groupby("well_id")
            .agg(
                present_in_canonical=("well_id","size"),
                max_cycle=("max_cycle","max"),
                true_ct=("true_ct", lambda s: pd.to_numeric(s, errors="coerce").dropna().iloc[0]
                         if pd.to_numeric(s, errors="coerce").dropna().shape[0] else float("nan")),
                has_label=("has_label","any"),
                has_enough_cycles=("has_enough_cycles","any"),
                usable=("usable","any"),
            )
            .reset_index())

    out = plate.merge(gw, on="well_id", how="left")

    out["present_in_canonical"] = out["present_in_canonical"].fillna(0).astype(int)
    out["has_label"] = out["has_label"].fillna(False).astype(bool)
    out["has_enough_cycles"] = out["has_enough_cycles"].fillna(False).astype(bool)
    out["usable"] = out["usable"].fillna(False).astype(bool)

    # 4) reason (네가 원하는 3분류를 반영)
    # (i) 라벨 누락/매칭 실패 -> NO_LABEL (곡선은 있는데 Cq 없음)
    # (iii) cycle 부족/QC -> SHORT_TRACE (곡선 있는데 cycle<cutoff)
    # (ii) NTC/Control/분석제외/업로드누락 -> NO_DATA_OR_EXCLUDED (canonical에 곡선 자체가 없음)
    out["reason_detail"] = "USABLE"
    out.loc[out["present_in_canonical"] == 0, "reason_detail"] = "NO_DATA_OR_EXCLUDED"
    out.loc[(out["present_in_canonical"] > 0) & (~out["has_label"]), "reason_detail"] = "NO_LABEL"
    out.loc[(out["present_in_canonical"] > 0) & (out["has_label"]) & (~out["has_enough_cycles"]), "reason_detail"] = "SHORT_TRACE"

    out["usable_bool"] = out["usable"]

    # 상위 bucket(네가 문장에 쓰는 i/ii/iii)
    out["reason_bucket"] = "INCLUDED"
    out.loc[out["reason_detail"] == "NO_LABEL", "reason_bucket"] = "EXCLUDE_LABEL_MISSING_OR_MATCH_FAIL"
    out.loc[out["reason_detail"] == "NO_DATA_OR_EXCLUDED", "reason_bucket"] = "EXCLUDE_NTC_CONTROL_OR_NO_DATA"
    out.loc[out["reason_detail"] == "SHORT_TRACE", "reason_bucket"] = "EXCLUDE_SHORT_TRACE_OR_QC"

    # 5) 요약표
    reason_counts = (out["reason_bucket"].value_counts()
                       .rename_axis("reason_bucket")
                       .reset_index(name="count"))
    totals = pd.DataFrame({
        "reason_bucket": ["__TOTAL__", "__USABLE_TRUE__", "__USABLE_FALSE__"],
        "count": [len(out), int(out["usable_bool"].sum()), int((~out["usable_bool"]).sum())]
    })
    summary = pd.concat([reason_counts, totals], ignore_index=True)

    usable_true = out[out["usable_bool"]].copy()
    usable_false = out[~out["usable_bool"]].copy()

    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        out.to_excel(w, sheet_name="all_384", index=False)
        usable_true.to_excel(w, sheet_name="usable_true", index=False)
        usable_false.to_excel(w, sheet_name="usable_false", index=False)

    print("[OK] wrote:", OUT_FILE)
    print(summary)

if __name__ == "__main__":
    main()
