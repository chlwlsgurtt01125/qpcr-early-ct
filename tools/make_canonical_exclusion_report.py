import re
import pandas as pd

def norm_well(x):
    x = str(x).strip().upper()
    m = re.match(r"^([A-Z]+)(\d+)$", x)
    if not m:
        return x
    return f"{m.group(1)}{int(m.group(2)):02d}"

def main(cutoff=24, out_file=None):
    if out_file is None:
        out_file = f"reports/exclusions/canonical_exclusion_report_cutoff{cutoff}.xlsx"

    df = pd.read_parquet("data/canonical/master_long.parquet").copy()
    df["run_id"] = df["run_id"].astype(str).str.strip()
    df["well_id"] = df["Well"].map(norm_well)

    g = (df.groupby(["run_id","well_id"])
           .agg(max_cycle=("Cycle","max"),
                has_label=("Cq", lambda s: pd.to_numeric(s, errors="coerce").notna().any()),
                true_ct=("Cq", lambda s: pd.to_numeric(s, errors="coerce").dropna().iloc[0]
                        if pd.to_numeric(s, errors="coerce").dropna().shape[0] else float("nan")),
                n_points=("Cycle","size"))
           .reset_index())

    g["has_enough_cycles"] = g["max_cycle"] >= cutoff
    g["usable"] = g["has_label"] & g["has_enough_cycles"]

    # reason bucket (canonical에서 가능한 범위)
    g["reason_bucket"] = "INCLUDED"
    g.loc[~g["has_label"], "reason_bucket"] = "EXCLUDE_NO_LABEL_OR_MATCH_FAIL"
    g.loc[g["has_label"] & (~g["has_enough_cycles"]), "reason_bucket"] = "EXCLUDE_SHORT_TRACE_OR_MISSING"

    # summary
    summary = (g["reason_bucket"].value_counts()
                 .rename_axis("reason_bucket")
                 .reset_index(name="count"))
    summary["ratio"] = summary["count"] / len(g)

    totals = pd.DataFrame({
        "reason_bucket": ["__TOTAL__", "__USABLE_TRUE__", "__USABLE_FALSE__"],
        "count": [len(g), int(g["usable"].sum()), int((~g["usable"]).sum())],
        "ratio": [1.0, float(g["usable"].mean()), float((~g["usable"]).mean())],
    })
    summary = pd.concat([summary, totals], ignore_index=True)

    # run-level breakdown
    per_run = (g.groupby(["run_id","reason_bucket"]).size()
                 .reset_index(name="count"))
    per_run["total_in_run"] = per_run.groupby("run_id")["count"].transform("sum")
    per_run["ratio_in_run"] = per_run["count"] / per_run["total_in_run"]
    per_run = per_run.sort_values(["run_id","reason_bucket"])

    usable_true  = g[g["usable"]].copy()
    usable_false = g[~g["usable"]].copy()

    with pd.ExcelWriter(out_file, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        per_run.to_excel(w, sheet_name="per_run_summary", index=False)
        usable_true.to_excel(w, sheet_name="usable_true", index=False)
        usable_false.to_excel(w, sheet_name="usable_false", index=False)

    print("[OK] wrote:", out_file)
    print(summary)

if __name__ == "__main__":
    import sys
    cutoff = int(sys.argv[1]) if len(sys.argv) >= 2 else 24
    main(cutoff=cutoff)
