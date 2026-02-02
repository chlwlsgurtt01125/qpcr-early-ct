#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize QC x Prediction performance outputs into a "next actions" report.

Auto-discovers:
  - outputs/**/qc_pred_merged.parquet (or similar)
  - outputs/**/bucket_performance.(parquet|csv)
  - outputs/**/figures/*.png

Produces:
  - outputs/reports/next_actions.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import pandas as pd
import numpy as np


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def find_latest_file(patterns: List[str], base: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for pat in patterns:
        candidates += list(base.glob(pat))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_best_bucket_file(base: Path) -> Optional[Path]:
    pats = [
        "outputs/**/bucket_performance.parquet",
        "outputs/**/bucket_performance.csv",
        "outputs/**/*bucket*performance*.parquet",
        "outputs/**/*bucket*performance*.csv",
    ]
    return find_latest_file(pats, base)


def find_best_merged_file(base: Path) -> Optional[Path]:
    pats = [
        "outputs/**/qc_pred_merged.parquet",
        "outputs/**/*merged*.parquet",
    ]
    return find_latest_file(pats, base)


def find_figures_dir(base: Path) -> Optional[Path]:
    preferred = base / "outputs" / "qc_performance_analysis" / "figures"
    if preferred.exists() and preferred.is_dir():
        return preferred
    dirs = [p for p in (base / "outputs").glob("**/figures") if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def ensure_metric_columns_from_merged(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()
    colmap = {c.lower(): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n in df.columns:
                return n
            if n.lower() in colmap:
                return colmap[n.lower()]
        return None

    pred_col = pick("pred_ct", "y_pred", "pred", "prediction")
    true_col = pick("true_ct", "y_true", "true", "label", "ct_true", "cq_true")
    if pred_col is None or true_col is None:
        raise ValueError(f"merged parquet must include pred/true ct columns. cols={df.columns.tolist()}")

    df["_pred_ct"] = pd.to_numeric(df[pred_col], errors="coerce")
    df["_true_ct"] = pd.to_numeric(df[true_col], errors="coerce")
    df = df.dropna(subset=["_pred_ct", "_true_ct"])

    df["err"] = df["_pred_ct"] - df["_true_ct"]
    df["abs_err"] = df["err"].abs()
    df["acc_0p5"] = (df["abs_err"] <= 0.5).astype(float)
    df["acc_1p0"] = (df["abs_err"] <= 1.0).astype(float)
    df["fold_error"] = np.power(2.0, df["abs_err"].astype(float))

    qc_col = pick("qc_status", "qc", "status")
    df["qc_status"] = df[qc_col].astype(str) if qc_col else "UNKNOWN"

    fr_col = pick("fail_reason", "reason", "qc_reason")
    df["fail_reason"] = df[fr_col].fillna("UNKNOWN").astype(str) if fr_col else "UNKNOWN"

    cb_col = pick("ct_bin", "ctbin", "bin")
    df["ct_bin"] = df[cb_col].fillna("UNKNOWN").astype(str) if cb_col else "UNKNOWN"

    wu_col = pick("well_uid", "well_id")
    if wu_col is not None and wu_col != "well_uid":
        df["well_uid"] = df[wu_col].astype(str)
    elif "well_uid" not in df.columns:
        df["well_uid"] = "UNKNOWN"

    return df


def compute_group_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    g = df.groupby(group_cols, dropna=False)
    out = g.agg(
        n=("abs_err", "count"),
        mae=("abs_err", "mean"),
        rmse=("err", lambda x: float(np.sqrt(np.mean(np.square(x.astype(float)))))),
        acc_0p5=("acc_0p5", "mean"),
        acc_1p0=("acc_1p0", "mean"),
        fold_error_mean=("fold_error", "mean"),
    ).reset_index()
    return out


def _fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x*100:5.1f}%"


def _fmt_float(x: float, nd: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{nd}f}"


def df_to_markdown_table(df: pd.DataFrame, cols: List[str], max_rows: int = 10) -> str:
    view = df.copy()
    if len(view) > max_rows:
        view = view.head(max_rows)
    view = view[cols]
    for c in cols:
        if c in ("mae", "rmse", "fold_error_mean"):
            view[c] = view[c].apply(lambda v: _fmt_float(v, 3))
        if c in ("acc_0p5", "acc_1p0"):
            view[c] = view[c].apply(lambda v: _fmt_pct(v))
    return view.to_markdown(index=False)


def pick_top_worst_buckets(bucket_df: pd.DataFrame, min_n: int = 5, topk: int = 3) -> pd.DataFrame:
    df = bucket_df.copy()
    for c in ["n", "mae", "acc_0p5", "fold_error_mean"]:
        if c not in df.columns:
            raise ValueError(f"bucket df missing required column: {c}")
    df = df[df["n"] >= min_n].copy()
    if df.empty:
        return df
    df = df.sort_values(by=["mae", "acc_0p5", "fold_error_mean"], ascending=[False, True, False])
    return df.head(topk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".", help="project root (default: .)")
    ap.add_argument("--min_n", type=int, default=5, help="min samples per bucket to consider")
    ap.add_argument("--out", default="outputs/reports/next_actions.txt", help="output report path")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    out_path = (base / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged_path = find_best_merged_file(base)
    bucket_path = find_best_bucket_file(base)
    fig_dir = find_figures_dir(base)

    print("============================================================")
    print("QC Report Summarizer")
    print("============================================================")
    print(f"Time: {_now_str()}")
    print(f"Base: {base}")
    print(f"Merged: {merged_path if merged_path else 'NOT FOUND'}")
    print(f"Bucket: {bucket_path if bucket_path else 'NOT FOUND'}")
    print(f"Figures: {fig_dir if fig_dir else 'NOT FOUND'}")

    if merged_path is None:
        print("\n❌ merged parquet not found under outputs/. Run analyze_qc_performance.py first.")
        sys.exit(2)

    merged = pd.read_parquet(merged_path)
    merged = normalize_columns(merged)
    merged = ensure_metric_columns_from_merged(merged)

    by_fail = compute_group_metrics(merged, ["fail_reason"])
    by_ctbin = compute_group_metrics(merged, ["ct_bin"])
    by_bucket = compute_group_metrics(merged, ["qc_status", "fail_reason", "ct_bin"])

    fail_mae_top5 = by_fail.sort_values("mae", ascending=False).head(5)
    fail_acc05_bottom5 = by_fail.sort_values("acc_0p5", ascending=True).head(5)
    ctbin_mae_top5 = by_ctbin.sort_values("mae", ascending=False).head(5)
    fold_top5 = by_bucket.sort_values("fold_error_mean", ascending=False).head(5)
    worst3 = pick_top_worst_buckets(by_bucket, min_n=args.min_n, topk=3)

    overall_mae = float(merged["abs_err"].mean()) if len(merged) else float("nan")
    overall_rmse = float(np.sqrt(np.mean(np.square(merged["err"].astype(float))))) if len(merged) else float("nan")
    overall_acc05 = float(merged["acc_0p5"].mean()) if len(merged) else float("nan")
    overall_acc10 = float(merged["acc_1p0"].mean()) if len(merged) else float("nan")
    overall_fold = float(merged["fold_error"].mean()) if len(merged) else float("nan")

    lines: List[str] = []
    lines.append("============================================================")
    lines.append("NEXT ACTIONS REPORT (QC x Prediction)")
    lines.append("============================================================")
    lines.append(f"Generated: {_now_str()}")
    lines.append("")
    lines.append("[Auto-discovered artifacts]")
    lines.append(f"- merged_parquet: {str(merged_path) if merged_path else 'NOT FOUND'}")
    lines.append(f"- bucket_file:    {str(bucket_path) if bucket_path else 'NOT FOUND'}")
    lines.append(f"- figures_dir:    {str(fig_dir) if fig_dir else 'NOT FOUND'}")
    if fig_dir and fig_dir.exists():
        figs = sorted([p.name for p in fig_dir.glob("*.png")])
        if figs:
            lines.append(f"- figures:        {', '.join(figs)}")
    lines.append("")
    lines.append("[Overall]")
    lines.append(f"- N: {len(merged)}")
    lines.append(f"- MAE:  {_fmt_float(overall_mae, 3)}")
    lines.append(f"- RMSE: {_fmt_float(overall_rmse, 3)}")
    lines.append(f"- acc(|err|<=0.5): {_fmt_pct(overall_acc05)}")
    lines.append(f"- acc(|err|<=1.0): {_fmt_pct(overall_acc10)}")
    lines.append(f"- mean fold_error (2^|err|): {_fmt_float(overall_fold, 2)}x")
    lines.append("")

    lines.append("------------------------------------------------------------")
    lines.append("1) fail_reason별 MAE 상위 5개 (worst by MAE)")
    lines.append("------------------------------------------------------------")
    lines.append(
        df_to_markdown_table(
            fail_mae_top5,
            cols=["fail_reason", "n", "mae", "rmse", "acc_0p5", "acc_1p0", "fold_error_mean"],
            max_rows=5,
        )
        if len(fail_mae_top5)
        else "No data."
    )
    lines.append("")

    lines.append("------------------------------------------------------------")
    lines.append("2) fail_reason별 acc_0p5 하위 5개 (worst by accuracy@0.5)")
    lines.append("------------------------------------------------------------")
    lines.append(
        df_to_markdown_table(
            fail_acc05_bottom5,
            cols=["fail_reason", "n", "mae", "rmse", "acc_0p5", "acc_1p0", "fold_error_mean"],
            max_rows=5,
        )
        if len(fail_acc05_bottom5)
        else "No data."
    )
    lines.append("")

    lines.append("------------------------------------------------------------")
    lines.append("3) ct_bin별 MAE 상위 5개 (worst Ct ranges)")
    lines.append("------------------------------------------------------------")
    lines.append(
        df_to_markdown_table(
            ctbin_mae_top5,
            cols=["ct_bin", "n", "mae", "rmse", "acc_0p5", "acc_1p0", "fold_error_mean"],
            max_rows=5,
        )
        if len(ctbin_mae_top5)
        else "No data."
    )
    lines.append("")

    lines.append("------------------------------------------------------------")
    lines.append("4) fold_error_mean 상위 5개 (largest fold error buckets)")
    lines.append("   (bucket = qc_status + fail_reason + ct_bin)")
    lines.append("------------------------------------------------------------")
    lines.append(
        df_to_markdown_table(
            fold_top5,
            cols=["qc_status", "fail_reason", "ct_bin", "n", "mae", "acc_0p5", "fold_error_mean"],
            max_rows=5,
        )
        if len(fold_top5)
        else "No data."
    )
    lines.append("")

    lines.append("------------------------------------------------------------")
    lines.append(f"5) Top 3 worst bucket (min_n={args.min_n})")
    lines.append("   Ranking: MAE desc → acc_0p5 asc → fold_error_mean desc")
    lines.append("------------------------------------------------------------")
    if len(worst3) == 0:
        lines.append("No bucket meets min_n threshold.")
    else:
        lines.append(
            df_to_markdown_table(
                worst3,
                cols=["qc_status", "fail_reason", "ct_bin", "n", "mae", "rmse", "acc_0p5", "acc_1p0", "fold_error_mean"],
                max_rows=3,
            )
        )
        lines.append("")
        lines.append("[Suggested next actions (auto)]")
        for i, row in worst3.reset_index(drop=True).iterrows():
            lines.append(
                f"- ({i+1}) qc_status={row.get('qc_status')}, fail_reason={row.get('fail_reason')}, ct_bin={row.get('ct_bin')}"
                f" | MAE={_fmt_float(row.get('mae'),3)}, acc@0.5={_fmt_pct(row.get('acc_0p5'))}, fold_mean={_fmt_float(row.get('fold_error_mean'),2)}x"
            )
            if str(row.get("qc_status")).upper() == "FAIL":
                lines.append("    - Action: 해당 fail_reason QC 규칙에 맞춰 데이터 정제/제외 또는 전처리 재검토")
            elif str(row.get("qc_status")).upper() == "FLAG":
                lines.append("    - Action: FLAG 샘플 수동 점검 후 포함/제외 정책 확정")
            else:
                lines.append("    - Action: PASS인데도 성능이 나쁘면 모델/feature/훈련 분할(특정 run 편향) 점검")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ Saved next-actions report: {out_path}")


if __name__ == "__main__":
    main()
