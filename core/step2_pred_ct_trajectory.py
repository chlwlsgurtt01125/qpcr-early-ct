# core/step2_pred_ct_trajectory.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import xgboost as xgb

# 재사용: 기존 step2_eval_cutoffs.py의 함수들
from core.step2_eval_cutoffs import (
    DATA,
    OUT_DIR,
    build_curve_matrix,
    fit_predict_xgb,
)

def oof_predict_for_cutoff(df: pd.DataFrame, cutoff: int, seed: int = 42, use_gpu: bool = True) -> pd.DataFrame:
    """
    cutoff까지 관측 가능한 curve들에 대해,
    run_id 그룹 기반 GroupKFold로 out-of-fold(OOF) 예측을 생성.

    반환: well_uid, run_id, true_cq, pred_cq, abs_err, cutoff, max_cycle
    """
    X, y, meta = build_curve_matrix(df, cutoff)
    # meta: well_uid, run_id, max_cycle

    if len(y) < 10 or meta["run_id"].nunique() < 2:
        # 너무 적으면 스킵
        out = meta.copy()
        out["true_cq"] = y if len(y) == len(out) else np.nan
        out["pred_cq"] = np.nan
        out["abs_err"] = np.nan
        out["cutoff"] = cutoff
        return out

    groups = meta["run_id"].to_numpy()
    n_runs = meta["run_id"].nunique()

    # run 수가 적을 수 있으니 분할 수 자동 조정
    n_splits = min(5, n_runs)
    gkf = GroupKFold(n_splits=n_splits)

    pred = np.full(shape=(len(y),), fill_value=np.nan, dtype=float)

    for fold, (trainval_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        # trainval -> train/val을 다시 run 기준으로 쪼개서 early stopping용 validation 구성
        meta_tv = meta.iloc[trainval_idx].copy()
        groups_tv = meta_tv["run_id"].to_numpy()
        idx_tv = np.arange(len(meta_tv))

        # val 비율은 적당히(0.2) 고정
        if meta_tv["run_id"].nunique() >= 2 and len(idx_tv) >= 20:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + fold)
            tr2, va2 = next(gss.split(idx_tv, groups=groups_tv))
            train_idx = trainval_idx[tr2]
            val_idx = trainval_idx[va2]
        else:
            # run이 너무 적거나 데이터가 너무 적으면 trainval 전체를 train으로, val은 train의 일부를 랜덤으로
            rng = np.random.default_rng(seed + fold)
            perm = rng.permutation(trainval_idx)
            split = max(1, int(0.8 * len(perm)))
            train_idx = perm[:split]
            val_idx = perm[split:] if split < len(perm) else perm[:split]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test = X[test_idx]

        bst = fit_predict_xgb(X_train, y_train, X_val, y_val, use_gpu=use_gpu)
        pred[test_idx] = bst.predict(xgb.DMatrix(X_test))

    out = meta.copy()
    out["true_cq"] = y
    out["pred_cq"] = pred
    out["abs_err"] = np.abs(out["pred_cq"] - out["true_cq"])
    out["cutoff"] = cutoff
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA)

    # 기본 정리
    df = df.dropna(subset=["well_uid", "run_id", "Cycle", "Fluor"])
    df["Cycle"] = df["Cycle"].astype(int)

    # ✅ “전체 샘플 148개”를 기준으로 보고 싶다면,
    # Ct가 존재하는 curve들만 대상으로 일단 고정(이게 너의 ‘샘플=환자’ 개념에 맞음)
    # (Ct가 NaN인 curve는 true 비교 자체가 불가능하니까)
    cq_first = df.groupby("well_uid")["Cq"].first()
    has_cq = cq_first.dropna().index
    df = df[df["well_uid"].isin(has_cq)].copy()

    total_curves = df["well_uid"].nunique()
    total_runs = df["run_id"].nunique()
    print(f"[DATA] curves_with_Cq={total_curves} runs={total_runs}")

    # cutoff 전체 범위
    cutoffs = list(range(1, 41))  # 1..40
    all_rows = []
    summary = []

    for cutoff in cutoffs:
        oof = oof_predict_for_cutoff(df, cutoff, seed=42, use_gpu=True)

        # cutoff에서 실제로 예측이 가능한 curve 수(= Cycle 1..cutoff 모두 존재 + Cq 존재)
        n_curves = int(oof["true_cq"].notna().sum())
        n_pred = int(oof["pred_cq"].notna().sum())
        n_runs = int(oof["run_id"].nunique())

        # 요약 통계 (예측값이 있는 것만)
        ok = oof.dropna(subset=["pred_cq", "true_cq"]).copy()
        if len(ok) > 0:
            mae = float(ok["abs_err"].mean())
            med = float(ok["abs_err"].median())
            p50 = med
            p90 = float(ok["abs_err"].quantile(0.90))
            within_05 = float((ok["abs_err"] <= 0.5).mean())
            within_10 = float((ok["abs_err"] <= 1.0).mean())
        else:
            mae = med = p50 = p90 = within_05 = within_10 = np.nan

        summary.append({
            "cutoff": cutoff,
            "n_curves": n_curves,
            "n_pred": n_pred,
            "n_runs": n_runs,
            "mae": mae,
            "median_abs_err": med,
            "p90_abs_err": p90,
            "pct_within_0.5": within_05,
            "pct_within_1.0": within_10,
        })

        all_rows.append(oof)

        print(
            f"[cutoff={cutoff:2d}] eligible={n_curves:3d} pred={n_pred:3d} "
            f"MAE={mae:.3f}  MED={med:.3f}  P90={p90:.3f}  "
            f"<=0.5:{within_05*100:.1f}%  <=1.0:{within_10*100:.1f}%"
        )

    long_df = pd.concat(all_rows, ignore_index=True)

    # 저장 (long + summary)
    out_long_parquet = OUT_DIR / "step2_pred_ct_trajectory_long.parquet"
    out_long_csv = OUT_DIR / "step2_pred_ct_trajectory_long.csv"
    out_summary_csv = OUT_DIR / "step2_pred_ct_trajectory_summary.csv"

    long_df.to_parquet(out_long_parquet, index=False)
    long_df.to_csv(out_long_csv, index=False, encoding="utf-8-sig")

    sum_df = pd.DataFrame(summary).sort_values("cutoff")
    sum_df.to_csv(out_summary_csv, index=False, encoding="utf-8-sig")

    print(f"\n[SAVED] long(parquet)  {out_long_parquet}")
    print(f"[SAVED] long(csv)      {out_long_csv}")
    print(f"[SAVED] summary(csv)   {out_summary_csv}")

    # “각 샘플이 몇 cycle에서 충분히 맞아지는지”를 보고 싶다면,
    # 여기서 샘플별 earliest cutoff 계산도 가능 (원하면 추가해줄게)


if __name__ == "__main__":
    main()
