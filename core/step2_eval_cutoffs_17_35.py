# core/step2_eval_cutoffs_17_35.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

# 기존 로직 재사용 (중복 방지)
from core.step2_eval_cutoffs import (
    build_curve_matrix,
    split_by_run,
    fit_predict_xgb,
    metrics,
    ROOT,
    DATA,
    OUT_DIR,
)

def main():
    df = pd.read_parquet(DATA)

    # 기본 정리
    df = df.dropna(subset=["well_uid", "run_id", "Cycle", "Fluor"])
    df["Cycle"] = df["Cycle"].astype(int)

    # ✅ 공정 비교를 위해: 35-cycle 이상 존재하는 curve만 고정
    # (이걸 안 하면 cutoff별로 표본이 바뀌어서 25<->30 비교가 흐려짐)
    max_cycle = df.groupby("well_uid")["Cycle"].max()
    keep = max_cycle[max_cycle >= 35].index
    df = df[df["well_uid"].isin(keep)].copy()

    fixed_n = df["well_uid"].nunique()
    fixed_runs = df["run_id"].nunique()
    print(f"[FIXED SET] curves_with_>=35cycles = {fixed_n}  runs={fixed_runs}")

    cutoffs = list(range(17, 36))  # 17..35 inclusive
    rows = []

    for cutoff in cutoffs:
        X, y, meta = build_curve_matrix(df, cutoff)
        n = len(y)
        if n < 10:
            rows.append({"cutoff": cutoff, "n_curves": n, "mae": np.nan, "rmse": np.nan, "r2": np.nan})
            print(f"[cutoff={cutoff:2d}] n_curves={n:3d}  (skip: too few)")
            continue

        train_idx, val_idx, test_idx = split_by_run(meta, test_size=0.2, val_size=0.2, seed=42)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        bst = fit_predict_xgb(X_train, y_train, X_val, y_val, use_gpu=True)
        pred = bst.predict(xgb.DMatrix(X_test))

        mae, rmse, r2 = metrics(y_test, pred)
        rows.append({"cutoff": cutoff, "n_curves": n, "mae": mae, "rmse": rmse, "r2": r2})

        print(f"[cutoff={cutoff:2d}] n_curves={n:3d}  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

    out = pd.DataFrame(rows).sort_values("cutoff")

    out_path = OUT_DIR / "step2_metrics_by_cutoff_17_35.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # best cutoff (MAE 최소, 컷오프 작은 것 선호)
    out2 = out.dropna(subset=["mae"]).copy()
    out2 = out2[out2["n_curves"] >= 10]
    if len(out2) > 0:
        best = out2.sort_values(["mae", "cutoff"]).iloc[0]
        best_path = OUT_DIR / "step2_best_cutoff_17_35.txt"
        best_path.write_text(
            f"best_cutoff={int(best['cutoff'])}\n"
            f"mae={best['mae']}\nrmse={best['rmse']}\nr2={best['r2']}\n"
            f"fixed_set_curves={fixed_n}\nfixed_set_runs={fixed_runs}\n",
            encoding="utf-8",
        )
        print(f"\n[BEST 17-35] cutoff={int(best['cutoff'])}  MAE={best['mae']:.3f} RMSE={best['rmse']:.3f} R2={best['r2']:.3f}")
    else:
        print("\n[BEST 17-35] not selected (too few curves).")

    print(f"\n[SAVED] {out_path}")

if __name__ == "__main__":
    main()
