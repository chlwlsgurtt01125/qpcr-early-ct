# core/step2_eval_cutoffs.py
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb

ROOT = Path("/home/cphotonic/qpcr_v2")
DATA = ROOT / "data" / "canonical" / "master_long.parquet"
OUT_DIR = ROOT / "data" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def build_curve_matrix(df: pd.DataFrame, cutoff: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build X, y for curves that have >= cutoff cycles and have Cq.
    X: (n_curves, cutoff) fluorescence vector (Cycle 1..cutoff)
    y: (n_curves,) Cq
    meta: dataframe with well_uid, run_id, max_cycle
    """
    # only rows up to cutoff
    sub = df[df["Cycle"] <= cutoff].copy()

    # curve eligibility: has all cycles 1..cutoff
    counts = sub.groupby("well_uid")["Cycle"].nunique()
    eligible = counts[counts >= cutoff].index

    sub = sub[sub["well_uid"].isin(eligible)]

    # target Cq per curve (should be constant per well_uid)
    cq = df.groupby("well_uid")["Cq"].first()
    cq = cq.loc[eligible]
    cq = cq.dropna()  # require Cq
    eligible2 = cq.index

    sub = sub[sub["well_uid"].isin(eligible2)]

    # pivot to matrix
    piv = sub.pivot_table(index="well_uid", columns="Cycle", values="Fluor", aggfunc="first")
    piv = piv.reindex(columns=list(range(1, cutoff + 1)))
    piv = piv.dropna(axis=0)

    y = cq.loc[piv.index].to_numpy(dtype=float)
    X_raw = piv.to_numpy(dtype=float)

    # ---- normalize copy (do NOT replace raw) ----
    Xn = X_raw.copy()
    b = min(5, cutoff)
    s = min(10, cutoff)

    baseline = Xn[:, :b].mean(axis=1, keepdims=True)
    Xn = Xn - baseline
    scale = np.maximum(1e-6, np.std(Xn[:, :s], axis=1, keepdims=True))
    Xn = Xn / scale

    # ---- delta features on normalized ----
    d1 = np.diff(Xn, axis=1)
    d2 = np.diff(d1, axis=1)

    # ---- final feature: raw + normalized + deltas ----
    X = np.hstack([X_raw, Xn, d1, d2])

    meta = df.groupby("well_uid")[["run_id"]].first().loc[piv.index].reset_index()
    max_cycle = df.groupby("well_uid")["Cycle"].max().loc[piv.index].to_numpy()
    meta["max_cycle"] = max_cycle
    return X, y, meta



def split_by_run(meta: pd.DataFrame, test_size=0.2, val_size=0.2, seed=42):
    """
    Split curves by run_id groups (no leakage).
    First split train+val vs test, then split train vs val.
    """
    groups = meta["run_id"].to_numpy()
    idx = np.arange(len(meta))

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(idx, groups=groups))

    # split trainval into train/val
    meta_tv = meta.iloc[trainval_idx]
    groups_tv = meta_tv["run_id"].to_numpy()
    idx_tv = np.arange(len(meta_tv))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + 1)
    train_idx2, val_idx2 = next(gss2.split(idx_tv, groups=groups_tv))

    train_idx = trainval_idx[train_idx2]
    val_idx = trainval_idx[val_idx2]
    return train_idx, val_idx, test_idx


def fit_predict_xgb(X_train, y_train, X_val, y_val, use_gpu=True):
    params = {
        "objective": "reg:squarederror",
        "max_depth": 4,
        "eta": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "min_child_weight": 5,
        "eval_metric": "rmse",
    }

    if use_gpu:
        # 최신 xgboost에서는 tree_method="hist" + device="cuda"가 권장
        params.update({"tree_method": "hist", "device": "cuda"})
    else:
        params.update({"tree_method": "hist", "device": "cpu"})

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=80,
        verbose_eval=False,
    )
    return bst


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def main():
    df = pd.read_parquet(DATA)

    # 기본 정리
    df = df.dropna(subset=["well_uid", "run_id", "Cycle", "Fluor"])
    # Cq 없는 curve도 있을 수 있으니, 평가용 build_curve_matrix에서 처리
    df["Cycle"] = df["Cycle"].astype(int)

    cutoffs = list(range(5, 41, 5))  # 5,10,...,40
    rows = []

    for cutoff in cutoffs:
        X, y, meta = build_curve_matrix(df, cutoff)
        n = len(y)
        if n < 10:
            rows.append({"cutoff": cutoff, "n_curves": n, "mae": np.nan, "rmse": np.nan, "r2": np.nan})
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
    out_path = OUT_DIR / "step2_metrics_by_cutoff.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 자동 best cutoff 선택 (MAE 최소, n_curves 충분한 것 중)
    out2 = out.dropna(subset=["mae"]).copy()
    out2 = out2[out2["n_curves"] >= 10]
    if len(out2) > 0:
        best = out2.sort_values(["mae", "cutoff"]).iloc[0]
        best_path = OUT_DIR / "step2_best_cutoff.txt"
        best_path.write_text(f"best_cutoff={int(best['cutoff'])}\nmae={best['mae']}\nrmse={best['rmse']}\nr2={best['r2']}\n", encoding="utf-8")
        print(f"\n[BEST] cutoff={int(best['cutoff'])}  MAE={best['mae']:.3f} RMSE={best['rmse']:.3f} R2={best['r2']:.3f}")
    else:
        print("\n[BEST] not selected (too few curves).")

    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
