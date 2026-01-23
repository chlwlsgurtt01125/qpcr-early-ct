from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

from core.model_features import build_xy_from_master_long
from core.model_store import save_model

DATA = Path("data/canonical/master_long.parquet")

def train_one_cutoff(df_long: pd.DataFrame, cutoff: int, seed: int = 42) -> tuple[dict, pd.DataFrame]:
    X, y, meta, feat_cols = build_xy_from_master_long(df_long, cutoff=cutoff)
    groups = meta["run_id"].astype(str).to_numpy()

    # run 기준 split (재현성/누설 방지)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]

    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=feat_cols)
    dte = xgb.DMatrix(Xte, label=yte, feature_names=feat_cols)

    params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        max_depth=6,
        eta=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        seed=seed,
        tree_method="hist",
        device="cuda",
    )

    bst = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=2000,
        evals=[(dtr, "train"), (dte, "test")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    pred = bst.predict(dte)
    mae = float(np.mean(np.abs(pred - yte)))
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))

    # 저장
    save_model(
        bst,
        cutoff=cutoff,
        feat_cols=feat_cols,
        extra={"mae_test": mae, "rmse_test": rmse, "n_curves": int(len(meta)), "n_runs": int(meta["run_id"].nunique())},
    )

    metrics = {
        "cutoff": int(cutoff),
        "mae_test": mae,
        "rmse_test": rmse,
        "n_curves": int(len(meta)),
        "n_runs": int(meta["run_id"].nunique()),
    }

    # --- per-sample prediction log (for Hard Review) ---
    meta_te = meta.iloc[te_idx].copy()

    # well id 컬럼 표준화
    well_col = None
    for cand in ["well_id", "Well", "well", "WELL"]:
        if cand in meta_te.columns:
            well_col = cand
            break
    if well_col is None:
        # fallback: 이름에 "well" 포함된 첫 컬럼
        for c in meta_te.columns:
            if "well" in str(c).lower():
                well_col = c
                break

    if well_col is None:
        # well 정보가 없으면 빈 값으로라도 맞춰줌
        pred_df = meta_te[["run_id"]].copy()
        pred_df["well_id"] = ""
    else:
        pred_df = meta_te[["run_id", well_col]].copy()
        if well_col != "well_id":
            pred_df = pred_df.rename(columns={well_col: "well_id"})

    pred_df["cutoff"] = int(cutoff)
    pred_df["true_ct"] = yte.astype(float)
    pred_df["pred_ct"] = pred.astype(float)

    return metrics, pred_df[["run_id", "well_id", "cutoff", "true_ct", "pred_ct"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_cutoff", type=int, default=1)
    ap.add_argument("--max_cutoff", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df_long = pd.read_parquet(DATA)
    metrics_rows = []
    pred_rows = []
    for c in range(args.min_cutoff, args.max_cutoff + 1):
        try:
            r, pred_df = train_one_cutoff(df_long, cutoff=c, seed=args.seed)
            print(f"[OK] cutoff={c:2d}  n_curves={r['n_curves']}  MAE={r['mae_test']:.3f}  RMSE={r['rmse_test']:.3f}")
            metrics_rows.append(r)
            pred_rows.append(pred_df)

        except Exception as e:
            print(f"[SKIP] cutoff={c:2d}  reason={e}")

    out = Path("data/models/train_report.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(out, index=False)
    print(f"[SAVED] {out}")
    
    # --- save parquet reports for Streamlit pages ---
    model_id = "model_server_latest_xgb"
    app_reports = Path(__file__).resolve().parents[1] / "app" / "data" / "reports" / model_id
    app_reports.mkdir(parents=True, exist_ok=True)

    rep = pd.DataFrame(metrics_rows).sort_values("cutoff").reset_index(drop=True)
    rep.to_parquet(app_reports / "metrics_by_cutoff.parquet", index=False)

    if pred_rows:
        pred_long = pd.concat(pred_rows, ignore_index=True)
        pred_long.to_parquet(app_reports / "predictions_long.parquet", index=False)

    # active model 지정(선택이지만 강력 추천)
    app_models = Path(__file__).resolve().parents[1] / "app" / "models"
    app_models.mkdir(parents=True, exist_ok=True)
    (app_models / "active_model.txt").write_text(model_id)

    print(f"[WROTE] {app_reports/'metrics_by_cutoff.parquet'}")
    print(f"[WROTE] {app_reports/'predictions_long.parquet'}")

if __name__ == "__main__":
    main()
