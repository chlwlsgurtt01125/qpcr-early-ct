from __future__ import annotations
import numpy as np
import pandas as pd

def build_xy_from_master_long(df_long: pd.DataFrame, cutoff: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """
    master_long.parquet 형태:
      cols: Cycle, Fluor, Well, run_id, channel, Cq, well_uid
    cutoff 사이클까지의 fluorescence를 feature로 만들어 Ct(Cq)를 예측.
    """
    need_cols = {"well_uid", "run_id", "Cycle", "Fluor", "Cq"}
    missing = need_cols - set(df_long.columns)
    if missing:
        raise ValueError(f"Missing columns in df_long: {missing}")

    # cutoff 이상 cycle이 있는 curve만 사용
    g = df_long.groupby("well_uid")["Cycle"].max()
    eligible_uids = g[g >= cutoff].index
    df = df_long[df_long["well_uid"].isin(eligible_uids)].copy()

    # pivot: rows=well_uid, cols=cycle_1..cycle_cutoff
    df = df[df["Cycle"].between(1, cutoff)]
    wide = (
        df.pivot_table(index="well_uid", columns="Cycle", values="Fluor", aggfunc="mean")
        .sort_index(axis=1)
    )

    # 일부 cycle이 비는 케이스 대비: forward fill 후 남은 NaN은 0
    wide = wide.ffill(axis=1).fillna(0.0)

    # y와 meta (wide.index 순서에 맞춰 정렬)
    meta = (
        df_long.drop_duplicates("well_uid")[["well_uid", "run_id", "Cq"]]
        .set_index("well_uid")
        .loc[wide.index]
        .reset_index()
        .rename(columns={"Cq": "true_ct"})
    )

    # ✅ true_ct 정리: 문자열/Undetermined -> NaN 처리, inf 제거, NaN 제거
    meta["true_ct"] = pd.to_numeric(meta["true_ct"], errors="coerce")
    meta = meta.replace([np.inf, -np.inf], np.nan).dropna(subset=["true_ct"])
    meta = meta[(meta["true_ct"] > 0) & (meta["true_ct"] < 60)]

    # ✅ meta에서 살아남은 well_uid 기준으로 wide도 줄여서 X-y 정렬 맞춤
    keep_uids = meta["well_uid"].values
    wide = wide.loc[keep_uids]

    # baseline correction + normalize (각 curve별 1~5cycle baseline)
    arr = wide.to_numpy(dtype=np.float32)
    b = np.mean(arr[:, : min(5, arr.shape[1])], axis=1, keepdims=True)
    arr = arr - b
    denom = (np.max(arr, axis=1, keepdims=True) - np.min(arr, axis=1, keepdims=True)) + 1e-6
    arr = arr / denom

    # 파생 feature 추가
    x = arr
    slope = (x[:, min(10, cutoff) - 1] - x[:, 0]).reshape(-1, 1)
    area = np.sum(x, axis=1, keepdims=True)
    mx = np.max(x, axis=1, keepdims=True)
    amx = (np.argmax(x, axis=1).reshape(-1, 1) / float(cutoff)).astype(np.float32)

    X = np.concatenate([x, slope, area, mx, amx], axis=1)
    y = meta["true_ct"].to_numpy(dtype=np.float32)

    feat_cols = [f"f_cycle_{i}" for i in range(1, cutoff + 1)] + ["f_slope10", "f_area", "f_max", "f_argmax"]
    return X, y, meta, feat_cols


def build_x_from_long(df_long: pd.DataFrame, cutoff: int) -> tuple[np.ndarray, pd.DataFrame]:
    """
    업로드한 단일 run(또는 여러 run) long df에서 X만 만든다.
    Cq가 없을 수도 있으니 meta에는 well_uid/run_id/Well만 최대한 담는다.
    """
    need_cols = {"Cycle", "Fluor", "Well", "run_id", "well_uid"}
    missing = need_cols - set(df_long.columns)
    if missing:
        raise ValueError(f"Missing columns in df_long: {missing}")

    g = df_long.groupby("well_uid")["Cycle"].max()
    eligible_uids = g[g >= cutoff].index
    df = df_long[df_long["well_uid"].isin(eligible_uids)].copy()
    df = df[df["Cycle"].between(1, cutoff)]

    wide = (
        df.pivot_table(index="well_uid", columns="Cycle", values="Fluor", aggfunc="mean")
        .sort_index(axis=1)
    )
    wide = wide.ffill(axis=1).fillna(0.0)

    arr = wide.to_numpy(dtype=np.float32)
    b = np.mean(arr[:, : min(5, arr.shape[1])], axis=1, keepdims=True)
    arr = arr - b
    denom = (np.max(arr, axis=1, keepdims=True) - np.min(arr, axis=1, keepdims=True)) + 1e-6
    arr = arr / denom

    x = arr
    slope = (x[:, min(10, cutoff)-1] - x[:, 0]).reshape(-1, 1)
    area = np.sum(x, axis=1, keepdims=True)
    mx = np.max(x, axis=1, keepdims=True)
    amx = np.argmax(x, axis=1).reshape(-1, 1) / float(cutoff)
    X = np.concatenate([x, slope, area, mx, amx], axis=1)

    # meta
    base = df_long.drop_duplicates("well_uid").set_index("well_uid").loc[wide.index]
    cols = ["well_uid", "run_id", "Well"]
    meta = base.reset_index()[cols]
    if "Cq" in base.columns:
        meta["true_ct"] = base["Cq"].values
    return X, meta
