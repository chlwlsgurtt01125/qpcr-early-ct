# app/streamlit_app.py
from __future__ import annotations

import os
import io
import re
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
import urllib.request
import urllib.error
from typing import Dict, Tuple, Optional, List  # ? 추가
from dataclasses import dataclass
from enum import Enum

import plotly.express as px
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import pyarrow.dataset as ds
from scipy.optimize import curve_fit
from scipy.stats import linregress
import altair as alt

# ? set_page_config는 반드시 import 직후, 다른 st 호출 전에!
st.set_page_config(page_title="CPHOTONICS | Early Ct Predictor", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ? 경로는 PROJECT_ROOT 기준으로
ASSETS_DIR = PROJECT_ROOT / "assets"
CATALOG_PATH = ASSETS_DIR / "data_catalog.json"
QC_DIR = PROJECT_ROOT / "outputs" / "qc"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "qc_performance_analysis"
MODELS_DIR = PROJECT_ROOT / "data" / "models" / "by_cutoff"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"


# ============================================
# Hard Sample 버킷 분류 관련 클래스/상수
# ============================================

class HardBucket(Enum):
    """Hard Sample 버킷 종류"""
    LATE_AMP = "late_amp"
    HIGH_RANGE = "high_range"
    NOISY = "noisy"
    NON_SIGMOID = "non_sigmoid"
    UNKNOWN = "unknown"
    NORMAL = "normal"


@dataclass
class BucketResult:
    """버킷 분류 결과"""
    bucket: HardBucket
    confidence: float
    details: Dict
    is_hard: bool


BUCKET_COLORS = {
    "late_amp": "#FFD700",
    "high_range": "#FF4444",
    "noisy": "#FFA500",
    "non_sigmoid": "#9370DB",
    "unknown": "#808080",
    "normal": "#00CC66",
    "error": "#000000"
}

BUCKET_EMOJI = {
    "late_amp": "??",
    "high_range": "??",
    "noisy": "??",
    "non_sigmoid": "??",
    "unknown": "?",
    "normal": "??",
    "error": "?"
}


# ============================================
# Sigmoid Fitting 함수 (한 번만 정의)
# ============================================

def sigmoid_4pl(x, a, b, c, d):
    """4-Parameter Logistic Sigmoid"""
    return d + (a - d) / (1 + (x / c) ** b)


def fit_sigmoid(cycles: np.ndarray, fluor: np.ndarray) -> Tuple[float, np.ndarray, Dict]:
    """
    Sigmoid fitting 수행
    Returns: (r2, fitted_values, params_dict)
    """
    try:
        a_init = np.min(fluor)
        d_init = np.max(fluor)
        c_init = cycles[len(cycles) // 2]
        b_init = 1.0
        
        popt, _ = curve_fit(
            sigmoid_4pl, cycles, fluor,
            p0=[a_init, b_init, c_init, d_init],
            bounds=([0, 0.1, 1, 0], [np.inf, 50, 100, np.inf]),
            maxfev=5000
        )
        
        fitted = sigmoid_4pl(cycles, *popt)
        params = {"a": popt[0], "b": popt[1], "c": popt[2], "d": popt[3]}
        
        ss_res = np.sum((fluor - fitted) ** 2)
        ss_tot = np.sum((fluor - np.mean(fluor)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return r2, fitted, params
        
    except Exception as e:
        return 0.0, np.zeros_like(fluor), {"error": str(e)}


# ============================================
# 버킷 판정 함수들
# ============================================

def check_late_amplification(true_ct: Optional[float], pred_ct: float, threshold: float = 35.0) -> Tuple[bool, float, Dict]:
    """Late Amplification 체크"""
    ct_value = true_ct if true_ct is not None else pred_ct
    is_late = ct_value > threshold
    confidence = min(1.0, (ct_value - threshold) / 10.0) if is_late else 0.0
    
    return is_late, confidence, {
        "true_ct": true_ct, "pred_ct": pred_ct, 
        "threshold": threshold, "ct_used": ct_value
    }


def check_high_range(fluor: np.ndarray, max_thr: float = 50000, min_thr: float = -100) -> Tuple[bool, float, Dict]:
    """과대 레인지 체크"""
    f_max, f_min = np.max(fluor), np.min(fluor)
    is_high = f_max > max_thr or f_min < min_thr
    
    confidence = 0.0
    if f_max > max_thr:
        confidence = max(confidence, min(1.0, (f_max - max_thr) / max_thr))
    if f_min < min_thr:
        confidence = max(confidence, min(1.0, abs(f_min - min_thr) / 1000))
    
    return is_high, confidence, {
        "fluor_max": float(f_max), "fluor_min": float(f_min),
        "max_threshold": max_thr, "min_threshold": min_thr
    }


def check_noisy(cycles: np.ndarray, fluor: np.ndarray, cutoff: int, 
                cv_thr: float = 0.15, snr_thr: float = 3.0) -> Tuple[bool, float, Dict]:
    """노이즈 체크 (CV, SNR)"""
    early_mask = cycles <= cutoff
    early_fluor = fluor[early_mask]
    
    if len(early_fluor) < 3:
        return False, 0.0, {"error": "early 구간 데이터 부족"}
    
    early_mean = np.mean(early_fluor)
    early_std = np.std(early_fluor)
    cv = early_std / early_mean if early_mean != 0 else 0
    
    late_fluor = fluor[cycles > cutoff] if np.any(cycles > cutoff) else early_fluor
    signal = np.max(late_fluor) - np.mean(early_fluor)
    noise = early_std if early_std > 0 else 1e-6
    snr = signal / noise
    
    is_high_cv = cv > cv_thr
    is_low_snr = snr < snr_thr
    is_noisy = is_high_cv or is_low_snr
    
    confidence = 0.0
    if is_high_cv:
        confidence = max(confidence, min(1.0, (cv - cv_thr) / cv_thr))
    if is_low_snr:
        confidence = max(confidence, min(1.0, (snr_thr - snr) / snr_thr))
    
    return is_noisy, confidence, {
        "cv_early": float(cv), "cv_threshold": cv_thr,
        "snr": float(snr), "snr_threshold": snr_thr
    }


def check_non_sigmoid(cycles: np.ndarray, fluor: np.ndarray, r2_thr: float = 0.95) -> Tuple[bool, float, Dict]:
    """비시그모이드 체크"""
    r2, fitted, params = fit_sigmoid(cycles, fluor)
    
    baseline = np.percentile(fluor, 10)
    plateau = np.percentile(fluor, 90)
    fold_change = plateau / baseline if baseline > 0 else 1.0
    
    slope, _, r_value, _, _ = linregress(cycles, fluor)
    is_increasing = slope > 0 and r_value > 0.5
    
    is_low_r2 = r2 < r2_thr
    is_flat = fold_change < 2.0
    is_non_sigmoid = is_low_r2 or is_flat or (not is_increasing)
    
    confidence = 0.0
    if is_low_r2:
        confidence = max(confidence, min(1.0, (r2_thr - r2) / r2_thr))
    if is_flat:
        confidence = max(confidence, min(1.0, (2.0 - fold_change) / 2.0))
    
    return is_non_sigmoid, confidence, {
        "r2_sigmoid": float(r2), "r2_threshold": r2_thr,
        "fold_change": float(fold_change), "slope": float(slope)
    }


def classify_hard_sample(
    curve_df: pd.DataFrame,
    true_ct: Optional[float],
    pred_ct: float,
    abs_error: float,
    cutoff: int,
    error_threshold: float = 2.0,
    late_amp_threshold: float = 35.0,
    fluor_max_threshold: float = 50000,
    fluor_min_threshold: float = -100,
    cv_threshold: float = 0.15,
    snr_threshold: float = 3.0,
    r2_threshold: float = 0.95,
) -> BucketResult:
    """Hard Sample 통합 분류"""
    
    is_hard = abs_error >= error_threshold
    if not is_hard:
        return BucketResult(HardBucket.NORMAL, 0.0, {"abs_error": abs_error}, False)
    
    cycles = curve_df["Cycle"].values.astype(float)
    fluor = curve_df["Fluor"].values.astype(float)
    sort_idx = np.argsort(cycles)
    cycles, fluor = cycles[sort_idx], fluor[sort_idx]
    
    checks = []
    
    is_late, conf_late, det_late = check_late_amplification(true_ct, pred_ct, late_amp_threshold)
    checks.append((HardBucket.LATE_AMP, is_late, conf_late, det_late))
    
    is_high, conf_high, det_high = check_high_range(fluor, fluor_max_threshold, fluor_min_threshold)
    checks.append((HardBucket.HIGH_RANGE, is_high, conf_high, det_high))
    
    is_noisy, conf_noisy, det_noisy = check_noisy(cycles, fluor, cutoff, cv_threshold, snr_threshold)
    checks.append((HardBucket.NOISY, is_noisy, conf_noisy, det_noisy))
    
    is_non_sig, conf_non_sig, det_non_sig = check_non_sigmoid(cycles, fluor, r2_threshold)
    checks.append((HardBucket.NON_SIGMOID, is_non_sig, conf_non_sig, det_non_sig))
    
    triggered = [(b, c, d) for b, is_triggered, c, d in checks if is_triggered]
    
    if triggered:
        triggered.sort(key=lambda x: x[1], reverse=True)
        best_bucket, best_conf, best_details = triggered[0]
        
        all_details = {
            "primary_bucket": best_bucket.value,
            "all_checks": {
                "late_amp": det_late, "high_range": det_high,
                "noisy": det_noisy, "non_sigmoid": det_non_sig
            },
            "triggered_buckets": [b.value for b, _, _ in triggered],
            "abs_error": abs_error
        }
        
        return BucketResult(best_bucket, best_conf, all_details, True)
    
    return BucketResult(
        HardBucket.UNKNOWN, 0.5,
        {"abs_error": abs_error, "note": "No specific pattern detected"},
        True
    )


def get_bucket_recommendations(bucket: str) -> Dict:
    """버킷별 원인 및 대응 전략"""
    recommendations = {
        "late_amp": {
            "원인": "템플릿 농도가 매우 낮거나, 증폭 효율이 떨어짐",
            "모델 특징": "Early cycle에서 신호 변화가 거의 없어 예측이 어려움",
            "대응 전략": [
                "Late Ct 샘플은 별도 모델 또는 threshold 적용 고려",
                "Ct > 35 샘플은 예측 신뢰도 경고 표시",
                "재검사 또는 희석 후 재검사 권장"
            ]
        },
        "high_range": {
            "원인": "장비 캘리브레이션 문제, 샘플 오염, 또는 데이터 전처리 오류",
            "모델 특징": "비정상적인 Fluor 범위로 feature 값이 왜곡됨",
            "대응 전략": [
                "원본 데이터와 정합성 확인 필요",
                "장비 캘리브레이션 상태 점검",
                "해당 샘플 제외 후 재분석 고려"
            ]
        },
        "noisy": {
            "원인": "낮은 시그널, 장비 노이즈, 또는 샘플 품질 문제",
            "모델 특징": "Early 구간 변동이 커서 feature 추출이 불안정",
            "대응 전략": [
                "SNR 기반 품질 필터링 강화",
                "Smoothing 전처리 적용 고려",
                "Low-quality 샘플 재검사 권장"
            ]
        },
        "non_sigmoid": {
            "원인": "비정상 증폭 (억제, 비특이적 증폭, primer-dimer 등)",
            "모델 특징": "정상 S-curve 가정이 깨져 예측 정확도 저하",
            "대응 전략": [
                "Sigmoid R² 기반 품질 필터 적용",
                "Melting curve 분석으로 특이성 확인",
                "Primer 재설계 또는 조건 최적화"
            ]
        },
        "unknown": {
            "원인": "명확한 패턴 없이 예측 오차 발생",
            "모델 특징": "기존 버킷으로 설명되지 않는 오차",
            "대응 전략": [
                "개별 사례 심층 분석 필요",
                "새로운 오류 패턴 발굴 기회",
                "추가 feature 엔지니어링 검토"
            ]
        }
    }
    return recommendations.get(bucket, {"원인": "Unknown", "모델 특징": "Unknown", "대응 전략": []})


# ============================================
# 유틸리티 함수들
# ============================================

def load_data_catalog(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        st.error(f"Failed to read data_catalog.json: {e}")
        return {}


catalog = load_data_catalog(CATALOG_PATH)


def find_file_url_in_catalog(catalog: dict, filename: str) -> str | None:
    for item in catalog.get("files", []):
        if item.get("filename") == filename:
            return item.get("url")
    return None


def download_to_path(url: str, dst_path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dst_path)


def ensure_asset_download(url: str, dst_path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() and dst_path.stat().st_size > 0:
        return False
    with urllib.request.urlopen(url) as r, open(dst_path, "wb") as f:
        f.write(r.read())
    return True


def running_on_streamlit_cloud() -> bool:
    return str(PROJECT_ROOT).startswith("/mount/src") or os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud"


def get_reports_root() -> Path:
    p = Path("reports")
    if p.exists():
        return p
    p2 = Path(__file__).resolve().parent / "data" / "reports"
    if p2.exists():
        return p2
    return Path("reports")


REPORTS_ROOT = get_reports_root()


def has_canonical_master_long() -> bool:
    return (PROJECT_ROOT / "data" / "canonical" / "master_long.parquet").exists()


def get_active_model_id() -> str:
    p = REPORTS_ROOT / "active_model.txt"
    mid = p.read_text().strip() if p.exists() else "model_server_latest_xgb"
    mid = Path(mid).name
    return mid


def get_best_cutoff_from_report() -> int | None:
    report_path = REPORTS_ROOT / "train_report.csv"
    if not report_path.exists():
        return None
    rep = pd.read_csv(report_path)
    cols = {str(c).lower(): c for c in rep.columns}
    cutoff_col = cols.get("cutoff")
    mae_col = cols.get("mae") or cols.get("mae_test")
    if not cutoff_col or not mae_col or rep.empty:
        return None
    best_row = rep.loc[rep[mae_col].idxmin()]
    return int(best_row[cutoff_col])


def normalize_well(x: object) -> str:
    """B2, b2, ' B2 '  -> 'B02'"""
    s = str(x).strip().upper()
    m = re.fullmatch(r"([A-H])\s*0*([0-9]{1,2})", s)
    if not m:
        return s
    row = m.group(1)
    col = int(m.group(2))
    return f"{row}{col:02d}"


def _safe_stem(name: str) -> str:
    s = Path(name).stem
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)
    return s[:80] if s else "uploaded"


def discover_cutoffs(models_dir: Path) -> list[int]:
    cutoffs: list[int] = []
    for p in models_dir.glob("ct_xgb_cutoff_*.json"):
        m = re.search(r"cutoff_(\d+)\.json$", p.name)
        if m:
            cutoffs.append(int(m.group(1)))
    return sorted(set(cutoffs))


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if str(c).strip().lower().startswith("unnamed")]
    return df.drop(columns=cols) if cols else df


def _line_y_eq_x(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame({"x":[0,1], "y":[0,1]})
    lo = float(min(df["true_ct"].min(), df["pred_ct"].min()))
    hi = float(max(df["true_ct"].max(), df["pred_ct"].max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    lo -= pad; hi += pad
    return pd.DataFrame({"x":[lo, hi], "y":[lo, hi]})


# ============================================
# 모델 로딩 관련
# ============================================

@st.cache_resource
def load_booster(cutoff: int) -> xgb.Booster:
    model_path = MODELS_DIR / f"ct_xgb_cutoff_{cutoff}.json"
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return booster


def load_meta(cutoff: int) -> dict:
    meta_path = MODELS_DIR / f"ct_xgb_cutoff_{cutoff}.meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


# ============================================
# Feature Engineering (? 누락되었던 함수 추가)
# ============================================

def build_x_from_long(df_long: pd.DataFrame, cutoff: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    df_long (Cycle, Fluor, Well, run_id, well_uid)에서
    cutoff cycle까지의 feature를 추출하여 X matrix 생성
    
    Returns:
        X: (n_wells, n_features) numpy array
        meta: DataFrame with run_id, Well, well_uid
    """
    df = df_long[df_long["Cycle"] <= cutoff].copy()
    
    index_cols = ["run_id", "Well"]
    if "well_uid" in df.columns:
        index_cols.append("well_uid")
    
    pivot = df.pivot_table(
        index=index_cols,
        columns="Cycle",
        values="Fluor",
        aggfunc="first"
    ).reset_index()
    
    feat_cols = [c for c in pivot.columns if isinstance(c, (int, float)) or (isinstance(c, str) and c.isdigit())]
    feat_cols = sorted(feat_cols, key=lambda x: int(x))
    
    meta = pivot[["run_id", "Well"]].copy()
    if "well_uid" in pivot.columns:
        meta["well_uid"] = pivot["well_uid"]
    
    X = pivot[feat_cols].values.astype(float)
    X = pd.DataFrame(X).ffill(axis=1).bfill(axis=1).values
    
    return X, meta


# ============================================
# 데이터 변환 함수들
# ============================================

def infer_long_df(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """업로드 테이블을 long 형태로 변환"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = _drop_unnamed(df)

    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    has_cycle = "cycle" in cols_lower

    fluor_key = None
    for k in ("fluor", "rfu", "signal"):
        if k in cols_lower:
            fluor_key = k
            break

    if has_cycle and fluor_key is not None:
        cycle_col = cols_lower["cycle"]
        fluor_col = cols_lower[fluor_key]
        well_col = (
            cols_lower.get("well")
            or cols_lower.get("well position")
            or cols_lower.get("well_position")
        )

        if well_col is None:
            df["Well"] = [f"R{i:03d}" for i in range(1, len(df) + 1)]
            well_col = "Well"

        out = df[[well_col, cycle_col, fluor_col]].copy()
        out.columns = ["Well", "Cycle", "Fluor"]

    elif has_cycle:
        cycle_col = cols_lower["cycle"]
        well_cols = [c for c in df.columns if c != cycle_col]
        if not well_cols:
            raise ValueError("Cycle 컬럼은 있는데 well 컬럼이 없어.")

        long = df.melt(
            id_vars=[cycle_col],
            value_vars=well_cols,
            var_name="Well",
            value_name="Fluor",
        )
        long.rename(columns={cycle_col: "Cycle"}, inplace=True)
        out = long[["Well", "Cycle", "Fluor"]].copy()

    else:
        well_col = None
        for cand in ["Well", "well", "WELL"]:
            if cand in df.columns:
                well_col = cand
                break
        if well_col is None:
            raise ValueError("Well 컬럼을 찾지 못했어.")

        cycle_cols: list[str] = []
        for c in df.columns:
            if c == well_col:
                continue
            if re.fullmatch(r"\d+", str(c).strip()):
                cycle_cols.append(c)
            elif re.search(r"cycle\s*\d+", str(c).strip(), flags=re.IGNORECASE):
                cycle_cols.append(c)

        if not cycle_cols:
            raise ValueError("long도 아니고 wide도 아닌 것 같아.")

        tmp = df[[well_col] + cycle_cols].copy()
        long = tmp.melt(id_vars=[well_col], var_name="Cycle", value_name="Fluor")
        long["Cycle"] = long["Cycle"].astype(str).str.extract(r"(\d+)").astype(int)
        long.rename(columns={well_col: "Well"}, inplace=True)
        out = long[["Well", "Cycle", "Fluor"]].copy()

    out = out.dropna(subset=["Well", "Cycle", "Fluor"]).copy()
    out["Cycle"] = pd.to_numeric(out["Cycle"], errors="coerce")
    out["Fluor"] = pd.to_numeric(out["Fluor"], errors="coerce")
    out = out.dropna(subset=["Cycle", "Fluor"]).copy()

    out["Cycle"] = out["Cycle"].astype(int)
    out["Well"] = out["Well"].astype(str).str.strip()

    out["run_id"] = run_id
    out["well_uid"] = out["run_id"].astype(str) + ":" + out["Well"].astype(str)

    return out[["Cycle", "Fluor", "Well", "run_id", "well_uid"]]


def predict_ct(df_long: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    booster = load_booster(cutoff)
    X, meta = build_x_from_long(df_long, cutoff=cutoff)

    m = load_meta(cutoff)
    feat_cols = m.get("feat_cols") or m.get("feature_cols")
    if feat_cols:
        dmat = xgb.DMatrix(X, feature_names=list(feat_cols))
    else:
        dmat = xgb.DMatrix(X)

    pred = booster.predict(dmat)

    out = meta.copy()
    out["pred_ct"] = pred.astype(float)
    out["cutoff_used"] = cutoff
    return out.sort_values(["run_id", "Well"]).reset_index(drop=True)


def split_excel_sheets(obj):
    """엑셀 시트 분리"""
    if not isinstance(obj, dict):
        return obj, None, None, None

    curve_priority = ["SYBR", "Amplification", "Data", "Raw"]
    truth_priority = ["Sheet1", "Ct", "Cq", "Truth", "Result"]

    def norm_cols(df):
        return [str(c).strip().lower() for c in df.columns]

    curve_df = None
    curve_sheet = None

    for nm in curve_priority:
        if nm in obj:
            cols = norm_cols(obj[nm])
            if "cycle" in cols:
                curve_df = obj[nm]
                curve_sheet = nm
                break

    if curve_df is None:
        for nm, df in obj.items():
            cols = norm_cols(df)
            if "cycle" in cols:
                curve_df = df
                curve_sheet = nm
                break

    if curve_df is None:
        curve_sheet = next(iter(obj.keys()))
        curve_df = obj[curve_sheet]

    truth_df = None
    truth_sheet = None
    truth_keys = {"cq", "ct", "true_ct", "truect"}

    for nm in truth_priority:
        if nm in obj:
            cols = set(norm_cols(obj[nm]))
            if "well" in cols and len(cols & truth_keys) > 0:
                truth_df = obj[nm]
                truth_sheet = nm
                break

    if truth_df is None:
        for nm, df in obj.items():
            cols = set(norm_cols(df))
            if "well" in cols and len(cols & truth_keys) > 0:
                truth_df = df
                truth_sheet = nm
                break

    return curve_df, truth_df, curve_sheet, truth_sheet


def read_uploaded_table(up):
    name = (up.name or "").lower()
    raw = up.getvalue() if hasattr(up, "getvalue") else up.read()
    buf = io.BytesIO(raw)

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(buf, sheet_name=None)
    return pd.read_csv(buf)


# ============================================
# Curve 로딩 함수들
# ============================================

def load_curve_from_master(run_id: str, well_id: str) -> pd.DataFrame:
    """canonical master_long.parquet에서 곡선 로드"""
    path = PROJECT_ROOT / "data" / "canonical" / "master_long.parquet"
    if not path.exists():
        raise FileNotFoundError(f"master_long.parquet not found: {path}")

    dataset = ds.dataset(str(path))
    cols = set(dataset.schema.names)
    
    cycle_col = "Cycle" if "Cycle" in cols else ("cycle" if "cycle" in cols else None)
    fluor_col = "Fluor" if "Fluor" in cols else ("fluor" if "fluor" in cols else None)
    run_col = "run_id" if "run_id" in cols else None
    
    if "Well" in cols:
        well_col = "Well"
    elif "well_id" in cols:
        well_col = "well_id"
    elif "well_uid" in cols:
        well_col = "well_uid"
    else:
        well_col = None
    
    if not all([cycle_col, fluor_col, run_col, well_col]):
        raise ValueError(f"master_long columns unexpected. found={sorted(cols)[:50]} ...")
    
    well_value = well_id
    if well_col == "well_uid":
        well_value = f"{run_id}:{well_id}"
    
    filt = (ds.field(run_col) == run_id) & (ds.field(well_col) == well_value)
    table = dataset.to_table(filter=filt, columns=[run_col, well_col, cycle_col, fluor_col])
    df = table.to_pandas()
    
    df = df.rename(columns={cycle_col: "Cycle", fluor_col: "Fluor"})
    df = df.sort_values("Cycle").reset_index(drop=True)
    return df


def load_one_curve_from_predictions_row(row) -> pd.DataFrame:
    """predictions_long.parquet row에서 곡선 복원"""
    cycles_json = row.get("curve_cycles_json", "") if isinstance(row, dict) else getattr(row, "curve_cycles_json", "")
    fluor_json = row.get("curve_fluor_json", "") if isinstance(row, dict) else getattr(row, "curve_fluor_json", "")

    if not cycles_json or not fluor_json:
        raise ValueError("curve_cycles_json / curve_fluor_json is empty.")

    cycles = json.loads(cycles_json)
    fluor = json.loads(fluor_json)

    df = pd.DataFrame({"Cycle": cycles, "Fluor": fluor})
    df = df.dropna().sort_values("Cycle").reset_index(drop=True)
    return df


# ============================================
# QC 관련
# ============================================

def decision_from_qc(qc_status: str) -> str:
    """QC 상태를 기반으로 운영 결정"""
    qc_status = str(qc_status).upper().strip()
    if qc_status == "PASS":
        return "PREDICT"
    if qc_status == "FLAG":
        return "WARN"
    return "RERUN"


# ============================================
# 시각화 함수들
# ============================================

def perf_accuracy_fraction_vs_cutoff(pred: pd.DataFrame, tol: float = 2.0) -> pd.DataFrame:
    """정확도 비율 계산"""
    df = pred.dropna(subset=["cutoff", "true_ct", "pred_ct"]).copy()
    df["abs_err"] = (df["pred_ct"] - df["true_ct"]).abs()
    out = df.groupby("cutoff").apply(lambda g: (g["abs_err"] <= tol).mean()).reset_index(name="acc_frac")
    out["cutoff"] = out["cutoff"].astype(int)
    return out.sort_values("cutoff")


def plot_pred_vs_true_facets(pred_long: pd.DataFrame, cutoffs: list[int], ncol: int = 4) -> None:
    """Faceted scatter plot"""
    df = pred_long.dropna(subset=["cutoff", "true_ct", "pred_ct"]).copy()
    df["cutoff"] = df["cutoff"].astype(int)
    df = df[df["cutoff"].isin([int(c) for c in cutoffs])].copy()
    if df.empty:
        st.info("선택한 cutoff에 표시할 데이터가 없어요.")
        return

    base = alt.Chart(df).mark_circle(size=60, opacity=0.75).encode(
        x=alt.X("true_ct:Q", title="True Ct/Cq"),
        y=alt.Y("pred_ct:Q", title="Pred Ct/Cq"),
        tooltip=["run_id", "well_id", "cutoff", "true_ct", "pred_ct"],
    )

    diag = (
        alt.Chart(df)
        .transform_aggregate(
            min_true="min(true_ct)",
            min_pred="min(pred_ct)",
            max_true="max(true_ct)",
            max_pred="max(pred_ct)",
            groupby=["cutoff"],
        )
        .transform_calculate(
            lo="datum.min_true < datum.min_pred ? datum.min_true : datum.min_pred",
            hi="datum.max_true > datum.max_pred ? datum.max_true : datum.max_pred",
        )
        .transform_fold(["lo", "hi"], as_=["k", "v"])
        .transform_calculate(x="datum.v", y="datum.v")
        .mark_line()
        .encode(x="x:Q", y="y:Q")
    )

    chart = alt.layer(diag, base).facet(
        facet=alt.Facet("cutoff:N", title=None),
        columns=ncol,
    ).resolve_scale(x="independent", y="independent").properties(title="Pred vs True across Cycle Cutoffs")

    st.altair_chart(chart, use_container_width=True)


def plot_error_by_true_ct_scatter(pred: pd.DataFrame, cutoff: int, tol: float = 2.0, bin_width: float = 2.0) -> None:
    """Bias Plot"""
    df = pred[pred["cutoff"] == int(cutoff)].dropna(subset=["true_ct", "pred_ct"]).copy()
    if df.empty:
        st.info("해당 cutoff에 scatter를 그릴 데이터가 없어요.")
        return

    df["err"] = df["pred_ct"] - df["true_ct"]

    x_min = float(df["true_ct"].min())
    x_max = float(df["true_ct"].max())
    pad = (x_max - x_min) * 0.03 if x_max > x_min else 1.0
    x_min -= pad
    x_max += pad

    band_df = pd.DataFrame({"x": [x_min, x_max], "y1": [-float(tol), -float(tol)], "y2": [float(tol), float(tol)]})

    bw = float(bin_width) if bin_width > 0 else 2.0
    tmp = df[["true_ct", "err"]].copy()
    tmp["bin"] = np.floor(tmp["true_ct"] / bw) * bw
    grp = tmp.groupby("bin").agg(mean_err=("err", "mean"), n=("err", "size")).reset_index().sort_values("bin")
    grp["bin_center"] = grp["bin"] + bw / 2.0

    band = alt.Chart(band_df).mark_area(opacity=0.12).encode(
        x=alt.X("x:Q", title="True Ct/Cq"),
        y=alt.Y("y1:Q", title="Error (pred - true)"),
        y2="y2:Q",
    )

    zero_line = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(strokeDash=[6, 4], opacity=0.6).encode(y="y:Q")

    points = alt.Chart(df).mark_circle(size=55, opacity=0.65).encode(
        x=alt.X("true_ct:Q", title="True Ct/Cq"),
        y=alt.Y("err:Q", title="Error (pred - true)"),
        tooltip=["run_id:N", "well_id:N", "true_ct:Q", "pred_ct:Q", "err:Q"],
    )

    bias_line = alt.Chart(grp).mark_line(point=True, opacity=0.9).encode(
        x=alt.X("bin_center:Q"),
        y=alt.Y("mean_err:Q"),
        tooltip=["bin_center:Q", "mean_err:Q", "n:Q"],
    )

    chart = alt.layer(band, zero_line, points, bias_line).properties(height=340).interactive()
    st.altair_chart(chart, use_container_width=True)


def plot_pred_vs_true_hard_colored(df_cut: pd.DataFrame, hard_ids: set[tuple[str, str]] | None = None,
                                  highlight: tuple[str, str] | None = None) -> None:
    """Hard 샘플 강조 scatter"""
    df = df_cut.dropna(subset=["true_ct", "pred_ct"]).copy()
    if df.empty:
        st.info("scatter를 그릴 데이터가 없어요.")
        return

    if hard_ids is None:
        df["group"] = "Inlier"
    else:
        df["group"] = df.apply(lambda r: "Hard" if (str(r["run_id"]), str(r["well_id"])) in hard_ids else "Inlier", axis=1)

    if highlight is not None:
        hr, hw = highlight
        df["is_selected"] = (df["run_id"].astype(str) == str(hr)) & (df["well_id"].astype(str) == str(hw))
        df.loc[df["is_selected"], "group"] = "Selected"
    else:
        df["is_selected"] = False

    base = alt.Chart(df).mark_circle(size=70, opacity=0.85).encode(
        x=alt.X("true_ct:Q", title="True Ct/Cq"),
        y=alt.Y("pred_ct:Q", title="Pred Ct/Cq"),
        color=alt.Color("group:N", title="Group"),
        tooltip=["run_id", "well_id", "true_ct", "pred_ct", "abs_err"],
    )

    diag = (
        alt.Chart(df)
        .transform_aggregate(min_true="min(true_ct)", min_pred="min(pred_ct)", max_true="max(true_ct)", max_pred="max(pred_ct)")
        .transform_calculate(lo="datum.min_true < datum.min_pred ? datum.min_true : datum.min_pred", hi="datum.max_true > datum.max_pred ? datum.max_true : datum.max_pred")
        .transform_fold(["lo", "hi"], as_=["k", "v"])
        .transform_calculate(x="datum.v", y="datum.v")
        .mark_line()
        .encode(x="x:Q", y="y:Q")
    )

    st.altair_chart(alt.layer(diag, base).properties(height=380, title="Hard Samples highlighted"), use_container_width=True)


def plot_uploaded_curve_preview(df_long: pd.DataFrame, cutoff: int, max_wells: int = 6) -> None:
    """업로드 곡선 미리보기"""
    if df_long.empty:
        st.info("df_long이 비어있어요.")
        return

    wells = sorted(df_long["Well"].dropna().unique().tolist())[:max_wells]
    sub = df_long[df_long["Well"].isin(wells)].copy()
    sub["segment"] = np.where(sub["Cycle"] <= int(cutoff), "early(<=cutoff)", "late")

    chart = alt.Chart(sub).mark_line().encode(
        x=alt.X("Cycle:Q", title="Cycle"),
        y=alt.Y("Fluor:Q", title="Fluor"),
        color=alt.Color("Well:N", legend=alt.Legend(title="Well")),
        tooltip=["Well", "Cycle", "Fluor", "segment"],
    ).properties(height=320).interactive()

    vline = alt.Chart(pd.DataFrame({"Cycle": [int(cutoff)]})).mark_rule(strokeDash=[6, 4]).encode(x="Cycle:Q")

    st.altair_chart(chart + vline, use_container_width=True)
    st.caption(f"미리보기: {len(wells)}개 well만 표시")


def plot_pred_ct_hist(pred_df: pd.DataFrame) -> None:
    """예측 Ct 히스토그램"""
    if pred_df.empty or "pred_ct" not in pred_df.columns:
        return

    hist = alt.Chart(pred_df).mark_bar().encode(
        x=alt.X("pred_ct:Q", bin=alt.Bin(maxbins=25), title="Predicted Ct"),
        y=alt.Y("count():Q", title="Count"),
    ).properties(height=280)
    st.altair_chart(hist, use_container_width=True)


def plot_cv_vs_ct(df_long: pd.DataFrame, pred_df: pd.DataFrame, cutoff: int) -> None:
    """CV vs Ct"""
    if df_long.empty or pred_df.empty:
        return

    early = df_long[df_long["Cycle"] <= int(cutoff)].copy()
    g = early.groupby(["run_id", "Well"])["Fluor"]
    cv = (g.std() / (g.mean().replace(0, np.nan))).reset_index()
    cv.rename(columns={"Fluor": "cv_early"}, inplace=True)

    m = pred_df.merge(cv, on=["run_id", "Well"], how="left")
    m = m.dropna(subset=["pred_ct", "cv_early"]).copy()
    if m.empty:
        st.info("CV vs Ct를 그릴 데이터가 부족해요.")
        return

    scat = alt.Chart(m).mark_circle(size=60).encode(
        x=alt.X("pred_ct:Q", title="Predicted Ct"),
        y=alt.Y("cv_early:Q", title="CV (early <= cutoff)"),
        tooltip=["Well", "pred_ct", "cv_early"],
    ).properties(height=300).interactive()
    st.altair_chart(scat, use_container_width=True)


# ============================================
# 재학습 관련
# ============================================

def run_retrain(min_cutoff: int, max_cutoff: int) -> tuple[int, str]:
    if running_on_streamlit_cloud():
        return 2, "Streamlit Cloud에서는 재학습이 비활성화되어 있습니다."
    
    cmd = [
        sys.executable, "-m", "core.step3_train_and_save_models",
        "--min_cutoff", str(min_cutoff),
        "--max_cutoff", str(max_cutoff),
    ]

    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")

    p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, env=env)
    log = (p.stdout or "") + "\n" + (p.stderr or "")
    return p.returncode, log


def sync_train_report_to_parquet(rep: pd.DataFrame) -> str:
    """train_report.csv를 parquet로 저장"""
    model_id = "model_server_latest_xgb"

    outdir = REPORTS_ROOT / model_id
    outdir.mkdir(parents=True, exist_ok=True)
    (REPORTS_ROOT / "active_model.txt").write_text(model_id, encoding="utf-8")
    
    cols = {str(c).lower(): c for c in rep.columns}
    cutoff_col = cols.get("cutoff")
    mae_col = cols.get("mae") or cols.get("mae_test")
    rmse_col = cols.get("rmse") or cols.get("rmse_test")

    if not (cutoff_col and mae_col and rmse_col):
        return model_id

    rep2 = rep[[cutoff_col, mae_col, rmse_col]].copy()
    rep2 = rep2.rename(columns={cutoff_col: "cutoff", mae_col: "mae_test", rmse_col: "rmse_test"})

    for extra in ["n_curves", "n_runs"]:
        if extra in cols:
            rep2[extra] = rep[cols[extra]].values

    rep2.to_parquet(outdir / "metrics_by_cutoff.parquet", index=False)
    return model_id


# ============================================
# 평가 함수
# ============================================

def try_eval_if_truth_exists(df_raw: pd.DataFrame, pred_df: pd.DataFrame, truth_df: pd.DataFrame | None = None) -> None:
    """truth가 있으면 즉석 평가"""
    src = truth_df if truth_df is not None else df_raw

    true_col = None
    for cand in ["true_ct", "TrueCt", "trueCt", "ct", "Ct", "CT", "Cq", "cq", "CQ"]:
        if cand in src.columns:
            true_col = cand
            break

    if true_col is None:
        st.info("업로드 파일에 정답 Ct/Cq 컬럼이 없어서 즉석 평가는 생략했어요.")
        return

    well_key = None
    for w in ["Well", "well", "WELL"]:
        if w in src.columns:
            well_key = w
            break

    eval_df = pred_df.copy()

    if well_key is not None and "Well" in eval_df.columns:
        truth2 = src[[well_key, true_col]].copy()
        truth2.columns = ["Well", "true_ct"]
        truth2["Well"] = truth2["Well"].map(normalize_well)
        eval_df["Well"] = eval_df["Well"].map(normalize_well)
        eval_df = eval_df.merge(truth2, on="Well", how="left")
    else:
        eval_df["true_ct"] = pd.to_numeric(src[true_col], errors="coerce").values[: len(eval_df)]

    eval_df["true_ct"] = pd.to_numeric(eval_df["true_ct"], errors="coerce")
    eval_df = eval_df.dropna(subset=["true_ct", "pred_ct"]).copy()
    
    if len(eval_df) == 0:
        st.warning("정답 Ct 컬럼은 찾았는데, pred와 매칭된 값이 없어요.")
        return

    eval_df["err"] = eval_df["pred_ct"] - eval_df["true_ct"]
    mae = float(np.mean(np.abs(eval_df["err"])))
    rmse = float(np.sqrt(np.mean(eval_df["err"] ** 2)))

    st.markdown("### ? 업로드 데이터 즉석 평가")
    st.write({"MAE": mae, "RMSE": rmse, "n": int(len(eval_df))})
    st.scatter_chart(eval_df[["true_ct", "pred_ct"]], x="true_ct", y="pred_ct", height=320)


# ============================================
# 메인 UI 함수들
# ============================================

def show_train_report() -> None:
    """Performance 탭 - 모델 성능 리포트"""
    st.subheader("?? 모델 성능 리포트")
    report_path = REPORTS_ROOT / "train_report.csv"
    if not report_path.exists():
        st.info("train_report.csv가 아직 없어요.")
        return

    rep = pd.read_csv(report_path)
    cols = {str(c).lower(): c for c in rep.columns}
    cutoff_col = cols.get("cutoff")
    mae_col = cols.get("mae") or cols.get("mae_test")
    rmse_col = cols.get("rmse") or cols.get("rmse_test")

    mid = sync_train_report_to_parquet(rep)

    if cutoff_col and mae_col and rmse_col:
        best_row = rep.loc[rep[mae_col].idxmin()]
        c1, c2, c3 = st.columns(3)
        c1.metric("? 추천 cutoff", int(best_row[cutoff_col]))
        c2.metric("최소 MAE", round(float(best_row[mae_col]), 4))
        c3.metric("해당 RMSE", round(float(best_row[rmse_col]), 4))
        st.divider()

    rep2 = rep.rename(columns={cutoff_col: "cutoff", mae_col: "mae", rmse_col: "rmse"})
    rep2["cutoff"] = pd.to_numeric(rep2["cutoff"], errors="coerce")
    rep2["mae"] = pd.to_numeric(rep2["mae"], errors="coerce")
    rep2["rmse"] = pd.to_numeric(rep2["rmse"], errors="coerce")
    rep2 = rep2.dropna(subset=["cutoff", "mae", "rmse"]).sort_values("cutoff")

    metric_choice = st.radio("보기 선택", ["MAE vs Cutoff", "RMSE vs Cutoff", "둘 다"], horizontal=True)

    if metric_choice == "MAE vs Cutoff":
        chart = alt.Chart(rep2).mark_line(point=True).encode(x="cutoff:Q", y="mae:Q").properties(height=320)
        st.altair_chart(chart, use_container_width=True)
    elif metric_choice == "RMSE vs Cutoff":
        chart = alt.Chart(rep2).mark_line(point=True).encode(x="cutoff:Q", y="rmse:Q").properties(height=320)
        st.altair_chart(chart, use_container_width=True)
    else:
        longm = rep2.melt(id_vars=["cutoff"], value_vars=["mae", "rmse"], var_name="metric", value_name="value")
        chart = alt.Chart(longm).mark_line(point=True).encode(x="cutoff:Q", y="value:Q", strokeDash="metric:N").properties(height=320)
        st.altair_chart(chart, use_container_width=True)


def show_hard_review_with_buckets() -> None:
    """Hard Sample Review - 버킷 분류 포함"""
    st.subheader("?? Hard Sample Review (버킷 분류)")

    model_id = get_active_model_id()
    pred_path = PROJECT_ROOT / "reports" / model_id / "predictions_long.parquet"

    if not pred_path.exists():
        st.info(f"predictions_long.parquet가 없어요: {pred_path}")
        return

    pred = pd.read_parquet(pred_path)
    pred = pred.copy()
    pred["abs_err"] = (pred["pred_ct"] - pred["true_ct"]).abs()

    c_list = sorted(pred["cutoff"].dropna().unique().astype(int).tolist())
    if not c_list:
        st.warning("cutoff 값이 비어있어요.")
        return

    st.markdown("### ?? 설정")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_cutoff = get_best_cutoff_from_report()
        default_idx = c_list.index(best_cutoff) if best_cutoff in c_list else 0
        cutoff = st.selectbox("Cutoff", c_list, index=default_idx, key="bucket_cutoff")
    
    with col2:
        error_threshold = st.slider("Hard 기준 |error|", 0.5, 5.0, 2.0, 0.5, key="bucket_err_thr")
    
    with col3:
        topk = st.slider("최대 표시 개수", 10, 200, 50, 10, key="bucket_topk")

    with st.expander("?? 고급 설정"):
        adv1, adv2 = st.columns(2)
        with adv1:
            late_amp_thr = st.number_input("Late Amp Ct 기준", value=35.0, step=1.0)
            cv_thr = st.number_input("노이즈 CV 기준", value=0.15, step=0.01)
            r2_thr = st.number_input("비시그모이드 R² 기준", value=0.95, step=0.01)
        with adv2:
            fluor_max_thr = st.number_input("과대 Fluor Max", value=50000.0, step=1000.0)
            snr_thr = st.number_input("노이즈 SNR 기준", value=3.0, step=0.5)

    st.divider()

    df = pred[pred["cutoff"] == int(cutoff)].copy()
    df = df.sort_values("abs_err", ascending=False).reset_index(drop=True)
    hard_candidates = df[df["abs_err"] >= error_threshold].head(topk).copy()
    
    if hard_candidates.empty:
        st.success(f"?? |error| >= {error_threshold} 인 샘플이 없어요!")
        return

    st.markdown("### ?? 버킷 분류 중...")
    progress = st.progress(0)
    
    results = []
    for i, (idx, row) in enumerate(hard_candidates.iterrows()):
        try:
            curve_df = load_curve_from_master(str(row["run_id"]), str(row["well_id"]))
            result = classify_hard_sample(
                curve_df=curve_df,
                true_ct=row.get("true_ct"),
                pred_ct=row["pred_ct"],
                abs_error=row["abs_err"],
                cutoff=int(cutoff),
                error_threshold=error_threshold,
                late_amp_threshold=late_amp_thr,
                fluor_max_threshold=fluor_max_thr,
                cv_threshold=cv_thr,
                snr_threshold=snr_thr,
                r2_threshold=r2_thr
            )
            results.append({
                "bucket": result.bucket.value,
                "confidence": result.confidence,
                "details": result.details
            })
        except Exception as e:
            results.append({"bucket": "error", "confidence": 0.0, "details": {"error": str(e)}})
        
        progress.progress((i + 1) / len(hard_candidates))
    
    classified = pd.concat([hard_candidates.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    progress.empty()

    # 버킷 분포
    st.markdown("### ?? 버킷 분포")
    bucket_counts = classified["bucket"].value_counts()
    
    cols = st.columns(min(len(bucket_counts), 6))
    for i, (bucket, count) in enumerate(bucket_counts.items()):
        pct = count / len(classified) * 100
        with cols[i % len(cols)]:
            st.metric(f"{BUCKET_EMOJI.get(bucket, '?')} {bucket}", f"{count}개", f"{pct:.1f}%")

    # Scatter
    st.markdown("### ?? Pred vs True (버킷별)")
    
    scatter = alt.Chart(classified).mark_circle(size=80, opacity=0.8).encode(
        x=alt.X("true_ct:Q", title="True Ct"),
        y=alt.Y("pred_ct:Q", title="Pred Ct"),
        color=alt.Color("bucket:N", scale=alt.Scale(domain=list(BUCKET_COLORS.keys()), range=list(BUCKET_COLORS.values()))),
        tooltip=["run_id", "well_id", "true_ct", "pred_ct", "abs_err", "bucket"]
    )
    
    x_min, x_max = classified["true_ct"].min(), classified["true_ct"].max()
    diag = alt.Chart(pd.DataFrame({"x": [x_min, x_max], "y": [x_min, x_max]})).mark_line(strokeDash=[5, 5], color="gray").encode(x="x:Q", y="y:Q")
    
    st.altair_chart((diag + scatter).properties(height=400).interactive(), use_container_width=True)

    # Box Plot
    st.markdown("### ?? 버킷별 Error 분포")
    box = alt.Chart(classified).mark_boxplot(size=40).encode(
        x=alt.X("bucket:N", sort=list(bucket_counts.index)),
        y=alt.Y("abs_err:Q", title="|Error|"),
        color=alt.Color("bucket:N", legend=None, scale=alt.Scale(domain=list(BUCKET_COLORS.keys()), range=list(BUCKET_COLORS.values())))
    ).properties(height=300)
    st.altair_chart(box, use_container_width=True)

    # 권장사항 탭
    st.markdown("### ?? 버킷별 대응 전략")
    active_buckets = [b for b in bucket_counts.index if b not in ["normal", "error"]]
    
    if active_buckets:
        tabs_bucket = st.tabs([f"{BUCKET_EMOJI.get(b, '?')} {b}" for b in active_buckets])
        
        for tab, bucket in zip(tabs_bucket, active_buckets):
            with tab:
                rec = get_bucket_recommendations(bucket)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**?? 원인**")
                    st.info(rec["원인"])
                    st.markdown("**?? 모델 특징**")
                    st.warning(rec["모델 특징"])
                with c2:
                    st.markdown("**?? 대응 전략**")
                    for i, s in enumerate(rec["대응 전략"], 1):
                        st.markdown(f"{i}. {s}")
                
                st.markdown(f"**?? {bucket} 샘플 목록**")
                st.dataframe(
                    classified[classified["bucket"] == bucket][["run_id", "well_id", "true_ct", "pred_ct", "abs_err", "confidence"]].sort_values("abs_err", ascending=False),
                    use_container_width=True, height=200
                )

    st.divider()

    # 개별 분석
    st.markdown("### ?? 개별 샘플 분석")
    
    def _fmt(i):
        r = classified.iloc[i]
        return f"{BUCKET_EMOJI.get(r['bucket'], '?')} {r['run_id']}:{r['well_id']} | err={r['abs_err']:.2f}"
    
    pick = st.selectbox("샘플 선택", range(len(classified)), format_func=_fmt, key="bucket_pick")
    sel = classified.iloc[pick]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bucket", f"{BUCKET_EMOJI.get(sel['bucket'], '?')} {sel['bucket']}")
    c2.metric("True Ct", f"{sel['true_ct']:.2f}")
    c3.metric("Pred Ct", f"{sel['pred_ct']:.2f}")
    c4.metric("|Error|", f"{sel['abs_err']:.3f}")

    try:
        curve_df = load_curve_from_master(str(sel["run_id"]), str(sel["well_id"]))
        if not curve_df.empty:
            curve_df = curve_df.sort_values("Cycle").reset_index(drop=True)
            cycles = curve_df["Cycle"].values.astype(float)
            fluor = curve_df["Fluor"].values.astype(float)
            r2, fitted, params = fit_sigmoid(cycles, fluor)
            curve_df["Fitted"] = fitted
            
            base = alt.Chart(curve_df).encode(x=alt.X("Cycle:Q"))
            orig = base.mark_line(color="steelblue").encode(y="Fluor:Q")
            fit_line = base.mark_line(color="red", strokeDash=[5,5]).encode(y="Fitted:Q")
            vline = alt.Chart(pd.DataFrame({"x": [int(cutoff)]})).mark_rule(strokeDash=[6,4], color="green").encode(x="x:Q")
            
            st.altair_chart((orig + fit_line + vline).properties(height=350, title=f"R²={r2:.4f}"), use_container_width=True)
    except Exception as e:
        st.error(f"곡선 로딩 실패: {e}")

    # 다운로드
    st.divider()
    st.markdown("### ?? 다운로드")
    c1, c2 = st.columns(2)
    with c1:
        csv = classified.drop(columns=["details"], errors="ignore").to_csv(index=False)
        st.download_button("?? 전체 결과 CSV", csv.encode(), f"hard_buckets_cutoff{cutoff}.csv")
    with c2:
        summary = classified.groupby("bucket").agg(count=("bucket", "size"), mean_err=("abs_err", "mean")).reset_index()
        st.download_button("?? 요약 CSV", summary.to_csv(index=False).encode(), "bucket_summary.csv")


# ============================================
# Streamlit Cloud 초기화
# ============================================

if running_on_streamlit_cloud():
    qc_local_path = QC_DIR / "master_catalog.parquet"
    if not qc_local_path.exists():
        if catalog:
            for item in catalog.get("files", []):
                if item.get("filename") == "master_catalog.parquet":
                    ensure_asset_download(item["url"], qc_local_path)
                    break


# ============================================
# 메인 UI
# ============================================

st.caption("업로드한 qPCR curve 데이터로 Ct를 예측하거나, 서버에 누적된 데이터로 모델을 재학습할 수 있어요.")

# Sidebar
cutoffs = discover_cutoffs(MODELS_DIR)
if not cutoffs:
    st.error(f"모델을 찾지 못했어: {MODELS_DIR}")
    st.stop()

with st.sidebar:
    st.title("CPHOTONICS | Early Ct Predictor")
    st.divider()
    
    best = get_best_cutoff_from_report()
    default_cutoff = best if (best in cutoffs) else (30 if 30 in cutoffs else cutoffs[-1] if cutoffs else 20)
    cutoff = int(st.selectbox(
        "Cutoff(사용 cycle 수)",
        cutoffs,
        index=cutoffs.index(default_cutoff) if default_cutoff in cutoffs else 0,
        key="sidebar_cutoff",
    ))
    
    st.divider()
    st.subheader("재학습 (서버 데이터 기준)")
    min_c = st.number_input("min_cutoff", min_value=1, max_value=200, value=10, step=1, key="sidebar_min_c")
    max_c = st.number_input("max_cutoff", min_value=1, max_value=200, value=40, step=1, key="sidebar_max_c")

cutoff = int(cutoff)
min_c = int(min_c)
max_c = int(max_c)

# Tabs
tabs = st.tabs(["?? Performance", "?? Data Catalog", "?? Predict (Upload)", "?? Hard Review", "?? Retrain(Admin)"])

with tabs[0]:
    show_train_report()

with tabs[1]:
    st.header("?? Data Quality Control & Catalog")
    
    @st.cache_data
    def load_qc_catalog():
        qc_path = QC_DIR / "master_catalog.parquet"
        if qc_path.exists():
            return pd.read_parquet(qc_path)
        return pd.DataFrame()
    
    qc_df = load_qc_catalog()
    
    if qc_df.empty:
        st.warning("?? QC catalog not found")
        st.stop()
    
    total = len(qc_df)
    pass_c = (qc_df['qc_status'] == 'PASS').sum() if 'qc_status' in qc_df.columns else 0
    fail_c = (qc_df['qc_status'] == 'FAIL').sum() if 'qc_status' in qc_df.columns else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Wells", f"{total:,}")
    col2.metric("? PASS", f"{pass_c:,}")
    col3.metric("? FAIL", f"{fail_c:,}")
    
    st.dataframe(qc_df, use_container_width=True, height=500)

with tabs[2]:
    st.subheader("?? Predict (Upload)")
    up = st.file_uploader("qPCR 파일 업로드 (csv/xlsx)", type=["csv", "xlsx", "xls"])
    
    if up is not None:
        run_id = _safe_stem(up.name) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_obj = read_uploaded_table(up)
        df_curve, df_truth, curve_sheet, truth_sheet = split_excel_sheets(raw_obj)
        
        st.dataframe(df_curve.head(30), use_container_width=True)
        
        try:
            df_long = infer_long_df(df_curve, run_id=run_id)
            pred_df = predict_ct(df_long, cutoff=int(cutoff))
            st.success("예측 완료!")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Cutoff", int(cutoff))
            c2.metric("Wells", int(pred_df["Well"].nunique()) if "Well" in pred_df.columns else len(pred_df))
            c3.metric("Pred Ct 범위", f"{pred_df['pred_ct'].min():.2f} ~ {pred_df['pred_ct'].max():.2f}")
            
            plot_uploaded_curve_preview(df_long, cutoff=int(cutoff))
            plot_pred_ct_hist(pred_df)
            
            st.dataframe(pred_df, use_container_width=True)
            
            if df_truth is not None:
                try_eval_if_truth_exists(df_curve, pred_df, truth_df=df_truth)
                
        except Exception as e:
            st.error(f"예측 실패: {e}")

with tabs[3]:
    show_hard_review_with_buckets()

with tabs[4]:
    st.subheader("?? Retrain (Admin)")
    
    if running_on_streamlit_cloud():
        st.warning("Streamlit Cloud에서는 재학습이 비활성화되어 있습니다.")
    else:
        if has_canonical_master_long():
            if st.button("재학습 실행", type="secondary"):
                with st.spinner("재학습 중..."):
                    code, log = run_retrain(int(min_c), int(max_c))
                st.text_area("학습 로그", log, height=380)
                if code == 0:
                    st.success("재학습 완료!")
                else:
                    st.error(f"재학습 실패 (code={code})")
        else:
            st.warning("master_long.parquet가 없어서 재학습을 실행할 수 없습니다.")

try:
    st.caption("VERSION: " + (PROJECT_ROOT / "VERSION.txt").read_text().strip())
except Exception:
    st.caption("VERSION: (missing)")