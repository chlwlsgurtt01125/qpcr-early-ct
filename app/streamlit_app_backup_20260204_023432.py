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
import plotly.express as px
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import pyarrow.dataset as ds
import argparse
from scipy.optimize import curve_fit
from scipy.stats import linregress
from dataclasses import dataclass
from enum import Enum

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
    "late_amp": "#FFD700",      # 노란색
    "high_range": "#FF4444",    # 빨간색
    "noisy": "#FFA500",         # 주황색
    "non_sigmoid": "#9370DB",   # 보라색
    "unknown": "#808080",       # 회색
    "normal": "#00CC66",        # 초록색
    "error": "#000000"          # 검정색
}

BUCKET_EMOJI = {
    "late_amp": "🟡",
    "high_range": "🔴",
    "noisy": "🟠",
    "non_sigmoid": "🟣",
    "unknown": "⚪",
    "normal": "🟢",
    "error": "⚫"
}


# ============================================
# PART 2: Sigmoid Fitting 함수
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

# ✅ set_page_config는 반드시 1번만, 그리고 최상단에서
st.set_page_config(page_title="CPHOTONICS | Early Ct Predictor", layout="wide")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ✅ 경로는 PROJECT_ROOT 기준으로
ASSETS_DIR = PROJECT_ROOT / "assets"
CATALOG_PATH = ASSETS_DIR / "data_catalog.json"
QC_DIR = PROJECT_ROOT / "outputs" / "qc"  # ✅ QC_DIR 정의 추가

OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "qc_performance_analysis"
MODELS_DIR = PROJECT_ROOT / "data" / "models" / "by_cutoff"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

# ========================================
# GitHub Release에서 QC 데이터 자동 다운로드
# ========================================
def load_data_catalog(catalog_path):
    try:
        with open(catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

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
    # GitHub release asset은 302 redirect가 뜰 수 있어서 urlretrieve/urllib가 안정적
    urllib.request.urlretrieve(url, dst_path)

def load_data_catalog_json(catalog_path):
    if not catalog_path.exists():
        return None
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_asset_download(url: str, dst_path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() and dst_path.stat().st_size > 0:
        return False  # already exists

    # GitHub release asset은 redirect가 있을 수 있어 urllib가 안전함
    with urllib.request.urlopen(url) as r, open(dst_path, "wb") as f:
        f.write(r.read())
    return True

def download_qc_data_from_github():
    """
    GitHub Release에서 QC 관련 parquet 파일들을 다운로드
    
    사용법:
    1. GitHub에서 Release 생성
    2. QC parquet 파일들을 Release에 첨부
    3. 이 함수가 자동으로 다운로드
    """
    # ✅ 여기에 실제 GitHub Release URL 입력
    GITHUB_RELEASE_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0.0/"
    
    QC_DIR.mkdir(parents=True, exist_ok=True)
    
    # 다운로드할 파일 목록
    files_to_download = [
        "master_catalog.parquet",
        "excluded_report.parquet",
    ]
    
    for filename in files_to_download:
        local_path = QC_DIR / filename
        
        # 이미 있으면 스킵
        if local_path.exists():
            continue
        
        url = GITHUB_RELEASE_URL + filename
        
        try:
            st.info(f"Downloading {filename} from GitHub Release...")
            urllib.request.urlretrieve(url, local_path)
            st.success(f"✅ Downloaded: {filename}")
        except urllib.error.HTTPError as e:
            st.warning(f"⚠️ Failed to download {filename}: {e}")
        except Exception as e:
            st.warning(f"⚠️ Error downloading {filename}: {e}")


# Streamlit Cloud에서 실행 중이면 자동 다운로드
def running_on_streamlit_cloud() -> bool:
    return str(PROJECT_ROOT).startswith("/mount/src") or os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud"


if running_on_streamlit_cloud():
    # Cloud에서는 QC 데이터를 GitHub Release에서 다운로드
    qc_local_path = QC_DIR / "master_catalog.parquet"

    if not qc_local_path.exists():
        if not catalog:
            st.error("data_catalog.json not loaded (catalog is None/empty).")
        else:
            found = False
            for item in catalog.get("files", []):
                if item.get("filename") == "master_catalog.parquet":
                    ensure_asset_download(item["url"], qc_local_path)
                    found = True
                    break

            if not found:
                st.error("master_catalog.parquet entry not found in assets/data_catalog.json")


# ✅ cutoff 먼저 정의
cutoff = int(st.sidebar.selectbox("Cutoff", [10, 20, 24, 30, 40], index=1))

OPS_DIR = PROJECT_ROOT / "outputs" / "qc_performance_analysis"
OPS_DIR.mkdir(parents=True, exist_ok=True)

ops_filename = f"ops_decisions_cutoff_{cutoff}.parquet"
parquet_path = OPS_DIR / ops_filename
csv_path     = OPS_DIR / f"ops_decisions_cutoff_{cutoff}.csv"

# ✅ Streamlit Cloud에서 ops decisions parquet 자동 다운로드
if running_on_streamlit_cloud():
    if not parquet_path.exists():
        ops_url = find_file_url_in_catalog(catalog, ops_filename)
        if ops_url:
            try:
                download_to_path(ops_url, parquet_path)
            except Exception as e:
                st.warning(f"Failed to download ops decisions ({ops_filename}): {e}")
        else:
            st.warning(
                f"Ops decisions file for cutoff={cutoff} not found in data_catalog.json: {ops_filename}"
            )
if 'show_data_catalog' not in st.session_state:
    st.session_state.show_data_catalog = False


if st.session_state.show_data_catalog:
    st.header("📊 Data Quality Control & Catalog")
    st.markdown("QC 상태(PASS/FAIL/FLAG), Ct bin, excluded 사유를 한 번에 정리/다운로드하는 페이지")

    # 1. master_catalog 로드
    @st.cache_data
    def load_master_catalog():
        path = QC_DIR / "master_catalog.parquet"
        if path.exists():
            return pd.read_parquet(path)
        else:
            st.error("master_catalog.parquet not found. Cloud에서는 GitHub Release에서 자동 다운로드됩니다.")
            return pd.DataFrame()

    df = load_master_catalog()
    # 컬럼 확인 & 안전 처리
    if "exclusion_reason" not in df.columns or df["exclusion_reason"].isna().all():
        df["exclusion_reason"] = "No specific reason"  # N/A 대신 의미있는 값
    
    # qc_status, ct_bin 안전 처리
    df["qc_status"] = df["qc_status"].fillna("UNKNOWN").astype(str)
    df["ct_bin"] = df["ct_bin"].fillna("UNKNOWN").astype(str)
    
    # Exclusion Reasons 차트 수정 (N/A 제외하고 실제 이유만)
    excluded_df = df[(df["qc_status"] != "PASS") & (df["exclusion_reason"] != "N/A") & (df["exclusion_reason"] != "No specific reason")]
    if not excluded_df.empty:
        reasons = excluded_df["exclusion_reason"].value_counts().head(10).reset_index()
        reasons = reasons[reasons["exclusion_reason"] != "N/A"]  # 강제 필터
        fig_ex = px.bar(reasons, x="count", y="exclusion_reason", orientation="h",
                        title="Top 10 Exclusion Reasons")
        fig_ex.update_layout(height=500)
        st.plotly_chart(fig_ex, use_container_width=True)
    else:
        st.info("Excluded 샘플이 없거나 exclusion_reason이 모두 N/A입니다.")
        
    if df.empty:
        st.stop()

    # 필요한 컬럼 확인 (에러 방지)
    required_cols = ["qc_status", "ct_bin"]
    if "exclusion_reason" not in df.columns:
        df["exclusion_reason"] = "N/A"

    # 2. Summary Statistics
    total = len(df)
    pass_c = len(df[df["qc_status"] == "PASS"])
    fail_c = len(df[df["qc_status"] == "FAIL"])
    flag_c = len(df[df["qc_status"] == "FLAG"])
    usable = pass_c
    excluded = total - usable

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Wells", f"{total:,}")
    col2.metric("✅ PASS", f"{pass_c:,}", f"{pass_c/total*100:.1f}%")
    col3.metric("❌ FAIL", f"{fail_c:,}", f"{fail_c/total*100:.1f}%")
    col4.metric("⚠️ FLAG", f"{flag_c:,}", f"{flag_c/total*100:.1f}%")
    col5.metric("🟢 Usable", f"{usable:,}", f"{usable/total*100:.1f}%")
    col6.metric("🔴 Excluded", f"{excluded:,}", f"{excluded/total*100:.1f}%")

    st.divider()

    # 3. Visualizations
        # 3. Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("QC Status Distribution")
        status_counts = df["qc_status"].value_counts()
        if not status_counts.empty:
            fig_pie = px.pie(
                status_counts.reset_index(),
                values="count", names="qc_status",
                color_discrete_map={"PASS": "#00FF00", "FAIL": "#FF0000", "FLAG": "#FFA500", "UNKNOWN": "#808080"}
            )
            fig_pie.update_layout(showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("QC Status 데이터가 없습니다.")

    with col2:
        st.subheader("Ct Bin Distribution")
        ct_order = sorted(df["ct_bin"].dropna().unique())
        if not ct_order:
            st.info("Ct Bin 데이터가 없습니다.")
        else:
            fig_ct = px.bar(
                df["ct_bin"].value_counts().reindex(ct_order).reset_index(),
                x="ct_bin", y="count"
            )
            st.plotly_chart(fig_ct, use_container_width=True)

    st.subheader("QC Status by Ct Bin")
    stacked = df.groupby(["ct_bin", "qc_status"]).size().reset_index(name="count")
    if not stacked.empty:
        stacked = stacked.sort_values("ct_bin")
        fig_stacked = px.bar(
            stacked, x="ct_bin", y="count", color="qc_status",
            color_discrete_map={"PASS": "#00FF00", "FAIL": "#FF0000", "FLAG": "#FFA500"},
            title="QC Status by Ct Bin"
        )
        st.plotly_chart(fig_stacked, use_container_width=True)
    else:
        st.info("QC Status by Ct Bin 데이터가 없습니다.")

    excluded_df = df[df["qc_status"] != "PASS"].copy()
    if not excluded_df.empty:
        st.subheader("🔍 Exclusion Analysis - Top 10 Reasons")
        # 실제 이유만 필터 (N/A 제외)
        reasons = excluded_df[
            (excluded_df["exclusion_reason"] != "N/A") &
            (excluded_df["exclusion_reason"] != "No specific reason") &
            (excluded_df["exclusion_reason"].notna())
        ]["exclusion_reason"].value_counts().head(10).reset_index()
        
        if not reasons.empty and reasons["count"].sum() > 0:
            fig_ex = px.bar(reasons, x="count", y="exclusion_reason", orientation="h",
                            title="Top 10 Exclusion Reasons")
            fig_ex.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_ex, use_container_width=True)
        else:
            st.info("실제 exclusion reason이 없습니다 (대부분 N/A). QC 스크립트에서 이유를 더 자세히 기록하세요.")
    else:
        st.info("Excluded 샘플이 없습니다.")

    # 4. Filterable Table
    st.subheader("📋 Master Catalog (Filterable & Sortable)")
    try:
        from st_aggrid import AgGrid, GridOptionsBuilder
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(groupable=True, sortable=True, filterable=True, editable=False)
        gb.configure_column("qc_status", rowGroup=True)
        gb.configure_column("ct_bin", rowGroup=True)
        grid_options = gb.build()
        AgGrid(df, gridOptions=grid_options, height=600, fit_columns_on_grid_load=True)
    except ImportError:
        st.warning("AgGrid not available. Using basic table.")
        st.dataframe(df, use_container_width=True)

    # 5. Download Buttons
    st.subheader("💾 Download Reports")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Master Catalog (CSV)",
            df.to_csv(index=False).encode('utf-8'),
            "master_catalog_full.csv",
            "text/csv"
        )
    with col2:
        st.download_button(
            "Download Excluded Report (CSV)",
            excluded_df.to_csv(index=False).encode('utf-8'),
            "excluded_report.csv",
            "text/csv"
        )

    # 6. 어두운 테마 (검은색 배경)
    st.markdown("""
    <style>
        .css-1d391kg {background-color: #0e1117;}
        .css-1y0t9cy {color: white;}
        section[data-testid="stSidebar"] {background-color: #262730;}
        .css-1cpxl2t {color: white;}
        h1, h2, h3, h4 {color: white !important;}
    </style>
    """, unsafe_allow_html=True)

    st.stop()

ops = None
try:
    if parquet_path.exists():
        ops = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        ops = pd.read_csv(csv_path, encoding="utf-8")
    
    else:
        st.warning(f"Ops decisions not found: {parquet_path} (or {csv_path})")
except Exception as e:
    st.error(f"Failed to load ops decisions: {e}")
    st.caption(f"Checked: {parquet_path} , {csv_path}")

# ✅ 중복 decision_from_qc 함수 제거 (하나만 유지)
def decision_from_qc(qc_status: str) -> str:
    """QC 상태를 기반으로 운영 결정"""
    qc_status = str(qc_status).upper().strip()
    if qc_status == "PASS":
        return "PREDICT"
    if qc_status == "FLAG":
        return "WARN"
    return "RERUN"
# ============================================
# PART 3: 개별 버킷 판정 함수
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


# ============================================
# PART 4: 통합 분류 함수
# ============================================

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
    
    # Hard 여부 판정
    is_hard = abs_error >= error_threshold
    if not is_hard:
        return BucketResult(HardBucket.NORMAL, 0.0, {"abs_error": abs_error}, False)
    
    # 데이터 추출 및 정렬
    cycles = curve_df["Cycle"].values.astype(float)
    fluor = curve_df["Fluor"].values.astype(float)
    sort_idx = np.argsort(cycles)
    cycles, fluor = cycles[sort_idx], fluor[sort_idx]
    
    # 각 버킷 체크
    checks = []
    
    is_late, conf_late, det_late = check_late_amplification(true_ct, pred_ct, late_amp_threshold)
    checks.append((HardBucket.LATE_AMP, is_late, conf_late, det_late))
    
    is_high, conf_high, det_high = check_high_range(fluor, fluor_max_threshold, fluor_min_threshold)
    checks.append((HardBucket.HIGH_RANGE, is_high, conf_high, det_high))
    
    is_noisy, conf_noisy, det_noisy = check_noisy(cycles, fluor, cutoff, cv_threshold, snr_threshold)
    checks.append((HardBucket.NOISY, is_noisy, conf_noisy, det_noisy))
    
    is_non_sig, conf_non_sig, det_non_sig = check_non_sigmoid(cycles, fluor, r2_threshold)
    checks.append((HardBucket.NON_SIGMOID, is_non_sig, conf_non_sig, det_non_sig))
    
    # 가장 confidence 높은 버킷 선택
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
    
    # UNKNOWN
    return BucketResult(
        HardBucket.UNKNOWN, 0.5,
        {"abs_error": abs_error, "note": "No specific pattern detected"},
        True
    )


# ============================================
# PART 5: 버킷별 권장사항
# ============================================

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
# PART 6: 메인 UI 함수
# ============================================

def show_hard_review_with_buckets() -> None:
    """Hard Sample Review - 버킷 분류 포함 버전"""
    import altair as alt
    
    st.subheader("🧨 Hard Sample Review (버킷 분류)")

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

    # ========== 설정 패널 ==========
    st.markdown("### ⚙️ 설정")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_cutoff = get_best_cutoff_from_report()
        default_idx = c_list.index(best_cutoff) if best_cutoff in c_list else 0
        cutoff = st.selectbox("Cutoff", c_list, index=default_idx, key="bucket_cutoff")
    
    with col2:
        error_threshold = st.slider("Hard 기준 |error|", 0.5, 5.0, 2.0, 0.5, key="bucket_err_thr")
    
    with col3:
        topk = st.slider("최대 표시 개수", 10, 200, 50, 10, key="bucket_topk")

    with st.expander("🔧 고급 설정"):
        adv1, adv2 = st.columns(2)
        with adv1:
            late_amp_thr = st.number_input("Late Amp Ct 기준", value=35.0, step=1.0)
            cv_thr = st.number_input("노이즈 CV 기준", value=0.15, step=0.01)
            r2_thr = st.number_input("비시그모이드 R² 기준", value=0.95, step=0.01)
        with adv2:
            fluor_max_thr = st.number_input("과대 Fluor Max", value=50000.0, step=1000.0)
            snr_thr = st.number_input("노이즈 SNR 기준", value=3.0, step=0.5)

    st.divider()

    # ========== 분류 실행 ==========
    df = pred[pred["cutoff"] == int(cutoff)].copy()
    df = df.sort_values("abs_err", ascending=False).reset_index(drop=True)
    hard_candidates = df[df["abs_err"] >= error_threshold].head(topk).copy()
    
    if hard_candidates.empty:
        st.success(f"🎉 |error| >= {error_threshold} 인 샘플이 없어요!")
        return

    st.markdown("### 🔄 버킷 분류 중...")
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
    
    classified = pd.concat([
        hard_candidates.reset_index(drop=True),
        pd.DataFrame(results)
    ], axis=1)
    progress.empty()

    # ========== 1. 버킷 분포 ==========
    st.markdown("### 📊 버킷 분포")
    bucket_counts = classified["bucket"].value_counts()
    
    cols = st.columns(min(len(bucket_counts), 6))
    for i, (bucket, count) in enumerate(bucket_counts.items()):
        pct = count / len(classified) * 100
        with cols[i % len(cols)]:
            st.metric(f"{BUCKET_EMOJI.get(bucket, '⚪')} {bucket}", f"{count}개", f"{pct:.1f}%")

    # ========== 2. Scatter ==========
    st.markdown("### 🎯 Pred vs True (버킷별)")
    
    scatter = alt.Chart(classified).mark_circle(size=80, opacity=0.8).encode(
        x=alt.X("true_ct:Q", title="True Ct"),
        y=alt.Y("pred_ct:Q", title="Pred Ct"),
        color=alt.Color("bucket:N", scale=alt.Scale(
            domain=list(BUCKET_COLORS.keys()), range=list(BUCKET_COLORS.values())
        )),
        tooltip=["run_id", "well_id", "true_ct", "pred_ct", "abs_err", "bucket"]
    )
    
    x_min, x_max = classified["true_ct"].min(), classified["true_ct"].max()
    diag = alt.Chart(pd.DataFrame({"x": [x_min, x_max], "y": [x_min, x_max]})).mark_line(
        strokeDash=[5, 5], color="gray"
    ).encode(x="x:Q", y="y:Q")
    
    st.altair_chart((diag + scatter).properties(height=400).interactive(), use_container_width=True)

    # ========== 3. Box Plot ==========
    st.markdown("### 📈 버킷별 Error 분포")
    box = alt.Chart(classified).mark_boxplot(size=40).encode(
        x=alt.X("bucket:N", sort=list(bucket_counts.index)),
        y=alt.Y("abs_err:Q", title="|Error|"),
        color=alt.Color("bucket:N", legend=None, scale=alt.Scale(
            domain=list(BUCKET_COLORS.keys()), range=list(BUCKET_COLORS.values())
        ))
    ).properties(height=300)
    st.altair_chart(box, use_container_width=True)

    # ========== 4. 권장사항 탭 ==========
    st.markdown("### 💡 버킷별 대응 전략")
    active_buckets = [b for b in bucket_counts.index if b not in ["normal", "error"]]
    
    if active_buckets:
        tabs = st.tabs([f"{BUCKET_EMOJI.get(b, '⚪')} {b}" for b in active_buckets])
        
        for tab, bucket in zip(tabs, active_buckets):
            with tab:
                rec = get_bucket_recommendations(bucket)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**🔍 원인**")
                    st.info(rec["원인"])
                    st.markdown("**🤖 모델 특징**")
                    st.warning(rec["모델 특징"])
                with c2:
                    st.markdown("**🛠 대응 전략**")
                    for i, s in enumerate(rec["대응 전략"], 1):
                        st.markdown(f"{i}. {s}")
                
                st.markdown(f"**📋 {bucket} 샘플 목록**")
                st.dataframe(
                    classified[classified["bucket"] == bucket][
                        ["run_id", "well_id", "true_ct", "pred_ct", "abs_err", "confidence"]
                    ].sort_values("abs_err", ascending=False),
                    use_container_width=True, height=200
                )

    st.divider()

    # ========== 5. 개별 분석 ==========
    st.markdown("### 🔬 개별 샘플 분석")
    
    def _fmt(i):
        r = classified.iloc[i]
        return f"{BUCKET_EMOJI.get(r['bucket'], '⚪')} {r['run_id']}:{r['well_id']} | err={r['abs_err']:.2f}"
    
    pick = st.selectbox("샘플 선택", range(len(classified)), format_func=_fmt, key="bucket_pick")
    sel = classified.iloc[pick]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bucket", f"{BUCKET_EMOJI.get(sel['bucket'], '⚪')} {sel['bucket']}")
    c2.metric("True Ct", f"{sel['true_ct']:.2f}")
    c3.metric("Pred Ct", f"{sel['pred_ct']:.2f}")
    c4.metric("|Error|", f"{sel['abs_err']:.3f}")

    # 곡선 시각화
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
            vline = alt.Chart(pd.DataFrame({"x": [int(cutoff)]})).mark_rule(
                strokeDash=[6,4], color="green"
            ).encode(x="x:Q")
            
            st.altair_chart(
                (orig + fit_line + vline).properties(height=350, title=f"R²={r2:.4f}"),
                use_container_width=True
            )
    except Exception as e:
        st.error(f"곡선 로딩 실패: {e}")

    # ========== 6. 다운로드 ==========
    st.divider()
    st.markdown("### 💾 다운로드")
    c1, c2 = st.columns(2)
    with c1:
        csv = classified.drop(columns=["details"], errors="ignore").to_csv(index=False)
        st.download_button("📥 전체 결과 CSV", csv.encode(), f"hard_buckets_cutoff{cutoff}.csv")
    with c2:
        summary = classified.groupby("bucket").agg(
            count=("bucket", "size"), mean_err=("abs_err", "mean")
        ).reset_index()
        st.download_button("📥 요약 CSV", summary.to_csv(index=False).encode(), "bucket_summary.csv")

# -------------------------
# Utilities
# -------------------------
def get_reports_root() -> Path:
    # 1) 가장 우선: 레포 루트의 reports/ (Streamlit Cloud 배포용)
    p = Path("reports")
    if p.exists():
        return p

    # 2) (레거시/로컬) app/data/reports 같은 위치를 쓰던 경우 대비
    p2 = Path(__file__).resolve().parent / "data" / "reports"
    if p2.exists():
        return p2

    # 3) 마지막 fallback
    return Path("reports")


REPORTS_ROOT = get_reports_root()


def has_canonical_master_long() -> bool:
    return (PROJECT_ROOT / "data" / "canonical" / "master_long.parquet").exists()


def running_on_streamlit_cloud() -> bool:
    # streamlit cloud는 보통 /mount/src 아래에서 실행됨
    return str(PROJECT_ROOT).startswith("/mount/src")

can_retrain = has_canonical_master_long() and (not running_on_streamlit_cloud())
if running_on_streamlit_cloud():
    pass
elif not has_canonical_master_long():
    pass

def get_active_model_id() -> str:
    p = REPORTS_ROOT / "active_model.txt"
    mid = p.read_text().strip() if p.exists() else "model_server_latest_xgb"
    mid = Path(mid).name
    return mid


def _line_y_eq_x(df: pd.DataFrame):
    # y=x 라인 그리기 위한 DataFrame
    if df.empty:
        return pd.DataFrame({"x":[0,1], "y":[0,1]})
    lo = float(min(df["true_ct"].min(), df["pred_ct"].min()))
    hi = float(max(df["true_ct"].max(), df["pred_ct"].max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    lo -= pad; hi += pad
    return pd.DataFrame({"x":[lo, hi], "y":[lo, hi]})

def plot_pred_vs_true_facets(pred_long: pd.DataFrame, cutoffs: list[int], ncol: int = 4) -> None:
    import altair as alt

    df = pred_long.dropna(subset=["cutoff", "true_ct", "pred_ct"]).copy()
    df["cutoff"] = df["cutoff"].astype(int)
    df = df[df["cutoff"].isin([int(c) for c in cutoffs])].copy()
    if df.empty:
        st.info("선택한 cutoff에 표시할 데이터가 없어요.")
        return

    # (1) 산점도
    base = alt.Chart(df).mark_circle(size=60, opacity=0.75).encode(
        x=alt.X("true_ct:Q", title="True Ct/Cq"),
        y=alt.Y("pred_ct:Q", title="Pred Ct/Cq"),
        tooltip=["run_id", "well_id", "cutoff", "true_ct", "pred_ct"],
    )

    # (2) y=x 대각선: ★ 같은 df를 쓰되 transform으로 2점짜리 라인 생성
    #     (cutoff facet마다 lo/hi 계산되도록 aggregate -> calculate -> fold)
    diag = (
        alt.Chart(df)
        .transform_aggregate(
            min_true="min(true_ct)",
            min_pred="min(pred_ct)",
            max_true="max(true_ct)",
            max_pred="max(pred_ct)",
            groupby=["cutoff"],  # facet 단위로 각각 lo/hi 만들기
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
    ).resolve_scale(
        x="independent",
        y="independent",
    ).properties(
        title="Pred vs True across Cycle Cutoffs"
    )

    st.altair_chart(chart, use_container_width=True)

import re

def normalize_well(x: object) -> str:
    """
    B2, b2, ' B2 '  -> 'B02'
    D07 -> 'D07'
    """
    s = str(x).strip().upper()
    m = re.fullmatch(r"([A-H])\s*0*([0-9]{1,2})", s)
    if not m:
        return s
    row = m.group(1)
    col = int(m.group(2))
    return f"{row}{col:02d}"


import altair as alt

def perf_accuracy_fraction_vs_cutoff(pred: pd.DataFrame, tol: float = 2.0) -> pd.DataFrame:
    """
    |pred-true| <= tol 비율을 cutoff별로 계산
    """
    df = pred.dropna(subset=["cutoff", "true_ct", "pred_ct"]).copy()
    df["abs_err"] = (df["pred_ct"] - df["true_ct"]).abs()
    out = df.groupby("cutoff").apply(lambda g: (g["abs_err"] <= tol).mean()).reset_index(name="acc_frac")
    out["cutoff"] = out["cutoff"].astype(int)
    return out.sort_values("cutoff")

def plot_error_by_true_ct_scatter(
    pred: pd.DataFrame,
    cutoff: int,
    tol: float = 2.0,
    bin_width: float = 2.0,
) -> None:
    """
    친절 버전 Bias Plot:
    - y=0 기준선 (과대/과소예측 바로 해석)
    - ±tol band (실무 허용 오차 대역)
    - True Ct 구간(bin)별 평균 error 라인 (bias가 어디서 생기는지 직관적으로)
    """
    import altair as alt

    df = pred[pred["cutoff"] == int(cutoff)].dropna(subset=["true_ct", "pred_ct"]).copy()
    if df.empty:
        st.info("해당 cutoff에 scatter를 그릴 데이터가 없어요.")
        return

    df["err"] = df["pred_ct"] - df["true_ct"]

    # x-range 잡기
    x_min = float(df["true_ct"].min())
    x_max = float(df["true_ct"].max())
    pad = (x_max - x_min) * 0.03 if x_max > x_min else 1.0
    x_min -= pad
    x_max += pad

    # (A) ±tol band 데이터
    band_df = pd.DataFrame(
        {"x": [x_min, x_max], "y1": [-float(tol), -float(tol)], "y2": [float(tol), float(tol)]}
    )

    # (B) bin별 평균 error (bias 라인)
    bw = float(bin_width)
    if bw <= 0:
        bw = 2.0

    tmp = df[["true_ct", "err"]].copy()
    tmp["bin"] = np.floor(tmp["true_ct"] / bw) * bw
    grp = (
        tmp.groupby("bin")
        .agg(mean_err=("err", "mean"), n=("err", "size"))
        .reset_index()
        .sort_values("bin")
    )
    grp["bin_center"] = grp["bin"] + bw / 2.0

    # -----------------
    # 차트 레이어 구성
    # -----------------

    # 1) tol band (연한 영역)
    band = (
        alt.Chart(band_df)
        .mark_area(opacity=0.12)
        .encode(
            x=alt.X("x:Q", title="True Ct/Cq"),
            y=alt.Y("y1:Q", title="Error (pred - true)"),
            y2="y2:Q",
            tooltip=[
                alt.Tooltip("y1:Q", title="-tol"),
                alt.Tooltip("y2:Q", title="+tol"),
            ],
        )
    )

    # 2) y=0 기준선
    zero_line = (
        alt.Chart(pd.DataFrame({"y": [0.0]}))
        .mark_rule(strokeDash=[6, 4], opacity=0.6)
        .encode(y="y:Q")
    )

    # 3) 점(샘플별 error)
    points = (
        alt.Chart(df)
        .mark_circle(size=55, opacity=0.65)
        .encode(
            x=alt.X("true_ct:Q", title="True Ct/Cq"),
            y=alt.Y("err:Q", title="Error (pred - true)"),
            tooltip=[
                alt.Tooltip("run_id:N", title="run_id"),
                alt.Tooltip("well_id:N", title="well_id"),
                alt.Tooltip("true_ct:Q", title="true"),
                alt.Tooltip("pred_ct:Q", title="pred"),
                alt.Tooltip("err:Q", title="err (pred-true)"),
            ],
        )
    )

    # 4) bin 평균 bias 라인 + 포인트
    bias_line = (
        alt.Chart(grp)
        .mark_line(point=True, opacity=0.9)
        .encode(
            x=alt.X("bin_center:Q"),
            y=alt.Y("mean_err:Q"),
            tooltip=[
                alt.Tooltip("bin_center:Q", title="Ct bin center"),
                alt.Tooltip("mean_err:Q", title="mean err"),
                alt.Tooltip("n:Q", title="n"),
            ],
        )
    )

    chart = (
        alt.layer(band, zero_line, points, bias_line)
        .properties(height=340)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"해석 팁: y=0 위(+)=과대예측(늦게 나온다고 판단), 아래(-)=과소예측(빨리 나온다고 판단). "
        f"연한 영역은 ±{float(tol):.1f} 오차 대역, 굵은 선은 Ct 구간(bin={float(bin_width):.1f})별 평균 오차(=bias)입니다."
    )


def plot_pred_vs_true_hard_colored(df_cut: pd.DataFrame, hard_ids: set[tuple[str, str]] | None = None,
                                  highlight: tuple[str, str] | None = None) -> None:
    import altair as alt

    df = df_cut.dropna(subset=["true_ct", "pred_ct"]).copy()
    if df.empty:
        st.info("scatter를 그릴 데이터가 없어요.")
        return

    # hard 여부
    if hard_ids is None:
        df["group"] = "Inlier"
    else:
        df["group"] = df.apply(lambda r: "Hard" if (str(r["run_id"]), str(r["well_id"])) in hard_ids else "Inlier", axis=1)

    # 선택 샘플 강조
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

    # y=x 선: 같은 df를 쓰는 transform 방식(Altair 안전)
    diag = (
        alt.Chart(df)
        .transform_aggregate(
            min_true="min(true_ct)",
            min_pred="min(pred_ct)",
            max_true="max(true_ct)",
            max_pred="max(pred_ct)",
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

    st.altair_chart(alt.layer(diag, base).properties(height=380, title="Hard Samples highlighted on Pred vs True"), use_container_width=True)

def plot_uploaded_curve_preview(df_long: pd.DataFrame, cutoff: int, max_wells: int = 6) -> None:
    """업로드한 df_long에서 몇 개 well만 뽑아 곡선 preview (동적)"""
    if df_long.empty:
        st.info("df_long이 비어있어요.")
        return

    wells = sorted(df_long["Well"].dropna().unique().tolist())[:max_wells]
    sub = df_long[df_long["Well"].isin(wells)].copy()
    sub["segment"] = np.where(sub["Cycle"] <= int(cutoff), "early(<=cutoff)", "late")

    chart = (
        alt.Chart(sub)
        .mark_line()
        .encode(
            x=alt.X("Cycle:Q", title="Cycle"),
            y=alt.Y("Fluor:Q", title="Fluor"),
            color=alt.Color("Well:N", legend=alt.Legend(title="Well")),
            tooltip=["Well", "Cycle", "Fluor", "segment"],
        )
        .properties(height=320)
        .interactive()
    )

    vline = (
        alt.Chart(pd.DataFrame({"Cycle": [int(cutoff)]}))
        .mark_rule(strokeDash=[6, 4])
        .encode(x="Cycle:Q")
    )

    st.altair_chart(chart + vline, use_container_width=True)
    st.caption(f"미리보기: {len(wells)}개 well만 표시 (전체 {df_long['Well'].nunique()} wells 중)")

def plot_pred_ct_hist(pred_df: pd.DataFrame) -> None:
    """예측 Ct 분포 히스토그램(동적)"""
    if pred_df.empty or "pred_ct" not in pred_df.columns:
        return

    hist = (
        alt.Chart(pred_df)
        .mark_bar()
        .encode(
            x=alt.X("pred_ct:Q", bin=alt.Bin(maxbins=25), title="Predicted Ct"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip("count():Q", title="count")],
        )
        .properties(height=280)
    )
    st.altair_chart(hist, use_container_width=True)

def plot_cv_vs_ct(df_long: pd.DataFrame, pred_df: pd.DataFrame, cutoff: int) -> None:
    """
    간단한 품질지표(CV) vs Ct (동적)
    - early 구간(<=cutoff)에서 Fluor의 CV(std/mean)를 계산해서 pred_ct와 연결
    """
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

    scat = (
        alt.Chart(m)
        .mark_circle(size=60)
        .encode(
            x=alt.X("pred_ct:Q", title="Predicted Ct"),
            y=alt.Y("cv_early:Q", title="CV (early <= cutoff)"),
            tooltip=["Well", "pred_ct", "cv_early"],
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(scat, use_container_width=True)
    st.caption("CV는 early 구간 Fluor의 std/mean 기반(간단 버전).")


def get_best_cutoff_from_report() -> int | None:
    """train_report.csv에서 mae_test(또는 mae) 최소 cutoff를 반환"""
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


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if str(c).strip().lower().startswith("unnamed")]
    return df.drop(columns=cols) if cols else df


def infer_long_df(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """
    업로드 테이블을 최대한 관대하게 long 형태로 변환.
    최종 반환 컬럼: Cycle, Fluor, Well, run_id, well_uid

    지원 포맷:
      A) long: (Well, Cycle, Fluor/RFU/Signal)
      B) wide-1: (Well + cycle columns "1","2",... 또는 "Cycle 1"...)
      C) wide-2: (Cycle + well columns "C3","C5","A01"... )  <-- 너가 말한 엑셀 형태
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = _drop_unnamed(df)

    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    has_cycle = "cycle" in cols_lower

    # ---- A) long 형태: Cycle + (Fluor/RFU/Signal)
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
            # Well이 없으면 행 번호로 임시 well 부여
            df["Well"] = [f"R{i:03d}" for i in range(1, len(df) + 1)]
            well_col = "Well"

        out = df[[well_col, cycle_col, fluor_col]].copy()
        out.columns = ["Well", "Cycle", "Fluor"]

    # ---- C) Cycle + 여러 well 컬럼 (Cycle이 행)
    elif has_cycle:
        cycle_col = cols_lower["cycle"]
        well_cols = [c for c in df.columns if c != cycle_col]
        if not well_cols:
            raise ValueError("Cycle 컬럼은 있는데 well 컬럼(C3, C5, A01 등)이 없어.")

        long = df.melt(
            id_vars=[cycle_col],
            value_vars=well_cols,
            var_name="Well",
            value_name="Fluor",
        )
        long.rename(columns={cycle_col: "Cycle"}, inplace=True)
        out = long[["Well", "Cycle", "Fluor"]].copy()

    # ---- B) Well + cycle 컬럼들 (wide)
    else:
        well_col = None
        for cand in ["Well", "well", "WELL"]:
            if cand in df.columns:
                well_col = cand
                break
        if well_col is None:
            raise ValueError("Well 컬럼을 찾지 못했어. (예: Well, well)")

        cycle_cols: list[str] = []
        for c in df.columns:
            if c == well_col:
                continue
            if re.fullmatch(r"\d+", str(c).strip()):
                cycle_cols.append(c)
            elif re.search(r"cycle\s*\d+", str(c).strip(), flags=re.IGNORECASE):
                cycle_cols.append(c)

        if not cycle_cols:
            raise ValueError("long도 아니고 wide(Well+cycle cols)도 아닌 것 같아. (Cycle+well cols도 아님)")

        tmp = df[[well_col] + cycle_cols].copy()
        long = tmp.melt(id_vars=[well_col], var_name="Cycle", value_name="Fluor")
        long["Cycle"] = long["Cycle"].astype(str).str.extract(r"(\d+)").astype(int)
        long.rename(columns={well_col: "Well"}, inplace=True)
        out = long[["Well", "Cycle", "Fluor"]].copy()

    # ---- 정리
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

    # feature_names mismatch 방지 (meta에 있으면 사용)
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


def run_retrain(min_cutoff: int, max_cutoff: int) -> tuple[int, str]:
    if running_on_streamlit_cloud():
        return 2, "Streamlit Cloud에서는 canonical 데이터가 없어서 재학습이 비활성화되어 있습니다. 서버/로컬에서 학습 후 reports/만 배포하세요."
    """
    현재 서버에 있는 canonical/master_long.parquet 기준으로 모델 전체 재학습.
    (Streamlit 버튼에서도 GPU 사용 가능하도록 env 전달)
    """
    cmd = [
        sys.executable,
        "-m",
        "core.step3_train_and_save_models",
        "--min_cutoff",
        str(min_cutoff),
        "--max_cutoff",
        str(max_cutoff),
    ]

    env = dict(os.environ)

    # ✅ GPU를 쓰고 싶으면 여기만 바꾸면 됨 (예: 1번 GPU 고정)
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")

    p = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    log = (p.stdout or "") + "\n" + (p.stderr or "")
    return p.returncode, log

def split_excel_sheets(obj):
    """
    obj가 dict(sheet_name -> df)일 때
    - curve_df: Cycle 컬럼 있는 시트(우선 SYBR)
    - truth_df: Well + (Cq/Ct/true_ct) 있는 시트(우선 Sheet1)
    """
    if not isinstance(obj, dict):
        return obj, None, None, None  # (curve_df, truth_df, curve_sheet, truth_sheet)

    # 후보 우선순위
    curve_priority = ["SYBR", "Amplification", "Data", "Raw"]
    truth_priority = ["Sheet1", "Ct", "Cq", "Truth", "Result"]

    def norm_cols(df):
        return [str(c).strip().lower() for c in df.columns]

    # 1) curve sheet 찾기
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

    # fallback: 첫 시트
    if curve_df is None:
        curve_sheet = next(iter(obj.keys()))
        curve_df = obj[curve_sheet]

    # 2) truth sheet 찾기
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
        # ✅ 모든 시트 읽기 (dict[str, DataFrame])
        return pd.read_excel(buf, sheet_name=None)
    return pd.read_csv(buf)


def sync_train_report_to_parquet(rep: pd.DataFrame) -> str:
    """
    train_report.csv(rep)를 Performance 페이지가 읽는 parquet로 저장한다.

    저장 위치:
      <repo>/reports/<model_id>/metrics_by_cutoff.parquet
    그리고:
      <repo>/reports/active_model.txt 를 업데이트한다.
    """
    model_id = "model_server_latest_xgb"

    outdir = REPORTS_ROOT / model_id
    outdir.mkdir(parents=True, exist_ok=True)
    (REPORTS_ROOT / "active_model.txt").write_text(model_id, encoding="utf-8")
    cols = {str(c).lower(): c for c in rep.columns}
    cutoff_col = cols.get("cutoff")
    mae_col = cols.get("mae") or cols.get("mae_test")
    rmse_col = cols.get("rmse") or cols.get("rmse_test")

    if not (cutoff_col and mae_col and rmse_col):
        print("Missing cols:", {"cutoff": cutoff_col, "mae": mae_col, "rmse": rmse_col})
        print("Available:", list(rep.columns))
        return model_id

    rep2 = rep[[cutoff_col, mae_col, rmse_col]].copy()
    rep2 = rep2.rename(columns={cutoff_col: "cutoff", mae_col: "mae_test", rmse_col: "rmse_test"})

    # optional extras
    for extra in ["n_curves", "n_runs"]:
        if extra in cols:
            rep2[extra] = rep[cols[extra]].values

    rep2.to_parquet(outdir / "metrics_by_cutoff.parquet", index=False)

    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    (PROJECT_ROOT / "reports" / "active_model.txt").write_text(model_id, encoding="utf-8")

    return model_id

def show_train_report() -> None:
    st.subheader("📊 모델 성능 리포트 (서버 학습 기준)")
    report_path = REPORTS_ROOT / "train_report.csv"
    if not report_path.exists():
        st.info("train_report.csv가 아직 없어요. 재학습 실행 후 생성됩니다.")
        return

    rep = pd.read_csv(report_path)

    # ✅ cols를 먼저 만든다 (여기가 핵심)
    cols = {str(c).lower(): c for c in rep.columns}
    cutoff_col = cols.get("cutoff")
    mae_col = cols.get("mae") or cols.get("mae_test")
    rmse_col = cols.get("rmse") or cols.get("rmse_test")
    ncurves_col = cols.get("n_curves")

    # Performance 페이지용 parquet 저장
    mid = sync_train_report_to_parquet(rep)
    st.caption(f"✅ Performance용 리포트 저장: reports/{mid}/metrics_by_cutoff.parquet")

    # ✅ 추천 cutoff 카드 (cols 만든 뒤에!)
    if cutoff_col and mae_col and rmse_col:
        best_row = rep.loc[rep[mae_col].idxmin()]
        c1, c2, c3 = st.columns(3)
        c1.metric("✅ 추천 cutoff (MAE 최소)", int(best_row[cutoff_col]))
        c2.metric("최소 MAE", round(float(best_row[mae_col]), 4))
        c3.metric("해당 RMSE", round(float(best_row[rmse_col]), 4))
        st.divider()

    # =========================
    # Figure 중심 Performance
    # =========================
    import altair as alt

    # 보기 옵션
    show_table = st.toggle("표(원본 rep)도 같이 보기", value=False)

    # 컬럼 정규화해서 쓰기 편하게
    rep2 = rep.copy()
    rep2 = rep2.rename(columns={
        cutoff_col: "cutoff",
        mae_col: "mae",
        rmse_col: "rmse",
    })
    if ncurves_col:
        rep2 = rep2.rename(columns={ncurves_col: "n_curves"})
    if "n_runs" in cols:
        rep2 = rep2.rename(columns={cols["n_runs"]: "n_runs"})

    rep2["cutoff"] = pd.to_numeric(rep2["cutoff"], errors="coerce")
    rep2["mae"] = pd.to_numeric(rep2["mae"], errors="coerce")
    rep2["rmse"] = pd.to_numeric(rep2["rmse"], errors="coerce")
    if "n_curves" in rep2.columns:
        rep2["n_curves"] = pd.to_numeric(rep2["n_curves"], errors="coerce")
    if "n_runs" in rep2.columns:
        rep2["n_runs"] = pd.to_numeric(rep2["n_runs"], errors="coerce")

    rep2 = rep2.dropna(subset=["cutoff", "mae", "rmse"]).sort_values("cutoff").reset_index(drop=True)

    # ---- 요약 카드(이미 위에서 metric 찍고 있어도, 여기서 더 직관적으로 보강 가능) ----
    best_i = int(rep2["mae"].idxmin())
    best_row2 = rep2.loc[best_i]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ Best cutoff (MAE 최소)", int(best_row2["cutoff"]))
    c2.metric("최소 MAE", round(float(best_row2["mae"]), 4))
    c3.metric("해당 RMSE", round(float(best_row2["rmse"]), 4))
    if "n_curves" in rep2.columns and pd.notna(best_row2.get("n_curves", np.nan)):
        c4.metric("n_curves", int(best_row2["n_curves"]))
    elif "n_runs" in rep2.columns and pd.notna(best_row2.get("n_runs", np.nan)):
        c4.metric("n_runs", int(best_row2["n_runs"]))
    else:
        c4.metric("rows", int(len(rep2)))

    st.divider()

    # ---- (1) MAE / RMSE vs Cutoff: 인터랙티브 라인 ----
    metric_choice = st.radio(
        "보기 선택",
        ["MAE vs Cutoff", "RMSE vs Cutoff", "MAE+RMSE 둘 다(겹쳐보기)"],
        horizontal=True,
    )

    base = alt.Chart(rep2).encode(
        x=alt.X("cutoff:Q", title="Cutoff"),
    )

    hover = alt.selection_point(fields=["cutoff"], on="mouseover", nearest=True, empty=False)

    def line_with_points(ycol: str, title: str):
        line = base.mark_line().encode(
            y=alt.Y(f"{ycol}:Q", title=title),
            tooltip=[alt.Tooltip("cutoff:Q", title="cutoff"), alt.Tooltip(f"{ycol}:Q", title=title)],
        )
        pts = base.mark_circle(size=70).encode(
            y=alt.Y(f"{ycol}:Q"),
            opacity=alt.condition(hover, alt.value(1.0), alt.value(0.15)),
            tooltip=[alt.Tooltip("cutoff:Q", title="cutoff"), alt.Tooltip(f"{ycol}:Q", title=title)],
        ).add_params(hover)

        vline = alt.Chart(rep2).mark_rule(strokeDash=[6, 4]).encode(
            x="cutoff:Q",
            opacity=alt.condition(hover, alt.value(0.6), alt.value(0.0)),
        ).transform_filter(hover)

        return (line + pts + vline).properties(height=320)

    if metric_choice == "MAE vs Cutoff":
        st.altair_chart(line_with_points("mae", "MAE"), use_container_width=True)

    elif metric_choice == "RMSE vs Cutoff":
        st.altair_chart(line_with_points("rmse", "RMSE"), use_container_width=True)

    else:
        # 겹쳐보기(롱 포맷으로 변환)
        longm = rep2.melt(id_vars=["cutoff"], value_vars=["mae", "rmse"], var_name="metric", value_name="value")
        longm["metric"] = longm["metric"].map({"mae": "MAE", "rmse": "RMSE"})

        hover2 = alt.selection_point(fields=["cutoff"], on="mouseover", nearest=True, empty=False)

        chart = alt.Chart(longm).encode(
            x=alt.X("cutoff:Q", title="Cutoff"),
            y=alt.Y("value:Q", title="Error"),
            tooltip=[
                alt.Tooltip("cutoff:Q", title="cutoff"),
                alt.Tooltip("metric:N", title="metric"),
                alt.Tooltip("value:Q", title="value"),
            ],
            strokeDash="metric:N",
        )

        line = chart.mark_line()
        pts = chart.mark_circle(size=70).encode(
            opacity=alt.condition(hover2, alt.value(1.0), alt.value(0.15)),
        ).add_params(hover2)

        vline = alt.Chart(rep2).mark_rule(strokeDash=[6, 4]).encode(
            x="cutoff:Q",
            opacity=alt.condition(hover2, alt.value(0.6), alt.value(0.0)),
        ).transform_filter(hover2)

        st.altair_chart((line + pts + vline).properties(height=320), use_container_width=True)

    st.divider()

    # ---- (2) n_curves vs Cutoff (있을 때만) ----
    if "n_curves" in rep2.columns and rep2["n_curves"].notna().any():
        st.markdown("#### 📦 데이터 커버리지 (#Curves vs Cutoff)")
        cov = alt.Chart(rep2).mark_line().encode(
            x=alt.X("cutoff:Q", title="Cutoff"),
            y=alt.Y("n_curves:Q", title="#Curves"),
            tooltip=[alt.Tooltip("cutoff:Q"), alt.Tooltip("n_curves:Q")],
        ).properties(height=220)
        st.altair_chart(cov, use_container_width=True)

    # =========================
    # (추가) predictions_long 기반 동적 성능 figure
    # =========================
    model_id = get_active_model_id()
    
    pred_path = PROJECT_ROOT / "reports" / model_id / "predictions_long.parquet"


    if pred_path.exists():
        pred_long = pd.read_parquet(pred_path)

        st.markdown("### 📌 추가 성능 Figure (서버 평가 로그 기반)")

        tol = st.slider("정확도 기준 |error| <= ?", 0.5, 5.0, 2.0, 0.5, key="perf_tol")
        acc_df = perf_accuracy_fraction_vs_cutoff(pred_long, tol=float(tol))

        acc_chart = (
            alt.Chart(acc_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("cutoff:Q", title="Cutoff"),
                y=alt.Y("acc_frac:Q", title=f"Accuracy Fraction (|err|<= {tol})"),
                tooltip=["cutoff", "acc_frac"],
            )
            .properties(height=260)
        )
        st.altair_chart(acc_chart, use_container_width=True)
        
        # =========================
        # (NEW) Pred vs True (Cutoff step별 Small Multiples)
        # =========================
        st.markdown("### 📌 Pred vs True (Cutoff step별 Small Multiples)")

        step = st.radio("cutoff step", [3, 5], horizontal=True, key="pvst_step")
        cmin = int(pred_long["cutoff"].min())
        cmax = int(pred_long["cutoff"].max())
        rng = st.slider("cutoff 범위", min_value=cmin, max_value=cmax, value=(cmin, cmax), step=1, key="pvst_rng")
        cols_per_row = st.slider("한 줄에 몇 개?", 2, 6, 4, 1, key="pvst_cols")

        cut_list = [c for c in range(rng[0], rng[1] + 1) if (c - rng[0]) % int(step) == 0]
        plot_pred_vs_true_facets(pred_long, cut_list, ncol=int(cols_per_row))

        st.divider()

        cutoff_sel = st.selectbox(
            "Scatter 볼 cutoff 선택",
            sorted(pred_long["cutoff"].dropna().unique().astype(int).tolist()),
            key="perf_cutoff_sel",
        )
        st.markdown("#### Error vs True Ct (Bias 확인)")
        # ✅ 친절 버전 옵션 (키 중복 방지용으로 perf_ prefix)
        bias_tol = st.slider("Bias 허용 대역(±tol)", 0.5, 5.0, 2.0, 0.5, key="perf_bias_tol")
        bias_binw = st.slider("Ct bin 폭(평균 bias 계산)", 1.0, 6.0, 2.0, 0.5, key="perf_bias_binw")
        plot_error_by_true_ct_scatter(
            pred_long,
            cutoff=int(cutoff_sel),
            tol=float(bias_tol),
            bin_width=float(bias_binw),
        )

        dfc = pred_long[pred_long["cutoff"] == int(cutoff_sel)].dropna(subset=["true_ct", "pred_ct"]).copy()
        dfc["abs_err"] = (dfc["pred_ct"] - dfc["true_ct"]).abs()
        hist = (
            alt.Chart(dfc)
            .mark_bar()
            .encode(
                x=alt.X("abs_err:Q", bin=alt.Bin(maxbins=30), title="|Error|"),
                y=alt.Y("count():Q", title="Count"),
                tooltip=[alt.Tooltip("count():Q", title="count")],
            )
            .properties(height=240)
        )
        st.altair_chart(hist, use_container_width=True)

    else:
        st.info("추가 figure를 그리려면 predictions_long.parquet가 필요해요. (Retrain 후 생성되는 파일)")

    # ---- (옵션) 표 ----
    if show_table:
        st.markdown("#### 원본 테이블")
        st.dataframe(rep, use_container_width=True)
        
        st.markdown("### 📊 Fold-change (증폭 배수) 분석")
    
        st.info("""
        **Fold-change 해석:**
        - Ct 차이 1 = 2배 증폭 차이
        - 임상적 허용 범위: 1.5배 이내 (ΔCt ≈ 0.58)
        - 2배 이상 차이: 생물학적으로 유의미한 차이
        """)
        
        # Fold-change 계산
        pred_long["fold_change"] = 2 ** abs(pred_long["pred_ct"] - pred_long["true_ct"])
        
        # 카테고리 분류
        fc_bins = [1.0, 1.2, 1.5, 2.0, 3.0, float('inf')]
        fc_labels = ["<1.2x (우수)", "1.2-1.5x (양호)", "1.5-2x (주의)", "2-3x (불량)", ">3x (심각)"]
        
        pred_long["fc_category"] = pd.cut(
            pred_long["fold_change"], 
            bins=fc_bins, 
            labels=fc_labels,
            include_lowest=True
        )
        
        # Cutoff별 Fold-change 분포
        fc_by_cutoff = pred_long.groupby(["cutoff", "fc_category"]).size().reset_index(name="count")
        fc_by_cutoff = fc_by_cutoff.sort_values("cutoff")
        
        import altair as alt
        
        fc_chart = alt.Chart(fc_by_cutoff).mark_bar().encode(
            x=alt.X("cutoff:Q", title="Cutoff"),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color("fc_category:N", 
                           title="Fold-change Category",
                           scale=alt.Scale(scheme="redyellowgreen", reverse=True)),
            tooltip=["cutoff", "fc_category", "count"]
        ).properties(height=300)
        
        st.altair_chart(fc_chart, use_container_width=True)
        
        # 전체 요약
        st.markdown("#### 📈 전체 Fold-change 요약")
        
        fc_summary = pred_long["fc_category"].value_counts().sort_index()
        total = len(pred_long)
        
        summary_df = pd.DataFrame({
            "Category": fc_summary.index,
            "Count": fc_summary.values,
            "Percentage": (fc_summary.values / total * 100).round(1)
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(summary_df, use_container_width=True)
        
        with col2:
            # 허용 가능(1.5x 이내) 비율
            acceptable = pred_long["fold_change"] <= 1.5
            acceptable_rate = acceptable.sum() / total
            
            st.metric(
                "임상 허용 범위 (≤1.5x)",
                f"{acceptable.sum():,} / {total:,}",
                f"{acceptable_rate*100:.1f}%"
            )
            
            # 심각한 오차 (>3x)
            severe = pred_long["fold_change"] > 3.0
            st.metric(
                "심각한 오차 (>3x)",
                f"{severe.sum():,}",
                f"{severe.sum()/total*100:.1f}%",
                delta_color="inverse"
            )
        
        st.divider()
    
    # =========================
    # (NEW) Ct 오차의 생물학적 의미
    # =========================
    st.markdown("### 🔢 Ct 오차의 생물학적 의미")
    
    with st.expander("💡 Ct 오차 해석 가이드", expanded=False):
        meaning_df = pd.DataFrame({
            "Ct Error (ΔCt)": [0.3, 0.5, 1.0, 1.5, 2.0, 3.0],
            "Fold Change": [1.23, 1.41, 2.0, 2.83, 4.0, 8.0],
            "생물학적 의미": [
                "무시 가능 (기술적 변동 범위)",
                "허용 범위 (Technical replicate SD)",
                "2배 차이 (주의 필요)",
                "임상 판단 경계",
                "4배 차이 (임상적으로 유의미)",
                "8배 차이 (심각한 오차)"
            ],
            "권장 조치": [
                "정상",
                "정상",
                "재확인 권장",
                "재검사 고려",
                "재검사 필요",
                "샘플/모델 재검토"
            ]
        })
        
        st.table(meaning_df)
        
        st.caption("""
        **참고 문헌:**
        - Technical replicate SD: < 0.5 Ct (Bustin et al., 2009)
        - Biological replicate SD: < 1.0 Ct (MIQE Guidelines)
        - Clinical threshold: ΔCt < 1.5 (1.5배 차이)
        """)
    
    st.divider()

def try_eval_if_truth_exists(df_raw: pd.DataFrame, pred_df: pd.DataFrame, truth_df: pd.DataFrame | None = None) -> None:
    # ✅ truth_df가 있으면 그걸 우선으로 평가
    src = truth_df if truth_df is not None else df_raw

    true_col = None
    for cand in ["true_ct", "TrueCt", "trueCt", "ct", "Ct", "CT", "Cq", "cq", "CQ"]:
        if cand in src.columns:
            true_col = cand
            break

    if true_col is None:
        st.info("업로드 파일에 정답 Ct/Cq 컬럼(true_ct/ct/cq 등)이 없어서 즉석 평가는 생략했어요.")
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
        
        # ✅ Well 표준화 (핵심)
        truth2["Well"] = truth2["Well"].map(normalize_well)
        eval_df["Well"] = eval_df["Well"].map(normalize_well)
        
        eval_df = eval_df.merge(truth2, on="Well", how="left")

    else:
        eval_df["true_ct"] = pd.to_numeric(src[true_col], errors="coerce").values[: len(eval_df)]

    eval_df["true_ct"] = pd.to_numeric(eval_df["true_ct"], errors="coerce")
    eval_df = eval_df.dropna(subset=["true_ct", "pred_ct"]).copy()
    if len(eval_df) == 0:
        st.warning("정답 Ct 컬럼은 찾았는데, pred와 매칭된 값이 없어요. Well 이름이 맞는지 확인해줘.")
        return

    eval_df["err"] = eval_df["pred_ct"] - eval_df["true_ct"]
    mae = float(np.mean(np.abs(eval_df["err"])))
    rmse = float(np.sqrt(np.mean(eval_df["err"] ** 2)))

    st.markdown("### ✅ 업로드 데이터 즉석 평가")
    st.write({"MAE": mae, "RMSE": rmse, "n": int(len(eval_df))})

    st.markdown("**Pred vs True (산점도)**")
    st.scatter_chart(eval_df[["true_ct", "pred_ct"]], x="true_ct", y="pred_ct", height=320)

    st.markdown("**Residual(오차) 분포**")
    # value_counts는 히스토그램 느낌이 약해서, bins로 가볍게
    bins = np.linspace(eval_df["err"].min(), eval_df["err"].max(), 30) if len(eval_df) > 3 else None
    if bins is not None:
        hist, edges = np.histogram(eval_df["err"].values, bins=bins)
        hist_df = pd.DataFrame({"err_bin_left": edges[:-1], "count": hist}).set_index("err_bin_left")
        st.line_chart(hist_df["count"], height=220)
    else:
        st.line_chart(eval_df["err"], height=220)

def load_curve_from_master(run_id: str, well_id: str) -> pd.DataFrame:
    """
    canonical master_long.parquet에서 (run_id, well_id) 한 곡선만 로드
    필요한 컬럼명이 환경마다 달라서 유연하게 매핑함.
    """
    path = PROJECT_ROOT / "data" / "canonical" / "master_long.parquet"
    if not path.exists():
        raise FileNotFoundError(f"master_long.parquet not found: {path}")

    dataset = ds.dataset(str(path))

    # 컬럼 후보들(프로젝트마다 이름이 약간 다를 수 있어서)
    cols = set(dataset.schema.names)
    cycle_col = "Cycle" if "Cycle" in cols else ("cycle" if "cycle" in cols else None)
    fluor_col = "Fluor" if "Fluor" in cols else ("fluor" if "fluor" in cols else None)
    run_col   = "run_id" if "run_id" in cols else None
    
    # ✅ 여기가 핵심: Well이 있으면 Well을 최우선으로
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
    
    # ✅ well_uid를 쓰는 경우에는 run_id:Well 형태로 맞춰줌
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
    """
    predictions_long.parquet row에 curve_cycles_json / curve_fluor_json 이 있으면
    그걸로 원본 곡선을 복원한다 (Streamlit Cloud fallback).
    Returns: DataFrame with columns ["Cycle","Fluor"] sorted by Cycle
    """
    import json
    cycles_json = row.get("curve_cycles_json", "") if isinstance(row, dict) else getattr(row, "curve_cycles_json", "")
    fluor_json  = row.get("curve_fluor_json", "")  if isinstance(row, dict) else getattr(row, "curve_fluor_json", "")

    if not cycles_json or not fluor_json:
        raise ValueError("curve_cycles_json / curve_fluor_json is empty (retrain on server with curve embedding).")

    cycles = json.loads(cycles_json)
    fluor  = json.loads(fluor_json)

    df = pd.DataFrame({"Cycle": cycles, "Fluor": fluor})
    df = df.dropna().sort_values("Cycle").reset_index(drop=True)
    return df

def show_hard_review() -> None:
    st.subheader("🧨 Hard Sample Review (서버 평가 로그 기반)")

    # active model id
    model_id = get_active_model_id()

    pred_path = PROJECT_ROOT / "reports" / model_id / "predictions_long.parquet"

    if not pred_path.exists():
        st.info(f"predictions_long.parquet가 없어요: {pred_path}\n재학습을 한 번 실행해서 생성해줘.")
        return

    pred = pd.read_parquet(pred_path)
    need_cols = {"run_id", "well_id", "cutoff", "true_ct", "pred_ct"}
    if not need_cols.issubset(set(pred.columns)):
        st.error(f"predictions_long.parquet 컬럼이 예상과 달라요. 필요: {need_cols}, 현재: {set(pred.columns)}")
        return

    pred = pred.copy()
    pred["abs_err"] = (pred["pred_ct"] - pred["true_ct"]).abs()

    c_list = sorted(pred["cutoff"].dropna().unique().astype(int).tolist())
    if not c_list:
        st.warning("cutoff 값이 비어있어요.")
        return

    col1, col2, col3 = st.columns([1.0, 1.0, 1.5])
    with col1:
        best_cutoff = get_best_cutoff_from_report()
        if best_cutoff in c_list:
            default_idx = c_list.index(best_cutoff)
        else:
            default_idx = min(len(c_list)-1, 0)
    
        cutoff = st.selectbox("cutoff 선택", c_list, index=default_idx)
    with col2:
        topk = st.slider("Hard Top-K", min_value=5, max_value=200, value=30, step=5)
    with col3:
        err_thr = st.number_input("또는 |error| 임계값", value=0.0, step=0.5, help="0이면 Top-K 기준만 사용")

    df = pred[pred["cutoff"] == int(cutoff)].copy()
    if err_thr > 0:
        hard = df[df["abs_err"] >= float(err_thr)].sort_values("abs_err", ascending=False)
    else:
        hard = df.sort_values("abs_err", ascending=False).head(int(topk))
        
    # =========================
    # 0) 전체 후보군 대비 hard 선정 이유/랭크 만들기
    # =========================
    df = df.copy()
    df["err"] = df["pred_ct"] - df["true_ct"]
    df = df.sort_values("abs_err", ascending=False).reset_index(drop=True)
    df["rank_abs_err"] = np.arange(1, len(df) + 1)  # 1이 가장 hard
    
    # hard set 표시
    if err_thr > 0:
        rule_text = f"|error| ≥ {float(err_thr):.3f}"
        df["is_hard"] = df["abs_err"] >= float(err_thr)
    else:
        rule_text = f"Top-{int(topk)} by |error|"
        df["is_hard"] = df["rank_abs_err"] <= int(topk)
    
    hard = df[df["is_hard"]].copy()
    hard = hard.sort_values("abs_err", ascending=False).reset_index(drop=True)
    
    st.markdown("## 🎯 Pred vs True에서 Hard 강조 보기")

    # hard id set 만들기 (run_id, well_id)
    hard_ids = set(zip(hard["run_id"].astype(str), hard["well_id"].astype(str)))
    
    # (아직 선택 샘플 rid/wid 만들기 전이면 highlight=None으로 먼저 그리고,
    #  선택 샘플 선택 후에 다시 한 번 그려도 됨)
    plot_pred_vs_true_hard_colored(
        df_cut=df,                 # 해당 cutoff 전체
        hard_ids=hard_ids,
        highlight=None
    )
    
    st.caption("선(y=x)에서 많이 벗어난 점들을 Hard로 표시해서 '우리가 이걸 분석 중'이라는 느낌을 줍니다.")
    
    # =========================
    # 1) Hard 선정 요약 카드 (전체 대비)
    # =========================
    st.markdown("## 🔎 Hard 선정 요약")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cutoff", int(cutoff))
    c2.metric("전체 후보(해당 cutoff)", int(len(df)))
    c3.metric("Hard 후보", int(len(hard)))
    hard_ratio = (len(hard) / len(df) * 100.0) if len(df) else 0.0
    c4.metric("Hard 비율", f"{hard_ratio:.1f}%")
    st.caption(f"Hard 선정 기준: **{rule_text}**")
    st.divider()
    
    # =========================
    # 2) 전체 분포에서 hard 위치 보기 (히스토그램)
    # =========================
    st.markdown("## 📊 전체 후보 대비 Hard 위치")
    import altair as alt
    
    hist_base = alt.Chart(df).mark_bar().encode(
        x=alt.X("abs_err:Q", bin=alt.Bin(maxbins=40), title="|error| (abs_err)"),
        y=alt.Y("count():Q", title="Count"),
        tooltip=[alt.Tooltip("count():Q", title="Count")]
    )
    
    layers = [hist_base]
    
    # threshold 라인 또는 topk 컷 라인
    if err_thr > 0:
        thr_df = pd.DataFrame({"abs_err_thr": [float(err_thr)]})
        thr_line = alt.Chart(thr_df).mark_rule(strokeDash=[6,4]).encode(x="abs_err_thr:Q")
        layers.append(thr_line)
    else:
        # TopK의 마지막 abs_err 값을 컷으로 표시
        if len(hard) > 0:
            kth = float(hard["abs_err"].min())
            kth_df = pd.DataFrame({"abs_err_kth": [kth]})
            kth_line = alt.Chart(kth_df).mark_rule(strokeDash=[6,4]).encode(x="abs_err_kth:Q")
            layers.append(kth_line)
    
    st.altair_chart(alt.layer(*layers).properties(height=220), use_container_width=True)
    st.divider()
    
    # 표는 접어서 보기 옵션
    with st.expander("📋 Hard 후보 표(접기/펼치기)", expanded=False):
        st.dataframe(
            hard[["run_id", "well_id", "true_ct", "pred_ct", "abs_err", "rank_abs_err"]],
            use_container_width=True,
            height=260
        )


    st.caption(f"model_id={model_id} / cutoff={cutoff} / candidates={len(hard)}")
    st.dataframe(hard[["run_id", "well_id", "true_ct", "pred_ct", "abs_err"]], use_container_width=True, height=320)

    # ---- 선택 ----
    if len(hard) == 0:
        return
    
    items = hard.reset_index(drop=True)  # 0..N-1 인덱스 고정
    
    # ✅ 세션 인덱스 키 (정수)
    if "hard_pick_i" not in st.session_state:
        st.session_state["hard_pick_i"] = 0
    
    # ✅ TopK/threshold 바뀌어 items 길이가 바뀌어도 안전하게 clamp
    st.session_state["hard_pick_i"] = max(
        0, min(int(st.session_state["hard_pick_i"]), len(items) - 1)
    )
    
    st.caption(f"📌 {st.session_state['hard_pick_i']+1} / {len(items)}")
    
    def _fmt(i: int) -> str:
        r = items.iloc[i]
        return f"{r['run_id']} | {r['well_id']} | {r['abs_err']:.3f}"
    
    pick_i = st.selectbox(
        "검토할 샘플 선택 (run_id | well_id | abs_err)",
        options=list(range(len(items))),
        index=int(st.session_state["hard_pick_i"]),
        format_func=_fmt,
        key="hard_pick_i_selectbox",
    )
    
    # ✅ 선택값 동기화 (사용자가 직접 드롭다운으로 바꿔도 idx 갱신)
    st.session_state["hard_pick_i"] = int(pick_i)
    
    rid = str(items.loc[pick_i, "run_id"])
    wid = str(items.loc[pick_i, "well_id"])
    
    # =========================
    # 3) 선택 샘플이 hard로 들어온 이유 설명
    # =========================
    chosen = df[(df["run_id"] == rid) & (df["well_id"] == wid)]
    if len(chosen) > 0:
        chosen = chosen.iloc[0]
        st.markdown("### 🧠 왜 이 샘플이 Hard인가?")
        reason_lines = [
            f"- Hard 선정 기준: **{rule_text}**",
            f"- 이 샘플 |error| = **{float(chosen['abs_err']):.3f}** (err={float(chosen['err']):+.3f})",
            f"- 전체 후보 {len(df)}개 중 |error| 순위: **{int(chosen['rank_abs_err'])}위**",
        ]
        if err_thr > 0:
            reason_lines.append(f"- 임계값 {float(err_thr):.3f} {'통과' if float(chosen['abs_err']) >= float(err_thr) else '미통과'}")
        else:
            reason_lines.append(f"- Top-{int(topk)} {'포함' if int(chosen['rank_abs_err']) <= int(topk) else '미포함'}")
        st.write("\n".join(reason_lines))

    
    sel = items.iloc[int(pick_i)]
    st.markdown("### 선택 샘플 요약")
    st.write(
        {
            "run_id": rid,
            "well_id": wid,
            "cutoff": int(cutoff),
            "true_ct": float(sel["true_ct"]),
            "pred_ct": float(sel["pred_ct"]),
            "abs_err": float(sel["abs_err"]),
        }
    )
    
    # ---- 플롯 ----
    st.markdown("### 📈 원본 qPCR 곡선 보기 (master_long 우선, 없으면 predictions_long JSON fallback)")
    cutoff_i = int(cutoff)

    try:
        curve = None

        # 1) master_long 있으면 우선 로드
        if has_canonical_master_long():
            curve = load_curve_from_master(rid, wid)

        # 2) 없거나(Cloud) / 못찾았거나 / 빈 df면 -> predictions_long의 JSON으로 복원
        if curve is None or len(curve) == 0:
            curve = load_one_curve_from_predictions_row(sel.to_dict())

        if curve is None or len(curve) == 0:
            st.info("곡선 데이터를 찾지 못했어. (master_long도 없고, predictions_long JSON도 비어있음)")
        else:
            curve = curve.sort_values("Cycle").reset_index(drop=True)
            curve["segment"] = np.where(curve["Cycle"] <= cutoff_i, "early(<=cutoff)", "late")

            import altair as alt

            line = (
                alt.Chart(curve)
                .mark_line()
                .encode(
                    x=alt.X("Cycle:Q", title="Cycle"),
                    y=alt.Y("Fluor:Q", title="Fluor"),
                    tooltip=["Cycle", "Fluor", "segment"],
                )
            )
            vline = (
                alt.Chart(pd.DataFrame({"Cycle": [cutoff_i]}))
                .mark_rule(strokeDash=[6, 4])
                .encode(x="Cycle:Q")
            )
            st.altair_chart(line + vline, use_container_width=True)

            st.markdown("#### 🔍 Early 구간 확대 (<= cutoff)")
            early = curve[curve["Cycle"] <= cutoff_i].copy()
            if len(early) < 2:
                st.info("early 구간 데이터가 너무 적어서 확대 플롯을 못 그려.")
            else:
                eline = (
                    alt.Chart(early)
                    .mark_line()
                    .encode(
                        x=alt.X("Cycle:Q", title="Cycle (early)"),
                        y=alt.Y("Fluor:Q", title="Fluor"),
                        tooltip=["Cycle", "Fluor"],
                    )
                )
                st.altair_chart(eline, use_container_width=True)

    except Exception as e:
        st.warning(f"곡선 로딩/플롯 실패: {e}")
    
    # ---- 빠른 이동 ----
    st.markdown("### ⏭️ 빠른 이동")
    
    def _next_sample():
        # 버튼 콜백에서만 idx 증가 (여기서 set하면 Streamlit이 자연스럽게 rerun함)
        i = int(st.session_state.get("hard_pick_i", 0))
        if i < len(items) - 1:
            st.session_state["hard_pick_i"] = i + 1
            st.session_state["hard_pick_i_selectbox"] = i + 1  # selectbox도 같이 밀어줌(안전)
        else:
            st.session_state["hard_pick_i"] = len(items) - 1
    
    st.button("다음 샘플 보기", type="secondary", key="btn_next_hard", on_click=_next_sample)

# -------------------------
# UI
# -------------------------
st.caption("업로드한 qPCR curve 데이터로 Ct를 예측하거나, 서버에 누적된 데이터로 모델을 재학습할 수 있어요.")

# -------------------------
# Sidebar (cutoff / retrain range)
# -------------------------
cutoffs = discover_cutoffs(MODELS_DIR)
if not cutoffs:
    st.error(f"모델을 찾지 못했어: {MODELS_DIR} (ct_xgb_cutoff_*.json 없음)")
    st.stop()
    
# discover_cutoffs(MODELS_DIR) 후, cutoff selectbox 전에 통째로

# discover_cutoffs 아래, cutoff selectbox 전에 이 코드 넣기 (기존 사이드바 코드 삭제)

# discover_cutoffs(MODELS_DIR) 후, cutoff selectbox 전에 이 코드로 교체

# 수정된 코드 (버튼 부분만 삭제):
with st.sidebar:
    st.title("CPHOTONICS | Early Ct Predictor")
    st.divider()
    
    # 기존 cutoff 등 (이 아래 그대로)
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

# 수정된 코드:
tabs = st.tabs(["📈 Performance", "📊 Data Catalog", "🧪 Predict (Upload)", "🧨 Hard Review", "🛠 Retrain(Admin)"])

with tabs[0]:
    show_train_report()

# ============================================
# Data Catalog 탭 - 완전히 새로운 버전
# streamlit_app.py의 tabs[1] 부분에 넣을 코드
# ============================================

with tabs[1]:  # Data Catalog 탭
    st.header("📊 Data Quality Control & Catalog")
    st.markdown("QC 분석 결과를 시각화하고 다운로드할 수 있습니다.")
    
    # ========================================
    # 1. QC Catalog 로드
    # ========================================
    @st.cache_data
    def load_qc_catalog():
        qc_path = QC_DIR / "master_catalog.parquet"
        if qc_path.exists():
            return pd.read_parquet(qc_path)
        return pd.DataFrame()
    
    qc_df = load_qc_catalog()
    
    if qc_df.empty:
        st.warning("⚠️ QC catalog not found")
        st.info("""
        **QC 데이터를 생성하려면:**
        
        서버에서 실행:
        ```bash
        cd ~/qpcr_v2
        python scripts/save_qc_results.py
        ```
        """)
        st.stop()
    
    # ========================================
    # 2. 전체 요약 통계 (상단 카드)
    # ========================================
    total = len(qc_df)
    pass_c = (qc_df['qc_status'] == 'PASS').sum()
    fail_c = (qc_df['qc_status'] == 'FAIL').sum()
    flag_c = (qc_df['qc_status'] == 'FLAG').sum()
    usable = qc_df['usable'].sum() if 'usable' in qc_df.columns else pass_c
    
    st.subheader("📈 전체 요약")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Wells", f"{total:,}")
    col2.metric("✅ PASS", f"{pass_c:,}", f"{pass_c/total*100:.1f}%")
    col3.metric("❌ FAIL", f"{fail_c:,}", f"{fail_c/total*100:.1f}%")
    col4.metric("⚠️ FLAG", f"{flag_c:,}", f"{flag_c/total*100:.1f}%")
    col5.metric("🟢 Usable", f"{usable:,}", f"{usable/total*100:.1f}%")
    
    st.divider()
    
    # ========================================
    # 3. 시각화 (2열)
    # ========================================
    st.subheader("📊 QC 분포 시각화")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown("#### QC Status Distribution")
        status_counts = qc_df['qc_status'].value_counts()
        
        import plotly.express as px
        fig_pie = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            color=status_counts.index,
            color_discrete_map={
                'PASS': '#00CC66',
                'FAIL': '#FF4444', 
                'FLAG': '#FFAA00'
            },
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=True, height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with viz_col2:
        st.markdown("#### Ct Bin Distribution")
        if 'ct_bin' in qc_df.columns:
            ct_counts = qc_df['ct_bin'].value_counts().sort_index()
            fig_bar = px.bar(
                x=ct_counts.index,
                y=ct_counts.values,
                labels={'x': 'Ct Bin', 'y': 'Count'},
                color=ct_counts.values,
                color_continuous_scale='Viridis'
            )
            fig_bar.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("ct_bin 컬럼이 없습니다.")
    
    # ========================================
    # 4. Fail Reason 분석 (Top 10)
    # ========================================
    if 'fail_reason' in qc_df.columns:
        st.subheader("🔍 Fail Reason Analysis (Top 10)")
        
        fail_df = qc_df[qc_df['qc_status'] != 'PASS'].copy()
        if not fail_df.empty:
            fail_counts = fail_df['fail_reason'].value_counts().head(10)
            
            fig_fail = px.bar(
                x=fail_counts.values,
                y=fail_counts.index,
                orientation='h',
                labels={'x': 'Count', 'y': 'Fail Reason'},
                color=fail_counts.values,
                color_continuous_scale='Reds'
            )
            fig_fail.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_fail, use_container_width=True)
        else:
            st.success("🎉 No failed samples!")
    
    st.divider()
    
    # ========================================
    # 5. QC Metrics 분포 (r2, snr)
    # ========================================
    if 'r2' in qc_df.columns and 'snr' in qc_df.columns:
        st.subheader("📐 QC Metrics Distribution")
        
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            fig_r2 = px.histogram(
                qc_df,
                x='r2',
                nbins=50,
                labels={'r2': 'R² Value'},
                title='R² Distribution',
                color_discrete_sequence=['#3498db']
            )
            fig_r2.add_vline(x=0.98, line_dash="dash", line_color="red", 
                            annotation_text="Threshold (0.98)")
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with metric_col2:
            fig_snr = px.histogram(
                qc_df,
                x='snr',
                nbins=50,
                labels={'snr': 'SNR Value'},
                title='SNR Distribution',
                color_discrete_sequence=['#e74c3c']
            )
            st.plotly_chart(fig_snr, use_container_width=True)
    
    st.divider()
    
    # ========================================
    # 6. 필터링 가능한 테이블
    # ========================================
    st.subheader("🔎 Detailed Data Table (Filterable)")
    
    # 필터 옵션
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        status_filter = st.multiselect(
            "QC Status",
            options=['PASS', 'FAIL', 'FLAG'],
            default=['PASS', 'FAIL', 'FLAG']
        )
    
    with filter_col2:
        if 'ct_bin' in qc_df.columns:
            ct_bins = sorted(qc_df['ct_bin'].dropna().unique())
            ct_filter = st.multiselect(
                "Ct Bin",
                options=ct_bins,
                default=ct_bins
            )
        else:
            ct_filter = None
    
    with filter_col3:
        search_well = st.text_input("Search Well ID", "")
    
    # 필터 적용
    filtered = qc_df[qc_df['qc_status'].isin(status_filter)].copy()
    
    if ct_filter and 'ct_bin' in qc_df.columns:
        filtered = filtered[filtered['ct_bin'].isin(ct_filter)]
    
    if search_well:
        if 'well_uid' in filtered.columns:
            filtered = filtered[filtered['well_uid'].str.contains(search_well, case=False, na=False)]
        elif 'Well' in filtered.columns:
            filtered = filtered[filtered['Well'].str.contains(search_well, case=False, na=False)]
    
    st.write(f"Showing **{len(filtered):,}** / **{total:,}** wells")
    
    # 표시할 컬럼 선택
    display_cols = ['well_uid', 'run_id', 'Well', 'ct_value', 'ct_bin', 
                   'qc_status', 'fail_reason', 'usable', 'r2', 'snr']
    available_cols = [c for c in display_cols if c in filtered.columns]
    
    # 테이블 표시
    st.dataframe(
        filtered[available_cols],
        use_container_width=True,
        height=500
    )
    
    # ========================================
    # 7. 다운로드 버튼
    # ========================================
    st.subheader("💾 Download Data")
    
    download_col1, download_col2 = st.columns(2)
    
    with download_col1:
        # 전체 데이터 다운로드
        csv_all = qc_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download All Data (CSV)",
            data=csv_all,
            file_name=f"qc_catalog_all_{cutoff}.csv",
            mime="text/csv"
        )
    
    with download_col2:
        # 필터된 데이터 다운로드
        csv_filtered = filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_filtered,
            file_name=f"qc_catalog_filtered_{cutoff}.csv",
            mime="text/csv"
        )
    
    # ========================================
    # 8. 상세 통계 (접기/펼치기)
    # ========================================
    with st.expander("📊 상세 통계 보기", expanded=False):
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        
        with stat_col1:
            st.markdown("**Ct Value 통계**")
            if 'ct_value' in qc_df.columns:
                st.write({
                    "Mean": f"{qc_df['ct_value'].mean():.2f}",
                    "Median": f"{qc_df['ct_value'].median():.2f}",
                    "Std": f"{qc_df['ct_value'].std():.2f}",
                    "Min": f"{qc_df['ct_value'].min():.2f}",
                    "Max": f"{qc_df['ct_value'].max():.2f}"
                })
        
        with stat_col2:
            st.markdown("**R² 통계**")
            if 'r2' in qc_df.columns:
                st.write({
                    "Mean": f"{qc_df['r2'].mean():.4f}",
                    "Median": f"{qc_df['r2'].median():.4f}",
                    "Min": f"{qc_df['r2'].min():.4f}",
                    "< 0.98": f"{(qc_df['r2'] < 0.98).sum()} wells"
                })
        
        with stat_col3:
            st.markdown("**SNR 통계**")
            if 'snr' in qc_df.columns:
                st.write({
                    "Mean": f"{qc_df['snr'].mean():.2f}",
                    "Median": f"{qc_df['snr'].median():.2f}",
                    "Min": f"{qc_df['snr'].min():.2f}",
                    "Max": f"{qc_df['snr'].max():.2f}"
                })
            
with tabs[2]:
    st.subheader("🧪 Predict (Upload)")
    up = st.file_uploader("qPCR 파일 업로드 (csv/xlsx)", type=["csv", "xlsx", "xls"])
    if up is None:
        st.info("파일을 업로드하면 예측을 진행해요.")
    else:
        run_id = _safe_stem(up.name) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        # ✅ 1) 업로드 읽기
        raw_obj = read_uploaded_table(up)

        # ✅ 2) 엑셀이면 시트 분리, CSV면 그대로
        df_curve, df_truth, curve_sheet, truth_sheet = split_excel_sheets(raw_obj)
        if isinstance(raw_obj, dict):
            st.write("📄 Sheets:", list(raw_obj.keys()))
            for nm, df_ in raw_obj.items():
                st.write(f"--- {nm} ---")
                st.write("columns:", [str(c) for c in df_.columns])

        # ✅ 3) 미리보기
        if curve_sheet is not None:
            st.caption(f"업로드 curve 시트: {curve_sheet}")
        st.dataframe(df_curve.head(30), use_container_width=True)

        if df_truth is not None:
            st.caption(f"정답 Ct/Cq 시트 감지됨: {truth_sheet}")
            st.dataframe(df_truth.head(30), use_container_width=True)

        # ✅ 4) long 변환은 curve_df로만!
        try:
            df_long = infer_long_df(df_curve, run_id=run_id)
        except Exception as e:
            st.error(f"업로드 파일을 long 형태로 변환 실패: {e}")
            st.stop()
        
        # =========================
        # (NEW) Multi-cutoff Sweep (업로드 데이터)
        # =========================
        do_sweep = st.toggle("여러 cutoff로 한 번에 비교(3/5 step)", value=False, key="do_sweep_upload")


        if do_sweep:
            step2 = st.radio("step", [3, 5], horizontal=True, key="upload_step")
            sweep_min = st.number_input("min cutoff", 1, 200, 10, 1, key="upload_min")
            sweep_max = st.number_input("max cutoff", 1, 200, 40, 1, key="upload_max")

            sweep_cutoffs = list(range(int(sweep_min), int(sweep_max) + 1, int(step2)))

            missing = []
            preds_all = []
            
            with st.spinner(f"Sweep 예측 중... (cutoffs={sweep_cutoffs})"):
                for c in sweep_cutoffs:
                    try:
                        p = predict_ct(df_long, cutoff=int(c))
                        p["cutoff"] = int(c)
            
                        # 업로드 예측 결과에 well_id 없을 수 있어서 안전하게 생성
                        if "well_id" not in p.columns:
                            if "Well" in p.columns:
                                p["well_id"] = p["Well"].astype(str)
                            elif "well_uid" in p.columns:
                                p["well_id"] = p["well_uid"].astype(str)
            
                        preds_all.append(p)
            
                    except Exception as e:
                        # 모델 파일 없는 cutoff(=No such file...)면 여기로 들어옴
                        missing.append((int(c), str(e)))
            
            if missing:
                st.warning("일부 cutoff는 모델 파일이 없어서 스킵했어: " + ", ".join(str(c) for c, _ in missing))
            
            if not preds_all:
                st.error("선택한 cutoff 범위에서 사용 가능한 모델이 하나도 없어. min/max를 줄여줘.")
                st.stop()
            
            preds_all = pd.concat(preds_all, ignore_index=True)


            # ---- truth 있으면 cutoff별 성능(MAE/RMSE) ----
            if df_truth is not None:
                tmp = preds_all.copy()

                # true 컬럼 찾기
                true_col = None
                for cand in ["true_ct", "ct", "Ct", "Cq", "cq", "CQ", "CT"]:
                    if cand in df_truth.columns:
                        true_col = cand
                        break

                # well 컬럼 찾기
                wcol = None
                for cand in ["Well", "well", "WELL"]:
                    if cand in df_truth.columns:
                        wcol = cand
                        break

                if true_col and wcol:
                    truth2 = df_truth[[wcol, true_col]].copy()
                    truth2.columns = ["Well", "true_ct"]

                    # ✅ Well 표준화(매칭 핵심)
                    truth2["Well"] = truth2["Well"].map(normalize_well)
                    if "Well" in tmp.columns:
                        tmp["Well"] = tmp["Well"].map(normalize_well)
                    elif "well_id" in tmp.columns:
                        tmp["well_id"] = tmp["well_id"].map(normalize_well)

                    # merge 키: tmp에 Well이 있으면 Well로, 없으면 well_id로
                    if "Well" in tmp.columns:
                        tmp = tmp.merge(truth2, on="Well", how="left")
                    else:
                        tmp = tmp.merge(truth2.rename(columns={"Well": "well_id"}), on="well_id", how="left")

                    tmp["err"] = tmp["pred_ct"] - tmp["true_ct"]
                    g = tmp.dropna(subset=["true_ct", "pred_ct"]).groupby("cutoff")

                    perf = g.apply(lambda x: pd.Series({
                        "mae": float(np.mean(np.abs(x["err"]))),
                        "rmse": float(np.sqrt(np.mean(x["err"] ** 2))),
                        "n": int(len(x)),
                    })).reset_index()

                    st.markdown("#### cutoff별 MAE/RMSE")
                    pm = perf.melt(id_vars=["cutoff"], value_vars=["mae", "rmse"], var_name="metric", value_name="value")
                    st.altair_chart(
                        alt.Chart(pm).mark_line(point=True).encode(
                            x="cutoff:Q",
                            y="value:Q",
                            strokeDash="metric:N",
                            tooltip=["cutoff", "metric", "value", "n:Q"],
                        ).properties(height=260),
                        use_container_width=True
                    )

                    # ---- Pred vs True small multiples (sweep) ----
                    st.markdown("#### Pred vs True (Sweep Small Multiples)")
                    ncol = st.slider("한 줄에 몇 개?", 2, 6, 4, 1, key="upload_sweep_cols")

                    # plot_pred_vs_true_facets가 요구하는 컬럼 맞추기
                    if "run_id" not in tmp.columns:
                        tmp["run_id"] = run_id
                    if "well_id" not in tmp.columns:
                        tmp["well_id"] = tmp["Well"].astype(str)

                    plot_pred_vs_true_facets(tmp.rename(columns={"pred_ct": "pred_ct", "true_ct": "true_ct"}), sweep_cutoffs, ncol=int(ncol))
                else:
                    st.info("Sweep 성능을 계산하려면 truth 시트에 Well + (ct/cq/true_ct) 컬럼이 있어야 해요.")

            st.divider()

        # ✅ 5) 예측
        pred_df = predict_ct(df_long, cutoff=int(cutoff))
        st.success("예측 완료!")

        # =========================
        # (NEW) Prediction 신뢰도/정답률(확률) 요약
        # =========================
        st.markdown("### ✅ 예측 신뢰도 / 몇 개 맞췄는지")
        
        tol_u = st.slider(
            "정답 판정 기준 (|pred-true| <= tol)",
            0.5, 5.0, 2.0, 0.5,
            key="upload_tol_summary",
        )
        
        # --- 서버 로그 기반 '예상 정답률(확률)' 계산 (있으면) ---
        active_path = PROJECT_ROOT / "reports" / "active_model.txt"
        model_id = active_path.read_text().strip() if active_path.exists() else "model_server_latest_xgb"
        
        pred_path = PROJECT_ROOT / "reports" / model_id / "predictions_long.parquet"

        expected_rate = None
        if pred_path.exists():
            try:
                server_pred_long = pd.read_parquet(pred_path)
                acc_df_srv = perf_accuracy_fraction_vs_cutoff(server_pred_long, tol=float(tol_u))
                hit = acc_df_srv[acc_df_srv["cutoff"] == int(cutoff)]
                if not hit.empty:
                    expected_rate = float(hit["acc_frac"].iloc[0])
            except Exception:
                expected_rate = None
        
        # --- 업로드 파일 truth가 있으면 "맞춘 개수" 계산 ---
        def _build_eval_df_with_truth(pred_df: pd.DataFrame, df_truth: pd.DataFrame) -> pd.DataFrame:
            # true_ct column 찾기
            true_col = None
            for cand in ["true_ct", "TrueCt", "trueCt", "ct", "Ct", "CT", "Cq", "cq", "CQ"]:
                if cand in df_truth.columns:
                    true_col = cand
                    break
            if true_col is None:
                return pd.DataFrame()
        
            # Well column 찾기
            wcol = None
            for cand in ["Well", "well", "WELL"]:
                if cand in df_truth.columns:
                    wcol = cand
                    break
            if wcol is None:
                return pd.DataFrame()
        
            truth2 = df_truth[[wcol, true_col]].copy()
            truth2.columns = ["Well", "true_ct"]
        
            out = pred_df.copy()
            if "Well" not in out.columns:
                return pd.DataFrame()
        
            # Well 표준화
            truth2["Well"] = truth2["Well"].map(normalize_well)
            out["Well"] = out["Well"].map(normalize_well)
        
            out = out.merge(truth2, on="Well", how="left")
            out["pred_ct"] = pd.to_numeric(out["pred_ct"], errors="coerce")
            out["true_ct"] = pd.to_numeric(out["true_ct"], errors="coerce")
            out = out.dropna(subset=["pred_ct", "true_ct"]).copy()
            if out.empty:
                return pd.DataFrame()
        
            out["abs_err"] = (out["pred_ct"] - out["true_ct"]).abs()
            out["within_tol"] = out["abs_err"] <= float(tol_u)
            return out
        
        eval_df = pd.DataFrame()
        if "pred_df" in locals() and pred_df is not None and df_truth is not None:
            eval_df = _build_eval_df_with_truth(pred_df, df_truth)
        else:
            st.info("예측 결과(pred_df) 또는 truth(df_truth)가 아직 없어서 신뢰도 계산을 생략했어.")

        # --- UI 출력 (truth 있으면 실제 정답률 / 없으면 예상 정답률) ---
        cA, cB, cC, cD = st.columns(4)
        
        if not eval_df.empty:
            n_total = int(len(eval_df))
            n_hit = int(eval_df["within_tol"].sum())
            hit_rate = (n_hit / n_total) if n_total else 0.0
            mae_u = float(eval_df["abs_err"].mean())
            rmse_u = float(np.sqrt(np.mean((eval_df["pred_ct"] - eval_df["true_ct"]) ** 2)))
        
            cA.metric("맞춘 개수", f"{n_hit} / {n_total}")
            cB.metric(f"정답률(±{tol_u:g})", f"{hit_rate*100:.1f}%")
            cC.metric("MAE(업로드)", f"{mae_u:.3f}")
            cD.metric("RMSE(업로드)", f"{rmse_u:.3f}")
        
            if expected_rate is not None:
                st.caption(f"참고: 서버 평가 로그 기준 이 cutoff의 **예상 정답률(±{tol_u:g}) ≈ {expected_rate*100:.1f}%**")
        
            st.progress(hit_rate)
            st.caption("해석: 정답률은 **|pred-true| <= tol** 만족 비율입니다.")
        
            # (옵션) Well별 오차 막대 + 색상으로 맞/틀 표시
            with st.expander("🔍 Well별 오차(맞/틀 색상) 보기", expanded=False):
                import altair as alt
                bar = (
                    alt.Chart(eval_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Well:N", sort=alt.SortField("abs_err", order="descending"), title="Well"),
                        y=alt.Y("abs_err:Q", title="|Error|"),
                        color=alt.Color("within_tol:N", title=f"within ±{tol_u:g}", legend=None),
                        tooltip=["Well", "true_ct", "pred_ct", "abs_err", "within_tol"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(bar, use_container_width=True)
        
        else:
            # truth 없을 때: 예상 정답률만
            if expected_rate is not None:
                cA.metric(f"예상 정답률(±{tol_u:g})", f"{expected_rate*100:.1f}%")
                cB.metric("cutoff", int(cutoff))
                cC.metric("Wells", int(df_long["Well"].nunique()) if "Well" in df_long.columns else int(df_long["well_uid"].nunique()))
                cD.metric("기준", "서버 로그 기반")
                st.progress(expected_rate)
                st.caption(
                    f"truth(정답 Ct/Cq)이 없어서 업로드 데이터의 '맞춘 개수'는 계산 불가. "
                    f"대신 서버 평가 로그에서 **이 cutoff가 ±{tol_u:g} 안에 들어갈 확률(예상 정답률)** 을 보여줘요."
                )
            else:
                st.info(
                    "truth(정답 Ct/Cq) 시트도 없고, 서버 평가 로그(predictions_long.parquet)도 없어서 "
                    "정답률/확률 요약을 표시할 수 없어요."
                )

        # ---- 요약 카드 ----
        c1, c2, c3 = st.columns(3)
        c1.metric("Cutoff", int(cutoff))
        c2.metric("Wells", int(pred_df["Well"].nunique()) if "Well" in pred_df.columns else len(pred_df))
        c3.metric("Pred Ct (min ~ max)", f"{pred_df['pred_ct'].min():.3f} ~ {pred_df['pred_ct'].max():.3f}")

        st.divider()

        # ✅ 표는 기본 숨김(원하면 켜기)
        show_tables = st.toggle("표도 같이 보기(디버깅용)", value=False, key="pred_show_tables")

        # ---- figure 1) 업로드 곡선 preview ----
        st.markdown("### 📈 업로드 곡선 Preview")
        plot_uploaded_curve_preview(df_long, cutoff=int(cutoff), max_wells=6)

        # ---- figure 2) Pred Ct 분포 ----
        st.markdown("### 📊 Pred Ct 분포")
        plot_pred_ct_hist(pred_df)

        # ---- figure 3) 품질지표(CV) vs Ct ----
        st.markdown("### 🧪 품질지표(CV) vs Ct")
        plot_cv_vs_ct(df_long, pred_df, cutoff=int(cutoff))

        # ---- 표는 옵션 ----
        if show_tables:
            st.markdown("### 📋 예측 결과 테이블")
            st.dataframe(pred_df, use_container_width=True)

        # ✅ 6) truth 있으면 즉석 평가 (중요: df_truth 넘기기!)
        try_eval_if_truth_exists(df_curve, pred_df, truth_df=df_truth)

        st.divider()
        
        # =========================
        # (A) Pred vs True + error color (Upload)
        # =========================
        st.markdown("### 🎯 Pred vs True (업로드 데이터, 오차 색상 표시)")
        
        if df_truth is not None:
            # eval_df 만들기: try_eval_if_truth_exists와 동일한 방식(Well normalize)
            true_col = None
            for cand in ["true_ct", "TrueCt", "trueCt", "ct", "Ct", "CT", "Cq", "cq", "CQ"]:
                if cand in df_truth.columns:
                    true_col = cand
                    break
        
            wcol = None
            for cand in ["Well", "well", "WELL"]:
                if cand in df_truth.columns:
                    wcol = cand
                    break
        
            if true_col and wcol:
                truth2 = df_truth[[wcol, true_col]].copy()
                truth2.columns = ["Well", "true_ct"]
                truth2["Well"] = truth2["Well"].map(normalize_well)
        
                tmp_eval = pred_df.copy()
                tmp_eval["Well"] = tmp_eval["Well"].map(normalize_well)
                tmp_eval = tmp_eval.merge(truth2, on="Well", how="left")
        
                tmp_eval["true_ct"] = pd.to_numeric(tmp_eval["true_ct"], errors="coerce")
                tmp_eval["pred_ct"] = pd.to_numeric(tmp_eval["pred_ct"], errors="coerce")
                tmp_eval = tmp_eval.dropna(subset=["true_ct", "pred_ct"]).copy()
        
                if not tmp_eval.empty:
                    tmp_eval["err"] = tmp_eval["pred_ct"] - tmp_eval["true_ct"]
                    tmp_eval["abs_err"] = tmp_eval["err"].abs()
        
                    import altair as alt
                    line_df = _line_y_eq_x(tmp_eval.rename(columns={"true_ct": "true_ct", "pred_ct": "pred_ct"}))
        
                    base = alt.Chart(tmp_eval).mark_circle(size=70, opacity=0.85).encode(
                        x=alt.X("true_ct:Q", title="True Ct/Cq"),
                        y=alt.Y("pred_ct:Q", title="Pred Ct/Cq"),
                        color=alt.Color("abs_err:Q", title="|Error|"),
                        tooltip=["Well", "true_ct", "pred_ct", "err", "abs_err"]
                    ).interactive()
        
                    diag = alt.Chart(line_df.rename(columns={"x": "true_ct", "y": "pred_ct"})).mark_line().encode(
                        x="true_ct:Q", y="pred_ct:Q"
                    )
        
                    st.altair_chart((diag + base).properties(height=360), use_container_width=True)
                    st.caption("✅ 점이 대각선(y=x)에 가까울수록 예측이 잘 맞는 거야. 색이 진할수록(|Error| 큼) 더 많이 틀린 샘플.")
                else:
                    st.info("truth는 있는데 pred와 매칭된 행이 없어(Well 매칭 확인 필요).")
            else:
                st.info("truth 시트에서 Well/true_ct(Ct/Cq) 컬럼을 못 찾아서 Pred vs True 차트는 생략했어.")
        else:
            st.info("truth 시트가 없어서 Pred vs True(정답 기반) 차트는 생략했어.")
        
        # =========================
        # (B) Upload Hard-like Review (Top-K |error|)
        # =========================
        st.markdown("### 🧨 Upload Hard Review (Top-K |error|)")
        
        if df_truth is not None and 'tmp_eval' in locals() and (tmp_eval is not None) and (not tmp_eval.empty):
            topk_u = st.slider("Hard Top-K (업로드)", 5, 50, 15, 5, key="pred_upload_hard_topk")
        
            hard_u = tmp_eval.sort_values("abs_err", ascending=False).head(int(topk_u)).reset_index(drop=True)
        
            c1, c2, c3 = st.columns(3)
            c1.metric("n (eval)", int(len(tmp_eval)))
            c2.metric("Hard Top-K", int(len(hard_u)))
            c3.metric("Worst |error|", f"{float(hard_u['abs_err'].iloc[0]):.3f}")
        
            with st.expander("📋 Hard 후보 표(업로드)", expanded=False):
                st.dataframe(hard_u[["Well", "true_ct", "pred_ct", "err", "abs_err"]], use_container_width=True, height=280)
        
            # 선택해서 곡선 보기
            def _fmt_u(i: int) -> str:
                r = hard_u.iloc[i]
                return f"{r['Well']} | |err|={r['abs_err']:.3f} (err={r['err']:+.3f})"
        
            pick_u = st.selectbox(
                "검토할 Hard 샘플 선택(업로드)",
                options=list(range(len(hard_u))),
                format_func=_fmt_u,
                key="pred_upload_hard_pick",
            )
        
            well_pick = str(hard_u.loc[int(pick_u), "Well"])
        
            st.markdown("#### 📈 선택 Hard 샘플의 원본 곡선(업로드 df_long)")
            curve_sel = df_long[df_long["Well"].astype(str) == str(well_pick)].copy()
            if curve_sel.empty:
                st.info("선택 well의 curve를 df_long에서 못 찾았어.")
            else:
                import altair as alt
                curve_sel = curve_sel.sort_values("Cycle").reset_index(drop=True)
                cutoff_i = int(cutoff)
                curve_sel["segment"] = np.where(curve_sel["Cycle"] <= cutoff_i, "early(<=cutoff)", "late")
        
                line = alt.Chart(curve_sel).mark_line().encode(
                    x=alt.X("Cycle:Q", title="Cycle"),
                    y=alt.Y("Fluor:Q", title="Fluor"),
                    tooltip=["Cycle", "Fluor", "segment"]
                )
                vline = alt.Chart(pd.DataFrame({"Cycle": [cutoff_i]})).mark_rule(strokeDash=[6,4]).encode(x="Cycle:Q")
                st.altair_chart(line + vline, use_container_width=True)
        
                st.markdown("#### 🔍 Early 확대(<=cutoff)")
                early = curve_sel[curve_sel["Cycle"] <= cutoff_i].copy()
                if len(early) >= 2:
                    eline = alt.Chart(early).mark_line().encode(
                        x=alt.X("Cycle:Q", title="Cycle (early)"),
                        y=alt.Y("Fluor:Q", title="Fluor"),
                        tooltip=["Cycle", "Fluor"]
                    )
                    st.altair_chart(eline, use_container_width=True)
        else:
            st.info("truth 기반 평가가 없어서(또는 매칭 실패) 업로드 Hard Review는 생략했어.")

        # =========================
        # (C) Pred stability across cutoffs (연결선)
        #  - Multi-cutoff Sweep을 켰을 때만 의미 있음
        # =========================
        st.markdown("### 🔗 Cutoff에 따른 예측 변화(선 연결)")
        
        if 'preds_all' in locals() and isinstance(preds_all, pd.DataFrame) and (not preds_all.empty):
            # 한 번에 너무 많으면 보기 힘들어서, 보여줄 well 수 제한
            max_w = st.slider("표시할 Well 개수(상위)", 5, 40, 15, 5, key="pred_stab_maxw")
        
            show_wells = preds_all["Well"].astype(str).unique().tolist()[:int(max_w)]
            stab = preds_all[preds_all["Well"].astype(str).isin(show_wells)].copy()
        
            # truth가 있으면 같이 그리기(가능하면)
            if df_truth is not None:
                true_col = None
                for cand in ["true_ct", "ct", "Ct", "Cq", "cq", "CQ"]:
                    if cand in df_truth.columns:
                        true_col = cand; break
                wcol = None
                for cand in ["Well", "well", "WELL"]:
                    if cand in df_truth.columns:
                        wcol = cand; break
        
                if true_col and wcol:
                    truth2 = df_truth[[wcol, true_col]].copy()
                    truth2.columns = ["Well", "true_ct"]
                    truth2["Well"] = truth2["Well"].map(normalize_well)
        
                    stab["Well"] = stab["Well"].map(normalize_well)
                    stab = stab.merge(truth2, on="Well", how="left")
        
            import altair as alt
            base = alt.Chart(stab.dropna(subset=["cutoff", "pred_ct"])).encode(
                x=alt.X("cutoff:Q", title="Cutoff"),
                y=alt.Y("pred_ct:Q", title="Pred Ct"),
                color=alt.Color("Well:N", legend=None),
                tooltip=["Well", "cutoff", "pred_ct"] + (["true_ct"] if "true_ct" in stab.columns else [])
            )
        
            lines = base.mark_line(point=True).properties(height=320)
        
            if "true_ct" in stab.columns and stab["true_ct"].notna().any():
                # True는 cutoff마다 변하지 않으니까 점선으로 같이 보여주면 “정답 대비 흔들림”이 바로 보임
                true_line = alt.Chart(stab.dropna(subset=["cutoff","true_ct"])).mark_line(strokeDash=[6,4]).encode(
                    x="cutoff:Q",
                    y="true_ct:Q",
                    detail="Well:N",
                    color=alt.value("gray"),
                )
                st.altair_chart((lines + true_line).interactive(), use_container_width=True)
                st.caption("실선=Pred 변화, 점선=True(고정). Pred가 cutoff에 따라 안정적이면 실선이 점선 주변에서 크게 흔들리지 않아.")
            else:
                st.altair_chart(lines.interactive(), use_container_width=True)
                st.caption("truth가 없어서 True 기준선은 생략했어. 그래도 cutoff에 따른 예측 안정성은 확인 가능.")
        else:
            st.info("Multi-cutoff Sweep을 켜면(그리고 preds_all 생성되면) cutoff별 예측 연결선이 생겨.")

                    
        # =========================
        # (추가) 그림 중심 4탭 UI (원하면 유지)
        # =========================
        import altair as alt

        t1, t2, t3, t4 = st.tabs(["📊 Ct Overview", "📈 Well별 Ct", "🧬 Curve 보기", "🧾 Data(표)"])

        with t1:
            hist = (
                alt.Chart(pred_df)
                .mark_bar()
                .encode(
                    x=alt.X("pred_ct:Q", bin=alt.Bin(maxbins=30), title="Predicted Ct"),
                    y=alt.Y("count():Q", title="Count"),
                    tooltip=[alt.Tooltip("count():Q", title="count")],
                )
                .properties(height=280)
            )
            st.altair_chart(hist, use_container_width=True)

        with t2:
            bar = (
                alt.Chart(pred_df)
                .mark_bar()
                .encode(
                    x=alt.X("Well:N", sort=alt.SortField("pred_ct", order="ascending"), title="Well"),
                    y=alt.Y("pred_ct:Q", title="Predicted Ct"),
                    tooltip=["Well", "pred_ct"],
                )
                .properties(height=320)
            )
            st.altair_chart(bar, use_container_width=True)

        with t3:
            wells = pred_df["Well"].astype(str).tolist()
            pick_well = st.selectbox("곡선 볼 Well 선택", wells, index=0, key="pred_pick_well")

            curve_sel = df_long[df_long["Well"].astype(str) == str(pick_well)].copy()
            if curve_sel.empty:
                st.info("선택한 Well의 curve 데이터를 찾지 못했어.")
            else:
                curve_sel = curve_sel.sort_values("Cycle").copy()
                cutoff_i = int(cutoff)

                line = (
                    alt.Chart(curve_sel)
                    .mark_line()
                    .encode(
                        x=alt.X("Cycle:Q", title="Cycle"),
                        y=alt.Y("Fluor:Q", title="Fluor"),
                        tooltip=["Cycle", "Fluor"],
                    )
                )
                vline = (
                    alt.Chart(pd.DataFrame({"Cycle": [cutoff_i]}))
                    .mark_rule(strokeDash=[6, 4])
                    .encode(x="Cycle:Q")
                )
                st.altair_chart(line + vline, use_container_width=True)

                ct_val = float(pred_df.loc[pred_df["Well"].astype(str) == str(pick_well), "pred_ct"].iloc[0])
                st.caption(f"✅ Well={pick_well} / Pred Ct={ct_val:.3f} (cutoff={cutoff_i})")

        with t4:
            with st.expander("표로 보기 (원본/예측 결과)", expanded=False):
                if isinstance(raw_obj, dict):
                    st.caption("curve 시트(상단 일부)")
                    st.dataframe(df_curve.head(30), use_container_width=True)
                    if df_truth is not None:
                        st.caption("truth 시트(상단 일부)")
                        st.dataframe(df_truth.head(30), use_container_width=True)
                else:
                    st.caption("업로드 원본(상단 일부)")
                    st.dataframe(df_curve.head(30), use_container_width=True)

                st.caption("예측 결과")
                st.dataframe(pred_df, use_container_width=True)


# -------------------------
# Tab 2: Hard Review
# -------------------------
with tabs[3]:
    show_hard_review_with_buckets()

# -------------------------
# Tab 3: Retrain (Admin)
# -------------------------
with tabs[4]:
    st.subheader("2) 누적 반영 후 재학습 (관리자 버튼)")

    if running_on_streamlit_cloud():
        st.warning(
            "Streamlit Cloud에는 canonical 데이터(master_long.parquet)가 없어서 재학습을 실행할 수 없어요.\n"
            "서버/로컬에서 학습 후 reports/ 결과물만 배포하세요."
        )
        st.stop()  # ✅ 여기서 탭 실행을 끝내버리면 step3가 절대 호출되지 않음

    st.info(
        "이 버튼은 **현재 서버에 저장된 canonical 데이터(master_long.parquet)** 기준으로 "
        "모델을 다시 학습하고 data/models/by_cutoff에 덮어씁니다.\n\n"
        "⚠️ 데이터 ingest(= raw -> canonical)는 이 버튼에 포함되어 있지 않아요. "
        "새 raw 데이터를 canonical로 반영하려면 ingest 파이프라인으로 먼저 master_long을 업데이트해줘야 해요."
    )

    meta = load_meta(int(cutoff))
    if meta:
        with st.expander("선택된 모델 메타 보기"):
            st.json(meta)

    can_retrain = has_canonical_master_long() and (not running_on_streamlit_cloud())

    if not can_retrain:
        st.warning(
            "Streamlit Cloud에는 학습 데이터(master_long.parquet)가 없어서 재학습을 실행할 수 없어요.\n"
            "로컬/서버에서 학습을 돌린 뒤, reports/ 결과물만 repo에 커밋해서 배포하는 방식으로 운영하세요."
        )
    
    if st.button("재학습 실행", type="secondary", key="btn_retrain", disabled=not can_retrain):
        with st.spinner("재학습 중... (로그 생성 중)"):
            code, log = run_retrain(int(min_c), int(max_c))
    
        st.text_area("학습 로그", log, height=380)
    
        if code == 0:
            st.success("재학습 완료! 모델 파일이 갱신됐어요.")
            show_train_report()
        else:
            st.error(f"재학습 실패 (return code={code}) - 로그를 확인해줘.")
            
try:
    st.caption("VERSION: " + (PROJECT_ROOT / "VERSION.txt").read_text().strip())
except Exception:
    st.caption("VERSION: (missing)")

