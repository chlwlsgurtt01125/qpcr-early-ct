"""
qPCR Data Quality Control Analyzer
Well 단위 QC 상태, Ct 구간 분류, 사용 가능성 판정
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import zscore
from typing import Dict, Tuple, List, Optional


class QPCRQualityControl:
    """qPCR 데이터 품질 관리 및 분류 시스템"""
    
    # Ct 구간 정의
    CT_BINS = {
        'BIN_A': (0, 3),      # EXTREME_LOW
        'BIN_B': (4, 5),      # VERY_LOW
        'BIN_C': (6, 24),     # NORMAL
        'BIN_D': (25, 29),    # MODERATE_LATE
        'BIN_E': (30, 35),    # LATE
        'BIN_F': (36, 38),    # ULTRA_LATE
        'BIN_G': (39, 40),    # NO_AMP
    }
    
    # QC 임계값
    THRESHOLDS = {
        'r2_min': 0.95,           # 시그모이드 적합도 최소값
        'snr_min': 3.0,           # Signal-to-Noise Ratio 최소값
        'baseline_std_max': 0.15, # 베이스라인 안정성
        'spike_threshold': 5.0,   # 스파이크 검출 (표준편차 배수)
        'min_amp_range': 0.3,     # 최소 증폭 범위
    }
    
    def __init__(self, cycle_columns: List[str] = None):
        """
        Args:
            cycle_columns: 형광 데이터 컬럼명 리스트 (예: ['cycle_1', 'cycle_2', ...])
                          None이면 자동으로 'cycle_1' ~ 'cycle_40' 사용
        """
        if cycle_columns is None:
            self.cycle_columns = [f'cycle_{i}' for i in range(1, 41)]
        else:
            self.cycle_columns = cycle_columns
    
    @staticmethod
    def sigmoid(x: np.ndarray, L: float, x0: float, k: float, b: float) -> np.ndarray:
        """4-parameter sigmoid function"""
        return L / (1 + np.exp(-k * (x - x0))) + b
    
    def analyze_curve_shape(self, fluorescence: np.ndarray) -> Dict[str, float]:
        """
        형광 곡선의 형태 분석
        
        Returns:
            dict: {
                'r2': 시그모이드 적합도 (R²),
                'snr': Signal-to-Noise Ratio,
                'baseline_std': 베이스라인 표준편차,
                'has_spike': 스파이크 존재 여부,
                'amp_range': 증폭 범위
            }
        """
        cycles = np.arange(len(fluorescence))
        
        # 1. 시그모이드 적합도 (R²)
        r2 = self._fit_sigmoid_r2(cycles, fluorescence)
        
        # 2. SNR 계산
        snr = self._calculate_snr(fluorescence)
        
        # 3. 베이스라인 안정성 (처음 10 사이클)
        baseline = fluorescence[:10]
        baseline_std = np.std(baseline) if len(baseline) > 0 else np.inf
        
        # 4. 스파이크 검출
        has_spike = self._detect_spike(fluorescence)
        
        # 5. 증폭 범위
        amp_range = np.max(fluorescence) - np.min(fluorescence)
        
        return {
            'r2': r2,
            'snr': snr,
            'baseline_std': baseline_std,
            'has_spike': has_spike,
            'amp_range': amp_range
        }
    
    def _fit_sigmoid_r2(self, x: np.ndarray, y: np.ndarray) -> float:
        """시그모이드 함수 적합 후 R² 계산"""
        try:
            # 초기 파라미터 추정
            L = np.max(y) - np.min(y)
            x0 = x[len(x) // 2]
            k = 0.3
            b = np.min(y)
            
            # Curve fitting
            popt, _ = curve_fit(
                self.sigmoid, x, y,
                p0=[L, x0, k, b],
                maxfev=5000,
                bounds=([0, 0, 0, -np.inf], [np.inf, 40, 5, np.inf])
            )
            
            # R² 계산
            y_pred = self.sigmoid(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return max(0, min(1, r2))  # 0~1 사이로 제한
            
        except Exception:
            return 0.0
    
    def _calculate_snr(self, fluorescence: np.ndarray) -> float:
        """Signal-to-Noise Ratio 계산"""
        baseline = fluorescence[:10]
        signal_max = np.max(fluorescence)
        
        noise = np.std(baseline) if len(baseline) > 0 else 1e-6
        signal = signal_max - np.mean(baseline)
        
        return signal / noise if noise > 1e-6 else 0
    
    def _detect_spike(self, fluorescence: np.ndarray) -> bool:
        """급격한 스파이크 검출"""
        if len(fluorescence) < 3:
            return False
        
        # 1차 미분
        diff = np.diff(fluorescence)
        
        # Z-score 기반 이상치 검출
        z_scores = np.abs(zscore(diff, nan_policy='omit'))
        
        return np.any(z_scores > self.THRESHOLDS['spike_threshold'])
    
    def classify_qc_status(
        self, 
        fluorescence: np.ndarray, 
        ct_value: Optional[float],
        sample_type: str = 'unknown'
    ) -> Tuple[str, str, Dict]:
        """
        QC 상태 분류 (PASS / FAIL / FLAG)
        
        Args:
            fluorescence: 형광 곡선 (40 cycles)
            ct_value: Ct 값 (None이면 NO_SIGNAL로 처리)
            sample_type: 'PC' (positive control), 'NC' (negative control), 'sample'
        
        Returns:
            (qc_status, fail_reason, metrics)
        """
        # 형태 분석
        metrics = self.analyze_curve_shape(fluorescence)
        
        # Ct 값 체크
        if ct_value is None or np.isnan(ct_value):
            return 'FAIL', 'NO_SIGNAL', metrics
        
        # 1. 형태 기반 FAIL 조건
        if metrics['r2'] < self.THRESHOLDS['r2_min']:
            if metrics['snr'] < self.THRESHOLDS['snr_min']:
                return 'FAIL', 'NOISY_BASELINE', metrics
            else:
                return 'FAIL', 'NON_SIGMOID', metrics
        
        if metrics['has_spike']:
            return 'FAIL', 'SPIKE_ARTIFACT', metrics
        
        if metrics['amp_range'] < self.THRESHOLDS['min_amp_range']:
            return 'FAIL', 'NO_AMPLIFICATION', metrics
        
        # 2. Ct 값 기반 분류
        if ct_value <= 3:
            return 'FAIL', 'CT_EXTREME_LOW', metrics
        
        elif 4 <= ct_value <= 5:
            return 'FLAG', 'CT_VERY_LOW', metrics
        
        elif 30 <= ct_value <= 35:
            if sample_type == 'NC':
                return 'FAIL', 'NC_LATE_SIGNAL', metrics
            else:
                return 'FLAG', 'CT_LATE', metrics
        
        elif 36 <= ct_value <= 40:
            if sample_type == 'NC':
                return 'FAIL', 'NC_ULTRA_LATE_SIGNAL', metrics
            else:
                return 'FLAG', 'CT_ULTRA_LATE', metrics
        
        # 3. PASS
        return 'PASS', 'OK', metrics
    
    def assign_ct_bin(self, ct_value: float) -> str:
        """Ct 값을 구간(BIN)으로 분류"""
        if pd.isna(ct_value):
            return 'BIN_UNKNOWN'
        
        for bin_name, (lower, upper) in self.CT_BINS.items():
            if lower <= ct_value <= upper:
                return bin_name
        
        return 'BIN_OUT_OF_RANGE'
    
    def create_catalog(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        전체 데이터프레임에 대해 QC 카탈로그 생성
        
        Args:
            df: 원본 데이터프레임 (cycle_1 ~ cycle_40 컬럼 필수, ct_true 컬럼 필수)
        
        Returns:
            카탈로그 데이터프레임 (well_id, qc_status, fail_reason, ct_bin, usable, ...)
        """
        results = []
        
        for idx, row in df.iterrows():
            # 형광 곡선 추출
            try:
                fluorescence = row[self.cycle_columns].values.astype(float)
            except KeyError:
                # 컬럼명이 다를 경우 자동 탐색
                cycle_cols = [c for c in df.columns if 'cycle' in c.lower()]
                if len(cycle_cols) >= 40:
                    fluorescence = row[cycle_cols[:40]].values.astype(float)
                else:
                    # 형광 데이터 없음
                    results.append({
                        'row_index': idx,
                        'qc_status': 'FAIL',
                        'fail_reason': 'MISSING_FLUORESCENCE_DATA',
                        'ct_bin': 'UNKNOWN',
                        'usable': False,
                        'r2': np.nan,
                        'snr': np.nan,
                    })
                    continue
            
            # Ct 값
            ct_value = row.get('ct_true', row.get('Ct', np.nan))
            
            # 샘플 타입 추정 (있다면)
            sample_type = row.get('sample_type', 'unknown')
            
            # QC 분류
            qc_status, fail_reason, metrics = self.classify_qc_status(
                fluorescence, ct_value, sample_type
            )
            
            # Ct 구간
            ct_bin = self.assign_ct_bin(ct_value)
            
            # 사용 가능 여부
            usable = (qc_status == 'PASS')
            
            results.append({
                'row_index': idx,
                'well_id': row.get('well_id', row.get('Well', f'well_{idx}')),
                'ct_value': ct_value,
                'ct_bin': ct_bin,
                'qc_status': qc_status,
                'fail_reason': fail_reason,
                'usable': usable,
                'r2': metrics['r2'],
                'snr': metrics['snr'],
                'baseline_std': metrics['baseline_std'],
                'has_spike': metrics['has_spike'],
                'amp_range': metrics['amp_range'],
            })
        
        catalog = pd.DataFrame(results)
        return catalog
    
    def create_excluded_report(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """
        사용 불가 데이터(excluded)에 대한 상세 리포트
        
        Returns:
            excluded_report: [well_id, excluded_major_reason, excluded_minor_reason, evidence]
        """
        excluded = catalog[~catalog['usable']].copy()
        
        if len(excluded) == 0:
            return pd.DataFrame(columns=[
                'well_id', 'excluded_major_reason', 'excluded_minor_reason', 
                'evidence_r2', 'evidence_snr', 'evidence_ct'
            ])
        
        # Major/Minor reason 분류
        def categorize_reason(fail_reason):
            shape_issues = ['NON_SIGMOID', 'NOISY_BASELINE', 'SPIKE_ARTIFACT', 'NO_AMPLIFICATION']
            ct_issues = ['CT_EXTREME_LOW', 'CT_VERY_LOW', 'CT_LATE', 'CT_ULTRA_LATE']
            nc_issues = ['NC_LATE_SIGNAL', 'NC_ULTRA_LATE_SIGNAL']
            
            if fail_reason in shape_issues:
                return 'QC_FAIL_SHAPE', fail_reason
            elif fail_reason in ct_issues:
                return 'CT_EXTREME', fail_reason
            elif fail_reason in nc_issues:
                return 'CONTROL_WELL', fail_reason
            elif fail_reason == 'NO_SIGNAL':
                return 'QC_FAIL_NOISE', 'NO_SIGNAL'
            else:
                return 'UNKNOWN', fail_reason
        
        excluded['excluded_major_reason'] = excluded['fail_reason'].apply(
            lambda x: categorize_reason(x)[0]
        )
        excluded['excluded_minor_reason'] = excluded['fail_reason'].apply(
            lambda x: categorize_reason(x)[1]
        )
        
        # Evidence 정리
        report = excluded[[
            'well_id', 'excluded_major_reason', 'excluded_minor_reason',
            'r2', 'snr', 'ct_value', 'ct_bin', 'qc_status'
        ]].copy()
        
        report.rename(columns={
            'r2': 'evidence_r2',
            'snr': 'evidence_snr',
            'ct_value': 'evidence_ct'
        }, inplace=True)
        
        return report


def demo_usage():
    """사용 예시"""
    # 1. 가상 데이터 생성 (실제로는 엑셀에서 로드)
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'well_id': [f'A{i:02d}' for i in range(1, n_samples+1)],
        'ct_true': np.random.uniform(5, 35, n_samples),
    }
    
    # 형광 곡선 시뮬레이션
    for i in range(1, 41):
        data[f'cycle_{i}'] = np.random.uniform(0.1, 3.0, n_samples)
    
    df = pd.DataFrame(data)
    
    # 2. QC 분석
    qc = QPCRQualityControl()
    catalog = qc.create_catalog(df)
    
    print("=== Master Catalog ===")
    print(catalog.head())
    print(f"\nTotal: {len(catalog)}")
    print(f"PASS: {(catalog['qc_status'] == 'PASS').sum()}")
    print(f"FAIL: {(catalog['qc_status'] == 'FAIL').sum()}")
    print(f"FLAG: {(catalog['qc_status'] == 'FLAG').sum()}")
    
    # 3. Excluded Report
    excluded_report = qc.create_excluded_report(catalog)
    print("\n=== Excluded Report ===")
    print(excluded_report)
    
    return catalog, excluded_report


if __name__ == "__main__":
    catalog, excluded_report = demo_usage()
