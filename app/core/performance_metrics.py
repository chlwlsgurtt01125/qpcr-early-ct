"""
qPCR Performance Metrics with Error Tolerance Thresholds
0.5 cycle, 1.0 cycle 기준 KPI 계산
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import plotly.graph_objects as go
import plotly.express as px


class PerformanceMetrics:
    """
    qPCR Early-Ct 예측 성능 평가 지표
    - 기존 MAE/RMSE/R2
    - 오차 허용 범위 기준 (0.5, 1.0 cycle)
    - Fold-change 관점 지표
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Args:
            y_true: 실제 Ct 값
            y_pred: 예측 Ct 값
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.errors = y_pred - y_true
        self.abs_errors = np.abs(self.errors)
    
    def calculate_basic_metrics(self) -> Dict[str, float]:
        """기본 회귀 지표"""
        from sklearn.metrics import (
            mean_absolute_error, 
            mean_squared_error, 
            r2_score
        )
        
        mae = mean_absolute_error(self.y_true, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
        r2 = r2_score(self.y_true, self.y_pred)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        }
    
    def calculate_tolerance_metrics(self) -> Dict[str, float]:
        """
        오차 허용 범위 기준 정확도
        
        Returns:
            dict: {
                'P(|error| ≤ 0.5)': 0.5 cycle 이내 비율,
                'P(|error| ≤ 1.0)': 1.0 cycle 이내 비율,
                'P(|error| ≤ 2.0)': 2.0 cycle 이내 비율,
                ...
            }
        """
        thresholds = [0.3, 0.5, 1.0, 2.0, 3.0]
        
        tolerance_metrics = {}
        
        for threshold in thresholds:
            within_threshold = (self.abs_errors <= threshold).sum()
            ratio = within_threshold / len(self.abs_errors)
            tolerance_metrics[f'P(|error| ≤ {threshold})'] = ratio
        
        return tolerance_metrics
    
    def calculate_fold_change_metrics(self) -> Dict[str, float]:
        """
        Fold-change (증폭 배수) 관점의 지표
        
        qPCR에서 Ct 차이 1 cycle = 2배 차이
        ΔCt = log2(fold_change)
        
        Returns:
            dict: {
                'Mean Fold Error': 평균 배수 오차,
                'Median Fold Error': 중앙값 배수 오차,
                'P(Fold Error < 1.5x)': 1.5배 이내 비율,
                ...
            }
        """
        # Ct 차이를 fold-change로 변환
        fold_errors = 2 ** self.abs_errors
        
        metrics = {
            'Mean Fold Error': np.mean(fold_errors),
            'Median Fold Error': np.median(fold_errors),
            'P(Fold Error < 1.5x)': (fold_errors < 1.5).sum() / len(fold_errors),
            'P(Fold Error < 2.0x)': (fold_errors < 2.0).sum() / len(fold_errors),
            'P(Fold Error < 3.0x)': (fold_errors < 3.0).sum() / len(fold_errors),
        }
        
        return metrics
    
    def calculate_ct_range_performance(self, bins: list = None) -> pd.DataFrame:
        """
        Ct 구간별 성능 분석
        
        Args:
            bins: Ct 구간 경계 (기본값: [0, 10, 15, 20, 25, 30, 35, 40])
        
        Returns:
            DataFrame: 구간별 MAE, RMSE, P(≤0.5), P(≤1.0)
        """
        if bins is None:
            bins = [0, 10, 15, 20, 25, 30, 35, 40]
        
        ct_ranges = pd.cut(self.y_true, bins=bins, include_lowest=True)
        
        results = []
        
        for ct_range in ct_ranges.cat.categories:
            mask = ct_ranges == ct_range
            
            if mask.sum() == 0:
                continue
            
            range_errors = self.abs_errors[mask]
            range_true = self.y_true[mask]
            range_pred = self.y_pred[mask]
            
            mae = np.mean(range_errors)
            rmse = np.sqrt(np.mean((range_pred - range_true) ** 2))
            p_05 = (range_errors <= 0.5).sum() / len(range_errors)
            p_10 = (range_errors <= 1.0).sum() / len(range_errors)
            
            results.append({
                'Ct Range': str(ct_range),
                'Count': mask.sum(),
                'MAE': mae,
                'RMSE': rmse,
                'P(≤0.5)': p_05,
                'P(≤1.0)': p_10
            })
        
        return pd.DataFrame(results)
    
    def get_all_metrics(self) -> Dict[str, any]:
        """모든 지표를 하나의 딕셔너리로 반환"""
        metrics = {}
        
        # 1. 기본 지표
        metrics.update(self.calculate_basic_metrics())
        
        # 2. 허용 범위 지표
        metrics.update(self.calculate_tolerance_metrics())
        
        # 3. Fold-change 지표
        metrics.update(self.calculate_fold_change_metrics())
        
        return metrics
    
    def plot_error_distribution(self) -> go.Figure:
        """오차 분포 시각화 (히스토그램 + 허용 범위 표시)"""
        fig = go.Figure()
        
        # 히스토그램
        fig.add_trace(go.Histogram(
            x=self.errors,
            nbinsx=50,
            name='Error Distribution',
            marker_color='steelblue',
            opacity=0.7
        ))
        
        # 허용 범위 표시
        for threshold, color in [(0.5, 'green'), (1.0, 'orange')]:
            fig.add_vline(
                x=threshold, 
                line_dash="dash", 
                line_color=color,
                annotation_text=f"+{threshold}",
                annotation_position="top"
            )
            fig.add_vline(
                x=-threshold, 
                line_dash="dash", 
                line_color=color,
                annotation_text=f"-{threshold}",
                annotation_position="top"
            )
        
        fig.update_layout(
            title="Prediction Error Distribution with Tolerance Thresholds",
            xaxis_title="Error (Predicted - True)",
            yaxis_title="Count",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def plot_cumulative_error(self) -> go.Figure:
        """누적 오차 분포 (CDF)"""
        sorted_errors = np.sort(self.abs_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sorted_errors,
            y=cumulative * 100,
            mode='lines',
            name='Cumulative Distribution',
            line=dict(color='steelblue', width=2)
        ))
        
        # 허용 범위 표시
        for threshold, color in [(0.5, 'green'), (1.0, 'orange'), (2.0, 'red')]:
            pct = (self.abs_errors <= threshold).sum() / len(self.abs_errors) * 100
            
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color=color,
                annotation_text=f"{threshold} cycle ({pct:.1f}%)",
                annotation_position="top right"
            )
        
        fig.update_layout(
            title="Cumulative Error Distribution",
            xaxis_title="Absolute Error (cycles)",
            yaxis_title="Cumulative Percentage (%)",
            height=400
        )
        
        return fig
    
    def create_performance_summary_table(self) -> pd.DataFrame:
        """성능 요약 테이블"""
        all_metrics = self.get_all_metrics()
        
        summary = pd.DataFrame([
            {'Metric': 'Mean Absolute Error (MAE)', 'Value': f"{all_metrics['MAE']:.3f} cycles"},
            {'Metric': 'Root Mean Squared Error (RMSE)', 'Value': f"{all_metrics['RMSE']:.3f} cycles"},
            {'Metric': 'R² Score', 'Value': f"{all_metrics['R²']:.4f}"},
            {'Metric': '', 'Value': ''},  # 구분선
            {'Metric': 'P(|error| ≤ 0.5 cycle)', 'Value': f"{all_metrics['P(|error| ≤ 0.5)']:.1%}"},
            {'Metric': 'P(|error| ≤ 1.0 cycle)', 'Value': f"{all_metrics['P(|error| ≤ 1.0)']:.1%}"},
            {'Metric': 'P(|error| ≤ 2.0 cycle)', 'Value': f"{all_metrics['P(|error| ≤ 2.0)']:.1%}"},
            {'Metric': '', 'Value': ''},  # 구분선
            {'Metric': 'Mean Fold Error', 'Value': f"{all_metrics['Mean Fold Error']:.2f}x"},
            {'Metric': 'P(Fold Error < 1.5x)', 'Value': f"{all_metrics['P(Fold Error < 1.5x)']:.1%}"},
            {'Metric': 'P(Fold Error < 2.0x)', 'Value': f"{all_metrics['P(Fold Error < 2.0x)']:.1%}"},
        ])
        
        return summary


# ===== Streamlit Integration Example =====
def render_enhanced_performance_page(y_true, y_pred):
    """
    기존 Performance 페이지에 추가할 향상된 메트릭
    """
    import streamlit as st
    
    st.header("📊 Performance Metrics (Enhanced)")
    
    # 성능 계산
    perf = PerformanceMetrics(y_true, y_pred)
    
    # 1. 요약 테이블
    st.subheader("Performance Summary")
    summary_table = perf.create_performance_summary_table()
    st.dataframe(summary_table, use_container_width=True, hide_index=True)
    
    # 2. 시각화
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(perf.plot_error_distribution(), use_container_width=True)
    
    with col2:
        st.plotly_chart(perf.plot_cumulative_error(), use_container_width=True)
    
    # 3. Ct 구간별 성능
    st.subheader("Performance by Ct Range")
    ct_range_perf = perf.calculate_ct_range_performance()
    
    st.dataframe(
        ct_range_perf.style.format({
            'MAE': '{:.3f}',
            'RMSE': '{:.3f}',
            'P(≤0.5)': '{:.1%}',
            'P(≤1.0)': '{:.1%}'
        }).background_gradient(subset=['MAE', 'RMSE'], cmap='RdYlGn_r'),
        use_container_width=True,
        hide_index=True
    )


if __name__ == "__main__":
    # 테스트
    np.random.seed(42)
    y_true = np.random.uniform(10, 35, 1000)
    y_pred = y_true + np.random.normal(0, 0.8, 1000)
    
    perf = PerformanceMetrics(y_true, y_pred)
    
    print("=== All Metrics ===")
    for k, v in perf.get_all_metrics().items():
        print(f"{k}: {v:.4f}")
    
    print("\n=== Ct Range Performance ===")
    print(perf.calculate_ct_range_performance())
