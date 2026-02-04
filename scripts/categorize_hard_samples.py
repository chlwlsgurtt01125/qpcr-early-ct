#!/usr/bin/env python3
"""
Hard Sample 카테고리 분류 및 분석
4가지 버킷으로 Hard Sample을 분류하고 각 원인 분석

사용법:
    python scripts/categorize_hard_samples.py --cutoff 24
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_ROOT = PROJECT_ROOT / "reports"
QC_DIR = PROJECT_ROOT / "outputs" / "qc"


def get_active_model_id() -> str:
    p = REPORTS_ROOT / "active_model.txt"
    return p.read_text().strip() if p.exists() else "model_server_latest_xgb"


def categorize_hard_sample(row) -> str:
    """
    Hard Sample을 4가지 카테고리로 분류:
    
    1. LATE_AMP: Ct > 35 (증폭이 너무 늦어서 예측 어려움)
    2. NOISY: SNR < 3.0 또는 R² < 0.95 (신호 품질 문제)
    3. NON_SIGMOID: fail_reason == "NON_SIGMOID" (비정상 곡선 형태)
    4. OUTLIER: 나머지 (모델의 한계 또는 알 수 없는 원인)
    """
    # LATE_AMP 체크
    if pd.notna(row.get('true_ct')) and row['true_ct'] > 35:
        return "LATE_AMP"
    
    # NOISY 체크
    if (pd.notna(row.get('snr')) and row['snr'] < 3.0) or \
       (pd.notna(row.get('r2')) and row['r2'] < 0.95):
        return "NOISY"
    
    # NON_SIGMOID 체크
    if pd.notna(row.get('fail_reason')) and 'SIGMOID' in str(row['fail_reason']).upper():
        return "NON_SIGMOID"
    
    return "OUTLIER"


def analyze_hard_samples(cutoff: int = 24, topk: int = 50):
    """
    Hard Sample 분석 및 카테고리별 통계 생성
    """
    print(f"\n?? Hard Sample 분석 시작 (cutoff={cutoff}, top-K={topk})")
    
    # 1. predictions_long 로드
    model_id = get_active_model_id()
    pred_path = REPORTS_ROOT / model_id / "predictions_long.parquet"
    
    if not pred_path.exists():
        print(f"? predictions_long.parquet not found: {pred_path}")
        return
    
    pred = pd.read_parquet(pred_path)
    pred["abs_err"] = abs(pred["pred_ct"] - pred["true_ct"])
    
    # 2. QC catalog 로드 (r2, snr, fail_reason 정보)
    qc_path = QC_DIR / "master_catalog.parquet"
    
    if qc_path.exists():
        qc = pd.read_parquet(qc_path)
        
        # well_uid 기반 병합
        pred = pred.merge(
            qc[["well_uid", "r2", "snr", "fail_reason", "qc_status"]],
            on="well_uid",
            how="left"
        )
        print(f"? QC 데이터 병합 완료 ({len(qc)} wells)")
    else:
        print(f"?? QC catalog not found: {qc_path}")
        pred["r2"] = np.nan
        pred["snr"] = np.nan
        pred["fail_reason"] = None
        pred["qc_status"] = None
    
    # 3. 해당 cutoff의 Hard Sample 추출
    df = pred[pred["cutoff"] == cutoff].copy()
    hard = df.sort_values("abs_err", ascending=False).head(topk).copy()
    
    print(f"?? Hard Sample 추출: {len(hard)} / {len(df)} wells")
    
    # 4. 카테고리 분류
    hard["category"] = hard.apply(categorize_hard_sample, axis=1)
    
    # 5. 카테고리별 통계
    category_stats = hard.groupby("category").agg({
        "abs_err": ["mean", "median", "std", "min", "max"],
        "well_uid": "count"
    }).round(3)
    
    category_stats.columns = ["_".join(col).strip() for col in category_stats.columns]
    category_stats = category_stats.rename(columns={"well_uid_count": "count"})
    category_stats = category_stats.reset_index()
    
    print("\n?? 카테고리별 통계:")
    print(category_stats.to_string(index=False))
    
    # 6. 각 카테고리별 원인 분석
    print("\n?? 카테고리별 원인 분석:")
    
    for cat in ["LATE_AMP", "NOISY", "NON_SIGMOID", "OUTLIER"]:
        cat_samples = hard[hard["category"] == cat]
        
        if len(cat_samples) == 0:
            continue
        
        print(f"\n  [{cat}] {len(cat_samples)} samples")
        print(f"    - 평균 |error|: {cat_samples['abs_err'].mean():.3f}")
        print(f"    - 평균 true_ct: {cat_samples['true_ct'].mean():.2f}")
        
        if "r2" in cat_samples.columns:
            print(f"    - 평균 R²: {cat_samples['r2'].mean():.4f}")
        if "snr" in cat_samples.columns:
            print(f"    - 평균 SNR: {cat_samples['snr'].mean():.2f}")
        
        # 대표 샘플 (가장 오차 큰 것)
        worst = cat_samples.iloc[0]
        print(f"    - 대표 샘플: {worst['run_id']}:{worst['well_id']}")
        print(f"      true={worst['true_ct']:.2f}, pred={worst['pred_ct']:.2f}, err={worst['abs_err']:.3f}")
    
    # 7. 결과 저장
    output_dir = PROJECT_ROOT / "outputs" / "hard_sample_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hard.to_csv(output_dir / f"hard_samples_cutoff_{cutoff}.csv", index=False)
    category_stats.to_csv(output_dir / f"category_stats_cutoff_{cutoff}.csv", index=False)
    
    print(f"\n? 결과 저장:")
    print(f"   - {output_dir / f'hard_samples_cutoff_{cutoff}.csv'}")
    print(f"   - {output_dir / f'category_stats_cutoff_{cutoff}.csv'}")
    
    # 8. 시각화 (선택)
    try:
        # 카테고리 분포
        fig = px.pie(
            hard,
            names="category",
            title=f"Hard Sample Category Distribution (cutoff={cutoff}, top-{topk})",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.write_html(output_dir / f"category_distribution_cutoff_{cutoff}.html")
        print(f"   - {output_dir / f'category_distribution_cutoff_{cutoff}.html'}")
        
        # 카테고리별 오차 분포
        fig2 = px.box(
            hard,
            x="category",
            y="abs_err",
            title=f"Error Distribution by Category (cutoff={cutoff})",
            color="category"
        )
        fig2.write_html(output_dir / f"error_by_category_cutoff_{cutoff}.html")
        print(f"   - {output_dir / f'error_by_category_cutoff_{cutoff}.html'}")
        
    except Exception as e:
        print(f"?? 시각화 생성 실패: {e}")
    
    return hard, category_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int, default=24, help="Cutoff to analyze")
    parser.add_argument("--topk", type=int, default=50, help="Number of hard samples to analyze")
    args = parser.parse_args()
    
    analyze_hard_samples(cutoff=args.cutoff, topk=args.topk)