#!/usr/bin/env python3
"""
QC 분석 결과를 저장하는 스크립트 (well_uid 기반)
outputs/qc/ 폴더에 master_catalog.parquet와 excluded_report.parquet 생성

사용법:
    python scripts/save_qc_results.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.qc_analyzer import QPCRQualityControl


def create_qc_catalog_from_long(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    master_long.parquet에서 직접 QC 카탈로그 생성
    well_uid(run_id__Well) 단위로 그룹핑하여 QC 분석
    
    Args:
        df_long: master_long.parquet (Cycle, Fluor, Well, run_id, channel, Cq, well_uid)
    
    Returns:
        QC catalog with well_uid as key
    """
    print("   Detecting unique wells...")
    
    # well_uid가 있으면 사용, 없으면 run_id + Well로 생성
    if 'well_uid' not in df_long.columns:
        print("   Creating well_uid from run_id + Well...")
        df_long['well_uid'] = df_long['run_id'] + '__' + df_long['Well']
    
    # well_uid별로 그룹핑
    wells = df_long.groupby('well_uid').first().reset_index()
    print(f"   Found {len(wells)} unique wells")
    
    # QC Analyzer 초기화
    qc = QPCRQualityControl()
    
    results = []
    
    print("   Running QC analysis per well...")
    for idx, well_uid in enumerate(wells['well_uid']):
        if idx % 100 == 0 and idx > 0:
            print(f"   Progress: {idx}/{len(wells)} wells...")
        
        # 해당 well의 모든 cycle 데이터
        well_data = df_long[df_long['well_uid'] == well_uid].sort_values('Cycle')
        
        if len(well_data) == 0:
            continue
        
        # 형광 곡선 (40 cycles)
        fluorescence = well_data['Fluor'].values
        
        # Cycle이 40개 미만이면 패딩 또는 스킵
        if len(fluorescence) < 40:
            # NaN으로 패딩
            fluorescence = np.pad(
                fluorescence, 
                (0, 40 - len(fluorescence)), 
                constant_values=np.nan
            )
        elif len(fluorescence) > 40:
            fluorescence = fluorescence[:40]
        
        # Ct 값
        ct_value = well_data['Cq'].iloc[0] if 'Cq' in well_data.columns else np.nan
        
        # 샘플 타입 (있다면)
        sample_type = well_data['sample_type'].iloc[0] if 'sample_type' in well_data.columns else 'unknown'
        
        # QC 분류
        qc_status, fail_reason, metrics = qc.classify_qc_status(
            fluorescence, ct_value, sample_type
        )
        
        # Ct 구간
        ct_bin = qc.assign_ct_bin(ct_value)
        
        # 사용 가능 여부
        usable = (qc_status == 'PASS')
        
        # 결과 수집
        result = {
            'well_uid': well_uid,
            'run_id': well_data['run_id'].iloc[0],
            'Well': well_data['Well'].iloc[0],
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
        }
        
        # channel 추가 (있다면)
        if 'channel' in well_data.columns:
            result['channel'] = well_data['channel'].iloc[0]
        
        results.append(result)
    
    catalog = pd.DataFrame(results)
    
    return catalog


def create_excluded_report(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    사용 불가 데이터(excluded)에 대한 상세 리포트
    """
    excluded = catalog[~catalog['usable']].copy()
    
    if len(excluded) == 0:
        return pd.DataFrame(columns=[
            'well_uid', 'run_id', 'Well', 'excluded_major_reason', 'excluded_minor_reason', 
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
    report_cols = ['well_uid', 'run_id', 'Well', 'excluded_major_reason', 'excluded_minor_reason',
                   'r2', 'snr', 'ct_value', 'ct_bin', 'qc_status']
    
    # channel 추가 (있다면)
    if 'channel' in excluded.columns:
        report_cols.insert(3, 'channel')
    
    report = excluded[report_cols].copy()
    
    report.rename(columns={
        'r2': 'evidence_r2',
        'snr': 'evidence_snr',
        'ct_value': 'evidence_ct'
    }, inplace=True)
    
    return report


def main():
    print("=" * 60)
    print("QC 결과 저장 스크립트 (well_uid 기반)")
    print("=" * 60)
    
    # 경로 설정
    master_long_path = PROJECT_ROOT / "data" / "canonical" / "master_long.parquet"
    output_dir = PROJECT_ROOT / "outputs" / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Input: {master_long_path}")
    print(f"📂 Output: {output_dir}")
    
    # 1. 데이터 로드
    print("\n🔄 Loading master_long.parquet...")
    df_long = pd.read_parquet(master_long_path)
    print(f"   Loaded {len(df_long):,} rows")
    print(f"   Columns: {df_long.columns.tolist()}")
    
    # 2. well_uid 기반 QC 분석
    print("\n🔬 Running QC analysis (well_uid based)...")
    catalog = create_qc_catalog_from_long(df_long)
    
    # 3. 요약 통계
    print("\n📊 QC Summary:")
    print(f"   Total:    {len(catalog):,}")
    print(f"   PASS:     {(catalog['qc_status'] == 'PASS').sum():,} ({(catalog['qc_status'] == 'PASS').sum()/len(catalog)*100:.1f}%)")
    print(f"   FAIL:     {(catalog['qc_status'] == 'FAIL').sum():,} ({(catalog['qc_status'] == 'FAIL').sum()/len(catalog)*100:.1f}%)")
    print(f"   FLAG:     {(catalog['qc_status'] == 'FLAG').sum():,} ({(catalog['qc_status'] == 'FLAG').sum()/len(catalog)*100:.1f}%)")
    print(f"   Usable:   {catalog['usable'].sum():,}")
    print(f"   Excluded: {(~catalog['usable']).sum():,}")
    
    # well_uid 샘플 출력
    print("\n📝 Sample well_uid:")
    print(f"   {catalog['well_uid'].head(5).tolist()}")
    
    # 4. Excluded Report 생성
    print("\n🔄 Creating excluded report...")
    excluded_report = create_excluded_report(catalog)
    print(f"   Excluded wells: {len(excluded_report):,}")
    
    # 5. 저장
    print("\n💾 Saving results...")
    
    catalog_path = output_dir / "master_catalog.parquet"
    catalog.to_parquet(catalog_path, index=False)
    print(f"   ✅ {catalog_path}")
    
    excluded_path = output_dir / "excluded_report.parquet"
    excluded_report.to_parquet(excluded_path, index=False)
    print(f"   ✅ {excluded_path}")
    
    # CSV도 저장 (편의성)
    catalog.to_csv(output_dir / "master_catalog.csv", index=False)
    excluded_report.to_csv(output_dir / "excluded_report.csv", index=False)
    print(f"   ✅ CSV files also saved")
    
    # 6. 검증
    print("\n🔍 Validation:")
    print(f"   well_uid format check:")
    sample_uids = catalog['well_uid'].head(3).tolist()
    for uid in sample_uids:
        parts = uid.split('__')
        print(f"      {uid} → run_id='{parts[0] if len(parts) > 0 else 'N/A'}', Well='{parts[1] if len(parts) > 1 else 'N/A'}'")
    
    print("\n" + "=" * 60)
    print("✅ QC 결과 저장 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()