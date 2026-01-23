from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List

ROOT = Path(".")
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "data" / "reports"

def get_active_model_id() -> Optional[str]:
    """
    models/active_model.txt 에 적힌 model_id를 읽는다.
    없으면 data/reports/ 안의 최신 폴더명을 fallback으로 반환.
    """
    p = MODELS_DIR / "active_model.txt"
    if p.exists():
        mid = p.read_text().strip()
        return mid or None

    # fallback: data/reports의 최신 폴더
    if REPORTS_DIR.exists():
        cands = [d for d in REPORTS_DIR.iterdir() if d.is_dir()]
        if cands:
            cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return cands[0].name
    return None

def list_report_model_ids() -> List[str]:
    if not REPORTS_DIR.exists():
        return []
    ids = [d.name for d in REPORTS_DIR.iterdir() if d.is_dir()]
    ids.sort()
    return ids

def resolve_report_paths(model_id: str) -> Dict[str, Path]:
    """
    우리가 기대하는 리포트 파일들 경로 반환.
    - metrics_by_cutoff.parquet: cutoff별 MAE/RMSE 테이블
    - predictions_long.parquet: (run_id, well_id, cutoff) 단위 예측/오차 long-format
    - errors_by_sample.parquet: (선택) cutoff 고정의 샘플별 에러 테이블(있으면 사용)
    """
    base = REPORTS_DIR / model_id
    return {
        "base": base,
        "metrics_by_cutoff": base / "metrics_by_cutoff.parquet",
        "predictions_long": base / "predictions_long.parquet",
        "errors_by_sample": base / "errors_by_sample.parquet",
    }
