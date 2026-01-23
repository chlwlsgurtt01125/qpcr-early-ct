from __future__ import annotations
import json
from pathlib import Path
import xgboost as xgb

MODELS_DIR = Path("data/models/by_cutoff")

def model_path(cutoff: int) -> Path:
    return MODELS_DIR / f"ct_xgb_cutoff_{cutoff:02d}.json"

def meta_path(cutoff: int) -> Path:
    return MODELS_DIR / f"ct_xgb_cutoff_{cutoff:02d}.meta.json"

def save_model(bst: xgb.Booster, cutoff: int, feat_cols: list[str], extra: dict | None = None) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(model_path(cutoff)))
    meta = {"cutoff": cutoff, "feat_cols": feat_cols}
    if extra:
        meta.update(extra)
    meta_path(cutoff).write_text(json.dumps(meta, ensure_ascii=False, indent=2))

def load_model(cutoff: int) -> tuple[xgb.Booster, dict]:
    mp = model_path(cutoff)
    if not mp.exists():
        raise FileNotFoundError(f"Model not found: {mp}. Train first.")
    bst = xgb.Booster()
    bst.load_model(str(mp))
    meta = json.loads(meta_path(cutoff).read_text())
    return bst, meta
