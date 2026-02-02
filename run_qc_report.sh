#!/usr/bin/env bash
set -euo pipefail
CUTOFF="${1:-20}"
PRED="reports/model_server_latest_xgb/predictions_long.parquet"

source .venv/bin/activate
python3 scripts/save_qc_results.py
python3 scripts/analyze_qc_performance.py --cutoff "$CUTOFF" --predictions "$PRED"
