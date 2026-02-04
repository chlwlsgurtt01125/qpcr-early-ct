#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Comparison Script for qPCR Early Ct Prediction

Compares: XGBoost, LightGBM, Random Forest, Ridge, MLP, LSTM, 1D-CNN
Metrics: MAE, RMSE, Accuracy (+-1.0, +-2.0), Fold-change <=1.5x ratio

Usage:
    python scripts/model_comparison.py --min_cutoff 10 --max_cutoff 40 --step 5
    python scripts/model_comparison.py --cutoffs 15,20,25,30
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Skipping LightGBM.")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed. Skipping LSTM and 1D-CNN.")

warnings.filterwarnings('ignore')

# ============================================
# Path Configuration
# ============================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = PROJECT_ROOT / "data" / "canonical" / "master_long.parquet"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "model_comparison"


# ============================================
# Data Loading & Feature Engineering
# ============================================

def load_master_long() -> pd.DataFrame:
    """Load master_long.parquet"""
    if not CANONICAL_PATH.exists():
        raise FileNotFoundError(f"master_long.parquet not found: {CANONICAL_PATH}")
    return pd.read_parquet(CANONICAL_PATH)


def build_features_for_cutoff(df_long: pd.DataFrame, cutoff: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build feature matrix X and target y for a given cutoff.
   
    Returns:
        X: (n_samples, cutoff) - Fluor values for cycles 1..cutoff
        y: (n_samples,) - True Ct values
        meta: DataFrame with well_uid, run_id, Well, true_ct
    """
    # Filter to cycles <= cutoff
    df = df_long[df_long["Cycle"] <= cutoff].copy()
   
    # Get unique wells with their true Ct (Cq 컬럼 우선)
    ct_candidates = ["Cq", "cq", "ct_value", "true_ct", "Ct", "CT"]
    ct_col = next((c for c in df_long.columns if c in ct_candidates), None)
    if ct_col is None:
        raise ValueError(f"Cannot find Ct column. Available: {df_long.columns.tolist()}")
    print(f"Using true Ct column: {ct_col}")  # 실행 시 출력됨

    # Pivot to wide format
    pivot = df.pivot_table(
        index=["run_id", "well_uid"],
        columns="Cycle",
        values="Fluor",
        aggfunc="first"
    ).reset_index()
   
    # Get Ct values (one per well)
    ct_df = df_long.groupby(["run_id", "well_uid"])[ct_col].first().reset_index()
    ct_df.columns = ["run_id", "well_uid", "true_ct"]
   
    # Merge
    merged = pivot.merge(ct_df, on=["run_id", "well_uid"], how="inner")
   
    # Feature columns (cycle numbers)
    feat_cols = [c for c in pivot.columns if isinstance(c, (int, float))]
    feat_cols = sorted(feat_cols)
   
    # Build X, y
    X = merged[feat_cols].values.astype(float)
    y = merged["true_ct"].values.astype(float)
   
    # Handle NaN (forward fill then backward fill)
    X = pd.DataFrame(X).ffill(axis=1).bfill(axis=1).values
   
    # Remove rows with NaN in y
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
   
    meta = merged[["run_id", "well_uid", "true_ct"]].iloc[valid_mask.nonzero()[0]].reset_index(drop=True)
   
    return X, y, meta

# ============================================
# Metrics Calculation
# ============================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate all evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    abs_err = np.abs(y_true - y_pred)
    acc_1 = np.mean(abs_err <= 1.0) * 100
    acc_2 = np.mean(abs_err <= 2.0) * 100
    
    # Fold-change: 2^|delta_Ct|
    fold_change = 2 ** abs_err
    fc_1_5 = np.mean(fold_change <= 1.5) * 100
    
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "acc_1.0": round(acc_1, 2),
        "acc_2.0": round(acc_2, 2),
        "fc_1.5x": round(fc_1_5, 2),
    }


# ============================================
# Model Definitions
# ============================================

class ModelWrapper:
    """Base class for model wrappers"""
    name: str = "BaseModel"
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None):
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def cross_val_predict(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, cv: int = 5) -> np.ndarray:
        """Default CV predict using GroupKFold"""
        gkf = GroupKFold(n_splits=cv)
        preds = np.zeros_like(y)
        
        for train_idx, val_idx in gkf.split(X, y, groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            self.fit(X_train, y_train)
            preds[val_idx] = self.predict(X_val)
        
        return preds


class XGBoostModel(ModelWrapper):
    name = "XGBoost"
    
    def __init__(self):
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 300,  # GPU면 늘려도 빠름
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbosity": 0,
            "tree_method": "hist",      # gpu_hist 대신 hist (CPU/GPU 자동)
            "device": "cuda",           # GPU 사용 (cuda 또는 gpu)
        }
            
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None):
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class LightGBMModel(ModelWrapper):
    name = "LightGBM"
    
    def __init__(self):
        self.params = {
            "objective": "regression",
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbosity": -1,
            "device": "gpu",         # GPU 핵심!
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed")
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class RandomForestModel(ModelWrapper):
    name = "RandomForest"
    
    def __init__(self):
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class RidgeModel(ModelWrapper):
    name = "Ridge"
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None):
        X_scaled = self.scaler.fit_transform(X)
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_scaled, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class MLPModel(ModelWrapper):
    name = "MLP"
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None):
        X_scaled = self.scaler.fit_transform(X)
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self.model.fit(X_scaled, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


# ============================================
# PyTorch Models (LSTM, 1D-CNN)
# ============================================

if HAS_TORCH:
    
    class LSTMNet(nn.Module):
        """LSTM for sequence regression"""
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
            )
        
        def forward(self, x):
            # x: (batch, seq_len) -> (batch, seq_len, 1)
            x = x.unsqueeze(-1)
            lstm_out, _ = self.lstm(x)
            # Use last hidden state
            out = self.fc(lstm_out[:, -1, :])
            return out.squeeze(-1)
    
    
    class CNN1DNet(nn.Module):
        """1D CNN for sequence regression"""
        def __init__(self, input_size: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
            )
        
        def forward(self, x):
            # x: (batch, seq_len) -> (batch, 1, seq_len)
            x = x.unsqueeze(1)
            x = self.conv(x)
            x = x.squeeze(-1)
            out = self.fc(x)
            return out.squeeze(-1)
    
    
    class PyTorchModelWrapper(ModelWrapper):
        """Base wrapper for PyTorch models"""
        
        def __init__(self, model_class, **kwargs):
            self.model_class = model_class
            self.model_kwargs = kwargs
            self.model = None
            self.scaler = StandardScaler()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None,
                epochs: int = 100, batch_size: int = 32, lr: float = 0.001):
            
            X_scaled = self.scaler.fit_transform(X)
            
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            self.model = self.model_class(input_size=X.shape[1], **self.model_kwargs).to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    pred = self.model(batch_X)
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                pred = self.model(X_tensor)
            
            return pred.cpu().numpy()
    
    
    class LSTMModel(PyTorchModelWrapper):
        name = "LSTM"
        
        def __init__(self):
            super().__init__(LSTMNet, hidden_size=64, num_layers=2)
    
    
    class CNN1DModel(PyTorchModelWrapper):
        name = "1D-CNN"
        
        def __init__(self):
            super().__init__(CNN1DNet)


# ============================================
# Main Comparison Runner
# ============================================

def get_all_models() -> List[ModelWrapper]:
    """Get list of all models to compare"""
    models = [
        XGBoostModel(),
        RandomForestModel(),
        RidgeModel(),
        MLPModel(),
    ]
    
    if HAS_LIGHTGBM:
        models.insert(1, LightGBMModel())
    
    if HAS_TORCH:
        models.extend([
            LSTMModel(),
            CNN1DModel(),
        ])
    
    return models


def run_comparison(
    df_long: pd.DataFrame,
    cutoffs: List[int],
    cv_folds: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run model comparison across all cutoffs.
    
    Returns:
        DataFrame with columns: model, cutoff, mae, rmse, acc_1.0, acc_2.0, fc_1.5x, train_time
    """
    models = get_all_models()
    results = []
    
    for cutoff in cutoffs:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Cutoff: {cutoff}")
            print(f"{'='*50}")
        
        # Build features
        try:
            X, y, meta = build_features_for_cutoff(df_long, cutoff)
        except Exception as e:
            print(f"  Error building features for cutoff {cutoff}: {e}")
            continue
        
        if verbose:
            print(f"  Samples: {len(y)}, Features: {X.shape[1]}")
        
        # Get groups for GroupKFold (by run_id)
        groups = meta["run_id"].values
        
        for model in models:
            if verbose:
                print(f"  Training {model.name}...", end=" ")
            
            try:
                start_time = time.time()
                
                # Cross-validation prediction
                y_pred = model.cross_val_predict(X, y, groups, cv=cv_folds)
                
                train_time = time.time() - start_time
                
                # Calculate metrics
                metrics = calculate_metrics(y, y_pred)
                metrics["model"] = model.name
                metrics["cutoff"] = cutoff
                metrics["train_time"] = round(train_time, 2)
                metrics["n_samples"] = len(y)
                
                results.append(metrics)
                
                if verbose:
                    print(f"MAE={metrics['mae']:.3f}, Acc@2={metrics['acc_2.0']:.1f}% ({train_time:.1f}s)")
                    
            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")
                results.append({
                    "model": model.name,
                    "cutoff": cutoff,
                    "mae": None,
                    "rmse": None,
                    "acc_1.0": None,
                    "acc_2.0": None,
                    "fc_1.5x": None,
                    "train_time": None,
                    "n_samples": len(y),
                    "error": str(e),
                })
    
    return pd.DataFrame(results)


def save_results(results: pd.DataFrame, output_dir: Path):
    """Save results to CSV and Parquet"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if results.empty:
        print("Warning: No results (all cutoffs failed). Saving empty files.")
        empty_df = pd.DataFrame(columns=["cutoff", "model", "mae", "rmse", "acc_1.0", "acc_2.0", "fc_1.5x", "n_samples"])
        csv_path = output_dir / f"comparison_results_{timestamp}.csv"
        parquet_path = output_dir / "comparison_results_latest.parquet"
        empty_df.to_csv(csv_path, index=False)
        empty_df.to_parquet(parquet_path, index=False)
        print(f"Saved empty: {csv_path}")
        print(f"Saved empty: {parquet_path}")
        return
    
    # Save as CSV
    csv_path = output_dir / f"comparison_results_{timestamp}.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    
    # Save as Parquet (for Streamlit)
    parquet_path = output_dir / "comparison_results_latest.parquet"
    results.to_parquet(parquet_path, index=False)
    print(f"Saved: {parquet_path}")
    
    # Save summary (best model per cutoff)
    summary = results.dropna(subset=["mae"]).groupby("cutoff").apply(
        lambda g: g.loc[g["mae"].idxmin()]
    ).reset_index(drop=True)
    
    summary_path = output_dir / "best_models_summary.csv"
    summary[["cutoff", "model", "mae", "rmse", "acc_2.0", "fc_1.5x"]].to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    
    return csv_path, parquet_path


def print_summary(results: pd.DataFrame):
    """Print summary table"""
    print("\n" + "="*70)
    print("SUMMARY: Best Model per Cutoff")
    print("="*70)
    
    valid = results.dropna(subset=["mae"])
    
    for cutoff in sorted(valid["cutoff"].unique()):
        cut_df = valid[valid["cutoff"] == cutoff]
        best = cut_df.loc[cut_df["mae"].idxmin()]
        print(f"Cutoff {cutoff:2d}: {best['model']:12s} | MAE={best['mae']:.3f} | Acc@2={best['acc_2.0']:.1f}% | FC<=1.5x={best['fc_1.5x']:.1f}%")
    
    print("\n" + "="*70)
    print("SUMMARY: Average Performance by Model")
    print("="*70)
    
    avg = valid.groupby("model").agg({
        "mae": "mean",
        "rmse": "mean",
        "acc_1.0": "mean",
        "acc_2.0": "mean",
        "fc_1.5x": "mean",
        "train_time": "mean",
    }).round(3)
    
    avg = avg.sort_values("mae")
    print(avg.to_string())


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Model Comparison for qPCR Ct Prediction")
    parser.add_argument("--min_cutoff", type=int, default=10, help="Minimum cutoff")
    parser.add_argument("--max_cutoff", type=int, default=40, help="Maximum cutoff")
    parser.add_argument("--step", type=int, default=5, help="Cutoff step")
    parser.add_argument("--cutoffs", type=str, default=None, help="Comma-separated cutoffs (overrides min/max/step)")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    # Determine cutoffs
    if args.cutoffs:
        cutoffs = [int(c.strip()) for c in args.cutoffs.split(",")]
    else:
        cutoffs = list(range(args.min_cutoff, args.max_cutoff + 1, args.step))
    
    print(f"Model Comparison Script")
    print(f"Cutoffs: {cutoffs}")
    print(f"CV Folds: {args.cv}")
    print(f"PyTorch available: {HAS_TORCH}")
    print(f"LightGBM available: {HAS_LIGHTGBM}")
    
    # Load data
    print(f"\nLoading data from {CANONICAL_PATH}...")
    df_long = load_master_long()
    print(f"Loaded {len(df_long):,} rows, {df_long['well_uid'].nunique()} unique wells")
    
    # Run comparison
    results = run_comparison(df_long, cutoffs, cv_folds=args.cv, verbose=True)
    
    # Save results
    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    save_results(results, output_dir)
    
    # Print summary
    print_summary(results)
    
    print("\nDone!")


if __name__ == "__main__":
    main()