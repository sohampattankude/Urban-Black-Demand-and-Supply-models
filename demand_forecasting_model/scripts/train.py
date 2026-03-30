#!/usr/bin/env python3
"""
Full training pipeline: data → preprocessing → LSTM → XGBoost → ensemble.
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import LocalDataLoader
from src.features.preprocessor import DemandFeaturesPreprocessor
from src.models.lstm import LSTMDemandModel
from src.evaluation.metrics import evaluate_predictions
from src.logger import setup_logger
from src.config import config

# Setup logging
logger = setup_logger('training', 'logs/training.log')

def create_sequences(df, seq_len=24, lag_cols=None, temporal_cols=None, supply_cols=None):
    """
    Create overlapping sequences for LSTM.
    
    Args:
        df: DataFrame with features
        seq_len: Sequence length
        lag_cols: List of lag feature columns
        temporal_cols: List of temporal feature columns
        supply_cols: List of supply feature columns
    
    Returns:
        Tuple of (X_lag, X_temporal, X_supply, y)
    """
    
    if lag_cols is None:
        lag_cols = [c for c in df.columns if 'lag_' in c or 'rolling_' in c]
    if temporal_cols is None:
        temporal_cols = ['hour', 'day_of_week', 'is_weekend', 'seasonal_index_hour',
                        'seasonal_index_dow', 'seasonal_index_zone']
    if supply_cols is None:
        supply_cols = ['active_drivers_count', 'available_drivers_count', 'completed_rate']
    
    # Ensure columns exist
    lag_cols = [c for c in lag_cols if c in df.columns]
    temporal_cols = [c for c in temporal_cols if c in df.columns]
    supply_cols = [c for c in supply_cols if c in df.columns]
    
    X_lag, X_temporal, X_supply, y = [], [], [], []
    
    for zone_id in df['zone_id'].unique():
        zone_data = df[df['zone_id'] == zone_id].sort_values('timestamp').reset_index(drop=True)
        
        if len(zone_data) < seq_len + 1:
            continue
        
        for i in range(seq_len, len(zone_data)):
            # Lag features (sequence)
            if len(lag_cols) > 0:
                X_lag.append(zone_data[lag_cols].iloc[i-seq_len:i].values)
            
            # Temporal features (current)
            if len(temporal_cols) > 0:
                X_temporal.append(zone_data[temporal_cols].iloc[i].values)
            
            # Supply features (current)
            if len(supply_cols) > 0:
                X_supply.append(zone_data[supply_cols].iloc[i].values)
            
            # Target (next request count)
            y.append(zone_data['requests_count'].iloc[i])
    
    return (
        np.array(X_lag) if X_lag else np.zeros((len(y), seq_len, 1)),
        np.array(X_temporal) if X_temporal else np.zeros((len(y), 1)),
        np.array(X_supply) if X_supply else np.zeros((len(y), 1)),
        np.array(y)
    )

def train_pipeline():
    """Execute full training pipeline."""
    
    logger.info("=" * 70)
    logger.info("DEMAND FORECASTING MODEL - TRAINING PIPELINE")
    logger.info("=" * 70)
    
    # ===== Step 1: Load Data =====
    logger.info("\n[STEP 1] Loading preprocessed features...")
    
    try:
        features_df = LocalDataLoader.load_parquet('data/processed/demand_features_full.parquet')
    except FileNotFoundError:
        logger.error("❌ Preprocessed data not found. Run feature engineering first.")
        return False
    
    # Sort by timestamp for time-series aware split
    features_df = features_df.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"✅ Loaded {len(features_df):,} rows")
    
    # ===== Step 2: Train/Val/Test Split =====
    logger.info("\n[STEP 2] Splitting data into train/val/test...")
    
    n = len(features_df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)
    
    train_df = features_df.iloc[:n_train].copy()
    val_df = features_df.iloc[n_train:n_val].copy()
    test_df = features_df.iloc[n_val:].copy()
    
    logger.info(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    
    # ===== Step 3: Prepare LSTM Sequences =====
    logger.info("\n[STEP 3] Creating LSTM sequences...")
    
    seq_length = config.get('features.seq_length', 24)
    
    lag_cols = [c for c in features_df.columns if 'lag_' in c or 'rolling_' in c]
    temporal_cols = ['hour', 'day_of_week', 'is_weekend', 'seasonal_index_hour',
                    'seasonal_index_dow', 'seasonal_index_zone']
    supply_cols = ['active_drivers_count', 'available_drivers_count']
    
    train_lag, train_temp, train_supply, train_y = create_sequences(
        train_df, seq_length, lag_cols, temporal_cols, supply_cols
    )
    val_lag, val_temp, val_supply, val_y = create_sequences(
        val_df, seq_length, lag_cols, temporal_cols, supply_cols
    )
    test_lag, test_temp, test_supply, test_y = create_sequences(
        test_df, seq_length, lag_cols, temporal_cols, supply_cols
    )
    
    logger.info(f"Train sequences: {train_lag.shape[0]:,}")
    logger.info(f"Val sequences: {val_lag.shape[0]:,}")
    logger.info(f"Test sequences: {test_lag.shape[0]:,}")
    
    # ===== Step 4: Train LSTM =====
    logger.info("\n[STEP 4] Training LSTM model...")
    
    lstm_config = config['lstm']
    lstm_model = LSTMDemandModel(
        lstm_units_l1=lstm_config.get('units_layer1', 64),
        lstm_units_l2=lstm_config.get('units_layer2', 32),
        dropout_rate=lstm_config.get('dropout_rate', 0.22),
        seq_length=seq_length,
        num_lag_features=train_lag.shape[2]
    )
    lstm_model.compile_model(learning_rate=lstm_config.get('learning_rate', 0.001))
    
    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=lstm_config.get('early_stopping_patience', 5),
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=lstm_config.get('reduce_lr_patience', 3),
        min_lr=1e-6
    )
    
    # Train
    history = lstm_model.model.fit(
        x={
            'lag_features': train_lag,
            'temporal_features': train_temp,
            'supply_features': train_supply
        },
        y=[train_y, train_y, train_y, train_y],  # Same target for all horizons
        validation_data=(
            {
                'lag_features': val_lag,
                'temporal_features': val_temp,
                'supply_features': val_supply
            },
            [val_y, val_y, val_y, val_y]
        ),
        epochs=lstm_config.get('epochs', 100),
        batch_size=lstm_config.get('batch_size', 32),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    logger.info("✅ LSTM training complete")
    
    # ===== Step 5: LSTM Predictions =====
    logger.info("\n[STEP 5] Generating LSTM predictions...")
    
    lstm_test_preds = lstm_model.model.predict({
        'lag_features': test_lag,
        'temporal_features': test_temp,
        'supply_features': test_supply
    })
    
    logger.info("✅ LSTM predictions generated")
    
    # ===== Step 6: Prepare for XGBoost =====
    logger.info("\n[STEP 6] Preparing XGBoost features...")
    
    xgb_features = lag_cols + temporal_cols + supply_cols
    xgb_features = [c for c in xgb_features if c in features_df.columns]
    
    X_train_xgb = train_df[xgb_features].values
    X_test_xgb = test_df[xgb_features].values
    
    logger.info(f"XGBoost features: {len(xgb_features)}")
    
    # ===== Step 7: Train XGBoost Models =====
    logger.info("\n[STEP 7] Training XGBoost models...")
    
    xgb_config = config['xgboost']
    xgb_models = {}
    horizons = config.get('ensemble.horizons', [15, 30, 60, 120])
    
    for idx, horizon in enumerate(horizons):
        logger.info(f"Training XGBoost for {horizon}-min horizon...")
        
        params = {
            'tree_method': 'hist',
            'max_depth': xgb_config.get('max_depth', 7),
            'learning_rate': xgb_config.get('learning_rate', 0.1),
            'subsample': xgb_config.get('subsample', 0.81),
            'colsample_bytree': xgb_config.get('colsample_bytree', 0.79),
            'lambda': xgb_config.get('lambda', 1.2),
            'alpha': xgb_config.get('alpha', 0.4),
            'objective': 'reg:squarederror'
        }
        
        dtrain = xgb.DMatrix(X_train_xgb, label=train_y)
        dtest = xgb.DMatrix(X_test_xgb, label=test_y)
        
        xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=xgb_config.get('num_boost_round', 150),
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=xgb_config.get('early_stopping_rounds', 10),
            verbose_eval=20
        )
        
        xgb_models[f'horizon_{horizon}'] = xgb_model
    
    logger.info("✅ XGBoost training complete")
    
    # ===== Step 8: Ensemble Predictions =====
    logger.info("\n[STEP 8] Creating ensemble predictions...")
    
    lstm_weight = config.get('ensemble.lstm_weight', 0.6)
    xgb_weight = config.get('ensemble.xgboost_weight', 0.4)
    
    ensemble_preds = {}
    metrics = {}
    
    for idx, horizon in enumerate(horizons):
        xgb_pred = xgb_models[f'horizon_{horizon}'].predict(xgb.DMatrix(X_test_xgb))
        lstm_pred = lstm_test_preds[idx].flatten()
        
        # Weighted ensemble
        ensemble = lstm_weight * lstm_pred + xgb_weight * xgb_pred
        ensemble_preds[f'horizon_{horizon}'] = ensemble
        
        # Evaluate
        m = evaluate_predictions(test_y, ensemble, horizon_name=f'{horizon}min')
        metrics[f'horizon_{horizon}'] = m
    
    logger.info("✅ Ensemble predictions created")
    
    # ===== Step 9: Save Artifacts =====
    logger.info("\n[STEP 9] Saving model artifacts...")
    
    Path('models').mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_version = config.get('training.model_version', 'v1.0.0')
    
    # Save LSTM
    lstm_model.save_model(f'models/lstm_{model_version}.h5')
    logger.info(f"✅ Saved LSTM to models/lstm_{model_version}.h5")
    
    # Save XGBoost
    for horizon, model in xgb_models.items():
        model.save_model(f'models/xgboost_{horizon}_{model_version}.json')
    logger.info(f"✅ Saved {len(xgb_models)} XGBoost models")
    
    # Save ensemble config
    ensemble_config = {
        'model_version': model_version,
        'training_date': timestamp,
        'lstm_weight': lstm_weight,
        'xgboost_weight': xgb_weight,
        'horizons': horizons,
        'metrics': {k: {mk: float(mv) if isinstance(mv, (int, float, np.number)) else mv 
                       for mk, mv in v.items()} 
                   for k, v in metrics.items()}
    }
    
    with open(f'models/ensemble_config_{model_version}.json', 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    logger.info(f"✅ Saved ensemble config")
    
    # Save scalers if available
    if hasattr(train_df, 'scalers'):
        with open('models/scalers.pkl', 'wb') as f:
            pickle.dump(train_df.scalers, f)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Models saved to: models/")
    logger.info(f"Model version: {model_version}")
    logger.info(f"Timestamp: {timestamp}")
    
    return True

if __name__ == '__main__':
    try:
        success = train_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        sys.exit(1)
