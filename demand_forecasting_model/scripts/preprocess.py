#!/usr/bin/env python3
"""
Feature engineering script: convert raw data to features.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import LocalDataLoader
from src.features.preprocessor import DemandFeaturesPreprocessor
from src.logger import setup_logger

# Setup logging
logger = setup_logger('preprocess', 'logs/preprocess.log')

def preprocess_data():
    """
    Execute feature engineering pipeline.
    """
    
    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 70)
    
    try:
        # Load raw data
        logger.info("\n[1/3] Loading raw data...")
        
        try:
            rides_df = LocalDataLoader.load_parquet('data/raw/rides_6months.parquet')
        except FileNotFoundError:
            logger.error("❌ Rides data not found. Run download_data.py first.")
            return False
        
        try:
            driver_locs_df = LocalDataLoader.load_parquet('data/raw/driver_locations_6months.parquet')
        except FileNotFoundError:
            logger.warning("⚠️  Driver locations not found. Proceeding without supply features.")
            driver_locs_df = None
        
        logger.info(f"✅ Loaded {len(rides_df):,} rides")
        
        # Feature engineering
        logger.info("\n[2/3] Executing feature engineering...")
        
        processor = DemandFeaturesPreprocessor(
            seq_length=24,
            geohash_precision=5
        )
        
        features_df, scalers = processor.preprocess(rides_df, driver_locs_df)
        logger.info(f"✅ Generated {features_df.shape[1]} features")
        
        # Save processed data
        logger.info("\n[3/3] Saving processed features...")
        
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        LocalDataLoader.save_parquet(features_df, 'data/processed/demand_features_full.parquet')
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Output: data/processed/demand_features_full.parquet")
        logger.info(f"Rows: {len(features_df):,}")
        logger.info(f"Features: {features_df.shape[1]}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    success = preprocess_data()
    sys.exit(0 if success else 1)
