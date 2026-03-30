#!/usr/bin/env python3
"""
Download raw data from BigQuery for demand forecasting model.
Output: Parquet files in data/raw/
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loaders import WarehouseClient, LocalDataLoader
from src.logger import setup_logger

# Setup logging
logger = setup_logger('download_data', 'logs/download_data.log')

def download_all_data(days_back: int = 180):
    """
    Download all required data from BigQuery.
    
    Args:
        days_back: Number of days of history to download
    """
    
    logger.info("=" * 70)
    logger.info("DEMAND FORECASTING MODEL - DATA DOWNLOAD")
    logger.info("=" * 70)
    logger.info(f"Downloading {days_back} days of historical data...")
    
    try:
        # Initialize warehouse client
        client = WarehouseClient()
        
        # Create data directories
        Path('data/raw').mkdir(parents=True, exist_ok=True)
        
        # Download rides data
        logger.info("\n[1/3] Downloading rides data...")
        rides_df = client.download_rides_data(days_back=days_back)
        LocalDataLoader.save_parquet(rides_df, 'data/raw/rides_6months.parquet')
        logger.info(f"✅ Rides data: {len(rides_df):,} rows")
        
        # Download driver locations
        logger.info("\n[2/3] Downloading driver locations...")
        driver_locs_df = client.download_driver_locations(days_back=days_back)
        LocalDataLoader.save_parquet(driver_locs_df, 'data/raw/driver_locations_6months.parquet')
        logger.info(f"✅ Driver locations: {len(driver_locs_df):,} rows")
        
        # Download shifts data
        logger.info("\n[3/3] Downloading shifts data...")
        shifts_df = client.download_driver_shifts(days_back=days_back)
        LocalDataLoader.save_parquet(shifts_df, 'data/raw/driver_shifts_6months.parquet')
        logger.info(f"✅ Shifts data: {len(shifts_df):,} rows")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ ALL DATA DOWNLOADED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Output location: data/raw/")
        logger.info(f"Download timestamp: {datetime.now()}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Download failed: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download data for demand forecasting model')
    parser.add_argument('--days', type=int, default=180, help='Number of days to download')
    args = parser.parse_args()
    
    success = download_all_data(days_back=args.days)
    sys.exit(0 if success else 1)
