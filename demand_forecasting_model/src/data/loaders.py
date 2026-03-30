"""
Data warehouse and BigQuery connection module.
"""

import os
from typing import Optional, List
from google.cloud import bigquery
from google.cloud import storage
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class WarehouseClient:
    """Client for connecting to BigQuery data warehouse."""
    
    def __init__(self, project_id: Optional[str] = None, credentials_path: Optional[str] = None):
        """
        Initialize warehouse client.
        
        Args:
            project_id: GCP project ID (defaults to env var BQ_PROJECT_ID)
            credentials_path: Path to service account JSON (optional)
        """
        self.project_id = project_id or os.environ.get('BQ_PROJECT_ID', 'urban-black-prod')
        self.dataset = os.environ.get('BQ_DATASET', 'ride_service')
        
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        self.client = bigquery.Client(project=self.project_id)
        logger.info(f"✅ Initialized BigQuery client for project: {self.project_id}")
    
    def query(self, sql: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Execute a query and return results as DataFrame.
        
        Args:
            sql: SQL query string
            use_cache: Use BigQuery cache (default: True)
        
        Returns:
            DataFrame with query results
        """
        job_config = bigquery.QueryJobConfig(use_query_cache=use_cache)
        query_job = self.client.query(sql, job_config=job_config)
        return query_job.to_dataframe()
    
    def download_rides_data(self, days_back: int = 180) -> pd.DataFrame:
        """
        Download rides with locations and timestamps.
        
        Args:
            days_back: Number of days of history to download
        
        Returns:
            DataFrame with rides data
        """
        start_date = (datetime.now() - timedelta(days=days_back)).date()
        
        query = f"""
        SELECT
            r.id,
            r.userId,
            r.driverId,
            r.pickupLat,
            r.pickupLng,
            r.dropLat,
            r.dropLng,
            r.vehicleType,
            r.status,
            r.fare,
            r.requestedAt,
            r.startedAt,
            r.completedAt,
            r.durationMin,
            rr.rideKm,
            rr.approachKm
        FROM `{self.project_id}.{self.dataset}.rides` r
        LEFT JOIN `{self.project_id}.{self.dataset}.ride_routes` rr ON r.id = rr.rideId
        WHERE DATE(r.requestedAt) >= '{start_date}'
            AND r.status IN ('COMPLETED', 'CANCELLED_BY_RIDER', 'CANCELLED_BY_DRIVER')
        ORDER BY r.requestedAt DESC
        """
        
        logger.info(f"Downloading rides data from {start_date}...")
        df = self.query(query)
        logger.info(f"✅ Downloaded {len(df):,} rides")
        
        return df
    
    def download_driver_locations(self, days_back: int = 180) -> pd.DataFrame:
        """
        Download driver location snapshots.
        
        Args:
            days_back: Number of days of history
        
        Returns:
            DataFrame with driver locations
        """
        query = f"""
        SELECT
            driverId,
            lat,
            lng,
            updatedAt
        FROM `{self.project_id}.{self.dataset}.driver_locations`
        WHERE updatedAt >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days_back} DAY)
        LIMIT 10000000
        """
        
        logger.info("Downloading driver locations...")
        df = self.query(query)
        logger.info(f"✅ Downloaded {len(df):,} location snapshots")
        
        return df
    
    def download_driver_shifts(self, days_back: int = 180) -> pd.DataFrame:
        """
        Download driver shift records.
        
        Args:
            days_back: Number of days of history
        
        Returns:
            DataFrame with shift data
        """
        query = f"""
        SELECT
            id,
            driverId,
            shiftStart,
            shiftEnd,
            status,
            goalKm,
            totalRideKm,
            totalDeadKm
        FROM `{self.project_id}.{self.dataset}.driver_shifts`
        WHERE shiftStart >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days_back} DAY)
        """
        
        logger.info("Downloading shifts data...")
        df = self.query(query)
        logger.info(f"✅ Downloaded {len(df):,} shifts")
        
        return df

class LocalDataLoader:
    """Load data from local parquet/CSV files."""
    
    @staticmethod
    def load_parquet(path: str) -> pd.DataFrame:
        """Load data from parquet file."""
        logger.info(f"Loading {path}...")
        df = pd.read_parquet(path)
        logger.info(f"✅ Loaded {len(df):,} rows from {path}")
        return df
    
    @staticmethod
    def load_csv(path: str, **kwargs) -> pd.DataFrame:
        """Load data from CSV file."""
        logger.info(f"Loading {path}...")
        df = pd.read_csv(path, **kwargs)
        logger.info(f"✅ Loaded {len(df):,} rows from {path}")
        return df
    
    @staticmethod
    def save_parquet(df: pd.DataFrame, path: str):
        """Save DataFrame to parquet."""
        df.to_parquet(path, index=False)
        logger.info(f"✅ Saved {len(df):,} rows to {path}")
    
    @staticmethod
    def save_csv(df: pd.DataFrame, path: str, **kwargs):
        """Save DataFrame to CSV."""
        df.to_csv(path, index=False, **kwargs)
        logger.info(f"✅ Saved {len(df):,} rows to {path}")
