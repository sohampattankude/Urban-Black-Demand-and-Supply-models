"""
Feature engineering and preprocessing pipeline.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Tuple, List
import logging
from sklearn.preprocessing import StandardScaler
import geohash2

logger = logging.getLogger(__name__)

class DemandFeaturesPreprocessor:
    """Complete feature engineering pipeline for demand forecasting."""
    
    def __init__(self, 
                 seq_length: int = 24,
                 time_based_indices_path: str = 'data/time_based_indices/seasonal_indices_v1.1.json',
                 geohash_precision: int = 5):
        """
        Initialize preprocessor.
        
        Args:
            seq_length: Sequence length for LSTM (in 15-min buckets)
            time_based_indices_path: Path to seasonal indices from Model 2
            geohash_precision: Precision for geohash zones
        """
        self.seq_length = seq_length
        self.geohash_precision = geohash_precision
        self.scalers = {}
        
        # Try to load seasonal indices (optional for initial setup)
        try:
            self.time_based_indices = self._load_seasonal_indices(time_based_indices_path)
        except Exception as e:
            logger.warning(f"Could not load seasonal indices: {e}. Using defaults.")
            self.time_based_indices = self._get_default_indices()
    
    def _load_seasonal_indices(self, path: str) -> Dict:
        """Load pre-computed seasonal indices from Model 2."""
        with open(path, 'r') as f:
            indices = json.load(f)
        logger.info(f"✅ Loaded seasonal indices from {path}")
        return indices
    
    def _get_default_indices(self) -> Dict:
        """Return default seasonal indices."""
        return {
            'hourly_coefficients': {},
            'day_of_week_coefficients': {},
            'zone_coefficients_normalized': {}
        }
    
    def preprocess(self, 
                   rides_df: pd.DataFrame,
                   driver_locs_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Main preprocessing pipeline.
        
        Args:
            rides_df: DataFrame with rides data
            driver_locs_df: DataFrame with driver locations (optional)
        
        Returns:
            Tuple of (features_df, scalers_dict)
        """
        
        # Step 1: Add temporal features
        rides_df = self._add_temporal_features(rides_df)
        logger.info("✅ Added temporal features")
        
        # Step 2: Map rides to zones
        rides_df = self._map_to_zones(rides_df)
        logger.info("✅ Mapped rides to zones")
        
        # Step 3: Aggregate to 15-min buckets
        demand_15min = self._aggregate_demand_15min(rides_df)
        logger.info("✅ Aggregated demand to 15-min buckets")
        
        # Step 4: Add supply context (if available)
        if driver_locs_df is not None and len(driver_locs_df) > 0:
            supply_15min = self._aggregate_supply_15min(driver_locs_df)
            merged = demand_15min.merge(supply_15min, 
                                       on=['zone_id', 'timestamp'], 
                                       how='left')
        else:
            logger.warning("No driver locations provided. Skipping supply features.")
            merged = demand_15min
            merged['active_drivers_count'] = 5  # Placeholder
            merged['available_drivers_count'] = 3
        
        logger.info("✅ Merged demand and supply data")
        
        # Step 5: Add lag features
        merged = self._add_lag_features(merged)
        logger.info("✅ Added lag features")
        
        # Step 6: Add seasonal indices
        merged = self._add_seasonal_indices(merged)
        logger.info("✅ Added seasonal indices")
        
        # Step 7: Data quality checks & imputation
        merged = self._validate_and_impute(merged)
        logger.info("✅ Validated and imputed data")
        
        # Step 8: Normalize features
        merged, scalers = self._normalize_features(merged)
        logger.info("✅ Normalized features")
        
        logger.info(f"✅ Preprocessing complete. Final shape: {merged.shape}")
        return merged, scalers
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features."""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['requestedAt'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_name'] = df['timestamp'].dt.day_name()
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        df['month'] = df['timestamp'].dt.month
        df['date'] = df['timestamp'].dt.date
        return df
    
    def _map_to_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map pickup lat/lng to geohash zones."""
        def get_zone(lat, lng):
            try:
                return geohash2.encode(lat, lng, precision=self.geohash_precision)
            except:
                return None
        
        df = df.copy()
        df['zone_id'] = df.apply(
            lambda row: get_zone(row['pickupLat'], row['pickupLng']),
            axis=1
        )
        
        # Drop rides without valid zones
        df = df[df['zone_id'].notna()]
        logger.info(f"✅ Mapped {len(df):,} rides to zones")
        return df
    
    def _aggregate_demand_15min(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate rides to 15-minute buckets per zone."""
        
        df = df.copy()
        df['bucket_timestamp'] = df['timestamp'].dt.floor('15min')
        
        # Aggregate
        demand = df.groupby(['zone_id', 'bucket_timestamp']).agg({
            'id': 'count',
            'status': lambda x: (x == 'COMPLETED').sum(),
            'fare': 'mean',
            'durationMin': 'mean',
            'rideKm': 'mean'
        }).reset_index()
        
        demand.columns = ['zone_id', 'timestamp', 'requests_count', 
                         'completed_requests', 'avg_fare', 'avg_duration_min', 'avg_distance_km']
        
        demand['completed_rate'] = (demand['completed_requests'] / 
                                    demand['requests_count'].clip(lower=1))
        
        logger.info(f"✅ Aggregated to {len(demand):,} (zone, 15-min) pairs")
        return demand
    
    def _aggregate_supply_15min(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count active drivers per zone at 15-min intervals."""
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['updatedAt'])
        df['bucket_timestamp'] = df['timestamp'].dt.floor('15min')
        
        df['zone_id'] = df.apply(
            lambda row: geohash2.encode(row['lat'], row['lng'], 
                                       precision=self.geohash_precision),
            axis=1
        )
        
        supply = df.groupby(['zone_id', 'bucket_timestamp']).agg({
            'driverId': 'count'
        }).reset_index()
        
        supply.columns = ['zone_id', 'timestamp', 'active_drivers_count']
        supply['available_drivers_count'] = supply['active_drivers_count'] * 0.6
        
        logger.info(f"✅ Aggregated supply for {len(supply):,} (zone, 15-min) pairs")
        return supply
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for demand patterns."""
        
        df = df.copy()
        lag_distances = [1, 2, 4, 6, 24, 96]
        
        # Add lag features for each zone
        for zone_id in df['zone_id'].unique():
            zone_mask = df['zone_id'] == zone_id
            zone_data = df[zone_mask].sort_values('timestamp')
            
            for lag in lag_distances:
                col_name = f'lag_{lag}x15min_requests'
                df.loc[zone_mask, col_name] = zone_data['requests_count'].shift(lag).values
        
        # Add rolling statistics
        for zone_id in df['zone_id'].unique():
            zone_mask = df['zone_id'] == zone_id
            zone_data = df[zone_mask].sort_values('timestamp')
            
            # Rolling mean and std (1 hour = 4 buckets)
            rolling_mean = zone_data['requests_count'].rolling(window=4, min_periods=1).mean()
            rolling_std = zone_data['requests_count'].rolling(window=4, min_periods=1).std()
            
            df.loc[zone_mask, 'rolling_mean_1h'] = rolling_mean.values
            df.loc[zone_mask, 'rolling_std_1h'] = rolling_std.values
        
        logger.info(f"✅ Added {len(lag_distances)} lag features")
        return df
    
    def _add_seasonal_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seasonal indices from time-based model."""
        
        df = df.copy()
        
        # Hourly indices
        df['seasonal_index_hour'] = df.apply(
            lambda row: self.time_based_indices['hourly_coefficients']
                        .get(row['day_name'], {})
                        .get(str(row['hour']), 1.0),
            axis=1
        )
        
        # Day-of-week indices
        df['seasonal_index_dow'] = df.apply(
            lambda row: self.time_based_indices['day_of_week_coefficients']
                        .get(row['day_name'], 1.0),
            axis=1
        )
        
        # Zone indices
        df['seasonal_index_zone'] = df['zone_id'].map(
            self.time_based_indices.get('zone_coefficients_normalized', {})
        ).fillna(1.0)
        
        logger.info("✅ Added seasonal indices")
        return df
    
    def _validate_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data quality checks and imputation."""
        
        df = df.copy()
        
        # Remove rows with NULL requests_count
        initial_len = len(df)
        df = df[df['requests_count'].notna()]
        logger.info(f"Removed {initial_len - len(df)} rows with NULL requests_count")
        
        # Forward-fill lag features
        lag_cols = [c for c in df.columns if 'lag_' in c]
        for col in lag_cols:
            df[col] = df.groupby('zone_id')[col].fillna(method='bfill').fillna(method='ffill')
        
        # Impute missing supply features
        df['active_drivers_count'] = df['active_drivers_count'].fillna(5)
        df['available_drivers_count'] = df['available_drivers_count'].fillna(3)
        
        # Impute rolling stats
        global_mean = df['requests_count'].mean()
        df['rolling_mean_1h'] = df['rolling_mean_1h'].fillna(global_mean)
        df['rolling_std_1h'] = df['rolling_std_1h'].fillna(global_mean * 0.3)
        
        # Fill remaining NaNs with 0
        df = df.fillna(0)
        
        logger.info(f"✅ Data quality checks passed. Final shape: {df.shape}")
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Standardize numeric features."""
        
        df = df.copy()
        
        # Features to normalize
        numeric_features = [
            'requests_count', 'avg_fare', 'avg_duration_min', 'avg_distance_km',
            'active_drivers_count', 'available_drivers_count', 'rolling_mean_1h', 'rolling_std_1h'
        ] + [c for c in df.columns if 'lag_' in c]
        
        scalers = {}
        for feat in numeric_features:
            if feat in df.columns:
                scaler = StandardScaler()
                df[f'{feat}_scaled'] = scaler.fit_transform(df[[feat]])
                scalers[feat] = scaler
        
        logger.info(f"✅ Normalized {len(numeric_features)} features")
        return df, scalers
