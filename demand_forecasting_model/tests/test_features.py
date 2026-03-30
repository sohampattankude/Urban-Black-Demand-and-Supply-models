"""
Unit tests for feature engineering module.
"""

import pytest
import numpy as np
from src.features.preprocessor import DemandFeaturesPreprocessor

def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    processor = DemandFeaturesPreprocessor(seq_length=24)
    
    assert processor.seq_length == 24
    assert processor.geohash_precision == 5
    assert processor.time_based_indices is not None

def test_add_temporal_features(sample_rides_df):
    """Test temporal feature extraction."""
    processor = DemandFeaturesPreprocessor()
    df = processor._add_temporal_features(sample_rides_df)
    
    assert 'hour' in df.columns
    assert 'day_of_week' in df.columns
    assert 'day_name' in df.columns
    assert 'is_weekend' in df.columns
    assert len(df) == len(sample_rides_df)

def test_map_to_zones(sample_rides_df):
    """Test zone mapping."""
    processor = DemandFeaturesPreprocessor()
    df = processor._add_temporal_features(sample_rides_df)
    df = processor._map_to_zones(df)
    
    assert 'zone_id' in df.columns
    assert df['zone_id'].notna().any()
    assert len(df) <= len(sample_rides_df)  # Some may be dropped

def test_aggregate_demand_15min(sample_rides_df):
    """Test demand aggregation to 15-min buckets."""
    processor = DemandFeaturesPreprocessor()
    df = processor._add_temporal_features(sample_rides_df)
    df = processor._map_to_zones(df)
    demand = processor._aggregate_demand_15min(df)
    
    assert 'zone_id' in demand.columns
    assert 'timestamp' in demand.columns
    assert 'requests_count' in demand.columns
    assert len(demand) > 0

def test_full_preprocessing(sample_rides_df, sample_driver_locations_df):
    """Test full preprocessing pipeline."""
    processor = DemandFeaturesPreprocessor()
    features, scalers = processor.preprocess(sample_rides_df, sample_driver_locations_df)
    
    assert len(features) > 0
    assert len(scalers) > 0
    assert 'requests_count' in features.columns
    assert 'zone_id' in features.columns
