"""
Pytest configuration and fixtures.
"""

import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

@pytest.fixture
def sample_rides_df():
    """Create sample rides DataFrame for testing."""
    base_date = datetime(2024, 1, 1)
    dates = [base_date + timedelta(hours=i) for i in range(1000)]
    
    df = pd.DataFrame({
        'id': range(1000),
        'userId': np.random.randint(1, 100, 1000),
        'driverId': np.random.randint(1, 50, 1000),
        'pickupLat': np.random.uniform(40.7, 40.8, 1000),
        'pickupLng': np.random.uniform(-74.0, -73.95, 1000),
        'dropLat': np.random.uniform(40.7, 40.8, 1000),
        'dropLng': np.random.uniform(-74.0, -73.95, 1000),
        'status': np.random.choice(['COMPLETED', 'CANCELLED_BY_RIDER', 'CANCELLED_BY_DRIVER'], 1000),
        'fare': np.random.uniform(5, 50, 1000),
        'durationMin': np.random.uniform(5, 60, 1000),
        'rideKm': np.random.uniform(1, 20, 1000),
        'requestedAt': dates,
        'startedAt': dates,
        'completedAt': [d + timedelta(minutes=m) for d, m in zip(dates, np.random.uniform(5, 60, 1000))],
    })
    
    return df

@pytest.fixture
def sample_driver_locations_df():
    """Create sample driver locations DataFrame for testing."""
    base_date = datetime(2024, 1, 1)
    dates = [base_date + timedelta(hours=i) for i in range(500)]
    
    df = pd.DataFrame({
        'driverId': np.random.randint(1, 50, 500),
        'lat': np.random.uniform(40.7, 40.8, 500),
        'lng': np.random.uniform(-74.0, -73.95, 500),
        'updatedAt': dates
    })
    
    return df

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory structure for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    raw_dir = data_dir / "raw"
    raw_dir.mkdir()
    
    processed_dir = data_dir / "processed"
    processed_dir.mkdir()
    
    return data_dir

# Configuration for pytest
pytest_plugins = []
