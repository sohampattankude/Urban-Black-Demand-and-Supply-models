# Quick Start Guide

## Project Setup (5 minutes)

### 1. Clone and Environment

```bash
cd demand-forecasting-model

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env

# Edit .env with your BigQuery credentials (if needed)
# BQ_PROJECT_ID=your-project
# BQ_DATASET=your-dataset
```

### 4. Create Required Directories

```bash
mkdir -p data/{raw,processed,time_based_indices}
mkdir -p models logs
```

---

## Development Pipeline

### Phase 1: Data Preparation (Days 1-3)

#### 1.1 Download Data
```bash
python scripts/download_data.py

# Or with custom lookback:
python scripts/download_data.py --days 180
```

**Output:** 
- `data/raw/rides_6months.parquet`
- `data/raw/driver_locations_6months.parquet`
- `data/raw/driver_shifts_6months.parquet`

#### 1.2 Exploratory Data Analysis (EDA)
```bash
# Open Jupyter notebook for data exploration
jupyter notebook notebooks/01_eda_demand_patterns.ipynb
```

**In Notebook:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
rides = pd.read_parquet('data/raw/rides_6months.parquet')

# Quick stats
print(f"Total rides: {len(rides):,}")
print(f"Date range: {rides['requestedAt'].min()} to {rides['requestedAt'].max()}")
print(f"Completion rate: {(rides.status == 'COMPLETED').mean():.1%}")

# Visualize patterns
rides['hour'] = rides['requestedAt'].dt.hour
rides['hour'].value_counts().sort_index().plot(title='Rides by Hour')
plt.show()
```

---

### Phase 2: Feature Engineering (Days 4-6)

```bash
python scripts/preprocess.py
```

**Output:** `data/processed/demand_features_full.parquet`

**What it does:**
- Extracts temporal features (hour, day_of_week, seasonality)
- Maps locations to zones (geohash)
- Aggregates demand to 15-min buckets
- Adds lag features & rolling statistics
- Normalizes all numeric features

---

### Phase 3: Model Training (Days 7-14)

```bash
python scripts/train.py
```

**Output:**
- `models/lstm_v1.0.0.h5`
- `models/xgboost_horizon_15_v1.0.0.json`
- `models/xgboost_horizon_30_v1.0.0.json`
- `models/xgboost_horizon_60_v1.0.0.json`
- `models/xgboost_horizon_120_v1.0.0.json`
- `models/ensemble_config_v1.0.0.json`

**Training Flow:**
1. ✅ Load preprocessed features
2. ✅ Split into train/val/test (70/15/15)
3. ✅ Create LSTM sequences (24 x 15-min buckets)
4. ✅ Train BiLSTM with dropout
5. ✅ Train 4 XGBoost models (one per horizon)
6. ✅ Create 60% LSTM + 40% XGBoost ensemble
7. ✅ Evaluate on test set

**Expected Results:**
- MAPE: < 15% (target)
- RMSE: < 5 requests per 15-min
- p90 Error: < 8 requests

---

### Phase 4: Model Evaluation

Detailed evaluation metrics by:
- Time of day (hourly performance)
- Demand level (high/medium/low)
- Zone performance (top/bottom zones)
- Horizon accuracy (15/30/60/120 min)

```python
# In Python
from src.evaluation.metrics import ValidationMetrics

metrics_hourly = ValidationMetrics.metrics_by_time_of_day(y_true, y_pred, hours)
metrics_demand = ValidationMetrics.metrics_by_demand_level(y_true, y_pred, bins=5)
```

---

### Phase 5: API Deployment

#### 5.1 Start API Server
```bash
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```

#### 5.2 Test Health Check
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "demand_forecasting",
  "models_loaded": {
    "lstm": true,
    "xgboost": true,
    "config": true
  }
}
```

#### 5.3 Get Predictions
```bash
curl -X POST http://localhost:8000/api/v1/demand/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "zones": ["9q8zw", "9q9a"],
    "horizons": [15, 30, 60, 120],
    "include_confidence_intervals": true
  }'
```

**Response:**
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "prediction_timestamp": "2024-01-15T10:30:00.000Z",
  "zones": [
    {
      "zone_id": "9q8zw",
      "horizons": [
        {
          "minutes_ahead": 15,
          "predicted_requests": 12.5,
          "confidence_score": 0.85,
          "lower_bound": 10.0,
          "upper_bound": 15.0
        }
      ]
    }
  ],
  "model_version": "v1.0.0"
}
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_features.py -v
```

---

## Project Structure

```
demand-forecasting-model/
├── data/                    # Data directory (not in git)
│   ├── raw/                 # Downloaded from warehouse
│   ├── processed/           # Feature engineering output
│   └── time_based_indices/  # Seasonal indices
│
├── src/                     # Source code
│   ├── data/               # Data loading
│   ├── features/           # Feature engineering
│   ├── models/             # Model architectures
│   ├── training/           # Training loops
│   ├── evaluation/         # Metrics
│   └── serving/            # API
│
├── scripts/                # Standalone executables
│   ├── download_data.py    # Download from warehouse
│   ├── preprocess.py       # Feature engineering
│   └── train.py            # Model training
│
├── models/                 # Saved models (not in git)
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks
├── logs/                   # Application logs
├── config.yaml             # Configuration
└── requirements.txt        # Dependencies
```

---

## Troubleshooting

### Issue: Missing BigQuery Credentials
**Solution:** Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

### Issue: Out of Memory During Training
**Solution:** Reduce batch size in `config.yaml`
```yaml
training:
  batch_size: 16  # Reduced from 32
```

### Issue: API Port Already in Use
**Solution:** Use different port
```bash
uvicorn src.serving.api:app --port 8001
```

---

## Next Steps

1. ✅ Setup local environment
2. ✅ Download data from warehouse
3. ✅ Run feature engineering
4. ✅ Train models
5. ✅ Evaluate performance
6. ✅ Deploy API
7. Install to staging environment
8. Run load tests
9. Setup monitoring
10. A/B test in production

---

## Documentation

- **Architecture:** See [ML_ARCHITECTURE_ROADMAP.md](../ML_ARCHITECTURE_ROADMAP.md)
- **Development Timeline:** See [DEVELOPMENT_GUIDE.md](../DEVELOPMENT_GUIDE.md)
- **Phase Checklist:** See [PHASE_1_EXECUTION_CHECKLIST.md](../PHASE_1_EXECUTION_CHECKLIST.md)

---

## Support

For issues, questions, or contributions:
- Create an issue on the project repo
- Contact: ml-team@urbanblack.com
- Slack: #demand-forecasting
