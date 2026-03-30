# Demand Forecasting Model

A machine learning model that predicts ride demand across Urban Black's service zones using LSTM and XGBoost ensemble methods.

## Overview

- **Objective**: Predict ride demand 15, 30, 60, and 120 minutes ahead by zone
- **Architecture**: BiLSTM + XGBoost ensemble with temporal and supply features
- **Data**: 6 months of historical ride data from Urban Black
- **Accuracy**: Target MAPE < 15% across all horizons

## Project Structure

```
demand-forecasting-model/
├── data/                      # Raw and processed data
│   ├── raw/                   # Downloaded from data warehouse
│   ├── processed/             # Feature engineering output
│   └── time_based_indices/    # Seasonal indices from Model 2
├── src/                       # Source code
│   ├── data/                  # Data loading & warehouse connection
│   ├── features/              # Feature engineering pipeline
│   ├── models/                # Model architectures
│   ├── training/              # Training loops & callbacks
│   ├── evaluation/            # Metrics & diagnostics
│   └── serving/               # REST API
├── notebooks/                 # Jupyter notebooks for EDA & exploration
├── scripts/                   # Standalone scripts
├── models/                    # Saved model artifacts
├── tests/                     # Unit tests
└── docker/                    # Docker configuration
```

## Quick Start

### 1. Setup Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Create Data Directories

```bash
mkdir -p data/{raw,processed,time_based_indices}
mkdir -p models logs
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 4. Download Data

```bash
python scripts/download_data.py
```

### 5. Run Training Pipeline

```bash
python scripts/train.py
```

### 6. Start API Server

```bash
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```

## Development Timeline

- **Phase 1 (Days 1-3)**: Data preparation & EDA
- **Phase 2 (Days 4-6)**: Feature engineering
- **Phase 3 (Days 7-14)**: Model training (LSTM + XGBoost)
- **Phase 4 (Days 15-17)**: Evaluation & diagnostics
- **Phase 5 (Days 18-20)**: API development
- **Phase 6 (Days 21+)**: Deployment & monitoring

## Key Features

- Multi-horizon demand forecasting (15/30/60/120 min)
- Temporal features (hour, day-of-week, seasonality)
- Supply context (active drivers per zone)
- Lag features for demand patterns
- Ensemble predictions (LSTM + XGBoost)
- REST API for real-time predictions
- Comprehensive evaluation metrics

## Model Architecture

### LSTM Component
- BiLSTM with 64 units (L1), 32 units (L2)
- 6-hour lookback window (24 x 15-min buckets)
- Features: lag, rolling statistics, temporal features
- Dropout: 0.22

### XGBoost Component
- One model per horizon (15/30/60/120 min)
- Max depth: 7, Learning rate: 0.1
- Trained on expanded feature set
- Early stopping with validation set

### Ensemble
- Weighted combination: 60% LSTM + 40% XGBoost
- Output: Point predictions + confidence intervals

## API Endpoints

### Forecast Demand
```bash
POST /api/v1/demand/forecast

Request:
{
  "zones": ["9q8zw", "9q9a"],
  "horizons": [15, 30, 60, 120],
  "include_confidence_intervals": true
}

Response:
{
  "request_id": "uuid",
  "prediction_timestamp": "2024-01-15T10:30:00Z",
  "zones": [
    {
      "zone_id": "9q8zw",
      "horizons": [
        {
          "minutes_ahead": 15,
          "predicted_requests": 12.5,
          "confidence_score": 0.85
        }
      ]
    }
  ],
  "model_version": "1.0.0"
}
```

### Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "model": "1.0.0"
}
```

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src  # With coverage report
```

## Configuration

Edit `config.yaml` to adjust:
- Model hyperparameters
- Training epochs and batch size
- Feature engineering parameters
- Ensemble weights
- API configuration

## Monitoring

Prometheus metrics available at:
```
http://localhost:8001/metrics
```

## Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Run tests before committing: `pytest tests/`
3. Format code: `black src/`
4. Submit pull request

## License

Urban Black © 2024

## Support

For issues & questions, contact the ML team.
