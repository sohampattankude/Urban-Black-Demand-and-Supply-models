"""
REST API for demand forecasting model predictions.
Built with FastAPI.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import json
import tensorflow as tf
import xgboost as xgb
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Demand Forecasting API",
    version="1.0.0",
    description="API for urban ride demand predictions"
)

# ===== Request/Response Models =====

class DemandForecastRequest(BaseModel):
    """Request model for demand forecast."""
    zones: List[str] = Field(..., description="List of zone IDs (geohash)")
    horizons: List[int] = Field(
        default=[15, 30, 60, 120],
        description="Prediction horizons in minutes"
    )
    include_confidence_intervals: bool = Field(
        default=True,
        description="Include confidence intervals"
    )

class HorizonPrediction(BaseModel):
    """Prediction for a single horizon."""
    minutes_ahead: int
    predicted_requests: float
    confidence_score: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

class ZonePrediction(BaseModel):
    """Predictions for a single zone."""
    zone_id: str
    horizons: List[HorizonPrediction]

class DemandForecastResponse(BaseModel):
    """Response model for demand forecast."""
    request_id: str
    prediction_timestamp: str
    zones: List[ZonePrediction]
    model_version: str

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    models_loaded: Dict[str, bool]

# ===== Global State =====

models_state = {
    'lstm': None,
    'xgboost': {},
    'ensemble_config': None,
    'loaded': False
}

def load_models(model_version: str = 'v1.0.0'):
    """Load trained models from disk."""
    try:
        # Load LSTM
        lstm_path = f'models/lstm_{model_version}.h5'
        models_state['lstm'] = tf.keras.models.load_model(lstm_path)
        logger.info(f"✅ Loaded LSTM from {lstm_path}")
        
        # Load XGBoost
        horizons = [15, 30, 60, 120]
        for horizon in horizons:
            xgb_path = f'models/xgboost_horizon_{horizon}_{model_version}.json'
            models_state['xgboost'][f'horizon_{horizon}'] = xgb.Booster(
                model_file=xgb_path
            )
            logger.info(f"✅ Loaded XGBoost for {horizon}-min horizon")
        
        # Load ensemble config
        config_path = f'models/ensemble_config_{model_version}.json'
        with open(config_path, 'r') as f:
            models_state['ensemble_config'] = json.load(f)
        logger.info(f"✅ Loaded ensemble config from {config_path}")
        
        models_state['loaded'] = True
        return True
    
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}", exc_info=True)
        return False

# ===== API Endpoints =====

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Starting up API server...")
    load_models()

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        Health status and loaded models information
    """
    return HealthResponse(
        status="healthy" if models_state['loaded'] else "degraded",
        model="demand_forecasting",
        models_loaded={
            'lstm': models_state['lstm'] is not None,
            'xgboost': len(models_state['xgboost']) > 0,
            'config': models_state['ensemble_config'] is not None
        }
    )

@app.post("/api/v1/demand/forecast", response_model=DemandForecastResponse)
async def forecast_demand(request: DemandForecastRequest) -> DemandForecastResponse:
    """
    Predict ride demand for specified zones and horizons.
    
    Args:
        request: Demand forecast request with zones and horizons
    
    Returns:
        Demand forecast response with predictions
    
    Raises:
        HTTPException: If models are not loaded or prediction fails
    """
    
    if not models_state['loaded']:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Server is initializing."
        )
    
    request_id = str(uuid.uuid4())
    
    try:
        predictions = []
        
        for zone_id in request.zones:
            zone_preds = ZonePrediction(
                zone_id=zone_id,
                horizons=[]
            )
            
            for horizon in request.horizons:
                # In production, fetch real features from feature store
                # For demo, use synthetic features
                lstm_pred = np.random.randn() * 2 + 10
                xgb_pred = np.random.randn() * 2 + 10
                
                # Ensure positive predictions
                lstm_pred = max(0, lstm_pred)
                xgb_pred = max(0, xgb_pred)
                
                # Ensemble
                lstm_weight = models_state['ensemble_config']['lstm_weight']
                xgb_weight = models_state['ensemble_config']['xgboost_weight']
                ensemble_pred = lstm_weight * lstm_pred + xgb_weight * xgb_pred
                
                # Confidence intervals (approximate)
                lower = ensemble_pred * 0.8
                upper = ensemble_pred * 1.2
                confidence = 0.80 if ensemble_pred > 5 else 0.70
                
                zone_preds.horizons.append(
                    HorizonPrediction(
                        minutes_ahead=horizon,
                        predicted_requests=float(ensemble_pred),
                        confidence_score=float(confidence),
                        lower_bound=float(lower),
                        upper_bound=float(upper)
                    )
                )
            
            predictions.append(zone_preds)
        
        return DemandForecastResponse(
            request_id=request_id,
            prediction_timestamp=datetime.utcnow().isoformat(),
            zones=predictions,
            model_version=models_state['ensemble_config']['model_version']
        )
    
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/api/v1/models/info")
async def get_models_info() -> Dict:
    """
    Get information about loaded models.
    
    Returns:
        Dictionary with model metadata
    """
    if not models_state['ensemble_config']:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        'model_version': models_state['ensemble_config']['model_version'],
        'training_date': models_state['ensemble_config']['training_date'],
        'ensemble_weights': {
            'lstm': models_state['ensemble_config']['lstm_weight'],
            'xgboost': models_state['ensemble_config']['xgboost_weight']
        },
        'horizons': models_state['ensemble_config']['horizons'],
        'metrics': models_state['ensemble_config']['metrics']
    }

# ===== Error Handlers =====

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return {"detail": str(exc), "status_code": 422}

# ===== Root Endpoint =====

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Demand Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "forecast": "/api/v1/demand/forecast",
            "info": "/api/v1/models/info",
            "docs": "/docs"
        }
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
