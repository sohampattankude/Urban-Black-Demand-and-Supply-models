from fastapi import FastAPI, HTTPException, Query
import joblib
import pandas as pd
from datetime import datetime
import os
import json

app = FastAPI(title="Urban Black Demand Prediction API")

# Load models
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/time_demand_model.pkl")
KMEANS_PATH = os.path.join(os.path.dirname(__file__), "../models/zone_model.pkl")
JSON_PATH = os.path.join(os.path.dirname(__file__), "../outputs/demand_patterns.json")

try:
    model = joblib.load(MODEL_PATH)
    kmeans = joblib.load(KMEANS_PATH)
except Exception as e:
    print(f"Error loading models: {e}")
    model, kmeans = None, None

@app.get("/health")
def health_check():
    """Service status heartbeat for production monitoring."""
    if model is None or kmeans is None:
        return {"status": "unhealthy", "error": "Models not loaded"}
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/predict")
def predict_demand(
    lat: float = Query(..., description="Latitude of the driver heartbeat"),
    lon: float = Query(..., description="Longitude of the driver heartbeat")
):
    """Predict ride demand for a given location and current time."""
    if model is None or kmeans is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    now = datetime.now()
    hour = now.hour
    day_of_week = now.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0

    # 1. Map Lat/Lon to Zone
    geo_df = pd.DataFrame([[lat, lon]], columns=['lat', 'lon'])
    zone_id = int(kmeans.predict(geo_df)[0])

    # 2. Predict Demand
    input_df = pd.DataFrame([[zone_id, hour, day_of_week, is_weekend]], 
                           columns=['zone_id', 'hour', 'day_of_week', 'is_weekend'])
    prediction = model.predict(input_df)[0]

    # 3. Get Zone Name from JSON
    zone_name = f"Zone {zone_id}"
    try:
        if os.path.exists(JSON_PATH):
            with open(JSON_PATH, "r") as f:
                patterns = json.load(f)
                zone_name = patterns.get("metadata", {}).get("zone_mapping", {}).get(f"zone_{zone_id}", zone_name)
    except Exception:
        pass

    return {
        "zone_id": zone_id,
        "zone_name": zone_name,
        "predicted_demand": round(float(prediction), 2),
        "timestamp": now.isoformat(),
        "input_features": {
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend
        }
    }

@app.get("/indices")
def get_indices():
    """Retrieve the full demand_patterns.json for downstream models (Soham's Model)."""
    if not os.path.exists(JSON_PATH):
        raise HTTPException(status_code=404, detail="Demand patterns JSON not found")
    with open(JSON_PATH, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
