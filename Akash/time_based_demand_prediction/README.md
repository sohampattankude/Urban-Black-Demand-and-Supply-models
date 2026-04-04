# Time-Based Demand Prediction Pipeline

## Overview
This repository contains the authoritative **Time-Based Demand Prediction** engine for the Urban Black platform. Designed as a standalone machine learning architecture, it predicts highly localized, time-specific taxi rider demand. By analyzing real-time historically authenticated ride data, this module empowers proactive driver repositioning, minimizes idle time, and maximizes booking fulfillment rates.

## Core Architecture & Workflow
1. **Data Ingestion (PostgreSQL & Kafka):** 
   The pipeline securely interfaces with the `urbanblack_ride` backend PostgreSQL database, safely extracting `COMPLETED` ride telemetry that is continuously streamed via Apache Kafka events.
   
2. **Spatiotemporal Clustering (K-Means):** 
   Raw GPS coordinates (`pickupLat`, `pickupLng`) are dynamically clustered into regional zones and mathematically mapped via the **Google Maps Reverse Geocoding API** to generate human-readable localities (e.g., "Manhattan", "Hinjewadi").

3. **Predictive Modeling (XGBoost):** 
   An `XGBRegressor` ensemble analyzes combined spatiotemporal features (Zone ID, Hour, Day of Week) to learn predictive curves, intelligently identifying recurring organic rider demand combinations.

4. **Automated Fine-Tuning (GridSearchCV):** 
   The architecture integrates an automated hyperparameter tuning loop. By executing cross-validation across combinations of `learning_rate`, `max_depth`, and `n_estimators`, it mathematically drives down the Mean Absolute Error (MAE) for high-precision deployment.

5. **Heuristic JSON Payload Generation:** 
   Upon each offline retraining cycle, the module seamlessly outputs a highly structured `demand_patterns.json` state payload. This optimized artifact serves as the direct intelligence bridge for downstream Demand Forecasting and Supply Allocation Spring Boot microservices.

## File Structure
*   `src/train.py`: The foundational machine learning script responsible for baseline training on proxy/dummy datasets (e.g., historical NYC Uber data) to establish the initial `.pkl` weights and JSON mappings before production deployment.
*   `src/retrain.py`: The production-grade live retraining engine. Once the backend actively records organic rider requests, this engine connects directly to the PostgreSQL `urbanblack_ride` database to download the real-time continuous data. When triggered by a Server Cron Job, it automatically re-runs the XGBoost algorithms over the new reality, replaces the `.pkl` files, and pushes customized JSON intelligence drops automatically.
*   `src/predict.py`: The real-time inference engine. Ingests localized driver heartbeat pings (`lat`, `lng`, `updatedAt`) to yield lightning-fast instantaneous demand volume forecasts.
*   `src/app.py`: The production REST API service. Built using **FastAPI**, it provides a real-time bridge for the backend to query demand predictions and model health.
*   `src/predict.py`: The local inference utility script.
*   `src/eda_visualizer.py`: The Exploratory Data Analysis plotting utility. Processes the database tables to render visual PNG maps (e.g. Demand Heatmaps, Spatiotemporal distributions, Hourly trends) for executive dashboard review.
*   `outputs/plots/*.png`: Generated visual representations of the dataset demand curves.
*   `outputs/demand_patterns.json`: The live heuristic intelligence matrix containing evaluation metrics (MAE, RMSE) and predictive zone configurations.
*   `models/*.pkl`: The serialized intelligence weights (KMeans Mapping and XGBoost Regressor).

## Setup Instructions
1. Install strictly defined module requirements via the root directory:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure the backend PostgreSQL database is active at `localhost:5432` and populated with a `rides` table schema.
3. Validate that a live Google Maps SDK API key is actively provisioned.

## Execution
*   **Run Baseline Training:** 
    ```bash
    python src/train.py
    ```
*   **Start the Production AI API:**
    ```bash
    python src/app.py
    ```
*   **Verified Performance Metrics:**
    - **R2 Score:** 0.9874 (High Precision)
    - **MAE:** ~8.35 rides
*   **API Endpoints (Port 8001):**
    - `GET /`: Interactive Welcome Dashboard.
    - `GET /health`: Model health status (Standard Heartbeat).
    - `GET /predict?lat=X&lon=Y`: Real-time demand volume prediction.
    - `GET /indices`: Intelligence drop for forecasting models.
    - `GET /docs`: Interactive API Documentation (Swagger UI).

*   **Generate Dashboard EDA Plots:**
    ```bash
    python src/eda_visualizer.py
    ```
*   **Trigger Market Retraining (Pune Sync):** 
    ```bash
    python src/retrain.py
    ```

---
*Engineered as part of the Urban Black Demand & Supply Models Infrastructure.*
