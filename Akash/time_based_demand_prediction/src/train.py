import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from geopy.geocoders import GoogleV3
import xgboost as xgb
import joblib
import json
import os

# -----------------------------
# LOAD DATA
# -----------------------------
files = [
    "../data/uber-raw-data-apr14.csv",
    "../data/uber-raw-data-may14.csv",
    "../data/uber-raw-data-jun14.csv",
    "../data/uber-raw-data-jul14.csv",
    "../data/uber-raw-data-aug14.csv",
    "../data/uber-raw-data-sep14.csv"
]

df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)

df.columns = ['datetime', 'lat', 'lon', 'base']
df = df[['datetime', 'lat', 'lon']]

# -----------------------------
# PREPROCESS
# -----------------------------
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.weekday
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['date'] = df['datetime'].dt.date

# -----------------------------
# KMEANS & REVERSE GEOCODING
# -----------------------------
kmeans = KMeans(n_clusters=10, random_state=42)
df['zone_id'] = kmeans.fit_predict(df[['lat', 'lon']])

print("🗺️ Reverse Geocoding via Google Maps API...")
GOOGLE_MAPS_API_KEY = "AIzaSyBgsKbZcmM1dG7SuIMMPBcP5Ta6_LEYgEc"
geolocator = GoogleV3(api_key=GOOGLE_MAPS_API_KEY)
zone_names = {}

for idx, center in enumerate(kmeans.cluster_centers_):
    try:
        location = geolocator.reverse(f"{center[0]}, {center[1]}", timeout=10)
        if location:
            address_components = location.raw.get('address_components', [])
            name = f"Zone_{idx}"
            for comp in address_components:
                # Prioritize neighborhood/sublocality for precise NYC results
                if 'sublocality' in comp['types'] or 'neighborhood' in comp['types'] or 'locality' in comp['types']:
                    name = comp['long_name']
                    break
            zone_names[f"zone_{idx}"] = name
        else:
            zone_names[f"zone_{idx}"] = f"Zone_{idx}"
    except Exception as e:
        print(f"⚠️ Google API Failure on Zone {idx}: {e}")
        zone_names[f"zone_{idx}"] = f"Zone_{idx}"

print(f"✅ Found NYC Localities: {list(zone_names.values())}")

# -----------------------------
# AGGREGATION
# -----------------------------
demand = df.groupby(['zone_id', 'hour', 'day_of_week', 'is_weekend']).size().reset_index(name='ride_count')

days_count = df.groupby('day_of_week')['date'].nunique().to_dict()

demand['ride_count'] = demand.apply(
    lambda row: row['ride_count'] / days_count[row['day_of_week']],
    axis=1
)

# -----------------------------
# TRAIN MODEL & FINE TUNING
# -----------------------------
X = demand[['zone_id', 'hour', 'day_of_week', 'is_weekend']]
y = demand['ride_count']

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Starting Hyperparameter Tuning (this may take a moment)...")
xgb_model = xgb.XGBRegressor(random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2]
}

# Grid Search with 3-fold cross-validation
grid_search = GridSearchCV(
    estimator=xgb_model, 
    param_grid=param_grid, 
    scoring='neg_mean_absolute_error', 
    cv=3, 
    verbose=1, 
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Get the best model
model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate the model on test data
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Model Evaluation Metrics on Test Set (20%):")
print(f" - MAE:  {mae:.2f} rides")
print(f" - RMSE: {rmse:.2f} rides")

# -----------------------------
# SAVE MODELS
# -----------------------------
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/time_demand_model.pkl")
joblib.dump(kmeans, "../models/zone_model.pkl")

# -----------------------------
# JSON GENERATION
# -----------------------------
day_map = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}

# Hourly (day-wise)
hour_day = df.groupby(['day_of_week','hour']).size().reset_index(name='count')
hourly_coeff = {}

for day in range(7):
    d = hour_day[hour_day['day_of_week'] == day]
    mean_val = d['count'].mean()
    hourly_coeff[day_map[day]] = {str(row['hour']): row['count']/mean_val for _, row in d.iterrows()}

# Day coeff
daywise = df.groupby('day_of_week').size()
daywise_coeff = {day_map[k]: v/daywise.mean() for k,v in daywise.items()}

# Zone coeff
zone = df.groupby('zone_id').size()
zone_mean = zone.mean()
zone_coeff = {zone_names[f"zone_{k}"]: v/zone_mean for k,v in zone.items()}

# Baseline
baseline = len(df) / df['datetime'].nunique()

output_json = {
    "metadata": {
        "model": "time_based_demand_prediction",
        "data_days": int(df['date'].nunique()),
        "test_mae": round(mae, 2),
        "test_rmse": round(rmse, 2),
        "best_params": grid_search.best_params_,
        "zone_mapping": zone_names
    },
    "hourly_coefficients": hourly_coeff,
    "day_of_week_coefficients": daywise_coeff,
    "zone_coefficients": zone_coeff,
    "baseline_demand": {
        "avg_rides_per_hour": baseline
    }
}

os.makedirs("../outputs", exist_ok=True)
with open("../outputs/demand_patterns.json", "w") as f:
    json.dump(output_json, f, indent=4)

print("[SUCCESS] Training + JSON complete")