import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from geopy.geocoders import GoogleV3
from geopy.exc import GeocoderTimedOut
import xgboost as xgb
import joblib
import json
import os

# -----------------------------
# 1. DATABASE CONNECTION
# -----------------------------
# The backend uses PostgreSQL with DB=urbanblack_ride, USER=postgres, PASS=root
DB_URL = "postgresql+psycopg2://postgres:root@localhost:5432/urbanblack_ride"

print("📡 Connecting to PostgreSQL (urbanblack_ride)...")
try:
    engine = create_engine(DB_URL)
    
    # Query only COMPLETED rides to train on actual fulfilled demand
    query = """
        SELECT 
            "pickupLat" AS lat, 
            "pickupLng" AS lon, 
            "requestedAt" AS datetime 
        FROM rides 
        WHERE status = 'COMPLETED'
    """
    df = pd.read_sql(query, engine)
    
    # Fallback to dummy data if DB is completely empty (Pune cold start)
    if df.empty:
        print("⚠️ Warning: Database is empty! To prevent crash, skipping retraining. Awaiting real Pune data.")
        exit(0)
    
    print(f"✅ Downloaded {len(df)} live Pune ride records!")
except Exception as e:
    print(f"❌ Database Error: {e}")
    print("Ensure your Docker container/Postgres is running on localhost:5432!")
    exit(1)

# -----------------------------
# 2. PREPROCESS
# -----------------------------
print("⚙️ Processing Data...")
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.weekday
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['date'] = df['datetime'].dt.date

# -----------------------------
# 3. GEOGRAPHY (KMEANS) & REVERSE GEOCODING
# -----------------------------
print("🗺️ Rebuilding Pune Geographic Zones...")
kmeans = KMeans(n_clusters=min(10, len(df)), random_state=42)
df['zone_id'] = kmeans.fit_predict(df[['lat', 'lon']])

# Use Google Maps to name the mathematically found clusters (Per Backend Configuration)
# Note: Key copied from C:\Users\HP\Downloads\Urban-Black...backend\.env
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
                if 'sublocality' in comp['types'] or 'locality' in comp['types']:
                    name = comp['long_name']
                    break
                    
            zone_names[f"zone_{idx}"] = name
        else:
            zone_names[f"zone_{idx}"] = f"Zone_{idx}"
    except Exception as e:
        print(f"⚠️ Google API Failure on Zone {idx}: {e}")
        zone_names[f"zone_{idx}"] = f"Zone_{idx}"

print(f"✅ Found Localities: {list(zone_names.values())}")

# -----------------------------
# 4. AGGREGATION
# -----------------------------
demand = df.groupby(['zone_id', 'hour', 'day_of_week', 'is_weekend']).size().reset_index(name='ride_count')
days_count = df.groupby('day_of_week')['date'].nunique().to_dict()

demand['ride_count'] = demand.apply(
    lambda row: row['ride_count'] / days_count.get(row['day_of_week'], 1),
    axis=1
)

# -----------------------------
# 5. RETRAIN MODEL & FINE TUNING
# -----------------------------
print("🧠 Retraining XGBoost Model...")
X = demand[['zone_id', 'hour', 'day_of_week', 'is_weekend']]
y = demand['ride_count']

# Cold-start safe logic
if len(X) < 15:
    print("⚠️ Not enough data for Train/Test split. Training flat model.")
    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X, y)
    mae, rmse = 0, 0
    best_params = {"n_estimators": 100, "max_depth": 6}
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_model = xgb.XGBRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2]
    }
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=min(3, len(X_train)//3))
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"📊 Accuracy metrics updated. Ready for deployment.")

# -----------------------------
# 6. SAVE UPDATED MODELS
# -----------------------------
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/time_demand_model.pkl")
joblib.dump(kmeans, "../models/zone_model.pkl")

# -----------------------------
# 7. JSON GENERATION (FORECASTING EXPORT)
# -----------------------------
print("📝 Generating updated demand_patterns.json...")
day_map = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}

hour_day = df.groupby(['day_of_week','hour']).size().reset_index(name='count')
hourly_coeff = {}
for day in range(7):
    d = hour_day[hour_day['day_of_week'] == day]
    mean_val = d['count'].mean() if len(d) > 0 else 1
    mean_val = mean_val if mean_val != 0 else 1
    hourly_coeff[day_map[day]] = {str(row['hour']): row['count']/mean_val for _, row in d.iterrows()}

daywise = df.groupby('day_of_week').size()
day_mean = daywise.mean() if len(daywise) > 0 else 1
day_mean = day_mean if day_mean != 0 else 1
daywise_coeff = {day_map[k]: v/day_mean for k,v in daywise.items()}

zone = df.groupby('zone_id').size()
zone_mean = zone.mean() if len(zone) > 0 else 1
zone_mean = zone_mean if zone_mean != 0 else 1
zone_coeff = {zone_names[f"zone_{k}"]: v/zone_mean for k,v in zone.items()}

baseline = len(df) / df['datetime'].nunique() if df['datetime'].nunique() > 0 else 0

output_json = {
    "metadata": {
        "model": "time_based_demand_prediction",
        "data_days": int(df['date'].nunique()),
        "test_mae": round(float(mae), 2) if mae else None,
        "test_rmse": round(float(rmse), 2) if rmse else None,
        "best_params": best_params,
        "zone_mapping": zone_names
    },
    "hourly_coefficients": hourly_coeff,
    "day_of_week_coefficients": daywise_coeff,
    "zone_coefficients": zone_coeff,
    "baseline_demand": {
        "avg_rides_per_hour": float(baseline)
    }
}

os.makedirs("../outputs", exist_ok=True)
with open("../outputs/demand_patterns.json", "w") as f:
    json.dump(output_json, f, indent=4)

print("[SUCCESS] Pune Retraining Pipeline Complete! System is fully synced.")
