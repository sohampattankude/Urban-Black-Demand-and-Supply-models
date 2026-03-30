import joblib
from datetime import datetime
import pandas as pd

model = joblib.load("../models/time_demand_model.pkl")
kmeans = joblib.load("../models/zone_model.pkl")

# sample driver input
lat = 40.75
lon = -73.98

now = datetime.now()
hour = now.hour
day_of_week = now.weekday()
is_weekend = 1 if day_of_week >= 5 else 0

# Convert inputs to pandas DataFrames to prevent Scikit-Learn missing feature name warnings
geo_df = pd.DataFrame([[lat, lon]], columns=['lat', 'lon'])
zone = kmeans.predict(geo_df)[0]

input_df = pd.DataFrame([[zone, hour, day_of_week, is_weekend]], columns=['zone_id', 'hour', 'day_of_week', 'is_weekend'])
prediction = model.predict(input_df)

import json

try:
    with open("../outputs/demand_patterns.json", "r") as f:
        patterns = json.load(f)
        zone_name = patterns.get("metadata", {}).get("zone_mapping", {}).get(f"zone_{zone}", f"Zone {zone}")
except Exception:
    zone_name = f"Zone {zone}"

print("Zone:", zone_name, f"(ID: {zone})")
print("Time:", now)
print("Predicted Demand:", int(prediction[0]), "rides")