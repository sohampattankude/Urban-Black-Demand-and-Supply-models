import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def generate_eda_plots(output_dir="outputs/plots"):
    """
    Generates Exploratory Data Analysis (EDA) visualizations for the Demand Prediction model.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading proxy datasets from data/uber-raw-data-*.csv...")
    try:
        files = [
            "data/uber-raw-data-apr14.csv",
            "data/uber-raw-data-may14.csv",
            "data/uber-raw-data-jun14.csv",
            "data/uber-raw-data-jul14.csv",
            "data/uber-raw-data-aug14.csv",
            "data/uber-raw-data-sep14.csv"
        ]
        df_list = [pd.read_csv(f) for f in files]
        df = pd.concat(df_list, ignore_index=True)
        # Standardize column names
        df.columns = ['requestedAt', 'pickupLat', 'pickupLng', 'base']
    except Exception as e:
        print(f"Error loading proxy data: {e}. Please ensure data files exist.")
        return
        
    print("Extracting Temporal Features...")
    df['requestedAt'] = pd.to_datetime(df['requestedAt'])
    df['hour'] = df['requestedAt'].dt.hour
    df['day_of_week'] = df['requestedAt'].dt.day_name()
    
    # Ensure correct day order for plotting
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=days_order, ordered=True)

    # Use a professional seaborn style
    sns.set_theme(style="whitegrid", palette="muted")
    
    # --------------------------------------------------------------------------
    # Plot 1: Overall Demand by Hour of Day
    # --------------------------------------------------------------------------
    print("Generating Hourly Demand Plot...")
    plt.figure(figsize=(10, 6))
    hourly_counts = df.groupby('hour').size()
    
    sns.barplot(x=hourly_counts.index, y=hourly_counts.values, color="#4c72b0")
    plt.title('Total Ride Demand Over 24 Hours', fontsize=16, weight='bold')
    plt.xlabel('Hour of Day (0-23)', fontsize=12)
    plt.ylabel('Number of Ride Requests', fontsize=12)
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    hourly_path = os.path.join(output_dir, '1_hourly_demand.png')
    plt.savefig(hourly_path, dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # Plot 2: Demand by Day of the Week
    # --------------------------------------------------------------------------
    print("Generating Day of Week Demand Plot...")
    plt.figure(figsize=(10, 6))
    daily_counts = df.groupby('day_of_week').size()
    
    sns.barplot(x=daily_counts.index, y=daily_counts.values, palette="rocket")
    plt.title('Total Ride Demand by Day of Week', fontsize=16, weight='bold')
    plt.xlabel('Day of the Week', fontsize=12)
    plt.ylabel('Number of Ride Requests', fontsize=12)
    
    plt.tight_layout()
    daily_path = os.path.join(output_dir, '2_daily_demand.png')
    plt.savefig(daily_path, dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # Plot 3: Heatmap of Hour vs Day of the Week
    # --------------------------------------------------------------------------
    print("Generating Demand Heatmap...")
    plt.figure(figsize=(12, 8))
    
    heatmap_data = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
    
    sns.heatmap(heatmap_data, cmap="YlOrRd", annot=False, fmt="d", linewidths=.5)
    plt.title('Ride Demand Heatmap: Hour vs. Day of Week', fontsize=16, weight='bold')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Day of Week', fontsize=12)
    
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, '3_demand_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # Plot 4: Spatial Distribution (GPS Coordinates)
    # --------------------------------------------------------------------------
    print("Generating Spatial Distribution Map...")
    plt.figure(figsize=(10, 10))
    
    sns.scatterplot(x='pickupLng', y='pickupLat', data=df, alpha=0.5, s=20, color="indigo", edgecolor=None)
    plt.title('GPS Cluster Geography (Pickups)', fontsize=16, weight='bold')
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    spatial_path = os.path.join(output_dir, '4_spatial_distribution.png')
    plt.savefig(spatial_path, dpi=300)
    plt.close()

    print(f"\n✅ All EDA plots successfully generated and saved to: {os.path.abspath(output_dir)}")
    print("   -> 1_hourly_demand.png")
    print("   -> 2_daily_demand.png")
    print("   -> 3_demand_heatmap.png")
    print("   -> 4_spatial_distribution.png")

if __name__ == "__main__":
    generate_eda_plots()
