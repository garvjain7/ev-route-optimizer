# import psycopg2
# import pandas as pd
# import datetime
# import requests
# import json
# import os

# # ---- Configuration ----
# DB_CONFIG = {
#     'host': 'localhost',
#     'port': 5433,
#     'user': 'postgres',
#     'password': 'geodata@server',
#     'dbname': 'geodata_db'
# }

# FLASK_URL = "http://127.0.0.1:5000"
# JSON_DUMP_PATH = "test_data/ev_station_dump.json"

# # Ensure directory exists
# os.makedirs(os.path.dirname(JSON_DUMP_PATH), exist_ok=True)

# # ---- Step 1: Load EV station data from PostgreSQL ----
# def fetch_ev_stations():
#     try:
#         conn = psycopg2.connect(**DB_CONFIG)
#         query = "SELECT * FROM ev_station;"
#         df = pd.read_sql_query(query, conn)
#         conn.close()
#         print(f"[DB] Fetched {len(df)} records from database.")

#         # Drop datetime-related columns to avoid JSON issues
#         cols_to_drop = []
#         for col in df.columns:
#             if df[col].apply(lambda x: isinstance(x, (datetime.date, datetime.datetime, pd.Timestamp))).any():
#                 cols_to_drop.append(col)

#         if cols_to_drop:
#             print(f"[INFO] Dropping datetime columns: {cols_to_drop}")
#             df = df.drop(columns=cols_to_drop)

#         return df

#     except Exception as e:
#         print(f"[ERROR] Failed to fetch data from DB: {e}")
#         return pd.DataFrame()

# # ---- Step 2: Export to JSON ----
# # def save_to_json(df, file_path=JSON_DUMP_PATH):
# #     try:
# #         stations_json = df.to_dict(orient='records')
# #         with open(file_path, 'w', encoding='utf-8') as f:
# #             json.dump(stations_json, f, indent=2)
# #         print(f"[FILE] Saved stations_data to '{file_path}'")
# #     except Exception as e:
# #         print(f"[ERROR] Failed to write JSON file: {e}")

# # ---- Step 3: Inject into Flask backend ----
# def send_data_to_flask(df):
#     stations_json = df.to_dict(orient='records')
#     res = requests.post(f"{FLASK_URL}/api/dev/set_stations", json={"stations_data": stations_json})
#     try:
#         res.raise_for_status()
#         print("[SET STATIONS]", res.json())
#         return True
#     except Exception as e:
#         print(f"[ERROR] Failed to inject stations_data: {e}")
#         print("Response:", res.text)
#         return False
    
# def enrich_station_data(df):
#     if 'ac_ports' in df.columns and 'dc_fast_ports' in df.columns:
#         df['charging_type'] = df.apply(
#             lambda row: 'DC' if row['dc_fast_ports'] > 0 else 'AC', axis=1
#         )
#     else:
#         df['charging_type'] = 'AC'  # fallback

#     df['power_level'] = df['charging_type'].map({
#         'AC': 'Level2',
#         'DC': 'DC_Fast'
#     }).fillna('Level2')

#     if 'rating' not in df.columns:
#         df['rating'] = 4.0

#     if 'predicted_congestion' not in df.columns:
#         df['predicted_congestion'] = 1  # medium congestion

#     if 'ml_station_score' not in df.columns:
#         df['ml_station_score'] = 0.75

#     if 'predicted_rating' not in df.columns:
#         df['predicted_rating'] = 4.1

#     return df
# # stations_df = enrich_station_data(stations_df)



# # ---- Step 4: Apply filtering ----
# def apply_filtering():
#     payload = {
#         "source_lat": 26.9,
#         "source_lon": 75.8,
#         "dest_lat": 27.1,
#         "dest_lon": 76.1,
#         "filtering_method": "Smart ML Filtering",
#         "params": {
#             "prefer_fast_charging": False,
#             "detour_factor": 3.0,
#             "corridor_width": 100,
#             "min_station_rating": 0.0,
#             "avoid_congestion": False,
#             "charging_type": "AC", 
#             "power_level": "Level2"
#         }
#     }

#     print("[FILTERING] Sending request with filtering params...")
#     res = requests.post(f"{FLASK_URL}/api/apply_filtering", json=payload)
#     try:
#         res.raise_for_status()
#         response_json = res.json()
#         print("[FILTERING RESPONSE]", response_json)
#         return response_json
#     except Exception as e:
#         print(f"[ERROR] Filtering failed: {e}")
#         print("Response:", res.text)
#         return None

# # ---- Step 5: Clustering ----
# def perform_clustering(n_clusters=5):
#     payload = {"n_clusters": n_clusters}
#     res = requests.post(f"{FLASK_URL}/api/perform_clustering", json=payload)
#     try:
#         res.raise_for_status()
#         response_json = res.json()
#         print("[CLUSTERING RESPONSE]", response_json)
#         return response_json
#     except Exception as e:
#         print(f"[ERROR] Clustering failed: {e}")
#         print("Response:", res.text)
#         return None

# # ---- Main Flow ----
# if __name__ == "__main__":
#     print("[START] Fetching stations from DB...")
#     df = fetch_ev_stations()

#     if df.empty:
#         print("[EXIT] No stations data found in DB.")
#     else:
#         # save_to_json(df)  # Save for future reuse

#         if send_data_to_flask(df):
#             filtering_result = apply_filtering()
#             if filtering_result and filtering_result.get("success"):
#                 perform_clustering(n_clusters=5)
#             else:
#                 print("[EXIT] Filtering failed or returned no results.")
#         else:
#             print("[EXIT] Failed to send data to Flask.")
import psycopg2
import pandas as pd
import datetime
import requests
import json
import os
import numpy as np
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# ---- Configuration ----
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'user': 'postgres',
    'password': 'geodata@server',
    'dbname': 'geodata_db'
}

FLASK_URL = "http://127.0.0.1:5000"
JSON_DUMP_PATH = "test_data/ev_station_dump.json"

# Ensure directory exists
os.makedirs(os.path.dirname(JSON_DUMP_PATH), exist_ok=True)

# ---- Step 1: Load EV station data from PostgreSQL ----
def fetch_ev_stations():
    try:
        # Method 1: Try with SQLAlchemy (with proper URL encoding)
        try:
            # URL encode the password to handle special characters
            encoded_password = quote_plus(DB_CONFIG['password'])
            connection_string = f"postgresql://{DB_CONFIG['user']}:{encoded_password}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
            engine = create_engine(connection_string)
            
            query = "SELECT * FROM ev_station;"
            df = pd.read_sql_query(query, engine)
            print(f"[DB] Fetched {len(df)} records from database using SQLAlchemy.")
            
        except Exception as sqlalchemy_error:
            print(f"[INFO] SQLAlchemy connection failed: {sqlalchemy_error}")
            print("[INFO] Falling back to direct psycopg2 connection...")
            
            # Method 2: Fallback to direct psycopg2 connection
            conn = psycopg2.connect(**DB_CONFIG)
            query = "SELECT * FROM ev_station;"
            df = pd.read_sql_query(query, conn)
            conn.close()
            print(f"[DB] Fetched {len(df)} records from database using psycopg2.")

        # Drop datetime-related columns to avoid JSON issues
        cols_to_drop = []
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (datetime.date, datetime.datetime, pd.Timestamp))).any():
                cols_to_drop.append(col)

        if cols_to_drop:
            print(f"[INFO] Dropping datetime columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        return df

    except Exception as e:
        print(f"[ERROR] Failed to fetch data from DB: {e}")
        print(f"[DEBUG] DB Config: host={DB_CONFIG['host']}, port={DB_CONFIG['port']}, user={DB_CONFIG['user']}, dbname={DB_CONFIG['dbname']}")
        return pd.DataFrame()

# ---- Step 2: Data Enrichment and Validation ----
def enrich_station_data(df):
    """Enrich station data with required fields and validate data quality"""
    
    # Ensure required columns exist
    required_columns = ['lat', 'lng', 'name']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"[ERROR] Missing required columns: {missing_columns}")
        return df
    
    # Remove rows with null coordinates
    initial_count = len(df)
    df = df.dropna(subset=['lat', 'lng'])
    print(f"[INFO] Removed {initial_count - len(df)} rows with null coordinates")
    
    # Validate coordinate ranges
    df = df[(df['lat'].between(-90, 90)) & (df['lng'].between(-180, 180))]
    print(f"[INFO] {len(df)} rows remaining after coordinate validation")
    
    # Add charging type based on available data
    if 'ac_ports' in df.columns and 'dc_fast_ports' in df.columns:
        df['charging_type'] = df.apply(
            lambda row: 'DC' if pd.notna(row['dc_fast_ports']) and row['dc_fast_ports'] > 0 else 'AC', 
            axis=1
        )
    else:
        df['charging_type'] = 'AC'  # fallback

    # Add power level
    df['power_level'] = df['charging_type'].map({
        'AC': 'Level2',
        'DC': 'DC_Fast'
    }).fillna('Level2')

    # Add default values for missing fields
    if 'rating' not in df.columns:
        df['rating'] = 4.0

    if 'predicted_congestion' not in df.columns:
        df['predicted_congestion'] = 1  # medium congestion

    if 'ml_station_score' not in df.columns:
        df['ml_station_score'] = 0.75

    if 'predicted_rating' not in df.columns:
        df['predicted_rating'] = 4.1
    
    # Ensure numeric fields are properly typed
    numeric_fields = ['lat', 'lng', 'rating', 'predicted_congestion', 'ml_station_score', 'predicted_rating']
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce')
    
    # Fill any remaining NaN values
    df = df.fillna({
        'rating': 4.0,
        'predicted_congestion': 1,
        'ml_station_score': 0.75,
        'predicted_rating': 4.1
    })
    
    print(f"[INFO] Data enrichment complete. Final dataset has {len(df)} rows")
    print(f"[INFO] Charging types: {df['charging_type'].value_counts().to_dict()}")
    print(f"[INFO] Coordinate ranges: Lat({df['lat'].min():.3f}, {df['lat'].max():.3f}), Lon({df['lng'].min():.3f}, {df['lng'].max():.3f})")
    
    return df

# ---- Step 3: Export to JSON ----
def save_to_json(df, file_path=JSON_DUMP_PATH):
    try:
        stations_json = df.to_dict(orient='records')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(stations_json, f, indent=2)
        print(f"[FILE] Saved {len(stations_json)} stations to '{file_path}'")
    except Exception as e:
        print(f"[ERROR] Failed to write JSON file: {e}")

# ---- Step 4: Inject into Flask backend ----
def send_data_to_flask(df):
    try:
        stations_json = df.to_dict(orient='records')
        
        # Convert any numpy types to native Python types
        for station in stations_json:
            for key, value in station.items():
                if isinstance(value, (np.integer, np.floating)):
                    station[key] = value.item()
                elif pd.isna(value):
                    station[key] = None
        
        res = requests.post(f"{FLASK_URL}/api/dev/set_stations", json={"stations_data": stations_json})
        res.raise_for_status()
        print("[SET STATIONS]", res.json())
        return True
    except Exception as e:
        print(f"[ERROR] Failed to inject stations_data: {e}")
        if 'res' in locals():
            print("Response:", res.text)
        return False

# ---- Step 5: Apply filtering with dynamic parameters ----
def apply_filtering(df):
    """Apply filtering with parameters based on actual data"""
    
    # Calculate reasonable bounds based on actual data
    lat_center = df['lat'].mean()
    lon_center = df['lng'].mean()
    lat_range = df['lat'].max() - df['lat'].min()
    lon_range = df['lng'].max() - df['lng'].min()
    
    # Create source and destination within the data bounds
    source_lat = lat_center - lat_range * 0.2
    source_lon = lon_center - lon_range * 0.2
    dest_lat = lat_center + lat_range * 0.2
    dest_lon = lon_center + lon_range * 0.2
    
    print(f"[INFO] Using route: ({source_lat:.3f}, {source_lon:.3f}) -> ({dest_lat:.3f}, {dest_lon:.3f})")
    
    payload = {
        "source_lat": source_lat,
        "source_lon": source_lon,
        "dest_lat": dest_lat,
        "dest_lon": dest_lon,
        "filtering_method": "Smart ML Filtering",
        "params": {
            "prefer_fast_charging": False,
            "detour_factor": 5.0,  # Increased to be more permissive
            "corridor_width": 200,  # Increased corridor width
            "min_station_rating": 0.0,
            "avoid_congestion": False,
            "charging_type": "AC",  # Use the most common type
            "power_level": "Level2"
        }
    }

    print("[FILTERING] Sending request with filtering params...")
    try:
        res = requests.post(f"{FLASK_URL}/api/apply_filtering", json=payload)
        res.raise_for_status()
        response_json = res.json()
        print("[FILTERING RESPONSE]", response_json)
        return response_json
    except Exception as e:
        print(f"[ERROR] Filtering failed: {e}")
        if 'res' in locals():
            print("Response:", res.text)
        return None

# ---- Step 6: Clustering ----
def perform_clustering(n_clusters=5):
    payload = {"n_clusters": n_clusters}
    try:
        res = requests.post(f"{FLASK_URL}/api/perform_clustering", json=payload)
        res.raise_for_status()
        response_json = res.json()
        print("[CLUSTERING RESPONSE]", response_json)
        return response_json
    except Exception as e:
        print(f"[ERROR] Clustering failed: {e}")
        if 'res' in locals():
            print("Response:", res.text)
        return None

# ---- Alternative: Test with broader parameters ----
def test_with_broader_filters(df):
    """Test filtering with very broad parameters to ensure some stations are found"""
    
    # Get the full bounds of the data
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lon_min, lon_max = df['lng'].min(), df['lng'].max()
    
    payload = {
        "source_lat": lat_min,
        "source_lon": lon_min,
        "dest_lat": lat_max,
        "dest_lon": lon_max,
        "filtering_method": "Smart ML Filtering",
        "params": {
            "prefer_fast_charging": False,
            "detour_factor": 10.0,  # Very permissive
            "corridor_width": 500,  # Very wide corridor
            "min_station_rating": 0.0,
            "avoid_congestion": False,
            "charging_type": "AC",
            "power_level": "Level2"
        }
    }

    print("[BROAD FILTERING] Testing with very broad parameters...")
    try:
        res = requests.post(f"{FLASK_URL}/api/apply_filtering", json=payload)
        res.raise_for_status()
        response_json = res.json()
        print("[BROAD FILTERING RESPONSE]", response_json)
        return response_json
    except Exception as e:
        print(f"[ERROR] Broad filtering failed: {e}")
        if 'res' in locals():
            print("Response:", res.text)
        return None

# ---- Main Flow ----
if __name__ == "__main__":
    print("[START] Fetching stations from DB...")
    df = fetch_ev_stations()

    if df.empty:
        print("[EXIT] No stations data found in DB.")
    else:
        # Enrich and validate data
        df = enrich_station_data(df)
        
        if df.empty:
            print("[EXIT] No valid stations after data validation.")
        else:
            # Save enriched data for debugging
            save_to_json(df)
            
            # Send data to Flask
            if send_data_to_flask(df):
                # Try normal filtering first
                filtering_result = apply_filtering(df)
                
                # If normal filtering fails, try with broader parameters
                if not filtering_result or not filtering_result.get("success"):
                    print("[RETRY] Trying with broader filter parameters...")
                    filtering_result = test_with_broader_filters(df)
                
                # Proceed with clustering if filtering succeeded
                if filtering_result and filtering_result.get("success"):
                    perform_clustering(n_clusters=5)
                else:
                    print("[EXIT] All filtering attempts failed.")
            else:
                print("[EXIT] Failed to send data to Flask.")