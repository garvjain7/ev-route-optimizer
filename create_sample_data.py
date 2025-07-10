import pandas as pd
import numpy as np
from database.db_connection import DatabaseConnection
import streamlit as st

def create_sample_ev_stations():
    """Create sample EV charging station data for testing"""
    
    # Sample data for major US cities and highways
    sample_stations = []
    
    # Major cities with EV stations
    cities = [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060, "stations": 50},
        {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437, "stations": 60},
        {"name": "Chicago", "lat": 41.8781, "lon": -87.6298, "stations": 40},
        {"name": "Houston", "lat": 29.7604, "lon": -95.3698, "stations": 35},
        {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740, "stations": 30},
        {"name": "Philadelphia", "lat": 39.9526, "lon": -75.1652, "stations": 45},
        {"name": "San Antonio", "lat": 29.4241, "lon": -98.4936, "stations": 25},
        {"name": "San Diego", "lat": 32.7157, "lon": -117.1611, "stations": 40},
        {"name": "Dallas", "lat": 32.7767, "lon": -96.7970, "stations": 35},
        {"name": "San Jose", "lat": 37.3382, "lon": -121.8863, "stations": 45},
        {"name": "Boston", "lat": 42.3601, "lon": -71.0589, "stations": 40},
        {"name": "Seattle", "lat": 47.6062, "lon": -122.3321, "stations": 50},
        {"name": "Denver", "lat": 39.7392, "lon": -104.9903, "stations": 30},
        {"name": "Las Vegas", "lat": 36.1699, "lon": -115.1398, "stations": 25},
        {"name": "Portland", "lat": 45.5152, "lon": -122.6784, "stations": 35},
        {"name": "Atlanta", "lat": 33.7490, "lon": -84.3880, "stations": 30},
        {"name": "Miami", "lat": 25.7617, "lon": -80.1918, "stations": 35},
        {"name": "Tampa", "lat": 27.9506, "lon": -82.4572, "stations": 20},
        {"name": "Orlando", "lat": 28.5383, "lon": -81.3792, "stations": 25},
        {"name": "Nashville", "lat": 36.1627, "lon": -86.7816, "stations": 20}
    ]
    
    # Networks for realistic distribution
    networks = [
        "Tesla Supercharger", "ChargePoint", "Electrify America", 
        "EVgo", "Blink", "SemaConnect", "Volta", "Greenlots"
    ]
    
    station_id = 1
    
    for city in cities:
        for i in range(city["stations"]):
            # Add some randomness to coordinates
            lat_offset = np.random.uniform(-0.1, 0.1)
            lon_offset = np.random.uniform(-0.1, 0.1)
            
            # Station characteristics
            network = np.random.choice(networks)
            has_dc_fast = np.random.choice([True, False], p=[0.6, 0.4])
            
            station = {
                "id": station_id,
                "name": f"{city['name']} Station {i+1}",
                "latitude": city["lat"] + lat_offset,
                "longitude": city["lon"] + lon_offset,
                "network": network,
                "access": np.random.choice(["Public", "Private"], p=[0.8, 0.2]),
                "state": "Various",  # Simplified for this demo
                "city": city["name"],
                "level1_ports": np.random.randint(0, 3),
                "level2_ports": np.random.randint(2, 8),
                "dc_fast_ports": np.random.randint(1, 6) if has_dc_fast else 0,
                "fuel_type": "ELEC",
                "status": "E"  # Available
            }
            
            sample_stations.append(station)
            station_id += 1
    
    # Add highway stations (between cities)
    highway_routes = [
        # I-95 corridor (East Coast)
        {"start": (40.7128, -74.0060), "end": (42.3601, -71.0589), "stations": 15},  # NYC to Boston
        {"start": (40.7128, -74.0060), "end": (39.9526, -75.1652), "stations": 10},  # NYC to Philadelphia
        {"start": (39.9526, -75.1652), "end": (33.7490, -84.3880), "stations": 20},  # Philadelphia to Atlanta
        {"start": (33.7490, -84.3880), "end": (25.7617, -80.1918), "stations": 18},  # Atlanta to Miami
        
        # I-10 corridor (South)
        {"start": (34.0522, -118.2437), "end": (33.4484, -112.0740), "stations": 12},  # LA to Phoenix
        {"start": (33.4484, -112.0740), "end": (29.7604, -95.3698), "stations": 25},  # Phoenix to Houston
        {"start": (29.7604, -95.3698), "end": (25.7617, -80.1918), "stations": 30},  # Houston to Miami
        
        # I-5 corridor (West Coast)
        {"start": (32.7157, -117.1611), "end": (34.0522, -118.2437), "stations": 8},   # San Diego to LA
        {"start": (34.0522, -118.2437), "end": (37.3382, -121.8863), "stations": 15},  # LA to San Jose
        {"start": (37.3382, -121.8863), "end": (47.6062, -122.3321), "stations": 20},  # San Jose to Seattle
        
        # I-80 corridor (Cross-country)
        {"start": (37.3382, -121.8863), "end": (40.7128, -74.0060), "stations": 40},  # San Jose to NYC
    ]
    
    for route in highway_routes:
        start_lat, start_lon = route["start"]
        end_lat, end_lon = route["end"]
        
        for i in range(route["stations"]):
            # Interpolate between start and end points
            progress = (i + 1) / (route["stations"] + 1)
            lat = start_lat + (end_lat - start_lat) * progress
            lon = start_lon + (end_lon - start_lon) * progress
            
            # Add some randomness
            lat += np.random.uniform(-0.05, 0.05)
            lon += np.random.uniform(-0.05, 0.05)
            
            network = np.random.choice(networks)
            has_dc_fast = np.random.choice([True, False], p=[0.8, 0.2])  # Higher probability for highway stations
            
            station = {
                "id": station_id,
                "name": f"Highway Station {station_id}",
                "latitude": lat,
                "longitude": lon,
                "network": network,
                "access": "Public",
                "state": "Various",
                "city": "Highway",
                "level1_ports": np.random.randint(0, 2),
                "level2_ports": np.random.randint(4, 12),
                "dc_fast_ports": np.random.randint(2, 8) if has_dc_fast else 0,
                "fuel_type": "ELEC",
                "status": "E"
            }
            
            sample_stations.append(station)
            station_id += 1
    
    return pd.DataFrame(sample_stations)

def insert_sample_data():
    """Insert sample data into PostgreSQL database"""
    try:
        # Create sample data
        stations_df = create_sample_ev_stations()
        
        # Connect to database
        db_conn = DatabaseConnection()
        if not db_conn.connect():
            st.error("Failed to connect to database")
            return False
        
        # Create table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS ev_stations (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255),
            latitude DECIMAL(10, 8),
            longitude DECIMAL(11, 8),
            network VARCHAR(255),
            access VARCHAR(50),
            state VARCHAR(50),
            city VARCHAR(255),
            level1_ports INTEGER,
            level2_ports INTEGER,
            dc_fast_ports INTEGER,
            fuel_type VARCHAR(10),
            status VARCHAR(10)
        );
        """
        
        with db_conn.engine.connect() as conn:
            conn.execute(create_table_query)
            conn.commit()
        
        # Insert data
        stations_df.to_sql('ev_stations', db_conn.engine, if_exists='replace', index=False)
        
        st.success(f"Successfully inserted {len(stations_df)} EV charging stations into database!")
        
        return True
        
    except Exception as e:
        st.error(f"Failed to insert sample data: {str(e)}")
        return False

if __name__ == "__main__":
    # Create and display sample data
    sample_data = create_sample_ev_stations()
    print(f"Created {len(sample_data)} sample EV charging stations")
    print("\nSample data preview:")
    print(sample_data.head())
    
    print("\nData summary:")
    print(sample_data.describe())