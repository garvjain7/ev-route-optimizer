import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
import streamlit as st

class DatabaseConnection:
    def __init__(self):
        self.engine = None
        self.connection = None
    
    def connect(self):
        """Establish connection to PostgreSQL database"""
        try:
            # Try to get connection from DATABASE_URL first
            database_url = os.getenv("DATABASE_URL")
            
            if database_url:
                self.engine = create_engine(database_url)
            else:
                # Fallback to individual parameters
                db_config = {
                    'host': os.getenv('PGHOST', 'localhost'),
                    'port': os.getenv('PGPORT', '5432'),
                    'database': os.getenv('PGDATABASE', 'ev_stations'),
                    'user': os.getenv('PGUSER', 'postgres'),
                    'password': os.getenv('PGPASSWORD', '')
                }
                
                # Create connection string
                conn_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
                self.engine = create_engine(conn_string)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            return True
        
        except Exception as e:
            st.error(f"Database connection failed: {str(e)}")
            return False
    
    def load_ev_stations(self):
        """Load EV charging station data from PostgreSQL"""
        if not self.connect():
            return None
        
        try:
            # Try common table names for EV stations
            table_queries = [
                "SELECT * FROM ev_stations",
                "SELECT * FROM charging_stations",
                "SELECT * FROM stations",
                "SELECT * FROM ev_charging_stations"
            ]
            
            for query in table_queries:
                try:
                    df = pd.read_sql_query(query, self.engine)
                    if not df.empty:
                        return self._process_stations_data(df)
                except Exception:
                    continue
            
            # If no standard tables found, try to find any table with coordinate columns
            tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            """
            
            tables_df = pd.read_sql_query(tables_query, self.engine)
            
            for table_name in tables_df['table_name']:
                try:
                    # Check if table has coordinate columns
                    columns_query = f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}' 
                    AND table_schema = 'public'
                    """
                    
                    columns_df = pd.read_sql_query(columns_query, self.engine)
                    columns = columns_df['column_name'].tolist()
                    
                    # Look for latitude/longitude columns
                    lat_cols = [col for col in columns if any(x in col.lower() for x in ['lat', 'y'])]
                    lon_cols = [col for col in columns if any(x in col.lower() for x in ['lon', 'lng', 'x'])]
                    
                    if lat_cols and lon_cols:
                        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 1000", self.engine)
                        if not df.empty:
                            return self._process_stations_data(df)
                
                except Exception:
                    continue
            
            st.error("No EV station data found in the database")
            return None
        
        except Exception as e:
            st.error(f"Error loading EV stations: {str(e)}")
            return None
    
    def _process_stations_data(self, df):
        """Process and standardize EV stations data"""
        try:
            # Standardize column names
            column_mapping = {}
            
            # Map latitude columns
            lat_candidates = ['latitude', 'lat', 'y', 'coord_lat', 'station_lat']
            for col in df.columns:
                if col.lower() in lat_candidates:
                    column_mapping[col] = 'latitude'
                    break
            
            # Map longitude columns
            lon_candidates = ['longitude', 'lon', 'lng', 'x', 'coord_lon', 'station_lon']
            for col in df.columns:
                if col.lower() in lon_candidates:
                    column_mapping[col] = 'longitude'
                    break
            
            # Map other common columns
            other_mappings = {
                'station_name': 'name',
                'station_id': 'id',
                'ev_network': 'network',
                'access_code': 'access',
                'fuel_type_code': 'fuel_type',
                'ev_level1_evse_num': 'level1_ports',
                'ev_level2_evse_num': 'level2_ports',
                'ev_dc_fast_num': 'dc_fast_ports'
            }
            
            for old_col, new_col in other_mappings.items():
                if old_col in df.columns:
                    column_mapping[old_col] = new_col
            
            # Apply column mapping
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns exist
            if 'latitude' not in df.columns or 'longitude' not in df.columns:
                st.error("Could not identify latitude and longitude columns")
                return None
            
            # Filter out invalid coordinates
            df = df.dropna(subset=['latitude', 'longitude'])
            df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
            df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]
            
            # Add missing columns with default values
            if 'name' not in df.columns:
                df['name'] = 'EV Station'
            if 'network' not in df.columns:
                df['network'] = 'Unknown'
            if 'access' not in df.columns:
                df['access'] = 'Public'
            
            # Convert coordinate columns to numeric
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            
            # Remove rows with invalid coordinates after conversion
            df = df.dropna(subset=['latitude', 'longitude'])
            
            st.success(f"Successfully processed {len(df)} EV stations")
            return df
        
        except Exception as e:
            st.error(f"Error processing stations data: {str(e)}")
            return None
    
    def get_stations_in_region(self, min_lat, max_lat, min_lon, max_lon):
        """Get stations within a specific geographic region"""
        if not self.connect():
            return None
        
        try:
            query = """
            SELECT * FROM ev_stations 
            WHERE latitude BETWEEN %s AND %s 
            AND longitude BETWEEN %s AND %s
            """
            
            df = pd.read_sql_query(query, self.engine, params=[min_lat, max_lat, min_lon, max_lon])
            return self._process_stations_data(df)
        
        except Exception as e:
            st.error(f"Error querying stations by region: {str(e)}")
            return None
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
