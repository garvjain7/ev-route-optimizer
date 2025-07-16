import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv  # ✅ Load variables from .env

# ✅ Load environment variables from .env file
load_dotenv()

class DatabaseConnection:
    def __init__(self):
        self.engine = None

    def connect(self):
        """Establish connection to PostgreSQL using .env variables"""
        try:
            # ✅ First priority: use full DATABASE_URL if provided (used in deployment)
            database_url = os.getenv("DATABASE_URL")

            if database_url:
                self.engine = create_engine(database_url)

            else:
                # ✅ Fallback to individual components (used in local dev)
                db_config = {
                    'host': os.getenv('PGHOST', 'localhost'),
                    'port': os.getenv('PGPORT', '5432'),
                    'database': os.getenv('PGDATABASE', 'ev_station'),
                    'user': os.getenv('PGUSER', 'postgres'),
                    'password': os.getenv('PGPASSWORD', '')
                }

                # ✅ Create PostgreSQL connection string manually
                conn_string = (
                    f"postgresql://{db_config['user']}:{db_config['password']}@"
                    f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
                )
                self.engine = create_engine(conn_string)

            # ✅ Test the database connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True

        except Exception as e:
            print(f"[ERROR] Database connection failed: {str(e)}")
            return False

    def load_ev_station(self, port_type='both'):
        """Load EV station data and apply smart port_type fallback logic"""
        if not self.connect():
            return None

        try:
            df = pd.read_sql_query("SELECT * FROM ev_station", self.engine)
            return self._process_stations_data(df, port_type=port_type)

        except Exception as e:
            print(f"[ERROR] Failed to load EV stations: {str(e)}")
            return None

    def _process_stations_data(self, df, port_type='both'):
        """Standardize and enrich station data based on port type fallback"""
        try:
            # ✅ Rename known coordinate columns
            df = df.rename(columns={
                'lat': 'latitude',
                'lng': 'longitude',
                'station_id': 'id'
            })

            # ✅ Filter invalid coordinates
            df = df.dropna(subset=['latitude', 'longitude'])
            df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]

            # ✅ Fill common metadata fields
            for col in ['name', 'owner', 'address', 'website', 'city']:
                if col not in df.columns:
                    df[col] = 'Unknown'

            # ✅ Add fallback 'id' if missing
            if 'id' not in df.columns:
                df['id'] = df.index.astype(str)

            # ✅ Use 'city' as 'state' if 'state' missing
            if 'state' not in df.columns:
                if 'city' in df.columns:
                    df['state'] = df['city']
                else:
                    df['state'] = 'Unknown'

            # ✅ Add 'network' (from owner if missing)
            if 'network' not in df.columns:
                if 'owner' in df.columns:
                    df['network'] = df['owner']
                else:
                    df['network'] = 'Unknown'
            
            # ✅ If 'ports' is missing, assume 0
            if 'ports' not in df.columns:
                df['ports'] = 0

            # ✅ Smart fallback for missing AC/DC columns
            if port_type.lower() == 'ac':
                df['ac_ports'] = df['ports']
                df['dc_fast_ports'] = 0

            elif port_type.lower() == 'dc':
                df['ac_ports'] = 0
                df['dc_fast_ports'] = df['ports']

            elif port_type.lower() == 'both':
                df['ac_ports'] = df['ports'] // 2
                df['dc_fast_ports'] = df['ports'] - df['ac_ports']

            else:
                df['ac_ports'] = df['ports']
                df['dc_fast_ports'] = 0

            return df

        except Exception as e:
            print(f"[ERROR] Failed to process stations data: {str(e)}")
            return None

    def get_stations_in_region(self, min_lat, max_lat, min_lon, max_lon, port_type='both'):
        """Filter EV stations by bounding box and apply smart port type fallback"""
        if not self.connect():
            return None

        try:
            query = """
            SELECT * FROM ev_station 
            WHERE lat BETWEEN %s AND %s 
              AND lng BETWEEN %s AND %s
            """

            df = pd.read_sql_query(query, self.engine, params=[min_lat, max_lat, min_lon, max_lon])
            return self._process_stations_data(df, port_type=port_type)

        except Exception as e:
            print(f"[ERROR] Failed to get stations in region: {str(e)}")
            return None

    def close(self):
        """Close SQLAlchemy connection pool"""
        if self.engine:
            self.engine.dispose()
