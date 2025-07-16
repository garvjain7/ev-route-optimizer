from database.db_connection import DatabaseConnection

try:
    db = DatabaseConnection()
    stations_df = db.load_ev_station()
    print(f"Loaded {len(stations_df)} EV stations.")
except Exception as e:
    print("DB connection failed:", str(e))
