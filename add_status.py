import os
import requests
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sqlalchemy import text
from datetime import datetime
import pytz  # to localize time if needed
import json

load_dotenv()

# Load DB & API credentials
api_key = os.getenv("GOOGLE_PLACES_API_KEY")
database_url = os.getenv("DATABASE_URL")

if not api_key:
    raise ValueError("Google Places API key is missing in .env file")

if not database_url:
    raise ValueError("DATABASE_URL is missing in .env file")

# Create DB connection
engine = create_engine(database_url)

# Fetch EV stations (name + address only)
query = "SELECT name, address FROM ev_station"
df = pd.read_sql_query(query, engine)

from datetime import datetime
import pytz  # to localize time if needed

def check_place_status(name, address):
    try:
        # Step 1: Get place_id
        search_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
        input_text = f"{name}, {address}"
        search_params = {
            "input": input_text,
            "inputtype": "textquery",
            "fields": "place_id",
            "key": api_key
        }

        r = requests.get(search_url, params=search_params)
        data = r.json()

        if not data.get("candidates"):
            return "NOT FOUND"

        place_id = data["candidates"][0]["place_id"]

        # Step 2: Get detailed status
        detail_url = "https://maps.googleapis.com/maps/api/place/details/json"
        detail_params = {
            "place_id": place_id,
            "fields": "business_status,opening_hours",
            "key": api_key
        }

        r2 = requests.get(detail_url, params=detail_params)
        details = r2.json()

        result = details.get("result", {})
        business_status = result.get("business_status", "")
        hours = result.get("opening_hours", {})

        # 🔍 DEBUG: Show result for testing
        print(f"\n{input_text}")
        print("Result:", json.dumps(result, indent=2))

        if business_status == "CLOSED_PERMANENTLY":
            return "Permanently closed"

        if hours.get("open_now") is True:
            periods = hours.get("periods", [])
            today = datetime.now(pytz.UTC).weekday()
            for p in periods:
                if p["open"]["day"] == today:
                    if "close" in p:
                        close_time = p["close"]["time"]
                        close_hour = int(close_time[:2])
                        close_min = int(close_time[2:])
                        return f"Open until {close_hour}:{close_min:02d}"
                    else:
                        return "Open 24 hours"
            return "Open now"
        elif hours.get("open_now") is False:
            return "Closed now"
        else:
            return "Open status unknown"

    except Exception as e:
        return f"ERROR: {e}"


# Run the check for each station
results = []
for _, row in df.iterrows():
    name, address = row['name'], row['address']
    status = check_place_status(name, address)
    results.append((name, address, status))
    print(f"[{status}] {name} | {address}")

# Step 1: Ensure 'status' column exists in PostgreSQL
# with engine.connect() as conn:
#     conn.execute(text("""
#         DO $$
#         BEGIN
#             IF NOT EXISTS (
#                 SELECT 1 FROM information_schema.columns
#                 WHERE table_name = 'ev_station'
#                 AND column_name = 'status'
#             ) THEN
#                 ALTER TABLE ev_station ADD COLUMN status TEXT;
#             END IF;
#         END;
#         $$;
#     """))

# Step 2: Update status values using name + address
with engine.begin() as conn:
    for name, address, status in results:
        conn.execute(text("""
            UPDATE ev_station
            SET status = :status
            WHERE name = :name AND address = :address
        """), {
            "status": status,
            "name": name,
            "address": address
        })

print("✅ All statuses updated in PostgreSQL ev_station table.")


