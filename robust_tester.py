import requests
import traceback
import json
import sys
import os
import math
import webbrowser

BASE_URL = "http://127.0.0.1:5000"

FILTERING_METHODS_MAP = {
    "1": "Smart ML Filtering",
    "2": "ML-Based Filtering",
    "3": "Geographic Bounding Box",
    "4": "Distance-Based",
    "5": "Corridor-Based",
    "6": "Combined"
}

FILTERING_METHODS = list(FILTERING_METHODS_MAP.values())

prev_station_sets = {}


def safe_post(endpoint, data=None):
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        if not result.get('success'):
            print(f"[WARNING] API {endpoint} failed: {result.get('error', 'No specific error')}")
        else:
            print(f"[INFO] {endpoint} succeeded: {result.get('message', '')}")
        return result
    except Exception as e:
        print(f"[ERROR] Failed to call {endpoint}: {e}\n{traceback.format_exc()}")
        return {'success': False, 'error': str(e)}


def safe_get(endpoint):
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        if not result.get('success'):
            print(f"[WARNING] API {endpoint} failed: {result.get('error', 'No specific error')}")
        else:
            print(f"[INFO] {endpoint} succeeded")
        return result
    except Exception as e:
        print(f"[ERROR] Failed to call {endpoint}: {e}\n{traceback.format_exc()}")
        return {'success': False, 'error': str(e)}


def validate_coords(source_lat, source_lon, dest_lat, dest_lon):
    try:
        assert -90 <= source_lat <= 90 and -180 <= source_lon <= 180
        assert -90 <= dest_lat <= 90 and -180 <= dest_lon <= 180
        return True
    except AssertionError:
        print("[ERROR] Invalid latitude or longitude values")
        return False


def fallback_top_n_stations(station_list, src_lat, src_lon, dst_lat, dst_lon):
    if not station_list:
        print("[ERROR] No filtered stations for fallback fallback_top_n_stations")
        return []
    try:
        if "ml_station_score" in station_list[0]:
            sorted_stations = sorted(station_list, key=lambda x: x.get("ml_station_score", 0), reverse=True)
        elif "predicted_rating" in station_list[0]:
            sorted_stations = sorted(station_list, key=lambda x: x.get("predicted_rating", 0), reverse=True)
        else:
            sorted_stations = station_list

        top = [s for s in sorted_stations if is_between_route(s['lat'], s['lon'], src_lat, src_lon, dst_lat, dst_lon)]
        return top[:8] if top else sorted_stations[:8]
    except Exception as e:
        print(f"[ERROR] Fallback selection failed: {e}")
        return station_list[:8] if station_list else []


def is_between_route(lat, lon, src_lat, src_lon, dst_lat, dst_lon, buffer_km=20):
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    dist_total = haversine(src_lat, src_lon, dst_lat, dst_lon)
    dist_1 = haversine(src_lat, src_lon, lat, lon)
    dist_2 = haversine(lat, lon, dst_lat, dst_lon)
    return abs((dist_1 + dist_2) - dist_total) <= buffer_km


def check_duplicate_stations(method, station_list):
    ids = set(f["station_id"] if "station_id" in f else f["name"] for f in station_list)
    for prev_method, prev_ids in prev_station_sets.items():
        overlap = ids.intersection(prev_ids)
        if len(overlap) >= 6:
            print(f"[NOTE] Filtering '{method}' has high overlap with '{prev_method}' ({len(overlap)} stations)")
    prev_station_sets[method] = ids


def debug_pipeline(source_lat, source_lon, dest_lat, dest_lon, filtering_method):
    print("========== EV Routing Full Debug Start ==========")

    if filtering_method not in FILTERING_METHODS:
        print(f"[FATAL] Invalid filtering method '{filtering_method}'. Choose from:")
        for key, val in FILTERING_METHODS_MAP.items():
            print(f"  {key}: {val}")
        return

    if not validate_coords(source_lat, source_lon, dest_lat, dest_lon):
        print("[FATAL] Invalid coordinates. Exiting.")
        return
    print("[STEP] Coordinates validated.")

    result = safe_post("/api/load_database", {"port_type": "both"})
    if not result['success']:
        print("[WARNING] Falling back to sample data")
        result = safe_post("/api/load_sample_data")
    print("[STEP] Data loading complete.")

    total_stations = result.get("total_stations", 0)
    print(f"[INFO] Total Stations Loaded: {total_stations}")

    filtering_payload = {
        "source_lat": source_lat,
        "source_lon": source_lon,
        "dest_lat": dest_lat,
        "dest_lon": dest_lon,
        "filtering_method": filtering_method,
        "params": {
            "detour_factor": 1.5,
            "corridor_width": 25,
            "max_distance_source": 200,
            "max_distance_dest": 200,
            "min_station_rating": 3.0,
            "avoid_congestion": True,
            "prefer_fast_charging": True,
            "charging_types": ["AC", "DC"],
            "power_levels": ["Level2", "DC_Fast"]
        }
    }
    result = safe_post("/api/apply_filtering", filtering_payload)
    filtered_count = result.get("filtered_count", 0)
    print(f"[INFO] Filtered Stations Count ({filtering_method}): {filtered_count}")
    print("[STEP] Filtering logic applied.")

    if filtered_count == 0:
        print("[WARNING] Filtering eliminated all stations. Trying fallback with Distance-Based filtering...")
        filtering_payload["filtering_method"] = "Distance-Based"
        result = safe_post("/api/apply_filtering", filtering_payload)
        filtered_count = result.get("filtered_count", 0)

    if filtered_count == 0:
        print("[FATAL] No stations found even with fallback. Exiting pipeline.")
        return

    geojson_result = safe_get("/api/export_geojson")
    station_list = geojson_result.get("geojson", {}).get("features", [])
    if station_list:
        station_list = [
            {
                "lat": f["geometry"]["coordinates"][1],
                "lon": f["geometry"]["coordinates"][0],
                **f["properties"]
            } for f in station_list
        ]
        if len(station_list) > 100:
            print("[WARNING] Very high station count after filtering. Likely ineffective filter logic.")
        check_duplicate_stations(filtering_method, station_list)

    result_cluster = safe_post("/api/perform_clustering", {"n_clusters": 8})
    if not result_cluster['success']:
        print("[WARNING] Clustering failed. Will fallback to top-N stations from filtered list.")
        fallback_stations = fallback_top_n_stations(station_list, source_lat, source_lon, dest_lat, dest_lon)
        if not fallback_stations:
            print("[FATAL] No fallback stations available. Exiting pipeline.")
            return
    print("[STEP] Clustering step completed.")

    routing_payload = {
        "source_lat": source_lat,
        "source_lon": source_lon,
        "dest_lat": dest_lat,
        "dest_lon": dest_lon,
        "battery_range": 300,
        "consumption_rate": 20,
        "charging_time": 30,
        "safety_margin": 15
    }
    result = safe_post("/api/optimize_route", routing_payload)
    if not result['success']:
        print("[ERROR] Route optimization failed. Trying default straight route...")
        return
    print("[STEP] Route optimization complete.")

    route_data = result.get("route", {})
    ev_stations = route_data.get("charging_stops", [])
    between_count = 0
    print("[INFO] Final Charging Stations used in route:")
    for stn in ev_stations:
        name = stn.get("name") or stn.get("station_id")
        lat = stn['lat']
        lon = stn['lon']
        if is_between_route(lat, lon, source_lat, source_lon, dest_lat, dest_lon):
            print(f"  - {name} ({lat}, {lon}) ✅")
            between_count += 1
        else:
            print(f"  - {name} ({lat}, {lon}) [OUTLIER ❌]")
    print(f"[CHECK] Stations between route: {between_count}/{len(ev_stations)}")

    result = safe_post("/api/generate_map", {
        "source_lat": source_lat,
        "source_lon": source_lon,
        "dest_lat": dest_lat,
        "dest_lon": dest_lon
    })
    if result['success']:
        map_html = result.get("map_html", "")
        output_file = "ev_route_map.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(map_html)
        print(f"[INFO] Map generation successful. Saved to '{output_file}'")
        webbrowser.open(f"file://{os.path.abspath(output_file)}")
    else:
        print("[WARNING] Map generation failed.")
    print("[STEP] Map generation complete.")

    result = safe_get("/api/export_route")
    if result['success']:
        filename = result.get("filename", "route_data.json")
        route_data = result.get("route_data", {})
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(route_data, f, indent=2)
        print(f"[INFO] Exported route data to {filename}")
    print("[STEP] Export step finished.")

    print("========== EV Routing Full Debug Complete ==========")

if __name__ == "__main__":
    src_lat = 26.9124  # Jaipur
    src_lon = 75.7873
    dst_lat = 26.8851  # Jaipur South (Sanganer)
    dst_lon = 75.8152

    if len(sys.argv) < 2:
        print("Usage: python robust_tester.py <filtering_method_number>")
        for key, val in FILTERING_METHODS_MAP.items():
            print(f"  {key}: {val}")
        sys.exit(1)

    filtering_choice = sys.argv[1]
    method = FILTERING_METHODS_MAP.get(filtering_choice)

    if not method:
        print(f"Invalid choice '{filtering_choice}'. Please use a number 1-6")
        for key, val in FILTERING_METHODS_MAP.items():
            print(f"  {key}: {val}")
        sys.exit(1)

    debug_pipeline(src_lat, src_lon, dst_lat, dst_lon, method)
    