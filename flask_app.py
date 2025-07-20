from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import io
import base64
from dotenv import load_dotenv
import inspect

# Import our ML components
from database.db_connection import DatabaseConnection
from filtering.station_filters import StationFilters
from clustering.clustering_engine import ClusteringEngine
from routing.ev_router import EVRouter
from visualization.map_visualizer import MapVisualizer
from utils.geojson_exporter import GeoJSONExporter
from create_sample_data import create_sample_ev_stations

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'fallback_dev_key')

# Global variables to store data
stations_data = None
filtered_stations = None
route_data = None
clustered_stations = None
current_dataset_type = 'none'  # 'sample', 'database', or 'uploaded'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/load_sample_data', methods=['POST'])
def load_sample_data():
    global stations_data, current_dataset_type
    try:
        stations_data = create_sample_ev_stations()
        current_dataset_type = 'sample'
        return jsonify({
            'success': True,
            'message': f'Created {len(stations_data)} sample stations',
            'total_stations': len(stations_data)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/load_database', methods=['POST'])
def load_database():
    global stations_data, current_dataset_type
    try:
        data = request.get_json(force=True) or {}
        port_type = data.get('port_type', 'both').lower()

        if port_type not in ['ac', 'dc', 'both']:
            return jsonify({'success': False, 'error': f"Invalid port_type '{port_type}'"}), 400

        db_conn = DatabaseConnection()
        stations_data = db_conn.load_ev_station(port_type=port_type)
        current_dataset_type = 'database'

        if stations_data is not None:
            # Preprocessing: fill missing essential columns
            if 'state' not in stations_data.columns:
                stations_data['state'] = 'Unknown'
            if 'network' not in stations_data.columns:
                stations_data['network'] = 'Unknown'
            if 'dc_fast_ports' not in stations_data.columns:
                stations_data['dc_fast_ports'] = 0

            return jsonify({
                'success': True,
                'message': f'Loaded {len(stations_data)} charging stations',
                'total_stations': len(stations_data)
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to load stations data'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/switch_dataset', methods=['POST'])
def switch_dataset():
    global stations_data, current_dataset_type
    try:
        data = request.get_json()
        new_dataset = data.get('dataset_type')

        if new_dataset == 'sample':
            stations_data = create_sample_ev_stations()
        elif new_dataset == 'database':
            db_conn = DatabaseConnection()
            stations_data = db_conn.load_ev_station()
        else:
            return jsonify({'success': False, 'error': 'Unsupported dataset type'})

        current_dataset_type = new_dataset
        if stations_data is None:
            return jsonify({'success': False, 'error': 'Failed to load stations data'})
        return jsonify({'success': True, 'message': f'Switched to {new_dataset} dataset', 'total_stations': len(stations_data)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_stats', methods=['GET'])
def get_stats():
    global stations_data, current_dataset_type
    if stations_data is None:
        return jsonify({'success': False, 'error': 'No data loaded'})

    try:
        stats = {
            'total_stations': len(stations_data),
            'states_covered': stations_data['state'].nunique() if 'state' in stations_data.columns else 'N/A',
            'networks': stations_data['network'].nunique() if 'network' in stations_data.columns else 'N/A',
            'dc_fast_stations': len(stations_data[stations_data.get('dc_fast_ports', 0) > 0]) if 'dc_fast_ports' in stations_data.columns else 'N/A',
            'dataset_type': current_dataset_type
        }
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/dev/set_stations', methods=['POST'])
def dev_set_stations():
    """
    DEV-ONLY ROUTE: Manually set stations_data for testing from a client script.
    """
    global stations_data, current_dataset_type
    try:
        data = request.get_json(force=True)
        stations_data_list = data.get("stations_data", [])

        if not stations_data_list:
            return jsonify({'success': False, 'error': 'stations_data is empty'}), 400

        stations_data = pd.DataFrame(stations_data_list)
        current_dataset_type = 'test'

        return jsonify({
            'success': True,
            'message': f'{len(stations_data)} stations loaded into memory for testing.'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/apply_filtering', methods=['POST'])
def apply_filtering():
    global stations_data, filtered_stations
    if stations_data is None:
        return jsonify({'success': False, 'error': 'No data loaded'})

    try:
        data = request.get_json()
        source_coords = (data['source_lat'], data['source_lon'])
        dest_coords = (data['dest_lat'], data['dest_lon'])
        filtering_method = data['filtering_method']
        params = data.get('params', {})

        station_filters = StationFilters()

        # Custom mappings for function references
        method_map = {
            "Smart ML Filtering": station_filters.smart_filtering,
            "ML-Based Filtering": station_filters.ml_based_filtering,
            "Geographic Bounding Box": station_filters.geographic_bounding_box,
            "Distance-Based": station_filters.distance_based_filtering,
            "Corridor-Based": station_filters.corridor_based_filtering,
            "Combined": station_filters.combined_filtering
        }

        if filtering_method not in method_map:
            return jsonify({'success': False, 'error': f'Unknown filtering method: {filtering_method}'})

        func = method_map[filtering_method]

        # Build kwargs dynamically
        kwargs = {
            "stations_df": stations_data,
            "source_coords": source_coords,
            "dest_coords": dest_coords
        }

        if filtering_method == "Smart ML Filtering":
            kwargs["user_preferences"] = {
                'prefer_fast_charging': params.get('prefer_fast_charging', True),
                'max_detour_factor': params.get('detour_factor', 1.5),
                'corridor_width': params.get('corridor_width', 25),
                'min_station_rating': params.get('min_station_rating', 3.0),
                'avoid_congestion': params.get('avoid_congestion', True),
                'charging_types': params.get('charging_types', ['AC', 'DC']),
                'power_levels': params.get('power_levels', ['Level2', 'DC_Fast'])
            }
        elif filtering_method == "ML-Based Filtering":
            kwargs["filter_criteria"] = {
                'min_rating': params.get('min_station_rating', 3.0),
                'max_congestion': 1 if params.get('avoid_congestion', True) else 2,
                'prefer_fast_charging': params.get('prefer_fast_charging', True),
                'min_station_score': params.get('min_station_rating', 3.0)
            }
        else:
            # Dynamically inject only matching parameters for non-ML methods
            sig = inspect.signature(func)
            for key in sig.parameters:
                if key not in kwargs and key in params:
                    kwargs[key] = params[key]

        # Call the filtering function with only valid arguments
        filtered_stations = func(**kwargs)

        if filtered_stations is not None:
            ml_summary = {}
            if 'predicted_congestion' in filtered_stations.columns:
                congestion_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
                avg_congestion = filtered_stations['predicted_congestion'].mean()
                ml_summary['avg_congestion'] = congestion_labels.get(round(avg_congestion), 'Unknown')

            if 'predicted_rating' in filtered_stations.columns:
                ml_summary['avg_rating'] = round(filtered_stations['predicted_rating'].mean(), 1)

            if 'ml_station_score' in filtered_stations.columns:
                ml_summary['avg_score'] = round(filtered_stations['ml_station_score'].mean(), 1)

            return jsonify({
                'success': True,
                'message': f'Filtered to {len(filtered_stations)} stations',
                'filtered_count': len(filtered_stations),
                'ml_summary': ml_summary
            })

        return jsonify({'success': False, 'error': 'No stations found with current filters'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# @app.route('/api/perform_clustering', methods=['POST'])
# def perform_clustering():
#     global filtered_stations, clustered_stations

#     if filtered_stations is None or filtered_stations.empty:
#         return jsonify({'success': False, 'error': 'No filtered stations available'}), 400

#     try:
#         data = request.get_json() or {}

#         user_clusters = data.get('n_clusters', None)
#         apply_additional_filters = data.get('apply_additional_filters', False)

#         print(f"[DEBUG] perform_clustering called with user_clusters={user_clusters}, apply_additional_filters={apply_additional_filters}")

#         clustering_engine = ClusteringEngine()

#         try:
#             clustered_result = clustering_engine.cluster_stations(
#                 filtered_stations,
#                 # n_clusters=user_clusters,
#                 requested_clusters=user_clusters,
#                 # apply_additional_filters=apply_additional_filters
#                 include_features = apply_additional_filters
#             )
#         except Exception as ce:
#             print(f"[DEBUG] ClusteringEngine.cluster_stations() threw: {ce}")
#             return jsonify({'success': False, 'error': 'ClusteringEngine crashed: ' + str(ce)}), 500

#         print(f"[DEBUG] clustered_result = {clustered_result}")

#         if clustered_result is None:
#             return jsonify({'success': False, 'error': 'Clustering engine returned None'}), 500

#         clustered_stations = clustered_result['stations']

#         return jsonify({
#             'success': True,
#             'message': f'Created {clustered_result.get("actual_n_clusters", "unknown")} clusters',
#             'n_clusters': clustered_result.get('actual_n_clusters'),
#             'cluster_stats': clustered_result.get('statistics', {}),
#             'centroids': clustered_result.get('centroids', {}),
#             'summary': clustering_engine.get_cluster_summary(clustered_result)
#         })

#     except Exception as e:
#         print(f"[ERROR] perform_clustering failed: {str(e)}")
#         return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/perform_clustering', methods=['POST'])
def perform_clustering():
    global filtered_stations, clustered_stations

    if filtered_stations is None or filtered_stations.empty:
        return jsonify({'success': False, 'error': 'No filtered stations available'}), 400

    try:
        data = request.get_json() or {}

        user_clusters = data.get('n_clusters', None)
        print(f"[DEBUG] perform_clustering called with user_clusters={user_clusters}")

        clustering_engine = ClusteringEngine()

        try:
            clustered_result, actual_clusters = clustering_engine.cluster_stations(
                filtered_stations,
                requested_clusters=user_clusters
            )
        except Exception as ce:
            print(f"[DEBUG] ClusteringEngine.cluster_stations() threw: {ce}")
            return jsonify({'success': False, 'error': 'ClusteringEngine crashed: ' + str(ce)}), 500

        print(f"[DEBUG] clustered_result = {clustered_result}")

        if clustered_result is None:
            return jsonify({'success': False, 'error': 'Clustering engine returned None'}), 500
        
        def convert_numpy(obj):
            """Recursively convert NumPy data types to native Python types."""
            if isinstance(obj, dict):
                return {convert_numpy(k): convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj

        clustered_stations = clustered_result['stations']

        return jsonify({
            'success': True,
            'message': f'Created {actual_clusters} clusters',
            'n_clusters': actual_clusters,
            'cluster_stats': convert_numpy(clustered_result.get('statistics', {})),
            'centroids': convert_numpy(clustered_result.get('centroids', {})),
            'summary': convert_numpy(clustering_engine.get_cluster_summary(clustered_result)),
            'clusters': clustered_result.get('clusters', [])
        })

    except Exception as e:
        print(f"[ERROR] perform_clustering failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/optimize_route', methods=['POST'])
def optimize_route():
    global filtered_stations, clustered_stations, route_data

    # Use clustered stations if available, fallback to filtered
    stations_to_use = clustered_stations if clustered_stations is not None and not getattr(clustered_stations, 'empty', True) else filtered_stations

    import pandas as pd
    # Convert to DataFrame if needed
    if isinstance(stations_to_use, list):
        stations_to_use = pd.DataFrame(stations_to_use)
    print("[DEBUG] stations_to_use type:", type(stations_to_use))
    print("[DEBUG] stations_to_use shape:", stations_to_use.shape if hasattr(stations_to_use, 'shape') else 'N/A')
    print("[DEBUG] stations_to_use columns:", stations_to_use.columns if hasattr(stations_to_use, 'columns') else 'N/A')

    if stations_to_use is None or len(stations_to_use) == 0:
        return jsonify({'success': False, 'error': 'No filtered or clustered stations available'}), 400
    if not {'latitude', 'longitude'}.issubset(stations_to_use.columns):
        return jsonify({'success': False, 'error': 'Stations missing latitude/longitude columns'}), 400

    try:
        data = request.get_json()

        required_fields = ['source_lat', 'source_lon', 'dest_lat', 'dest_lon']
        if not all(k in data for k in required_fields):
            return jsonify({'success': False, 'error': 'Missing required coordinates'}), 400

        source_coords = (float(data['source_lat']), float(data['source_lon']))
        dest_coords = (float(data['dest_lat']), float(data['dest_lon']))

        ev_specs = {
            'battery_range': float(data.get('battery_range', 300)),
            'consumption_rate': float(data.get('consumption_rate', 20)),
            'charging_time': float(data.get('charging_time', 30)),
            'safety_margin': float(data.get('safety_margin', 15))
        }

        ev_router = EVRouter()

        print(f"[DEBUG] Optimizing route from {source_coords} to {dest_coords}")
        print(f"[DEBUG] Stations available for routing: {len(stations_to_use)}")

        route_result = ev_router.optimize_route(source_coords, dest_coords, stations_to_use, ev_specs)

        if route_result is None:
            return jsonify({'success': False, 'error': 'Route optimization failed'}), 500

        route_data = route_result
        route_stats = route_result.get('statistics', {})

        return jsonify({
            'success': True,
            'message': 'Route optimized successfully',
            'route_id': route_result.get('route_id'),
            'route_stats': route_stats,
            'has_ml_predictions': 'ml_predictions' in route_result,
            'ml_predictions': route_result.get('ml_predictions', {}),
            'route': route_result
        }), 200

    except Exception as e:
        print(f"[ERROR] optimize_route failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
        
@app.route('/api/generate_map', methods=['POST'])
def generate_map():
    global filtered_stations, clustered_stations, route_data

    try:
        data = request.get_json()
        source_coords = (data['source_lat'], data['source_lon'])
        dest_coords = (data['dest_lat'], data['dest_lon'])

        # Decide which station set to use: clustered or filtered
        stations_to_use = None

        if clustered_stations is not None:
            if isinstance(clustered_stations, pd.DataFrame):
                if not clustered_stations.empty:
                    stations_to_use = clustered_stations
            elif isinstance(clustered_stations, list):
                if len(clustered_stations) > 0:
                    stations_to_use = pd.DataFrame(clustered_stations)

        if stations_to_use is None:
            print("[WARNING] clustered_stations is empty or not set, falling back to filtered_stations")
            stations_to_use = filtered_stations
        else:
            print(f"[DEBUG] Using clustered_stations with {len(stations_to_use)} entries")

        # Create map
        map_visualizer = MapVisualizer()
        map_obj = map_visualizer.create_route_map(
            source_coords,
            dest_coords,
            stations_df=stations_to_use,
            route_data=route_data
        )

        map_html = map_obj._repr_html_()

        return jsonify({
            'success': True,
            'map_html': map_html
        })

    except Exception as e:
        print(f"[ERROR] generate_map failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/export_geojson', methods=['GET'])
def export_geojson():
    global filtered_stations
    if filtered_stations is None:
        return jsonify({'success': False, 'error': 'No filtered stations available'})
    
    try:
        geojson_exporter = GeoJSONExporter()
        geojson_data = geojson_exporter.export_stations(filtered_stations)
        
        return jsonify({
            'success': True,
            'geojson': geojson_data,
            'filename': 'filtered_stations.geojson'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/export_route', methods=['GET'])
def export_route():
    global route_data
    if route_data is None:
        return jsonify({'success': False, 'error': 'No route data available'})
    
    try:
        return jsonify({
            'success': True,
            'route_data': route_data,
            'filename': 'route_data.json'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/train_models', methods=['POST'])
def train_models():
    global stations_data
    if stations_data is None:
        return jsonify({'success': False, 'error': 'No station data available'})
    
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'station_predictor')
        
        if model_type == 'station_predictor':
            from ml_models.station_predictor import StationPredictor
            predictor = StationPredictor()
            success = predictor.train_models(stations_data)
            return jsonify({
                'success': success,
                'message': 'Station predictor trained successfully!' if success else 'Training failed'
            })
        elif model_type == 'travel_time_predictor':
            from ml_models.travel_time_predictor import TravelTimePredictor
            predictor = TravelTimePredictor()
            success = predictor.train_model()
            return jsonify({
                'success': success,
                'message': 'Travel time predictor trained successfully!' if success else 'Training failed'
            })
        elif model_type == 'adaptive_router':
            from ml_models.adaptive_router import AdaptiveRouter
            router = AdaptiveRouter()
            success = router.train_efficiency_model()
            return jsonify({
                'success': success,
                'message': 'Adaptive router trained successfully!' if success else 'Training failed'
            })
        else:
            return jsonify({'success': False, 'error': 'Unknown model type'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/submit_feedback', methods=['POST'])
def submit_feedback():
    global route_data
    if route_data is None:
        return jsonify({'success': False, 'error': 'No route data available'})
    
    try:
        feedback = request.get_json()
        
        from ml_models.adaptive_router import AdaptiveRouter
        router = AdaptiveRouter()
        success = router.add_route_feedback(route_data, feedback)
        
        return jsonify({
            'success': success,
            'message': 'Feedback submitted successfully!' if success else 'Failed to save feedback'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_analytics', methods=['GET'])
def get_analytics():
    try:
        from ml_models.adaptive_router import AdaptiveRouter
        router = AdaptiveRouter()
        analytics = router.get_performance_analytics()
        
        if analytics:
            return jsonify({
                'success': True,
                'analytics': analytics
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No analytics data available yet'
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_route_details', methods=['GET'])
def get_route_details():
    global route_data
    if route_data is None:
        return jsonify({'success': False, 'error': 'No route data available'})
    
    try:
        # Extract charging stops and energy profile
        charging_stops = route_data.get('charging_stops', [])
        energy_profile = route_data.get('energy_profile', {})
        ml_predictions = route_data.get('ml_predictions', {})
        
        return jsonify({
            'success': True,
            'charging_stops': charging_stops,
            'energy_profile': energy_profile,
            'ml_predictions': ml_predictions
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
