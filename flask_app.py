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

# Import our ML components
from database.db_connection import DatabaseConnection
from filtering.station_filters import StationFilters
from clustering.clustering_engine import ClusteringEngine
from routing.ev_router import EVRouter
from visualization.map_visualizer import MapVisualizer
from utils.geojson_exporter import GeoJSONExporter
from create_sample_data import create_sample_ev_stations

app = Flask(__name__)
app.secret_key = 'ev_routing_secret_key'

# Global variables to store data
stations_data = None
filtered_stations = None
route_data = None
clustered_stations = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/load_sample_data', methods=['POST'])
def load_sample_data():
    global stations_data
    try:
        stations_data = create_sample_ev_stations()
        return jsonify({
            'success': True,
            'message': f'Created {len(stations_data)} sample stations',
            'total_stations': len(stations_data)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/load_database', methods=['POST'])
def load_database():
    global stations_data
    try:
        db_conn = DatabaseConnection()
        stations_data = db_conn.load_ev_stations()
        if stations_data is not None:
            return jsonify({
                'success': True,
                'message': f'Loaded {len(stations_data)} charging stations',
                'total_stations': len(stations_data)
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to load stations data'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_stats', methods=['GET'])
def get_stats():
    global stations_data
    if stations_data is None:
        return jsonify({'success': False, 'error': 'No data loaded'})
    
    try:
        stats = {
            'total_stations': len(stations_data),
            'states_covered': stations_data['state'].nunique() if 'state' in stations_data.columns else 0,
            'networks': stations_data['network'].nunique() if 'network' in stations_data.columns else 0,
            'dc_fast_stations': len(stations_data[stations_data['dc_fast_ports'] > 0]) if 'dc_fast_ports' in stations_data.columns else 0
        }
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

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
        
        # Get filtering parameters
        params = data.get('params', {})
        
        station_filters = StationFilters()
        
        if filtering_method == "Smart ML Filtering":
            user_preferences = {
                'prefer_fast_charging': params.get('prefer_fast_charging', True),
                'max_detour_factor': params.get('detour_factor', 1.5),
                'corridor_width': params.get('corridor_width', 25),
                'min_station_rating': params.get('min_station_rating', 3.0),
                'avoid_congestion': params.get('avoid_congestion', True),
                'charging_types': params.get('charging_types', ['AC', 'DC']),
                'power_levels': params.get('power_levels', ['Level2', 'DC_Fast'])
            }
            filtered_stations = station_filters.smart_filtering(
                stations_data, source_coords, dest_coords, user_preferences=user_preferences
            )
        elif filtering_method == "ML-Based Filtering":
            filter_criteria = {
                'min_rating': params.get('min_station_rating', 3.0),
                'max_congestion': 1 if params.get('avoid_congestion', True) else 2,
                'prefer_fast_charging': params.get('prefer_fast_charging', True),
                'min_station_score': params.get('min_station_rating', 3.0)
            }
            filtered_stations = station_filters.ml_based_filtering(
                stations_data, source_coords, dest_coords, filter_criteria
            )
        elif filtering_method == "Geographic Bounding Box":
            filtered_stations = station_filters.geographic_bounding_box(
                stations_data, source_coords, dest_coords
            )
        elif filtering_method == "Distance-Based":
            filtered_stations = station_filters.distance_based_filtering(
                stations_data, source_coords, dest_coords,
                params.get('detour_factor', 1.5),
                params.get('max_distance_source', 200),
                params.get('max_distance_dest', 200)
            )
        elif filtering_method == "Corridor-Based":
            filtered_stations = station_filters.corridor_based_filtering(
                stations_data, source_coords, dest_coords, params.get('corridor_width', 25)
            )
        else:  # Combined Approach
            filtered_stations = station_filters.combined_filtering(
                stations_data, source_coords, dest_coords,
                params.get('detour_factor', 1.5),
                params.get('corridor_width', 25),
                params.get('max_distance_source', 200),
                params.get('max_distance_dest', 200)
            )
        
        if filtered_stations is not None:
            # Calculate ML prediction summary
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
        else:
            return jsonify({'success': False, 'error': 'No stations found with current filters'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/perform_clustering', methods=['POST'])
def perform_clustering():
    global filtered_stations, clustered_stations
    if filtered_stations is None:
        return jsonify({'success': False, 'error': 'No filtered stations available'})
    
    try:
        data = request.get_json()
        n_clusters = data.get('n_clusters', 8)
        
        clustering_engine = ClusteringEngine()
        clustered_stations = clustering_engine.cluster_stations(filtered_stations, n_clusters)
        
        return jsonify({
            'success': True,
            'message': f'Created {n_clusters} clusters',
            'n_clusters': n_clusters
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/optimize_route', methods=['POST'])
def optimize_route():
    global filtered_stations, route_data
    if filtered_stations is None:
        return jsonify({'success': False, 'error': 'No filtered stations available'})
    
    try:
        data = request.get_json()
        source_coords = (data['source_lat'], data['source_lon'])
        dest_coords = (data['dest_lat'], data['dest_lon'])
        
        ev_specs = {
            'battery_range': data.get('battery_range', 300),
            'consumption_rate': data.get('consumption_rate', 20),
            'charging_time': data.get('charging_time', 30),
            'safety_margin': data.get('safety_margin', 15)
        }
        
        ev_router = EVRouter()
        route_result = ev_router.optimize_route(
            source_coords, dest_coords, filtered_stations, ev_specs
        )
        
        route_data = route_result
        
        # Extract route statistics
        route_stats = route_result.get('statistics', {})
        
        return jsonify({
            'success': True,
            'message': 'Route optimized successfully',
            'route_stats': route_stats,
            'has_ml_predictions': 'ml_predictions' in route_result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/generate_map', methods=['POST'])
def generate_map():
    global filtered_stations, route_data
    try:
        data = request.get_json()
        source_coords = (data['source_lat'], data['source_lon'])
        dest_coords = (data['dest_lat'], data['dest_lon'])
        
        map_visualizer = MapVisualizer()
        map_obj = map_visualizer.create_route_map(
            source_coords, dest_coords, filtered_stations, route_data
        )
        
        # Convert map to HTML
        map_html = map_obj._repr_html_()
        
        return jsonify({
            'success': True,
            'map_html': map_html
        })
    except Exception as e:
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
    app.run(host='0.0.0.0', port=5000, debug=True)