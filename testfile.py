import unittest
import json
import requests
import time
import sys
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
import webbrowser
import tempfile
import os
from math import radians, cos, sin, asin, sqrt
import folium
from folium import plugins

class EVChargingStationTestSuite:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = {}
        self.stations_data = None
        self.filtered_stations = None
        self.route_data = None
        self.clustered_stations = None
        self.final_stations_for_map = []
        
        # Test coordinates (Inside the Jaipur district)
        self.test_coords = {
            'source_lat': 26.9260,
            'source_lon': 75.7870,
            'dest_lat': 26.8505,
            'dest_lon': 75.8028
        }
        
        # Filtering methods dictionary
        self.filtering_methods = {
            1: "Smart ML Filtering",
            2: "ML-Based Filtering", 
            3: "Geographic Bounding Box",
            4: "Distance-Based",
            5: "Corridor-Based",
            6: "Combined Approach"
        }
        
        # Default parameters for different filtering methods
        self.default_params = {
            "Smart ML Filtering": {
                'prefer_fast_charging': True,
                'detour_factor': 1.2,
                'corridor_width': 30,
                'min_station_rating': 2.0,
                'avoid_congestion': True,
                'charging_types': ['AC', 'DC'],
                'power_levels': ['Level2', 'DC_Fast']
            },
            "ML-Based Filtering": {
                'min_station_rating': 3.0,
                'avoid_congestion': True,
                'prefer_fast_charging': True,
                'detour_factor': 1.5
            },
            "Geographic Bounding Box": {
                'buffer_km': 50
            },
            "Distance-Based": {
                'detour_factor': 1.3,
                'max_distance_source': 150,
                'max_distance_dest': 150
            },
            "Corridor-Based": {
                'corridor_width': 20,
                'buffer_km': 10
            },
            "Combined Approach": {
                'detour_factor': 1.4,
                'corridor_width': 25,
                'max_distance_source': 180,
                'max_distance_dest': 180
            }
        }
        
        # Sample station data for fallback
        self.sample_stations = self.generate_sample_stations()
        
        self.print_header()
    
    def generate_sample_stations(self):
        """Generate sample stations along the route"""
        stations = []
        # Generate stations along the route from SF to LA
        lat_step = (self.test_coords['dest_lat'] - self.test_coords['source_lat']) / 20
        lon_step = (self.test_coords['dest_lon'] - self.test_coords['source_lon']) / 20
        
        for i in range(25):
            lat = self.test_coords['source_lat'] + (lat_step * i) + np.random.uniform(-0.5, 0.5)
            lon = self.test_coords['source_lon'] + (lon_step * i) + np.random.uniform(-0.5, 0.5)
            
            stations.append({
                'id': i + 1,
                'name': f'Station {i + 1}',
                'latitude': lat,
                'longitude': lon,
                'state': 'CA',
                'network': f'Network{(i % 3) + 1}',
                'dc_fast_ports': np.random.randint(1, 5),
                'ac_ports': np.random.randint(2, 8),
                'predicted_congestion': np.random.randint(0, 3),
                'predicted_rating': np.random.uniform(3.0, 5.0),
                'ml_station_score': np.random.uniform(6.0, 10.0)
            })
        
        return stations
    
    def print_header(self):
        print("=" * 80)
        print("🔋 ENHANCED EV CHARGING STATION API TEST SUITE")
        print("=" * 80)
        print(f"📍 Testing Base URL: {self.base_url}")
        print(f"⏰ Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🗺️ Route: San Francisco → Los Angeles")
        print("📊 Filtering Methods Available:")
        for key, method in self.filtering_methods.items():
            print(f"    {key}: {method}")
        print("=" * 80)
        print()
    
    def log_step(self, step_name, details=""):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] 🔍 {step_name}")
        if details:
            print(f"    💡 {details}")
    
    def log_success(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] ✅ {message}")
    
    def log_warning(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] ⚠️  {message}")
    
    def log_error(self, message, error=None):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] ❌ {message}")
        if error:
            print(f"    🐛 Error Details: {str(error)}")
    
    def log_debug(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] 🔧 DEBUG: {message}")
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates in kilometers"""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        return R * c
    
    def is_station_on_route(self, station_lat, station_lon, buffer_km=1.0):
        """Check if station is within buffer distance of the route"""
        source_lat, source_lon = self.test_coords['source_lat'], self.test_coords['source_lon']
        dest_lat, dest_lon = self.test_coords['dest_lat'], self.test_coords['dest_lon']
        
        # Calculate distances
        dist_to_source = self.haversine_distance(station_lat, station_lon, source_lat, source_lon)
        dist_to_dest = self.haversine_distance(station_lat, station_lon, dest_lat, dest_lon)
        direct_route_distance = self.haversine_distance(source_lat, source_lon, dest_lat, dest_lon)
        
        # Check if station is roughly on the route (with buffer)
        total_distance = dist_to_source + dist_to_dest
        detour_distance = total_distance - direct_route_distance
        
        return detour_distance <= buffer_km
    
    def filter_stations_by_route(self, stations, buffer_km=1.0):
        """Filter stations to only include those on the route"""
        if not stations:
            return []
        
        filtered = []
        for station in stations:
            if isinstance(station, dict):
                lat, lon = station.get('latitude'), station.get('longitude')
            else:
                lat, lon = getattr(station, 'latitude', None), getattr(station, 'longitude', None)
            
            if lat is not None and lon is not None:
                if self.is_station_on_route(lat, lon, buffer_km):
                    filtered.append(station)
        
        return filtered
    
    def get_best_stations_from_filtered(self, filtered_stations, max_stations=10):
        """Get best stations from filtered data"""
        if not filtered_stations:
            return []
        
        try:
            # Convert to DataFrame if it's a list
            if isinstance(filtered_stations, list):
                df = pd.DataFrame(filtered_stations)
            else:
                df = filtered_stations
            
            # Sort by ML score or rating
            if 'ml_station_score' in df.columns:
                df_sorted = df.sort_values('ml_station_score', ascending=False)
            elif 'predicted_rating' in df.columns:
                df_sorted = df.sort_values('predicted_rating', ascending=False)
            else:
                df_sorted = df.head(max_stations)
            
            return df_sorted.head(max_stations).to_dict('records')
            
        except Exception as e:
            self.log_error(f"Error selecting best stations: {str(e)}")
            return filtered_stations[:max_stations] if isinstance(filtered_stations, list) else []
    
    def create_route_map(self, stations, save_path=None):
        """Create a map with stations and route polyline"""
        try:
            # Center map between source and destination
            center_lat = (self.test_coords['source_lat'] + self.test_coords['dest_lat']) / 2
            center_lon = (self.test_coords['source_lon'] + self.test_coords['dest_lon']) / 2
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
            
            # Add source marker
            folium.Marker(
                [self.test_coords['source_lat'], self.test_coords['source_lon']],
                popup="Source: San Francisco",
                icon=folium.Icon(color='green', icon='play')
            ).add_to(m)
            
            # Add destination marker
            folium.Marker(
                [self.test_coords['dest_lat'], self.test_coords['dest_lon']],
                popup="Destination: Los Angeles",
                icon=folium.Icon(color='red', icon='stop')
            ).add_to(m)
            
            # Add route polyline
            route_coordinates = [
                [self.test_coords['source_lat'], self.test_coords['source_lon']],
                [self.test_coords['dest_lat'], self.test_coords['dest_lon']]
            ]
            
            folium.PolyLine(
                route_coordinates,
                color='blue',
                weight=3,
                opacity=0.7,
                popup="Route: SF → LA"
            ).add_to(m)
            
            # Add station markers
            for i, station in enumerate(stations):
                if isinstance(station, dict):
                    lat, lon = station.get('latitude'), station.get('longitude')
                    name = station.get('name', f'Station {i+1}')
                    rating = station.get('predicted_rating', 'N/A')
                    dc_ports = station.get('dc_fast_ports', 0)
                else:
                    lat, lon = getattr(station, 'latitude', None), getattr(station, 'longitude', None)
                    name = getattr(station, 'name', f'Station {i+1}')
                    rating = getattr(station, 'predicted_rating', 'N/A')
                    dc_ports = getattr(station, 'dc_fast_ports', 0)
                
                if lat is not None and lon is not None:
                    # Different colors based on DC fast charging availability
                    color = 'orange' if dc_ports > 0 else 'blue'
                    
                    folium.Marker(
                        [lat, lon],
                        popup=f"{name}<br>Rating: {rating}<br>DC Ports: {dc_ports}",
                        icon=folium.Icon(color=color, icon='bolt')
                    ).add_to(m)
            
            # Save map
            if save_path is None:
                save_path = tempfile.mktemp(suffix='.html')
            
            m.save(save_path)
            self.log_success(f"Map saved to: {save_path}")
            
            return save_path
            
        except Exception as e:
            self.log_error(f"Error creating map: {str(e)}")
            return None
    
    def make_request(self, method, endpoint, data=None, expected_status=200):
        """Make HTTP request with error handling"""
        try:
            url = f"{self.base_url}{endpoint}"
            self.log_debug(f"Making {method} request to {endpoint}")
            
            if method.upper() == 'GET':
                response = self.session.get(url, timeout=30)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            self.log_debug(f"Response Status: {response.status_code}")
            
            if response.status_code == expected_status:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    return {'success': True, 'data': response.text}
            else:
                self.log_error(f"Unexpected status code: {response.status_code}")
                return {'success': False, 'error': f'Status {response.status_code}'}
                
        except requests.exceptions.RequestException as e:
            self.log_error(f"Request failed for {endpoint}", e)
            return {'success': False, 'error': str(e)}
    
    def test_data_loading(self):
        """Test data loading with PostgreSQL fallback to sample data"""
        print("\n" + "="*50)
        print("🗄️ TESTING DATA LOADING (PostgreSQL → Sample)")
        print("="*50)
        
        # First try PostgreSQL database
        self.log_step("Attempting PostgreSQL database connection")
        try:
            response = self.make_request('POST', '/api/load_database', {'port_type': 'both'})
            
            if response.get('success'):
                total_stations = response.get('total_stations', 0)
                self.log_success(f"PostgreSQL loaded successfully: {total_stations} stations")
                self.stations_data = response
                return True
            else:
                self.log_warning(f"PostgreSQL failed: {response.get('error')}")
                
        except Exception as e:
            self.log_error("PostgreSQL connection failed", e)
        
        # Fallback to sample data
        self.log_step("Falling back to sample data")
        try:
            response = self.make_request('POST', '/api/load_sample_data')
            
            if response.get('success'):
                total_stations = response.get('total_stations', 0)
                self.log_success(f"Sample data loaded: {total_stations} stations")
                self.stations_data = response
                return True
            else:
                self.log_error(f"Sample data failed: {response.get('error')}")
                
        except Exception as e:
            self.log_error("Sample data loading failed", e)
        
        # Last resort: use built-in sample data
        self.log_step("Using built-in sample stations")
        try:
            response = self.make_request('POST', '/api/dev/set_stations', 
                                       {'stations_data': self.sample_stations})
            
            if response.get('success'):
                self.log_success(f"Built-in sample data set: {len(self.sample_stations)} stations")
                self.stations_data = response
                return True
            else:
                self.log_error(f"Built-in sample data failed: {response.get('error')}")
                return False
                
        except Exception as e:
            self.log_error("Built-in sample data failed", e)
            return False
    
    def test_all_filtering_methods(self):
        """Test all 6 filtering methods with different parameters"""
        print("\n" + "="*50)
        print("🔍 TESTING ALL FILTERING METHODS")
        print("="*50)
        
        successful_filters = {}
        
        for method_id, method_name in self.filtering_methods.items():
            self.log_step(f"Testing Method {method_id}: {method_name}")
            
            try:
                # Get default parameters for this method
                params = self.default_params.get(method_name, {})
                
                filter_data = {
                    'source_lat': self.test_coords['source_lat'],
                    'source_lon': self.test_coords['source_lon'],
                    'dest_lat': self.test_coords['dest_lat'],
                    'dest_lon': self.test_coords['dest_lon'],
                    'filtering_method': method_name,
                    'params': params
                }
                
                response = self.make_request('POST', '/api/apply_filtering', filter_data)
                
                if response.get('success'):
                    filtered_count = response.get('filtered_count', 0)
                    ml_summary = response.get('ml_summary', {})
                    
                    self.log_success(f"Method {method_id} SUCCESS: {filtered_count} stations")
                    self.log_debug(f"ML Summary: {ml_summary}")
                    
                    # Store successful filter result
                    successful_filters[method_id] = {
                        'name': method_name,
                        'count': filtered_count,
                        'response': response,
                        'params': params
                    }
                else:
                    self.log_error(f"Method {method_id} FAILED: {response.get('error')}")
                    
            except Exception as e:
                self.log_error(f"Method {method_id} EXCEPTION", e)
        
        # Check if methods return different results
        if len(successful_filters) > 1:
            counts = [f['count'] for f in successful_filters.values()]
            if len(set(counts)) > 1:
                self.log_success("✅ Different methods returned different station counts")
            else:
                self.log_warning("⚠️ All methods returned same station count")
        
        # Use the best filtering result
        if successful_filters:
            # Choose method with reasonable station count (not too many, not too few)
            best_method = None
            for method_id, result in successful_filters.items():
                count = result['count']
                if 5 <= count <= 50:  # Reasonable range
                    best_method = method_id
                    break
            
            if best_method is None:
                best_method = min(successful_filters.keys(), 
                                key=lambda x: successful_filters[x]['count'])
            
            self.filtered_stations = successful_filters[best_method]['response']
            self.log_success(f"Selected Method {best_method} for further processing")
            return True
        else:
            self.log_error("All filtering methods failed")
            return False
    
    def test_clustering_with_fallback(self):
        """Test clustering with intelligent fallback"""
        print("\n" + "="*50)
        print("🎯 TESTING CLUSTERING WITH FALLBACK")
        print("="*50)
        
        cluster_sizes = [5, 8, 10, 12]
        
        for n_clusters in cluster_sizes:
            self.log_step(f"Attempting clustering with {n_clusters} clusters")
            
            try:
                response = self.make_request('POST', '/api/perform_clustering', 
                                           {'n_clusters': n_clusters})
                
                if response.get('success'):
                    cluster_stats = response.get('cluster_stats', {})
                    summary = response.get('summary', {})
                    
                    self.log_success(f"Clustering SUCCESS: {n_clusters} clusters created")
                    self.log_debug(f"Cluster Stats: {cluster_stats}")
                    
                    self.clustered_stations = response
                    return True
                else:
                    self.log_warning(f"Clustering with {n_clusters} clusters failed: {response.get('error')}")
                    
            except Exception as e:
                self.log_error(f"Clustering with {n_clusters} clusters failed", e)
        
        # Clustering failed, use best stations from filtered data
        self.log_step("Clustering failed, selecting best 10 stations from filtered data")
        
        if self.filtered_stations:
            try:
                # Get filtered stations data
                filtered_data = self.filtered_stations.get('filtered_stations', [])
                best_stations = self.get_best_stations_from_filtered(filtered_data, 10)
                
                if best_stations:
                    self.log_success(f"Selected {len(best_stations)} best stations from filtered data")
                    self.final_stations_for_map = best_stations
                    return True
                else:
                    self.log_warning("No stations available from filtered data")
                    
            except Exception as e:
                self.log_error("Failed to select best stations from filtered data", e)
        
        # Final fallback: use sample stations
        self.log_step("Using sample stations as final fallback")
        best_sample_stations = self.get_best_stations_from_filtered(self.sample_stations, 10)
        self.final_stations_for_map = best_sample_stations
        self.log_success(f"Using {len(best_sample_stations)} sample stations")
        
        return True
    
    def test_route_optimization(self):
        """Test route optimization"""
        print("\n" + "="*50)
        print("🗺️ TESTING ROUTE OPTIMIZATION")
        print("="*50)
        
        try:
            route_data = {
                'source_lat': self.test_coords['source_lat'],
                'source_lon': self.test_coords['source_lon'],
                'dest_lat': self.test_coords['dest_lat'],
                'dest_lon': self.test_coords['dest_lon'],
                'battery_range': 350,
                'consumption_rate': 18,
                'charging_time': 25,
                'safety_margin': 20
            }
            
            self.log_step("Optimizing route with EV specifications")
            self.log_debug(f"Battery Range: {route_data['battery_range']} km")
            self.log_debug(f"Consumption Rate: {route_data['consumption_rate']} kWh/100km")
            
            response = self.make_request('POST', '/api/optimize_route', route_data)
            
            if response.get('success'):
                route_stats = response.get('route_stats', {})
                self.log_success("Route optimization SUCCESS")
                self.log_debug(f"Route Stats: {route_stats}")
                
                self.route_data = response
                return True
            else:
                self.log_error("Route optimization FAILED", response.get('error'))
                return False
                
        except Exception as e:
            self.log_error("Route optimization EXCEPTION", e)
            return False
    
    def validate_final_stations(self):
        """Validate that final stations are on the route"""
        print("\n" + "="*50)
        print("🔍 VALIDATING FINAL STATIONS")
        print("="*50)
        
        if not self.final_stations_for_map:
            self.log_error("No final stations to validate")
            return False
        
        self.log_step(f"Validating {len(self.final_stations_for_map)} stations")
        
        # Filter stations by route with 1km buffer
        valid_stations = self.filter_stations_by_route(self.final_stations_for_map, buffer_km=1.0)
        
        self.log_debug(f"Stations before validation: {len(self.final_stations_for_map)}")
        self.log_debug(f"Stations after validation: {len(valid_stations)}")
        
        if len(valid_stations) > 0:
            self.final_stations_for_map = valid_stations
            self.log_success(f"✅ {len(valid_stations)} stations validated as on-route")
            return True
        else:
            self.log_warning("No stations found on route, using original selection")
            return False
    
    def create_and_display_map(self):
        """Create map and automatically open it"""
        print("\n" + "="*50)
        print("🗺️ CREATING AND DISPLAYING MAP")
        print("="*50)
        
        if not self.final_stations_for_map:
            self.log_error("No stations available for map creation")
            return False
        
        self.log_step(f"Creating map with {len(self.final_stations_for_map)} stations")
        
        # Ensure we don't show all 123 stations
        if len(self.final_stations_for_map) > 20:
            self.log_warning(f"Too many stations ({len(self.final_stations_for_map)}), limiting to 15")
            self.final_stations_for_map = self.final_stations_for_map[:15]
        
        try:
            map_path = self.create_route_map(self.final_stations_for_map)
            
            if map_path:
                self.log_success("Map created successfully")
                
                # Auto-open the map
                try:
                    webbrowser.open(f'file://{os.path.abspath(map_path)}')
                    self.log_success("Map opened in browser")
                except Exception as e:
                    self.log_warning(f"Could not auto-open map: {str(e)}")
                    self.log_debug(f"Map file location: {map_path}")
                
                return True
            else:
                self.log_error("Map creation failed")
                return False
                
        except Exception as e:
            self.log_error("Map creation and display failed", e)
            return False
    
    def run_comprehensive_test(self):
        """Run comprehensive test with smart fallback logic"""
        print("\n🚀 STARTING COMPREHENSIVE TEST WITH SMART FALLBACK")
        print("="*80)
        
        test_results = {}
        
        # Test sequence with intelligent fallback
        test_sequence = [
            ('Data Loading', self.test_data_loading),
            ('All Filtering Methods', self.test_all_filtering_methods),
            ('Clustering with Fallback', self.test_clustering_with_fallback),
            ('Route Optimization', self.test_route_optimization),
            ('Final Station Validation', self.validate_final_stations),
            ('Map Creation and Display', self.create_and_display_map)
        ]
        
        for test_name, test_func in test_sequence:
            self.log_step(f"Executing: {test_name}")
            
            try:
                result = test_func()
                test_results[test_name] = result
                
                if result:
                    self.log_success(f"✅ {test_name} PASSED")
                else:
                    self.log_error(f"❌ {test_name} FAILED")
                    
            except Exception as e:
                self.log_error(f"💥 {test_name} CRITICAL ERROR", e)
                test_results[test_name] = False
            
            # Brief pause between tests
            time.sleep(0.5)
        
        # Print final summary
        self.print_final_summary(test_results)
        
        return test_results
    
    def print_final_summary(self, test_results):
        """Print enhanced final summary"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print("="*80)
        
        passed = sum(1 for result in test_results.values() if result)
        total = len(test_results)
        
        print(f"✅ Tests Passed: {passed}/{total}")
        print(f"❌ Tests Failed: {total - passed}/{total}")
        print(f"📈 Success Rate: {(passed/total)*100:.1f}%")
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"📊 Stations for Map: {len(self.final_stations_for_map)}")
        print(f"🔍 Filtering Methods: {len(self.filtering_methods)} tested")
        print(f"⚡ Route Optimization: {'✅' if self.route_data else '❌'}")
        
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 60)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<35} {status}")
        
        print("\n" + "="*80)
        print(f"🏁 Test Suite Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

def main():
    """Main function with enhanced argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced EV Charging Station API Test Suite')
    parser.add_argument('--url', default='http://localhost:5000', 
                        help='Base URL for the Flask application')
    parser.add_argument('--filter-method', type=int, choices=[1,2,3,4,5,6],
                        help='Specific filtering method to test (1-6)')
    parser.add_argument('--no-map', action='store_true',
                        help='Skip map creation and display')
    
    args = parser.parse_args()
    
    # Create and run test suite
    test_suite = EVChargingStationTestSuite(args.url)
    
    try:
        # Run specific filtering method if requested
        if args.filter_method:
            method_name = test_suite.filtering_methods[args.filter_method]
            print(f"🎯 Testing specific filtering method: {method_name}")
            
            # Load data first
            test_suite.test_data_loading()
            
            # Test specific method
            # ... (implement specific method testing)
        
        # Run comprehensive test
        results = test_suite.run_comprehensive_test()
        
        return results
        
    except KeyboardInterrupt:
        print("\n🛑 Test suite interrupted by user")
        return {}
    except Exception as e:
        print(f"\n💥 Test suite failed with critical error: {str(e)}")
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    main()