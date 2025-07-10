import json
import pandas as pd
import streamlit as st

class GeoJSONExporter:
    def __init__(self):
        pass
    
    def export_stations(self, stations_df):
        """Export charging stations to GeoJSON format"""
        try:
            if stations_df is None or len(stations_df) == 0:
                return None
            
            # Create GeoJSON structure
            geojson_data = {
                "type": "FeatureCollection",
                "features": []
            }
            
            for idx, station in stations_df.iterrows():
                # Create feature for each station
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [station['longitude'], station['latitude']]
                    },
                    "properties": self._create_station_properties(station)
                }
                
                geojson_data["features"].append(feature)
            
            # Add metadata
            geojson_data["metadata"] = {
                "total_stations": len(stations_df),
                "export_timestamp": pd.Timestamp.now().isoformat(),
                "coordinate_system": "WGS84"
            }
            
            return geojson_data
        
        except Exception as e:
            st.error(f"GeoJSON export failed: {str(e)}")
            return None
    
    def _create_station_properties(self, station):
        """Create properties object for a station"""
        try:
            # Basic properties
            properties = {
                "name": station.get('name', 'EV Station'),
                "network": station.get('network', 'Unknown'),
                "access": station.get('access', 'Unknown'),
                "latitude": station['latitude'],
                "longitude": station['longitude']
            }
            
            # Add optional properties if available
            optional_fields = [
                'id', 'state', 'city', 'zip', 'street_address',
                'level1_ports', 'level2_ports', 'dc_fast_ports',
                'fuel_type', 'status', 'open_date'
            ]
            
            for field in optional_fields:
                if field in station and pd.notna(station[field]):
                    properties[field] = station[field]
            
            # Add calculated fields if available
            calculated_fields = [
                'distance_from_source', 'distance_from_dest',
                'corridor_distance', 'detour_ratio', 'cluster'
            ]
            
            for field in calculated_fields:
                if field in station and pd.notna(station[field]):
                    properties[field] = float(station[field])
            
            return properties
        
        except Exception as e:
            st.error(f"Station properties creation failed: {str(e)}")
            return {"name": "EV Station", "error": str(e)}
    
    def export_route(self, route_data):
        """Export route data to GeoJSON format"""
        try:
            if route_data is None or 'waypoints' not in route_data:
                return None
            
            # Create GeoJSON structure
            geojson_data = {
                "type": "FeatureCollection",
                "features": []
            }
            
            # Add route line
            coordinates = []
            for waypoint in route_data['waypoints']:
                coordinates.append([waypoint['coordinates'][1], waypoint['coordinates'][0]])
            
            route_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                },
                "properties": {
                    "type": "route",
                    "name": "EV Optimized Route",
                    "statistics": route_data.get('statistics', {})
                }
            }
            
            geojson_data["features"].append(route_feature)
            
            # Add waypoints as point features
            for i, waypoint in enumerate(route_data['waypoints']):
                waypoint_feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [waypoint['coordinates'][1], waypoint['coordinates'][0]]
                    },
                    "properties": {
                        "type": waypoint['type'],
                        "name": waypoint['name'],
                        "order": i
                    }
                }
                
                geojson_data["features"].append(waypoint_feature)
            
            # Add charging stops with detailed information
            if 'charging_stops' in route_data:
                for stop in route_data['charging_stops']:
                    stop_feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [stop['coordinates'][1], stop['coordinates'][0]]
                        },
                        "properties": {
                            "type": "charging_stop",
                            "stop_number": stop['stop_number'],
                            "station_name": stop['station_name'],
                            "distance_from_start": stop['distance_from_start'],
                            "battery_on_arrival": stop['battery_on_arrival'],
                            "charging_time": stop['charging_time']
                        }
                    }
                    
                    geojson_data["features"].append(stop_feature)
            
            # Add metadata
            geojson_data["metadata"] = {
                "route_type": "ev_optimized",
                "export_timestamp": pd.Timestamp.now().isoformat(),
                "coordinate_system": "WGS84",
                "total_features": len(geojson_data["features"])
            }
            
            return geojson_data
        
        except Exception as e:
            st.error(f"Route GeoJSON export failed: {str(e)}")
            return None
    
    def export_clusters(self, clustered_data):
        """Export cluster data to GeoJSON format"""
        try:
            if not clustered_data or 'stations' not in clustered_data:
                return None
            
            stations_df = clustered_data['stations']
            centroids = clustered_data['centroids']
            
            # Create GeoJSON structure
            geojson_data = {
                "type": "FeatureCollection",
                "features": []
            }
            
            # Add stations with cluster information
            for idx, station in stations_df.iterrows():
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [station['longitude'], station['latitude']]
                    },
                    "properties": self._create_station_properties(station)
                }
                
                geojson_data["features"].append(feature)
            
            # Add cluster centroids
            for cluster_id, centroid in centroids.items():
                centroid_feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [centroid['longitude'], centroid['latitude']]
                    },
                    "properties": {
                        "type": "cluster_centroid",
                        "cluster_id": cluster_id,
                        "station_count": centroid['station_count'],
                        "name": f"Cluster {cluster_id} Centroid"
                    }
                }
                
                geojson_data["features"].append(centroid_feature)
            
            # Add metadata
            geojson_data["metadata"] = {
                "data_type": "clustered_stations",
                "export_timestamp": pd.Timestamp.now().isoformat(),
                "coordinate_system": "WGS84",
                "total_clusters": len(centroids),
                "total_stations": len(stations_df)
            }
            
            return geojson_data
        
        except Exception as e:
            st.error(f"Cluster GeoJSON export failed: {str(e)}")
            return None
    
    def save_to_file(self, geojson_data, filename):
        """Save GeoJSON data to file"""
        try:
            if geojson_data is None:
                return False
            
            with open(filename, 'w') as f:
                json.dump(geojson_data, f, indent=2)
            
            return True
        
        except Exception as e:
            st.error(f"File save failed: {str(e)}")
            return False
    
    def validate_geojson(self, geojson_data):
        """Validate GeoJSON data structure"""
        try:
            if not isinstance(geojson_data, dict):
                return False, "GeoJSON must be a dictionary"
            
            if geojson_data.get('type') != 'FeatureCollection':
                return False, "GeoJSON must be a FeatureCollection"
            
            if 'features' not in geojson_data:
                return False, "GeoJSON must contain features array"
            
            if not isinstance(geojson_data['features'], list):
                return False, "Features must be an array"
            
            # Validate each feature
            for i, feature in enumerate(geojson_data['features']):
                if not isinstance(feature, dict):
                    return False, f"Feature {i} must be a dictionary"
                
                if feature.get('type') != 'Feature':
                    return False, f"Feature {i} must have type 'Feature'"
                
                if 'geometry' not in feature:
                    return False, f"Feature {i} must have geometry"
                
                if 'properties' not in feature:
                    return False, f"Feature {i} must have properties"
            
            return True, "GeoJSON validation successful"
        
        except Exception as e:
            return False, f"GeoJSON validation failed: {str(e)}"
