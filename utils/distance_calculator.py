import numpy as np
import pandas as pd
from geopy.distance import geodesic
# import streamlit as st

class DistanceCalculator:
    def __init__(self):
        pass
    
    def calculate_haversine_distance(self, coord1, coord2):
        """Calculate Haversine distance between two coordinates"""
        try:
            return geodesic(coord1, coord2).kilometers
        except Exception as e:
            print(f"Distance calculation failed: {str(e)}")
            return None
    
    def calculate_distance_matrix(self, coords_list):
        """Calculate distance matrix for a list of coordinates"""
        try:
            n = len(coords_list)
            distance_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i + 1, n):
                    dist = self.calculate_haversine_distance(coords_list[i], coords_list[j])
                    distance_matrix[i][j] = dist
                    distance_matrix[j][i] = dist
            
            return distance_matrix
        
        except Exception as e:
            print(f"Distance matrix calculation failed: {str(e)}")
            return None
    
    def find_nearest_stations(self, reference_coord, stations_df, n_stations=5):
        """Find the nearest N stations to a reference coordinate"""
        try:
            stations_df = stations_df.copy()
            
            # Calculate distances
            stations_df['distance'] = stations_df.apply(
                lambda row: self.calculate_haversine_distance(
                    reference_coord, (row['latitude'], row['longitude'])
                ),
                axis=1
            )
            
            # Sort by distance and return top N
            nearest_stations = stations_df.nsmallest(n_stations, 'distance')
            
            return nearest_stations
        
        except Exception as e:
            print(f"Nearest stations calculation failed: {str(e)}")
            return None
    
    def calculate_route_distance(self, waypoints):
        """Calculate total distance for a route with multiple waypoints"""
        try:
            if len(waypoints) < 2:
                return 0
            
            total_distance = 0
            for i in range(len(waypoints) - 1):
                distance = self.calculate_haversine_distance(waypoints[i], waypoints[i + 1])
                total_distance += distance
            
            return total_distance
        
        except Exception as e:
            print(f"Route distance calculation failed: {str(e)}")
            return None
    
    def calculate_bearing(self, coord1, coord2):
        """Calculate bearing between two coordinates"""
        try:
            lat1, lon1 = np.radians(coord1)
            lat2, lon2 = np.radians(coord2)
            
            dlon = lon2 - lon1
            
            y = np.sin(dlon) * np.cos(lat2)
            x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
            
            bearing = np.arctan2(y, x)
            bearing = np.degrees(bearing)
            bearing = (bearing + 360) % 360
            
            return bearing
        
        except Exception as e:
            print(f"Bearing calculation failed: {str(e)}")
            return None
    
    def calculate_midpoint(self, coord1, coord2):
        """Calculate midpoint between two coordinates"""
        try:
            lat1, lon1 = np.radians(coord1)
            lat2, lon2 = np.radians(coord2)
            
            dlon = lon2 - lon1
            
            Bx = np.cos(lat2) * np.cos(dlon)
            By = np.cos(lat2) * np.sin(dlon)
            
            lat3 = np.arctan2(np.sin(lat1) + np.sin(lat2),
                             np.sqrt((np.cos(lat1) + Bx) * (np.cos(lat1) + Bx) + By * By))
            lon3 = lon1 + np.arctan2(By, np.cos(lat1) + Bx)
            
            return (np.degrees(lat3), np.degrees(lon3))
        
        except Exception as e:
            print(f"Midpoint calculation failed: {str(e)}")
            return None
    
    def calculate_bounding_box(self, center_coord, radius_km):
        """Calculate bounding box around a center coordinate"""
        try:
            # Approximate conversion (1 degree ≈ 111 km)
            lat_delta = radius_km / 111
            lon_delta = radius_km / (111 * np.cos(np.radians(center_coord[0])))
            
            min_lat = center_coord[0] - lat_delta
            max_lat = center_coord[0] + lat_delta
            min_lon = center_coord[1] - lon_delta
            max_lon = center_coord[1] + lon_delta
            
            return {
                'min_lat': min_lat,
                'max_lat': max_lat,
                'min_lon': min_lon,
                'max_lon': max_lon
            }
        
        except Exception as e:
            print(f"Bounding box calculation failed: {str(e)}")
            return None
    
    def calculate_travel_time(self, distance_km, speed_kmh=80):
        """Calculate travel time based on distance and speed"""
        try:
            if distance_km <= 0:
                return 0
            
            travel_time_hours = distance_km / speed_kmh
            return travel_time_hours
        
        except Exception as e:
            print(f"Travel time calculation failed: {str(e)}")
            return None
    
    def calculate_energy_consumption(self, distance_km, consumption_rate_kwh_per_100km):
        """Calculate energy consumption for a given distance"""
        try:
            if distance_km <= 0:
                return 0
            
            energy_consumed = (distance_km / 100) * consumption_rate_kwh_per_100km
            return energy_consumed
        
        except Exception as e:
            print(f"Energy consumption calculation failed: {str(e)}")
            return None
    
    def calculate_detour_factor(self, source_coord, dest_coord, via_coord):
        """Calculate detour factor when going through an intermediate point"""
        try:
            # Direct distance
            direct_distance = self.calculate_haversine_distance(source_coord, dest_coord)
            
            # Distance via intermediate point
            via_distance = (
                self.calculate_haversine_distance(source_coord, via_coord) +
                self.calculate_haversine_distance(via_coord, dest_coord)
            )
            
            if direct_distance == 0:
                return float('inf')
            
            detour_factor = via_distance / direct_distance
            return detour_factor
        
        except Exception as e:
            print(f"Detour factor calculation failed: {str(e)}")
            return None
