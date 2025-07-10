import pandas as pd
import numpy as np
from geopy.distance import geodesic
import streamlit as st

class StationFilters:
    def __init__(self):
        pass
    
    def geographic_bounding_box(self, stations_df, source_coords, dest_coords):
        """Filter stations within a geographic bounding box between source and destination"""
        try:
            # Calculate bounding box with some padding
            min_lat = min(source_coords[0], dest_coords[0]) - 0.5
            max_lat = max(source_coords[0], dest_coords[0]) + 0.5
            min_lon = min(source_coords[1], dest_coords[1]) - 0.5
            max_lon = max(source_coords[1], dest_coords[1]) + 0.5
            
            # Filter stations within bounding box
            filtered_stations = stations_df[
                (stations_df['latitude'] >= min_lat) &
                (stations_df['latitude'] <= max_lat) &
                (stations_df['longitude'] >= min_lon) &
                (stations_df['longitude'] <= max_lon)
            ].copy()
            
            if len(filtered_stations) == 0:
                st.warning("No stations found within geographic bounding box")
                return None
            
            # Calculate distances from source and destination
            filtered_stations['distance_from_source'] = filtered_stations.apply(
                lambda row: geodesic(source_coords, (row['latitude'], row['longitude'])).kilometers,
                axis=1
            )
            
            filtered_stations['distance_from_dest'] = filtered_stations.apply(
                lambda row: geodesic(dest_coords, (row['latitude'], row['longitude'])).kilometers,
                axis=1
            )
            
            # Sort by distance from source
            filtered_stations = filtered_stations.sort_values('distance_from_source')
            
            return filtered_stations
        
        except Exception as e:
            st.error(f"Geographic bounding box filtering failed: {str(e)}")
            return None
    
    def distance_based_filtering(self, stations_df, source_coords, dest_coords, 
                                detour_factor=1.5, max_distance_source=200, max_distance_dest=200):
        """Filter stations based on distance constraints and detour factor"""
        try:
            # Calculate direct distance between source and destination
            direct_distance = geodesic(source_coords, dest_coords).kilometers
            
            # Calculate distances from source and destination for all stations
            stations_df_copy = stations_df.copy()
            
            stations_df_copy['distance_from_source'] = stations_df_copy.apply(
                lambda row: geodesic(source_coords, (row['latitude'], row['longitude'])).kilometers,
                axis=1
            )
            
            stations_df_copy['distance_from_dest'] = stations_df_copy.apply(
                lambda row: geodesic(dest_coords, (row['latitude'], row['longitude'])).kilometers,
                axis=1
            )
            
            # Apply distance constraints
            filtered_stations = stations_df_copy[
                (stations_df_copy['distance_from_source'] <= max_distance_source) &
                (stations_df_copy['distance_from_dest'] <= max_distance_dest)
            ].copy()
            
            if len(filtered_stations) == 0:
                st.warning("No stations found within distance constraints")
                return None
            
            # Apply detour factor constraint
            # A station is acceptable if source->station->dest <= detour_factor * direct_distance
            filtered_stations['total_detour_distance'] = (
                filtered_stations['distance_from_source'] + 
                filtered_stations['distance_from_dest']
            )
            
            filtered_stations['detour_ratio'] = (
                filtered_stations['total_detour_distance'] / direct_distance
            )
            
            # Filter by detour factor
            filtered_stations = filtered_stations[
                filtered_stations['detour_ratio'] <= detour_factor
            ]
            
            if len(filtered_stations) == 0:
                st.warning(f"No stations found within detour factor of {detour_factor}")
                return None
            
            # Sort by detour ratio (most efficient first)
            filtered_stations = filtered_stations.sort_values('detour_ratio')
            
            return filtered_stations
        
        except Exception as e:
            st.error(f"Distance-based filtering failed: {str(e)}")
            return None
    
    def corridor_based_filtering(self, stations_df, source_coords, dest_coords, corridor_width=25):
        """Filter stations within a corridor around the direct route"""
        try:
            stations_df_copy = stations_df.copy()
            
            # Calculate perpendicular distance from each station to the direct route line
            stations_df_copy['corridor_distance'] = stations_df_copy.apply(
                lambda row: self._point_to_line_distance(
                    (row['latitude'], row['longitude']), source_coords, dest_coords
                ),
                axis=1
            )
            
            # Filter stations within corridor
            filtered_stations = stations_df_copy[
                stations_df_copy['corridor_distance'] <= corridor_width
            ].copy()
            
            if len(filtered_stations) == 0:
                st.warning(f"No stations found within corridor width of {corridor_width} km")
                return None
            
            # Calculate distances from source and destination
            filtered_stations['distance_from_source'] = filtered_stations.apply(
                lambda row: geodesic(source_coords, (row['latitude'], row['longitude'])).kilometers,
                axis=1
            )
            
            filtered_stations['distance_from_dest'] = filtered_stations.apply(
                lambda row: geodesic(dest_coords, (row['latitude'], row['longitude'])).kilometers,
                axis=1
            )
            
            # Sort by distance from source
            filtered_stations = filtered_stations.sort_values('distance_from_source')
            
            return filtered_stations
        
        except Exception as e:
            st.error(f"Corridor-based filtering failed: {str(e)}")
            return None
    
    def combined_filtering(self, stations_df, source_coords, dest_coords, 
                          detour_factor=1.5, corridor_width=25, 
                          max_distance_source=200, max_distance_dest=200):
        """Apply multiple filtering methods in combination"""
        try:
            # Start with geographic bounding box
            filtered_stations = self.geographic_bounding_box(stations_df, source_coords, dest_coords)
            
            if filtered_stations is None:
                return None
            
            # Apply corridor-based filtering
            filtered_stations = self.corridor_based_filtering(
                filtered_stations, source_coords, dest_coords, corridor_width
            )
            
            if filtered_stations is None:
                return None
            
            # Apply distance-based filtering
            filtered_stations = self.distance_based_filtering(
                filtered_stations, source_coords, dest_coords, 
                detour_factor, max_distance_source, max_distance_dest
            )
            
            if filtered_stations is None:
                return None
            
            # Add combined score for ranking
            # Lower score is better (combination of detour ratio and corridor distance)
            filtered_stations['combined_score'] = (
                filtered_stations['detour_ratio'] * 0.7 + 
                (filtered_stations['corridor_distance'] / corridor_width) * 0.3
            )
            
            # Sort by combined score
            filtered_stations = filtered_stations.sort_values('combined_score')
            
            return filtered_stations
        
        except Exception as e:
            st.error(f"Combined filtering failed: {str(e)}")
            return None
    
    def _point_to_line_distance(self, point, line_start, line_end):
        """Calculate the perpendicular distance from a point to a line segment"""
        try:
            # Convert to numpy arrays for easier calculation
            point = np.array([point[0], point[1]])
            line_start = np.array([line_start[0], line_start[1]])
            line_end = np.array([line_end[0], line_end[1]])
            
            # Calculate line vector
            line_vec = line_end - line_start
            
            # Calculate point vector
            point_vec = point - line_start
            
            # Calculate projection of point onto line
            line_len = np.linalg.norm(line_vec)
            
            if line_len == 0:
                # Line start and end are the same point
                return geodesic(point, line_start).kilometers
            
            # Normalize line vector
            line_unit = line_vec / line_len
            
            # Calculate projection length
            proj_length = np.dot(point_vec, line_unit)
            
            # Clamp projection to line segment
            proj_length = max(0, min(proj_length, line_len))
            
            # Calculate closest point on line
            closest_point = line_start + proj_length * line_unit
            
            # Calculate distance using geodesic
            distance = geodesic(point, closest_point).kilometers
            
            return distance
        
        except Exception as e:
            # Fallback to simple distance calculation
            dist_to_start = geodesic(point, line_start).kilometers
            dist_to_end = geodesic(point, line_end).kilometers
            return min(dist_to_start, dist_to_end)
