import pandas as pd
import numpy as np
from geopy.distance import geodesic
import streamlit as st
from ml_models.station_predictor import StationPredictor

class StationFilters:
    def __init__(self):
        self.station_predictor = StationPredictor()
    
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
    
    def ml_based_filtering(self, stations_df, source_coords, dest_coords, filter_criteria=None):
        """Filter stations using machine learning predictions"""
        try:
            if filter_criteria is None:
                filter_criteria = {
                    'min_rating': 3.0,
                    'max_congestion': 1,  # 0=Low, 1=Medium, 2=High
                    'prefer_fast_charging': True,
                    'min_station_score': 3.0
                }
            
            # Calculate basic distance metrics
            stations_df_copy = stations_df.copy()
            stations_df_copy['distance_from_source'] = stations_df_copy.apply(
                lambda row: geodesic(source_coords, (row['latitude'], row['longitude'])).kilometers,
                axis=1
            )
            
            stations_df_copy['distance_from_dest'] = stations_df_copy.apply(
                lambda row: geodesic(dest_coords, (row['latitude'], row['longitude'])).kilometers,
                axis=1
            )
            
            # Get ML predictions
            ml_predictions = self.station_predictor.predict_station_metrics(
                stations_df_copy, 
                route_info=True, 
                time_info=True
            )
            
            if ml_predictions is None:
                st.warning("ML predictions unavailable, using basic filtering")
                return self.combined_filtering(stations_df, source_coords, dest_coords)
            
            # Apply ML-based filters
            filtered_stations = ml_predictions.copy()
            
            # Filter by predicted rating
            if 'predicted_rating' in filtered_stations.columns:
                filtered_stations = filtered_stations[
                    filtered_stations['predicted_rating'] >= filter_criteria['min_rating']
                ]
            
            # Filter by predicted congestion
            if 'predicted_congestion' in filtered_stations.columns:
                filtered_stations = filtered_stations[
                    filtered_stations['predicted_congestion'] <= filter_criteria['max_congestion']
                ]
            
            # Filter by ML station score
            if 'ml_station_score' in filtered_stations.columns:
                filtered_stations = filtered_stations[
                    filtered_stations['ml_station_score'] >= filter_criteria['min_station_score']
                ]
            
            # Prefer fast charging if requested
            if filter_criteria['prefer_fast_charging'] and 'dc_fast_ports' in filtered_stations.columns:
                # Prioritize stations with DC fast charging
                dc_stations = filtered_stations[
                    pd.to_numeric(filtered_stations['dc_fast_ports'], errors='coerce').fillna(0) > 0
                ]
                if not dc_stations.empty:
                    filtered_stations = dc_stations
            
            if len(filtered_stations) == 0:
                st.warning("No stations meet ML filtering criteria, relaxing constraints")
                # Relax constraints and try again
                relaxed_criteria = filter_criteria.copy()
                relaxed_criteria['min_rating'] = max(1.0, relaxed_criteria['min_rating'] - 1.0)
                relaxed_criteria['max_congestion'] = min(2, relaxed_criteria['max_congestion'] + 1)
                relaxed_criteria['min_station_score'] = max(1.0, relaxed_criteria['min_station_score'] - 1.0)
                
                return self.ml_based_filtering(stations_df, source_coords, dest_coords, relaxed_criteria)
            
            # Sort by ML station score
            filtered_stations = filtered_stations.sort_values('ml_station_score', ascending=False)
            
            return filtered_stations
            
        except Exception as e:
            st.error(f"ML-based filtering failed: {str(e)}")
            return self.combined_filtering(stations_df, source_coords, dest_coords)
    
    def station_type_filtering(self, stations_df, charging_types=None, power_levels=None):
        """Filter stations by charging type and power level"""
        try:
            if charging_types is None:
                charging_types = ['AC', 'DC']
            if power_levels is None:
                power_levels = ['Level1', 'Level2', 'DC_Fast']
            
            filtered_stations = stations_df.copy()
            
            # Filter by charging type
            if 'AC' not in charging_types:
                # Remove AC-only stations
                filtered_stations = filtered_stations[
                    pd.to_numeric(filtered_stations.get('dc_fast_ports', 0), errors='coerce').fillna(0) > 0
                ]
            
            if 'DC' not in charging_types:
                # Remove DC stations
                filtered_stations = filtered_stations[
                    pd.to_numeric(filtered_stations.get('dc_fast_ports', 0), errors='coerce').fillna(0) == 0
                ]
            
            # Filter by power level
            power_filter = pd.Series(False, index=filtered_stations.index)
            
            if 'Level1' in power_levels and 'level1_ports' in filtered_stations.columns:
                power_filter |= (
                    pd.to_numeric(filtered_stations['level1_ports'], errors='coerce').fillna(0) > 0
                )
            
            if 'Level2' in power_levels and 'level2_ports' in filtered_stations.columns:
                power_filter |= (
                    pd.to_numeric(filtered_stations['level2_ports'], errors='coerce').fillna(0) > 0
                )
            
            if 'DC_Fast' in power_levels and 'dc_fast_ports' in filtered_stations.columns:
                power_filter |= (
                    pd.to_numeric(filtered_stations['dc_fast_ports'], errors='coerce').fillna(0) > 0
                )
            
            # Apply power level filter
            filtered_stations = filtered_stations[power_filter]
            
            if len(filtered_stations) == 0:
                st.warning("No stations found matching charging type and power level criteria")
                return None
            
            return filtered_stations
            
        except Exception as e:
            st.error(f"Station type filtering failed: {str(e)}")
            return stations_df
    
    def smart_filtering(self, stations_df, source_coords, dest_coords, 
                       ev_specs=None, user_preferences=None):
        """Smart filtering combining all methods with ML optimization"""
        try:
            # Default preferences
            if user_preferences is None:
                user_preferences = {
                    'prefer_fast_charging': True,
                    'max_detour_factor': 1.5,
                    'corridor_width': 25,
                    'min_station_rating': 3.0,
                    'avoid_congestion': True,
                    'charging_types': ['AC', 'DC'],
                    'power_levels': ['Level2', 'DC_Fast']
                }
            
            # Start with geographic filtering
            filtered_stations = self.geographic_bounding_box(stations_df, source_coords, dest_coords)
            
            if filtered_stations is None:
                return None
            
            # Apply station type filtering
            filtered_stations = self.station_type_filtering(
                filtered_stations, 
                user_preferences.get('charging_types', ['AC', 'DC']),
                user_preferences.get('power_levels', ['Level2', 'DC_Fast'])
            )
            
            if filtered_stations is None:
                return None
            
            # Apply corridor filtering
            filtered_stations = self.corridor_based_filtering(
                filtered_stations, 
                source_coords, 
                dest_coords, 
                user_preferences.get('corridor_width', 25)
            )
            
            if filtered_stations is None:
                return None
            
            # Apply distance-based filtering with detour factor
            filtered_stations = self.distance_based_filtering(
                filtered_stations, 
                source_coords, 
                dest_coords, 
                user_preferences.get('max_detour_factor', 1.5)
            )
            
            if filtered_stations is None:
                return None
            
            # Apply ML-based filtering for final optimization
            ml_criteria = {
                'min_rating': user_preferences.get('min_station_rating', 3.0),
                'max_congestion': 1 if user_preferences.get('avoid_congestion', True) else 2,
                'prefer_fast_charging': user_preferences.get('prefer_fast_charging', True),
                'min_station_score': user_preferences.get('min_station_rating', 3.0)
            }
            
            final_filtered = self.ml_based_filtering(
                filtered_stations, 
                source_coords, 
                dest_coords, 
                ml_criteria
            )
            
            if final_filtered is None or len(final_filtered) == 0:
                st.warning("Smart filtering too restrictive, returning basic filtered results")
                return filtered_stations
            
            return final_filtered
            
        except Exception as e:
            st.error(f"Smart filtering failed: {str(e)}")
            return self.combined_filtering(stations_df, source_coords, dest_coords)
