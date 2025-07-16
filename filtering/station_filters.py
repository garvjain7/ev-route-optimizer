# import pandas as pd
# import numpy as np
# from geopy.distance import geodesic
# # import streamlit as st
# from ml_models.station_predictor import StationPredictor

# class StationFilters:
#     def __init__(self):
#         self.station_predictor = StationPredictor()
    
#     def geographic_bounding_box(self, stations_df, source_coords, dest_coords):
#         """Filter stations within a geographic bounding box between source and destination"""
#         try:
#             # Calculate bounding box with some padding
#             min_lat = min(source_coords[0], dest_coords[0]) - 0.5
#             max_lat = max(source_coords[0], dest_coords[0]) + 0.5
#             min_lon = min(source_coords[1], dest_coords[1]) - 0.5
#             max_lon = max(source_coords[1], dest_coords[1]) + 0.5
            
#             # Filter stations within bounding box
#             filtered_stations = stations_df[
#                 (stations_df['latitude'] >= min_lat) &
#                 (stations_df['latitude'] <= max_lat) &
#                 (stations_df['longitude'] >= min_lon) &
#                 (stations_df['longitude'] <= max_lon)
#             ].copy()
            
#             if len(filtered_stations) == 0:
#                 print("No stations found within geographic bounding box")
#                 return None
            
#             # Calculate distances from source and destination
#             filtered_stations['distance_from_source'] = filtered_stations.apply(
#                 lambda row: geodesic(source_coords, (row['latitude'], row['longitude'])).kilometers,
#                 axis=1
#             )
            
#             filtered_stations['distance_from_dest'] = filtered_stations.apply(
#                 lambda row: geodesic(dest_coords, (row['latitude'], row['longitude'])).kilometers,
#                 axis=1
#             )
            
#             # Sort by distance from source
#             filtered_stations = filtered_stations.sort_values('distance_from_source')
            
#             return filtered_stations
        
#         except Exception as e:
#             print(f"Geographic bounding box filtering failed: {str(e)}")
#             return None
    
#     def distance_based_filtering(self, stations_df, source_coords, dest_coords, 
#                                 detour_factor=1.5, max_distance_source=200, max_distance_dest=200):
#         """Filter stations based on distance constraints and detour factor"""
#         try:
#             # Calculate direct distance between source and destination
#             direct_distance = geodesic(source_coords, dest_coords).kilometers
            
#             # Calculate distances from source and destination for all stations
#             stations_df_copy = stations_df.copy()
            
#             stations_df_copy['distance_from_source'] = stations_df_copy.apply(
#                 lambda row: geodesic(source_coords, (row['latitude'], row['longitude'])).kilometers,
#                 axis=1
#             )
            
#             stations_df_copy['distance_from_dest'] = stations_df_copy.apply(
#                 lambda row: geodesic(dest_coords, (row['latitude'], row['longitude'])).kilometers,
#                 axis=1
#             )
            
#             # Apply distance constraints
#             filtered_stations = stations_df_copy[
#                 (stations_df_copy['distance_from_source'] <= max_distance_source) &
#                 (stations_df_copy['distance_from_dest'] <= max_distance_dest)
#             ].copy()
            
#             if len(filtered_stations) == 0:
#                 print("No stations found within distance constraints")
#                 return None
            
#             # Apply detour factor constraint
#             # A station is acceptable if source->station->dest <= detour_factor * direct_distance
#             filtered_stations['total_detour_distance'] = (
#                 filtered_stations['distance_from_source'] + 
#                 filtered_stations['distance_from_dest']
#             )
            
#             filtered_stations['detour_ratio'] = (
#                 filtered_stations['total_detour_distance'] / direct_distance
#             )
            
#             # Filter by detour factor
#             filtered_stations = filtered_stations[
#                 filtered_stations['detour_ratio'] <= detour_factor
#             ]
            
#             if len(filtered_stations) == 0:
#                 print(f"No stations found within detour factor of {detour_factor}")
#                 return None
            
#             # Sort by detour ratio (most efficient first)
#             filtered_stations = filtered_stations.sort_values('detour_ratio')
            
#             return filtered_stations
        
#         except Exception as e:
#             print(f"Distance-based filtering failed: {str(e)}")
#             return None
    
#     def corridor_based_filtering(self, stations_df, source_coords, dest_coords, corridor_width=25):
#         """Filter stations within a corridor around the direct route"""
#         try:
#             stations_df_copy = stations_df.copy()
            
#             # Calculate perpendicular distance from each station to the direct route line
#             stations_df_copy['corridor_distance'] = stations_df_copy.apply(
#                 lambda row: self._point_to_line_distance(
#                     (row['latitude'], row['longitude']), source_coords, dest_coords
#                 ),
#                 axis=1
#             )
            
#             # Filter stations within corridor
#             filtered_stations = stations_df_copy[
#                 stations_df_copy['corridor_distance'] <= corridor_width
#             ].copy()
            
#             if len(filtered_stations) == 0:
#                 print(f"No stations found within corridor width of {corridor_width} km")
#                 return None
            
#             # Calculate distances from source and destination
#             filtered_stations['distance_from_source'] = filtered_stations.apply(
#                 lambda row: geodesic(source_coords, (row['latitude'], row['longitude'])).kilometers,
#                 axis=1
#             )
            
#             filtered_stations['distance_from_dest'] = filtered_stations.apply(
#                 lambda row: geodesic(dest_coords, (row['latitude'], row['longitude'])).kilometers,
#                 axis=1
#             )
            
#             # Sort by distance from source
#             filtered_stations = filtered_stations.sort_values('distance_from_source')
            
#             return filtered_stations
        
#         except Exception as e:
#             print(f"Corridor-based filtering failed: {str(e)}")
#             return None
    
#     def combined_filtering(self, stations_df, source_coords, dest_coords, 
#                           detour_factor=1.5, corridor_width=25, 
#                           max_distance_source=200, max_distance_dest=200):
#         """Apply multiple filtering methods in combination"""
#         try:
#             # Start with geographic bounding box
#             filtered_stations = self.geographic_bounding_box(stations_df, source_coords, dest_coords)
            
#             if filtered_stations is None:
#                 return None
            
#             # Apply corridor-based filtering
#             filtered_stations = self.corridor_based_filtering(
#                 filtered_stations, source_coords, dest_coords, corridor_width
#             )
            
#             if filtered_stations is None:
#                 return None
            
#             # Apply distance-based filtering
#             filtered_stations = self.distance_based_filtering(
#                 filtered_stations, source_coords, dest_coords, 
#                 detour_factor, max_distance_source, max_distance_dest
#             )
            
#             if filtered_stations is None:
#                 return None
            
#             # Add combined score for ranking
#             # Lower score is better (combination of detour ratio and corridor distance)
#             filtered_stations['combined_score'] = (
#                 filtered_stations['detour_ratio'] * 0.7 + 
#                 (filtered_stations['corridor_distance'] / corridor_width) * 0.3
#             )
            
#             # Sort by combined score
#             filtered_stations = filtered_stations.sort_values('combined_score')
            
#             return filtered_stations
        
#         except Exception as e:
#             print(f"Combined filtering failed: {str(e)}")
#             return None
    
#     def _point_to_line_distance(self, point, line_start, line_end):
#         """Calculate the perpendicular distance from a point to a line segment"""
#         try:
#             # Convert to numpy arrays for easier calculation
#             point = np.array([point[0], point[1]])
#             line_start = np.array([line_start[0], line_start[1]])
#             line_end = np.array([line_end[0], line_end[1]])
            
#             # Calculate line vector
#             line_vec = line_end - line_start
            
#             # Calculate point vector
#             point_vec = point - line_start
            
#             # Calculate projection of point onto line
#             line_len = np.linalg.norm(line_vec)
            
#             if line_len == 0:
#                 # Line start and end are the same point
#                 return geodesic(point, line_start).kilometers
            
#             # Normalize line vector
#             line_unit = line_vec / line_len
            
#             # Calculate projection length
#             proj_length = np.dot(point_vec, line_unit)
            
#             # Clamp projection to line segment
#             proj_length = max(0, min(proj_length, line_len))
            
#             # Calculate closest point on line
#             closest_point = line_start + proj_length * line_unit
            
#             # Calculate distance using geodesic
#             distance = geodesic(point, closest_point).kilometers
            
#             return distance
        
#         except Exception as e:
#             # Fallback to simple distance calculation
#             dist_to_start = geodesic(point, line_start).kilometers
#             dist_to_end = geodesic(point, line_end).kilometers
#             return min(dist_to_start, dist_to_end)
    
#     def ml_based_filtering(self, stations_df, source_coords, dest_coords, filter_criteria=None):
#         """Filter stations using machine learning predictions"""
#         try:
#             if filter_criteria is None:
#                 filter_criteria = {
#                     'min_rating': 3.0,
#                     'max_congestion': 1,  # 0=Low, 1=Medium, 2=High
#                     'prefer_fast_charging': True,
#                     'min_station_score': 3.0
#                 }
            
#             # Calculate basic distance metrics
#             stations_df_copy = stations_df.copy()
#             stations_df_copy['distance_from_source'] = stations_df_copy.apply(
#                 lambda row: geodesic(source_coords, (row['latitude'], row['longitude'])).kilometers,
#                 axis=1
#             )
            
#             stations_df_copy['distance_from_dest'] = stations_df_copy.apply(
#                 lambda row: geodesic(dest_coords, (row['latitude'], row['longitude'])).kilometers,
#                 axis=1
#             )
            
#             # Get ML predictions
#             ml_predictions = self.station_predictor.predict_station_metrics(
#                 stations_df_copy, 
#                 route_info=True, 
#                 time_info=True
#             )
            
#             if ml_predictions is None or len(ml_predictions) == 0:
#                 print("[WARNING] ML model failed or returned empty. Skipping ML filter.")
#                 return stations_df

#             # Apply ML-based filters
#             filtered = ml_predictions.copy()
            
#             # Filter by predicted rating
#             if 'predicted_rating' in filtered.columns:
#                 filtered = filtered[
#                     filtered['predicted_rating'] >= filter_criteria['min_rating']
#                 ]
            
#             # Filter by predicted congestion
#             if 'predicted_congestion' in filtered.columns:
#                 filtered = filtered[
#                     filtered['predicted_congestion'] <= filter_criteria['max_congestion']
#                 ]
            
#             # Filter by ML station score
#             if 'ml_station_score' in filtered.columns:
#                 filtered = filtered[
#                     filtered['ml_station_score'] >= filter_criteria['min_station_score']
#                 ]
            
#             # Prefer fast charging if requested
#             if filter_criteria['prefer_fast_charging'] and 'dc_fast_ports' in filtered.columns:
#                 # Prioritize stations with DC fast charging
#                 dc_stations = filtered[
#                     pd.to_numeric(filtered['dc_fast_ports'], errors='coerce').fillna(0) > 0
#                 ]
#                 if not dc_stations.empty:
#                     filtered_stations = dc_stations
            
#             if len(filtered_stations) == 0:
#                 print("No stations meet ML filtering criteria, relaxing constraints")
#                 # Relax constraints and try again
#                 relaxed_criteria = {
#                 'min_rating': max(1.0, filter_criteria['min_rating'] - 1.0),
#                 'max_congestion': min(2, filter_criteria['max_congestion'] + 1),
#                 'prefer_fast_charging': filter_criteria['prefer_fast_charging'],
#                 'min_station_score': max(1.0, filter_criteria['min_station_score'] - 1.0)
#                 }
                
#                 return self.ml_based_filtering(stations_df, source_coords, dest_coords, relaxed_criteria)
            
#             # Sort by ML station score
#             return filtered.sort_values('ml_station_score', ascending=False)
            
#         except Exception as e:
#             print(f"ML-based filtering failed: {str(e)}")
#             # return self.combined_filtering(stations_df, source_coords, dest_coords)
#             return stations_df
    
#     def station_type_filtering(self, stations_df, charging_types=None, power_levels=None):
#         """Filter stations by charging type and power level"""
#         try:
#             if charging_types is None:
#                 charging_types = ['AC', 'DC']
#             if power_levels is None:
#                 power_levels = ['Level1', 'Level2', 'DC_Fast']
            
#             filtered_stations = stations_df.copy()
            
#             # Filter by charging type
#             if 'AC' not in charging_types:
#                 # Remove AC-only stations
#                 filtered_stations = filtered_stations[
#                     pd.to_numeric(filtered_stations.get('dc_fast_ports', 0), errors='coerce').fillna(0) > 0
#                 ]
            
#             if 'DC' not in charging_types:
#                 # Remove DC stations
#                 filtered_stations = filtered_stations[
#                     pd.to_numeric(filtered_stations.get('dc_fast_ports', 0), errors='coerce').fillna(0) == 0
#                 ]
            
#             # Filter by power level
#             power_filter = pd.Series(False, index=filtered_stations.index)
            
#             if 'Level1' in power_levels and 'level1_ports' in filtered_stations.columns:
#                 power_filter |= (
#                     pd.to_numeric(filtered_stations['level1_ports'], errors='coerce').fillna(0) > 0
#                 )
            
#             if 'Level2' in power_levels and 'level2_ports' in filtered_stations.columns:
#                 power_filter |= (
#                     pd.to_numeric(filtered_stations['level2_ports'], errors='coerce').fillna(0) > 0
#                 )
            
#             if 'DC_Fast' in power_levels and 'dc_fast_ports' in filtered_stations.columns:
#                 power_filter |= (
#                     pd.to_numeric(filtered_stations['dc_fast_ports'], errors='coerce').fillna(0) > 0
#                 )
            
#             # Apply power level filter
#             filtered_stations = filtered_stations[power_filter]
            
#             if len(filtered_stations) == 0:
#                 print("No stations found matching charging type and power level criteria")
#                 return None
            
#             return filtered_stations
            
#         except Exception as e:
#             print(f"Station type filtering failed: {str(e)}")
#             return stations_df
    
#     def smart_filtering(self, stations_df, source_coords, dest_coords, 
#                        ev_specs=None, user_preferences=None):
#         """Smart filtering combining all methods with ML optimization"""
#         try:
#             # Default preferences
#             if user_preferences is None:
#                 user_preferences = {
#                     'prefer_fast_charging': True,
#                     'max_detour_factor': 1.5,
#                     'corridor_width': 25,
#                     'min_station_rating': 3.0,
#                     'avoid_congestion': True,
#                     'charging_types': ['AC', 'DC'],
#                     'power_levels': ['Level2', 'DC_Fast']
#                 }
            
#             # Start with geographic filtering
#             filtered_stations = self.geographic_bounding_box(stations_df, source_coords, dest_coords)
            
#             if filtered_stations is None or len(filtered_stations) == 0:
#                 print("[WARNING] Geographic filtering returned no stations, skipping to original data")
#                 filtered_stations = stations_df.copy()
            
#             # Apply station type filtering
#             filtered = self.station_type_filtering(
#                 filtered_stations, 
#                 user_preferences.get('charging_types', ['AC', 'DC']),
#                 user_preferences.get('power_levels', ['Level2', 'DC_Fast'])
#             )
            
#             if filtered is not None and len(filtered) > 0:
#                 filtered_stations = filtered
#             else:
#                 print("[WARNING] Station type filtering skipped due to no results")
            
#             # Apply corridor filtering
#             filtered = self.corridor_based_filtering(
#                 filtered_stations, 
#                 source_coords, 
#                 dest_coords, 
#                 user_preferences.get('corridor_width', 25)
#             )
            
#             if filtered_stations is not None and len(filtered) > 0:
#                 filtered_stations = filtered
#             else:
#                 print("[WARNING] Corridor filtering skipped due to no results")
            
#             # Apply distance-based filtering with detour factor
#             filtered = self.distance_based_filtering(
#                 filtered_stations, 
#                 source_coords, 
#                 dest_coords, 
#                 user_preferences.get('max_detour_factor', 1.5)
#             )
            
#             if filtered is not None and len(filtered) > 0:
#                 filtered_stations = filtered
#             else:
#                 print("[WARNING] Distance filtering skipped due to no results")
            
#             # Apply ML-based filtering for final optimization
#             ml_criteria = {
#                 'min_rating': user_preferences.get('min_station_rating', 3.0),
#                 'max_congestion': 1 if user_preferences.get('avoid_congestion', True) else 2,
#                 'prefer_fast_charging': user_preferences.get('prefer_fast_charging', True),
#                 'min_station_score': user_preferences.get('min_station_rating', 3.0)
#             }
            
#             final_filtered = self.ml_based_filtering(
#                 filtered_stations, 
#                 source_coords, 
#                 dest_coords, 
#                 ml_criteria
#             )
            
#             if final_filtered is None or len(final_filtered) == 0:
#                 print("[WARNING] ML filtering returned no stations, using pre-ML filtered data")
#                 return filtered_stations
            
#             return final_filtered
            
#         except Exception as e:
#             print(f"Smart filtering failed: {str(e)}")
#             return self.combined_filtering(stations_df, source_coords, dest_coords)













import pandas as pd
import numpy as np
from geopy.distance import geodesic
# import streamlit as st
from ml_models.station_predictor import StationPredictor

class StationFilters:
    def __init__(self):
        self.station_predictor = StationPredictor()
    
    def geographic_bounding_box(self, stations_df, source_coords, dest_coords, padding_factor=0.2):
        """Filter stations within a geographic bounding box between source and destination"""
        try:
            # Calculate route distance for dynamic padding
            route_distance = geodesic(source_coords, dest_coords).kilometers
            
            # Dynamic padding based on route distance (more padding for longer routes)
            padding = max(0.1, min(1.0, route_distance * padding_factor / 100))
            
            # Calculate bounding box with dynamic padding
            min_lat = min(source_coords[0], dest_coords[0]) - padding
            max_lat = max(source_coords[0], dest_coords[0]) + padding
            min_lon = min(source_coords[1], dest_coords[1]) - padding
            max_lon = max(source_coords[1], dest_coords[1]) + padding
            
            # Filter stations within bounding box
            filtered_stations = stations_df[
                (stations_df['latitude'] >= min_lat) &
                (stations_df['latitude'] <= max_lat) &
                (stations_df['longitude'] >= min_lon) &
                (stations_df['longitude'] <= max_lon)
            ].copy()
            
            if len(filtered_stations) == 0:
                print("No stations found within geographic bounding box")
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
            
            # Filter out stations that are too far from both source and destination
            max_reasonable_distance = route_distance * 1.5  # Allow up to 1.5x route distance
            
            filtered_stations = filtered_stations[
                (filtered_stations['distance_from_source'] <= max_reasonable_distance) |
                (filtered_stations['distance_from_dest'] <= max_reasonable_distance)
            ]
            
            # Sort by distance from source
            filtered_stations = filtered_stations.sort_values('distance_from_source')
            
            return filtered_stations
        
        except Exception as e:
            print(f"Geographic bounding box filtering failed: {str(e)}")
            return None
    
    def distance_based_filtering(self, stations_df, source_coords, dest_coords, 
                                detour_factor=1.3, max_distance_from_route=None):
        """Improved distance-based filtering with dynamic constraints"""
        try:
            # Calculate direct distance between source and destination
            direct_distance = geodesic(source_coords, dest_coords).kilometers
            
            # Set dynamic max distance if not provided
            if max_distance_from_route is None:
                max_distance_from_route = max(50, direct_distance * 0.3)  # 30% of route or min 50km
            
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
            
            # Calculate total detour distance
            stations_df_copy['total_detour_distance'] = (
                stations_df_copy['distance_from_source'] + 
                stations_df_copy['distance_from_dest']
            )
            
            stations_df_copy['detour_ratio'] = (
                stations_df_copy['total_detour_distance'] / direct_distance
            )
            
            # Apply progressive filtering based on route length
            if direct_distance < 100:  # Short routes
                detour_threshold = min(detour_factor, 1.5)
                distance_threshold = min(max_distance_from_route, 100)
            elif direct_distance < 300:  # Medium routes
                detour_threshold = min(detour_factor, 1.4)
                distance_threshold = min(max_distance_from_route, 150)
            else:  # Long routes
                detour_threshold = min(detour_factor, 1.3)
                distance_threshold = max_distance_from_route
            
            # Filter by detour ratio and reasonable distance constraints
            filtered_stations = stations_df_copy[
                (stations_df_copy['detour_ratio'] <= detour_threshold) &
                (stations_df_copy['distance_from_source'] <= distance_threshold) &
                (stations_df_copy['distance_from_dest'] <= distance_threshold)
            ]
            
            if len(filtered_stations) == 0:
                print(f"No stations found within detour factor {detour_threshold}, relaxing constraints")
                # Relax constraints progressively
                filtered_stations = stations_df_copy[
                    (stations_df_copy['detour_ratio'] <= detour_threshold * 1.2) &
                    (stations_df_copy['distance_from_source'] <= distance_threshold * 1.3) &
                    (stations_df_copy['distance_from_dest'] <= distance_threshold * 1.3)
                ]
            
            if len(filtered_stations) == 0:
                print("Distance-based filtering too restrictive, returning closest stations")
                # Return closest stations as fallback
                return stations_df_copy.nsmallest(20, 'distance_from_source')
            
            # Sort by combined score (detour ratio + distance from source)
            filtered_stations['distance_score'] = (
                filtered_stations['detour_ratio'] * 0.6 + 
                (filtered_stations['distance_from_source'] / direct_distance) * 0.4
            )
            
            return filtered_stations.sort_values('distance_score')
        
        except Exception as e:
            print(f"Distance-based filtering failed: {str(e)}")
            return None
    
    def corridor_based_filtering(self, stations_df, source_coords, dest_coords, corridor_width=None):
        """Improved corridor-based filtering with dynamic width"""
        try:
            # Calculate route distance for dynamic corridor width
            route_distance = geodesic(source_coords, dest_coords).kilometers
            
            # Set dynamic corridor width if not provided
            if corridor_width is None:
                if route_distance < 50:
                    corridor_width = 15  # Narrow corridor for short routes
                elif route_distance < 200:
                    corridor_width = 25  # Medium corridor for medium routes
                else:
                    corridor_width = 35  # Wider corridor for long routes
            
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
                print(f"No stations found within corridor width of {corridor_width} km, expanding")
                # Expand corridor width progressively
                expanded_width = corridor_width * 1.5
                filtered_stations = stations_df_copy[
                    stations_df_copy['corridor_distance'] <= expanded_width
                ].copy()
                
                if len(filtered_stations) == 0:
                    print("Corridor too narrow, returning closest stations to route")
                    # Return stations closest to the route
                    return stations_df_copy.nsmallest(15, 'corridor_distance')
            
            # Calculate distances from source and destination
            filtered_stations['distance_from_source'] = filtered_stations.apply(
                lambda row: geodesic(source_coords, (row['latitude'], row['longitude'])).kilometers,
                axis=1
            )
            
            filtered_stations['distance_from_dest'] = filtered_stations.apply(
                lambda row: geodesic(dest_coords, (row['latitude'], row['longitude'])).kilometers,
                axis=1
            )
            
            # Calculate progress along route (0 = at source, 1 = at destination)
            filtered_stations['route_progress'] = filtered_stations.apply(
                lambda row: self._calculate_route_progress(
                    (row['latitude'], row['longitude']), source_coords, dest_coords
                ),
                axis=1
            )
            
            # Sort by route progress (stations appear in order along the route)
            filtered_stations = filtered_stations.sort_values('route_progress')
            
            return filtered_stations
        
        except Exception as e:
            print(f"Corridor-based filtering failed: {str(e)}")
            return None
    
    def _calculate_route_progress(self, point, source_coords, dest_coords):
        """Calculate how far along the route a point is (0 = source, 1 = destination)"""
        try:
            # Vector from source to destination
            route_vector = np.array([dest_coords[0] - source_coords[0], 
                                   dest_coords[1] - source_coords[1]])
            
            # Vector from source to point
            point_vector = np.array([point[0] - source_coords[0], 
                                   point[1] - source_coords[1]])
            
            # Calculate projection
            route_length_squared = np.dot(route_vector, route_vector)
            
            if route_length_squared == 0:
                return 0
            
            progress = np.dot(point_vector, route_vector) / route_length_squared
            
            # Clamp to [0, 1] range
            return max(0, min(1, progress))
        
        except Exception:
            # Fallback: use distance ratio
            dist_from_source = geodesic(point, source_coords).kilometers
            total_route_distance = geodesic(source_coords, dest_coords).kilometers
            
            if total_route_distance == 0:
                return 0
            
            return min(1, dist_from_source / total_route_distance)
    
    def combined_filtering(self, stations_df, source_coords, dest_coords, 
                          detour_factor=1.3, corridor_width=None, 
                          max_distance_from_route=None):
        """Improved combined filtering with better fallback logic"""
        try:
            route_distance = geodesic(source_coords, dest_coords).kilometers
            
            # Start with geographic bounding box with appropriate padding
            filtered_stations = self.geographic_bounding_box(
                stations_df, source_coords, dest_coords, padding_factor=0.15
            )
            
            if filtered_stations is None or len(filtered_stations) < 5:
                print("Geographic filtering too restrictive, expanding search area")
                filtered_stations = self.geographic_bounding_box(
                    stations_df, source_coords, dest_coords, padding_factor=0.3
                )
            
            if filtered_stations is None:
                print("Geographic filtering failed, using all stations")
                filtered_stations = stations_df.copy()
            
            # Apply corridor-based filtering
            corridor_filtered = self.corridor_based_filtering(
                filtered_stations, source_coords, dest_coords, corridor_width
            )
            
            if corridor_filtered is not None and len(corridor_filtered) >= 5:
                filtered_stations = corridor_filtered
            else:
                print("Corridor filtering too restrictive, keeping geographic results")
            
            # Apply distance-based filtering
            distance_filtered = self.distance_based_filtering(
                filtered_stations, source_coords, dest_coords, 
                detour_factor, max_distance_from_route
            )
            
            if distance_filtered is not None and len(distance_filtered) >= 3:
                filtered_stations = distance_filtered
            else:
                print("Distance filtering too restrictive, keeping previous results")
            
            # Ensure we have a reasonable number of stations
            if len(filtered_stations) < 3:
                print("Too few stations after filtering, adding closest stations")
                # Add closest stations from original dataset
                all_stations = stations_df.copy()
                all_stations['distance_from_source'] = all_stations.apply(
                    lambda row: geodesic(source_coords, (row['latitude'], row['longitude'])).kilometers,
                    axis=1
                )
                
                closest_stations = all_stations.nsmallest(10, 'distance_from_source')
                filtered_stations = pd.concat([filtered_stations, closest_stations]).drop_duplicates()
            
            # Add final combined score for ranking
            filtered_stations['combined_score'] = self._calculate_combined_score(
                filtered_stations, source_coords, dest_coords, route_distance
            )
            
            # Sort by combined score and return top results
            return filtered_stations.sort_values('combined_score').head(20)
            
        except Exception as e:
            print(f"Combined filtering failed: {str(e)}")
            return None
    
    def _calculate_combined_score(self, stations_df, source_coords, dest_coords, route_distance):
        """Calculate a combined score for station ranking"""
        try:
            scores = []
            
            for _, row in stations_df.iterrows():
                score = 0
                
                # Distance score (lower is better)
                if 'distance_from_source' in row:
                    distance_score = row['distance_from_source'] / route_distance
                    score += distance_score * 0.3
                
                # Detour ratio score (lower is better)
                if 'detour_ratio' in row:
                    score += row['detour_ratio'] * 0.4
                
                # Corridor distance score (lower is better)
                if 'corridor_distance' in row:
                    corridor_score = row['corridor_distance'] / 50  # Normalize to ~50km
                    score += corridor_score * 0.2
                
                # Route progress score (prefer stations spread along route)
                if 'route_progress' in row:
                    # Favor stations that are not too close to start or end
                    progress_score = abs(row['route_progress'] - 0.5) * 0.1
                    score += progress_score
                
                scores.append(score)
            
            return scores
            
        except Exception:
            # Fallback to distance from source
            return stations_df.get('distance_from_source', 0)
    
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
            line_len_squared = np.dot(line_vec, line_vec)
            
            if line_len_squared == 0:
                # Line start and end are the same point
                return geodesic(point, line_start).kilometers
            
            # Calculate projection parameter
            t = np.dot(point_vec, line_vec) / line_len_squared
            
            # Clamp to line segment
            t = max(0, min(1, t))
            
            # Calculate closest point on line
            closest_point = line_start + t * line_vec
            
            # Calculate distance using geodesic
            distance = geodesic(point, closest_point).kilometers
            
            return distance
        
        except Exception as e:
            # Fallback to simple distance calculation
            dist_to_start = geodesic(point, line_start).kilometers
            dist_to_end = geodesic(point, line_end).kilometers
            return min(dist_to_start, dist_to_end)
    
    def ml_based_filtering(self, stations_df, source_coords, dest_coords, filter_criteria=None):
        """Improved ML-based filtering with better fallback logic"""
        try:
            if filter_criteria is None:
                filter_criteria = {
                    'min_rating': 2.5,  # Lowered from 3.0
                    'max_congestion': 1,  # 0=Low, 1=Medium, 2=High
                    'prefer_fast_charging': True,
                    'min_station_score': 2.5  # Lowered from 3.0
                }
            
            # Calculate basic distance metrics
            stations_df_copy = stations_df.copy()
            route_distance = geodesic(source_coords, dest_coords).kilometers
            
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
            
            if ml_predictions is None or len(ml_predictions) == 0:
                print("[WARNING] ML model failed, using distance-based filtering")
                return self.distance_based_filtering(stations_df_copy, source_coords, dest_coords)
            
            # Apply ML-based filters progressively
            filtered = ml_predictions.copy()
            original_count = len(filtered)
            
            # Filter by predicted rating
            if 'predicted_rating' in filtered.columns:
                rating_filtered = filtered[
                    filtered['predicted_rating'] >= filter_criteria['min_rating']
                ]
                if len(rating_filtered) >= max(3, original_count * 0.3):
                    filtered = rating_filtered
                else:
                    print(f"Rating filter too restrictive ({len(rating_filtered)} stations), relaxing")
            
            # Filter by predicted congestion
            if 'predicted_congestion' in filtered.columns:
                congestion_filtered = filtered[
                    filtered['predicted_congestion'] <= filter_criteria['max_congestion']
                ]
                if len(congestion_filtered) >= max(3, len(filtered) * 0.4):
                    filtered = congestion_filtered
                else:
                    print(f"Congestion filter too restrictive ({len(congestion_filtered)} stations), relaxing")
            
            # Filter by ML station score
            if 'ml_station_score' in filtered.columns:
                score_filtered = filtered[
                    filtered['ml_station_score'] >= filter_criteria['min_station_score']
                ]
                if len(score_filtered) >= max(3, len(filtered) * 0.4):
                    filtered = score_filtered
                else:
                    print(f"ML score filter too restrictive ({len(score_filtered)} stations), relaxing")
            
            # Prefer fast charging if requested, but don't eliminate all stations
            if filter_criteria['prefer_fast_charging'] and 'dc_fast_ports' in filtered.columns:
                dc_stations = filtered[
                    pd.to_numeric(filtered['dc_fast_ports'], errors='coerce').fillna(0) > 0
                ]
                if len(dc_stations) >= max(3, len(filtered) * 0.3):
                    filtered = dc_stations
                else:
                    print("Fast charging preference applied as bonus, not filter")
                    # Add bonus score for fast charging instead of filtering
                    filtered['fast_charging_bonus'] = (
                        pd.to_numeric(filtered['dc_fast_ports'], errors='coerce').fillna(0) > 0
                    ).astype(int) * 0.5
                    
                    if 'ml_station_score' in filtered.columns:
                        filtered['ml_station_score'] += filtered['fast_charging_bonus']
            
            # Ensure minimum number of stations
            if len(filtered) < 3:
                print("ML filtering too restrictive, using top stations by ML score")
                if 'ml_station_score' in ml_predictions.columns:
                    filtered = ml_predictions.nlargest(10, 'ml_station_score')
                else:
                    filtered = ml_predictions.nsmallest(10, 'distance_from_source')
            
            # Sort by ML station score if available, otherwise by distance
            if 'ml_station_score' in filtered.columns:
                return filtered.sort_values('ml_station_score', ascending=False)
            else:
                return filtered.sort_values('distance_from_source')
            
        except Exception as e:
            print(f"ML-based filtering failed: {str(e)}")
            return self.distance_based_filtering(stations_df, source_coords, dest_coords)
    
    def station_type_filtering(self, stations_df, charging_types=None, power_levels=None):
        """Improved station type filtering with better fallback"""
        try:
            if charging_types is None:
                charging_types = ['AC', 'DC']
            if power_levels is None:
                power_levels = ['Level2', 'DC_Fast']
            
            filtered_stations = stations_df.copy()
            original_count = len(filtered_stations)
            
            # Create power level filter
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
            
            # Apply power level filter only if it doesn't eliminate too many stations
            power_filtered = filtered_stations[power_filter]
            
            if len(power_filtered) >= max(3, original_count * 0.3):
                filtered_stations = power_filtered
            else:
                print(f"Power level filter too restrictive ({len(power_filtered)} stations), keeping all")
            
            # Apply charging type filter
            if 'AC' not in charging_types and 'dc_fast_ports' in filtered_stations.columns:
                # Remove AC-only stations (keep stations with DC fast charging)
                dc_filtered = filtered_stations[
                    pd.to_numeric(filtered_stations['dc_fast_ports'], errors='coerce').fillna(0) > 0
                ]
                if len(dc_filtered) >= max(3, len(filtered_stations) * 0.3):
                    filtered_stations = dc_filtered
                else:
                    print("AC type filter too restrictive, keeping all stations")
            
            if 'DC' not in charging_types and 'dc_fast_ports' in filtered_stations.columns:
                # Remove DC stations (keep AC-only stations)
                ac_filtered = filtered_stations[
                    pd.to_numeric(filtered_stations['dc_fast_ports'], errors='coerce').fillna(0) == 0
                ]
                if len(ac_filtered) >= max(3, len(filtered_stations) * 0.3):
                    filtered_stations = ac_filtered
                else:
                    print("DC type filter too restrictive, keeping all stations")
            
            if len(filtered_stations) == 0:
                print("Station type filtering eliminated all stations, returning original")
                return stations_df
            
            return filtered_stations
            
        except Exception as e:
            print(f"Station type filtering failed: {str(e)}")
            return stations_df
    
    def smart_filtering(self, stations_df, source_coords, dest_coords, 
                       ev_specs=None, user_preferences=None):
        """Enhanced smart filtering with adaptive logic"""
        try:
            route_distance = geodesic(source_coords, dest_coords).kilometers
            
            # Adaptive preferences based on route distance
            if user_preferences is None:
                user_preferences = {}
            
            # Set adaptive defaults
            default_preferences = {
                'prefer_fast_charging': True,
                'max_detour_factor': 1.4 if route_distance > 200 else 1.3,
                'corridor_width': 35 if route_distance > 300 else 25,
                'min_station_rating': 2.5,
                'avoid_congestion': True,
                'charging_types': ['AC', 'DC'],
                'power_levels': ['Level2', 'DC_Fast'],
                'max_stations': 15
            }
            
            # Merge user preferences with defaults
            for key, value in default_preferences.items():
                if key not in user_preferences:
                    user_preferences[key] = value
            
            print(f"Starting smart filtering for {len(stations_df)} stations on {route_distance:.1f}km route")
            
            # Step 1: Geographic filtering with adaptive constraints
            filtered_stations = self.geographic_bounding_box(
                stations_df, source_coords, dest_coords, 
                padding_factor=0.2 if route_distance > 200 else 0.15
            )
            
            if filtered_stations is None or len(filtered_stations) < 5:
                print("Geographic filtering too restrictive, expanding search area")
                filtered_stations = self.geographic_bounding_box(
                    stations_df, source_coords, dest_coords, padding_factor=0.4
                )
            
            if filtered_stations is None:
                print("Geographic filtering failed, using all stations")
                filtered_stations = stations_df.copy()
            
            print(f"After geographic filtering: {len(filtered_stations)} stations")
            
            # Step 2: Station type filtering (less restrictive)
            type_filtered = self.station_type_filtering(
                filtered_stations, 
                user_preferences.get('charging_types', ['AC', 'DC']),
                user_preferences.get('power_levels', ['Level2', 'DC_Fast'])
            )
            
            if type_filtered is not None and len(type_filtered) >= 3:
                filtered_stations = type_filtered
                print(f"After station type filtering: {len(filtered_stations)} stations")
            else:
                print("Station type filtering skipped (too restrictive)")
            
            # Step 3: Combined geometric filtering
            combined_filtered = self.combined_filtering(
                filtered_stations, 
                source_coords, 
                dest_coords, 
                user_preferences.get('max_detour_factor', 1.3),
                user_preferences.get('corridor_width', 25)
            )
            
            if combined_filtered is not None and len(combined_filtered) >= 3:
                filtered_stations = combined_filtered
                print(f"After combined filtering: {len(filtered_stations)} stations")
            else:
                print("Combined filtering skipped (too restrictive)")
            
            # Step 4: ML-based optimization
            ml_criteria = {
                'min_rating': user_preferences.get('min_station_rating', 2.5),
                'max_congestion': 1 if user_preferences.get('avoid_congestion', True) else 2,
                'prefer_fast_charging': user_preferences.get('prefer_fast_charging', True),
                'min_station_score': user_preferences.get('min_station_rating', 2.5)
            }
            
            final_filtered = self.ml_based_filtering(
                filtered_stations, 
                source_coords, 
                dest_coords, 
                ml_criteria
            )
            
            if final_filtered is not None and len(final_filtered) >= 3:
                filtered_stations = final_filtered
                print(f"After ML filtering: {len(filtered_stations)} stations")
            else:
                print("ML filtering skipped (too restrictive)")
            
            # Step 5: Final ranking and selection
            max_stations = user_preferences.get('max_stations', 15)
            
            if len(filtered_stations) > max_stations:
                # Use the best scoring method available
                if 'ml_station_score' in filtered_stations.columns:
                    final_stations = filtered_stations.nlargest(max_stations, 'ml_station_score')
                elif 'combined_score' in filtered_stations.columns:
                    final_stations = filtered_stations.nsmallest(max_stations, 'combined_score')
                else:
                    final_stations = filtered_stations.nsmallest(max_stations, 'distance_from_source')
            else:
                final_stations = filtered_stations
            
            print(f"Final result: {len(final_stations)} stations")
            
            # Ensure we have reasonable results
            if len(final_stations) < 3:
                print("Too few stations after all filtering, adding closest stations")
                all_stations = stations_df.copy()
                all_stations['distance_from_source'] = all_stations.apply(
                    lambda row: geodesic(source_coords, (row['latitude'], row['longitude'])).kilometers,
                    axis=1
                )
                closest_stations = all_stations.nsmallest(max_stations, 'distance_from_source')
                final_stations = pd.concat([final_stations, closest_stations]).drop_duplicates().head(max_stations)
            
            return final_stations
            
        except Exception as e:
            print(f"Smart filtering failed: {str(e)}")
            # Fallback to simple distance-based filtering
            stations_with_distance = stations_df.copy()
            stations_with_distance['distance_from_source'] = stations_with_distance.apply(
                lambda row: geodesic(source_coords, (row['latitude'], row['longitude'])).kilometers,
                axis=1
            )
            return stations_with_distance.nsmallest(15, 'distance_from_source')