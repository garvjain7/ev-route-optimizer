import pandas as pd
import numpy as np
from geopy.distance import geodesic
# import streamlit as st
from utils.distance_calculator import DistanceCalculator
from ml_models.station_predictor import StationPredictor
from ml_models.travel_time_predictor import TravelTimePredictor
from ml_models.adaptive_router import AdaptiveRouter

class EVRouter:
    def __init__(self):
        self.distance_calc = DistanceCalculator()
        self.station_predictor = StationPredictor()
        self.travel_time_predictor = TravelTimePredictor()
        self.adaptive_router = AdaptiveRouter()
        self.adaptive_router.load_route_feedback()
    
    def optimize_route(self, source_coords, dest_coords, stations_df, ev_specs):
        """Optimize EV route with charging stops"""
        try:
            # Extract EV specifications
            battery_range = ev_specs['battery_range']
            # consumption_rate = ev_specs['consumption_rate']
            # charging_time = ev_specs['charging_time']
            safety_margin = ev_specs['safety_margin']
            
            # Calculate effective range (with safety margin)
            effective_range = battery_range * (1 - safety_margin / 100)
            
            # Calculate direct distance
            direct_distance = geodesic(source_coords, dest_coords).kilometers
            
            # Simple routing if within range
            if direct_distance <= effective_range:
                return self._create_direct_route(source_coords, dest_coords, ev_specs)
            
            # Multi-stop routing with charging
            return self._create_multi_stop_route(
                source_coords, dest_coords, stations_df, ev_specs, effective_range
            )
        
        except Exception as e:
            print(f"Route optimization failed: {str(e)}")
            return None
    
    def _create_direct_route(self, source_coords, dest_coords, ev_specs):
        """Create a direct route without charging stops"""
        try:
            distance = geodesic(source_coords, dest_coords).kilometers
            
            # Calculate energy consumption
            energy_consumed = (distance / 100) * ev_specs['consumption_rate']
            battery_used = (energy_consumed / ev_specs['battery_range']) * 100
            
            # Calculate travel time (assuming 80 km/h average speed)
            travel_time = distance / 80
            
            route_data = {
                'waypoints': [
                    {'type': 'source', 'coordinates': source_coords, 'name': 'Start'},
                    {'type': 'destination', 'coordinates': dest_coords, 'name': 'End'}
                ],
                'charging_stops': [],
                'statistics': {
                    'total_distance': distance,
                    'charging_stops': 0,
                    'total_time': travel_time,
                    'energy_consumed': energy_consumed,
                    'battery_used': battery_used,
                    'efficiency': ev_specs['consumption_rate']
                },
                'energy_profile': {
                    'distance': [0, distance],
                    'battery_level': [100, 100 - battery_used]
                }
            }
            
            return route_data
        
        except Exception as e:
            print(f"Direct route creation failed: {str(e)}")
            return None
    
    def _create_multi_stop_route(self, source_coords, dest_coords, stations_df, ev_specs, effective_range):
        """Create route with multiple charging stops"""
        try:
            if stations_df is None or len(stations_df) == 0:
                print("[ERROR] No stations provided to multi-stop route.")
                return {
                    'route_id': None,
                    'waypoints': [],
                    'charging_stops': [],
                    'statistics': {},
                    'energy_profile': {},
                    'ml_predictions': {},
                    'warning': 'No stations available for routing. Please check your filters or data.'
                }
            required_cols = {'latitude', 'longitude'}
            if not required_cols.issubset(stations_df.columns):
                print(f"[ERROR] Station DataFrame missing required columns: {required_cols - set(stations_df.columns)}")
                return {
                    'route_id': None,
                    'waypoints': [],
                    'charging_stops': [],
                    'statistics': {},
                    'energy_profile': {},
                    'ml_predictions': {},
                    'warning': 'Station data missing latitude/longitude.'
                }

            visited_indices = set()
            current_position = source_coords
            remaining_distance = geodesic(source_coords, dest_coords).kilometers
            waypoints = [{'type': 'source', 'coordinates': source_coords, 'name': 'Start'}]
            charging_stops = []
            current_battery = 100
            total_distance = 0
            total_time = 0
            energy_profile = {'distance': [0], 'battery_level': [100]}
            stop_counter = 1

            while remaining_distance > effective_range:
                available_stations = stations_df.drop(list(visited_indices), errors='ignore') if visited_indices else stations_df
                if len(available_stations) == 0:
                    print("[ERROR] No more available stations to visit.")
                    break
                charging_station = self._find_optimal_charging_station(
                    current_position, dest_coords, available_stations, effective_range
                )
                print(f"[DEBUG] Remaining distance: {remaining_distance:.2f} km")
                print(f"[DEBUG] Current position: {current_position}")
                print(f"[DEBUG] Stations to choose from: {len(available_stations)}")

                fallback_used = False
                if charging_station is None:
                    print("[WARNING] No stations within effective range. Falling back to nearest station.")
                    available_stations['distance_from_current'] = available_stations.apply(
                        lambda row: geodesic(current_position, (row['latitude'], row['longitude'])).kilometers,
                        axis=1
                    )
                    nearest_idx = available_stations['distance_from_current'].idxmin()
                    charging_station = available_stations.loc[nearest_idx]
                    fallback_used = True
                    if charging_station is None or pd.isnull(charging_station['latitude']) or pd.isnull(charging_station['longitude']):
                        print("[ERROR] Could not find any fallback charging station.")
                        break
                    print(f"[INFO] Using fallback station at ({charging_station['latitude']}, {charging_station['longitude']})")

                # Prevent revisiting the same station robustly
                if 'id' in charging_station:
                    visited_indices.add(charging_station['id'])
                elif hasattr(charging_station, 'name'):
                    visited_indices.add(charging_station.name)
                elif hasattr(charging_station, 'index'):
                    visited_indices.add(charging_station.index)

                station_coords = (charging_station['latitude'], charging_station['longitude'])
                distance_to_station = geodesic(current_position, station_coords).kilometers

                if fallback_used and distance_to_station < 1.0:
                    print("[ERROR] Fallback station is too close to current position. No progress possible.")
                    break

                total_distance += distance_to_station

                try:
                    predicted_travel_time = self.travel_time_predictor.predict_travel_time(
                        current_position, station_coords
                    )
                    travel_time = predicted_travel_time if predicted_travel_time is not None else distance_to_station / 80
                except Exception as e:
                    travel_time = distance_to_station / 80

                total_time += travel_time + (ev_specs['charging_time'] / 60)

                energy_consumed = (distance_to_station / 100) * ev_specs['consumption_rate']
                battery_used = (energy_consumed / ev_specs['battery_range']) * 100
                current_battery -= battery_used

                energy_profile['distance'].append(total_distance)
                energy_profile['battery_level'].append(current_battery)

                charging_stop = {
                    'stop_number': stop_counter,
                    'station_name': charging_station.get('name', f'Station {stop_counter}') if hasattr(charging_station, 'get') else charging_station['name'] if 'name' in charging_station else f'Station {stop_counter}',
                    'coordinates': station_coords,
                    'distance_from_start': total_distance,
                    'battery_on_arrival': current_battery,
                    'charging_time': ev_specs['charging_time'],
                    'distance_to_next': None
                }

                charging_stops.append(charging_stop)
                waypoints.append({
                    'type': 'charging_station',
                    'coordinates': station_coords,
                    'name': charging_stop['station_name']
                })

                print(f"[ROUTE] Added stop {stop_counter} at {station_coords}, remaining distance: {remaining_distance:.2f} km")

                current_battery = 100
                energy_profile['distance'].append(total_distance)
                energy_profile['battery_level'].append(100)

                current_position = station_coords
                remaining_distance = geodesic(current_position, dest_coords).kilometers
                stop_counter += 1

                if stop_counter > 20:
                    print("Maximum number of charging stops reached (20). Aborting to prevent infinite loop.")
                    break

            print(f"[ROUTE] Total charging stops: {len(charging_stops)}")

            # If no charging stops found, return warning and clustered stations
            if len(charging_stops) == 0:
                print("[WARNING] No charging stops could be found for this route. Returning clustered stations for map display.")
                return {
                    'route_id': None,
                    'waypoints': [],
                    'charging_stops': [],
                    'statistics': {},
                    'energy_profile': {},
                    'ml_predictions': {},
                    'warning': 'No valid route found. Please review clustered stations on the map.',
                    'clustered_stations': stations_df.to_dict(orient='records')
                }

            final_distance = geodesic(current_position, dest_coords).kilometers
            total_distance += final_distance

            try:
                predicted_final_time = self.travel_time_predictor.predict_travel_time(
                    current_position, dest_coords
                )
                final_travel_time = predicted_final_time if predicted_final_time is not None else final_distance / 80
            except Exception:
                final_travel_time = final_distance / 80

            total_time += final_travel_time

            final_energy_consumed = (final_distance / 100) * ev_specs['consumption_rate']
            final_battery_used = (final_energy_consumed / ev_specs['battery_range']) * 100
            current_battery -= final_battery_used

            if charging_stops:
                charging_stops[-1]['distance_to_next'] = final_distance

            waypoints.append({'type': 'destination', 'coordinates': dest_coords, 'name': 'End'})

            energy_profile['distance'].append(total_distance)
            energy_profile['battery_level'].append(current_battery)

            total_energy = (total_distance / 100) * ev_specs['consumption_rate']

            route_data = {
                'route_id': f"route_{len(self.adaptive_router.route_history)}",
                'waypoints': waypoints,
                'charging_stops': charging_stops,
                'statistics': {
                    'total_distance': total_distance,
                    'charging_stops': len(charging_stops),
                    'total_time': total_time,
                    'energy_consumed': total_energy,
                    'battery_used': None,
                    'efficiency': ev_specs['consumption_rate']
                },
                'energy_profile': energy_profile,
                'ml_predictions': {
                    'efficiency_score': self.adaptive_router.predict_route_efficiency(route_data) if hasattr(self.adaptive_router, 'predict_route_efficiency') else None,
                    'recommendations': self.adaptive_router.get_adaptive_recommendations(route_data) if hasattr(self.adaptive_router, 'get_adaptive_recommendations') else []
                }
            }

            return route_data

        except Exception as e:
            print(f"Multi-stop route creation failed: {str(e)}")
            return {
                'route_id': None,
                'waypoints': [],
                'charging_stops': [],
                'statistics': {},
                'energy_profile': {},
                'ml_predictions': {},
                'warning': f'Route creation failed: {str(e)}'
            }
    
    def _find_optimal_charging_station(self, current_position, dest_coords, stations_df, effective_range):
        """Find the optimal charging station using ML predictions"""
        try:
            if stations_df is None or len(stations_df) == 0:
                print("[ERROR] No stations available for optimal charging station selection.")
                return None
            if not {'latitude', 'longitude'}.issubset(stations_df.columns):
                print("[ERROR] Station DataFrame missing required columns for optimal charging station selection.")
                return None
            # Calculate reachable stations from current position
            stations_df = stations_df.copy()
            stations_df['distance_from_current'] = stations_df.apply(
                lambda row: geodesic(current_position, (row['latitude'], row['longitude'])).kilometers,
                axis=1
            )

            # print(f"[DEBUG] Effective range: {effective_range} km")

            # distances = stations_df.apply(
            #     lambda row: haversine(current_position[0], current_position[1],
            #                             row['latitude'], row['longitude']),
            #     axis=1
            # )

            # stations_df['distance_from_current'] = distances
            # Filter stations within effective range
            reachable_stations = stations_df[
                stations_df['distance_from_current'] <= effective_range
            ]
            print(f"[DEBUG] Reachable stations: {len(reachable_stations)} / {len(stations_df)} within {effective_range}")
            # If none are reachable, return None (caller will fallback to nearest)
            if len(reachable_stations) == 0:
                print("[DEBUG] No stations within effective range — increase battery_range or reduce safety_margin.")
                return None
            # Calculate distance from each station to destination
            reachable_stations['distance_to_dest'] = reachable_stations.apply(
                lambda row: geodesic(dest_coords, (row['latitude'], row['longitude'])).kilometers,
                axis=1
            )
            # Use ML model to predict optimal stations
            try:
                # Get ML predictions for stations
                ml_predictions = self.station_predictor.predict_station_metrics(
                    reachable_stations,
                    route_info=True,
                    time_info=True
                )
                if ml_predictions is not None:
                    # Use ML score as primary criterion
                    reachable_stations = ml_predictions
                    reachable_stations['optimization_score'] = (
                        reachable_stations['distance_to_dest'] * 0.4 +
                        reachable_stations['distance_from_current'] * 0.2 +
                        (5 - reachable_stations['ml_station_score']) * 0.4
                    )
                else:
                    # Fallback to traditional scoring
                    reachable_stations['optimization_score'] = (
                        reachable_stations['distance_to_dest'] * 0.7 +
                        reachable_stations['distance_from_current'] * 0.3
                    )
                    # Add bonus for fast charging capabilities
                    if 'dc_fast_ports' in reachable_stations.columns:
                        dc_ports = pd.to_numeric(reachable_stations['dc_fast_ports'], errors='coerce').fillna(0)
                        reachable_stations['optimization_score'] -= dc_ports * 2
            except Exception as e:
                print(f"ML prediction failed, using traditional scoring: {str(e)}")
                # Traditional scoring as fallback
                reachable_stations['optimization_score'] = (
                    reachable_stations['distance_to_dest'] * 0.7 +
                    reachable_stations['distance_from_current'] * 0.3
                )

                if 'dc_fast_ports' in reachable_stations.columns:
                    dc_ports = pd.to_numeric(reachable_stations['dc_fast_ports'], errors='coerce').fillna(0)
                    reachable_stations['optimization_score'] -= dc_ports * 2
            # Select station with best score
            best_station = reachable_stations.loc[reachable_stations['optimization_score'].idxmin()]
            if best_station is None:
                print("[DEBUG] No best station could be selected — possible filtering too strict or ML fallback failed.")
            else:
                print(f"[DEBUG] Best station selected: {best_station.get('name', best_station.get('id'))} at ({best_station['latitude']}, {best_station['longitude']})")

            print(f"[DEBUG] Effective range: {effective_range:.2f} km")
            print(f"[DEBUG] Reachable stations count: {len(reachable_stations)}")

            return best_station
        
        except Exception as e:
            print(f"Charging station selection failed: {str(e)}")
            return None

    def calculate_route_efficiency(self, route_data):
        """Calculate route efficiency metrics"""
        try:
            if not route_data or 'statistics' not in route_data:
                return None
            
            stats = route_data['statistics']
            
            # Calculate efficiency metrics
            efficiency_metrics = {
                'distance_efficiency': None,
                'time_efficiency': None,
                'energy_efficiency': stats['efficiency'],
                'charging_efficiency': None
            }
            
            # Distance efficiency (actual vs direct)
            if len(route_data['waypoints']) >= 2:
                source = route_data['waypoints'][0]['coordinates']
                dest = route_data['waypoints'][-1]['coordinates']
                direct_distance = geodesic(source, dest).kilometers
                
                efficiency_metrics['distance_efficiency'] = (
                    direct_distance / stats['total_distance']
                ) * 100
            
            # Time efficiency (driving time vs total time)
            driving_time = stats['total_distance'] / 80  # Assuming 80 km/h
            efficiency_metrics['time_efficiency'] = (
                driving_time / stats['total_time']
            ) * 100
            
            # Charging efficiency (stops vs distance)
            if stats['charging_stops'] > 0:
                efficiency_metrics['charging_efficiency'] = (
                    stats['total_distance'] / stats['charging_stops']
                )
            
            return efficiency_metrics
        
        except Exception as e:
            print(f"Efficiency calculation failed: {str(e)}")
            return None
    
    def validate_route(self, route_data, ev_specs):
        """Validate the generated route"""
        try:
            if not route_data:
                return False, "No route data provided"
            
            # Check if all required fields are present
            required_fields = ['waypoints', 'charging_stops', 'statistics']
            for field in required_fields:
                if field not in route_data:
                    return False, f"Missing required field: {field}"
            
            # Validate waypoints
            if len(route_data['waypoints']) < 2:
                return False, "Route must have at least source and destination"
            
            # Validate battery constraints
            if 'energy_profile' in route_data:
                battery_levels = route_data['energy_profile']['battery_level']
                if any(level < 0 for level in battery_levels):
                    return False, "Battery level goes below 0%"
                
                # Check if battery level goes below safety margin
                safety_threshold = ev_specs['safety_margin']
                if any(level < safety_threshold for level in battery_levels[:-1]):  # Exclude final level
                    return False, f"Battery level goes below safety margin ({safety_threshold}%)"
            
            return True, "Route validation successful"
        
        except Exception as e:
            return False, f"Route validation failed: {str(e)}"
