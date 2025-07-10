import pandas as pd
import numpy as np
from geopy.distance import geodesic
import streamlit as st
from utils.distance_calculator import DistanceCalculator

class EVRouter:
    def __init__(self):
        self.distance_calc = DistanceCalculator()
    
    def optimize_route(self, source_coords, dest_coords, stations_df, ev_specs):
        """Optimize EV route with charging stops"""
        try:
            # Extract EV specifications
            battery_range = ev_specs['battery_range']
            consumption_rate = ev_specs['consumption_rate']
            charging_time = ev_specs['charging_time']
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
            st.error(f"Route optimization failed: {str(e)}")
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
            st.error(f"Direct route creation failed: {str(e)}")
            return None
    
    def _create_multi_stop_route(self, source_coords, dest_coords, stations_df, ev_specs, effective_range):
        """Create route with multiple charging stops"""
        try:
            # Use greedy approach to find charging stops
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
                # Find optimal charging station
                charging_station = self._find_optimal_charging_station(
                    current_position, dest_coords, stations_df, effective_range
                )
                
                if charging_station is None:
                    st.warning("Could not find suitable charging station")
                    break
                
                # Calculate distance to charging station
                station_coords = (charging_station['latitude'], charging_station['longitude'])
                distance_to_station = geodesic(current_position, station_coords).kilometers
                
                # Update route data
                total_distance += distance_to_station
                travel_time = distance_to_station / 80  # Assuming 80 km/h
                total_time += travel_time + (ev_specs['charging_time'] / 60)  # Add charging time
                
                # Update battery level
                energy_consumed = (distance_to_station / 100) * ev_specs['consumption_rate']
                battery_used = (energy_consumed / ev_specs['battery_range']) * 100
                current_battery -= battery_used
                
                # Add to energy profile
                energy_profile['distance'].append(total_distance)
                energy_profile['battery_level'].append(current_battery)
                
                # Add charging stop
                charging_stop = {
                    'stop_number': stop_counter,
                    'station_name': charging_station.get('name', f'Station {stop_counter}'),
                    'coordinates': station_coords,
                    'distance_from_start': total_distance,
                    'battery_on_arrival': current_battery,
                    'charging_time': ev_specs['charging_time'],
                    'distance_to_next': None  # Will be calculated
                }
                
                charging_stops.append(charging_stop)
                waypoints.append({
                    'type': 'charging_station',
                    'coordinates': station_coords,
                    'name': charging_stop['station_name']
                })
                
                # Reset battery after charging (assume full charge)
                current_battery = 100
                energy_profile['distance'].append(total_distance)
                energy_profile['battery_level'].append(100)
                
                # Update position and remaining distance
                current_position = station_coords
                remaining_distance = geodesic(current_position, dest_coords).kilometers
                stop_counter += 1
                
                # Safety check to prevent infinite loop
                if stop_counter > 10:
                    st.warning("Maximum number of charging stops reached")
                    break
            
            # Add final leg to destination
            final_distance = geodesic(current_position, dest_coords).kilometers
            total_distance += final_distance
            final_travel_time = final_distance / 80
            total_time += final_travel_time
            
            # Final battery level
            final_energy_consumed = (final_distance / 100) * ev_specs['consumption_rate']
            final_battery_used = (final_energy_consumed / ev_specs['battery_range']) * 100
            current_battery -= final_battery_used
            
            # Update last charging stop with distance to destination
            if charging_stops:
                charging_stops[-1]['distance_to_next'] = final_distance
            
            # Add destination waypoint
            waypoints.append({'type': 'destination', 'coordinates': dest_coords, 'name': 'End'})
            
            # Final energy profile
            energy_profile['distance'].append(total_distance)
            energy_profile['battery_level'].append(current_battery)
            
            # Calculate total energy consumption
            total_energy = (total_distance / 100) * ev_specs['consumption_rate']
            
            route_data = {
                'waypoints': waypoints,
                'charging_stops': charging_stops,
                'statistics': {
                    'total_distance': total_distance,
                    'charging_stops': len(charging_stops),
                    'total_time': total_time,
                    'energy_consumed': total_energy,
                    'battery_used': None,  # Not applicable for multi-stop
                    'efficiency': ev_specs['consumption_rate']
                },
                'energy_profile': energy_profile
            }
            
            return route_data
        
        except Exception as e:
            st.error(f"Multi-stop route creation failed: {str(e)}")
            return None
    
    def _find_optimal_charging_station(self, current_position, dest_coords, stations_df, effective_range):
        """Find the optimal charging station for the next leg"""
        try:
            # Calculate reachable stations from current position
            stations_df = stations_df.copy()
            stations_df['distance_from_current'] = stations_df.apply(
                lambda row: geodesic(current_position, (row['latitude'], row['longitude'])).kilometers,
                axis=1
            )
            
            # Filter stations within effective range
            reachable_stations = stations_df[
                stations_df['distance_from_current'] <= effective_range
            ]
            
            if len(reachable_stations) == 0:
                return None
            
            # Calculate distance from each station to destination
            reachable_stations['distance_to_dest'] = reachable_stations.apply(
                lambda row: geodesic(dest_coords, (row['latitude'], row['longitude'])).kilometers,
                axis=1
            )
            
            # Calculate optimization score
            # Prefer stations that are: closer to destination, not too far from current position
            reachable_stations['optimization_score'] = (
                reachable_stations['distance_to_dest'] * 0.7 +  # Closer to destination is better
                reachable_stations['distance_from_current'] * 0.3  # Not too far from current
            )
            
            # Add bonus for fast charging capabilities
            if 'dc_fast_ports' in reachable_stations.columns:
                dc_ports = pd.to_numeric(reachable_stations['dc_fast_ports'], errors='coerce').fillna(0)
                reachable_stations['optimization_score'] -= dc_ports * 2  # Bonus for DC fast charging
            
            # Select station with best score
            best_station = reachable_stations.loc[reachable_stations['optimization_score'].idxmin()]
            
            return best_station
        
        except Exception as e:
            st.error(f"Charging station selection failed: {str(e)}")
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
            st.error(f"Efficiency calculation failed: {str(e)}")
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
