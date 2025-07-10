import folium
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import streamlit as st

class MapVisualizer:
    def __init__(self):
        self.color_palette = {
            'source': 'green',
            'destination': 'red',
            'charging_station': 'blue',
            'cluster_0': 'purple',
            'cluster_1': 'orange',
            'cluster_2': 'darkgreen',
            'cluster_3': 'lightred',
            'cluster_4': 'beige',
            'cluster_5': 'darkblue',
            'cluster_6': 'lightgreen',
            'cluster_7': 'cadetblue',
            'cluster_8': 'darkpurple',
            'cluster_9': 'lightblue'
        }
    
    def create_route_map(self, source_coords, dest_coords, stations_df=None, route_data=None):
        """Create an interactive map showing the route and charging stations"""
        try:
            # Calculate map center
            center_lat = (source_coords[0] + dest_coords[0]) / 2
            center_lon = (source_coords[1] + dest_coords[1]) / 2
            
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=8,
                tiles='OpenStreetMap'
            )
            
            # Add source marker
            folium.Marker(
                location=source_coords,
                popup="🚗 Start Location",
                tooltip="Source",
                icon=folium.Icon(color='green', icon='play')
            ).add_to(m)
            
            # Add destination marker
            folium.Marker(
                location=dest_coords,
                popup="🏁 Destination",
                tooltip="Destination",
                icon=folium.Icon(color='red', icon='stop')
            ).add_to(m)
            
            # Add direct route line
            folium.PolyLine(
                locations=[source_coords, dest_coords],
                color='gray',
                weight=2,
                opacity=0.5,
                popup="Direct Route"
            ).add_to(m)
            
            # Add charging stations
            if stations_df is not None:
                self._add_charging_stations(m, stations_df)
            
            # Add optimized route
            if route_data is not None:
                self._add_optimized_route(m, route_data)
            
            # Add map controls
            folium.LayerControl().add_to(m)
            
            # Add fullscreen button
            folium.plugins.Fullscreen().add_to(m)
            
            # Add measure tool
            folium.plugins.MeasureControl().add_to(m)
            
            # Fit bounds to show all markers
            self._fit_bounds(m, source_coords, dest_coords, stations_df)
            
            return m
        
        except Exception as e:
            st.error(f"Map creation failed: {str(e)}")
            return self._create_fallback_map(source_coords, dest_coords)
    
    def _add_charging_stations(self, map_obj, stations_df):
        """Add charging stations to the map"""
        try:
            # Create feature groups for different types of stations
            all_stations_group = folium.FeatureGroup(name="All Charging Stations")
            
            # Add clustered stations if cluster column exists
            if 'cluster' in stations_df.columns:
                cluster_groups = {}
                for cluster_id in stations_df['cluster'].unique():
                    cluster_groups[cluster_id] = folium.FeatureGroup(
                        name=f"Cluster {cluster_id}"
                    )
            
            for idx, station in stations_df.iterrows():
                # Determine marker color
                if 'cluster' in stations_df.columns:
                    cluster_id = station['cluster']
                    color = self.color_palette.get(f'cluster_{cluster_id}', 'blue')
                else:
                    color = 'blue'
                
                # Create popup content
                popup_content = self._create_station_popup(station)
                
                # Create marker
                marker = folium.Marker(
                    location=[station['latitude'], station['longitude']],
                    popup=popup_content,
                    tooltip=station.get('name', 'EV Station'),
                    icon=folium.Icon(color=color, icon='bolt')
                )
                
                # Add to appropriate groups
                marker.add_to(all_stations_group)
                
                if 'cluster' in stations_df.columns:
                    cluster_id = station['cluster']
                    marker.add_to(cluster_groups[cluster_id])
            
            # Add feature groups to map
            all_stations_group.add_to(map_obj)
            
            if 'cluster' in stations_df.columns:
                for group in cluster_groups.values():
                    group.add_to(map_obj)
        
        except Exception as e:
            st.error(f"Failed to add charging stations: {str(e)}")
    
    def _create_station_popup(self, station):
        """Create detailed popup content for charging station"""
        try:
            popup_html = f"""
            <div style="width: 200px;">
                <h4>🔌 {station.get('name', 'EV Station')}</h4>
                <p><strong>Network:</strong> {station.get('network', 'Unknown')}</p>
                <p><strong>Access:</strong> {station.get('access', 'Unknown')}</p>
                <p><strong>Location:</strong> {station['latitude']:.4f}, {station['longitude']:.4f}</p>
            """
            
            # Add port information if available
            if 'level2_ports' in station:
                l2_ports = pd.to_numeric(station['level2_ports'], errors='coerce')
                if not pd.isna(l2_ports) and l2_ports > 0:
                    popup_html += f"<p><strong>Level 2 Ports:</strong> {int(l2_ports)}</p>"
            
            if 'dc_fast_ports' in station:
                dc_ports = pd.to_numeric(station['dc_fast_ports'], errors='coerce')
                if not pd.isna(dc_ports) and dc_ports > 0:
                    popup_html += f"<p><strong>DC Fast Ports:</strong> {int(dc_ports)}</p>"
            
            # Add distance information if available
            if 'distance_from_source' in station:
                popup_html += f"<p><strong>Distance from Source:</strong> {station['distance_from_source']:.1f} km</p>"
            
            if 'distance_from_dest' in station:
                popup_html += f"<p><strong>Distance to Destination:</strong> {station['distance_from_dest']:.1f} km</p>"
            
            # Add cluster information if available
            if 'cluster' in station:
                popup_html += f"<p><strong>Cluster:</strong> {station['cluster']}</p>"
            
            popup_html += "</div>"
            
            return popup_html
        
        except Exception as e:
            return f"<div>EV Station<br>Error loading details: {str(e)}</div>"
    
    def _add_optimized_route(self, map_obj, route_data):
        """Add optimized route to the map"""
        try:
            if not route_data or 'waypoints' not in route_data:
                return
            
            # Create route feature group
            route_group = folium.FeatureGroup(name="Optimized Route")
            
            # Extract coordinates from waypoints
            route_coords = []
            for waypoint in route_data['waypoints']:
                route_coords.append(waypoint['coordinates'])
            
            # Add route line
            folium.PolyLine(
                locations=route_coords,
                color='blue',
                weight=4,
                opacity=0.8,
                popup="Optimized EV Route"
            ).add_to(route_group)
            
            # Add charging stop markers
            if 'charging_stops' in route_data:
                for i, stop in enumerate(route_data['charging_stops']):
                    # Create popup for charging stop
                    popup_content = f"""
                    <div style="width: 200px;">
                        <h4>🔋 Charging Stop {stop['stop_number']}</h4>
                        <p><strong>Station:</strong> {stop['station_name']}</p>
                        <p><strong>Distance from Start:</strong> {stop['distance_from_start']:.1f} km</p>
                        <p><strong>Battery on Arrival:</strong> {stop['battery_on_arrival']:.1f}%</p>
                        <p><strong>Charging Time:</strong> {stop['charging_time']} minutes</p>
                    </div>
                    """
                    
                    folium.Marker(
                        location=stop['coordinates'],
                        popup=popup_content,
                        tooltip=f"Charging Stop {stop['stop_number']}",
                        icon=folium.Icon(color='orange', icon='flash')
                    ).add_to(route_group)
            
            # Add route statistics
            if 'statistics' in route_data:
                stats = route_data['statistics']
                stats_popup = f"""
                <div style="width: 250px;">
                    <h4>📊 Route Statistics</h4>
                    <p><strong>Total Distance:</strong> {stats['total_distance']:.1f} km</p>
                    <p><strong>Charging Stops:</strong> {stats['charging_stops']}</p>
                    <p><strong>Total Time:</strong> {stats['total_time']:.1f} hours</p>
                    <p><strong>Energy Consumed:</strong> {stats['energy_consumed']:.1f} kWh</p>
                    <p><strong>Efficiency:</strong> {stats['efficiency']:.1f} kWh/100km</p>
                </div>
                """
                
                # Add stats marker at midpoint
                if len(route_coords) > 1:
                    midpoint_idx = len(route_coords) // 2
                    midpoint = route_coords[midpoint_idx]
                    
                    folium.Marker(
                        location=midpoint,
                        popup=stats_popup,
                        tooltip="Route Statistics",
                        icon=folium.Icon(color='purple', icon='info-sign')
                    ).add_to(route_group)
            
            route_group.add_to(map_obj)
        
        except Exception as e:
            st.error(f"Failed to add optimized route: {str(e)}")
    
    def _fit_bounds(self, map_obj, source_coords, dest_coords, stations_df=None):
        """Fit map bounds to show all relevant points"""
        try:
            # Start with source and destination
            all_coords = [source_coords, dest_coords]
            
            # Add station coordinates if available
            if stations_df is not None:
                for _, station in stations_df.iterrows():
                    all_coords.append([station['latitude'], station['longitude']])
            
            # Calculate bounds
            if len(all_coords) > 0:
                lats = [coord[0] for coord in all_coords]
                lons = [coord[1] for coord in all_coords]
                
                southwest = [min(lats), min(lons)]
                northeast = [max(lats), max(lons)]
                
                map_obj.fit_bounds([southwest, northeast], padding=(20, 20))
        
        except Exception as e:
            st.error(f"Failed to fit map bounds: {str(e)}")
    
    def _create_fallback_map(self, source_coords, dest_coords):
        """Create a simple fallback map if main map creation fails"""
        try:
            center_lat = (source_coords[0] + dest_coords[0]) / 2
            center_lon = (source_coords[1] + dest_coords[1]) / 2
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=8
            )
            
            # Add basic markers
            folium.Marker(
                location=source_coords,
                popup="Source",
                icon=folium.Icon(color='green')
            ).add_to(m)
            
            folium.Marker(
                location=dest_coords,
                popup="Destination",
                icon=folium.Icon(color='red')
            ).add_to(m)
            
            # Add direct line
            folium.PolyLine(
                locations=[source_coords, dest_coords],
                color='blue',
                weight=2
            ).add_to(m)
            
            return m
        
        except Exception as e:
            st.error(f"Failed to create fallback map: {str(e)}")
            return None
    
    def create_cluster_analysis_map(self, clustered_data):
        """Create a map specifically for cluster analysis"""
        try:
            if not clustered_data or 'stations' not in clustered_data:
                return None
            
            stations_df = clustered_data['stations']
            centroids = clustered_data['centroids']
            
            # Calculate map center
            center_lat = stations_df['latitude'].mean()
            center_lon = stations_df['longitude'].mean()
            
            # Create map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=10
            )
            
            # Add cluster centroids
            for cluster_id, centroid in centroids.items():
                folium.Marker(
                    location=[centroid['latitude'], centroid['longitude']],
                    popup=f"Cluster {cluster_id} Centroid<br>Stations: {centroid['station_count']}",
                    tooltip=f"Cluster {cluster_id} Center",
                    icon=folium.Icon(color='black', icon='certificate')
                ).add_to(m)
            
            # Add stations with cluster colors
            for idx, station in stations_df.iterrows():
                cluster_id = station['cluster']
                color = self.color_palette.get(f'cluster_{cluster_id}', 'blue')
                
                folium.CircleMarker(
                    location=[station['latitude'], station['longitude']],
                    radius=5,
                    popup=f"Station: {station.get('name', 'Unknown')}<br>Cluster: {cluster_id}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
            
            return m
        
        except Exception as e:
            st.error(f"Cluster analysis map creation failed: {str(e)}")
            return None
