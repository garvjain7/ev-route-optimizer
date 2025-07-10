import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic
import streamlit as st

class ClusteringEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
    
    def cluster_stations(self, stations_df, n_clusters=8, include_features=True):
        """Perform K-means clustering on charging stations"""
        try:
            if len(stations_df) < n_clusters:
                st.warning(f"Not enough stations ({len(stations_df)}) for {n_clusters} clusters")
                n_clusters = max(1, len(stations_df))
            
            # Prepare features for clustering
            features = self._prepare_features(stations_df, include_features)
            
            # Perform K-means clustering
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = self.kmeans.fit_predict(features)
            
            # Add cluster labels to the dataframe
            clustered_stations = stations_df.copy()
            clustered_stations['cluster'] = cluster_labels
            
            # Calculate cluster statistics
            cluster_stats = self._calculate_cluster_stats(clustered_stations)
            
            # Add cluster centroids
            centroids = self._calculate_centroids(clustered_stations)
            
            return {
                'stations': clustered_stations,
                'centroids': centroids,
                'statistics': cluster_stats,
                'n_clusters': n_clusters
            }
        
        except Exception as e:
            st.error(f"Clustering failed: {str(e)}")
            return None
    
    def _prepare_features(self, stations_df, include_features=True):
        """Prepare features for clustering"""
        try:
            # Start with coordinates
            features = stations_df[['latitude', 'longitude']].copy()
            
            if include_features:
                # Add additional features if available
                feature_columns = []
                
                # Charging port counts
                if 'level1_ports' in stations_df.columns:
                    features['level1_ports'] = pd.to_numeric(stations_df['level1_ports'], errors='coerce').fillna(0)
                    feature_columns.append('level1_ports')
                
                if 'level2_ports' in stations_df.columns:
                    features['level2_ports'] = pd.to_numeric(stations_df['level2_ports'], errors='coerce').fillna(0)
                    feature_columns.append('level2_ports')
                
                if 'dc_fast_ports' in stations_df.columns:
                    features['dc_fast_ports'] = pd.to_numeric(stations_df['dc_fast_ports'], errors='coerce').fillna(0)
                    feature_columns.append('dc_fast_ports')
                
                # Network encoding (if available)
                if 'network' in stations_df.columns:
                    # Use simple encoding for network (count of stations per network)
                    network_counts = stations_df['network'].value_counts()
                    features['network_size'] = stations_df['network'].map(network_counts).fillna(1)
                    feature_columns.append('network_size')
                
                # Access type encoding
                if 'access' in stations_df.columns:
                    features['is_public'] = (stations_df['access'].str.lower() == 'public').astype(int)
                    feature_columns.append('is_public')
                
                # Scale features (coordinates get higher weight)
                coord_features = features[['latitude', 'longitude']].values
                
                if feature_columns:
                    other_features = features[feature_columns].values
                    # Normalize other features
                    other_features_scaled = self.scaler.fit_transform(other_features)
                    # Combine with coordinates (coordinates get 70% weight)
                    combined_features = np.hstack([
                        coord_features * 0.7,
                        other_features_scaled * 0.3
                    ])
                    return combined_features
                else:
                    return coord_features
            else:
                return features[['latitude', 'longitude']].values
        
        except Exception as e:
            st.error(f"Feature preparation failed: {str(e)}")
            return stations_df[['latitude', 'longitude']].values
    
    def _calculate_cluster_stats(self, clustered_stations):
        """Calculate statistics for each cluster"""
        try:
            stats = {}
            
            for cluster_id in clustered_stations['cluster'].unique():
                cluster_data = clustered_stations[clustered_stations['cluster'] == cluster_id]
                
                # Basic statistics
                stats[cluster_id] = {
                    'station_count': len(cluster_data),
                    'avg_latitude': cluster_data['latitude'].mean(),
                    'avg_longitude': cluster_data['longitude'].mean(),
                    'lat_std': cluster_data['latitude'].std(),
                    'lon_std': cluster_data['longitude'].std()
                }
                
                # Calculate cluster spread (average distance from centroid)
                centroid = (stats[cluster_id]['avg_latitude'], stats[cluster_id]['avg_longitude'])
                distances = cluster_data.apply(
                    lambda row: geodesic(centroid, (row['latitude'], row['longitude'])).kilometers,
                    axis=1
                )
                stats[cluster_id]['avg_spread'] = distances.mean()
                stats[cluster_id]['max_spread'] = distances.max()
                
                # Port statistics if available
                if 'level2_ports' in cluster_data.columns:
                    stats[cluster_id]['total_level2_ports'] = pd.to_numeric(
                        cluster_data['level2_ports'], errors='coerce'
                    ).sum()
                
                if 'dc_fast_ports' in cluster_data.columns:
                    stats[cluster_id]['total_dc_fast_ports'] = pd.to_numeric(
                        cluster_data['dc_fast_ports'], errors='coerce'
                    ).sum()
            
            return stats
        
        except Exception as e:
            st.error(f"Cluster statistics calculation failed: {str(e)}")
            return {}
    
    def _calculate_centroids(self, clustered_stations):
        """Calculate centroid coordinates for each cluster"""
        try:
            centroids = {}
            
            for cluster_id in clustered_stations['cluster'].unique():
                cluster_data = clustered_stations[clustered_stations['cluster'] == cluster_id]
                
                centroid_lat = cluster_data['latitude'].mean()
                centroid_lon = cluster_data['longitude'].mean()
                
                centroids[cluster_id] = {
                    'latitude': centroid_lat,
                    'longitude': centroid_lon,
                    'station_count': len(cluster_data)
                }
            
            return centroids
        
        except Exception as e:
            st.error(f"Centroid calculation failed: {str(e)}")
            return {}
    
    def select_optimal_stations(self, clustered_data, max_stations=10):
        """Select optimal stations from clusters for routing"""
        try:
            if not clustered_data or 'stations' not in clustered_data:
                return None
            
            stations_df = clustered_data['stations']
            centroids = clustered_data['centroids']
            
            selected_stations = []
            
            # Select one representative station from each cluster
            for cluster_id in stations_df['cluster'].unique():
                cluster_stations = stations_df[stations_df['cluster'] == cluster_id]
                
                if len(cluster_stations) == 0:
                    continue
                
                # Find station closest to cluster centroid
                centroid = centroids[cluster_id]
                centroid_coords = (centroid['latitude'], centroid['longitude'])
                
                cluster_stations = cluster_stations.copy()
                cluster_stations['distance_to_centroid'] = cluster_stations.apply(
                    lambda row: geodesic(centroid_coords, (row['latitude'], row['longitude'])).kilometers,
                    axis=1
                )
                
                # Select station with best combination of centrality and features
                # Priority: closest to centroid, then most ports
                cluster_stations['selection_score'] = cluster_stations['distance_to_centroid']
                
                # Bonus for more charging ports
                if 'dc_fast_ports' in cluster_stations.columns:
                    dc_ports = pd.to_numeric(cluster_stations['dc_fast_ports'], errors='coerce').fillna(0)
                    cluster_stations['selection_score'] -= dc_ports * 0.5  # Lower score is better
                
                if 'level2_ports' in cluster_stations.columns:
                    l2_ports = pd.to_numeric(cluster_stations['level2_ports'], errors='coerce').fillna(0)
                    cluster_stations['selection_score'] -= l2_ports * 0.2
                
                best_station = cluster_stations.loc[cluster_stations['selection_score'].idxmin()]
                selected_stations.append(best_station)
            
            # Convert to DataFrame and limit to max_stations
            selected_df = pd.DataFrame(selected_stations)
            if len(selected_df) > max_stations:
                # Keep stations with best selection scores
                selected_df = selected_df.nsmallest(max_stations, 'selection_score')
            
            return selected_df
        
        except Exception as e:
            st.error(f"Station selection failed: {str(e)}")
            return None
    
    def get_cluster_summary(self, clustered_data):
        """Get a summary of the clustering results"""
        try:
            if not clustered_data or 'statistics' not in clustered_data:
                return None
            
            stats = clustered_data['statistics']
            
            summary = {
                'total_clusters': len(stats),
                'total_stations': sum(cluster['station_count'] for cluster in stats.values()),
                'avg_stations_per_cluster': np.mean([cluster['station_count'] for cluster in stats.values()]),
                'avg_cluster_spread': np.mean([cluster['avg_spread'] for cluster in stats.values()]),
                'max_cluster_spread': max([cluster['max_spread'] for cluster in stats.values()])
            }
            
            return summary
        
        except Exception as e:
            st.error(f"Cluster summary generation failed: {str(e)}")
            return None
