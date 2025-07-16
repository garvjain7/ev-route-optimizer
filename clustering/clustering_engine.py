import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic

class ClusteringEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None

    def cluster_stations(self, stations_df, n_clusters=8, include_features=True):
        try:
            required_cols = {'latitude', 'longitude'}
            if not required_cols.issubset(stations_df.columns):
                print("[ERROR] Missing latitude or longitude columns in input dataset.")
                return None
            if stations_df.empty:
                print("[ERROR] Input DataFrame is empty. Cannot perform clustering.")
                return None
            if len(stations_df) < n_clusters:
                print(f"[WARNING] Not enough stations ({len(stations_df)}) for {n_clusters} clusters. Adjusting to minimum.")
                n_clusters = max(1, len(stations_df))
            features = self._prepare_features(stations_df, include_features)
            if features.shape[0] == 0:
                print("[ERROR] No features available for clustering.")
                return None
            # Remove n_init for compatibility with older sklearn versions
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = self.kmeans.fit_predict(features)
            clustered_stations = stations_df.copy()
            clustered_stations['cluster'] = cluster_labels
            cluster_stats = self._calculate_cluster_stats(clustered_stations)
            centroids = self._calculate_centroids(clustered_stations)
            return {
                'stations': clustered_stations,
                'centroids': centroids,
                'statistics': cluster_stats,
                'n_clusters': n_clusters
            }
        except Exception as e:
            print(f"[ERROR] Clustering failed: {str(e)}")
            return None

    def _prepare_features(self, stations_df, include_features=True):
        try:
            if stations_df.empty:
                print("[ERROR] Input DataFrame is empty. Cannot prepare features.")
                return np.empty((0, 2))
            features = stations_df[['latitude', 'longitude']].copy()
            feature_columns = []
            if include_features:
                if 'level1_ports' in stations_df.columns:
                    col = pd.to_numeric(stations_df['level1_ports'], errors='coerce')
                    if isinstance(col, pd.Series):
                        features['level1_ports'] = col.fillna(0)
                    else:
                        features['level1_ports'] = col
                    feature_columns.append('level1_ports')
                if 'level2_ports' in stations_df.columns:
                    col = pd.to_numeric(stations_df['level2_ports'], errors='coerce')
                    if isinstance(col, pd.Series):
                        features['level2_ports'] = col.fillna(0)
                    else:
                        features['level2_ports'] = col
                    feature_columns.append('level2_ports')
                if 'dc_fast_ports' in stations_df.columns:
                    col = pd.to_numeric(stations_df['dc_fast_ports'], errors='coerce')
                    if isinstance(col, pd.Series):
                        features['dc_fast_ports'] = col.fillna(0)
                    else:
                        features['dc_fast_ports'] = col
                    feature_columns.append('dc_fast_ports')
                if 'network' in stations_df.columns:
                    network_counts = stations_df['network'].value_counts()
                    features['network_size'] = stations_df['network'].map(network_counts).fillna(1)
                    feature_columns.append('network_size')
                if 'access' in stations_df.columns:
                    features['is_public'] = (stations_df['access'].str.lower() == 'public').astype(int)
                    feature_columns.append('is_public')
                coord_features = features[['latitude', 'longitude']].values
                if feature_columns:
                    other_features = features[feature_columns].values
                    other_features_scaled = self.scaler.fit_transform(other_features)
                    combined = np.hstack([coord_features * 0.7, other_features_scaled * 0.3])
                    return combined
                else:
                    # No extra features, use only coordinates
                    return features[['latitude', 'longitude']].values
            return features[['latitude', 'longitude']].values
        except Exception as e:
            print(f"[ERROR] Feature preparation failed: {str(e)}")
            return stations_df[['latitude', 'longitude']].fillna(0).values

    def _calculate_cluster_stats(self, clustered_stations):
        try:
            stats = {}
            for cluster_id in clustered_stations['cluster'].unique():
                cluster_data = clustered_stations[clustered_stations['cluster'] == cluster_id]
                if cluster_data.empty:
                    print(f"[WARNING] Cluster {cluster_id} is empty. Skipping stats calculation.")
                    continue
                stats[cluster_id] = {
                    'station_count': len(cluster_data),
                    'avg_latitude': cluster_data['latitude'].mean(),
                    'avg_longitude': cluster_data['longitude'].mean(),
                    'lat_std': cluster_data['latitude'].std(),
                    'lon_std': cluster_data['longitude'].std()
                }
                centroid = (stats[cluster_id]['avg_latitude'], stats[cluster_id]['avg_longitude'])
                distances = cluster_data.apply(
                    lambda row: geodesic(centroid, (row['latitude'], row['longitude'])).kilometers, axis=1)
                stats[cluster_id]['avg_spread'] = distances.mean() if not distances.empty else 0
                stats[cluster_id]['max_spread'] = distances.max() if not distances.empty else 0
                if 'level2_ports' in cluster_data.columns:
                    col = pd.to_numeric(cluster_data['level2_ports'], errors='coerce')
                    if isinstance(col, pd.Series):
                        stats[cluster_id]['total_level2_ports'] = col.sum()
                    else:
                        stats[cluster_id]['total_level2_ports'] = col
                if 'dc_fast_ports' in cluster_data.columns:
                    col = pd.to_numeric(cluster_data['dc_fast_ports'], errors='coerce')
                    if isinstance(col, pd.Series):
                        stats[cluster_id]['total_dc_fast_ports'] = col.sum()
                    else:
                        stats[cluster_id]['total_dc_fast_ports'] = col
            return stats
        except Exception as e:
            print(f"[ERROR] Cluster statistics calculation failed: {str(e)}")
            return {}

    def _calculate_centroids(self, clustered_stations):
        try:
            centroids = {}
            for cluster_id in clustered_stations['cluster'].unique():
                cluster_data = clustered_stations[clustered_stations['cluster'] == cluster_id]
                if cluster_data.empty:
                    print(f"[WARNING] Cluster {cluster_id} is empty. Skipping centroid calculation.")
                    continue
                centroids[cluster_id] = {
                    'latitude': cluster_data['latitude'].mean(),
                    'longitude': cluster_data['longitude'].mean(),
                    'station_count': len(cluster_data)
                }
            return centroids
        except Exception as e:
            print(f"[ERROR] Centroid calculation failed: {str(e)}")
            return {}

    def select_optimal_stations(self, clustered_data, max_stations=10):
        try:
            if not clustered_data or 'stations' not in clustered_data:
                print("[ERROR] Clustered data is missing or malformed.")
                return pd.DataFrame([])
            stations_df = clustered_data['stations']
            centroids = clustered_data['centroids']
            selected_stations = []
            for cluster_id in stations_df['cluster'].unique():
                cluster_stations = stations_df[stations_df['cluster'] == cluster_id].copy()
                if cluster_stations.empty:
                    print(f"[WARNING] No stations in cluster {cluster_id}. Skipping.")
                    continue
                centroid = centroids[cluster_id]
                centroid_coords = (centroid['latitude'], centroid['longitude'])
                cluster_stations['distance_to_centroid'] = cluster_stations.apply(
                    lambda row: geodesic(centroid_coords, (row['latitude'], row['longitude'])).kilometers, axis=1)
                cluster_stations['selection_score'] = cluster_stations['distance_to_centroid']
                if 'dc_fast_ports' in cluster_stations.columns:
                    dc_ports = pd.to_numeric(cluster_stations['dc_fast_ports'], errors='coerce')
                    if isinstance(dc_ports, pd.Series):
                        dc_ports = dc_ports.fillna(0).astype(float).to_numpy()
                        cluster_stations['selection_score'] = cluster_stations['selection_score'] - dc_ports * 0.5
                    else:
                        try:
                            val = float(dc_ports)
                            cluster_stations['selection_score'] = cluster_stations['selection_score'].apply(lambda x: x - val * 0.5)
                        except Exception:
                            pass
                if 'level2_ports' in cluster_stations.columns:
                    l2_ports = pd.to_numeric(cluster_stations['level2_ports'], errors='coerce')
                    if isinstance(l2_ports, pd.Series):
                        l2_ports = l2_ports.fillna(0).astype(float).to_numpy()
                        cluster_stations['selection_score'] = cluster_stations['selection_score'] - l2_ports * 0.2
                    else:
                        try:
                            val = float(l2_ports)
                            cluster_stations['selection_score'] = cluster_stations['selection_score'].apply(lambda x: x - val * 0.2)
                        except Exception:
                            pass
                best_station = cluster_stations.loc[cluster_stations['selection_score'].idxmin()]
                selected_stations.append(best_station)
            selected_df = pd.DataFrame(selected_stations)
            if selected_df.empty:
                print("[WARNING] No optimal stations selected from clusters.")
                return selected_df
            if len(selected_df) > max_stations:
                selected_df = selected_df.nsmallest(max_stations, 'selection_score')
            return selected_df
        except Exception as e:
            print(f"[ERROR] Station selection failed: {str(e)}")
            return pd.DataFrame([])

    def get_cluster_summary(self, clustered_data):
        try:
            if not clustered_data or 'statistics' not in clustered_data:
                return None
            stats = clustered_data['statistics']
            summary = {
                'total_clusters': len(stats),
                'total_stations': sum(c['station_count'] for c in stats.values()),
                'avg_stations_per_cluster': np.mean([c['station_count'] for c in stats.values()]),
                'avg_cluster_spread': np.mean([c['avg_spread'] for c in stats.values()]),
                'max_cluster_spread': max([c['max_spread'] for c in stats.values()])
            }
            return summary
        except Exception as e:
            print(f"[ERROR] Cluster summary generation failed: {str(e)}")
            return None
