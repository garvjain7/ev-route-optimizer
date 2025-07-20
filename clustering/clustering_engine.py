import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic

class ClusteringEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
        self.max_stations_limit = 10000  # Prevent memory issues for large datasets

    def determine_optimal_clusters(self, num_stations):
        """Determine optimal number of clusters based on dataset size."""
        if num_stations <= 1:
            return 1
        elif num_stations <= 10:
            return 1
        elif num_stations <= 30:
            return 2
        elif num_stations <= 50:
            return 3
        elif num_stations <= 70:
            return 4
        elif num_stations <= 100:
            return 6
        elif num_stations <= 150:
            return 7
        else:
            return min(8, num_stations // 10)  # Scale clusters for large datasets

    def _validate_data(self, stations_df):
        """Validate input DataFrame for required columns, coordinate ranges, and duplicates."""
        if stations_df.empty:
            print("[ERROR] Input DataFrame is empty.")
            return False

        required_cols = {'latitude', 'longitude'}
        if not required_cols.issubset(stations_df.columns):
            print("[ERROR] Missing required columns: 'latitude' and/or 'longitude'.")
            return False

        # Check for valid coordinate ranges
        invalid_coords = (
            (stations_df['latitude'].abs() > 90) | 
            (stations_df['longitude'].abs() > 180) |
            (stations_df['latitude'].isna()) |
            (stations_df['longitude'].isna())
        )
        if invalid_coords.any():
            print(f"[ERROR] Invalid coordinates found in {invalid_coords.sum()} rows.")
            return False

        # Check for duplicates
        duplicates = stations_df.duplicated(subset=['latitude', 'longitude'], keep=False)
        if duplicates.any():
            print(f"[WARNING] Found {duplicates.sum()} duplicate stations. Dropping duplicates.")
            stations_df.drop_duplicates(subset=['latitude', 'longitude'], inplace=True)
            if stations_df.empty:
                print("[ERROR] DataFrame is empty after removing duplicates.")
                return False

        # Check dataset size
        if len(stations_df) > self.max_stations_limit:
            print(f"[ERROR] Dataset size ({len(stations_df)}) exceeds limit ({self.max_stations_limit}).")
            return False

        return True

    def cluster_stations(self, stations_df, requested_clusters=None, include_features=True):
        """Cluster stations with improved validation and consistent error handling."""
        try:
            if not self._validate_data(stations_df):
                return None, 0

            # Handle single station case
            if len(stations_df) == 1:
                print("[INFO] Single station, assigning to cluster 0.")
                stations_df['cluster'] = 0
                return {
                    'stations': stations_df,
                    'centroids': self._calculate_centroids(stations_df),
                    'statistics': self._calculate_cluster_stats(stations_df),
                    'n_clusters': 1
                }, 1

            # Handle few stations case
            if len(stations_df) <= 10:
                print("[INFO] Few stations, assigning to single cluster.")
                stations_df['cluster'] = 0
                return {
                    'stations': stations_df,
                    'centroids': self._calculate_centroids(stations_df),
                    'statistics': self._calculate_cluster_stats(stations_df),
                    'n_clusters': 1
                }, 1

            optimal_clusters = self.determine_optimal_clusters(len(stations_df))
            n_clusters = min(requested_clusters, len(stations_df)) if requested_clusters and requested_clusters > 0 else optimal_clusters

            features = self._prepare_features(stations_df, include_features)
            if features.shape[0] == 0:
                print("[ERROR] No features available for clustering.")
                return None, 0

            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            stations_df['cluster'] = self.kmeans.fit_predict(features)

            # Check for empty clusters
            unique_clusters = stations_df['cluster'].nunique()
            if unique_clusters < n_clusters:
                print(f"[WARNING] Only {unique_clusters} clusters formed instead of requested {n_clusters}.")

            return {
                'stations': stations_df,
                'centroids': self._calculate_centroids(stations_df),
                'statistics': self._calculate_cluster_stats(stations_df),
                'n_clusters': unique_clusters
            }, unique_clusters

        except Exception as e:
            print(f"[ERROR] Clustering failed: {str(e)}")
            return None, 0

    def _prepare_features(self, stations_df, include_features=True):
        """Prepare features with consistent scaling and additional EV-specific features."""
        try:
            if stations_df.empty:
                print("[ERROR] Input DataFrame is empty. Cannot prepare features.")
                return np.empty((0, 2))

            features = stations_df[['latitude', 'longitude']].copy()
            feature_columns = []
            if include_features:
                for col in ['level1_ports', 'level2_ports', 'dc_fast_ports', 'rating', 'power_kw', 'congestion_score']:
                    if col in stations_df.columns:
                        converted = pd.to_numeric(stations_df[col], errors='coerce')
                        if isinstance(converted, pd.Series):
                            features[col] = converted.fillna(0)
                        else:
                            features[col] = float(converted) if pd.notna(converted) else 0
                        feature_columns.append(col)
                if 'network' in stations_df.columns:
                    network_counts = stations_df['network'].value_counts()
                    features['network_size'] = stations_df['network'].map(network_counts).fillna(1)
                    feature_columns.append('network_size')
                if 'access' in stations_df.columns:
                    features['is_public'] = (stations_df['access'].str.lower() == 'public').astype(int)
                    feature_columns.append('is_public')

            # Apply consistent scaling across all features
            if feature_columns:
                all_features = self.scaler.fit_transform(features[['latitude', 'longitude'] + feature_columns])
                weights = np.array([0.7, 0.7] + [0.3 / len(feature_columns)] * len(feature_columns))
                return all_features * weights
            return self.scaler.fit_transform(features[['latitude', 'longitude']]) * 0.7

        except Exception as e:
            print(f"[ERROR] Feature preparation failed: {str(e)}")
            return stations_df[['latitude', 'longitude']].fillna(0).values

    def _calculate_cluster_stats(self, clustered_stations):
        """Calculate cluster statistics with additional metrics."""
        try:
            stats = {}
            for cluster_id in clustered_stations['cluster'].unique():
                group = clustered_stations[clustered_stations['cluster'] == cluster_id]
                if group.empty:
                    print(f"[WARNING] Cluster {cluster_id} is empty. Skipping stats calculation.")
                    continue
                centroid = (group['latitude'].mean(), group['longitude'].mean())
                coords = group[['latitude', 'longitude']].values
                distances = np.array([geodesic((lat, lon), centroid).km for lat, lon in coords])
                stats[cluster_id] = {
                    'station_count': len(group),
                    'avg_latitude': centroid[0],
                    'avg_longitude': centroid[1],
                    'lat_std': group['latitude'].std() if len(group) > 1 else 0,
                    'lon_std': group['longitude'].std() if len(group) > 1 else 0,
                    'avg_spread': distances.mean() if len(distances) > 0 else 0,
                    'max_spread': distances.max() if len(distances) > 0 else 0
                }
                for col in ['level2_ports', 'dc_fast_ports']:
                    if col in group.columns:
                        converted = pd.to_numeric(group[col], errors='coerce')
                        if isinstance(converted, pd.Series):
                            stats[cluster_id][f'total_{col}'] = converted.sum()
                        else:
                            stats[cluster_id][f'total_{col}'] = float(converted) if pd.notna(converted) else 0
            return stats
        except Exception as e:
            print(f"[ERROR] Cluster statistics calculation failed: {str(e)}")
            return {}

    def _calculate_centroids(self, clustered_stations):
        """Calculate centroids efficiently."""
        try:
            centroids = {}
            for cluster_id in clustered_stations['cluster'].unique():
                group = clustered_stations[clustered_stations['cluster'] == cluster_id]
                if group.empty:
                    print(f"[WARNING] Cluster {cluster_id} is empty. Skipping centroid calculation.")
                    continue
                centroids[cluster_id] = {
                    'latitude': group['latitude'].mean(),
                    'longitude': group['longitude'].mean(),
                    'station_count': len(group)
                }
            return centroids
        except Exception as e:
            print(f"[ERROR] Centroid calculation failed: {str(e)}")
            return {}

    def select_optimal_stations(self, clustered_data, max_stations=10):
        """Select one optimal station per cluster, capped at max_stations."""
        try:
            if not clustered_data or 'stations' not in clustered_data or clustered_data['stations'].empty:
                print("[ERROR] Invalid or empty clustered data.")
                return pd.DataFrame([])

            stations_df = clustered_data['stations']
            centroids = clustered_data['centroids']
            if not centroids:
                print("[ERROR] No centroids available for station selection.")
                return pd.DataFrame([])

            selected_stations = []
            for cluster_id in stations_df['cluster'].unique():
                cluster_stations = stations_df[stations_df['cluster'] == cluster_id].copy()
                if cluster_stations.empty:
                    print(f"[WARNING] No stations in cluster {cluster_id}. Skipping.")
                    continue
                centroid = centroids.get(cluster_id, {'latitude': cluster_stations['latitude'].mean(), 'longitude': cluster_stations['longitude'].mean()})
                coords = cluster_stations[['latitude', 'longitude']].values
                distances = np.array([geodesic((lat, lon), (centroid['latitude'], centroid['longitude'])).km for lat, lon in coords])
                cluster_stations['distance_to_centroid'] = distances
                cluster_stations['selection_score'] = cluster_stations['distance_to_centroid']

                for col, weight in [('dc_fast_ports', -0.5), ('level2_ports', -0.2), ('rating', -0.5), ('power_kw', -0.1), ('congestion_score', 0.3)]:
                    if col in cluster_stations.columns:
                        converted = pd.to_numeric(cluster_stations[col], errors='coerce')
                        if isinstance(converted, pd.Series):
                            cluster_stations['selection_score'] += converted.fillna(0) * weight
                        else:
                            try:
                                val = float(converted) if pd.notna(converted) else 0
                                cluster_stations['selection_score'] += val * weight
                            except Exception:
                                pass

                best_station = cluster_stations.loc[cluster_stations['selection_score'].idxmin()]
                selected_stations.append(best_station)

            if not selected_stations:
                print("[WARNING] No optimal stations selected from clusters.")
                return pd.DataFrame([])

            selected_df = pd.DataFrame(selected_stations)
            if len(selected_df) > max_stations:
                selected_df = selected_df.nsmallest(max_stations, 'selection_score')
            return selected_df

        except Exception as e:
            print(f"[ERROR] Station selection failed: {str(e)}")
            return pd.DataFrame([])

    def get_cluster_summary(self, clustered_data):
        """Generate cluster summary with robust error handling."""
        try:
            if not clustered_data or 'statistics' not in clustered_data:
                print("[ERROR] Invalid or missing clustered data statistics.")
                return None
            stats = clustered_data['statistics']
            if not stats:
                print("[ERROR] No statistics available for summary.")
                return None
            return {
                'total_clusters': len(stats),
                'total_stations': sum(s['station_count'] for s in stats.values()),
                'avg_stations_per_cluster': np.mean([s['station_count'] for s in stats.values()]) if stats else 0,
                'avg_cluster_spread': np.mean([s['avg_spread'] for s in stats.values()]) if stats else 0,
                'max_cluster_spread': max([s['max_spread'] for s in stats.values()]) if stats else 0
            }
        except Exception as e:
            print(f"[ERROR] Summary generation failed: {str(e)}")
            return None
        