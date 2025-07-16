import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, classification_report
import joblib
# import streamlit as st
from datetime import datetime
import os

class StationPredictor:
    """ML model to predict optimal charging stations and congestion levels"""
    
    def __init__(self):
        self.congestion_model = GradientBoostingClassifier(random_state=42)
        self.rating_model = RandomForestRegressor(random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.model_path = "ml_models/saved_models/"
        
    def prepare_features(self, stations_df, route_info=None, time_info=None):
        """Prepare features for ML models"""
        try:
            features_df = stations_df.copy()
            
            # Basic station features
            feature_columns = []
            
            # Power level features
            if 'dc_fast_ports' in features_df.columns:
                features_df['dc_fast_ports'] = pd.to_numeric(features_df['dc_fast_ports'], errors='coerce').fillna(0)
                feature_columns.append('dc_fast_ports')
            
            if 'level2_ports' in features_df.columns:
                features_df['level2_ports'] = pd.to_numeric(features_df['level2_ports'], errors='coerce').fillna(0)
                feature_columns.append('level2_ports')
            
            if 'level1_ports' in features_df.columns:
                features_df['level1_ports'] = pd.to_numeric(features_df['level1_ports'], errors='coerce').fillna(0)
                feature_columns.append('level1_ports')
            
            # Calculate total charging capacity
            features_df['total_ports'] = (
                features_df.get('dc_fast_ports', 0) + 
                features_df.get('level2_ports', 0) + 
                features_df.get('level1_ports', 0)
            )
            feature_columns.append('total_ports')
            
            # Fast charging capability
            features_df['has_fast_charging'] = (features_df.get('dc_fast_ports', 0) > 0).astype(int)
            feature_columns.append('has_fast_charging')
            
            # Network encoding
            if 'network' in features_df.columns:
                # Network reliability score (based on network size)
                network_counts = features_df['network'].value_counts()
                features_df['network_reliability'] = features_df['network'].map(network_counts).fillna(1)
                feature_columns.append('network_reliability')
            
            # Access type
            if 'access' in features_df.columns:
                features_df['is_public'] = (features_df['access'].str.lower() == 'public').astype(int)
                feature_columns.append('is_public')
            
            # Time-based features
            if time_info:
                current_time = datetime.now()
                features_df['hour_of_day'] = current_time.hour
                features_df['day_of_week'] = current_time.weekday()
                features_df['is_weekend'] = (current_time.weekday() >= 5).astype(int)
                features_df['is_rush_hour'] = ((current_time.hour >= 7) & (current_time.hour <= 9) | 
                                             (current_time.hour >= 17) & (current_time.hour <= 19)).astype(int)
                feature_columns.extend(['hour_of_day', 'day_of_week', 'is_weekend', 'is_rush_hour'])
            
            # Route-based features
            if route_info:
                # Distance from source and destination
                if 'distance_from_source' in features_df.columns:
                    feature_columns.append('distance_from_source')
                if 'distance_from_dest' in features_df.columns:
                    feature_columns.append('distance_from_dest')
                if 'corridor_distance' in features_df.columns:
                    feature_columns.append('corridor_distance')
                
                # Route efficiency
                if 'detour_ratio' in features_df.columns:
                    feature_columns.append('detour_ratio')
            
            # Geographic density features
            features_df['station_density'] = self._calculate_station_density(features_df)
            feature_columns.append('station_density')
            
            return features_df[feature_columns].fillna(0)
            
        except Exception as e:
            print(f"Feature preparation failed: {str(e)}")
            return None
    
    def _calculate_station_density(self, stations_df, radius_km=10):
        """Calculate station density around each station"""
        try:
            from geopy.distance import geodesic
            
            densities = []
            for idx, station in stations_df.iterrows():
                station_coord = (station['latitude'], station['longitude'])
                nearby_count = 0
                
                for idx2, other_station in stations_df.iterrows():
                    if idx != idx2:
                        other_coord = (other_station['latitude'], other_station['longitude'])
                        if geodesic(station_coord, other_coord).kilometers <= radius_km:
                            nearby_count += 1
                
                densities.append(nearby_count)
            
            return densities
        except Exception:
            return [0] * len(stations_df)
    
    def generate_synthetic_training_data(self, stations_df, n_samples=1000):
        """Generate synthetic training data for ML models"""
        try:
            np.random.seed(42)
            
            # Prepare features
            features = self.prepare_features(stations_df, route_info=True, time_info=True)
            if features is None:
                return None, None, None, None
            
            # Generate synthetic congestion levels (0=Low, 1=Medium, 2=High)
            congestion_labels = []
            station_ratings = []
            
            for idx, row in features.iterrows():
                # Congestion logic: higher during rush hours, weekends, popular networks
                congestion_score = 0
                
                # Time-based congestion
                if row.get('is_rush_hour', 0):
                    congestion_score += 0.4
                if row.get('is_weekend', 0):
                    congestion_score += 0.3
                
                # Station popularity affects congestion
                if row.get('network_reliability', 0) > 10:
                    congestion_score += 0.2
                if row.get('has_fast_charging', 0):
                    congestion_score += 0.3
                
                # Station density affects congestion
                if row.get('station_density', 0) < 3:
                    congestion_score += 0.2
                
                # Add some randomness
                congestion_score += np.random.normal(0, 0.1)
                
                # Convert to categorical
                if congestion_score < 0.3:
                    congestion_labels.append(0)  # Low
                elif congestion_score < 0.7:
                    congestion_labels.append(1)  # Medium
                else:
                    congestion_labels.append(2)  # High
                
                # Station rating logic (0-5 scale)
                rating = 3.0  # Base rating
                
                if row.get('has_fast_charging', 0):
                    rating += 0.8
                if row.get('is_public', 0):
                    rating += 0.5
                if row.get('total_ports', 0) > 4:
                    rating += 0.4
                if row.get('network_reliability', 0) > 20:
                    rating += 0.3
                
                # Subtract points for congestion
                rating -= congestion_score * 0.5
                
                # Add randomness and clamp to 1-5 range
                rating += np.random.normal(0, 0.3)
                rating = max(1.0, min(5.0, rating))
                
                station_ratings.append(rating)
            
            return features, np.array(congestion_labels), np.array(station_ratings), stations_df
            
        except Exception as e:
            print(f"Synthetic data generation failed: {str(e)}")
            return None, None, None, None
    
    def train_models(self, stations_df):
        """Train ML models with synthetic data"""
        try:
            # Generate training data
            features, congestion_labels, ratings, _ = self.generate_synthetic_training_data(stations_df)
            
            if features is None:
                return False
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Split data
            X_train, X_test, y_cong_train, y_cong_test = train_test_split(
                features_scaled, congestion_labels, test_size=0.2, random_state=42
            )
            
            _, _, y_rat_train, y_rat_test = train_test_split(
                features_scaled, ratings, test_size=0.2, random_state=42
            )
            
            # Train congestion prediction model
            self.congestion_model.fit(X_train, y_cong_train)
            
            # Train rating prediction model
            self.rating_model.fit(X_train, y_rat_train)
            
            # Evaluate models
            cong_pred = self.congestion_model.predict(X_test)
            rat_pred = self.rating_model.predict(X_test)
            
            # Calculate metrics
            cong_accuracy = np.mean(cong_pred == y_cong_test)
            rat_mse = mean_squared_error(y_rat_test, rat_pred)
            
            self.is_trained = True
            
            # Save models
            self._save_models()
            
            print(f"Models trained successfully! Congestion Accuracy: {cong_accuracy:.3f}, Rating MSE: {rat_mse:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Model training failed: {str(e)}")
            return False
    
    def predict_station_metrics(self, stations_df, route_info=None, time_info=None):
        """Predict congestion and rating for stations"""
        try:
            if not self.is_trained:
                # Try to load saved models
                if not self._load_models():
                    print("Models not trained. Training with available data...")
                    if not self.train_models(stations_df):
                        return None
            
            # Prepare features
            features = self.prepare_features(stations_df, route_info, time_info)
            if features is None:
                return None
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make predictions
            congestion_pred = self.congestion_model.predict(features_scaled)
            congestion_proba = self.congestion_model.predict_proba(features_scaled)
            rating_pred = self.rating_model.predict(features_scaled)
            
            # Add predictions to dataframe
            result_df = stations_df.copy()
            result_df['predicted_congestion'] = congestion_pred
            result_df['congestion_confidence'] = np.max(congestion_proba, axis=1)
            result_df['predicted_rating'] = rating_pred
            
            # Calculate overall station score
            result_df['ml_station_score'] = (
                result_df['predicted_rating'] * 0.6 +  # Rating weight
                (3 - result_df['predicted_congestion']) * 0.4  # Congestion weight (inverted)
            )
            
            return result_df
            
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            return None
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            joblib.dump(self.congestion_model, f"{self.model_path}/congestion_model.pkl")
            joblib.dump(self.rating_model, f"{self.model_path}/rating_model.pkl")
            joblib.dump(self.scaler, f"{self.model_path}/scaler.pkl")
            
            return True
        except Exception as e:
            print(f"Model saving failed: {str(e)}")
            return False
    
    def _load_models(self):
        """Load trained models from disk"""
        try:
            if not os.path.exists(self.model_path):
                return False
            
            self.congestion_model = joblib.load(f"{self.model_path}/congestion_model.pkl")
            self.rating_model = joblib.load(f"{self.model_path}/rating_model.pkl")
            self.scaler = joblib.load(f"{self.model_path}/scaler.pkl")
            
            self.is_trained = True
            return True
        except Exception:
            return False
    
    def get_optimal_stations(self, stations_df, n_stations=10, route_info=None, time_info=None):
        """Get optimal stations using ML predictions"""
        try:
            # Get predictions
            predicted_df = self.predict_station_metrics(stations_df, route_info, time_info)
            if predicted_df is None:
                return stations_df.head(n_stations)
            
            # Sort by ML score and return top N
            optimal_stations = predicted_df.sort_values('ml_station_score', ascending=False).head(n_stations)
            
            return optimal_stations
            
        except Exception as e:
            print(f"Optimal station selection failed: {str(e)}")
            return stations_df.head(n_stations)