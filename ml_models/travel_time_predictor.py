import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
# import streamlit as st
from geopy.distance import geodesic
from datetime import datetime, timedelta
import os
import requests

class TravelTimePredictor:
    """ML model to predict realistic travel times considering traffic patterns"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = "ml_models/saved_models/"
        
    def calculate_base_travel_time(self, coord1, coord2, transport_mode='driving'):
        """Calculate base travel time using distance and speed estimates"""
        try:
            distance_km = geodesic(coord1, coord2).kilometers
            
            # Speed estimates by transport mode and road type
            speed_estimates = {
                'highway': {'driving': 100, 'city': 50},
                'city': {'driving': 40, 'city': 30},
                'rural': {'driving': 80, 'city': 60}
            }
            
            # Estimate road type based on distance
            if distance_km > 100:
                road_type = 'highway'
            elif distance_km > 20:
                road_type = 'rural'
            else:
                road_type = 'city'
            
            # Calculate base time
            base_speed = speed_estimates[road_type]['driving']
            base_time_hours = distance_km / base_speed
            
            return base_time_hours, distance_km, road_type
            
        except Exception as e:
            print(f"Base travel time calculation failed: {str(e)}")
            return None, None, None
    
    def generate_traffic_factors(self, coord1, coord2, departure_time=None):
        """Generate traffic adjustment factors based on time and location"""
        try:
            if departure_time is None:
                departure_time = datetime.now()
            
            # Time-based factors
            hour = departure_time.hour
            day_of_week = departure_time.weekday()
            
            # Traffic multipliers
            traffic_multiplier = 1.0
            
            # Rush hour adjustments
            if (7 <= hour <= 9) or (17 <= hour <= 19):
                traffic_multiplier *= 1.3  # 30% slower during rush hours
            elif (22 <= hour <= 6):
                traffic_multiplier *= 0.9  # 10% faster during night
            
            # Weekend adjustments
            if day_of_week >= 5:  # Weekend
                traffic_multiplier *= 0.95  # 5% faster on weekends
            
            # Urban area density adjustment (simplified)
            distance_km = geodesic(coord1, coord2).kilometers
            if distance_km < 50:  # Urban area
                traffic_multiplier *= 1.15  # 15% slower in urban areas
            
            return traffic_multiplier
            
        except Exception as e:
            print(f"Traffic factor calculation failed: {str(e)}")
            return 1.0
    
    def prepare_features(self, route_segments):
        """Prepare features for travel time prediction"""
        try:
            features = []
            
            for segment in route_segments:
                coord1 = segment['start_coord']
                coord2 = segment['end_coord']
                departure_time = segment.get('departure_time', datetime.now())
                
                # Basic distance and time features
                base_time, distance_km, road_type = self.calculate_base_travel_time(coord1, coord2)
                
                if base_time is None:
                    continue
                
                # Time-based features
                hour = departure_time.hour
                day_of_week = departure_time.weekday()
                is_weekend = int(day_of_week >= 5)
                is_rush_hour = int((7 <= hour <= 9) or (17 <= hour <= 19))
                
                # Geographic features
                lat_diff = abs(coord1[0] - coord2[0])
                lon_diff = abs(coord1[1] - coord2[1])
                
                # Road type encoding
                road_type_highway = int(road_type == 'highway')
                road_type_city = int(road_type == 'city')
                road_type_rural = int(road_type == 'rural')
                
                # Calculate traffic density proxy
                traffic_density = self._estimate_traffic_density(coord1, coord2)
                
                feature_vector = [
                    distance_km,
                    base_time,
                    hour,
                    day_of_week,
                    is_weekend,
                    is_rush_hour,
                    lat_diff,
                    lon_diff,
                    road_type_highway,
                    road_type_city,
                    road_type_rural,
                    traffic_density
                ]
                
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception as e:
            print(f"Feature preparation failed: {str(e)}")
            return None
    
    def _estimate_traffic_density(self, coord1, coord2):
        """Estimate traffic density based on coordinates"""
        try:
            # Simple heuristic: urban areas have higher traffic density
            avg_lat = (coord1[0] + coord2[0]) / 2
            avg_lon = (coord1[1] + coord2[1]) / 2
            
            # Major city centers (simplified)
            major_cities = [
                (40.7128, -74.0060),  # NYC
                (34.0522, -118.2437),  # LA
                (41.8781, -87.6298),  # Chicago
                (37.7749, -122.4194),  # SF
                (42.3601, -71.0589),  # Boston
            ]
            
            min_distance = float('inf')
            for city_coord in major_cities:
                distance = geodesic((avg_lat, avg_lon), city_coord).kilometers
                min_distance = min(min_distance, distance)
            
            # Traffic density decreases with distance from major cities
            if min_distance < 50:
                return 0.8  # High traffic
            elif min_distance < 100:
                return 0.5  # Medium traffic
            else:
                return 0.2  # Low traffic
                
        except Exception:
            return 0.3  # Default medium-low traffic
    
    def generate_synthetic_training_data(self, n_samples=2000):
        """Generate synthetic training data for travel time prediction"""
        try:
            np.random.seed(42)
            
            # Generate random route segments
            route_segments = []
            actual_times = []
            
            for _ in range(n_samples):
                # Random coordinates (continental US)
                start_lat = np.random.uniform(25, 49)
                start_lon = np.random.uniform(-125, -65)
                
                # Random destination within reasonable distance
                distance_km = np.random.exponential(50)  # Most trips are short
                distance_km = min(distance_km, 500)  # Cap at 500km
                
                # Random direction
                bearing = np.random.uniform(0, 360)
                
                # Calculate end coordinates (simplified)
                end_lat = start_lat + (distance_km / 111) * np.cos(np.radians(bearing))
                end_lon = start_lon + (distance_km / 111) * np.sin(np.radians(bearing))
                
                # Random departure time
                departure_time = datetime.now() + timedelta(
                    hours=np.random.uniform(-24, 24),
                    minutes=np.random.uniform(0, 60)
                )
                
                route_segments.append({
                    'start_coord': (start_lat, start_lon),
                    'end_coord': (end_lat, end_lon),
                    'departure_time': departure_time
                })
                
                # Generate realistic travel time
                base_time, _, road_type = self.calculate_base_travel_time(
                    (start_lat, start_lon), (end_lat, end_lon)
                )
                
                if base_time is None:
                    continue
                
                # Apply traffic factors
                traffic_multiplier = self.generate_traffic_factors(
                    (start_lat, start_lon), (end_lat, end_lon), departure_time
                )
                
                # Add some randomness for realistic variation
                noise_factor = np.random.normal(1.0, 0.1)
                actual_time = base_time * traffic_multiplier * noise_factor
                
                # Ensure positive time
                actual_time = max(actual_time, 0.1)
                
                actual_times.append(actual_time)
            
            # Prepare features
            features = self.prepare_features(route_segments)
            
            if features is None or len(features) == 0:
                return None, None
            
            return features, np.array(actual_times[:len(features)])
            
        except Exception as e:
            print(f"Synthetic data generation failed: {str(e)}")
            return None, None
    
    def train_model(self):
        """Train the travel time prediction model"""
        try:
            # Generate training data
            features, actual_times = self.generate_synthetic_training_data()
            
            if features is None:
                return False
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, actual_times, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            self.is_trained = True
            
            # Save model
            self._save_model()
            
            print(f"Travel time model trained! Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Model training failed: {str(e)}")
            return False
    
    def predict_travel_time(self, coord1, coord2, departure_time=None):
        """Predict travel time between two coordinates"""
        try:
            if not self.is_trained:
                if not self._load_model():
                    print("Training travel time model...")
                    if not self.train_model():
                        # Fallback to simple calculation
                        base_time, _, _ = self.calculate_base_travel_time(coord1, coord2)
                        if base_time:
                            traffic_multiplier = self.generate_traffic_factors(coord1, coord2, departure_time)
                            return base_time * traffic_multiplier
                        return None
            
            # Prepare features for prediction
            route_segments = [{
                'start_coord': coord1,
                'end_coord': coord2,
                'departure_time': departure_time or datetime.now()
            }]
            
            features = self.prepare_features(route_segments)
            if features is None or len(features) == 0:
                return None
            
            # Scale features and predict
            features_scaled = self.scaler.transform(features)
            predicted_time = self.model.predict(features_scaled)[0]
            
            return max(predicted_time, 0.1)  # Ensure positive time
            
        except Exception as e:
            print(f"Travel time prediction failed: {str(e)}")
            # Fallback to simple calculation
            base_time, _, _ = self.calculate_base_travel_time(coord1, coord2)
            if base_time:
                traffic_multiplier = self.generate_traffic_factors(coord1, coord2, departure_time)
                return base_time * traffic_multiplier
            return None
    
    def predict_route_times(self, waypoints, departure_time=None):
        """Predict travel times for a complete route"""
        try:
            if len(waypoints) < 2:
                return []
            
            current_time = departure_time or datetime.now()
            segment_times = []
            
            for i in range(len(waypoints) - 1):
                # Predict time for this segment
                segment_time = self.predict_travel_time(
                    waypoints[i], waypoints[i + 1], current_time
                )
                
                if segment_time is None:
                    segment_time = 1.0  # Default 1 hour
                
                segment_times.append(segment_time)
                
                # Update current time for next segment
                current_time += timedelta(hours=segment_time)
            
            return segment_times
            
        except Exception as e:
            print(f"Route time prediction failed: {str(e)}")
            return []
    
    def _save_model(self):
        """Save trained model to disk"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            joblib.dump(self.model, f"{self.model_path}/travel_time_model.pkl")
            joblib.dump(self.scaler, f"{self.model_path}/travel_time_scaler.pkl")
            
            return True
        except Exception as e:
            print(f"Model saving failed: {str(e)}")
            return False
    
    def _load_model(self):
        """Load trained model from disk"""
        try:
            if not os.path.exists(self.model_path):
                return False
            
            self.model = joblib.load(f"{self.model_path}/travel_time_model.pkl")
            self.scaler = joblib.load(f"{self.model_path}/travel_time_scaler.pkl")
            
            self.is_trained = True
            return True
        except Exception:
            return False