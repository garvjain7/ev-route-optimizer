import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
# import streamlit as st
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict

class AdaptiveRouter:
    """Adaptive ML system that learns from route feedback and optimizes future routes"""
    
    def __init__(self):
        self.efficiency_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = "ml_models/saved_models/"
        self.feedback_path = "ml_models/route_feedback.json"
        self.route_history = []
        self.station_performance = defaultdict(list)
        
    def load_route_feedback(self):
        """Load historical route feedback"""
        try:
            if os.path.exists(self.feedback_path):
                with open(self.feedback_path, 'r') as f:
                    data = json.load(f)
                    self.route_history = data.get('route_history', [])
                    self.station_performance = defaultdict(list, data.get('station_performance', {}))
                return True
            return False
        except Exception as e:
            print(f"Failed to load route feedback: {str(e)}")
            return False
    
    def save_route_feedback(self):
        """Save route feedback to disk"""
        try:
            os.makedirs(os.path.dirname(self.feedback_path), exist_ok=True)
            
            data = {
                'route_history': self.route_history,
                'station_performance': dict(self.station_performance)
            }
            
            with open(self.feedback_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Failed to save route feedback: {str(e)}")
            return False
    
    def add_route_feedback(self, route_data, user_feedback):
        """Add feedback for a completed route"""
        try:
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'route_id': route_data.get('route_id', len(self.route_history)),
                'waypoints': route_data.get('waypoints', []),
                'charging_stops': route_data.get('charging_stops', []),
                'statistics': route_data.get('statistics', {}),
                'user_feedback': user_feedback
            }
            
            self.route_history.append(feedback_entry)
            
            # Update station performance
            for stop in route_data.get('charging_stops', []):
                station_id = stop.get('station_id', stop.get('station_name', ''))
                if station_id:
                    self.station_performance[station_id].append({
                        'timestamp': datetime.now().isoformat(),
                        'actual_charging_time': user_feedback.get('actual_charging_time', stop.get('charging_time', 30)),
                        'station_availability': user_feedback.get('station_availability', True),
                        'charging_speed': user_feedback.get('charging_speed', 'normal'),
                        'overall_rating': user_feedback.get('station_rating', 3)
                    })
            
            self.save_route_feedback()
            
            return True
        except Exception as e:
            print(f"Failed to add route feedback: {str(e)}")
            return False
    
    def prepare_route_features(self, route_data):
        """Prepare features from route data for ML model"""
        try:
            features = []
            
            stats = route_data.get('statistics', {})
            
            # Basic route features
            total_distance = stats.get('total_distance', 0)
            charging_stops = stats.get('charging_stops', 0)
            total_time = stats.get('total_time', 0)
            
            # Calculate route complexity
            complexity_score = 0
            if charging_stops > 0:
                complexity_score = charging_stops / (total_distance / 100)  # Stops per 100km
            
            # Time-based features
            if 'waypoints' in route_data and len(route_data['waypoints']) > 0:
                # Assuming first waypoint is departure
                departure_time = datetime.now()  # Simplified
                hour = departure_time.hour
                day_of_week = departure_time.weekday()
                is_weekend = int(day_of_week >= 5)
                is_rush_hour = int((7 <= hour <= 9) or (17 <= hour <= 19))
            else:
                hour = day_of_week = is_weekend = is_rush_hour = 0
            
            # Station quality features
            avg_station_rating = self._calculate_avg_station_rating(route_data)
            
            # Route efficiency features
            direct_distance = self._calculate_direct_distance(route_data)
            route_efficiency = direct_distance / total_distance if total_distance > 0 else 0
            
            features = [
                total_distance,
                charging_stops,
                total_time,
                complexity_score,
                hour,
                day_of_week,
                is_weekend,
                is_rush_hour,
                avg_station_rating,
                route_efficiency
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Route feature preparation failed: {str(e)}")
            return None
    
    def _calculate_avg_station_rating(self, route_data):
        """Calculate average rating of stations in the route"""
        try:
            ratings = []
            for stop in route_data.get('charging_stops', []):
                station_id = stop.get('station_id', stop.get('station_name', ''))
                if station_id in self.station_performance:
                    station_ratings = [
                        perf['overall_rating'] for perf in self.station_performance[station_id]
                    ]
                    if station_ratings:
                        ratings.append(np.mean(station_ratings))
                    else:
                        ratings.append(3.0)  # Default rating
                else:
                    ratings.append(3.0)  # Default rating
            
            return np.mean(ratings) if ratings else 3.0
        except Exception:
            return 3.0
    
    def _calculate_direct_distance(self, route_data):
        """Calculate direct distance from source to destination"""
        try:
            from geopy.distance import geodesic
            
            waypoints = route_data.get('waypoints', [])
            if len(waypoints) >= 2:
                start = waypoints[0]['coordinates']
                end = waypoints[-1]['coordinates']
                return geodesic(start, end).kilometers
            return 0
        except Exception:
            return 0
    
    def train_efficiency_model(self):
        """Train model to predict route efficiency"""
        try:
            if len(self.route_history) < 10:
                # Generate synthetic data for initial training
                return self._train_with_synthetic_data()
            
            # Prepare training data from historical routes
            X_features = []
            y_efficiency = []
            
            for route in self.route_history:
                features = self.prepare_route_features(route)
                if features is not None:
                    X_features.append(features.flatten())
                    
                    # Calculate efficiency score from user feedback
                    user_feedback = route.get('user_feedback', {})
                    efficiency = self._calculate_efficiency_score(route, user_feedback)
                    y_efficiency.append(efficiency)
            
            if len(X_features) == 0:
                return self._train_with_synthetic_data()
            
            X = np.array(X_features)
            y = np.array(y_efficiency)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.efficiency_model.fit(X_scaled, y)
            
            # Calculate training score
            train_score = self.efficiency_model.score(X_scaled, y)
            
            self.is_trained = True
            self._save_model()
            
            print(f"Adaptive efficiency model trained! R²: {train_score:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Efficiency model training failed: {str(e)}")
            return False
    
    def _train_with_synthetic_data(self):
        """Train model with synthetic data when insufficient historical data"""
        try:
            np.random.seed(42)
            n_samples = 500
            
            # Generate synthetic route features
            X_synthetic = []
            y_synthetic = []
            
            for _ in range(n_samples):
                # Random route parameters
                distance = np.random.exponential(200)  # Average 200km
                distance = min(distance, 800)  # Cap at 800km
                
                stops = int(distance / 250) + np.random.poisson(1)  # Stops based on distance
                stops = max(0, min(stops, 6))  # 0-6 stops
                
                time = distance / 60 + stops * 0.5  # Base time + charging time
                time *= np.random.uniform(0.8, 1.3)  # Add variance
                
                complexity = stops / (distance / 100) if distance > 0 else 0
                
                # Random time features
                hour = np.random.randint(0, 24)
                day_of_week = np.random.randint(0, 7)
                is_weekend = int(day_of_week >= 5)
                is_rush_hour = int((7 <= hour <= 9) or (17 <= hour <= 19))
                
                # Random station rating
                station_rating = np.random.normal(3.5, 0.8)
                station_rating = max(1, min(5, station_rating))
                
                # Random route efficiency
                route_efficiency = np.random.beta(2, 3)  # Skewed towards efficient routes
                
                features = [
                    distance, stops, time, complexity, hour, day_of_week,
                    is_weekend, is_rush_hour, station_rating, route_efficiency
                ]
                
                # Calculate synthetic efficiency score
                efficiency = (
                    route_efficiency * 0.3 +
                    (station_rating / 5) * 0.3 +
                    (1 - complexity / 0.1) * 0.2 +  # Lower complexity is better
                    (1 - (time / distance * 60)) * 0.2  # Faster relative time is better
                )
                
                # Add noise
                efficiency += np.random.normal(0, 0.1)
                efficiency = max(0, min(1, efficiency))
                
                X_synthetic.append(features)
                y_synthetic.append(efficiency)
            
            X = np.array(X_synthetic)
            y = np.array(y_synthetic)
            
            # Scale and train
            X_scaled = self.scaler.fit_transform(X)
            self.efficiency_model.fit(X_scaled, y)
            
            train_score = self.efficiency_model.score(X_scaled, y)
            
            self.is_trained = True
            self._save_model()
            
            print(f"Synthetic efficiency model trained! R²: {train_score:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Synthetic training failed: {str(e)}")
            return False
    
    def _calculate_efficiency_score(self, route, user_feedback):
        """Calculate efficiency score based on route data and user feedback"""
        try:
            # Base efficiency from route statistics
            stats = route.get('statistics', {})
            total_distance = stats.get('total_distance', 0)
            total_time = stats.get('total_time', 0)
            
            # Time efficiency
            if total_distance > 0:
                time_efficiency = (total_distance / 80) / total_time  # Compared to 80km/h
            else:
                time_efficiency = 0.5
            
            # User satisfaction
            user_satisfaction = user_feedback.get('overall_satisfaction', 3) / 5
            
            # Station performance
            station_issues = user_feedback.get('station_issues', 0)
            station_performance = max(0, 1 - station_issues / 10)
            
            # Charging efficiency
            planned_charging_time = sum(
                stop.get('charging_time', 30) for stop in route.get('charging_stops', [])
            )
            actual_charging_time = user_feedback.get('total_charging_time', planned_charging_time)
            
            charging_efficiency = planned_charging_time / actual_charging_time if actual_charging_time > 0 else 0.5
            
            # Combined efficiency score
            efficiency = (
                time_efficiency * 0.3 +
                user_satisfaction * 0.3 +
                station_performance * 0.2 +
                charging_efficiency * 0.2
            )
            
            return max(0, min(1, efficiency))
            
        except Exception:
            return 0.5  # Default neutral efficiency
    
    def predict_route_efficiency(self, route_data):
        """Predict efficiency score for a route"""
        try:
            if not self.is_trained:
                if not self._load_model():
                    if not self.train_efficiency_model():
                        return 0.5  # Default score
            
            features = self.prepare_route_features(route_data)
            if features is None:
                return 0.5
            
            # Scale features and predict
            features_scaled = self.scaler.transform(features)
            efficiency_score = self.efficiency_model.predict(features_scaled)[0]
            
            return max(0, min(1, efficiency_score))
            
        except Exception as e:
            print(f"Route efficiency prediction failed: {str(e)}")
            return 0.5
    
    def get_adaptive_recommendations(self, route_data):
        """Get adaptive recommendations based on historical data"""
        try:
            recommendations = []
            
            # Predict efficiency
            efficiency_score = self.predict_route_efficiency(route_data)
            
            if efficiency_score < 0.6:
                recommendations.append({
                    'type': 'efficiency_warning',
                    'message': 'This route may be less efficient than optimal alternatives',
                    'confidence': 1 - efficiency_score
                })
            
            # Station recommendations based on performance
            for stop in route_data.get('charging_stops', []):
                station_id = stop.get('station_id', stop.get('station_name', ''))
                if station_id in self.station_performance:
                    station_data = self.station_performance[station_id]
                    
                    # Calculate recent performance
                    recent_ratings = [
                        perf['overall_rating'] for perf in station_data[-10:]  # Last 10 visits
                    ]
                    
                    if recent_ratings:
                        avg_rating = np.mean(recent_ratings)
                        if avg_rating < 3.0:
                            recommendations.append({
                                'type': 'station_warning',
                                'message': f'Station {station_id} has low recent ratings ({avg_rating:.1f}/5)',
                                'station_id': station_id,
                                'confidence': 1 - (avg_rating / 5)
                            })
            
            # Time-based recommendations
            current_time = datetime.now()
            if 17 <= current_time.hour <= 19:
                recommendations.append({
                    'type': 'time_warning',
                    'message': 'Traveling during rush hour may increase charging wait times',
                    'confidence': 0.7
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Adaptive recommendations failed: {str(e)}")
            return []
    
    def _save_model(self):
        """Save trained model to disk"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            joblib.dump(self.efficiency_model, f"{self.model_path}/efficiency_model.pkl")
            joblib.dump(self.scaler, f"{self.model_path}/efficiency_scaler.pkl")
            
            return True
        except Exception as e:
            print(f"Model saving failed: {str(e)}")
            return False
    
    def _load_model(self):
        """Load trained model from disk"""
        try:
            if not os.path.exists(self.model_path):
                return False
            
            self.efficiency_model = joblib.load(f"{self.model_path}/efficiency_model.pkl")
            self.scaler = joblib.load(f"{self.model_path}/efficiency_scaler.pkl")
            
            self.is_trained = True
            return True
        except Exception:
            return False
    
    def get_performance_analytics(self):
        """Get analytics on route performance"""
        try:
            if not self.route_history:
                return None
            
            # Calculate overall statistics
            total_routes = len(self.route_history)
            
            # Efficiency trends
            efficiency_scores = []
            for route in self.route_history:
                user_feedback = route.get('user_feedback', {})
                efficiency = self._calculate_efficiency_score(route, user_feedback)
                efficiency_scores.append(efficiency)
            
            # Station performance
            station_stats = {}
            for station_id, performance_data in self.station_performance.items():
                ratings = [p['overall_rating'] for p in performance_data]
                station_stats[station_id] = {
                    'avg_rating': np.mean(ratings),
                    'total_visits': len(ratings),
                    'recent_rating': np.mean(ratings[-5:]) if len(ratings) >= 5 else np.mean(ratings)
                }
            
            return {
                'total_routes': total_routes,
                'avg_efficiency': np.mean(efficiency_scores),
                'efficiency_trend': efficiency_scores[-10:],  # Last 10 routes
                'station_stats': station_stats,
                'best_stations': sorted(station_stats.items(), key=lambda x: x[1]['avg_rating'], reverse=True)[:5],
                'worst_stations': sorted(station_stats.items(), key=lambda x: x[1]['avg_rating'])[:5]
            }
            
        except Exception as e:
            print(f"Performance analytics failed: {str(e)}")
            return None