import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from database.db_connection import DatabaseConnection
from filtering.station_filters import StationFilters
from clustering.clustering_engine import ClusteringEngine
from routing.ev_router import EVRouter
from visualization.map_visualizer import MapVisualizer
from utils.geojson_exporter import GeoJSONExporter
import json

# Page configuration
st.set_page_config(
    page_title="EV Routing & Charging Station Optimization",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'stations_data' not in st.session_state:
    st.session_state.stations_data = None
if 'filtered_stations' not in st.session_state:
    st.session_state.filtered_stations = None
if 'route_data' not in st.session_state:
    st.session_state.route_data = None

def main():
    st.title("🚗🔋 EV Routing & Charging Station Optimization Pipeline")
    st.markdown("A comprehensive pipeline for optimizing EV routes with intelligent charging station selection")
    
    # Initialize components
    db_conn = DatabaseConnection()
    station_filters = StationFilters()
    clustering_engine = ClusteringEngine()
    ev_router = EVRouter()
    map_visualizer = MapVisualizer()
    geojson_exporter = GeoJSONExporter()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Database connection section
    st.sidebar.subheader("📊 Database Connection")
    if st.sidebar.button("Connect to Database"):
        with st.spinner("Connecting to PostgreSQL database..."):
            try:
                st.session_state.stations_data = db_conn.load_ev_stations()
                if st.session_state.stations_data is not None:
                    st.sidebar.success(f"✅ Loaded {len(st.session_state.stations_data)} charging stations")
                else:
                    st.sidebar.error("❌ Failed to load stations data")
            except Exception as e:
                st.sidebar.error(f"❌ Database connection failed: {str(e)}")
    
    # Route input section
    st.sidebar.subheader("🗺️ Route Configuration")
    
    # Source and destination inputs
    col1, col2 = st.sidebar.columns(2)
    with col1:
        source_lat = st.number_input("Source Latitude", value=40.7128, format="%.6f")
        source_lon = st.number_input("Source Longitude", value=-74.0060, format="%.6f")
    
    with col2:
        dest_lat = st.number_input("Destination Latitude", value=42.3601, format="%.6f")
        dest_lon = st.number_input("Destination Longitude", value=-71.0589, format="%.6f")
    
    source_coords = (source_lat, source_lon)
    dest_coords = (dest_lat, dest_lon)
    
    # EV specifications
    st.sidebar.subheader("🔋 EV Specifications")
    battery_range = st.sidebar.slider("Battery Range (km)", 100, 600, 300)
    consumption_rate = st.sidebar.slider("Energy Consumption (kWh/100km)", 10, 30, 20)
    charging_time = st.sidebar.slider("Charging Time (minutes)", 15, 60, 30)
    safety_margin = st.sidebar.slider("Safety Margin (%)", 5, 30, 15)
    
    # Filtering parameters
    st.sidebar.subheader("🔍 Filtering Parameters")
    detour_factor = st.sidebar.slider("Detour Factor", 1.1, 3.0, 1.5)
    corridor_width = st.sidebar.slider("Corridor Width (km)", 5, 100, 25)
    max_distance_source = st.sidebar.slider("Max Distance from Source (km)", 50, 500, 200)
    max_distance_dest = st.sidebar.slider("Max Distance from Destination (km)", 50, 500, 200)
    
    # Clustering parameters
    st.sidebar.subheader("🎯 Clustering Parameters")
    n_clusters = st.sidebar.slider("Number of Clusters", 3, 20, 8)
    
    # ML preferences
    st.sidebar.subheader("🤖 ML Preferences")
    prefer_fast_charging = st.sidebar.checkbox("Prefer Fast Charging", value=True)
    avoid_congestion = st.sidebar.checkbox("Avoid Congested Stations", value=True)
    min_station_rating = st.sidebar.slider("Min Station Rating", 1.0, 5.0, 3.0)
    
    # Station type preferences
    st.sidebar.subheader("⚡ Station Type Preferences")
    charging_types = st.sidebar.multiselect(
        "Charging Types", 
        ["AC", "DC"], 
        default=["AC", "DC"]
    )
    power_levels = st.sidebar.multiselect(
        "Power Levels", 
        ["Level1", "Level2", "DC_Fast"], 
        default=["Level2", "DC_Fast"]
    )
    
    # Main content area
    if st.session_state.stations_data is not None:
        # Display database statistics
        st.subheader("📊 Database Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stations", len(st.session_state.stations_data))
        
        with col2:
            unique_states = st.session_state.stations_data['state'].nunique() if 'state' in st.session_state.stations_data.columns else 0
            st.metric("States Covered", unique_states)
        
        with col3:
            route_distance = geodesic(source_coords, dest_coords).kilometers
            st.metric("Direct Distance", f"{route_distance:.1f} km")
        
        with col4:
            estimated_charging_stops = max(0, int(route_distance / (battery_range * (1 - safety_margin/100))) - 1)
            st.metric("Est. Charging Stops", estimated_charging_stops)
        
        # Filtering section
        st.subheader("🔍 Station Filtering")
        
        filtering_method = st.selectbox(
            "Select Filtering Method",
            ["Smart ML Filtering", "ML-Based Filtering", "Station Type Filtering", 
             "Geographic Bounding Box", "Distance-Based", "Corridor-Based", "Combined Approach"]
        )
        
        if st.button("Apply Filtering"):
            with st.spinner("Filtering charging stations..."):
                try:
                    if filtering_method == "Smart ML Filtering":
                        user_preferences = {
                            'prefer_fast_charging': prefer_fast_charging,
                            'max_detour_factor': detour_factor,
                            'corridor_width': corridor_width,
                            'min_station_rating': min_station_rating,
                            'avoid_congestion': avoid_congestion,
                            'charging_types': charging_types,
                            'power_levels': power_levels
                        }
                        st.session_state.filtered_stations = station_filters.smart_filtering(
                            st.session_state.stations_data, source_coords, dest_coords, 
                            user_preferences=user_preferences
                        )
                    elif filtering_method == "ML-Based Filtering":
                        filter_criteria = {
                            'min_rating': min_station_rating,
                            'max_congestion': 1 if avoid_congestion else 2,
                            'prefer_fast_charging': prefer_fast_charging,
                            'min_station_score': min_station_rating
                        }
                        st.session_state.filtered_stations = station_filters.ml_based_filtering(
                            st.session_state.stations_data, source_coords, dest_coords, 
                            filter_criteria
                        )
                    elif filtering_method == "Station Type Filtering":
                        st.session_state.filtered_stations = station_filters.station_type_filtering(
                            st.session_state.stations_data, charging_types, power_levels
                        )
                    elif filtering_method == "Geographic Bounding Box":
                        st.session_state.filtered_stations = station_filters.geographic_bounding_box(
                            st.session_state.stations_data, source_coords, dest_coords
                        )
                    elif filtering_method == "Distance-Based":
                        st.session_state.filtered_stations = station_filters.distance_based_filtering(
                            st.session_state.stations_data, source_coords, dest_coords, 
                            detour_factor, max_distance_source, max_distance_dest
                        )
                    elif filtering_method == "Corridor-Based":
                        st.session_state.filtered_stations = station_filters.corridor_based_filtering(
                            st.session_state.stations_data, source_coords, dest_coords, corridor_width
                        )
                    else:  # Combined Approach
                        st.session_state.filtered_stations = station_filters.combined_filtering(
                            st.session_state.stations_data, source_coords, dest_coords,
                            detour_factor, corridor_width, max_distance_source, max_distance_dest
                        )
                    
                    if st.session_state.filtered_stations is not None:
                        st.success(f"✅ Filtered to {len(st.session_state.filtered_stations)} stations")
                        
                        # Show ML predictions if available
                        if ('predicted_congestion' in st.session_state.filtered_stations.columns or 
                            'predicted_rating' in st.session_state.filtered_stations.columns):
                            st.subheader("🤖 ML Predictions Summary")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if 'predicted_congestion' in st.session_state.filtered_stations.columns:
                                    avg_congestion = st.session_state.filtered_stations['predicted_congestion'].mean()
                                    congestion_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
                                    st.metric("Avg Congestion", congestion_labels.get(round(avg_congestion), 'Unknown'))
                            
                            with col2:
                                if 'predicted_rating' in st.session_state.filtered_stations.columns:
                                    avg_rating = st.session_state.filtered_stations['predicted_rating'].mean()
                                    st.metric("Avg Rating", f"{avg_rating:.1f}/5")
                            
                            with col3:
                                if 'ml_station_score' in st.session_state.filtered_stations.columns:
                                    avg_score = st.session_state.filtered_stations['ml_station_score'].mean()
                                    st.metric("Avg ML Score", f"{avg_score:.1f}/5")
                    else:
                        st.error("❌ No stations found with current filters")
                except Exception as e:
                    st.error(f"❌ Filtering failed: {str(e)}")
        
        # Clustering and route optimization
        if st.session_state.filtered_stations is not None:
            st.subheader("🎯 Clustering & Route Optimization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Perform Clustering"):
                    with st.spinner("Clustering stations..."):
                        try:
                            clustered_stations = clustering_engine.cluster_stations(
                                st.session_state.filtered_stations, n_clusters
                            )
                            st.session_state.clustered_stations = clustered_stations
                            st.success(f"✅ Created {n_clusters} clusters")
                        except Exception as e:
                            st.error(f"❌ Clustering failed: {str(e)}")
            
            with col2:
                if st.button("Optimize Route"):
                    with st.spinner("Optimizing EV route..."):
                        try:
                            ev_specs = {
                                'battery_range': battery_range,
                                'consumption_rate': consumption_rate,
                                'charging_time': charging_time,
                                'safety_margin': safety_margin
                            }
                            
                            route_result = ev_router.optimize_route(
                                source_coords, dest_coords, 
                                st.session_state.filtered_stations, 
                                ev_specs
                            )
                            
                            st.session_state.route_data = route_result
                            st.success("✅ Route optimized successfully")
                        except Exception as e:
                            st.error(f"❌ Route optimization failed: {str(e)}")
            
            # Visualization section
            st.subheader("🗺️ Interactive Visualization")
            
            # Create and display map
            map_obj = map_visualizer.create_route_map(
                source_coords, dest_coords,
                st.session_state.filtered_stations,
                st.session_state.route_data
            )
            
            st_folium(map_obj, width=1200, height=600)
            
            # Route analysis
            if st.session_state.route_data is not None:
                st.subheader("📈 Route Analysis")
                
                # Display route statistics
                route_stats = st.session_state.route_data.get('statistics', {})
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Distance", f"{route_stats.get('total_distance', 0):.1f} km")
                
                with col2:
                    st.metric("Charging Stops", route_stats.get('charging_stops', 0))
                
                with col3:
                    st.metric("Total Time", f"{route_stats.get('total_time', 0):.1f} hours")
                
                with col4:
                    st.metric("Energy Efficiency", f"{route_stats.get('efficiency', 0):.1f} kWh/100km")
                
                # ML predictions and recommendations
                if 'ml_predictions' in st.session_state.route_data:
                    ml_data = st.session_state.route_data['ml_predictions']
                    
                    if ml_data.get('efficiency_score') is not None:
                        st.subheader("🤖 ML Route Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            efficiency_score = ml_data['efficiency_score']
                            st.metric("Predicted Efficiency", f"{efficiency_score:.1%}")
                            
                            # Efficiency indicator
                            if efficiency_score >= 0.8:
                                st.success("🟢 Highly efficient route")
                            elif efficiency_score >= 0.6:
                                st.warning("🟡 Moderately efficient route")
                            else:
                                st.error("🔴 Low efficiency route")
                        
                        with col2:
                            recommendations = ml_data.get('recommendations', [])
                            if recommendations:
                                st.subheader("💡 Recommendations")
                                for rec in recommendations[:3]:  # Show top 3
                                    if rec['type'] == 'efficiency_warning':
                                        st.warning(f"⚠️ {rec['message']}")
                                    elif rec['type'] == 'station_warning':
                                        st.warning(f"🔌 {rec['message']}")
                                    elif rec['type'] == 'time_warning':
                                        st.info(f"⏰ {rec['message']}")
                            else:
                                st.info("No specific recommendations for this route")
                
                # Charging stops details
                if 'charging_stops' in st.session_state.route_data:
                    st.subheader("🔌 Charging Stops Details")
                    
                    charging_df = pd.DataFrame(st.session_state.route_data['charging_stops'])
                    if not charging_df.empty:
                        # Add ML predictions to charging stops if available
                        if 'predicted_congestion' in st.session_state.filtered_stations.columns:
                            # Try to match stations with predictions
                            for idx, stop in charging_df.iterrows():
                                station_name = stop.get('station_name', '')
                                matching_stations = st.session_state.filtered_stations[
                                    st.session_state.filtered_stations['name'].str.contains(station_name, case=False, na=False)
                                ]
                                if not matching_stations.empty:
                                    station_data = matching_stations.iloc[0]
                                    charging_df.at[idx, 'predicted_congestion'] = station_data.get('predicted_congestion', 'Unknown')
                                    charging_df.at[idx, 'predicted_rating'] = station_data.get('predicted_rating', 'Unknown')
                        
                        st.dataframe(charging_df, use_container_width=True)
                
                # Energy consumption chart
                if 'energy_profile' in st.session_state.route_data:
                    st.subheader("⚡ Energy Consumption Profile")
                    
                    energy_data = st.session_state.route_data['energy_profile']
                    fig = px.line(
                        x=energy_data.get('distance', []),
                        y=energy_data.get('battery_level', []),
                        title="Battery Level vs Distance",
                        labels={'x': 'Distance (km)', 'y': 'Battery Level (%)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            st.subheader("📁 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export Filtered Stations (GeoJSON)"):
                    try:
                        geojson_data = geojson_exporter.export_stations(st.session_state.filtered_stations)
                        st.download_button(
                            label="Download GeoJSON",
                            data=json.dumps(geojson_data, indent=2),
                            file_name="filtered_stations.geojson",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"❌ Export failed: {str(e)}")
            
            with col2:
                if st.session_state.route_data is not None:
                    if st.button("Export Route Data (JSON)"):
                        try:
                            route_json = json.dumps(st.session_state.route_data, indent=2, default=str)
                            st.download_button(
                                label="Download Route Data",
                                data=route_json,
                                file_name="route_data.json",
                                mime="application/json"
                            )
                        except Exception as e:
                            st.error(f"❌ Export failed: {str(e)}")
            
            # ML Model Training Section
            st.subheader("🤖 ML Model Management")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Train Station Predictor"):
                    if st.session_state.stations_data is not None:
                        with st.spinner("Training ML models..."):
                            try:
                                from ml_models.station_predictor import StationPredictor
                                predictor = StationPredictor()
                                success = predictor.train_models(st.session_state.stations_data)
                                if success:
                                    st.success("✅ Station predictor trained successfully!")
                                else:
                                    st.error("❌ Station predictor training failed")
                            except Exception as e:
                                st.error(f"❌ Training failed: {str(e)}")
                    else:
                        st.warning("⚠️ Load station data first")
            
            with col2:
                if st.button("Train Travel Time Predictor"):
                    with st.spinner("Training travel time model..."):
                        try:
                            from ml_models.travel_time_predictor import TravelTimePredictor
                            predictor = TravelTimePredictor()
                            success = predictor.train_model()
                            if success:
                                st.success("✅ Travel time predictor trained successfully!")
                            else:
                                st.error("❌ Travel time predictor training failed")
                        except Exception as e:
                            st.error(f"❌ Training failed: {str(e)}")
            
            with col3:
                if st.button("Train Adaptive Router"):
                    with st.spinner("Training adaptive router..."):
                        try:
                            from ml_models.adaptive_router import AdaptiveRouter
                            router = AdaptiveRouter()
                            success = router.train_efficiency_model()
                            if success:
                                st.success("✅ Adaptive router trained successfully!")
                            else:
                                st.error("❌ Adaptive router training failed")
                        except Exception as e:
                            st.error(f"❌ Training failed: {str(e)}")
            
            # Route Feedback Section
            if st.session_state.route_data is not None:
                st.subheader("📝 Route Feedback")
                
                with st.expander("Provide Route Feedback"):
                    st.write("Help improve the ML models by providing feedback on your route experience:")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        overall_satisfaction = st.slider("Overall Satisfaction", 1, 5, 3)
                        actual_charging_time = st.number_input("Total Charging Time (minutes)", min_value=0, value=30)
                        station_issues = st.number_input("Number of Station Issues", min_value=0, value=0)
                    
                    with col2:
                        charging_speed = st.selectbox("Charging Speed", ["slow", "normal", "fast"])
                        station_availability = st.checkbox("All Stations Available", value=True)
                        would_recommend = st.checkbox("Would Recommend Route", value=True)
                    
                    if st.button("Submit Feedback"):
                        feedback = {
                            'overall_satisfaction': overall_satisfaction,
                            'actual_charging_time': actual_charging_time,
                            'total_charging_time': actual_charging_time,
                            'station_issues': station_issues,
                            'charging_speed': charging_speed,
                            'station_availability': station_availability,
                            'would_recommend': would_recommend,
                            'station_rating': overall_satisfaction  # Use overall satisfaction as station rating
                        }
                        
                        try:
                            from ml_models.adaptive_router import AdaptiveRouter
                            router = AdaptiveRouter()
                            router.load_route_feedback()
                            success = router.add_route_feedback(st.session_state.route_data, feedback)
                            
                            if success:
                                st.success("✅ Thank you for your feedback! This will help improve future route recommendations.")
                            else:
                                st.error("❌ Failed to save feedback")
                        except Exception as e:
                            st.error(f"❌ Feedback submission failed: {str(e)}")
            
            # Performance Analytics
            st.subheader("📊 Performance Analytics")
            
            if st.button("View Performance Analytics"):
                try:
                    from ml_models.adaptive_router import AdaptiveRouter
                    router = AdaptiveRouter()
                    router.load_route_feedback()
                    analytics = router.get_performance_analytics()
                    
                    if analytics:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Routes", analytics['total_routes'])
                        
                        with col2:
                            st.metric("Avg Efficiency", f"{analytics['avg_efficiency']:.1%}")
                        
                        with col3:
                            if analytics['best_stations']:
                                best_station = analytics['best_stations'][0]
                                st.metric("Best Station", f"{best_station[0][:20]}...")
                        
                        # Efficiency trend
                        if analytics['efficiency_trend']:
                            st.subheader("Efficiency Trend")
                            import plotly.graph_objects as go
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=analytics['efficiency_trend'],
                                mode='lines+markers',
                                name='Efficiency',
                                line=dict(color='blue')
                            ))
                            fig.update_layout(
                                title="Route Efficiency Over Time",
                                xaxis_title="Route Number",
                                yaxis_title="Efficiency Score"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Station performance
                        if analytics['station_stats']:
                            st.subheader("Station Performance")
                            station_df = pd.DataFrame([
                                {'Station': k, 'Avg Rating': v['avg_rating'], 'Visits': v['total_visits']}
                                for k, v in analytics['station_stats'].items()
                            ])
                            st.dataframe(station_df.sort_values('Avg Rating', ascending=False), use_container_width=True)
                    else:
                        st.info("No performance data available yet. Complete some routes and provide feedback to see analytics.")
                        
                except Exception as e:
                    st.error(f"❌ Analytics failed: {str(e)}")
    
    else:
        st.info("👆 Please connect to the database to begin the optimization process")
        
        # Show sample coordinates for testing
        st.subheader("🗺️ Sample Coordinates")
        st.markdown("""
        **New York to Boston:**
        - Source: 40.7128, -74.0060
        - Destination: 42.3601, -71.0589
        
        **Los Angeles to San Francisco:**
        - Source: 34.0522, -118.2437
        - Destination: 37.7749, -122.4194
        """)

if __name__ == "__main__":
    main()
