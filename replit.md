# EV Routing & Charging Station Optimization Pipeline

## Overview

This is a comprehensive Streamlit application for optimizing electric vehicle (EV) routes with intelligent charging station selection. The system combines geospatial analysis, machine learning clustering, and route optimization to provide EV drivers with optimal charging strategies for long-distance travel.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web interface
- **Interactive Components**: 
  - Streamlit-folium for interactive maps
  - Plotly for data visualizations
  - Sidebar-based configuration panel
- **State Management**: Streamlit session state for maintaining data across user interactions

### Backend Architecture
- **Modular Design**: Component-based architecture with clear separation of concerns
- **Core Modules**:
  - Database connection layer
  - Station filtering engine
  - Clustering algorithms
  - Route optimization engine
  - Map visualization
  - Data export utilities

### Data Storage Solutions
- **Primary Database**: PostgreSQL for storing EV charging station data
- **Connection Strategy**: SQLAlchemy ORM with environment-based configuration
- **Fallback Support**: Multiple connection methods (DATABASE_URL or individual parameters)

## Key Components

### 1. Database Connection (`database/db_connection.py`)
- **Purpose**: Manages PostgreSQL connections and data retrieval
- **Features**: 
  - Environment-based configuration
  - Connection pooling via SQLAlchemy
  - Graceful error handling
  - Multiple table name support for EV station data

### 2. Station Filtering (`filtering/station_filters.py`)
- **Purpose**: Filters charging stations based on geographic and route constraints
- **Key Features**:
  - Geographic bounding box filtering
  - Distance-based filtering from source/destination
  - Route corridor analysis

### 3. Clustering Engine (`clustering/clustering_engine.py`)
- **Purpose**: Groups charging stations using K-means clustering
- **Features**:
  - Multi-dimensional feature clustering (location + station attributes)
  - Automatic cluster count adjustment
  - Cluster statistics calculation
  - Centroid computation

### 4. Route Optimization (`routing/ev_router.py`)
- **Purpose**: Optimizes EV routes considering battery constraints
- **Key Features**:
  - Battery range calculations with safety margins
  - Multi-stop route planning
  - Energy consumption modeling
  - Charging time integration

### 5. Map Visualization (`visualization/map_visualizer.py`)
- **Purpose**: Creates interactive maps for route and station visualization
- **Features**:
  - Folium-based interactive maps
  - Color-coded clustering visualization
  - Route path rendering
  - Station information popups

### 6. Utilities
- **Distance Calculator**: Haversine distance calculations and matrix operations
- **GeoJSON Exporter**: Export station data in GeoJSON format for external use

## Data Flow

1. **Data Ingestion**: PostgreSQL database connection retrieves EV station data
2. **Preprocessing**: Geographic filtering based on route parameters
3. **Clustering**: K-means algorithm groups stations by location and features
4. **Route Optimization**: Algorithm selects optimal charging stops considering:
   - Battery range constraints
   - Charging times
   - Route efficiency
5. **Visualization**: Interactive maps display optimized routes and station clusters
6. **Export**: Results can be exported as GeoJSON for external applications

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms (K-means clustering)
- **Folium**: Interactive mapping
- **Plotly**: Data visualization
- **GeoPy**: Geographic calculations
- **SQLAlchemy**: Database ORM
- **psycopg2**: PostgreSQL adapter

### Database Requirements
- **PostgreSQL**: Primary data storage
- **Expected Schema**: Tables containing EV station data with latitude/longitude coordinates

## Deployment Strategy

### Environment Configuration
- **Database Connection**: Environment variables for PostgreSQL connection
- **Fallback Strategy**: Multiple connection methods for different deployment scenarios
- **Configuration Variables**:
  - `DATABASE_URL`: Primary connection string
  - `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`: Individual parameters

### Deployment Considerations
- **Dependencies**: Requirements include scientific computing libraries
- **Database Setup**: Requires PostgreSQL instance with EV station data
- **Resource Requirements**: Moderate computational resources for clustering and route optimization
- **Scalability**: Modular design allows for easy component replacement or scaling

### Error Handling
- **Graceful Degradation**: Application continues functioning with reduced features if components fail
- **User Feedback**: Clear error messages and warnings for troubleshooting
- **Logging**: Streamlit's built-in error display for debugging

## Architecture Decisions

### Technology Choices
- **Streamlit over Flask/Django**: Chosen for rapid prototyping and built-in UI components
- **PostgreSQL**: Selected for robust geospatial data handling
- **K-means Clustering**: Simple and effective for geographic clustering
- **Folium for Mapping**: Interactive capabilities with Python integration

### Design Patterns
- **Component-based Architecture**: Enables modularity and testability
- **Session State Management**: Maintains user data across interactions
- **Error Boundary Pattern**: Prevents cascading failures between components