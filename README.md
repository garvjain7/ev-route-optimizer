# EV Routing & Charging Station Optimization Pipeline

## Local Setup Instructions

### Prerequisites
- Python 3.8 or higher
- PostgreSQL database
- Git (optional, for cloning)

### 1. Installation Steps

#### Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd ev-routing-pipeline

# Or download and extract the files to a folder
```

#### Install Python Dependencies
```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install streamlit pandas numpy scikit-learn joblib folium plotly geopy sqlalchemy psycopg2-binary streamlit-folium
```

### 2. Database Setup

#### Install PostgreSQL
- Download and install PostgreSQL from https://www.postgresql.org/download/
- Create a database for the project

#### Create Database and User
```sql
-- Connect to PostgreSQL as admin user
CREATE DATABASE ev_routing;
CREATE USER ev_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE ev_routing TO ev_user;
```

#### Set Environment Variables
Create a `.env` file in the project root directory:
```bash
# Database connection settings
DATABASE_URL=postgresql://ev_user:your_password@localhost:5432/ev_routing

# Alternative individual parameters
PGHOST=localhost
PGPORT=5432
PGDATABASE=ev_routing
PGUSER=ev_user
PGPASSWORD=your_password
```

### 3. Database Configuration

The database connection is handled in `database/db_connection.py`. The system automatically tries multiple connection methods:

1. **Primary**: Uses `DATABASE_URL` environment variable
2. **Fallback**: Uses individual PG* environment variables
3. **Default**: Uses hardcoded localhost settings

#### Load Sample Data
```bash
# Run the sample data creation script
python create_sample_data.py
```

### 4. Running the Application

#### Start the Streamlit Server
```bash
# Navigate to project directory
cd ev-routing-pipeline

# Run the main application
streamlit run app.py

# The app will be available at http://localhost:8501
```

#### Alternative: Specify Port
```bash
# Run on specific port
streamlit run app.py --server.port 8080
```

### 5. Key Files and Their Purpose

#### Main Application Files
- **`app.py`** - Main Streamlit application (THIS IS THE FILE YOU RUN)
- **`create_sample_data.py`** - Creates sample EV station data for testing

#### Core Components
- **`database/db_connection.py`** - Database connection handler
- **`filtering/station_filters.py`** - Station filtering algorithms
- **`routing/ev_router.py`** - Route optimization engine
- **`clustering/clustering_engine.py`** - Station clustering algorithms
- **`visualization/map_visualizer.py`** - Interactive map creation

#### ML Models
- **`ml_models/station_predictor.py`** - Predicts station congestion and ratings
- **`ml_models/travel_time_predictor.py`** - Predicts realistic travel times
- **`ml_models/adaptive_router.py`** - Learning-based route optimization

#### Utilities
- **`utils/distance_calculator.py`** - Geographic distance calculations
- **`utils/geojson_exporter.py`** - Data export functionality

### 6. First-Time Setup Workflow

1. **Install dependencies** (see step 1 above)
2. **Setup PostgreSQL database** (see step 2 above)
3. **Set environment variables** (create .env file)
4. **Load sample data**:
   ```bash
   python create_sample_data.py
   ```
5. **Run the application**:
   ```bash
   streamlit run app.py
   ```
6. **Access the application** at http://localhost:8501

### 7. Using the Application

#### Basic Workflow
1. **Load Station Data** - Click "Load EV Station Data" to connect to database
2. **Set Route Parameters** - Enter source and destination coordinates
3. **Configure Filtering** - Choose filtering method (try "Smart ML Filtering")
4. **Apply Filters** - Click "Apply Filtering" to process stations
5. **Perform Clustering** - Click "Perform Clustering" to group stations
6. **Optimize Route** - Click "Optimize Route" to find best path
7. **View Results** - Interactive map shows optimized route

#### ML Features
- **Train Models** - Use buttons in "ML Model Management" section
- **Provide Feedback** - Use "Route Feedback" section after creating routes
- **View Analytics** - Click "View Performance Analytics" to see trends

### 8. Troubleshooting

#### Database Connection Issues
- Verify PostgreSQL is running: `pg_ctl status`
- Check connection parameters in `.env` file
- Ensure database and user exist
- Check firewall settings

#### Missing Dependencies
```bash
# Install missing packages
pip install package_name

# Or reinstall all
pip install -r requirements.txt  # if you create this file
```

#### Port Already in Use
```bash
# Use different port
streamlit run app.py --server.port 8080
```

#### No Station Data
```bash
# Load sample data
python create_sample_data.py
```

### 9. Environment Variables Reference

#### Required Variables
- `DATABASE_URL` - Complete PostgreSQL connection string
- OR individual variables:
  - `PGHOST` - Database host (default: localhost)
  - `PGPORT` - Database port (default: 5432)
  - `PGDATABASE` - Database name
  - `PGUSER` - Database username
  - `PGPASSWORD` - Database password

#### Optional Variables
- `STREAMLIT_SERVER_PORT` - Custom port for Streamlit
- `STREAMLIT_SERVER_ADDRESS` - Custom address (default: localhost)

### 10. Development Mode

#### Enable Debug Mode
```bash
# Run with debug logging
streamlit run app.py --logger.level=debug
```

#### Auto-reload on Changes
Streamlit automatically reloads when files change. No restart needed during development.

### 11. Data Requirements

The application expects EV station data with these columns:
- `id` - Unique identifier
- `name` - Station name
- `latitude` - GPS latitude
- `longitude` - GPS longitude
- `network` - Charging network name
- `level1_ports` - Number of Level 1 ports
- `level2_ports` - Number of Level 2 ports
- `dc_fast_ports` - Number of DC fast charging ports
- `access` - Public/Private access
- `city` - City name
- `state` - State name

### 12. Production Deployment

For production deployment, consider:
- Using environment-specific configuration
- Setting up proper database backups
- Configuring SSL/TLS
- Using a process manager like PM2 or systemd
- Setting up monitoring and logging