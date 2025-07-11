# Local Setup Guide for EV Routing Application

## Quick Start - What You Need to Know

### Main File to Run
**Run this command:** `streamlit run app.py`

### Database Connection File
**File:** `database/db_connection.py` - This handles all database connections

### Required Dependencies
```bash
pip install streamlit pandas numpy scikit-learn joblib folium plotly geopy sqlalchemy psycopg2-binary streamlit-folium
```

## Step-by-Step Setup

### 1. Install Python Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install packages
pip install streamlit pandas numpy scikit-learn joblib folium plotly geopy sqlalchemy psycopg2-binary streamlit-folium
```

### 2. Database Setup Options

#### Option A: Use PostgreSQL (Recommended)
1. Install PostgreSQL on your machine
2. Create a database:
```sql
CREATE DATABASE ev_routing;
CREATE USER ev_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE ev_routing TO ev_user;
```

3. Set environment variables (create `.env` file):
```
DATABASE_URL=postgresql://ev_user:your_password@localhost:5432/ev_routing
```

#### Option B: Quick Test with Sample Data
1. Run the sample data script:
```bash
python create_sample_data.py
```

### 3. Database Connection Details

The database connection is handled in `database/db_connection.py`. It tries these methods in order:

1. **Environment variable `DATABASE_URL`**
2. **Individual environment variables:**
   - `PGHOST` (default: localhost)
   - `PGPORT` (default: 5432)
   - `PGDATABASE` (your database name)
   - `PGUSER` (your username)
   - `PGPASSWORD` (your password)

### 4. Running the Application

```bash
# Navigate to project folder
cd your-project-folder

# Run the main application
streamlit run app.py

# App opens at: http://localhost:8501
```

### 5. First Time Using the App

1. **Load Data**: Click "Load EV Station Data" button
2. **Set Route**: Enter coordinates (example: 40.7128,-74.0060 for NYC)
3. **Filter Stations**: Choose "Smart ML Filtering" for best results
4. **Apply Filters**: Click "Apply Filtering"
5. **Cluster**: Click "Perform Clustering"
6. **Optimize Route**: Click "Optimize Route"
7. **View Map**: See your optimized route with charging stations

### 6. Understanding Key Files

#### Main Files:
- `app.py` - **Main application (run this file)**
- `create_sample_data.py` - Creates test data if no database

#### Database:
- `database/db_connection.py` - **Database connection handler**

#### ML Models:
- `ml_models/station_predictor.py` - Predicts station quality
- `ml_models/travel_time_predictor.py` - Predicts travel times
- `ml_models/adaptive_router.py` - Learning from user feedback

#### Core Logic:
- `filtering/station_filters.py` - Filters charging stations
- `routing/ev_router.py` - Optimizes routes
- `clustering/clustering_engine.py` - Groups stations
- `visualization/map_visualizer.py` - Creates maps

### 7. Common Issues and Solutions

#### "No module named 'streamlit'"
```bash
pip install streamlit
```

#### "Database connection failed"
- Check PostgreSQL is running
- Verify database name and credentials
- Try running `create_sample_data.py` first

#### "No station data found"
- Click "Load EV Station Data" button in the app
- Or run `python create_sample_data.py` to create sample data

#### Port already in use
```bash
streamlit run app.py --server.port 8080
```

### 8. Environment Variables (Optional)

Create a `.env` file in your project folder:
```
# Database connection
DATABASE_URL=postgresql://username:password@localhost:5432/database_name

# Or individual variables
PGHOST=localhost
PGPORT=5432
PGDATABASE=ev_routing
PGUSER=your_username
PGPASSWORD=your_password
```

### 9. Testing the ML Features

1. **Train Models**: Use the "ML Model Management" section
2. **Smart Filtering**: Select "Smart ML Filtering" in the dropdown
3. **Route Feedback**: Provide feedback after creating routes
4. **Analytics**: View performance data and trends

### 10. Sample Coordinates for Testing

Try these coordinates for testing:
- **New York**: 40.7128, -74.0060
- **Los Angeles**: 34.0522, -118.2437
- **Chicago**: 41.8781, -87.6298
- **Boston**: 42.3601, -71.0589

### 11. Project Structure
```
ev-routing-pipeline/
├── app.py                          # Main application file
├── create_sample_data.py           # Sample data generator
├── database/
│   └── db_connection.py           # Database connection
├── ml_models/
│   ├── station_predictor.py       # ML station predictions
│   ├── travel_time_predictor.py   # ML travel time predictions
│   └── adaptive_router.py         # ML adaptive routing
├── filtering/
│   └── station_filters.py         # Station filtering logic
├── routing/
│   └── ev_router.py               # Route optimization
├── clustering/
│   └── clustering_engine.py       # Station clustering
├── visualization/
│   └── map_visualizer.py          # Interactive maps
└── utils/
    ├── distance_calculator.py     # Distance calculations
    └── geojson_exporter.py        # Data export
```

## Quick Commands Summary

```bash
# Install dependencies
pip install streamlit pandas numpy scikit-learn joblib folium plotly geopy sqlalchemy psycopg2-binary streamlit-folium

# Create sample data (if no database)
python create_sample_data.py

# Run the application
streamlit run app.py

# Access at: http://localhost:8501
```

The application will guide you through the process once it's running!