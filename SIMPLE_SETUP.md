# Simple Setup Guide

## What You Need to Do

### 1. Install Python Packages
```bash
pip install streamlit pandas numpy scikit-learn joblib folium plotly geopy sqlalchemy psycopg2-binary streamlit-folium
```

### 2. Run the Application
```bash
streamlit run app.py
```

### 3. Open in Browser
Go to: http://localhost:8501

## Database Setup (Choose One Option)

### Option A: PostgreSQL Database
1. Install PostgreSQL
2. Create database:
```sql
CREATE DATABASE ev_routing;
```
3. Edit the file `database/db_connection.py` and change these lines (around line 25):
```python
# Change these default values:
'host': os.getenv('PGHOST', 'localhost'),        # Your database host
'port': os.getenv('PGPORT', '5432'),            # Your database port  
'database': os.getenv('PGDATABASE', 'ev_routing'), # Your database name
'user': os.getenv('PGUSER', 'postgres'),        # Your username
'password': os.getenv('PGPASSWORD', 'your_password')  # Your password
```

### Option B: Quick Test (No Database Setup)
1. Run sample data script:
```bash
python create_sample_data.py
```
2. This creates test data automatically

## Using the App

1. Click "Load EV Station Data"
2. Enter coordinates (example: 40.7128,-74.0060)
3. Select "Smart ML Filtering"
4. Click "Apply Filtering"
5. Click "Perform Clustering"
6. Click "Optimize Route"
7. View your route on the map!

## Key Files

- **`app.py`** - Main file to run
- **`database/db_connection.py`** - Database settings (edit this for your database)
- **`create_sample_data.py`** - Creates test data

## Common Issues

### "Module not found"
```bash
pip install streamlit
```

### "Database connection failed"
- Check PostgreSQL is running
- Update credentials in `database/db_connection.py`
- Or run `python create_sample_data.py` for test data

### "Port already in use"
```bash
streamlit run app.py --server.port 8080
```

That's it! The app will guide you through the rest.