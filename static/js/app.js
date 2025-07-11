// Global variables
let dataLoaded = false;
let filteredStations = false;
let routeOptimized = false;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    updateRangeValues();
});

// Initialize all event listeners
function initializeEventListeners() {
    // Data loading
    document.getElementById('loadSampleData').addEventListener('click', loadSampleData);
    document.getElementById('loadDatabase').addEventListener('click', loadDatabase);
    
    // Filtering and optimization
    document.getElementById('applyFiltering').addEventListener('click', applyFiltering);
    document.getElementById('performClustering').addEventListener('click', performClustering);
    document.getElementById('optimizeRoute').addEventListener('click', optimizeRoute);
    
    // Export functions
    document.getElementById('exportGeoJSON').addEventListener('click', exportGeoJSON);
    document.getElementById('exportRoute').addEventListener('click', exportRoute);
    
    // ML Model training
    document.getElementById('trainStationPredictor').addEventListener('click', () => trainModel('station_predictor'));
    document.getElementById('trainTravelTimePredictor').addEventListener('click', () => trainModel('travel_time_predictor'));
    document.getElementById('trainAdaptiveRouter').addEventListener('click', () => trainModel('adaptive_router'));
    
    // Feedback and analytics
    document.getElementById('submitFeedback').addEventListener('click', submitFeedback);
    document.getElementById('viewAnalytics').addEventListener('click', viewAnalytics);
    
    // Range input updates
    setupRangeInputs();
}

// Setup range input value updates
function setupRangeInputs() {
    const rangeInputs = [
        'batteryRange', 'consumptionRate', 'chargingTime', 'safetyMargin',
        'detourFactor', 'corridorWidth', 'minStationRating', 'nClusters',
        'overallSatisfaction'
    ];
    
    rangeInputs.forEach(id => {
        const input = document.getElementById(id);
        const valueSpan = document.getElementById(id + 'Value');
        if (input && valueSpan) {
            input.addEventListener('input', function() {
                valueSpan.textContent = this.value;
            });
        }
    });
}

// Update range values on page load
function updateRangeValues() {
    const rangeInputs = [
        'batteryRange', 'consumptionRate', 'chargingTime', 'safetyMargin',
        'detourFactor', 'corridorWidth', 'minStationRating', 'nClusters',
        'overallSatisfaction'
    ];
    
    rangeInputs.forEach(id => {
        const input = document.getElementById(id);
        const valueSpan = document.getElementById(id + 'Value');
        if (input && valueSpan) {
            valueSpan.textContent = input.value;
        }
    });
}

// Show loading indicator
function showLoading(show = true) {
    const loading = document.getElementById('globalLoading');
    loading.style.display = show ? 'block' : 'none';
}

// Show alert message
function showAlert(message, type = 'info') {
    const alertClass = `alert-${type}`;
    const alertHtml = `
        <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
            <i class="fas fa-${getIconForType(type)} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    // Find a suitable container and prepend the alert
    const container = document.querySelector('.main-content');
    container.insertAdjacentHTML('afterbegin', alertHtml);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const alert = container.querySelector('.alert');
        if (alert) {
            alert.remove();
        }
    }, 5000);
}

// Get icon for alert type
function getIconForType(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'exclamation-triangle',
        'warning': 'exclamation-circle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Load sample data
async function loadSampleData() {
    showLoading(true);
    try {
        const response = await fetch('/api/load_sample_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert(data.message, 'success');
            dataLoaded = true;
            showDataSections();
            updateStats();
        } else {
            showAlert(data.error, 'danger');
        }
    } catch (error) {
        showAlert('Failed to load sample data: ' + error.message, 'danger');
    } finally {
        showLoading(false);
    }
}

// Load database
async function loadDatabase() {
    showLoading(true);
    try {
        const response = await fetch('/api/load_database', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert(data.message, 'success');
            dataLoaded = true;
            showDataSections();
            updateStats();
        } else {
            showAlert(data.error, 'danger');
        }
    } catch (error) {
        showAlert('Failed to load database: ' + error.message, 'danger');
    } finally {
        showLoading(false);
    }
}

// Show data sections after loading
function showDataSections() {
    document.getElementById('welcomeSection').style.display = 'none';
    document.getElementById('statsSection').style.display = 'block';
    document.getElementById('filteringSection').style.display = 'block';
    document.getElementById('mlManagementSection').style.display = 'block';
    document.getElementById('analyticsSection').style.display = 'block';
}

// Update statistics
async function updateStats() {
    try {
        const response = await fetch('/api/get_stats');
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('totalStations').textContent = data.stats.total_stations;
            document.getElementById('statesCovered').textContent = data.stats.states_covered;
            
            // Calculate direct distance
            const sourceLat = parseFloat(document.getElementById('sourceLat').value);
            const sourceLon = parseFloat(document.getElementById('sourceLon').value);
            const destLat = parseFloat(document.getElementById('destLat').value);
            const destLon = parseFloat(document.getElementById('destLon').value);
            
            const distance = calculateDistance(sourceLat, sourceLon, destLat, destLon);
            document.getElementById('directDistance').textContent = distance.toFixed(1);
            
            // Calculate estimated stops
            const batteryRange = parseFloat(document.getElementById('batteryRange').value);
            const safetyMargin = parseFloat(document.getElementById('safetyMargin').value);
            const effectiveRange = batteryRange * (1 - safetyMargin / 100);
            const estimatedStops = Math.max(0, Math.ceil(distance / effectiveRange) - 1);
            document.getElementById('estimatedStops').textContent = estimatedStops;
        }
    } catch (error) {
        console.error('Failed to update stats:', error);
    }
}

// Calculate distance between two points
function calculateDistance(lat1, lon1, lat2, lon2) {
    const R = 6371; // Earth's radius in km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
}

// Apply filtering
async function applyFiltering() {
    if (!dataLoaded) {
        showAlert('Please load data first', 'warning');
        return;
    }
    
    showLoading(true);
    
    try {
        const requestData = {
            source_lat: parseFloat(document.getElementById('sourceLat').value),
            source_lon: parseFloat(document.getElementById('sourceLon').value),
            dest_lat: parseFloat(document.getElementById('destLat').value),
            dest_lon: parseFloat(document.getElementById('destLon').value),
            filtering_method: document.getElementById('filteringMethod').value,
            params: {
                prefer_fast_charging: document.getElementById('preferFastCharging').checked,
                avoid_congestion: document.getElementById('avoidCongestion').checked,
                min_station_rating: parseFloat(document.getElementById('minStationRating').value),
                detour_factor: parseFloat(document.getElementById('detourFactor').value),
                corridor_width: parseFloat(document.getElementById('corridorWidth').value),
                max_distance_source: 200,
                max_distance_dest: 200,
                charging_types: ['AC', 'DC'],
                power_levels: ['Level2', 'DC_Fast']
            }
        };
        
        const response = await fetch('/api/apply_filtering', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert(data.message, 'success');
            filteredStations = true;
            showFilteringResults(data);
            document.getElementById('optimizationSection').style.display = 'block';
            document.getElementById('exportSection').style.display = 'block';
        } else {
            showAlert(data.error, 'danger');
        }
    } catch (error) {
        showAlert('Filtering failed: ' + error.message, 'danger');
    } finally {
        showLoading(false);
    }
}

// Show filtering results
function showFilteringResults(data) {
    let resultsHtml = `
        <div class="alert alert-success">
            <i class="fas fa-check-circle me-2"></i>
            Filtered to ${data.filtered_count} stations
        </div>
    `;
    
    if (data.ml_summary && Object.keys(data.ml_summary).length > 0) {
        resultsHtml += `
            <div class="card mt-3">
                <div class="card-header">
                    <i class="fas fa-robot me-2"></i>ML Predictions Summary
                </div>
                <div class="card-body">
                    <div class="row">
        `;
        
        if (data.ml_summary.avg_congestion) {
            resultsHtml += `
                <div class="col-md-4">
                    <div class="metric-card">
                        <div class="metric-value">${data.ml_summary.avg_congestion}</div>
                        <div class="metric-label">Avg Congestion</div>
                    </div>
                </div>
            `;
        }
        
        if (data.ml_summary.avg_rating) {
            resultsHtml += `
                <div class="col-md-4">
                    <div class="metric-card">
                        <div class="metric-value">${data.ml_summary.avg_rating}/5</div>
                        <div class="metric-label">Avg Rating</div>
                    </div>
                </div>
            `;
        }
        
        if (data.ml_summary.avg_score) {
            resultsHtml += `
                <div class="col-md-4">
                    <div class="metric-card">
                        <div class="metric-value">${data.ml_summary.avg_score}/5</div>
                        <div class="metric-label">Avg ML Score</div>
                    </div>
                </div>
            `;
        }
        
        resultsHtml += `
                    </div>
                </div>
            </div>
        `;
    }
    
    document.getElementById('filteringResults').innerHTML = resultsHtml;
}

// Perform clustering
async function performClustering() {
    if (!filteredStations) {
        showAlert('Please apply filtering first', 'warning');
        return;
    }
    
    showLoading(true);
    
    try {
        const requestData = {
            n_clusters: parseInt(document.getElementById('nClusters').value)
        };
        
        const response = await fetch('/api/perform_clustering', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert(data.message, 'success');
            updateOptimizationResults('clustering', data);
        } else {
            showAlert(data.error, 'danger');
        }
    } catch (error) {
        showAlert('Clustering failed: ' + error.message, 'danger');
    } finally {
        showLoading(false);
    }
}

// Optimize route
async function optimizeRoute() {
    if (!filteredStations) {
        showAlert('Please apply filtering first', 'warning');
        return;
    }
    
    showLoading(true);
    
    try {
        const requestData = {
            source_lat: parseFloat(document.getElementById('sourceLat').value),
            source_lon: parseFloat(document.getElementById('sourceLon').value),
            dest_lat: parseFloat(document.getElementById('destLat').value),
            dest_lon: parseFloat(document.getElementById('destLon').value),
            battery_range: parseFloat(document.getElementById('batteryRange').value),
            consumption_rate: parseFloat(document.getElementById('consumptionRate').value),
            charging_time: parseFloat(document.getElementById('chargingTime').value),
            safety_margin: parseFloat(document.getElementById('safetyMargin').value)
        };
        
        const response = await fetch('/api/optimize_route', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert(data.message, 'success');
            routeOptimized = true;
            updateOptimizationResults('routing', data);
            generateMap();
            showRouteAnalysis();
            document.getElementById('feedbackSection').style.display = 'block';
        } else {
            showAlert(data.error, 'danger');
        }
    } catch (error) {
        showAlert('Route optimization failed: ' + error.message, 'danger');
    } finally {
        showLoading(false);
    }
}

// Update optimization results
function updateOptimizationResults(type, data) {
    let resultsHtml = document.getElementById('optimizationResults').innerHTML;
    
    if (type === 'clustering') {
        resultsHtml += `
            <div class="alert alert-success">
                <i class="fas fa-project-diagram me-2"></i>
                Created ${data.n_clusters} clusters successfully
            </div>
        `;
    } else if (type === 'routing') {
        resultsHtml += `
            <div class="alert alert-success">
                <i class="fas fa-route me-2"></i>
                Route optimized successfully
            </div>
        `;
        
        if (data.route_stats) {
            resultsHtml += `
                <div class="row mt-3">
                    <div class="col-md-3">
                        <div class="metric-card">
                            <div class="metric-value">${data.route_stats.total_distance || 0}</div>
                            <div class="metric-label">Total Distance (km)</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <div class="metric-value">${data.route_stats.charging_stops || 0}</div>
                            <div class="metric-label">Charging Stops</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <div class="metric-value">${data.route_stats.total_time || 0}</div>
                            <div class="metric-label">Total Time (hours)</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <div class="metric-value">${data.route_stats.efficiency || 0}</div>
                            <div class="metric-label">Efficiency (kWh/100km)</div>
                        </div>
                    </div>
                </div>
            `;
        }
    }
    
    document.getElementById('optimizationResults').innerHTML = resultsHtml;
}

// Generate map
async function generateMap() {
    document.getElementById('mapSection').style.display = 'block';
    document.getElementById('mapLoading').style.display = 'block';
    
    try {
        const requestData = {
            source_lat: parseFloat(document.getElementById('sourceLat').value),
            source_lon: parseFloat(document.getElementById('sourceLon').value),
            dest_lat: parseFloat(document.getElementById('destLat').value),
            dest_lon: parseFloat(document.getElementById('destLon').value)
        };
        
        const response = await fetch('/api/generate_map', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('mapContainer').innerHTML = data.map_html;
        } else {
            showAlert('Map generation failed: ' + data.error, 'danger');
            document.getElementById('mapContainer').innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    Map visualization failed. You can still use the export features.
                </div>
            `;
        }
    } catch (error) {
        showAlert('Map generation failed: ' + error.message, 'danger');
    } finally {
        document.getElementById('mapLoading').style.display = 'none';
    }
}

// Show route analysis
async function showRouteAnalysis() {
    document.getElementById('routeAnalysisSection').style.display = 'block';
    
    try {
        const response = await fetch('/api/get_route_details');
        const data = await response.json();
        
        if (data.success) {
            // Display charging stops
            if (data.charging_stops && data.charging_stops.length > 0) {
                let stopsHtml = `
                    <div class="card mt-3">
                        <div class="card-header">
                            <i class="fas fa-charging-station me-2"></i>Charging Stops Details
                        </div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-striped">
                                    <thead>
                                        <tr>
                                            <th>Station</th>
                                            <th>Distance (km)</th>
                                            <th>Charging Time</th>
                                            <th>Battery Level</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                `;
                
                data.charging_stops.forEach(stop => {
                    stopsHtml += `
                        <tr>
                            <td>${stop.station_name || 'Unknown'}</td>
                            <td>${stop.distance_from_source || 0}</td>
                            <td>${stop.charging_time || 0} min</td>
                            <td>${stop.battery_level || 0}%</td>
                        </tr>
                    `;
                });
                
                stopsHtml += `
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                `;
                
                document.getElementById('chargingStops').innerHTML = stopsHtml;
            }
            
            // Display ML analysis
            if (data.ml_predictions && data.ml_predictions.efficiency_score) {
                const efficiency = data.ml_predictions.efficiency_score;
                let efficiencyClass = 'success';
                let efficiencyText = 'Highly efficient route';
                
                if (efficiency < 0.6) {
                    efficiencyClass = 'danger';
                    efficiencyText = 'Low efficiency route';
                } else if (efficiency < 0.8) {
                    efficiencyClass = 'warning';
                    efficiencyText = 'Moderately efficient route';
                }
                
                let mlHtml = `
                    <div class="card mt-3">
                        <div class="card-header">
                            <i class="fas fa-robot me-2"></i>ML Route Analysis
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="metric-card">
                                        <div class="metric-value">${(efficiency * 100).toFixed(1)}%</div>
                                        <div class="metric-label">Predicted Efficiency</div>
                                    </div>
                                    <div class="alert alert-${efficiencyClass} mt-2">
                                        <i class="fas fa-circle me-2"></i>${efficiencyText}
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <h6>Recommendations</h6>
                `;
                
                if (data.ml_predictions.recommendations && data.ml_predictions.recommendations.length > 0) {
                    data.ml_predictions.recommendations.slice(0, 3).forEach(rec => {
                        let alertClass = 'info';
                        if (rec.type === 'efficiency_warning' || rec.type === 'station_warning') {
                            alertClass = 'warning';
                        }
                        
                        mlHtml += `
                            <div class="alert alert-${alertClass}">
                                <i class="fas fa-lightbulb me-2"></i>${rec.message}
                            </div>
                        `;
                    });
                } else {
                    mlHtml += `
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle me-2"></i>No specific recommendations for this route
                        </div>
                    `;
                }
                
                mlHtml += `
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                
                document.getElementById('mlAnalysis').innerHTML = mlHtml;
            }
        }
    } catch (error) {
        console.error('Failed to load route details:', error);
    }
}

// Export GeoJSON
async function exportGeoJSON() {
    if (!filteredStations) {
        showAlert('Please apply filtering first', 'warning');
        return;
    }
    
    try {
        const response = await fetch('/api/export_geojson');
        const data = await response.json();
        
        if (data.success) {
            downloadFile(JSON.stringify(data.geojson, null, 2), data.filename, 'application/json');
            showAlert('GeoJSON exported successfully', 'success');
        } else {
            showAlert(data.error, 'danger');
        }
    } catch (error) {
        showAlert('Export failed: ' + error.message, 'danger');
    }
}

// Export route data
async function exportRoute() {
    if (!routeOptimized) {
        showAlert('Please optimize route first', 'warning');
        return;
    }
    
    try {
        const response = await fetch('/api/export_route');
        const data = await response.json();
        
        if (data.success) {
            downloadFile(JSON.stringify(data.route_data, null, 2), data.filename, 'application/json');
            showAlert('Route data exported successfully', 'success');
        } else {
            showAlert(data.error, 'danger');
        }
    } catch (error) {
        showAlert('Export failed: ' + error.message, 'danger');
    }
}

// Download file utility
function downloadFile(content, filename, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Train ML model
async function trainModel(modelType) {
    if (!dataLoaded) {
        showAlert('Please load data first', 'warning');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch('/api/train_models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_type: modelType })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert(data.message, 'success');
        } else {
            showAlert(data.message || 'Training failed', 'danger');
        }
    } catch (error) {
        showAlert('Training failed: ' + error.message, 'danger');
    } finally {
        showLoading(false);
    }
}

// Submit feedback
async function submitFeedback() {
    if (!routeOptimized) {
        showAlert('Please optimize route first', 'warning');
        return;
    }
    
    const feedback = {
        overall_satisfaction: parseInt(document.getElementById('overallSatisfaction').value),
        actual_charging_time: parseInt(document.getElementById('actualChargingTime').value),
        total_charging_time: parseInt(document.getElementById('actualChargingTime').value),
        station_issues: parseInt(document.getElementById('stationIssues').value),
        charging_speed: document.getElementById('chargingSpeed').value,
        station_availability: document.getElementById('stationAvailability').checked,
        would_recommend: document.getElementById('wouldRecommend').checked
    };
    
    try {
        const response = await fetch('/api/submit_feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(feedback)
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert(data.message, 'success');
        } else {
            showAlert(data.message || 'Feedback submission failed', 'danger');
        }
    } catch (error) {
        showAlert('Feedback submission failed: ' + error.message, 'danger');
    }
}

// View analytics
async function viewAnalytics() {
    try {
        const response = await fetch('/api/get_analytics');
        const data = await response.json();
        
        if (data.success) {
            displayAnalytics(data.analytics);
        } else {
            showAlert(data.message || 'No analytics data available', 'info');
        }
    } catch (error) {
        showAlert('Analytics failed: ' + error.message, 'danger');
    }
}

// Display analytics
function displayAnalytics(analytics) {
    let analyticsHtml = `
        <div class="card mt-3">
            <div class="card-header">
                <i class="fas fa-chart-line me-2"></i>Performance Metrics
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-4">
                        <div class="metric-card">
                            <div class="metric-value">${analytics.total_routes || 0}</div>
                            <div class="metric-label">Total Routes</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="metric-card">
                            <div class="metric-value">${((analytics.avg_efficiency || 0) * 100).toFixed(1)}%</div>
                            <div class="metric-label">Avg Efficiency</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="metric-card">
                            <div class="metric-value">${analytics.best_stations ? analytics.best_stations.length : 0}</div>
                            <div class="metric-label">Best Stations</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    if (analytics.efficiency_trend && analytics.efficiency_trend.length > 0) {
        analyticsHtml += `
            <div class="card mt-3">
                <div class="card-header">
                    <i class="fas fa-chart-line me-2"></i>Efficiency Trend
                </div>
                <div class="card-body">
                    <canvas id="efficiencyChart" width="400" height="200"></canvas>
                </div>
            </div>
        `;
    }
    
    document.getElementById('analyticsResults').innerHTML = analyticsHtml;
    
    // Create chart if trend data exists
    if (analytics.efficiency_trend && analytics.efficiency_trend.length > 0) {
        const ctx = document.getElementById('efficiencyChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: analytics.efficiency_trend.map((_, i) => `Route ${i + 1}`),
                datasets: [{
                    label: 'Efficiency Score',
                    data: analytics.efficiency_trend,
                    borderColor: 'rgb(31, 119, 180)',
                    backgroundColor: 'rgba(31, 119, 180, 0.1)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }
}