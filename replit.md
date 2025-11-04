# Breast Cancer Prediction App

## Overview
This is an **interactive machine learning application** that predicts whether a breast tumor is **Malignant** or **Benign** using two different models:
- Logistic Regression (with feature scaling)
- Random Forest Classifier

The app is built with **Streamlit** and uses the Breast Cancer Wisconsin Dataset.

## Project Structure
- `ml_project_ui.py` - Main Streamlit application (enhanced multi-page app)
- `pages/Model_Comparison.py` - Model performance comparison page with visualizations
- `Breast Cancer Wisconsin Dataset.csv` - Training dataset
- `sample_upload_template.csv` - Template for batch prediction uploads
- `auto-mpg.csv` - Additional dataset (not used in main app)
- `first.py`, `4.py` - Python practice scripts (not part of main app)
- `.streamlit/config.toml` - Streamlit configuration for Replit environment

## Features

### Main Page
1. **Dataset Overview** - Displays sample data and class distribution pie chart
2. **Feature Correlation Heatmap** - Interactive seaborn heatmap showing feature correlations
3. **Model Performance** - Side-by-side comparison of accuracy, precision, recall, and F1-score
4. **Single Patient Prediction** - Interactive form with all features for individual predictions
5. **Batch CSV Upload** - Upload patient data CSV for bulk predictions with downloadable results
6. **Sidebar Model Selection** - Choose between Logistic Regression or Random Forest for predictions

### Model Comparison Page
1. **Performance Metrics Table** - Tabular comparison of both models
2. **Bar Chart Visualization** - Grouped bar chart comparing all metrics
3. **Line Chart Visualization** - Line plot showing performance across metrics
4. **Best Model Indicators** - Metric cards highlighting top performers

## Dependencies
- streamlit - Web application framework
- pandas - Data manipulation
- numpy - Numerical operations
- scikit-learn - Machine learning models and metrics
- seaborn - Statistical data visualization
- matplotlib - Plotting library

## Technical Details
- **Data Preprocessing**: StandardScaler applied to features for Logistic Regression
- **Model Training**: Cached using @st.cache_resource for performance
- **Batch Predictions**: Validates CSV columns and generates confidence scores
- **Download Functionality**: Results exportable as CSV with predictions and confidence levels

## CSV Upload Format
For batch predictions, upload a CSV file with the following 30 features (same as training data):
- radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, etc.
- Use `sample_upload_template.csv` as a reference template

## Workflow
The app runs on port 5000 using Streamlit's built-in server. The workflow is configured to:
- Bind to 0.0.0.0:5000 (required for Replit's proxy)
- Allow all hosts (CORS disabled for Replit iframe compatibility)
- Run in headless mode
- Wide layout for better visualization space

## Deployment
Configured for autoscale deployment (suitable for stateless web applications).

## Recent Changes (November 4, 2025)

### Initial Setup
- Installed Python 3.11 and all required dependencies
- Created Streamlit configuration for Replit environment
- Set up workflow to run on port 5000 with webview output
- Added .gitignore for Python project
- Configured deployment settings for production

### Enhancement Update
- Added seaborn and matplotlib for advanced visualizations
- Created multi-page app structure with Model Comparison page
- Implemented correlation heatmap on main page
- Added sidebar with model selection functionality
- Built CSV batch prediction with download capability
- Improved UI/UX with professional styling, emojis, and layout
- Fixed convergence warnings by increasing max_iter and adding feature scaling

## User Preferences
- Prefers interactive and professional interfaces
- Wants visual comparisons and data exploration features
- Values batch processing capabilities

## Project Architecture
- **Frontend**: Streamlit multi-page web application (port 5000)
- **Backend**: None (single-tier application)
- **Database**: None (uses CSV file for dataset)
- **ML Models**: Trained on startup using scikit-learn with caching
- **Pages**: Main app + Model Comparison page
