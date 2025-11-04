# Breast Cancer Prediction App

## Overview
This is a machine learning application that predicts whether a breast tumor is **Malignant** or **Benign** using two different models:
- Logistic Regression
- Random Forest Classifier

The app is built with **Streamlit** and uses the Breast Cancer Wisconsin Dataset.

## Project Structure
- `ml_project_ui.py` - Main Streamlit application
- `Breast Cancer Wisconsin Dataset.csv` - Training dataset
- `auto-mpg.csv` - Additional dataset (not used in main app)
- `first.py`, `4.py` - Python practice scripts (not part of main app)
- `.streamlit/config.toml` - Streamlit configuration for Replit environment

## Features
1. **Dataset Overview** - Displays the first few rows and class distribution
2. **Model Performance** - Shows accuracy, precision, recall, and F1-score for both models
3. **Interactive Prediction** - Users can input patient data to get predictions from both models

## Dependencies
- streamlit - Web application framework
- pandas - Data manipulation
- numpy - Numerical operations
- scikit-learn - Machine learning models and metrics

## Workflow
The app runs on port 5000 using Streamlit's built-in server. The workflow is configured to:
- Bind to 0.0.0.0:5000 (required for Replit's proxy)
- Allow all hosts (CORS disabled for Replit iframe compatibility)
- Run in headless mode

## Deployment
Configured for autoscale deployment (suitable for stateless web applications).

## Recent Changes (November 4, 2025)
- Installed Python 3.11 and all required dependencies
- Created Streamlit configuration for Replit environment
- Set up workflow to run on port 5000 with webview output
- Added .gitignore for Python project
- Configured deployment settings for production
- Created project documentation

## User Preferences
None specified yet.

## Project Architecture
- **Frontend**: Streamlit web application (port 5000)
- **Backend**: None (single-tier application)
- **Database**: None (uses CSV file for dataset)
- **ML Models**: Trained on startup using scikit-learn
