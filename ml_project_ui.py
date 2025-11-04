# ml_project_ui.py
# Breast Cancer Prediction App (Logistic Regression & Random Forest)
#streamlit run ml_project_ui.py

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("Breast Cancer Wisconsin Dataset.csv")
    data['label'] = data['diagnosis'].map({'M': 0, 'B': 1})
    X = data.drop(columns=['label', 'id', 'diagnosis'], axis=1)
    Y = data['label']
    return X, Y, data

X, Y, full_data = load_data()

# -----------------------------
# Train Models
# -----------------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, Y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔬 Breast Cancer Prediction App")
st.write("This app predicts whether a breast tumor is **Malignant** or **Benign** using Logistic Regression and Random Forest models.")

st.header("📊 Dataset Overview")
st.write(full_data.head())

# Show class distribution
st.write("### Class Distribution:")
st.write(full_data['diagnosis'].value_counts())

# -----------------------------
# Model Performance
# -----------------------------
st.header("⚡ Model Performance")

# Logistic Regression
y_pred_log = log_model.predict(X_test_scaled)
st.subheader("Logistic Regression")
st.write(f"Accuracy: {accuracy_score(Y_test, y_pred_log):.2f}")
st.write(f"Precision: {precision_score(Y_test, y_pred_log):.2f}")
st.write(f"Recall: {recall_score(Y_test, y_pred_log):.2f}")
st.write(f"F1-Score: {f1_score(Y_test, y_pred_log):.2f}")

# Random Forest
y_pred_rf = rf_model.predict(X_test)
st.subheader("Random Forest")
st.write(f"Accuracy: {accuracy_score(Y_test, y_pred_rf):.2f}")
st.write(f"Precision: {precision_score(Y_test, y_pred_rf):.2f}")
st.write(f"Recall: {recall_score(Y_test, y_pred_rf):.2f}")
st.write(f"F1-Score: {f1_score(Y_test, y_pred_rf):.2f}")

# -----------------------------
# User Input Prediction
# -----------------------------
st.header("🧑‍⚕️ Try Your Own Prediction")

def user_input():
    features = []
    for col in X.columns:
        value = st.number_input(f"Enter value for {col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
        features.append(value)
    return np.array(features).reshape(1, -1)

if st.checkbox("Enter Patient Data"):
    input_data = user_input()
    input_data_scaled = scaler.transform(input_data)

    log_prediction = log_model.predict(input_data_scaled)[0]
    rf_prediction = rf_model.predict(input_data)[0]

    st.subheader("Prediction Results")
    st.write(f"**Logistic Regression:** {'Malignant' if log_prediction == 0 else 'Benign'}")
    st.write(f"**Random Forest:** {'Malignant' if rf_prediction == 0 else 'Benign'}")
