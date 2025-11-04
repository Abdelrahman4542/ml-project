import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Model Comparison", page_icon="📊")

st.title("📊 Model Performance Comparison")
st.write("Compare the performance metrics of Logistic Regression and Random Forest models")

@st.cache_data
def load_data():
    data = pd.read_csv("Breast Cancer Wisconsin Dataset.csv")
    data['label'] = data['diagnosis'].map({'M': 0, 'B': 1})
    X = data.drop(columns=['label', 'id', 'diagnosis'], axis=1)
    Y = data['label']
    return X, Y

X, Y = load_data()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, Y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

y_pred_log = log_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test)

metrics = {
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [accuracy_score(Y_test, y_pred_log), accuracy_score(Y_test, y_pred_rf)],
    'Precision': [precision_score(Y_test, y_pred_log), precision_score(Y_test, y_pred_rf)],
    'Recall': [recall_score(Y_test, y_pred_log), recall_score(Y_test, y_pred_rf)],
    'F1-Score': [f1_score(Y_test, y_pred_log), f1_score(Y_test, y_pred_rf)]
}

df_metrics = pd.DataFrame(metrics)

st.subheader("📈 Performance Metrics Table")
st.dataframe(df_metrics.set_index('Model'), width='stretch')

st.subheader("📊 Visual Comparison")

tab1, tab2 = st.tabs(["Bar Chart", "Line Chart"])

with tab1:
    st.write("### Metric Comparison - Bar Chart")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics['Model']))
    width = 0.2
    
    ax.bar(x - 1.5*width, df_metrics['Accuracy'], width, label='Accuracy', color='#1f77b4')
    ax.bar(x - 0.5*width, df_metrics['Precision'], width, label='Precision', color='#ff7f0e')
    ax.bar(x + 0.5*width, df_metrics['Recall'], width, label='Recall', color='#2ca02c')
    ax.bar(x + 1.5*width, df_metrics['F1-Score'], width, label='F1-Score', color='#d62728')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_metrics['Model'])
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    st.pyplot(fig)

with tab2:
    st.write("### Metric Comparison - Line Chart")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    log_scores = [df_metrics.loc[0, m] for m in metric_names]
    rf_scores = [df_metrics.loc[1, m] for m in metric_names]
    
    ax.plot(metric_names, log_scores, marker='o', linewidth=2, markersize=8, label='Logistic Regression', color='#1f77b4')
    ax.plot(metric_names, rf_scores, marker='s', linewidth=2, markersize=8, label='Random Forest', color='#ff7f0e')
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Across Metrics', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="🏆 Best Accuracy",
        value=f"{max(df_metrics['Accuracy']):.2%}",
        delta=f"{df_metrics.loc[df_metrics['Accuracy'].idxmax(), 'Model']}"
    )

with col2:
    st.metric(
        label="🎯 Best F1-Score",
        value=f"{max(df_metrics['F1-Score']):.2%}",
        delta=f"{df_metrics.loc[df_metrics['F1-Score'].idxmax(), 'Model']}"
    )
