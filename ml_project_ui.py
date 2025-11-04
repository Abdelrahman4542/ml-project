import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from io import BytesIO

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Breast Cancer Prediction App")
st.markdown("""
<style>
    .main-header {
        font-size: 20px;
        color: #1f77b4;
        padding: 10px 0;
    }
    .stAlert {
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.write("This app predicts whether a breast tumor is **Malignant** or **Benign** using machine learning models.")

@st.cache_data
def load_data():
    data = pd.read_csv("Breast Cancer Wisconsin Dataset.csv")
    data['label'] = data['diagnosis'].map({'M': 0, 'B': 1})
    X = data.drop(columns=['label', 'id', 'diagnosis'], axis=1)
    Y = data['label']
    return X, Y, data

X, Y, full_data = load_data()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

@st.cache_resource
def train_models():
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, Y_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, Y_train)
    
    return log_model, rf_model

log_model, rf_model = train_models()

st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

selected_model = st.sidebar.selectbox(
    "🤖 Select Model for Predictions",
    ["Logistic Regression", "Random Forest"],
    help="Choose which model to use for single and batch predictions"
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Current Model:** {selected_model}")
st.sidebar.markdown("""
### 📌 About
This application uses machine learning to classify breast tumors.
- **M**: Malignant (Cancerous)
- **B**: Benign (Non-cancerous)
""")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📊 Dataset Overview")
    st.write(f"**Total Samples:** {len(full_data)}")
    st.dataframe(full_data.head(10), use_container_width=True)

with col2:
    st.header("📈 Class Distribution")
    diagnosis_counts = full_data['diagnosis'].value_counts()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#ff6b6b', '#4ecdc4']
    ax.pie(diagnosis_counts, labels=['Benign', 'Malignant'], autopct='%1.1f%%', 
           colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
    ax.set_title('Diagnosis Distribution', fontsize=14, fontweight='bold', pad=20)
    st.pyplot(fig)

st.divider()

st.header("🔥 Feature Correlation Heatmap")
st.write("Visualizing correlations between different features in the dataset")

fig, ax = plt.subplots(figsize=(16, 12))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
st.pyplot(fig)

st.divider()

st.header("⚡ Model Performance")

col1, col2 = st.columns(2)

y_pred_log = log_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test)

with col1:
    st.subheader("🤖 Logistic Regression")
    st.metric("Accuracy", f"{accuracy_score(Y_test, y_pred_log):.2%}")
    st.metric("Precision", f"{precision_score(Y_test, y_pred_log):.2%}")
    st.metric("Recall", f"{recall_score(Y_test, y_pred_log):.2%}")
    st.metric("F1-Score", f"{f1_score(Y_test, y_pred_log):.2%}")

with col2:
    st.subheader("🌲 Random Forest")
    st.metric("Accuracy", f"{accuracy_score(Y_test, y_pred_rf):.2%}")
    st.metric("Precision", f"{precision_score(Y_test, y_pred_rf):.2%}")
    st.metric("Recall", f"{recall_score(Y_test, y_pred_rf):.2%}")
    st.metric("F1-Score", f"{f1_score(Y_test, y_pred_rf):.2%}")

st.divider()

st.header("🧑‍⚕️ Single Patient Prediction")

with st.expander("📝 Enter Patient Data for Prediction", expanded=False):
    st.write(f"Using **{selected_model}** for prediction")
    
    cols = st.columns(3)
    features = []
    
    for idx, col_name in enumerate(X.columns):
        with cols[idx % 3]:
            value = st.number_input(
                f"{col_name}", 
                float(X[col_name].min()), 
                float(X[col_name].max()), 
                float(X[col_name].mean()),
                key=f"input_{col_name}"
            )
            features.append(value)
    
    if st.button("🔍 Make Prediction", type="primary"):
        input_data = np.array(features).reshape(1, -1)
        
        if selected_model == "Logistic Regression":
            input_data_scaled = scaler.transform(input_data)
            prediction = log_model.predict(input_data_scaled)[0]
            probability = log_model.predict_proba(input_data_scaled)[0]
        else:
            prediction = rf_model.predict(input_data)[0]
            probability = rf_model.predict_proba(input_data)[0]
        
        result = "Benign" if prediction == 1 else "Malignant"
        confidence = probability[prediction] * 100
        
        if result == "Benign":
            st.success(f"### ✅ Prediction: **{result}**")
            st.info(f"Confidence: **{confidence:.2f}%**")
        else:
            st.error(f"### ⚠️ Prediction: **{result}**")
            st.warning(f"Confidence: **{confidence:.2f}%**")

st.divider()

st.header("📂 Batch Prediction from CSV")
st.write(f"Upload a CSV file with patient data to get predictions using **{selected_model}**")

uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], help="Upload CSV with same features as training data")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("### 📊 Uploaded Data Preview")
        st.dataframe(batch_data.head(), use_container_width=True)
        
        required_columns = X.columns.tolist()
        missing_columns = [col for col in required_columns if col not in batch_data.columns]
        
        if missing_columns:
            st.error(f"❌ Missing columns: {', '.join(missing_columns)}")
        else:
            batch_features = batch_data[required_columns]
            
            if st.button("🚀 Run Batch Predictions", type="primary"):
                if selected_model == "Logistic Regression":
                    batch_scaled = scaler.transform(batch_features)
                    predictions = log_model.predict(batch_scaled)
                    probabilities = log_model.predict_proba(batch_scaled)
                else:
                    predictions = rf_model.predict(batch_features)
                    probabilities = rf_model.predict_proba(batch_features)
                
                results_df = batch_data.copy()
                results_df['Prediction'] = ['Benign' if p == 1 else 'Malignant' for p in predictions]
                results_df['Confidence'] = [f"{max(prob)*100:.2f}%" for prob in probabilities]
                results_df['Prediction_Code'] = predictions
                
                st.write("### 📊 Prediction Results")
                st.dataframe(results_df, use_container_width=True)
                
                st.write(f"**Summary:** {len(results_df)} predictions made")
                benign_count = sum(predictions == 1)
                malignant_count = sum(predictions == 0)
                st.write(f"- **Benign:** {benign_count} ({benign_count/len(predictions)*100:.1f}%)")
                st.write(f"- **Malignant:** {malignant_count} ({malignant_count/len(predictions)*100:.1f}%)")
                
                csv_buffer = BytesIO()
                results_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_buffer,
                    file_name="breast_cancer_predictions.csv",
                    mime="text/csv",
                    type="primary"
                )
                
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

st.divider()
st.markdown("---")
st.caption("💡 Tip: Visit the **Model Comparison** page to see detailed performance visualizations!")
