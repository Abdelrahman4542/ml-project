# 🔬 Breast Cancer Prediction App

## 📌 Overview

This project is an interactive machine learning web application that predicts whether a breast tumor is **Malignant** or **Benign** using multiple models.

The application is built using **Streamlit** and trained on the Breast Cancer Wisconsin Dataset.

---

## 🚀 Features

* 📊 Dataset preview and class distribution visualization
* 🔥 Feature correlation heatmap
* 🤖 Multiple ML models:

  * Logistic Regression (with scaling)
  * Random Forest Classifier
* 📈 Model performance comparison (Accuracy, Precision, Recall, F1-score)
* 🧑‍⚕️ Single patient prediction
* 📂 Batch prediction using CSV upload
* 📥 Download prediction results

---

## 🧠 Models Used

* Logistic Regression (with StandardScaler)
* Random Forest Classifier

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Seaborn
* Matplotlib

---

## 📂 Project Structure

```bash
ml-project/
│
├── ml_project_ui.py
├── pages/
│   └── Model_Comparison.py
├── Breast Cancer Wisconsin Dataset.csv
├── sample_upload_template.csv
├── requirements.txt
└── .streamlit/
    └── config.toml
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run ml_project_ui.py
```

Then open:

```
http://localhost:8501
```

---

## 📊 Dataset

The model is trained on the **Breast Cancer Wisconsin Dataset**, which contains features extracted from digitized images of breast masses.

---

## 📌 Key Functionality

* Predict tumor type based on input features
* Compare model performance
* Visualize data relationships
* Process multiple patients at once

---

## 🎯 Use Case

This application demonstrates how machine learning can assist in medical diagnosis by providing fast and data-driven predictions.

---


