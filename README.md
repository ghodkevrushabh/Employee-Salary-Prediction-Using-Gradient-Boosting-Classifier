# 🧠 Employee Salary Prediction App with SHAP Explainability & Streamlit Deployment

Welcome to the **Employee Salary Prediction** project! This Machine Learning app predicts an employee’s income based on various demographic and professional features. It uses **Gradient Boosting Classifier** and offers model explainability using **SHAP (SHapley Additive exPlanations)**. The model is deployed via a **Streamlit web application** for interactive usage.

---

## 📌 Project Highlights

- 🧮 **Model:** Gradient Boosting Classifier (Scikit-learn)
- 📊 **Explainability:** SHAP for feature contribution and improvement suggestions
- 🌐 **Deployment:** Streamlit app ready for hosting on platforms like Hugging Face/Streamlit Cloud
- 📁 **Data Source:** (https://www.kaggle.com/datasets/vrushabhghodke/employee-salary-prediction-dataset)

---

## 🚀 Features

- 📈 Predicts salary classification (`<=50K` or `>50K`)
- 📋 Takes various inputs such as `education`, `occupation`, `relationship`, etc.
- **  Eliminated some features to remove outliers and reduce the bias.
- 🔍 Visualizes feature impact using SHAP plots
- 🎯 Recommends improvements to increase predicted salary
- 🧩 Handles both categorical and numerical data

---

## 🛠️ Tech Stack

| Category            | Tools Used                          |
|---------------------|-------------------------------------|
| Programming Language| Python                              |
| ML Model            | Gradient Boosting Classifier        |
| Data Processing     | Pandas, NumPy, LabelEncoder         |
| Visualization       | Matplotlib, Seaborn, SHAP           |
| Web Framework       | Streamlit                           |
| Deployment          | Streamlit Community Cloud / HF      |
| Model Export        | Joblib                              |


