# 🧠 COVID-19 Test Result Prediction (Logistic Regression)

### Author: Серік Нұртуған  
### University: Әл-Фараби атындағы Қазақ ұлттық университеті  
### Faculty: Ақпараттық технологиялар факультеті  
📅 Year: 2025  
📁 Dataset: [Kaggle — Symptoms and COVID Presence](https://www.kaggle.com/datasets/hemanthhari/symptoms-and-covid-presence)

---

## 🎯 Project Overview

This project focuses on predicting **COVID-19 test results** using a **logistic regression model** trained on the `covid.csv` dataset.  
The dataset includes various **symptoms, health conditions, and social behavior factors** that help the model determine whether a patient is likely infected with COVID-19.

The project demonstrates the complete **machine learning pipeline**, including:
- Data exploration  
- Data preprocessing  
- Model training  
- Model evaluation  
- Interactive diagnosis through a Streamlit web app.

---

## 🧩 Project Structure

```
├── app.py                 # Streamlit web application
├── covid.csv              # Dataset used for training
├── README.md              # Documentation
└── requirements.txt       # Python dependencies
```

---

## ⚙️ Installation

Make sure **Python 3.9+** is installed.

Then install all required packages:

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn
```

---

## 🚀 How to Run

1. Download all necessary files (`app.py`, `covid.csv`, and `README.md`).
2. Open the terminal and navigate to the project directory.
3. Run the Streamlit app:

```bash
streamlit run app.py
```

If the browser doesn’t open automatically, visit:  
👉 [http://localhost:8501](http://localhost:8501)

---

## 📊 Application Features

### 🧠 1. Data Exploration
- Displays the first rows of the dataset.  
- Shows the number of patients and distribution of COVID-19 results.  
- Provides summary statistics for all features.  
- Visualizes correlations between symptoms.

### 🧹 2. Data Preprocessing
- Converts categorical `Yes/No` values into numeric `1/0`.  
- Separates the target column (`COVID-19`) from features.  
- Splits data into **train/test** sets.  
- Normalizes data using `StandardScaler`.  

### 🧮 3. Model Building
A **Logistic Regression** model is trained on the following features:

- Breathing Problem  
- Fever  
- Dry Cough  
- Sore Throat  
- Running Nose  
- Asthma  
- Chronic Lung Disease  
- Headache  
- Heart Disease  
- Diabetes  
- Hyper Tension  
- Fatigue  
- Gastrointestinal Problem  
- Abroad Travel  
- Contact with COVID Patient  
- Attended Large Gathering  
- Visited Public Exposed Places  
- Family Working in Public Places  
- Wearing Masks  
- Sanitization from Market  

---

## 📈 Model Evaluation

### 🧾 Performance Metrics

| Metric | Value | Description |
|--------|--------|-------------|
| **Accuracy** | 0.969 | Overall correctness of predictions |
| **Precision (Positive)** | 0.977 | How many predicted positives were actually positive |
| **Recall (Positive)** | 0.984 | How many real positives were correctly identified |
| **F1-Score (Positive)** | 0.981 | Balance between precision and recall |
| **ROC-AUC** | 0.994 | Ability to distinguish between positive and negative cases |

---

## 🔍 Key Insights

- **Highest impact factors:**  
  `Attended Large Gathering`, `Abroad Travel`, and `Fever` — these are the most influential features for predicting infection.  
- **Negative correlation:**  
  `Running Nose` is more associated with common colds, not COVID-19.  
- **Clinical value:**  
  The model helps medical professionals optimize testing and detect likely infections earlier.

---

## 🧪 Example Diagnosis

| Scenario | Fever | Attended Large Gathering | Predicted Result |
|-----------|--------|---------------------------|------------------|
| All symptoms “No” | ❌ | ❌ | COVID-19 negative |
| Fever and attended gathering | ✅ | ✅ | ~98.9% chance of COVID-19 |

---

## 🧭 Streamlit Pages Overview

| Page | Description |
|------|--------------|
| **📊 Data Exploration** | View dataset info, class distribution, and feature correlations |
| **🔍 Diagnosis** | Enter new patient symptoms and get a COVID-19 prediction |
| **📈 Model Statistics** | Display model metrics, confusion matrix, and ROC curve |
| **📉 Feature Impact** | Shows top 10 most influential features |
| **ℹ️ Info** | Displays dataset and model details |

---

## 💡 Practical Importance

The results show that the logistic regression model achieves **high accuracy and robustness** in predicting COVID-19 outcomes.  
It can be a valuable **decision-support tool** for early detection and prioritization of testing.

### Summary of Effectiveness:
- **Accuracy:** 96.9%  
- **Recall:** 98.4%  
- **ROC-AUC:** 0.994  

### Most impactful features:
- Social & travel factors: *Attended Large Gathering*, *Abroad Travel*  
- Key symptoms: *Fever*, *Dry Cough*, *Sore Throat*  
- Less significant: *Running Nose*

---

## 🧾 Conclusion

This project successfully demonstrates how **logistic regression** can be used to predict **COVID-19 test results** with high accuracy using a combination of symptom and behavioral data.

The approach highlights:
- Simplicity of implementation  
- High interpretability of results  
- Strong generalization performance  

The developed web app can easily be adapted for other medical prediction tasks or real-time screening systems.

---

## 📚 Dataset Source

**Kaggle Dataset:**  
[Symptoms and COVID Presence](https://www.kaggle.com/datasets/hemanthhari/symptoms-and-covid-presence)

---

## 👨‍💻 Author Information

**Name:** Серік Нұртуған  
**Major:** Computer Engineering  
**Course:** 4th Year  
**University:** Әл-Фараби атындағы Қазақ ұлттық университеті  
**Year:** 2025  

---

## 🧠 License

This project is open for educational and research purposes.  
You are free to use and modify it for your own academic or personal work, with proper attribution.
