# 🧠 Stroke Prediction — End-to-End AI/ML/DL Project

> **An end-to-end machine learning and deep learning project for predicting stroke risk using clinical patient data — featuring EDA, 7 ML models, a Neural Network, SHAP explainability, an interactive Streamlit web app, and a Plotly Dash analytics dashboard.**

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Pipeline](#-pipeline)
- [Models & Results](#-models--results)
- [Dashboards & App](#-dashboards--app)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Key Insights](#-key-insights)
- [Disclaimer](#-disclaimer)

---

## 🔍 Overview

Stroke is the **second leading cause of death globally**, accounting for approximately 11% of total deaths. Early identification of high-risk patients can enable preventive intervention and save lives.

This project builds a complete **AI-powered stroke risk prediction system** that:

- Analyzes **5,110 patient records** with 10 clinical features
- Handles **class imbalance** (only ~4.9% stroke cases) using SMOTE
- Trains and compares **7 Machine Learning models** and **1 Deep Learning model**
- Provides **explainable predictions** using SHAP values
- Deploys as a **Streamlit multi-page web app** for clinical use
- Visualizes insights through a **Plotly Dash analytics dashboard**

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Name** | Healthcare Dataset — Stroke Data |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) |
| **Rows** | 5,110 patients |
| **Target** | `stroke` (1 = had stroke, 0 = no stroke) |
| **Stroke Rate** | ~4.9% (highly imbalanced) |
| **Location** | `data/raw/healthcare-dataset-stroke-data.csv` |

### 📋 Features

| Feature | Type | Description |
|---|---|---|
| `id` | int | Unique patient ID (dropped) |
| `gender` | categorical | Male / Female / Other |
| `age` | float | Patient age in years |
| `hypertension` | binary | 0 = No, 1 = Yes |
| `heart_disease` | binary | 0 = No, 1 = Yes |
| `ever_married` | categorical | Yes / No |
| `work_type` | categorical | Private / Self-employed / Govt_job / children / Never_worked |
| `Residence_type` | categorical | Urban / Rural |
| `avg_glucose_level` | float | Average blood glucose level (mg/dL) |
| `bmi` | float | Body Mass Index (has missing values → imputed) |
| `smoking_status` | categorical | never smoked / formerly smoked / smokes / Unknown |

---

## 📁 Project Structure

```
stroke_prediction/
│
├── 📄 README.md                        ← You are here
├── 📄 requirements.txt                 ← Python dependencies
├── 📄 config.yaml                      ← Global config (paths, hyperparams)
├── 📄 .gitignore
│
├── 📁 data/
│   ├── raw/
│   │   └── healthcare-dataset-stroke-data.csv
│   └── processed/                      ← Cleaned & encoded CSVs (auto-generated)
│
├── 📁 notebooks/                       ← Step-by-step Jupyter notebooks
│   ├── stroke_prediction.ipynb         ← Full combined notebook
│   ├── 01_eda.ipynb                    ← Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb         ← Cleaning & Feature Engineering
│   ├── 03_ml_models.ipynb             ← 7 Machine Learning Models
│   ├── 04_deep_learning.ipynb         ← Keras Neural Network
│   ├── 05_shap_explainability.ipynb   ← SHAP Explainability
│   └── 06_model_comparison.ipynb      ← Final Leaderboard
│
├── 📁 src/                             ← Reusable Python source modules
│   ├── preprocessing/
│   │   ├── cleaner.py                  ← Missing values, outlier handling
│   │   ├── feature_engineering.py     ← Age groups, risk score, categories
│   │   └── encoder.py                  ← Label encoding & one-hot encoding
│   ├── training/
│   │   ├── train_ml.py                 ← Train all ML models with SMOTE
│   │   └── train_dl.py                 ← Train Keras Neural Network
│   ├── evaluation/
│   │   ├── metrics.py                  ← AUC, F1, Recall, Precision
│   │   └── visualizer.py              ← ROC curves, confusion matrices
│   └── utils/
│       ├── helpers.py                  ← Shared utility functions
│       └── logger.py                   ← Logging setup
│
├── 📁 models/
│   ├── saved/                          ← Trained model files (.pkl, .keras)
│   └── metrics/
│       └── results.csv                 ← Model performance comparison table
│
├── 📁 plots/
│   ├── eda/                            ← EDA visualizations (PNG)
│   ├── models/                         ← ROC curves, confusion matrices (PNG)
│   └── shap/                           ← SHAP summary, bar, waterfall plots
│
├── 📁 app/                             ← Streamlit multi-page web app
│   ├── main.py                         ← Entry point
│   └── pages/
│       ├── 01_prediction.py            ← Patient risk predictor
│       ├── 02_dashboard.py             ← Overview analytics dashboard
│       ├── 03_eda_explorer.py          ← Interactive EDA explorer
│       ├── 04_model_comparison.py      ← Compare all models
│       ├── 05_shap_explainer.py        ← Explain individual predictions
│       └── 06_about.py                 ← Project info & team
│
├── 📁 dashboard/                       ← Plotly Dash standalone dashboard
│   ├── layout.py                       ← Overall page layout
│   ├── callbacks.py                    ← Interactivity & callback logic
│   ├── components/
│   │   ├── charts.py                   ← Chart components
│   │   ├── cards.py                    ← KPI / stat cards
│   │   └── filters.py                  ← Dropdowns, sliders, filters
│   └── assets/
│       └── custom.css                  ← Dashboard custom styling
│
├── 📁 reports/
│   ├── project_report.md               ← Full project write-up
│   └── model_summary.md                ← Model performance summary
│
└── 📁 tests/
    ├── test_preprocessing.py
    └── test_model.py
```

---

## 🛠 Tech Stack

| Category | Libraries / Tools |
|---|---|
| **Language** | Python 3.11 |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **ML Models** | Scikit-learn, XGBoost |
| **Deep Learning** | TensorFlow / Keras |
| **Imbalance Handling** | imbalanced-learn (SMOTE) |
| **Explainability** | SHAP |
| **Web App** | Streamlit |
| **Dashboard** | Plotly Dash |
| **Model Persistence** | Joblib |
| **Notebook** | JupyterLab |

---

## ⚙️ Pipeline

```
Raw CSV
  │
  ▼
1. Data Cleaning
   ├── Convert BMI 'N/A' → NaN
   ├── Median imputation for BMI
   └── Remove 'Other' gender (1 row)
  │
  ▼
2. Feature Engineering
   ├── age_group     (Child / Teen / YoungAdult / MiddleAge / Senior / Elderly)
   ├── risk_score    (0–5 composite score)
   ├── glucose_cat   (Low / Normal / Prediabetic / Diabetic / VeryHigh)
   └── bmi_cat       (Underweight / Normal / Overweight / Obese)
  │
  ▼
3. Encoding
   ├── Binary encoding  (gender, ever_married, Residence_type)
   └── One-Hot encoding (work_type, smoking_status, age_group, glucose_cat, bmi_cat)
  │
  ▼
4. Train / Test Split  (80% / 20%, stratified)
  │
  ▼
5. Feature Scaling     (RobustScaler — handles outliers)
  │
  ▼
6. SMOTE               (balance training set: ~4.9% → 50/50)
  │
  ▼
7. Model Training
   ├── ML:  Logistic Regression, Decision Tree, KNN,
   │        Random Forest, Gradient Boosting, XGBoost, SVM
   └── DL:  Keras Neural Network (256→128→64→32→1)
  │
  ▼
8. Evaluation
   ├── Accuracy, F1, Precision, Recall, ROC-AUC
   ├── Confusion Matrix
   └── ROC Curves
  │
  ▼
9. SHAP Explainability
   ├── Summary Plot
   ├── Bar Plot (feature importance)
   └── Waterfall Plot (individual patient)
  │
  ▼
10. Save & Deploy
    ├── models/saved/best_ml_model.pkl
    ├── models/saved/scaler.pkl
    └── models/saved/stroke_nn_model.keras
```

---

## 🤖 Models & Results

> Results are populated after running the notebooks.

| Model | Accuracy | F1 Score | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Decision Tree | — | — | — | — |
| K-Nearest Neighbors | — | — | — | — |
| Random Forest | — | — | — | — |
| Gradient Boosting | — | — | — | — |
| **XGBoost** | — | — | — | **—** |
| SVM | — | — | — | — |
| Neural Network (Keras) | — | — | — | — |

> 📌 **Note:** For stroke prediction, **Recall** and **ROC-AUC** are prioritized over Accuracy due to class imbalance — missing a stroke case is far more costly than a false positive.

---

## 🌐 Dashboards & App

### Streamlit Web App — `app/`
A **multi-page interactive application** for clinical use.

| Page | Description |
|---|---|
| 🎯 Prediction | Enter patient vitals → get stroke risk % + gauge chart |
| 📊 Dashboard | Overview analytics with KPI cards |
| 🔍 EDA Explorer | Interact with charts and filter data |
| 📈 Model Comparison | View all model metrics and ROC curves |
| 💡 SHAP Explainer | Understand why a patient is high-risk |
| ℹ️ About | Project info and tech stack |

**Run the app:**
```bash
cd app
streamlit run main.py
```

---

### Plotly Dash Dashboard — `dashboard/`
A **standalone analytics dashboard** with rich interactive charts, KPI cards, and filter controls.

| Component | Description |
|---|---|
| KPI Cards | Total patients, stroke cases, stroke rate, avg age |
| Age vs Stroke Chart | Bar chart of stroke rate by age group |
| Glucose Distribution | Histogram by stroke status |
| Feature Filters | Filter by gender, age range, hypertension, etc. |
| Correlation Heatmap | Feature-level correlation matrix |

**Run the dashboard:**
```bash
python dashboard/layout.py
```

---

## 🔧 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/stroke-prediction.git
cd stroke-prediction/stroke_prediction
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1 — Run the notebooks (trains & saves models)
```bash
jupyter lab notebooks/
```
Run notebooks in order: `01` → `02` → `03` → `04` → `05` → `06`

### Step 2 — Launch the Streamlit app
```bash
cd app
streamlit run main.py
```
Opens at: `http://localhost:8501`

### Step 3 — Launch the Plotly Dash dashboard
```bash
python dashboard/layout.py
```
Opens at: `http://localhost:8050`

---

## 🚀 Deployment & Hosting

### 1. Streamlit Community Cloud (Free)
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **New app**.
4. Select the repository, branch `main`, and set Main file path to `app.py`.
5. Click **Deploy**.

### 2. Docker (AWS / GCP / Local)
This project includes a `Dockerfile` and `docker-compose.yml`.
```bash
docker-compose up --build
```
The app will run at `http://localhost:8501`.

### 3. Heroku / Render
This project includes a `Procfile` and `setup.sh` for easy deployment on Heroku or Render. Simply link your GitHub repo to their service, and they will detect the Python environment and run the setup automatically.

---

## 💡 Key Insights

- **Age** is the single strongest predictor — stroke risk rises sharply after 60
- Patients with **both hypertension and heart disease** have ~3× higher stroke rate
- **Glucose > 140 mg/dL** (diabetic range) is a critical risk threshold
- **BMI > 30** (obesity) combined with high glucose creates compounding risk
- **Smoking history** has moderate but consistent impact on risk
- The dataset is **severely imbalanced** — SMOTE is essential for fair model training
- **Recall** matters most clinically: missing a true stroke is far worse than a false alarm

---

## ⚠️ Disclaimer

> This project is built for **educational and academic purposes only**.
> The predictions made by these models are **not intended for clinical diagnosis**.
> Always consult a **qualified medical professional** for health decisions.

---

## 👩‍💻 Author

**Hetvi Sheth**  
B.Tech — Data Science & AI  
Navrachana University  

---

*Built with ❤️ using Python, Scikit-learn, TensorFlow, SHAP, Streamlit & Plotly Dash*
