"""
Page 6 — ℹ️ About
Project overview, pipeline, tech stack, dataset info, and team.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
from utils import inject_css, hero, section, info_box, PLOTLY_LAYOUT

# ── Page config ────────────────────────────────────────────────
st.set_page_config(page_title="ℹ️ About | Stroke AI",
                   page_icon="ℹ️", layout="wide")
inject_css()

# ── Hero ───────────────────────────────────────────────────────
hero(
    "ℹ️ About This Project",
    "End-to-end stroke risk prediction using AI/ML/DL — from raw CSV to interactive web app.",
)

# ── Overview ───────────────────────────────────────────────────
section("🎯 Project Overview")
st.markdown("""
Stroke is the **second leading cause of death globally**, accounting for ~11% of all deaths.
This project builds a complete AI-powered clinical decision support tool that predicts
whether a patient is at risk of stroke based on their clinical vitals.

The project covers the full data science lifecycle:
raw data → cleaning → EDA → feature engineering → model training → evaluation → deployment.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **What this project demonstrates:**
    - Handling **highly imbalanced** medical data (SMOTE)
    - Building and comparing **7 ML + 1 DL** models
    - **Explainable AI** with SHAP values
    - Multi-page **Streamlit** deployment
    - **Plotly Dash** analytics dashboard
    """)
with col2:
    st.markdown("""
    **Clinical significance:**
    - Early prediction enables **preventive intervention**
    - Recall is prioritised over Accuracy — missing a stroke is critical
    - SHAP explanations build **clinician trust** in AI predictions
    - Composite risk score adds **interpretable** clinical value
    """)

st.markdown("<hr class='gradient-divider'>", unsafe_allow_html=True)

# ── Dataset ────────────────────────────────────────────────────
section("📦 Dataset")
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    | Property | Detail |
    |---|---|
    | **Name** | Healthcare Dataset — Stroke Prediction |
    | **Source** | [Kaggle — fedesoriano](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) |
    | **Rows** | 5,110 patients |
    | **Columns** | 12 (11 features + 1 target) |
    | **Target** | `stroke` (1 = stroke, 0 = no stroke) |
    | **Stroke Rate** | ~4.9% (highly imbalanced) |
    | **File** | `data/raw/healthcare-dataset-stroke-data.csv` |
    """)
with col2:
    st.markdown("""
    **Features:**
    - `age`, `gender`
    - `hypertension`, `heart_disease`
    - `ever_married`, `work_type`
    - `Residence_type`
    - `avg_glucose_level`
    - `bmi` *(has missing values)*
    - `smoking_status`
    """)

st.markdown("<hr class='gradient-divider'>", unsafe_allow_html=True)

# ── Pipeline ───────────────────────────────────────────────────
section("⚙️ Project Pipeline")

pipeline_steps = [
    ("1️⃣", "Data Loading & Cleaning",   "Convert 'N/A' BMI → NaN. Median imputation. Remove 1 'Other' gender row."),
    ("2️⃣", "EDA",                        "Distribution plots, correlation heatmap, stroke rate by category, scatter plots."),
    ("3️⃣", "Feature Engineering",        "age_group, risk_score (0–5), glucose_cat, bmi_cat — all derived from raw features."),
    ("4️⃣", "Encoding",                   "Label encoding for binary, One-Hot Encoding for multi-class categoricals."),
    ("5️⃣", "Train/Test Split",           "80/20 stratified split — preserving stroke class ratio."),
    ("6️⃣", "Scaling",                    "RobustScaler — robust to outliers in BMI and glucose."),
    ("7️⃣", "SMOTE",                      "Synthetic Minority Oversampling — balances training set from 4.9% → 50/50."),
    ("8️⃣", "ML Model Training",          "7 models: Logistic Regression, Decision Tree, KNN, RF, GB, XGBoost, SVM."),
    ("9️⃣", "Deep Learning",             "Keras Sequential: 256→128→64→32→1. BatchNorm + Dropout. EarlyStopping."),
    ("🔟", "Evaluation",                 "ROC-AUC, F1, Recall, Precision, Confusion Matrix, ROC Curves."),
    ("💡", "SHAP Explainability",        "Global (summary + bar) and local (waterfall) SHAP analysis."),
    ("🚀", "Deployment",                 "Streamlit multi-page app + Plotly Dash analytics dashboard."),
]

col1, col2 = st.columns(2)
for i, (num, title, desc) in enumerate(pipeline_steps):
    target = col1 if i % 2 == 0 else col2
    target.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">{num} {title}</div>
        <div class="insight-text">{desc}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr class='gradient-divider'>", unsafe_allow_html=True)

# ── Tech Stack ─────────────────────────────────────────────────
section("🛠️ Tech Stack")
stack = {
    "Data":           ["pandas", "numpy"],
    "Visualisation":  ["matplotlib", "seaborn", "plotly"],
    "ML":             ["scikit-learn", "xgboost"],
    "Deep Learning":  ["tensorflow", "keras"],
    "Imbalance":      ["imbalanced-learn (SMOTE)"],
    "Explainability": ["shap"],
    "App":            ["streamlit"],
    "Dashboard":      ["plotly dash"],
    "Model Saving":   ["joblib"],
}

cols = st.columns(3)
for i, (category, libs) in enumerate(stack.items()):
    col = cols[i % 3]
    lib_tags = "".join([f'<span class="tag">{lib}</span>' for lib in libs])
    col.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">{category}</div>
        <div style="margin-top:0.4rem;">{lib_tags}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr class='gradient-divider'>", unsafe_allow_html=True)

# ── App Guide ──────────────────────────────────────────────────
section("📱 App Navigation Guide")
pages = [
    ("🎯", "Prediction",       "01_prediction.py",      "Enter patient vitals → AI stroke risk prediction + gauge chart"),
    ("📊", "Dashboard",        "02_dashboard.py",        "KPI cards, charts, heatmaps across the full dataset"),
    ("🔍", "EDA Explorer",     "03_eda_explorer.py",     "Filter-driven interactive exploration of any feature"),
    ("📈", "Model Comparison", "04_model_comparison.py", "Compare all models with leaderboard, bar chart, radar chart"),
    ("💡", "SHAP Explainer",   "05_shap_explainer.py",   "Global & local model explainability with SHAP plots"),
    ("ℹ️",  "About",           "06_about.py",            "This page — project overview, pipeline, and team"),
]
table_df = __import__('pandas').DataFrame(pages, columns=["Icon", "Page", "File", "Description"])
st.dataframe(table_df, hide_index=True, use_container_width=True)

st.markdown("<hr class='gradient-divider'>", unsafe_allow_html=True)

# ── How to Run ─────────────────────────────────────────────────
section("▶️ How to Run")
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 1. Install dependencies")
    st.code("pip install -r requirements.txt", language="bash")

    st.markdown("#### 2. Run the notebooks")
    st.code("""cd notebooks
jupyter lab
# Run: 01 → 02 → 03 → 04 → 05 → 06""", language="bash")

with col2:
    st.markdown("#### 3. Launch Streamlit app")
    st.code("""cd app
streamlit run main.py""", language="bash")

    st.markdown("#### 4. Launch Dash dashboard")
    st.code("""python dashboard/layout.py
# Opens at http://localhost:8050""", language="bash")

st.markdown("<hr class='gradient-divider'>", unsafe_allow_html=True)

# ── Key Insights ───────────────────────────────────────────────
section("💡 Key Insights from the Data")
insights = [
    ("🎂 Age is the #1 predictor",          "Stroke risk increases sharply after 60. Patients 70+ have 10× higher stroke rate than patients under 40."),
    ("💊 Hypertension + Heart Disease",      "Patients with both conditions have ~3× higher stroke rate than those with neither."),
    ("🍬 Glucose threshold at 140 mg/dL",   "Above 140 mg/dL (diabetic range), stroke risk increases dramatically."),
    ("⚖️ BMI compounds other risks",         "BMI alone has moderate impact, but BMI > 30 combined with high glucose is a critical combination."),
    ("🚬 Smoking adds measurable risk",      "Both current and former smokers show elevated stroke risk vs never-smokers."),
    ("⚠️ SMOTE is essential",               "Without SMOTE, models predict only 'No Stroke' (trivial 95% accuracy). SMOTE enables genuine learning."),
]
c1, c2 = st.columns(2)
for i, (title, desc) in enumerate(insights):
    col = c1 if i % 2 == 0 else c2
    col.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">{title}</div>
        <div class="insight-text">{desc}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr class='gradient-divider'>", unsafe_allow_html=True)

# ── Author ─────────────────────────────────────────────────────
section("👩‍💻 Author")
st.markdown("""
<div style="text-align:center;padding:2rem;">
    <div style="font-size:3rem;">👩‍💻</div>
    <div style="font-size:1.4rem;font-weight:700;color:#667eea;margin:0.5rem 0;">Hetvi Sheth</div>
    <div style="color:#aaa;">B.Tech — Data Science & AI</div>
    <div style="color:#aaa;">Navrachana University</div>
</div>""", unsafe_allow_html=True)

# ── Disclaimer ─────────────────────────────────────────────────
st.markdown("""
<div class="warning-box">
    ⚠️ <b>Disclaimer:</b> This application is built for <b>educational and academic purposes only</b>.
    It is NOT intended for clinical diagnosis or medical advice.
    Always consult a <b>qualified healthcare professional</b> for any health-related decisions.
</div>""", unsafe_allow_html=True)
