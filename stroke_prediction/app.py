"""
🧠 Stroke Risk Predictor — Streamlit Web App
Beautiful interactive app using the trained ML model
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import os

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="🧠 Stroke Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

    .hero-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102,126,234,0.4);
    }
    .hero-card h1 { font-size: 2.8rem; font-weight: 700; margin: 0; }
    .hero-card p  { font-size: 1.1rem; margin-top: 0.5rem; opacity: 0.9; }

    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #667eea; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 0.2rem; }

    .risk-low  { background: linear-gradient(135deg, #11998e, #38ef7d); color: white; }
    .risk-med  { background: linear-gradient(135deg, #f7971e, #ffd200); color: white; }
    .risk-high { background: linear-gradient(135deg, #cb2d3e, #ef473a); color: white; }

    .risk-box {
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #667eea;
        padding: 0.5rem 0;
        border-bottom: 2px solid #667eea;
        margin: 1.5rem 0 1rem 0;
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(102,126,234,0.4);
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102,126,234,0.6); }

    .sidebar .sidebar-content { background: #1a1a2e; }
    .info-box {
        background: rgba(102,126,234,0.15);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: rgba(239,71,58,0.15);
        border-left: 4px solid #ef473a;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "models/best_ml_model.pkl"
    scaler_path = "models/scaler.pkl"
    features_path = "models/feature_names.json"

    if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
        return None, None, None

    model   = joblib.load(model_path)
    scaler  = joblib.load(scaler_path)
    with open(features_path) as f:
        features = json.load(f)
    return model, scaler, features


@st.cache_data
def load_data():
    df = pd.read_csv('data/raw/healthcare-dataset-stroke-data.csv')
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df = df[df['gender'] != 'Other']
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['age'] = df['age'].astype(int)
    return df


def preprocess_input(input_dict, feature_names):
    """Preprocess user input to match training features"""
    df_input = pd.DataFrame([input_dict])

    # Feature Engineering (same as training)
    df_input['age_group'] = pd.cut(df_input['age'],
        bins=[0,12,17,35,55,65,120],
        labels=['Child','Teen','YoungAdult','MiddleAge','Senior','Elderly'])
    df_input['risk_score'] = (
        df_input['hypertension'] +
        df_input['heart_disease'] +
        (df_input['avg_glucose_level'] > 140).astype(int) +
        (df_input['bmi'] > 30).astype(int) +
        (df_input['age'] > 60).astype(int)
    )
    df_input['glucose_cat'] = pd.cut(df_input['avg_glucose_level'],
        bins=[0,70,100,125,200,1000],
        labels=['Low','Normal','Prediabetic','Diabetic','VeryHigh'])
    df_input['bmi_cat'] = pd.cut(df_input['bmi'],
        bins=[0,18.5,25,30,100],
        labels=['Underweight','Normal','Overweight','Obese'])

    # Encode
    from sklearn.preprocessing import LabelEncoder
    df_input['gender']         = 1 if input_dict['gender'] == 'Male' else 0
    df_input['ever_married']   = 1 if input_dict['ever_married'] == 'Yes' else 0
    df_input['Residence_type'] = 1 if input_dict['Residence_type'] == 'Urban' else 0

    df_input = pd.get_dummies(df_input,
        columns=['work_type','smoking_status','age_group','glucose_cat','bmi_cat'],
        drop_first=True)

    bool_cols = df_input.select_dtypes(include='bool').columns
    df_input[bool_cols] = df_input[bool_cols].astype(int)

    # Align to training features
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_names]

    return df_input


# ── Main App ───────────────────────────────────────────────────
def main():
    model, scaler, feature_names = load_model()

    # ── HERO SECTION ───────────────────────────────────────────
    st.markdown("""
    <div class="hero-card">
        <h1>🧠 Stroke Risk Predictor</h1>
        <p>AI-powered clinical decision support system using Machine Learning & Deep Learning</p>
        <small>⚠️ For educational purposes only. Always consult a medical professional.</small>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 👤 Patient Information")
        st.markdown("---")

        st.markdown("### 📋 Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age    = st.slider("Age", 1, 82, 45)
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        residence    = st.selectbox("Residence Type", ["Urban", "Rural"])
        work_type    = st.selectbox("Work Type",
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])

        st.markdown("### 🏥 Medical History")
        hypertension  = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        smoking       = st.selectbox("Smoking Status",
            ["never smoked", "formerly smoked", "smokes", "Unknown"])

        st.markdown("### 🔬 Lab Values")
        glucose = st.slider("Avg Glucose Level (mg/dL)", 50.0, 300.0, 100.0, 0.5)
        bmi     = st.slider("BMI", 10.0, 70.0, 25.0, 0.1)

        st.markdown("---")
        predict_btn = st.button("🔍 Predict Stroke Risk", use_container_width=True)

    # ── TABS ───────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Data Analysis", "ℹ️ About", "📈 Advanced Analytics"])

    # ── TAB 1: PREDICTION ──────────────────────────────────────
    with tab1:
        col1, col2, col3 = st.columns(3)

        # Risk indicators
        risk_score = (
            (hypertension == "Yes") +
            (heart_disease == "Yes") +
            (glucose > 140) +
            (bmi > 30) +
            (age > 60)
        )
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{'⚠️' if hypertension=='Yes' else '✅'}</div>
                <div class="metric-label">Hypertension</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{'⚠️' if heart_disease=='Yes' else '✅'}</div>
                <div class="metric-label">Heart Disease</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {'#ef473a' if risk_score>=3 else '#f7971e' if risk_score>=2 else '#38ef7d'}">{risk_score}/5</div>
                <div class="metric-label">Risk Score</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        if predict_btn:
            if model is None:
                st.error("⚠️ Model not found! Please run the Jupyter notebook first to train and save the model.")
                st.info("📓 Run `stroke_prediction.ipynb` to train models and save them to the `models/` folder.")
            else:
                with st.spinner("🔄 Analyzing patient data..."):
                    input_dict = {
                        'gender': gender, 'age': age,
                        'hypertension': 1 if hypertension=="Yes" else 0,
                        'heart_disease': 1 if heart_disease=="Yes" else 0,
                        'ever_married': ever_married,
                        'work_type': work_type,
                        'Residence_type': residence,
                        'avg_glucose_level': glucose,
                        'bmi': bmi,
                        'smoking_status': smoking
                    }

                    try:
                        X_input = preprocess_input(input_dict, feature_names)
                        X_scaled = scaler.transform(X_input)
                        prob = model.predict_proba(X_scaled)[0][1]
                        pred = int(prob >= 0.4)

                        # Result Display
                        if prob < 0.3:
                            risk_class = "risk-low"
                            risk_emoji = "✅"
                            risk_label = "LOW RISK"
                            advice = "Maintain your healthy lifestyle! Regular check-ups recommended."
                        elif prob < 0.6:
                            risk_class = "risk-med"
                            risk_emoji = "⚠️"
                            risk_label = "MODERATE RISK"
                            advice = "Consider lifestyle modifications. Consult a physician."
                        else:
                            risk_class = "risk-high"
                            risk_emoji = "🚨"
                            risk_label = "HIGH RISK"
                            advice = "Seek immediate medical attention. Multiple risk factors detected."

                        st.markdown(f"""
                        <div class="risk-box {risk_class}">
                            {risk_emoji} {risk_label}<br>
                            <span style="font-size:2.5rem;">{prob*100:.1f}%</span><br>
                            <small style="font-size:1rem; font-weight:400;">Stroke Probability</small>
                        </div>""", unsafe_allow_html=True)

                        st.markdown(f'<div class="info-box">💡 <b>Recommendation:</b> {advice}</div>',
                                    unsafe_allow_html=True)

                        # Gauge Chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=prob * 100,
                            number={'suffix': "%", 'font': {'size': 40}},
                            delta={'reference': 5, 'suffix': '% (baseline)'},
                            title={'text': "Stroke Risk Probability", 'font': {'size': 20}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1},
                                'bar': {'color': "#667eea"},
                                'steps': [
                                    {'range': [0, 30],   'color': "#1a9e3f"},
                                    {'range': [30, 60],  'color': "#f7971e"},
                                    {'range': [60, 100], 'color': "#cb2d3e"}
                                ],
                                'threshold': {
                                    'line': {'color': "white", 'width': 4},
                                    'thickness': 0.75,
                                    'value': prob * 100
                                }
                            }
                        ))
                        fig.update_layout(
                            height=350,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': 'white'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Patient Summary
                        st.markdown('<div class="section-header">📋 Patient Summary</div>',
                                    unsafe_allow_html=True)
                        summary_data = {
                            "Feature": ["Age", "BMI", "Glucose Level", "Hypertension",
                                        "Heart Disease", "Smoking", "Risk Score"],
                            "Value": [f"{age} yrs",
                                      f"{bmi:.1f} {'(Obese)' if bmi>30 else '(Normal)' if bmi>18.5 else '(Under)'}",
                                      f"{glucose:.0f} mg/dL {'(High)' if glucose>140 else '(Normal)'}",
                                      "Yes ⚠️" if hypertension=="Yes" else "No ✅",
                                      "Yes ⚠️" if heart_disease=="Yes" else "No ✅",
                                      smoking,
                                      f"{risk_score}/5 {'🔴' if risk_score>=3 else '🟡' if risk_score>=2 else '🟢'}"]
                        }
                        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        st.info("Make sure to run the notebook first to generate the model files.")

        else:
            st.markdown("""
            <div class="info-box">
                👈 <b>Fill in patient details in the sidebar</b> and click <b>Predict Stroke Risk</b> to get the analysis.
            </div>
            """, unsafe_allow_html=True)

            # Feature importance info
            st.markdown('<div class="section-header">🔑 Key Stroke Risk Factors</div>',
                        unsafe_allow_html=True)
            factors = {
                "🎂 Age > 60": "Older patients have significantly higher stroke risk",
                "❤️ Heart Disease": "Pre-existing cardiac conditions double the risk",
                "💊 Hypertension": "High blood pressure is the #1 modifiable risk factor",
                "🍬 High Glucose": "Diabetic glucose levels (>125 mg/dL) increase risk",
                "⚖️ Obesity (BMI>30)": "Higher BMI correlates with elevated stroke risk",
                "🚬 Smoking": "Active/former smokers face greater cardiovascular risk"
            }
            for factor, desc in factors.items():
                st.markdown(f"**{factor}** — {desc}")

    # ── TAB 2: DATA ANALYSIS ───────────────────────────────────
    with tab2:
        try:
            df = load_data()
            df_chart = load_data()

            st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

            # --- KPIs ---
            total_patients = len(df_chart)
            avg_age = df_chart["age"].mean()
            avg_bmi = df_chart["bmi"].mean()
            stroke_cases = df_chart["stroke"].sum()
            avg_glucose = df_chart["avg_glucose_level"].mean()
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Patients", f"{total_patients/1000:.3f}K" if total_patients >= 1000 else f"{total_patients}")
            c2.metric("Average Age (Years)", f"{avg_age:.2f}")
            c3.metric("Average BMI", f"{avg_bmi:.2f}")
            c4.metric("Total Stroke Cases", f"{stroke_cases}")
            c5.metric("Average Glucose Level", f"{avg_glucose:.2f}")
            
            st.markdown("---")
            
            # --- ROW 1 CHARTS ---
            r1c1, r1c2, r1c3 = st.columns(3)
            
            with r1c1:
                stroke_by_age = df_chart.groupby('age')['stroke'].sum().reset_index()
                fig1 = px.bar(stroke_by_age, x='age', y='stroke', title="Stroke Cases by Age",
                              labels={'age': 'age', 'stroke': 'Sum of stroke'}, color_discrete_sequence=['#0078D4'])
                fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=20, l=20, r=20))
                st.plotly_chart(fig1, use_container_width=True)
                
            with r1c2:
                df_rounded = df_chart.copy()
                df_rounded['bmi_r'] = df_rounded['bmi'].round(0)
                df_rounded['glucose_r'] = df_rounded['avg_glucose_level'].round(0)
                grouped = df_rounded.groupby(['bmi_r', 'glucose_r']).agg(count=('stroke', 'size')).reset_index()
                fig2 = px.scatter(grouped, x='bmi_r', y='glucose_r', size='count', title="BMI vs Average Glucose Level",
                                  labels={'bmi_r': 'bmi', 'glucose_r': 'Average of avg_glucose_level'}, color_discrete_sequence=['#0078D4'])
                fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=20, l=20, r=20))
                st.plotly_chart(fig2, use_container_width=True)
                
            with r1c3:
                trend = df_chart.groupby('age')['stroke'].count().reset_index()
                fig3 = px.line(trend, x='age', y='stroke', title="Stroke Trend Across Age",
                               labels={'age': 'age', 'stroke': 'Count of stroke'}, color_discrete_sequence=['#0078D4'])
                fig3.update_traces(line=dict(width=3))
                fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=20, l=20, r=20))
                st.plotly_chart(fig3, use_container_width=True)
                
            # --- ROW 2 CHARTS ---
            r2c1, r2c2, r2c3 = st.columns(3)
            
            with r2c1:
                gender_counts = df_chart['gender'].value_counts().reset_index()
                gender_counts.columns = ['gender', 'count']
                gender_counts['gender'] = gender_counts['gender'].astype(str)
                fig4 = px.pie(gender_counts, values='count', names='gender', hole=0.6, title="Stroke Distribution by Gender",
                              color_discrete_sequence=['#0078D4', '#002050', '#83B4FF'])
                fig4.update_traces(textinfo='value+percent')
                fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=20, l=20, r=20))
                st.plotly_chart(fig4, use_container_width=True)
                
            with r2c2:
                hyper_stroke = df_chart.groupby('hypertension')['stroke'].sum().reset_index()
                hyper_stroke['hypertension'] = hyper_stroke['hypertension'].astype(str)
                fig5 = px.bar(hyper_stroke, x='stroke', y='hypertension', orientation='h', title="Stroke Cases by Hypertension Status",
                              labels={'hypertension': 'hypertension', 'stroke': 'Sum of stroke'}, color_discrete_sequence=['#0078D4'])
                fig5.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=20, l=20, r=20))
                st.plotly_chart(fig5, use_container_width=True)
                
            with r2c3:
                stroke_df = df_chart[df_chart['stroke'] == 1].copy()
                if not stroke_df.empty:
                    stroke_df['age_bin'] = pd.cut(stroke_df['age'], bins=[0,40,60,80,100], labels=['0-40', '41-60', '61-80', '81+'])
                    grouped_tree = stroke_df.groupby(['smoking_status', 'age_bin'], observed=False).size().reset_index(name='count')
                    grouped_tree = grouped_tree[grouped_tree['count'] > 0]
                    fig6 = px.sunburst(grouped_tree, path=['smoking_status', 'age_bin'], values='count', title="Stroke Risk Factor Analysis",
                                       color_discrete_sequence=['#0078D4', '#5C9CE6', '#99C3F0', '#002050'])
                    fig6.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=20, l=20, r=20))
                    st.plotly_chart(fig6, use_container_width=True)

        except Exception as e:
            st.error(f"Could not load data: {e}")

    # ── TAB 3: ABOUT ───────────────────────────────────────────
    with tab3:
        st.markdown("""
        ## 🧠 About This Project

        This is a **complete end-to-end machine learning project** for stroke prediction.

        ### 📊 Dataset
        - **Source:** Kaggle Healthcare Stroke Dataset
        - **Size:** 5,110 patients
        - **Features:** Age, BMI, Glucose, Hypertension, Heart Disease, Smoking, etc.
        - **Target:** Binary classification (Stroke / No Stroke)

        ### 🔧 Technical Pipeline
        | Step | Method |
        |------|--------|
        | Preprocessing | BMI imputation, Label Encoding, One-Hot Encoding |
        | Feature Engineering | Age groups, Risk score, Glucose/BMI categories |
        | Imbalance Handling | SMOTE (Synthetic Minority Oversampling) |
        | Scaling | RobustScaler (handles outliers) |
        | ML Models | Logistic Regression, Decision Tree, KNN, Random Forest, Gradient Boosting, XGBoost, SVM |
        | Deep Learning | 4-layer Keras Neural Network + BatchNorm + Dropout |
        | Explainability | SHAP (SHapley Additive exPlanations) |

        ### 📈 Key Insights
        - Stroke risk **increases dramatically after age 60**
        - **Hypertension + Heart Disease** together are the strongest predictors
        - **High glucose (>140 mg/dL)** is a major risk indicator
        - Dataset is **highly imbalanced** (~95% non-stroke) — SMOTE is critical

        ### ⚠️ Disclaimer
        > This tool is for **educational and research purposes only**.
        > It is NOT a substitute for professional medical diagnosis.
        > Always consult a qualified healthcare professional for medical advice.
        """)

    # ── TAB 4: ADVANCED ANALYTICS ───────────────────────────────
    with tab4:
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            import numpy as np

            df_aa = load_data()
            st.markdown('<div class="section-header">📈 Advanced Statistical Analytics</div>', unsafe_allow_html=True)

            # Feature Correlation
            st.subheader("1. Feature Correlation Heatmap")
            st.write("This heatmap shows how different numerical features correlate with each other. A higher magnitude (positive or negative) indicates a stronger relationship.")
            
            numeric_df = df_aa.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr, vmin=-1, vmax=1)
            st.pyplot(fig_corr)

            st.markdown("---")

            # KDE Plots
            st.subheader("2. Density Distribution by Stroke Status")
            st.write("Comparing the probability density of Age, BMI, and Glucose levels for patients with and without stroke.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                fig_kde1, ax_kde1 = plt.subplots(figsize=(5, 4))
                sns.kdeplot(data=df_aa, x='age', hue='stroke', fill=True, common_norm=False, palette={0: '#2ecc71', 1: '#e74c3c'}, ax=ax_kde1)
                ax_kde1.set_title("Age Distribution")
                st.pyplot(fig_kde1)
            with col2:
                fig_kde2, ax_kde2 = plt.subplots(figsize=(5, 4))
                sns.kdeplot(data=df_aa.dropna(subset=['bmi']), x='bmi', hue='stroke', fill=True, common_norm=False, palette={0: '#2ecc71', 1: '#e74c3c'}, ax=ax_kde2)
                ax_kde2.set_title("BMI Distribution")
                st.pyplot(fig_kde2)
            with col3:
                fig_kde3, ax_kde3 = plt.subplots(figsize=(5, 4))
                sns.kdeplot(data=df_aa, x='avg_glucose_level', hue='stroke', fill=True, common_norm=False, palette={0: '#2ecc71', 1: '#e74c3c'}, ax=ax_kde3)
                ax_kde3.set_title("Glucose Distribution")
                st.pyplot(fig_kde3)

            st.markdown("---")

            # Violin plots
            st.subheader("3. Outlier and Distribution Analysis (Violin Plots)")
            col1, col2 = st.columns(2)
            
            # Use both int and string keys just in case seaborn casts the hue/x column to string
            palette_dict = {0: '#2ecc71', 1: '#e74c3c', '0': '#2ecc71', '1': '#e74c3c'}
            
            with col1:
                fig_v1, ax_v1 = plt.subplots(figsize=(6, 5))
                sns.violinplot(data=df_aa, x='stroke', y='age', palette=palette_dict, inner="quartile", ax=ax_v1)
                ax_v1.set_title("Age vs Stroke (Violin)")
                st.pyplot(fig_v1)
            with col2:
                fig_v2, ax_v2 = plt.subplots(figsize=(6, 5))
                sns.violinplot(data=df_aa, x='stroke', y='avg_glucose_level', palette=palette_dict, inner="quartile", ax=ax_v2)
                ax_v2.set_title("Glucose Level vs Stroke (Violin)")
                st.pyplot(fig_v2)

        except Exception as e:
            st.error(f"Error loading analytics: {e}")


if __name__ == "__main__":
    main()
