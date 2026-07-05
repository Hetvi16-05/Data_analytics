"""
Page 1 — 🎯 Stroke Risk Prediction
Enter patient vitals and get an AI-powered stroke risk assessment.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import (inject_css, hero, section, info_box, warning_box,
                   kpi_card, load_model, preprocess_input)

# ── Page config ────────────────────────────────────────────────
st.set_page_config(page_title="🎯 Prediction | Stroke AI",
                   page_icon="🎯", layout="wide")
inject_css()

# ── Load model ─────────────────────────────────────────────────
model, scaler, feature_names = load_model()

# ── Hero ───────────────────────────────────────────────────────
hero(
    "🎯 Stroke Risk Predictor",
    "Enter patient clinical data to receive an AI-powered stroke risk assessment.",
    note="⚠️ Educational purposes only — not a clinical diagnostic tool."
)

# ── Patient Input (Top instead of Sidebar) ─────────────────────
with st.container():
    st.markdown("## 👤 Patient Details")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📋 Demographics")
        gender       = st.selectbox("Gender", ["Male", "Female"])
        age          = st.slider("Age (years)", 1, 82, 55)
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        residence    = st.selectbox("Residence Type", ["Urban", "Rural"])
        work_type    = st.selectbox("Work Type",
                        ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])

    with col2:
        st.markdown("### 🏥 Medical History")
        hypertension  = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease",  ["No", "Yes"])
        smoking       = st.selectbox("Smoking Status",
                        ["never smoked", "formerly smoked", "smokes", "Unknown"])

    with col3:
        st.markdown("### 🔬 Lab Values")
        glucose = st.slider("Avg Glucose Level (mg/dL)", 50.0, 300.0, 105.0, 0.5)
        bmi     = st.slider("BMI (Body Mass Index)",     10.0, 70.0,  27.0, 0.1)
        st.markdown("---")
        predict_btn = st.button("🔍 Predict Stroke Risk", use_container_width=True)

# ── Compute live risk score ────────────────────────────────────
risk_score = int(
    (hypertension  == "Yes") +
    (heart_disease == "Yes") +
    (glucose > 140) +
    (bmi > 30) +
    (age > 60)
)

# ── Live KPI Cards ─────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    (c1, f"{age} yrs",      "Age",           "🔴 Senior" if age > 60 else "🟢 Young"),
    (c2, f"{bmi:.1f}",      "BMI",           "🔴 Obese"  if bmi > 30 else "🟢 Normal"),
    (c3, f"{glucose:.0f}",  "Glucose mg/dL", "🔴 High"   if glucose > 140 else "🟢 OK"),
    (c4, "⚠️ Yes" if hypertension  == "Yes" else "✅ No", "Hypertension", ""),
    (c5, f"{risk_score}/5", "Risk Score",
        "🔴 High" if risk_score >= 3 else "🟡 Moderate" if risk_score >= 2 else "🟢 Low"),
]
for col, val, lbl, delta in cards:
    col.markdown(kpi_card(val, lbl, delta), unsafe_allow_html=True)

st.markdown("<hr class='gradient-divider'>", unsafe_allow_html=True)

# ── Prediction Result ──────────────────────────────────────────
if predict_btn:
    if model is None:
        st.error("⚠️ No trained model found.")
        warning_box("📓 Run the notebooks first: <b>notebooks/03_ml_models.ipynb</b> to train and save the model.")
    else:
        with st.spinner("🔄 Analysing patient data with AI model..."):
            input_dict = {
                'gender': gender, 'age': age,
                'hypertension':  1 if hypertension  == "Yes" else 0,
                'heart_disease': 1 if heart_disease == "Yes" else 0,
                'ever_married':  ever_married,
                'work_type':     work_type,
                'Residence_type': residence,
                'avg_glucose_level': glucose,
                'bmi':            bmi,
                'smoking_status': smoking
            }
            try:
                X_proc   = preprocess_input(input_dict, feature_names)
                X_scaled = scaler.transform(X_proc)
                prob     = model.predict_proba(X_scaled)[0][1]

                # ── Risk level ────────────────────────────────
                if prob < 0.30:
                    risk_cls, emoji, label = "risk-low",  "✅", "LOW RISK"
                    advice = "Maintain a healthy lifestyle — regular check-ups are still recommended."
                elif prob < 0.60:
                    risk_cls, emoji, label = "risk-med",  "⚠️", "MODERATE RISK"
                    advice = "Consider lifestyle changes and consult a physician soon."
                else:
                    risk_cls, emoji, label = "risk-high", "🚨", "HIGH RISK"
                    advice = "Seek immediate medical evaluation — multiple critical risk factors detected."

                col_res, col_gauge = st.columns([1, 1])

                with col_res:
                    st.markdown(f"""
                    <div class="risk-box {risk_cls}">
                        {emoji}&nbsp; {label}<br>
                        <span style="font-size:2.8rem;">{prob*100:.1f}%</span><br>
                        <small style="font-size:0.95rem;font-weight:400;">Stroke Probability</small>
                    </div>""", unsafe_allow_html=True)

                    info_box(f"💡 <b>Recommendation:</b> {advice}")

                    section("📋 Patient Summary")
                    summary = pd.DataFrame({
                        "Feature": ["Age", "BMI", "Glucose", "Hypertension",
                                    "Heart Disease", "Smoking", "Composite Risk"],
                        "Value":   [
                            f"{age} yrs",
                            f"{bmi:.1f}  {'(Obese)'    if bmi > 30   else '(Normal)' if bmi > 18.5 else '(Underweight)'}",
                            f"{glucose:.0f} mg/dL  {'(High)' if glucose > 140 else '(Normal)'}",
                            "Yes ⚠️" if hypertension  == "Yes" else "No ✅",
                            "Yes ⚠️" if heart_disease == "Yes" else "No ✅",
                            smoking,
                            f"{risk_score}/5  {'🔴' if risk_score >= 3 else '🟡' if risk_score >= 2 else '🟢'}"
                        ]
                    })
                    st.dataframe(summary, hide_index=True, use_container_width=True)

                with col_gauge:
                    # ── Gauge Chart ───────────────────────────
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number={'suffix': "%", 'font': {'size': 48, 'color': 'white'}},
                        title={'text': "Stroke Risk", 'font': {'size': 18, 'color': '#ccc'}},
                        gauge={
                            'axis':  {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#666'},
                            'bar':   {'color': "#667eea", 'thickness': 0.25},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 0,
                            'steps': [
                                {'range': [0,  30],  'color': 'rgba(46,204,113,0.25)'},
                                {'range': [30, 60],  'color': 'rgba(243,156,18,0.25)'},
                                {'range': [60, 100], 'color': 'rgba(231,76,60,0.25)'},
                            ],
                            'threshold': {
                                'line':      {'color': 'white', 'width': 3},
                                'thickness': 0.75,
                                'value':     prob * 100
                            }
                        }
                    ))
                    fig.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)',
                                      font={'color': 'white', 'family': 'Inter'})
                    st.plotly_chart(fig, use_container_width=True)

                    # ── Probability bar ───────────────────────
                    fig2 = go.Figure(go.Bar(
                        x=[prob * 100, (1 - prob) * 100],
                        y=["Stroke", "No Stroke"],
                        orientation='h',
                        marker_color=["#e74c3c", "#2ecc71"],
                        text=[f"{prob*100:.1f}%", f"{(1-prob)*100:.1f}%"],
                        textposition='inside',
                        textfont=dict(color='white', size=13)
                    ))
                    fig2.update_layout(
                        title="Probability Breakdown",
                        xaxis=dict(range=[0, 100], title="Probability (%)"),
                        height=180, **{k: v for k, v in {
                            'paper_bgcolor': 'rgba(0,0,0,0)',
                            'plot_bgcolor':  'rgba(255,255,255,0.03)',
                            'font': {'color': '#ccc', 'family': 'Inter'},
                            'margin': dict(t=40, b=30, l=10, r=10)
                        }.items()}
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                info_box("Make sure to run the training notebooks first.")

else:
    # ── Idle state — show risk factor guide ───────────────────
    info_box("☝️ <b>Fill in the patient details</b> above, then click <b>Predict Stroke Risk</b>.")

    section("🔑 Key Stroke Risk Factors")
    factors = [
        ("🎂", "Age > 60",          "Stroke risk rises sharply after 60 — the strongest single predictor."),
        ("💊", "Hypertension",      "High blood pressure is the #1 modifiable cardiovascular risk factor."),
        ("❤️",  "Heart Disease",    "Pre-existing cardiac conditions significantly increase stroke risk."),
        ("🍬", "High Glucose",      "Diabetic glucose levels (>140 mg/dL) are strongly correlated with stroke."),
        ("⚖️",  "BMI > 30 (Obese)", "Obesity compounds cardiovascular risk, especially with glucose elevation."),
        ("🚬", "Smoking History",   "Active and former smokers face measurably greater stroke risk."),
    ]
    c1, c2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(factors):
        col = c1 if i % 2 == 0 else c2
        col.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">{icon} {title}</div>
            <div class="insight-text">{desc}</div>
        </div>""", unsafe_allow_html=True)
