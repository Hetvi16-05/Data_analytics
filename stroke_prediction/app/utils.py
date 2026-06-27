"""
Shared CSS styles, helpers, and data loaders for all app pages.
Import this in every page file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ── Paths (relative to the app/ directory) ─────────────────────
DATA_PATH    = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'healthcare-dataset-stroke-data.csv')
MODEL_PATH   = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved', 'best_ml_model.pkl')
SCALER_PATH  = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved', 'scaler.pkl')
FEATURE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved', 'feature_names.json')

# ── Brand Colors ───────────────────────────────────────────────
PURPLE      = "#667eea"
PURPLE_DARK = "#764ba2"
GREEN       = "#2ecc71"
RED         = "#e74c3c"
ORANGE      = "#f39c12"
DARK_BG     = "#0f0c29"


# ── Shared CSS ─────────────────────────────────────────────────
SHARED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero Banner ─────────────────── */
.hero-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem 3rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(102,126,234,0.35);
}
.hero-card h1  { font-size: 2.6rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.hero-card p   { font-size: 1.05rem; margin-top: 0.5rem; opacity: 0.88; }
.hero-card small { opacity: 0.7; font-size: 0.85rem; }

/* ── Section Headers ─────────────── */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #667eea;
    padding: 0.4rem 0 0.6rem 0;
    border-bottom: 2px solid #667eea;
    margin: 1.8rem 0 1rem 0;
}

/* ── KPI / Metric Cards ──────────── */
.kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.3rem 1rem;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(102,126,234,0.2); }
.kpi-value { font-size: 2rem; font-weight: 800; color: #667eea; }
.kpi-label { font-size: 0.8rem; color: #aaa; margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-delta { font-size: 0.75rem; margin-top: 0.15rem; }

/* ── Risk Boxes ──────────────────── */
.risk-box {
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 1rem 0;
    box-shadow: 0 12px 35px rgba(0,0,0,0.3);
    color: white;
}
.risk-low  { background: linear-gradient(135deg, #11998e, #38ef7d); }
.risk-med  { background: linear-gradient(135deg, #f7971e, #ffd200); }
.risk-high { background: linear-gradient(135deg, #cb2d3e, #ef473a); }

/* ── Info / Alert Boxes ──────────── */
.info-box {
    background: rgba(102,126,234,0.12);
    border-left: 4px solid #667eea;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
    color: #ddd;
}
.warning-box {
    background: rgba(239,71,58,0.12);
    border-left: 4px solid #ef473a;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
    color: #ddd;
}
.success-box {
    background: rgba(46,204,113,0.12);
    border-left: 4px solid #2ecc71;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
    color: #ddd;
}

/* ── Buttons ─────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.8rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 5px 20px rgba(102,126,234,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(102,126,234,0.55) !important;
}

/* ── Tag Badges ──────────────────── */
.tag {
    display: inline-block;
    background: rgba(102,126,234,0.18);
    color: #a0a8f0;
    border: 1px solid rgba(102,126,234,0.4);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0.15rem;
}

/* ── Insight Cards ───────────────── */
.insight-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
    border-left: 4px solid #667eea;
}
.insight-title  { font-size: 1rem; font-weight: 600; color: #667eea; margin-bottom: 0.3rem; }
.insight-text   { font-size: 0.9rem; color: #bbb; line-height: 1.5; }

/* ── Divider ─────────────────────── */
.gradient-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #667eea, transparent);
    border: none;
    margin: 1.5rem 0;
}
</style>
"""


# ── Plotly Theme ───────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(255,255,255,0.03)',
    font=dict(family='Inter', color='#ccc', size=12),
    margin=dict(t=50, b=40, l=40, r=20),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#ccc')),
    xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.1)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.1)'),
)


# ── Loaders ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    return df


@st.cache_resource
def load_model():
    for p in [MODEL_PATH, SCALER_PATH, FEATURE_PATH]:
        if not os.path.exists(p):
            return None, None, None
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_PATH) as f:
        features = json.load(f)
    return model, scaler, features


# ── Preprocessing (shared with prediction page) ────────────────
def preprocess_input(input_dict, feature_names):
    df_i = pd.DataFrame([input_dict])

    df_i['age_group'] = pd.cut(df_i['age'],
        bins=[0, 12, 17, 35, 55, 65, 120],
        labels=['Child', 'Teen', 'YoungAdult', 'MiddleAge', 'Senior', 'Elderly'])
    df_i['risk_score'] = (
        df_i['hypertension'] +
        df_i['heart_disease'] +
        (df_i['avg_glucose_level'] > 140).astype(int) +
        (df_i['bmi'] > 30).astype(int) +
        (df_i['age'] > 60).astype(int)
    )
    df_i['glucose_cat'] = pd.cut(df_i['avg_glucose_level'],
        bins=[0, 70, 100, 125, 200, 1000],
        labels=['Low', 'Normal', 'Prediabetic', 'Diabetic', 'VeryHigh'])
    df_i['bmi_cat'] = pd.cut(df_i['bmi'],
        bins=[0, 18.5, 25, 30, 100],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    df_i['gender']         = 1 if input_dict['gender'] == 'Male' else 0
    df_i['ever_married']   = 1 if input_dict['ever_married'] == 'Yes' else 0
    df_i['Residence_type'] = 1 if input_dict['Residence_type'] == 'Urban' else 0

    df_i = pd.get_dummies(df_i,
        columns=['work_type', 'smoking_status', 'age_group', 'glucose_cat', 'bmi_cat'],
        drop_first=True)

    bool_cols = df_i.select_dtypes(include='bool').columns
    df_i[bool_cols] = df_i[bool_cols].astype(int)

    for col in feature_names:
        if col not in df_i.columns:
            df_i[col] = 0
    return df_i[feature_names]


# ── Helper: apply CSS ──────────────────────────────────────────
def inject_css():
    st.markdown(SHARED_CSS, unsafe_allow_html=True)


# ── Helper: hero banner ────────────────────────────────────────
def hero(title: str, subtitle: str, note: str = ""):
    note_html = f"<small>{note}</small>" if note else ""
    st.markdown(f"""
    <div class="hero-card">
        <h1>{title}</h1>
        <p>{subtitle}</p>
        {note_html}
    </div>""", unsafe_allow_html=True)


# ── Helper: section header ─────────────────────────────────────
def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ── Helper: info / warning / success box ──────────────────────
def info_box(text: str):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def warning_box(text: str):
    st.markdown(f'<div class="warning-box">{text}</div>', unsafe_allow_html=True)

def success_box(text: str):
    st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)


# ── Helper: KPI card ───────────────────────────────────────────
def kpi_card(value: str, label: str, delta: str = ""):
    delta_html = f'<div class="kpi-delta" style="color:#aaa">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>"""
