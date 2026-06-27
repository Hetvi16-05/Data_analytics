"""
🧠 Stroke Prediction App — Entry Point
Run with: streamlit run main.py
"""

import sys, os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
from utils import inject_css, SHARED_CSS

st.set_page_config(
    page_title="🧠 Stroke Prediction AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_css()

st.markdown("""
<div class="hero-card">
    <h1>🧠 Stroke Prediction AI</h1>
    <p>End-to-end Machine Learning & Deep Learning system for stroke risk assessment</p>
    <small>⚠️ Educational purposes only — not a clinical diagnostic tool</small>
</div>""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">🎯 Predict</div>
        <div class="insight-text">Enter patient vitals and get an AI-powered stroke risk score with a gauge chart and patient summary.</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">📊 Analyse</div>
        <div class="insight-text">Explore the dataset with interactive charts, filters, heatmaps, and categorical breakdowns.</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">💡 Explain</div>
        <div class="insight-text">Understand model decisions with SHAP explainability — globally and for individual patients.</div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box" style="margin-top:1.5rem;">
    👈 <b>Use the sidebar</b> to navigate between pages.
</div>""", unsafe_allow_html=True)
