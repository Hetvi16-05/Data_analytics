"""
Page 5 — 💡 SHAP Explainer
Understand WHY the model makes its predictions using SHAP values.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import inject_css, hero, section, info_box, warning_box, PLOTLY_LAYOUT

# ── Page config ────────────────────────────────────────────────
st.set_page_config(page_title="💡 SHAP Explainer | Stroke AI",
                   page_icon="💡", layout="wide")
inject_css()

# ── Hero ───────────────────────────────────────────────────────
hero(
    "💡 SHAP Explainability",
    "Understand <em>why</em> the model predicts a stroke — globally and for individual patients.",
    note="SHAP = SHapley Additive exPlanations — a game-theory-based method for model interpretability."
)

# ── Paths ──────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), '..', '..')
PLOT_DIR = os.path.join(BASE, 'plots', 'shap')
shap_summary   = os.path.join(PLOT_DIR, 'shap_summary.png')
shap_bar       = os.path.join(PLOT_DIR, 'shap_bar.png')
shap_waterfall = os.path.join(PLOT_DIR, 'shap_waterfall.png')

# ── What is SHAP? ──────────────────────────────────────────────
section("🤔 What is SHAP?")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    SHAP (SHapley Additive exPlanations) answers the question:
    > **"How much did each feature contribute to THIS prediction?"**

    It is based on **cooperative game theory** — each feature is treated as a
    "player" and SHAP assigns each player a fair share of the model's output.

    **Why SHAP matters:**
    - Works with ANY model (tree-based, neural network, linear)
    - Gives both **global** (dataset-level) and **local** (patient-level) explanations
    - Satisfies mathematical fairness properties (Efficiency, Symmetry, Linearity)
    """)
with col2:
    st.markdown("""
    **How to read SHAP values:**

    | SHAP Value | Meaning |
    |---|---|
    | **Positive (+)** | This feature INCREASED stroke probability |
    | **Negative (−)** | This feature DECREASED stroke probability |
    | **Near 0** | This feature had little impact |

    **Plot types used:**
    - 🔴🔵 **Summary Plot** — feature importance + direction (color = high/low value)
    - 📊 **Bar Plot** — mean absolute SHAP = overall importance
    - 💧 **Waterfall Plot** — single patient breakdown
    """)

st.markdown("<hr class='gradient-divider'>", unsafe_allow_html=True)

# ── Check if SHAP plots exist ──────────────────────────────────
plots_exist = all(os.path.exists(p) for p in [shap_summary, shap_bar, shap_waterfall])

if not plots_exist:
    warning_box(
        "📓 <b>SHAP plots not found.</b><br>"
        "Run <b>notebooks/05_shap_explainability.ipynb</b> to generate them.<br>"
        "They will be saved to <code>plots/shap/</code> automatically."
    )

# ── Global SHAP — Summary Plot ─────────────────────────────────
section("🌍 Global Explainability — SHAP Summary Plot")
info_box("Each dot = one patient. Color = feature value (🔴 high, 🔵 low). "
         "X position = SHAP value (impact on stroke probability).")
if os.path.exists(shap_summary):
    st.image(shap_summary,
             caption="SHAP Summary Plot — Feature Impact Direction & Magnitude",
             use_container_width=True)
else:
    st.markdown("""
    <div style='background:rgba(255,255,255,0.03);border:1px dashed rgba(255,255,255,0.15);
    border-radius:12px;padding:3rem;text-align:center;color:#666;font-size:1rem;'>
        📊 SHAP Summary Plot will appear here after running the notebook.
    </div>""", unsafe_allow_html=True)

# ── Global SHAP — Bar Plot ─────────────────────────────────────
section("📊 Global Feature Importance — Mean |SHAP|")
info_box("The bar chart shows the <b>average absolute SHAP value</b> for each feature — "
         "i.e., how much on average each feature pushes the prediction up or down.")

col1, col2 = st.columns([2, 1])
with col1:
    if os.path.exists(shap_bar):
        st.image(shap_bar,
                 caption="SHAP Bar Plot — Mean Absolute SHAP Values",
                 use_container_width=True)
    else:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.03);border:1px dashed rgba(255,255,255,0.15);
        border-radius:12px;padding:3rem;text-align:center;color:#666;'>
            📊 SHAP Bar Plot will appear here.
        </div>""", unsafe_allow_html=True)

with col2:
    section("🔍 Typical Top Features")
    expected_top = [
        ("🎂", "age",               "Increases with age — especially 60+"),
        ("🍬", "avg_glucose_level", "High glucose → higher stroke risk"),
        ("💊", "hypertension",      "Strong direct risk factor"),
        ("❤️",  "heart_disease",    "Cardiac history elevates risk"),
        ("⚖️",  "bmi",              "Obesity adds compounding risk"),
        ("🚬", "smoking_status",    "Former/current smokers at higher risk"),
        ("📅", "risk_score",        "Composite indicator of overall risk"),
    ]
    for icon, feat, desc in expected_top:
        st.markdown(f"""
        <div class="insight-card" style="margin-bottom:0.5rem;">
            <div class="insight-title">{icon} {feat.replace('_',' ').title()}</div>
            <div class="insight-text" style="font-size:0.82rem;">{desc}</div>
        </div>""", unsafe_allow_html=True)

# ── Local SHAP — Waterfall Plot ────────────────────────────────
section("💧 Local Explainability — Individual Patient Waterfall")
info_box("The waterfall plot shows the step-by-step contribution of each feature "
         "for a <b>single high-risk patient</b>. It starts from the base (expected) "
         "value and shows how each feature pushes the prediction up (🔴) or down (🔵).")

if os.path.exists(shap_waterfall):
    st.image(shap_waterfall,
             caption="SHAP Waterfall — Why This Patient Is High-Risk",
             use_container_width=True)
else:
    st.markdown("""
    <div style='background:rgba(255,255,255,0.03);border:1px dashed rgba(255,255,255,0.15);
    border-radius:12px;padding:3rem;text-align:center;color:#666;font-size:1rem;'>
        💧 Waterfall plot for an individual patient will appear here.
    </div>""", unsafe_allow_html=True)

st.markdown("<hr class='gradient-divider'>", unsafe_allow_html=True)

# ── Interactive SHAP Demo ──────────────────────────────────────
section("🎮 Interactive SHAP Demo (Simulated)")
info_box("This is a <b>simulated</b> SHAP waterfall based on known risk factor directions. "
         "Actual SHAP values will reflect the trained model's learned patterns.")

col_a, col_b = st.columns(2)
with col_a:
    demo_age      = st.slider("Patient Age",           1,  82, 68)
    demo_glucose  = st.slider("Avg Glucose (mg/dL)", 50.0, 300.0, 210.0)
    demo_bmi      = st.slider("BMI",                 10.0,  70.0,  31.5)
with col_b:
    demo_hyp  = st.selectbox("Hypertension",  ["No", "Yes"], index=1)
    demo_hd   = st.selectbox("Heart Disease", ["No", "Yes"], index=1)
    demo_smk  = st.selectbox("Smoking",       ["never smoked", "formerly smoked", "smokes"], index=1)

# Simulated SHAP contributions
base_val = 0.049  # dataset stroke rate
shap_contribs = {
    'age':               (demo_age - 43) / 120,
    'avg_glucose_level': (demo_glucose - 106) / 400,
    'bmi':               (demo_bmi - 28.9) / 100,
    'hypertension':       0.05 if demo_hyp == "Yes" else -0.01,
    'heart_disease':      0.04 if demo_hd  == "Yes" else -0.01,
    'smoking_status':     0.02 if demo_smk == "smokes" else 0.01 if demo_smk == "formerly smoked" else -0.005,
}
pred_prob = min(0.99, max(0.01, base_val + sum(shap_contribs.values())))

# Sort by absolute impact
sorted_contribs = sorted(shap_contribs.items(), key=lambda x: abs(x[1]), reverse=True)
labels   = [k.replace('_', ' ').title() for k, _ in sorted_contribs]
values   = [v for _, v in sorted_contribs]
colors   = ['#e74c3c' if v > 0 else '#3498db' for v in values]

fig_demo = go.Figure(go.Bar(
    y=labels, x=values,
    orientation='h',
    marker_color=colors,
    text=[f"{'+' if v > 0 else ''}{v*100:.1f}%" for v in values],
    textposition='outside',
    textfont=dict(color='white', size=11)
))
fig_demo.add_vline(x=0, line_dash='dash', line_color='rgba(255,255,255,0.4)')
fig_demo.update_layout(
    title=f"Simulated SHAP — Predicted Stroke Risk: {pred_prob*100:.1f}%",
    xaxis_title="SHAP Contribution (→ increases risk, ← decreases risk)",
    **PLOTLY_LAYOUT, height=370
)
st.plotly_chart(fig_demo, use_container_width=True)

risk_color = "#e74c3c" if pred_prob > 0.5 else "#f39c12" if pred_prob > 0.25 else "#2ecc71"
st.markdown(f"""
<div style="text-align:center;padding:1rem;border-radius:12px;
background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);">
    Simulated Stroke Probability:&nbsp;
    <span style="font-size:1.6rem;font-weight:800;color:{risk_color};">{pred_prob*100:.1f}%</span>
    &nbsp;(base rate was {base_val*100:.1f}%)
</div>""", unsafe_allow_html=True)
