"""
Page 4 — 📈 Model Comparison
Compare all trained ML & DL models with interactive metrics, ROC curves, and radar chart.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import inject_css, hero, section, info_box, warning_box, PLOTLY_LAYOUT

# ── Page config ────────────────────────────────────────────────
st.set_page_config(page_title="📈 Model Comparison | Stroke AI",
                   page_icon="📈", layout="wide")
inject_css()

# ── Hero ───────────────────────────────────────────────────────
hero("📈 Model Comparison",
     "Compare all trained ML & Deep Learning models across every evaluation metric.")

# ── Load saved metrics ─────────────────────────────────────────
METRICS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'metrics', 'results.csv')

info_box("📋 <b>Note:</b> Run the training notebooks first to populate results. A sample results table is shown if the file is not yet available.")

# Sample placeholder data (replaced once notebooks are run)
SAMPLE_RESULTS = pd.DataFrame({
    'Model':     ['Logistic Regression', 'Decision Tree', 'KNN',
                  'Random Forest', 'Gradient Boosting', 'XGBoost', 'SVM', 'Neural Network'],
    'Accuracy':  [0.00]*8,
    'F1 Score':  [0.00]*8,
    'Precision': [0.00]*8,
    'Recall':    [0.00]*8,
    'ROC-AUC':   [0.00]*8,
})

if os.path.exists(METRICS_PATH):
    try:
        results = pd.read_csv(METRICS_PATH)
        st.success("✅ Live model results loaded from `models/metrics/results.csv`")
    except Exception:
        results = SAMPLE_RESULTS
        warning_box("⚠️ Could not read results file. Showing placeholder table.")
else:
    results = SAMPLE_RESULTS
    warning_box("📓 Run <b>notebooks/03_ml_models.ipynb</b> and <b>04_deep_learning.ipynb</b> to generate results.")

# ── Metric Table ───────────────────────────────────────────────
section("🏆 Leaderboard")

metric_cols = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC-AUC']
sort_by = st.selectbox("Sort by", metric_cols, index=4)
df_disp = results.sort_values(sort_by, ascending=False).reset_index(drop=True)
df_disp.index = df_disp.index + 1  # 1-indexed rank

st.dataframe(
    df_disp.style
        .background_gradient(subset=metric_cols, cmap='YlOrRd')
        .format({m: '{:.4f}' for m in metric_cols}),
    use_container_width=True
)

st.markdown("<hr class='gradient-divider'>", unsafe_allow_html=True)

# ── Bar Chart Comparison ───────────────────────────────────────
section("📊 Metric Comparison — All Models")

selected_metrics = st.multiselect(
    "Select metrics to compare",
    metric_cols,
    default=['F1 Score', 'Recall', 'ROC-AUC']
)

if selected_metrics:
    fig_bar = go.Figure()
    colors  = ['#667eea', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    for metric, color in zip(selected_metrics, colors):
        fig_bar.add_trace(go.Bar(
            name=metric,
            x=df_disp['Model'],
            y=df_disp[metric],
            marker_color=color,
            opacity=0.85,
            text=df_disp[metric].round(3),
            textposition='outside',
            texttemplate='%{text:.3f}'
        ))
    fig_bar.update_layout(
        barmode='group', title="Model Performance Comparison",
        xaxis_tickangle=-30, yaxis=dict(range=[0, 1.15]),
        **PLOTLY_LAYOUT, height=430
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Radar Chart — Top Models ───────────────────────────────────
section("🕸️ Radar Chart — Top Models")

top_n   = st.slider("Number of top models to show", 2, min(8, len(df_disp)), 4)
top_df  = df_disp.head(top_n)
categories = metric_cols

radar_colors = ['#667eea', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
                '#1abc9c', '#e67e22', '#3498db']

fig_radar = go.Figure()
for i, (_, row) in enumerate(top_df.iterrows()):
    vals = row[categories].tolist()
    vals += vals[:1]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals,
        theta=categories + [categories[0]],
        fill='toself',
        name=row['Model'],
        line_color=radar_colors[i % len(radar_colors)],
        opacity=0.65,
        line_width=2.5
    ))
fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1],
                        tickfont=dict(color='#aaa', size=10)),
        angularaxis=dict(tickfont=dict(color='white', size=12))
    ),
    title="Model Strengths Across All Metrics",
    **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ['xaxis', 'yaxis']},
    height=500,
    showlegend=True
)
st.plotly_chart(fig_radar, use_container_width=True)

# ── ROC Curve Section ──────────────────────────────────────────
section("📉 ROC Curves")

roc_path = os.path.join(os.path.dirname(__file__), '..', '..', 'plots', 'models', 'roc_curves.png')
if os.path.exists(roc_path):
    st.image(roc_path, caption="ROC Curves — All Models", use_container_width=True)
else:
    info_box("📓 ROC curve plot will appear here after running the training notebooks. "
             "It is saved to <code>plots/models/roc_curves.png</code>.")

# ── Metric Explanations ────────────────────────────────────────
section("ℹ️ Metric Guide")
col1, col2 = st.columns(2)
metric_guide = [
    ("🎯 Accuracy",   "% of all predictions that are correct. Misleading on imbalanced data."),
    ("⚖️  F1 Score",  "Harmonic mean of Precision & Recall. Best single metric for imbalance."),
    ("🔍 Precision",  "Of all predicted stroke cases, how many were actually strokes?"),
    ("📢 Recall",     "Of all actual stroke cases, how many did we catch? CRITICAL for medicine."),
    ("📈 ROC-AUC",    "Area Under ROC Curve — measures model's ability to distinguish classes."),
]
for i, (title, desc) in enumerate(metric_guide):
    target_col = col1 if i % 2 == 0 else col2
    target_col.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">{title}</div>
        <div class="insight-text">{desc}</div>
    </div>""", unsafe_allow_html=True)

warning_box("⚕️ <b>Clinical Priority:</b> In stroke prediction, <b>Recall</b> is the most important metric — "
            "missing a true stroke case has far greater consequences than a false alarm.")
