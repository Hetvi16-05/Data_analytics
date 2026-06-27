"""
Page 2 — 📊 Analytics Dashboard
High-level KPIs and interactive charts across the entire dataset.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import inject_css, hero, section, kpi_card, PLOTLY_LAYOUT, load_data

# ── Page config ────────────────────────────────────────────────
st.set_page_config(page_title="📊 Dashboard | Stroke AI",
                   page_icon="📊", layout="wide")
inject_css()

# ── Data ───────────────────────────────────────────────────────
df = load_data()
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

# ── Hero ───────────────────────────────────────────────────────
hero("📊 Stroke Analytics Dashboard",
     "Dataset-wide insights — KPIs, distributions, and risk trends at a glance.")

# ──────────────────────────────────────────────────────────────
# ROW 1 — KPI Cards
# ──────────────────────────────────────────────────────────────
section("📈 Dataset Overview")
c1, c2, c3, c4, c5, c6 = st.columns(6)
kpis = [
    (c1, f"{len(df):,}",                      "Total Patients",     ""),
    (c2, f"{df['stroke'].sum():,}",            "Stroke Cases",       "⚠️ Imbalanced"),
    (c3, f"{df['stroke'].mean()*100:.1f}%",    "Stroke Rate",        "vs 95.1% healthy"),
    (c4, f"{df['age'].median():.0f} yrs",      "Median Age",         f"Range {df['age'].min():.0f}–{df['age'].max():.0f}"),
    (c5, f"{df['bmi'].median():.1f}",          "Median BMI",         "WHO Normal: 18.5–25"),
    (c6, f"{df['avg_glucose_level'].mean():.0f}", "Avg Glucose",     "Normal: <100 mg/dL"),
]
for col, val, lbl, delta in kpis:
    col.markdown(kpi_card(val, lbl, delta), unsafe_allow_html=True)

st.markdown("<hr class='gradient-divider'>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# ROW 2 — Target + Age Distribution
# ──────────────────────────────────────────────────────────────
section("🎯 Target Variable & Age Distribution")
col1, col2 = st.columns(2)

with col1:
    counts  = df['stroke'].value_counts()
    fig_pie = go.Figure(go.Pie(
        labels=['No Stroke', 'Stroke'],
        values=counts.values,
        hole=0.55,
        marker=dict(colors=['#2ecc71', '#e74c3c'],
                    line=dict(color='#1a1a2e', width=3)),
        textinfo='label+percent',
        textfont=dict(size=13)
    ))
    fig_pie.add_annotation(text=f"<b>{len(df):,}</b><br>Patients",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=15, color='white'))
    fig_pie.update_layout(title="Class Distribution", **PLOTLY_LAYOUT, height=350,
                          showlegend=True, legend=dict(orientation='h', y=-0.1))
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_age = go.Figure()
    for val, name, color in [(0, 'No Stroke', '#2ecc71'), (1, 'Stroke', '#e74c3c')]:
        fig_age.add_trace(go.Histogram(
            x=df[df['stroke'] == val]['age'],
            name=name, nbinsx=40,
            marker_color=color, opacity=0.75
        ))
    fig_age.update_layout(title="Age Distribution by Stroke", barmode='overlay',
                          xaxis_title="Age", yaxis_title="Count",
                          **PLOTLY_LAYOUT, height=350)
    st.plotly_chart(fig_age, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# ROW 3 — Stroke Rate by Categorical Features
# ──────────────────────────────────────────────────────────────
section("🏷️ Stroke Rate by Key Categories")

cat_features = {
    'Gender':         'gender',
    'Hypertension':   'hypertension',
    'Heart Disease':  'heart_disease',
    'Work Type':      'work_type',
    'Smoking Status': 'smoking_status',
    'Residence':      'Residence_type'
}

cols = st.columns(3)
palette = px.colors.qualitative.Pastel

for i, (label, col_name) in enumerate(cat_features.items()):
    rate = (df.groupby(col_name)['stroke'].mean() * 100).reset_index()
    rate.columns = [col_name, 'Stroke Rate (%)']
    fig = px.bar(rate, x=col_name, y='Stroke Rate (%)',
                 color='Stroke Rate (%)', color_continuous_scale='Reds',
                 title=f"By {label}", text_auto='.1f')
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(**PLOTLY_LAYOUT, height=280,
                      coloraxis_showscale=False,
                      margin=dict(t=45, b=30, l=20, r=10))
    cols[i % 3].plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# ROW 4 — Numerical Distributions
# ──────────────────────────────────────────────────────────────
section("📉 Numerical Feature Distributions")
col1, col2, col3 = st.columns(3)

for col, feat, title, unit in [
    (col1, 'age',               'Age',          'years'),
    (col2, 'avg_glucose_level', 'Glucose Level','mg/dL'),
    (col3, 'bmi',               'BMI',          '')
]:
    fig = go.Figure()
    for val, name, color in [(0, 'No Stroke', '#2ecc71'), (1, 'Stroke', '#e74c3c')]:
        data = df[df['stroke'] == val][feat].dropna()
        fig.add_trace(go.Violin(
            y=data, name=name,
            box_visible=True, meanline_visible=True,
            line_color=color, fillcolor=color,
            opacity=0.5
        ))
    fig.update_layout(title=f"{title} ({unit})", violinmode='overlay',
                      yaxis_title=unit, **PLOTLY_LAYOUT, height=300,
                      margin=dict(t=45, b=30, l=20, r=10))
    col.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# ROW 5 — Stroke Rate by Age Band (Heatmap style)
# ──────────────────────────────────────────────────────────────
section("🔥 Stroke Risk Heatmap — Age × Glucose")
df_heat = df.copy()
df_heat['age_band'] = pd.cut(df_heat['age'],
    bins=[0, 20, 30, 40, 50, 60, 70, 82],
    labels=['≤20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+'])
df_heat['glucose_band'] = pd.cut(df_heat['avg_glucose_level'],
    bins=[0, 80, 100, 130, 160, 200, 350],
    labels=['<80', '80-100', '100-130', '130-160', '160-200', '>200'])

pivot = df_heat.groupby(['age_band', 'glucose_band'], observed=True)['stroke'].mean() * 100
pivot = pivot.unstack(fill_value=0)

fig_heat = go.Figure(go.Heatmap(
    z=pivot.values,
    x=[str(c) for c in pivot.columns],
    y=[str(i) for i in pivot.index],
    colorscale='Reds',
    colorbar=dict(title='Stroke Rate %', ticksuffix='%'),
    text=pivot.values.round(1),
    texttemplate='%{text}%',
    hovertemplate='Age: %{y}<br>Glucose: %{x}<br>Stroke Rate: %{z:.1f}%<extra></extra>'
))
fig_heat.update_layout(
    title="Stroke Rate (%) by Age Group & Glucose Level",
    xaxis_title="Avg Glucose Level Band (mg/dL)",
    yaxis_title="Age Group",
    **PLOTLY_LAYOUT, height=380
)
st.plotly_chart(fig_heat, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# ROW 6 — Comorbidity Analysis
# ──────────────────────────────────────────────────────────────
section("🫀 Comorbidity Risk Matrix")
comorbid = df.groupby(['hypertension', 'heart_disease'])['stroke'].mean() * 100
comorbid = comorbid.reset_index()
comorbid['Hypertension']   = comorbid['hypertension'].map({0: 'No Hypertension', 1: 'Hypertension'})
comorbid['Heart Disease']  = comorbid['heart_disease'].map({0: 'No Heart Disease', 1: 'Heart Disease'})

fig_combo = px.bar(
    comorbid, x='Hypertension', y='stroke', color='Heart Disease',
    barmode='group', text_auto='.1f',
    color_discrete_map={'No Heart Disease': '#3498db', 'Heart Disease': '#e74c3c'},
    labels={'stroke': 'Stroke Rate (%)', 'Hypertension': ''},
    title="Stroke Rate by Hypertension & Heart Disease Combination"
)
fig_combo.update_traces(texttemplate='%{text}%', textposition='outside')
fig_combo.update_layout(**PLOTLY_LAYOUT, height=350)
st.plotly_chart(fig_combo, use_container_width=True)
