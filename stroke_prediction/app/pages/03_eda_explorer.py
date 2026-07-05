"""
Page 3 — 🔍 EDA Explorer
Interactive, filter-driven exploratory data analysis.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import inject_css, hero, section, info_box, PLOTLY_LAYOUT, load_data

# ── Page config ────────────────────────────────────────────────
st.set_page_config(page_title="🔍 EDA Explorer | Stroke AI",
                   page_icon="🔍", layout="wide")
inject_css()

df_raw = load_data()

# ── Hero ───────────────────────────────────────────────────────
hero("🔍 EDA Explorer",
     "Interactively explore the stroke dataset — filter, slice, and visualise any feature.")

# ── Top Filters ────────────────────────────────────────────
with st.container():
    st.markdown("## 🎛️ Filters")

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        age_range   = st.slider("Age Range", 0, 82, (0, 82))
        gender_sel  = st.multiselect("Gender",
                        df_raw['gender'].unique().tolist(),
                        default=df_raw['gender'].unique().tolist())
        stroke_sel  = st.multiselect("Stroke Status",
                        [0, 1], default=[0, 1],
                        format_func=lambda x: "Stroke" if x == 1 else "No Stroke")
    with f_col2:
        work_sel    = st.multiselect("Work Type",
                        df_raw['work_type'].unique().tolist(),
                        default=df_raw['work_type'].unique().tolist())
        smoke_sel   = st.multiselect("Smoking Status",
                        df_raw['smoking_status'].unique().tolist(),
                        default=df_raw['smoking_status'].unique().tolist())
    with f_col3:
        gluc_range  = st.slider("Glucose Level (mg/dL)", 50.0, 300.0, (50.0, 300.0))
        bmi_range   = st.slider("BMI Range", 10.0, 70.0, (10.0, 70.0))

    st.markdown("### 📊 Chart Options")
    c_col1, c_col2, c_col3 = st.columns(3)
    with c_col1:
        x_axis  = st.selectbox("X Axis", ['age', 'avg_glucose_level', 'bmi'])
    with c_col2:
        y_axis  = st.selectbox("Y Axis", ['avg_glucose_level', 'bmi', 'age'])
    with c_col3:
        color_by = st.selectbox("Color By",
                    ['stroke', 'gender', 'hypertension', 'heart_disease', 'smoking_status'])
    st.markdown("---")

# ── Apply filters ──────────────────────────────────────────────
df = df_raw.copy()
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df = df[
    (df['age'].between(*age_range)) &
    (df['gender'].isin(gender_sel)) &
    (df['stroke'].isin(stroke_sel)) &
    (df['work_type'].isin(work_sel)) &
    (df['smoking_status'].isin(smoke_sel)) &
    (df['avg_glucose_level'].between(*gluc_range)) &
    (df['bmi'].between(*bmi_range) | df['bmi'].isna())
]

# ── Filtered summary ───────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filtered Patients", f"{len(df):,}", delta=f"{len(df)-len(df_raw):,} from full dataset")
c2.metric("Stroke Cases",      f"{df['stroke'].sum():,}")
c3.metric("Stroke Rate",       f"{df['stroke'].mean()*100:.1f}%" if len(df) > 0 else "—")
c4.metric("Avg Age",           f"{df['age'].mean():.1f} yrs" if len(df) > 0 else "—")

st.markdown("<hr class='gradient-divider'>", unsafe_allow_html=True)

if len(df) == 0:
    st.warning("⚠️ No data matches the current filters. Adjust the sidebar controls.")
    st.stop()

# ── Scatter Plot ───────────────────────────────────────────────
section(f"🔵 Scatter: {x_axis.replace('_',' ').title()} vs {y_axis.replace('_',' ').title()}")

color_map = None
if color_by == 'stroke':
    df['stroke_label'] = df['stroke'].map({0: 'No Stroke', 1: 'Stroke'})
    color_col = 'stroke_label'
    color_map = {'No Stroke': '#2ecc71', 'Stroke': '#e74c3c'}
else:
    color_col = color_by

fig_scatter = px.scatter(
    df.dropna(subset=[x_axis, y_axis]),
    x=x_axis, y=y_axis,
    color=color_col,
    color_discrete_map=color_map,
    opacity=0.6, size_max=6,
    hover_data=['age', 'bmi', 'avg_glucose_level'],
    labels={x_axis: x_axis.replace('_', ' ').title(),
            y_axis: y_axis.replace('_', ' ').title()},
    title=f"{x_axis.replace('_',' ').title()} vs {y_axis.replace('_',' ').title()}"
)
fig_scatter.update_traces(marker=dict(size=5))
fig_scatter.update_layout(**PLOTLY_LAYOUT, height=420)
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Distribution Plots ─────────────────────────────────────────
section("📊 Feature Distributions")
col1, col2 = st.columns(2)

with col1:
    num_feat = st.selectbox("Select feature to plot",
                ['age', 'avg_glucose_level', 'bmi'], key='dist_feat')
    fig_hist = px.histogram(
        df.dropna(subset=[num_feat]),
        x=num_feat, color='stroke',
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
        nbins=40, opacity=0.8, barmode='overlay',
        labels={'stroke': 'Stroke', num_feat: num_feat.replace('_', ' ').title()},
        title=f"Distribution of {num_feat.replace('_', ' ').title()}"
    )
    fig_hist.update_layout(**PLOTLY_LAYOUT, height=340,
                            margin=dict(t=45, b=30, l=20, r=10))
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    fig_box = px.box(
        df.dropna(subset=[num_feat]),
        x='stroke', y=num_feat,
        color='stroke',
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
        labels={'stroke': 'Stroke Status', num_feat: num_feat.replace('_', ' ').title()},
        title=f"Box Plot: {num_feat.replace('_', ' ').title()} by Stroke"
    )
    fig_box.update_xaxes(ticktext=['No Stroke', 'Stroke'], tickvals=[0, 1])
    fig_box.update_layout(**PLOTLY_LAYOUT, height=340,
                           margin=dict(t=45, b=30, l=20, r=10), showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

# ── Correlation Heatmap ────────────────────────────────────────
section("🔥 Correlation Heatmap (Filtered Data)")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr     = df[num_cols].corr().round(2)
mask     = np.triu(np.ones_like(corr, dtype=bool))
corr_masked = corr.mask(mask)

fig_heat = go.Figure(go.Heatmap(
    z=corr_masked.values,
    x=corr_masked.columns.tolist(),
    y=corr_masked.index.tolist(),
    colorscale='RdBu', zmin=-1, zmax=1,
    colorbar=dict(title='r'),
    text=corr_masked.round(2).values,
    texttemplate='%{text}',
    hovertemplate='%{y} — %{x}<br>r = %{z:.2f}<extra></extra>'
))
fig_heat.update_layout(title="Pearson Correlation Matrix",
                        **PLOTLY_LAYOUT, height=380)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Categorical Breakdown ──────────────────────────────────────
section("🏷️ Categorical Breakdown")
cat_col = st.selectbox("Choose categorical feature",
            ['gender', 'work_type', 'smoking_status',
             'hypertension', 'heart_disease', 'Residence_type', 'ever_married'])

cat_rate = df.groupby(cat_col)['stroke'].agg(['mean', 'count']).reset_index()
cat_rate.columns = [cat_col, 'Stroke Rate', 'Count']
cat_rate['Stroke Rate (%)'] = (cat_rate['Stroke Rate'] * 100).round(2)

col_a, col_b = st.columns(2)
with col_a:
    fig_bar = px.bar(cat_rate, x=cat_col, y='Stroke Rate (%)',
                     color='Stroke Rate (%)', color_continuous_scale='Reds',
                     title=f"Stroke Rate by {cat_col.replace('_', ' ').title()}",
                     text='Stroke Rate (%)')
    fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_bar.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, height=340)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_b:
    fig_count = px.bar(cat_rate, x=cat_col, y='Count',
                       color=cat_col, title=f"Patient Count by {cat_col.replace('_', ' ').title()}",
                       text='Count')
    fig_count.update_traces(textposition='outside')
    fig_count.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
    st.plotly_chart(fig_count, use_container_width=True)

# ── Raw Data Table ─────────────────────────────────────────────
section("🗂️ Filtered Raw Data")
info_box(f"Showing <b>{min(200, len(df)):,}</b> of <b>{len(df):,}</b> filtered records.")
st.dataframe(df.drop(columns=['id'], errors='ignore').head(200),
             use_container_width=True, hide_index=True)

csv = df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Filtered CSV", csv,
                   file_name="stroke_filtered.csv", mime="text/csv")
