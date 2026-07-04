import plotly.express as px
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
from dash import dcc, html
import pandas as pd
import numpy as np

def _apply_clean_layout(fig):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=30, b=20, l=20, r=20),
        xaxis=dict(showgrid=True, gridcolor='#eaeaea', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#eaeaea', zeroline=False)
    )
    return fig

def create_stroke_by_age(df: pd.DataFrame):
    if df.empty: return go.Figure()
    # Sum of stroke by age
    stroke_by_age = df.groupby('age')['stroke'].sum().reset_index()
    fig = px.bar(
        stroke_by_age, x='age', y='stroke',
        title="Stroke Cases by Age",
        labels={'age': 'age', 'stroke': 'Sum of stroke'},
        color_discrete_sequence=['#0078D4']
    )
    return _apply_clean_layout(fig)

def create_bmi_glucose(df: pd.DataFrame):
    if df.empty: return go.Figure()
    # Scatter of BMI vs glucose
    df_rounded = df.copy()
    df_rounded['bmi_r'] = df_rounded['bmi'].round(0)
    df_rounded['glucose_r'] = df_rounded['avg_glucose_level'].round(0)
    grouped = df_rounded.groupby(['bmi_r', 'glucose_r']).agg(count=('stroke', 'size')).reset_index()
    
    fig = px.scatter(
        grouped, x='bmi_r', y='glucose_r', size='count',
        title="BMI vs Average Glucose Level",
        labels={'bmi_r': 'bmi', 'glucose_r': 'Average of avg_glucose_level'},
        color_discrete_sequence=['#0078D4']
    )
    return _apply_clean_layout(fig)

def create_stroke_trend(df: pd.DataFrame):
    if df.empty: return go.Figure()
    # Stroke Trend across age
    trend = df.groupby('age')['stroke'].count().reset_index()
    fig = px.line(
        trend, x='age', y='stroke',
        title="Stroke Trend Across Age",
        labels={'age': 'age', 'stroke': 'Count of stroke'},
        color_discrete_sequence=['#0078D4']
    )
    fig.update_traces(line=dict(width=3))
    return _apply_clean_layout(fig)

def create_gender_donut(df: pd.DataFrame):
    if df.empty: return go.Figure()
    # Gender (0 or 1).
    gender_counts = df['gender'].value_counts().reset_index()
    gender_counts.columns = ['gender', 'count']
    gender_counts['gender'] = gender_counts['gender'].astype(str)
    
    fig = px.pie(
        gender_counts, values='count', names='gender',
        hole=0.6,
        title="Stroke Distribution by Gender",
        color_discrete_sequence=['#0078D4', '#002050', '#83B4FF']
    )
    fig.update_traces(textinfo='value+percent')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=20, l=20, r=20))
    return fig

def create_hypertension_bar(df: pd.DataFrame):
    if df.empty: return go.Figure()
    # Sum of stroke by hypertension
    hyper_stroke = df.groupby('hypertension')['stroke'].sum().reset_index()
    hyper_stroke['hypertension'] = hyper_stroke['hypertension'].astype(str)
    
    fig = px.bar(
        hyper_stroke, x='stroke', y='hypertension', orientation='h',
        title="Stroke Cases by Hypertension Status",
        labels={'hypertension': 'hypertension', 'stroke': 'Sum of stroke'},
        color_discrete_sequence=['#0078D4']
    )
    return _apply_clean_layout(fig)

def create_risk_tree(df: pd.DataFrame):
    if df.empty: return go.Figure()
    stroke_df = df[df['stroke'] == 1].copy()
    if stroke_df.empty: return go.Figure()
    
    stroke_df['age_bin'] = pd.cut(stroke_df['age'], bins=[0,40,60,80,100], labels=['0-40', '41-60', '61-80', '81+'])
    grouped = stroke_df.groupby(['smoking_status', 'age_bin'], observed=False).size().reset_index(name='count')
    grouped = grouped[grouped['count'] > 0]
    
    fig = px.sunburst(
        grouped, path=['smoking_status', 'age_bin'], values='count',
        title="Stroke Risk Factor Analysis",
        color_discrete_sequence=['#0078D4', '#5C9CE6', '#99C3F0', '#002050']
    )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=20, l=20, r=20))
    return fig

def render_charts():
    return html.Div(
        className="charts-container",
        children=[
            html.Div(
                className="chart-row",
                children=[
                    html.Div(className="chart-card", children=[dcc.Graph(id="chart-stroke-age")]),
                    html.Div(className="chart-card", children=[dcc.Graph(id="chart-bmi-glucose")]),
                    html.Div(className="chart-card", children=[dcc.Graph(id="chart-stroke-trend")]),
                ]
            ),
            html.Div(
                className="chart-row",
                children=[
                    html.Div(className="chart-card", children=[dcc.Graph(id="chart-gender")]),
                    html.Div(className="chart-card", children=[dcc.Graph(id="chart-hypertension")]),
                    html.Div(className="chart-card", children=[dcc.Graph(id="chart-risk-tree")]),
                ]
            )
        ]
    )
