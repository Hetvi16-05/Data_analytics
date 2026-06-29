import plotly.express as px
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
from dash import dcc, html
import pandas as pd
import numpy as np

def create_age_stroke_chart(df: pd.DataFrame):
    if df.empty:
        return go.Figure()
        
    # Group by age and calculate stroke rate
    # Let's create age bins for a better bar chart
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
    labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81+']
    df_chart = df.copy()
    df_chart['age_group'] = pd.cut(df_chart['age'], bins=bins, labels=labels, right=False)
    
    age_stroke = df_chart.groupby('age_group', observed=False)['stroke'].mean().reset_index()
    age_stroke['stroke'] = age_stroke['stroke'] * 100 # Convert to percentage
    
    fig = px.bar(
        age_stroke, 
        x='age_group', 
        y='stroke',
        title="Stroke Rate by Age Group",
        labels={'age_group': 'Age Group', 'stroke': 'Stroke Rate (%)'},
        color_discrete_sequence=['#ef553b']
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=20, l=20, r=20),
        xaxis_title="Age Group",
        yaxis_title="Stroke Rate (%)"
    )
    return fig

def create_glucose_dist_chart(df: pd.DataFrame):
    if df.empty:
        return go.Figure()
        
    fig = px.histogram(
        df, 
        x="avg_glucose_level", 
        color="stroke",
        barmode="overlay",
        title="Glucose Distribution by Stroke Status",
        labels={'avg_glucose_level': 'Average Glucose Level', 'stroke': 'Stroke'},
        color_discrete_map={0: '#636efa', 1: '#ef553b'}
    )
    # Update legend labels
    newnames = {'0': 'No Stroke', '1': 'Stroke'}
    fig.for_each_trace(lambda t: t.update(name = newnames.get(t.name, t.name),
                                      legendgroup = newnames.get(t.name, t.name),
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames.get(t.name, t.name))))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=20, l=20, r=20)
    )
    return fig

def create_correlation_heatmap(df: pd.DataFrame):
    if df.empty:
        return go.Figure()
        
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return go.Figure()
        
    corr = numeric_df.corr()
    
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=20, l=20, r=20)
    )
    return fig

def render_charts():
    return html.Div(
        className="charts-container",
        children=[
            html.Div(
                className="chart-row",
                children=[
                    html.Div(
                        className="chart-card",
                        children=[dcc.Graph(id="age-stroke-chart")]
                    ),
                    html.Div(
                        className="chart-card",
                        children=[dcc.Graph(id="glucose-dist-chart")]
                    )
                ]
            ),
            html.Div(
                className="chart-row",
                children=[
                    html.Div(
                        className="chart-card full-width",
                        children=[dcc.Graph(id="correlation-heatmap")]
                    )
                ]
            )
        ]
    )
