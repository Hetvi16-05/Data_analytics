# pyrefly: ignore [missing-import]
from dash import dcc, html
import pandas as pd

def render_filters(df: pd.DataFrame):
    return html.Div(
        className="filter-panel",
        children=[
            html.H4("Filters", className="filter-header"),
            
            # Gender Filter
            html.Div(
                className="filter-group",
                children=[
                    html.Label("Gender"),
                    dcc.Dropdown(
                        id="filter-gender",
                        options=[{"label": g, "value": g} for g in df["gender"].dropna().unique()],
                        multi=True,
                        placeholder="Select Gender..."
                    )
                ]
            ),
            
            # Age Range Filter
            html.Div(
                className="filter-group",
                children=[
                    html.Label("Age Range"),
                    dcc.RangeSlider(
                        id="filter-age",
                        min=int(df["age"].min()),
                        max=int(df["age"].max()),
                        step=1,
                        marks={
                            int(df["age"].min()): str(int(df["age"].min())),
                            int(df["age"].max()): str(int(df["age"].max()))
                        },
                        value=[int(df["age"].min()), int(df["age"].max())],
                        tooltip={"placement": "bottom", "always_visible": False}
                    )
                ]
            ),
            
            # Hypertension Filter
            html.Div(
                className="filter-group",
                children=[
                    html.Label("Hypertension"),
                    dcc.Dropdown(
                        id="filter-hypertension",
                        options=[
                            {"label": "Yes", "value": 1},
                            {"label": "No", "value": 0}
                        ],
                        multi=True,
                        placeholder="Select..."
                    )
                ]
            ),
            
            # Heart Disease Filter
            html.Div(
                className="filter-group",
                children=[
                    html.Label("Heart Disease"),
                    dcc.Dropdown(
                        id="filter-heart-disease",
                        options=[
                            {"label": "Yes", "value": 1},
                            {"label": "No", "value": 0}
                        ],
                        multi=True,
                        placeholder="Select..."
                    )
                ]
            ),
            
            # Smoking Status Filter
            html.Div(
                className="filter-group",
                children=[
                    html.Label("Smoking Status"),
                    dcc.Dropdown(
                        id="filter-smoking",
                        options=[{"label": s, "value": s} for s in df["smoking_status"].dropna().unique()],
                        multi=True,
                        placeholder="Select..."
                    )
                ]
            ),
            
            # Clear Filters Button
            html.Button("Reset Filters", id="reset-filters-btn", className="reset-btn", n_clicks=0)
        ]
    )
