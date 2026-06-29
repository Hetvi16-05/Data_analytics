# pyrefly: ignore [missing-import]
import dash
# pyrefly: ignore [missing-import]
from dash import html
import pandas as pd
import os
import sys

# Add the project root to the path so we can import dashboard components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.components.filters import render_filters
from dashboard.components.cards import render_cards
from dashboard.components.charts import render_charts

# Initialize the Dash app
app = dash.Dash(
    __name__,
    title="Stroke Prediction Dashboard",
    external_stylesheets=[
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    ]
)
server = app.server

# Load data
try:
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "healthcare-dataset-stroke-data.csv")
    df = pd.read_csv(data_path)
    # Basic cleaning matching pipeline
    df['bmi'] = pd.to_numeric(df['bmi'].replace('N/A', pd.NA), errors='coerce')
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df = df[df['gender'] != 'Other']
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame()

# Define the layout
app.layout = html.Div(
    id="app-container",
    children=[
        # Sidebar for filters
        html.Div(
            className="sidebar",
            children=[
                html.H2("Stroke Analytics"),
                html.P("Interactive dashboard for exploring patient clinical records and stroke incidence."),
                render_filters(df) if not df.empty else html.Div("Data not loaded")
            ]
        ),
        
        # Main content area
        html.Div(
            className="main-content",
            children=[
                render_cards(),
                render_charts()
            ]
        )
    ]
)

# Import callbacks
import dashboard.callbacks

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
