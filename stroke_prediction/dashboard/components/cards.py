# pyrefly: ignore [missing-import]
from dash import html

def create_card(title, value, id_suffix):
    return html.Div(
        className="kpi-card",
        children=[
            html.H3(value, id=f"kpi-{id_suffix}", className="kpi-value"),
            html.P(title, className="kpi-title")
        ]
    )

def render_cards():
    return html.Div(
        className="kpi-container",
        children=[
            create_card("Total Patients", "0", "total-patients"),
            create_card("Average Age (Years)", "0", "avg-age"),
            create_card("Average BMI", "0", "avg-bmi"),
            create_card("Total Stroke Cases", "0", "stroke-cases"),
            create_card("Average Glucose Level", "0", "avg-glucose"),
        ]
    )
