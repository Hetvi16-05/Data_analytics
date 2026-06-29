# pyrefly: ignore [missing-import]
from dash import html

def create_card(title, value, id_suffix, icon_class="fas fa-chart-line"):
    return html.Div(
        className="kpi-card",
        children=[
            html.Div(
                className="kpi-icon",
                children=[html.I(className=icon_class)]
            ),
            html.Div(
                className="kpi-details",
                children=[
                    html.H5(title, className="kpi-title"),
                    html.H3(value, id=f"kpi-{id_suffix}", className="kpi-value")
                ]
            )
        ]
    )

def render_cards():
    return html.Div(
        className="kpi-container",
        children=[
            create_card("Total Patients", "0", "total-patients", "fas fa-users"),
            create_card("Stroke Cases", "0", "stroke-cases", "fas fa-heartbeat"),
            create_card("Stroke Rate", "0%", "stroke-rate", "fas fa-percentage"),
            create_card("Avg Age", "0", "avg-age", "fas fa-user-clock"),
        ]
    )
