# pyrefly: ignore [missing-import]
from dash.dependencies import Input, Output, State
from dashboard.layout import app, df
from dashboard.components.charts import create_age_stroke_chart, create_glucose_dist_chart, create_correlation_heatmap
# pyrefly: ignore [missing-import]
import dash

@app.callback(
    [Output("kpi-total-patients", "children"),
     Output("kpi-stroke-cases", "children"),
     Output("kpi-stroke-rate", "children"),
     Output("kpi-avg-age", "children"),
     Output("age-stroke-chart", "figure"),
     Output("glucose-dist-chart", "figure"),
     Output("correlation-heatmap", "figure")],
    [Input("filter-gender", "value"),
     Input("filter-age", "value"),
     Input("filter-hypertension", "value"),
     Input("filter-heart-disease", "value"),
     Input("filter-smoking", "value")]
)
def update_dashboard(gender, age_range, hypertension, heart_disease, smoking):
    if df.empty:
        return "0", "0", "0%", "0", {}, {}, {}

    filtered_df = df.copy()

    # Apply filters
    if gender:
        filtered_df = filtered_df[filtered_df["gender"].isin(gender)]
    if age_range:
        filtered_df = filtered_df[(filtered_df["age"] >= age_range[0]) & (filtered_df["age"] <= age_range[1])]
    if hypertension:
        filtered_df = filtered_df[filtered_df["hypertension"].isin(hypertension)]
    if heart_disease:
        filtered_df = filtered_df[filtered_df["heart_disease"].isin(heart_disease)]
    if smoking:
        filtered_df = filtered_df[filtered_df["smoking_status"].isin(smoking)]

    # Calculate KPIs
    total_patients = len(filtered_df)
    stroke_cases = filtered_df["stroke"].sum()
    stroke_rate = (stroke_cases / total_patients * 100) if total_patients > 0 else 0
    avg_age = filtered_df["age"].mean() if total_patients > 0 else 0

    # Format KPIs
    total_patients_str = f"{total_patients:,}"
    stroke_cases_str = f"{stroke_cases:,}"
    stroke_rate_str = f"{stroke_rate:.2f}%"
    avg_age_str = f"{avg_age:.1f}"

    # Generate Charts
    age_stroke_fig = create_age_stroke_chart(filtered_df)
    glucose_dist_fig = create_glucose_dist_chart(filtered_df)
    correlation_fig = create_correlation_heatmap(filtered_df)

    return (
        total_patients_str, 
        stroke_cases_str, 
        stroke_rate_str, 
        avg_age_str,
        age_stroke_fig,
        glucose_dist_fig,
        correlation_fig
    )

@app.callback(
    [Output("filter-gender", "value"),
     Output("filter-age", "value"),
     Output("filter-hypertension", "value"),
     Output("filter-heart-disease", "value"),
     Output("filter-smoking", "value")],
    [Input("reset-filters-btn", "n_clicks")],
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    if not df.empty:
        return None, [int(df["age"].min()), int(df["age"].max())], None, None, None
    return None, [0, 100], None, None, None
