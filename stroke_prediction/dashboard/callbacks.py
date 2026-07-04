# pyrefly: ignore [missing-import]
from dash.dependencies import Input, Output, State
from dashboard.layout import app, df
from dashboard.components.charts import (
    create_stroke_by_age,
    create_bmi_glucose,
    create_stroke_trend,
    create_gender_donut,
    create_hypertension_bar,
    create_risk_tree
)
# pyrefly: ignore [missing-import]
import dash

@app.callback(
    [Output("kpi-total-patients", "children"),
     Output("kpi-avg-age", "children"),
     Output("kpi-avg-bmi", "children"),
     Output("kpi-stroke-cases", "children"),
     Output("kpi-avg-glucose", "children"),
     Output("chart-stroke-age", "figure"),
     Output("chart-bmi-glucose", "figure"),
     Output("chart-stroke-trend", "figure"),
     Output("chart-gender", "figure"),
     Output("chart-hypertension", "figure"),
     Output("chart-risk-tree", "figure")],
    [Input("filter-gender", "value"),
     Input("filter-age", "value"),
     Input("filter-hypertension", "value"),
     Input("filter-heart-disease", "value"),
     Input("filter-smoking", "value")]
)
def update_dashboard(gender, age_range, hypertension, heart_disease, smoking):
    if df.empty:
        return "0", "0", "0", "0", "0", {}, {}, {}, {}, {}, {}

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
    avg_age = filtered_df["age"].mean() if total_patients > 0 else 0
    avg_bmi = filtered_df["bmi"].mean() if total_patients > 0 else 0
    stroke_cases = filtered_df["stroke"].sum()
    avg_glucose = filtered_df["avg_glucose_level"].mean() if total_patients > 0 else 0

    # Format KPIs
    total_patients_str = f"{total_patients/1000:.3f}K" if total_patients >= 1000 else f"{total_patients}"
    avg_age_str = f"{avg_age:.2f}"
    avg_bmi_str = f"{avg_bmi:.2f}"
    stroke_cases_str = f"{stroke_cases}"
    avg_glucose_str = f"{avg_glucose:.2f}"

    # Generate Charts
    fig_stroke_age = create_stroke_by_age(filtered_df)
    fig_bmi_glucose = create_bmi_glucose(filtered_df)
    fig_stroke_trend = create_stroke_trend(filtered_df)
    fig_gender = create_gender_donut(filtered_df)
    fig_hyper = create_hypertension_bar(filtered_df)
    fig_risk = create_risk_tree(filtered_df)

    return (
        total_patients_str, 
        avg_age_str, 
        avg_bmi_str, 
        stroke_cases_str, 
        avg_glucose_str,
        fig_stroke_age,
        fig_bmi_glucose,
        fig_stroke_trend,
        fig_gender,
        fig_hyper,
        fig_risk
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
