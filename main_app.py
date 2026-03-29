import json
import ast
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Baguio City Dengue Forecast Dashboard",
    layout="wide"
)

st.title("Baguio City Dengue Forecast Dashboard")
st.caption("Interactive web-based dashboard for dengue prediction and visualization")

ARTIFACTS_DIR = Path("artifacts")

DEFAULT_FEATURE_COLS = [
    "rainfall", "relative_humidity", "temp_mid",
    "cases_lag_1", "cases_lag_2", "cases_lag_3",
    "rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3",
    "relative_humidity_lag_1", "relative_humidity_lag_2", "relative_humidity_lag_3",
    "temp_mid_lag_1", "temp_mid_lag_2", "temp_mid_lag_3",
    "cases_roll3_mean", "cases_roll3_max",
    "month_sin", "month_cos"
]


def safe_read_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def safe_read_json(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def safe_load_model(path: Path):
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_data
def load_artifacts():
    monthly = safe_read_csv(ARTIFACTS_DIR / "monthly_modeling_dataset.csv")
    model_comparison = safe_read_csv(ARTIFACTS_DIR / "model_comparison.csv")
    feature_importance = safe_read_csv(ARTIFACTS_DIR / "feature_importance.csv")
    feature_sensitivity = safe_read_csv(ARTIFACTS_DIR / "feature_sensitivity.csv")
    forecast = safe_read_csv(ARTIFACTS_DIR / "forecast_5yr.csv")

    barangay_monthly = safe_read_csv(ARTIFACTS_DIR / "barangay_monthly.csv")
    top_barangay_monthly = safe_read_csv(ARTIFACTS_DIR / "top_barangay_monthly.csv")
    top3_barangays_yearly = safe_read_csv(ARTIFACTS_DIR / "top3_barangays_yearly.csv")
    top3_barangays_overall = safe_read_csv(ARTIFACTS_DIR / "top3_barangays_overall.csv")

    test_predictions = safe_read_csv(ARTIFACTS_DIR / "test_predictions.csv")
    confusion_matrix_detail = safe_read_csv(ARTIFACTS_DIR / "confusion_matrix_detail.csv")
    climate_case_correlation = safe_read_csv(ARTIFACTS_DIR / "climate_case_correlation.csv")
    month_profile = safe_read_csv(ARTIFACTS_DIR / "month_profile.csv")

    forecast_barangay_ranking = safe_read_csv(ARTIFACTS_DIR / "forecast_barangay_ranking.csv")
    forecast_top3_barangays = safe_read_csv(ARTIFACTS_DIR / "forecast_top3_barangays.csv")
    barangay_risk_profile = safe_read_csv(ARTIFACTS_DIR / "barangay_risk_profile.csv")

    meta = safe_read_json(ARTIFACTS_DIR / "meta.json")

    return (
        monthly,
        model_comparison,
        feature_importance,
        feature_sensitivity,
        forecast,
        barangay_monthly,
        top_barangay_monthly,
        top3_barangays_yearly,
        top3_barangays_overall,
        test_predictions,
        confusion_matrix_detail,
        climate_case_correlation,
        month_profile,
        forecast_barangay_ranking,
        forecast_top3_barangays,
        barangay_risk_profile,
        meta,
    )


(
    monthly,
    model_comparison,
    feature_importance,
    feature_sensitivity,
    forecast,
    barangay_monthly,
    top_barangay_monthly,
    top3_barangays_yearly,
    top3_barangays_overall,
    test_predictions,
    confusion_matrix_detail,
    climate_case_correlation,
    month_profile,
    forecast_barangay_ranking,
    forecast_top3_barangays,
    barangay_risk_profile,
    meta,
) = load_artifacts()

model = safe_load_model(ARTIFACTS_DIR / "best_model.joblib")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("About")
st.sidebar.write(
    "This dashboard displays historical dengue cases, model results, feature contributions, "
    "and forecast outputs from your Google Colab workflow."
)

if meta:
    st.sidebar.success(f"Best Model: {meta.get('best_model', 'Unknown')}")
    threshold_val = meta.get("outbreak_threshold_cases", "N/A")
    if isinstance(threshold_val, (int, float)):
        st.sidebar.info(f"Outbreak Threshold: {threshold_val:.2f}")
    else:
        st.sidebar.info(f"Outbreak Threshold: {threshold_val}")
else:
    st.sidebar.warning("Metadata not found.")

st.sidebar.subheader("Upload files manually (optional)")

uploaded_monthly = st.sidebar.file_uploader("Upload monthly_modeling_dataset.csv", type=["csv"])
uploaded_model_comparison = st.sidebar.file_uploader("Upload model_comparison.csv", type=["csv"])
uploaded_feature_importance = st.sidebar.file_uploader("Upload feature_importance.csv", type=["csv"])
uploaded_feature_sensitivity = st.sidebar.file_uploader("Upload feature_sensitivity.csv", type=["csv"])
uploaded_forecast = st.sidebar.file_uploader("Upload forecast_5yr.csv", type=["csv"])
uploaded_barangay_monthly = st.sidebar.file_uploader("Upload barangay_monthly.csv", type=["csv"])
uploaded_top_barangay_monthly = st.sidebar.file_uploader("Upload top_barangay_monthly.csv", type=["csv"])
uploaded_top3_yearly = st.sidebar.file_uploader("Upload top3_barangays_yearly.csv", type=["csv"])
uploaded_top3_overall = st.sidebar.file_uploader("Upload top3_barangays_overall.csv", type=["csv"])
uploaded_test_predictions = st.sidebar.file_uploader("Upload test_predictions.csv", type=["csv"])
uploaded_cm_detail = st.sidebar.file_uploader("Upload confusion_matrix_detail.csv", type=["csv"])
uploaded_climate_corr = st.sidebar.file_uploader("Upload climate_case_correlation.csv", type=["csv"])
uploaded_month_profile = st.sidebar.file_uploader("Upload month_profile.csv", type=["csv"])
uploaded_forecast_barangay_ranking = st.sidebar.file_uploader("Upload forecast_barangay_ranking.csv", type=["csv"])
uploaded_forecast_top3_barangays = st.sidebar.file_uploader("Upload forecast_top3_barangays.csv", type=["csv"])
uploaded_barangay_risk_profile = st.sidebar.file_uploader("Upload barangay_risk_profile.csv", type=["csv"])
uploaded_meta = st.sidebar.file_uploader("Upload meta.json", type=["json"])
uploaded_model = st.sidebar.file_uploader("Upload best_model.joblib", type=["joblib", "pkl"])

if uploaded_monthly is not None:
    monthly = pd.read_csv(uploaded_monthly)
if uploaded_model_comparison is not None:
    model_comparison = pd.read_csv(uploaded_model_comparison)
if uploaded_feature_importance is not None:
    feature_importance = pd.read_csv(uploaded_feature_importance)
if uploaded_feature_sensitivity is not None:
    feature_sensitivity = pd.read_csv(uploaded_feature_sensitivity)
if uploaded_forecast is not None:
    forecast = pd.read_csv(uploaded_forecast)
if uploaded_barangay_monthly is not None:
    barangay_monthly = pd.read_csv(uploaded_barangay_monthly)
if uploaded_top_barangay_monthly is not None:
    top_barangay_monthly = pd.read_csv(uploaded_top_barangay_monthly)
if uploaded_top3_yearly is not None:
    top3_barangays_yearly = pd.read_csv(uploaded_top3_yearly)
if uploaded_top3_overall is not None:
    top3_barangays_overall = pd.read_csv(uploaded_top3_overall)
if uploaded_test_predictions is not None:
    test_predictions = pd.read_csv(uploaded_test_predictions)
if uploaded_cm_detail is not None:
    confusion_matrix_detail = pd.read_csv(uploaded_cm_detail)
if uploaded_climate_corr is not None:
    climate_case_correlation = pd.read_csv(uploaded_climate_corr)
if uploaded_month_profile is not None:
    month_profile = pd.read_csv(uploaded_month_profile)
if uploaded_forecast_barangay_ranking is not None:
    forecast_barangay_ranking = pd.read_csv(uploaded_forecast_barangay_ranking)
if uploaded_forecast_top3_barangays is not None:
    forecast_top3_barangays = pd.read_csv(uploaded_forecast_top3_barangays)
if uploaded_barangay_risk_profile is not None:
    barangay_risk_profile = pd.read_csv(uploaded_barangay_risk_profile)
if uploaded_meta is not None:
    meta = json.load(uploaded_meta)
if uploaded_model is not None:
    model = joblib.load(uploaded_model)

if monthly is None:
    st.error("monthly_modeling_dataset.csv is required.")
    st.stop()

for df_name in [
    "monthly",
    "forecast",
    "top_barangay_monthly",
    "barangay_monthly",
    "test_predictions",
    "forecast_barangay_ranking",
    "forecast_top3_barangays",
]:
    df_obj = locals().get(df_name)
    if df_obj is not None and "Date" in df_obj.columns:
        df_obj["Date"] = pd.to_datetime(df_obj["Date"], errors="coerce")
        locals()[df_name] = df_obj


def safe_metric_value(value, decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def outbreak_label_from_binary(x):
    return "Outbreak" if int(x) == 1 else "Non-outbreak"


def month_name_from_number(m):
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    return month_names.get(int(m), str(m))


def get_month_profile_row(month_num, month_profile_df):
    if month_profile_df is None or month_profile_df.empty:
        return None
    if "Month" in month_profile_df.columns:
        subset = month_profile_df[month_profile_df["Month"] == int(month_num)]
        if not subset.empty:
            return subset.iloc[0]
    return None


def get_profile_value(month_num, col_name, month_profile_df, fallback_df=None, default=0.0):
    row = get_month_profile_row(month_num, month_profile_df)
    if row is not None and col_name in row.index and pd.notna(row[col_name]):
        return float(row[col_name])

    if fallback_df is not None and col_name in fallback_df.columns:
        series = pd.to_numeric(fallback_df[col_name], errors="coerce").dropna()
        if len(series) > 0:
            return float(series.mean())

    return float(default)


def get_previous_months(month_num):
    m1 = 12 if month_num - 1 <= 0 else month_num - 1
    m2 = 12 if m1 - 1 <= 0 else m1 - 1
    m3 = 12 if m2 - 1 <= 0 else m2 - 1
    return m1, m2, m3


def get_reasonable_range(df, col_name, fallback_min=0.0, fallback_max=100.0):
    if df is not None and col_name in df.columns:
        series = pd.to_numeric(df[col_name], errors="coerce").dropna()
        if len(series) > 0:
            vmin = float(series.min())
            vmax = float(series.max())
            if vmin == vmax:
                vmax = vmin + 1.0
            return vmin, vmax
    return fallback_min, fallback_max


def build_live_prediction_features(
    month_num,
    rainfall_now,
    humidity_now,
    temp_now,
    cases_lag_1,
    cases_lag_2,
    cases_lag_3,
    month_profile_df
):
    prev1, prev2, prev3 = get_previous_months(month_num)

    rainfall_lag_1 = get_profile_value(prev1, "rainfall", month_profile_df, monthly, 0.0)
    rainfall_lag_2 = get_profile_value(prev2, "rainfall", month_profile_df, monthly, 0.0)
    rainfall_lag_3 = get_profile_value(prev3, "rainfall", month_profile_df, monthly, 0.0)

    rh_lag_1 = get_profile_value(prev1, "relative_humidity", month_profile_df, monthly, 0.0)
    rh_lag_2 = get_profile_value(prev2, "relative_humidity", month_profile_df, monthly, 0.0)
    rh_lag_3 = get_profile_value(prev3, "relative_humidity", month_profile_df, monthly, 0.0)

    temp_lag_1 = get_profile_value(prev1, "temp_mid", month_profile_df, monthly, 0.0)
    temp_lag_2 = get_profile_value(prev2, "temp_mid", month_profile_df, monthly, 0.0)
    temp_lag_3 = get_profile_value(prev3, "temp_mid", month_profile_df, monthly, 0.0)

    cases_roll3_mean = float(np.mean([cases_lag_1, cases_lag_2, cases_lag_3]))
    cases_roll3_max = float(np.max([cases_lag_1, cases_lag_2, cases_lag_3]))

    month_sin = float(np.sin(2 * np.pi * month_num / 12))
    month_cos = float(np.cos(2 * np.pi * month_num / 12))

    feature_dict = {
        "rainfall": float(rainfall_now),
        "relative_humidity": float(humidity_now),
        "temp_mid": float(temp_now),
        "cases_lag_1": float(cases_lag_1),
        "cases_lag_2": float(cases_lag_2),
        "cases_lag_3": float(cases_lag_3),
        "rainfall_lag_1": float(rainfall_lag_1),
        "rainfall_lag_2": float(rainfall_lag_2),
        "rainfall_lag_3": float(rainfall_lag_3),
        "relative_humidity_lag_1": float(rh_lag_1),
        "relative_humidity_lag_2": float(rh_lag_2),
        "relative_humidity_lag_3": float(rh_lag_3),
        "temp_mid_lag_1": float(temp_lag_1),
        "temp_mid_lag_2": float(temp_lag_2),
        "temp_mid_lag_3": float(temp_lag_3),
        "cases_roll3_mean": float(cases_roll3_mean),
        "cases_roll3_max": float(cases_roll3_max),
        "month_sin": float(month_sin),
        "month_cos": float(month_cos),
    }

    return feature_dict


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Barangay Analytics",
    "Model Results",
    "Feature Transparency",
    "Forecast & Prediction"
])

with tab1:
    st.header("Historical Dengue Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Months", len(monthly))
    col2.metric(
        "Total Cases",
        int(monthly["CHSO_cases"].sum()) if "CHSO_cases" in monthly.columns else 0
    )
    col3.metric(
        "Average Monthly Cases",
        safe_metric_value(monthly["CHSO_cases"].mean()) if "CHSO_cases" in monthly.columns else "N/A"
    )

    st.subheader("What is the model predicting?")
    if meta:
        st.info(
            f"Problem Definition: {meta.get('problem_definition', 'Monthly outbreak classification')}  \n"
            f"Outbreak Definition: {meta.get('outbreak_definition', 'Not available')}"
        )
    else:
        st.info("The model predicts whether a month is outbreak or non-outbreak.")

    st.subheader("Monthly Dengue Cases")
    if {"Date", "CHSO_cases"}.issubset(monthly.columns):
        fig_line = px.line(
            monthly,
            x="Date",
            y="CHSO_cases",
            markers=True,
            title="Monthly Dengue Cases in Baguio City (CHSO)"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Year-Month Heatmap of Dengue Cases")
    if {"Year", "Month", "CHSO_cases"}.issubset(monthly.columns):
        heat = monthly.pivot_table(index="Year", columns="Month", values="CHSO_cases", aggfunc="sum")
        fig_heat = px.imshow(
            heat,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Year-Month Heatmap of Dengue Cases"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Rainfall vs Relative Humidity Sized by Dengue Cases")
    if {"rainfall", "relative_humidity", "CHSO_cases"}.issubset(monthly.columns):
        hover_cols = ["Date"]
        if "temp_mid" in monthly.columns:
            hover_cols.append("temp_mid")

        fig_bubble = px.scatter(
            monthly,
            x="rainfall",
            y="relative_humidity",
            size="CHSO_cases",
            color="CHSO_cases",
            hover_data=hover_cols,
            title="Rainfall vs Relative Humidity Sized by Dengue Cases"
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    st.subheader("Climate-Case Correlation")
    if climate_case_correlation is not None and not climate_case_correlation.empty:
        st.dataframe(climate_case_correlation, use_container_width=True)

        if {"feature", "pearson_corr_with_CHSO_cases"}.issubset(climate_case_correlation.columns):
            fig_corr = px.bar(
                climate_case_correlation,
                x="feature",
                y="pearson_corr_with_CHSO_cases",
                title="Climate-Case Correlation"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("climate_case_correlation.csv not found or empty.")

    st.subheader("Average Monthly Profile")
    if month_profile is not None and not month_profile.empty:
        st.dataframe(month_profile, use_container_width=True)

        if {"MonthName", "CHSO_cases"}.issubset(month_profile.columns):
            fig_month_profile = px.bar(
                month_profile,
                x="MonthName",
                y="CHSO_cases",
                title="Average CHSO Cases by Month"
            )
            st.plotly_chart(fig_month_profile, use_container_width=True)
    else:
        st.warning("month_profile.csv not found or empty.")

with tab2:
    st.header("Barangay Analytics")

    st.subheader("Highest-Risk barangay by month")
    if top_barangay_monthly is not None:
        st.dataframe(top_barangay_monthly, use_container_width=True)

    st.subheader("Highest-Risk Barangays by Dengue Cases")
    ranking_choice = st.radio(
        "Choose ranking view",
        ["Top 3 per year", "Top 3 overall"],
        horizontal=True
    )

    if ranking_choice == "Top 3 per year" and top3_barangays_yearly is not None and not top3_barangays_yearly.empty:
        fig_tree = px.treemap(
            top3_barangays_yearly,
            path=["Year", "Barangay"],
            values="Barangay_cases",
            color="Barangay_cases",
            title="Top 3 Barangays per Year Based on Dengue Cases"
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    elif ranking_choice == "Top 3 overall" and top3_barangays_overall is not None and not top3_barangays_overall.empty:
        fig_top3 = px.bar(
            top3_barangays_overall,
            x="Barangay",
            y="Barangay_cases",
            text="Barangay_cases",
            title="Top 3 Barangays Overall Based on Dengue Cases"
        )
        st.plotly_chart(fig_top3, use_container_width=True)

    st.subheader("Barangay monthly records")
    if barangay_monthly is not None:
        st.dataframe(barangay_monthly, use_container_width=True)

with tab3:
    st.header("Model Comparison")

    if meta:
        st.success(f"Selected Model: {meta.get('best_model', 'Unknown')}")

    if model_comparison is not None and not model_comparison.empty:
        display_cols = ["model", "accuracy", "precision", "recall", "f1_score"]
        available_display_cols = [c for c in display_cols if c in model_comparison.columns]
        st.dataframe(model_comparison[available_display_cols], use_container_width=True)

        st.subheader("Model Comparison by Metric")
        results_long = model_comparison.melt(
            id_vars="model",
            value_vars=["accuracy", "precision", "recall", "f1_score"],
            var_name="Metric",
            value_name="Score"
        )

        fig_model = px.bar(
            results_long,
            x="model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Model Comparison by Metric"
        )
        st.plotly_chart(fig_model, use_container_width=True)

    st.subheader("How to read the metrics")
    st.markdown(
        """
- **Accuracy**: overall percentage of correct predictions  
- **Precision**: when the model says outbreak, how often it is correct  
- **Recall / Sensitivity**: among real outbreak months, how many the model correctly catches  
- **F1 Score**: balance between precision and recall  

Higher values are better.  

A model can have high accuracy but still miss outbreak months.  
That is why **precision, recall, and F1 score** must also be checked.
"""
    )

    st.subheader("Confusion Matrix")
    if model_comparison is not None and "confusion_matrix" in model_comparison.columns:
        selected_model_for_cm = st.selectbox(
            "Select model to view confusion matrix",
            model_comparison["model"].tolist(),
            index=0
        )

        selected_row = model_comparison[model_comparison["model"] == selected_model_for_cm].iloc[0]
        cm_raw = selected_row["confusion_matrix"]

        if isinstance(cm_raw, str):
            cm = np.array(ast.literal_eval(cm_raw))
        else:
            cm = np.array(cm_raw)

        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0 (Non-outbreak)", "Actual 1 (Outbreak)"],
            columns=["Predicted 0 (Non-outbreak)", "Predicted 1 (Outbreak)"]
        )

        st.dataframe(cm_df, use_container_width=True)

        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto",
            title=f"Confusion Matrix - {selected_model_for_cm}"
        )
        fig_cm.update_xaxes(title="Predicted")
        fig_cm.update_yaxes(title="Actual")
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.warning("confusion_matrix column not found in model_comparison.csv.")

    st.subheader("How many months were correctly predicted?")
    if test_predictions is not None and not test_predictions.empty:
        total_test = len(test_predictions)
        correct_test = int(test_predictions["is_correct"].sum()) if "is_correct" in test_predictions.columns else None

        c1, c2 = st.columns(2)
        c1.metric("Test Set Months", total_test)
        c2.metric("Correct Predictions", correct_test if correct_test is not None else "N/A")

        st.dataframe(test_predictions, use_container_width=True)

with tab4:
    st.header("What contributed to the prediction?")

    st.subheader("Feature Importance")
    if feature_importance is not None and not feature_importance.empty:
        st.dataframe(feature_importance, use_container_width=True)

        fig_importance = px.bar(
            feature_importance.head(15),
            x="importance_mean",
            y="feature",
            orientation="h",
            title="Top Contributing Features"
        )
        st.plotly_chart(fig_importance, use_container_width=True)

    st.subheader("Sensitivity Analysis")
    if feature_sensitivity is not None and not feature_sensitivity.empty:
        st.dataframe(feature_sensitivity, use_container_width=True)

        fig_sens = px.bar(
            feature_sensitivity,
            x="feature",
            y="delta_probability",
            title="Effect of +10% Change in Climate Variable on Outbreak Probability"
        )
        st.plotly_chart(fig_sens, use_container_width=True)

    st.subheader("How to interpret this")
    st.markdown(
        """
- **Feature importance** shows which variables the model relied on most.  
- **Lagged case variables** mean the model uses recent dengue history.  
- **Sensitivity analysis** shows what happens to outbreak probability if rainfall, humidity, or temperature is changed.  
- These do not prove biological causation by themselves, but they help explain the model's behavior.
"""
    )

with tab5:
    st.header("Forecast")

    if forecast is not None and not forecast.empty:
        st.dataframe(forecast.head(30), use_container_width=True)

        if {"Date", "predicted_outbreak_probability"}.issubset(forecast.columns):
            fig_forecast = px.line(
                forecast,
                x="Date",
                y="predicted_outbreak_probability",
                markers=True,
                title="5-Year Forecasted Outbreak Probability"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

        if {"Year", "Month", "predicted_outbreak_probability"}.issubset(forecast.columns):
            forecast_heat = forecast.pivot_table(
                index="Year",
                columns="Month",
                values="predicted_outbreak_probability"
            )
            fig_forecast_heat = px.imshow(
                forecast_heat,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                title="Forecast Heatmap of Outbreak Probability"
            )
            st.plotly_chart(fig_forecast_heat, use_container_width=True)

    st.subheader("Top 3 Likely Barangays for Forecast Months")
    if forecast_top3_barangays is not None and not forecast_top3_barangays.empty:
        st.dataframe(forecast_top3_barangays, use_container_width=True)

        month_options = forecast_top3_barangays["Date"].dropna().astype(str).unique().tolist()
        selected_month = st.selectbox("Select forecast month for barangay ranking", month_options)

        selected_barangay_forecast = forecast_top3_barangays[
            forecast_top3_barangays["Date"].astype(str) == selected_month
        ].copy()

        fig_barangay_forecast = px.bar(
            selected_barangay_forecast,
            x="Barangay",
            y="predicted_barangay_cases_proxy",
            color="Barangay",
            title=f"Top 3 Likely Barangays - {selected_month}"
        )
        st.plotly_chart(fig_barangay_forecast, use_container_width=True)
    else:
        st.warning("forecast_top3_barangays.csv not found or empty.")

    st.subheader("Live Prediction")

    st.info(
        """
This section helps estimate whether a selected month is likely to be an **outbreak** or **non-outbreak** month.

To use this section:
- Select the **month** you want to evaluate.
- Adjust the **climate values** only if you have updated field data.
- Enter the **recent dengue case counts** from the last 3 months.
- Click **Predict**.

The system will automatically prepare the technical model inputs in the background.
"""
    )

    with st.expander("How to understand the inputs"):
        st.markdown(
            """
### Input Guide

**1. Select Month**  
Choose the month you want the system to evaluate.

**2. Rainfall**  
This refers to the rainfall amount for the selected month.  
Use observed rainfall data or an expected estimate if available.

**3. Relative Humidity**  
This refers to the average humidity level for the selected month.  
Higher humidity may support mosquito survival, but it does not always mean outbreak by itself.

**4. Temperature**  
This refers to the average temperature for the selected month.  
Temperature can affect mosquito breeding and virus development.

**5. Cases Last Month / 2 Months Ago / 3 Months Ago**  
These are the dengue case counts from the three months before the month you want to predict.  
The model uses recent case history because dengue trends often depend on recent transmission patterns.

### Important Reminder
You do **not** need to enter lag variables, rolling averages, or seasonal encoded values manually.  
The dashboard computes those automatically.

### How to read the result
- **Predicted Class = Outbreak** means the model sees outbreak-like conditions for that month.
- **Predicted Class = Non-outbreak** means the month is less likely to be outbreak-like.
- **Predicted Outbreak Probability** shows how strongly the model leans toward outbreak.
- **Likely Highest-Risk Barangays** helps identify which barangays may need closer attention.
"""
        )

    if model is None:
        st.warning("Model file not found. Live prediction is unavailable.")
    else:
        if meta and "feature_columns" in meta:
            feature_columns = meta["feature_columns"]
        else:
            feature_columns = DEFAULT_FEATURE_COLS

        month_options = list(range(1, 13))
        selected_month_num = st.selectbox(
            "Select Month",
            month_options,
            format_func=lambda x: f"{x} - {month_name_from_number(x)}",
            index=0
        )

        rainfall_default = get_profile_value(selected_month_num, "rainfall", month_profile, monthly, 0.0)
        humidity_default = get_profile_value(selected_month_num, "relative_humidity", month_profile, monthly, 0.0)
        temp_default = get_profile_value(selected_month_num, "temp_mid", month_profile, monthly, 0.0)

        rain_min, rain_max = 0.0, 1000.0
        rh_min, rh_max = get_reasonable_range(monthly, "relative_humidity", 60.0, 100.0)
        temp_min, temp_max = 10.0, 35.0  # realistic Baguio range
        cases_min, cases_max = get_reasonable_range(monthly, "CHSO_cases", 0.0, 3000.0)

        st.markdown("### Climate Inputs")
        climate_col1, climate_col2, climate_col3 = st.columns(3)

        st.info("""
        Climate inputs are based on the selected month.
        - Temperature is in Celsius (°C)
        - Rainfall is in millimeters (mm)
        - Humidity is in percentage (%)
        
        These values should represent the conditions for the month you want to predict.
        """)

        with climate_col1:
            st.markdown("**Rainfall**")
            rainfall_now = st.slider(
                "Current Rainfall (mm)",
                min_value=float(round(rain_min, 2)),
                max_value=float(round(rain_max, 2)),
                value=float(round(rainfall_default, 2)),
                step=1.0,
                help="Enter the rainfall amount for the selected month."
            )

        with climate_col2:
            st.markdown("**Relative Humidity**")
            humidity_now = st.slider(
                "Current Relative Humidity (%)",
                min_value=float(round(rh_min, 2)),
                max_value=float(round(rh_max, 2)),
                value=float(round(humidity_default, 2)),
                step=0.1,
                help="Enter the average relative humidity for the selected month."
            )

        with climate_col3:
            st.markdown("**Temperature**")
            temp_now = st.slider(
                "Current Temperature (°C)",
                min_value=float(temp_min),
                max_value=float(temp_max),
                value=float(round(temp_default, 2)),
                step=0.1,
                help="Enter the average temperature in Celsius (°C) for the selected month."
            )

        st.markdown("### Recent Dengue Case History")

        default_cases_lag_1 = float(monthly["CHSO_cases"].iloc[-1]) if "CHSO_cases" in monthly.columns and len(monthly) >= 1 else 0.0
        default_cases_lag_2 = float(monthly["CHSO_cases"].iloc[-2]) if "CHSO_cases" in monthly.columns and len(monthly) >= 2 else default_cases_lag_1
        default_cases_lag_3 = float(monthly["CHSO_cases"].iloc[-3]) if "CHSO_cases" in monthly.columns and len(monthly) >= 3 else default_cases_lag_2

        case_col1, case_col2, case_col3 = st.columns(3)

        with case_col1:
            cases_lag_1 = st.slider(
                "Cases Last Month",
                min_value=int(cases_min),
                max_value=int(cases_max),
                value=int(round(default_cases_lag_1)),
                step=1,
                help="Dengue cases recorded in the month immediately before the selected month."
            )

        with case_col2:
            cases_lag_2 = st.slider(
                "Cases 2 Months Ago",
                min_value=int(cases_min),
                max_value=int(cases_max),
                value=int(round(default_cases_lag_2)),
                step=1,
                help="Dengue cases recorded two months before the selected month."
            )

        with case_col3:
            cases_lag_3 = st.slider(
                "Cases 3 Months Ago",
                min_value=int(cases_min),
                max_value=int(cases_max),
                value=int(round(default_cases_lag_3)),
                step=1,
                help="Dengue cases recorded three months before the selected month."
            )

        live_month_number = st.selectbox(
            "Month Number for Barangay Ranking",
            list(range(1, 13)),
            index=selected_month_num - 1,
            help="This is used to rank likely higher-risk barangays for the chosen month."
        )

        auto_feature_values = build_live_prediction_features(
            month_num=selected_month_num,
            rainfall_now=rainfall_now,
            humidity_now=humidity_now,
            temp_now=temp_now,
            cases_lag_1=cases_lag_1,
            cases_lag_2=cases_lag_2,
            cases_lag_3=cases_lag_3,
            month_profile_df=month_profile
        )

        input_values = {}
        for feature in feature_columns:
            input_values[feature] = auto_feature_values.get(feature, 0.0)

        with st.expander("Show automatically prepared model inputs"):
            preview_df = pd.DataFrame([input_values])
            st.dataframe(preview_df, use_container_width=True)

        if st.button("Predict"):
            input_df = pd.DataFrame([input_values])
            pred = int(model.predict(input_df)[0])

            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(input_df)[0][1])
            else:
                prob = np.nan

            result_col1, result_col2 = st.columns(2)
            result_col1.success(f"Predicted Class: {outbreak_label_from_binary(pred)}")
            result_col2.info(
                f"Predicted Outbreak Probability: {prob:.4f}"
                if not pd.isna(prob) else
                "Probability not available"
            )

            st.markdown(
                """
**How to read this result**
- **0** means **Non-outbreak month**
- **1** means **Outbreak month**
- The probability is the model's estimated likelihood that the month is an outbreak month
- This is about **monthly outbreak classification**, not percentage of people or percentage of the population
"""
            )

            st.subheader("Likely Highest-Risk Barangays")

            if barangay_risk_profile is not None and not barangay_risk_profile.empty:
                barangay_live = barangay_risk_profile.copy()

                if "overall_share" not in barangay_live.columns:
                    barangay_live["overall_share"] = 0.0
                if "recent_share" not in barangay_live.columns:
                    barangay_live["recent_share"] = 0.0
                if "seasonal_share" not in barangay_live.columns:
                    barangay_live["seasonal_share"] = 0.0

                if "Month" in barangay_live.columns:
                    seasonal_subset = barangay_live[barangay_live["Month"] == live_month_number].copy()
                    if not seasonal_subset.empty:
                        barangay_live = seasonal_subset

                barangay_live["risk_score_raw"] = (
                    0.40 * barangay_live["overall_share"] +
                    0.35 * barangay_live["recent_share"] +
                    0.25 * barangay_live["seasonal_share"]
                )

                total_score = barangay_live["risk_score_raw"].sum()
                if total_score > 0:
                    barangay_live["risk_score"] = barangay_live["risk_score_raw"] / total_score
                else:
                    barangay_live["risk_score"] = 0.0

                city_cases_proxy = float(input_values.get("cases_roll3_mean", 0.0)) * (1 + (0 if pd.isna(prob) else prob))
                barangay_live["predicted_city_cases_proxy"] = city_cases_proxy
                barangay_live["predicted_barangay_cases_proxy"] = barangay_live["risk_score"] * city_cases_proxy
                barangay_live["predicted_barangay_label"] = "Higher Risk"

                keep_cols = [c for c in [
                    "Barangay",
                    "overall_share",
                    "recent_share",
                    "seasonal_share",
                    "risk_score_raw",
                    "risk_score",
                    "predicted_city_cases_proxy",
                    "predicted_barangay_cases_proxy",
                    "predicted_barangay_label"
                ] if c in barangay_live.columns]

                barangay_live_top3 = barangay_live.sort_values(
                    "predicted_barangay_cases_proxy",
                    ascending=False
                ).head(3)

                st.dataframe(barangay_live_top3[keep_cols], use_container_width=True)

                fig_live_barangay = px.bar(
                    barangay_live_top3,
                    x="Barangay",
                    y="predicted_barangay_cases_proxy",
                    color="Barangay",
                    title="Top 3 Likely Barangays for the Predicted Month"
                )
                st.plotly_chart(fig_live_barangay, use_container_width=True)
            else:
                st.warning("barangay_risk_profile.csv not found or empty.")

st.markdown("---")
st.caption("Baguio City Dengue Forecast Dashboard")
