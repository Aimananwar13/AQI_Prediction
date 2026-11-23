
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "city_day.csv"
MODEL_PATH = BASE_DIR / "models" / "aqi_model.pkl"
METRICS_PATH = BASE_DIR / "outputs" / "metrics.json"

st.set_page_config(
    page_title="India AQI Predictor",
    page_icon="🌍",
    layout="wide"
)

# AQI category based on Indian AQI ranges
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    return "Severe"


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, low_memory=False)
    data.columns = data.columns.str.strip()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["AQI"] = pd.to_numeric(data["AQI"], errors="coerce")
    return data


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# Check required files
if not DATA_PATH.exists():
    st.error("city_day.csv not found. Put it inside the dataset folder.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python train.py` first.")
    st.stop()

df = load_data()
bundle = load_model()

model = bundle["model"]
features = bundle["features"]
cities = bundle["cities"]

# Sidebar
st.sidebar.title("🌍 India AQI Predictor")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "AQI Prediction"]
)

st.title("🌍 India Air Quality Index Prediction")
st.caption("Explore historical air quality and predict AQI using machine learning.")

# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    st.header("Air Quality Dashboard")

    valid_aqi = df.dropna(subset=["AQI"]).copy()

    city_options = sorted(valid_aqi["City"].dropna().unique())

    selected_city = st.selectbox(
        "Select a city",
        ["All Cities"] + city_options
    )

    if selected_city != "All Cities":
        view = valid_aqi[valid_aqi["City"] == selected_city].copy()
    else:
        view = valid_aqi.copy()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    avg_aqi = view["AQI"].mean()
    max_aqi = view["AQI"].max()
    min_aqi = view["AQI"].min()

    col1.metric("Average AQI", f"{avg_aqi:.1f}")
    col2.metric("Maximum AQI", f"{max_aqi:.1f}")
    col3.metric("Minimum AQI", f"{min_aqi:.1f}")
    col4.metric("Records", f"{len(view):,}")

    st.divider()

    # AQI trend
    st.subheader("AQI Trend Over Time")

    trend = (
        view.dropna(subset=["Date"])
        .groupby("Date", as_index=False)["AQI"]
        .mean()
    )

    fig = px.line(
        trend,
        x="Date",
        y="AQI",
        title="Daily Average AQI",
        markers=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # City comparison
    st.subheader("City-wise Average AQI")

    city_aqi = (
        valid_aqi.groupby("City", as_index=False)["AQI"]
        .mean()
        .sort_values("AQI", ascending=False)
    )

    fig2 = px.bar(
        city_aqi,
        x="City",
        y="AQI",
        color="AQI",
        title="Average AQI by City"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # AQI category distribution
    st.subheader("AQI Category Distribution")

    category_data = valid_aqi.copy()
    category_data["Category"] = category_data["AQI"].apply(
        get_aqi_category
    )

    category_counts = (
        category_data["Category"]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Number of Days")
    )

    fig3 = px.pie(
        category_counts,
        names="Category",
        values="Number of Days",
        title="Distribution of AQI Categories"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # Dataset preview
    with st.expander("View dataset"):
        st.dataframe(view.head(100), use_container_width=True)


# ---------------- AQI PREDICTION ----------------
elif page == "AQI Prediction":

    st.header("Predict Air Quality Index")

    st.write(
        "Choose a city and enter pollutant readings. "
        "The trained Random Forest model will estimate the AQI."
    )

    city = st.selectbox("Select City", cities)

    # Use city-wise historical medians as sensible starting values
    city_data = df[df["City"] == city]

    st.subheader("Enter pollutant measurements")

    st.caption(
        "Values are prefilled using historical city medians. "
        "Change them to test different readings."
    )

    input_values = {}

    columns = st.columns(3)

    for i, feature in enumerate(features):
        if feature in city_data.columns:
            values = pd.to_numeric(
                city_data[feature], errors="coerce"
            ).dropna()
        else:
            values = pd.Series(dtype=float)

        default_value = float(values.median()) if not values.empty else 0.0

        with columns[i % 3]:
            input_values[feature] = st.number_input(
                feature,
                min_value=0.0,
                value=default_value,
                step=0.1,
                format="%.2f",
                key=f"input_{feature}"
            )

    if st.button("Predict AQI", type="primary"):

        input_data = pd.DataFrame([{
            "City": city,
            **input_values
        }])

        prediction = float(model.predict(input_data)[0])
        prediction = max(0, prediction)

        category = get_aqi_category(prediction)

        st.divider()
        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric("Predicted AQI", f"{prediction:.2f}")
        col2.metric("Air Quality Category", category)

        if prediction <= 50:
            st.success("Good air quality.")
        elif prediction <= 100:
            st.success("Satisfactory air quality.")
        elif prediction <= 200:
            st.warning("Moderate air quality.")
        elif prediction <= 300:
            st.warning("Poor air quality.")
        elif prediction <= 400:
            st.error("Very poor air quality.")
        else:
            st.error("Severe air quality.")

        st.info(
            "This is a model estimate based on historical data, "
            "not a live CPCB monitoring reading."
        )