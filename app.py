
import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("bike_demand_model.pkl")

st.set_page_config(
    page_title="Bike Sharing Demand Prediction",
    page_icon="🚲",
    layout="centered"
)

st.title("🚲 Bike Sharing Demand Prediction")
st.write(
    "Enter the weather and calendar details to predict "
    "the number of bike rentals."
)

col1, col2 = st.columns(2)

with col1:
    date = st.date_input("Date")
    hour = st.slider("Hour of Day", 0, 23, 12)
    temp = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=45.0,
        value=20.0
    )
    humidity = st.slider(
        "Humidity (%)",
        1, 100, 60
    )

with col2:
    season = st.selectbox(
        "Season",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Spring",
            2: "Summer",
            3: "Fall",
            4: "Winter"
        }[x]
    )

    weather = st.selectbox(
        "Weather Condition",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Clear / Few clouds",
            2: "Mist / Cloudy",
            3: "Light rain / Snow",
            4: "Heavy rain / Snow"
        }[x]
    )

    windspeed = st.number_input(
        "Windspeed",
        min_value=0.0,
        max_value=60.0,
        value=12.0
    )

    holiday = st.selectbox(
        "Holiday",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

workingday = st.selectbox(
    "Working Day",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

year = date.year
month = date.month
day = date.day

date_obj = pd.Timestamp(date)

day_of_week = date_obj.dayofweek
is_weekend = int(day_of_week >= 5)

hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

input_data = pd.DataFrame({
    'season': [season],
    'holiday': [holiday],
    'workingday': [workingday],
    'weather': [weather],
    'temp': [temp],
    'humidity': [humidity],
    'windspeed': [windspeed],
    'hour': [hour],
    'year': [year],
    'month': [month],
    'day': [day],
    'day_of_week': [day_of_week],
    'is_weekend': [is_weekend],
    'hour_sin': [hour_sin],
    'hour_cos': [hour_cos]
})


if st.button("🚲 Predict Bike Demand"):

    prediction = model.predict(input_data)[0]

    prediction = max(0, prediction)

    st.success(
        f"Predicted Bike Rentals: **{prediction:.0f} bikes per hour**"
    )

    st.info(
        "The prediction is generated using the trained Random Forest regression model."
    )
