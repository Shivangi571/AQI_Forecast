import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

CITIES = {
    "Delhi": (28.61, 77.21),
    "Ghaziabad": (28.67, 77.45),
    "Noida": (28.54, 77.39),
    "Greater Noida": (28.47, 77.50),
    "Gurugram": (28.46, 77.03),
    "Faridabad": (28.41, 77.31),
    "Meerut": (28.98, 77.71),
    "Sonipat": (28.99, 77.02),
}
# During training, city_code = position of the city in alphabetical order
CITY_CODE = {c: i for i, c in enumerate(sorted(CITIES))}

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://api.open-meteo.com/v1/forecast"

import os, glob
BASE = os.path.dirname(os.path.abspath(__file__))
found = glob.glob(os.path.join(BASE, "**", "aqi_model.joblib"), recursive=True)
if not found:
    st.error("Model file not found. Files I can see: " + str(os.listdir(BASE)))
    st.stop()
saved = joblib.load(found[0])
model, features = saved["model"], saved["features"]


@st.cache_data(ttl=3600)
def get_recent_data(city):
    """Download the last 5 days of pollution + weather for one city."""
    lat, lon = CITIES[city]
    common = {"latitude": lat, "longitude": lon, "past_days": 5,
              "forecast_days": 1, "timezone": "Asia/Kolkata"}
    aq = requests.get(AQ_URL, params={
        **common, "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,us_aqi"},
        timeout=60).json()["hourly"]
    wx = requests.get(WX_URL, params={
        **common, "hourly": "temperature_2m,relative_humidity_2m,"
        "wind_speed_10m,wind_direction_10m,precipitation"},
        timeout=60).json()["hourly"]
    df = pd.DataFrame(aq).merge(pd.DataFrame(wx), on="time")
    df["time"] = pd.to_datetime(df["time"])
    # keep only hours that have already happened
    now = pd.Timestamp.now(tz="Asia/Kolkata").floor("h").tz_localize(None)
    df = df[df["time"] <= now].set_index("time")
    return df.interpolate(limit=6).ffill()


def make_latest_row(df, city):
    """Create the same clues we used in training, for the latest hour."""
    g = df.copy()
    for lag in [1, 3, 6, 12, 24, 48]:
        g[f"aqi_lag{lag}"] = g["us_aqi"].shift(lag)
    g["aqi_roll24"] = g["us_aqi"].rolling(24).mean()
    g["aqi_change6"] = g["us_aqi"] - g["us_aqi"].shift(6)
    rad = np.deg2rad(g["wind_direction_10m"])
    g["wind_sin"], g["wind_cos"] = np.sin(rad), np.cos(rad)
    g["hour"] = g.index.hour
    g["dayofweek"] = g.index.dayofweek
    g["month"] = g.index.month
    g["city_code"] = CITY_CODE[city]
    return g.iloc[[-1]][features]


def category(aqi):
    if aqi <= 50: return "Good", "Enjoy outdoor activities."
    if aqi <= 100: return "Moderate", "Sensitive people should limit long outdoor exertion."
    if aqi <= 150: return "Unhealthy for sensitive groups", "Children, elderly and asthma patients should reduce outdoor activity."
    if aqi <= 200: return "Unhealthy", "Avoid long outdoor exercise. Consider a mask."
    if aqi <= 300: return "Very unhealthy", "Avoid outdoor exercise. Wear an N95 mask outside."
    return "Hazardous", "Stay indoors, use an air purifier, wear an N95 if you must go out."


st.title("NCR AQI Forecast")
st.write("Predicts the AQI **24 hours ahead** using an XGBoost model.")

city = st.selectbox("Choose a city", sorted(CITIES), index=sorted(CITIES).index("Ghaziabad"))

try:
    data = get_recent_data(city)
    row = make_latest_row(data, city)
    current = float(data["us_aqi"].iloc[-1])
    forecast = float(model.predict(row)[0])

    c1, c2 = st.columns(2)
    c1.metric("Current AQI", int(current))
    c2.metric("Forecast (next 24h)", int(forecast), delta=int(forecast - current),
              delta_color="inverse")

    label, advice = category(forecast)
    st.subheader(f"Tomorrow: {label}")
    st.info(advice)

    st.write("AQI over the last 48 hours")
    st.line_chart(data["us_aqi"].tail(48))
except Exception as e:
    st.error(f"Could not load data. Please try again later. ({e})")

st.caption("Data: Open-Meteo (CAMS air quality model and weather). "
           "These are model estimates, not ground-sensor readings.")
