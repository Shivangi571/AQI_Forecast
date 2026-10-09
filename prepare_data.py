import numpy as np
import pandas as pd

HORIZON = 24  # we predict AQI 24 hours ahead

df = pd.read_csv("ncr_data.csv", parse_dates=["time"])
df = df.sort_values(["city", "time"]).reset_index(drop=True)

num_cols = [c for c in df.columns if c not in ("time", "city")]


def prepare_city(g):
    g = g.set_index("time")
    g[num_cols] = g[num_cols].interpolate(limit=6)  # fill small gaps

    for lag in [1, 3, 6, 12, 24, 48]:               # AQI of past hours
        g[f"aqi_lag{lag}"] = g["us_aqi"].shift(lag)

    g["aqi_roll24"] = g["us_aqi"].rolling(24).mean()
    g["aqi_change6"] = g["us_aqi"] - g["us_aqi"].shift(6)

    rad = np.deg2rad(g["wind_direction_10m"])        # wind direction
    g["wind_sin"], g["wind_cos"] = np.sin(rad), np.cos(rad)

    g["hour"] = g.index.hour                         # time clues
    g["dayofweek"] = g.index.dayofweek
    g["month"] = g.index.month

    g["target"] = g["us_aqi"].shift(-HORIZON)        # the answer to predict
    return g.reset_index()


df = pd.concat([prepare_city(g.assign(city=c)) for c, g in df.groupby("city")],
               ignore_index=True)
df["city_code"] = df["city"].astype("category").cat.codes

before = len(df)
df = df.dropna().reset_index(drop=True)
print("Rows before cleaning:", before, " after:", len(df))

df.to_csv("features.csv", index=False)
print("Saved features.csv")
