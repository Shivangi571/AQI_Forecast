import time
from datetime import date, timedelta
import pandas as pd
import requests

cities = {
    "Delhi": (28.61, 77.21),
    "Ghaziabad": (28.67, 77.45),
    "Noida": (28.54, 77.39),
    "Greater Noida": (28.47, 77.50),
    "Gurugram": (28.46, 77.03),
    "Faridabad": (28.41, 77.31),
    "Meerut": (28.98, 77.71),
    "Sonipat": (28.99, 77.02),
}

START = "2022-08-01"
END = str(date.today() - timedelta(days=7))

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://archive-api.open-meteo.com/v1/archive"


def get_json(url, params):
    for attempt in range(5):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 429:
            print("Website is busy, waiting...")
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Website kept refusing. Try again later.")


all_data = []
for city, (lat, lon) in cities.items():
    print("Fetching", city, "...")
    aq = get_json(AQ_URL, {
        "latitude": lat, "longitude": lon,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,us_aqi",
        "start_date": START, "end_date": END, "timezone": "Asia/Kolkata",
    })
    wx = get_json(WX_URL, {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation",
        "start_date": START, "end_date": END, "timezone": "Asia/Kolkata",
    })
    df = pd.DataFrame(aq["hourly"]).merge(pd.DataFrame(wx["hourly"]), on="time")
    df["city"] = city
    all_data.append(df)
    time.sleep(3)

final = pd.concat(all_data, ignore_index=True)
final.to_csv("ncr_data.csv", index=False)
print("Done!")
print(final.shape)
print(final.groupby("city")["us_aqi"].mean())
