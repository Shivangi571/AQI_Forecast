import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

HORIZONS = [6, 24, 48]
WINTER_MONTHS = [10, 11, 12, 1]

df = pd.read_csv("features.csv", parse_dates=["time"])
base_cols = [c for c in df.columns if c not in ("time", "city", "target")]


def add_target(data, h):
    future = data[["city", "time", "us_aqi"]].copy()
    future["time"] = future["time"] - pd.Timedelta(hours=h)
    future = future.rename(columns={"us_aqi": "target_h"})
    return data.merge(future, on=["city", "time"], how="left")


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


rows = []
os.makedirs("models", exist_ok=True)

for h in HORIZONS:
    print(f"\n===== Forecast {h} hours ahead =====")
    d = add_target(df, h).dropna(subset=["target_h"]).sort_values("time")

    cutoff = d["time"].quantile(0.8)
    train, test = d[d["time"] <= cutoff], d[d["time"] > cutoff]
    X_tr, y_tr = train[base_cols], train["target_h"]
    X_te, y_te = test[base_cols], test["target_h"]
    winter = test["time"].dt.month.isin(WINTER_MONTHS).values

    models = {
        "Persistence (baseline)": None,
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42),
    }

    for name, m in models.items():
        t0 = time.time()
        if m is None:
            pred = X_te["us_aqi"].values
        else:
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)
        row = {
            "horizon_h": h, "model": name,
            "RMSE": rmse(y_te, pred),
            "MAE": mean_absolute_error(y_te, pred),
            "Winter_RMSE": rmse(y_te[winter], pred[winter]) if winter.any() else np.nan,
        }
        rows.append(row)
        print(f"{name:24s} RMSE {row['RMSE']:6.1f}  MAE {row['MAE']:6.1f}  "
              f"Winter RMSE {row['Winter_RMSE']:6.1f}   ({time.time()-t0:.0f}s)")
        if name == "XGBoost":
            joblib.dump({"model": m, "features": base_cols}, f"models/xgb_{h}h.joblib")

results = pd.DataFrame(rows).round(2)
results.to_csv("results_table.csv", index=False)

print("\n\n===== FINAL TABLE (RMSE, lower is better) =====")
print(results.pivot(index="model", columns="horizon_h", values="RMSE"))

base = results[results.model == "Persistence (baseline)"].set_index("horizon_h")["RMSE"]
xgb = results[results.model == "XGBoost"].set_index("horizon_h")["RMSE"]
print("\nXGBoost improvement over baseline (RMSE):")
for h in HORIZONS:
    print(f"  {h}h: {100 * (1 - xgb[h] / base[h]):.1f}%")

plt.figure(figsize=(7, 4.5))
for name, g in results.groupby("model"):
    plt.plot(g["horizon_h"], g["RMSE"], marker="o", label=name)
plt.xlabel("Forecast horizon (hours)")
plt.ylabel("RMSE (lower is better)")
plt.xticks(HORIZONS)
plt.title("Model comparison across forecast horizons")
plt.legend()
plt.tight_layout()
plt.savefig("results_horizon.png", dpi=120)
print("\nSaved: results_table.csv, results_horizon.png, models/xgb_*h.joblib")
