import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

df = pd.read_csv("features.csv", parse_dates=["time"]).sort_values("time")

# Split by time: old data for learning, newest 20% for testing
cutoff = df["time"].quantile(0.8)
train = df[df["time"] <= cutoff]
test = df[df["time"] > cutoff]
print("Training rows:", len(train), " Testing rows:", len(test))

features = [c for c in df.columns if c not in ("time", "city", "target")]
X_train, y_train = train[features], train["target"]
X_test, y_test = test[features], test["target"]


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


# Simple baseline: "tomorrow's AQI = today's AQI"
base = X_test["us_aqi"]
print("\nBASELINE  RMSE: %.1f   MAE: %.1f" % (rmse(y_test, base), mean_absolute_error(y_test, base)))

# The model
model = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("XGBOOST   RMSE: %.1f   MAE: %.1f" % (rmse(y_test, pred), mean_absolute_error(y_test, pred)))

improve = 100 * (1 - rmse(y_test, pred) / rmse(y_test, base))
print("\nImprovement over baseline: %.1f%%" % improve)

# Save the model
os.makedirs("models", exist_ok=True)
joblib.dump({"model": model, "features": features}, "models/aqi_model.joblib")

# Graph: actual vs predicted for Ghaziabad (last 30 days)
res = test[["time", "city"]].copy()
res["actual"], res["predicted"] = y_test.values, pred
g = res[res["city"] == "Ghaziabad"].tail(24 * 30)
plt.figure(figsize=(11, 4))
plt.plot(g["time"], g["actual"], label="Actual")
plt.plot(g["time"], g["predicted"], label="Predicted")
plt.title("Ghaziabad: actual vs predicted AQI")
plt.legend()
plt.tight_layout()
plt.savefig("results_plot.png")
print("Saved model and results_plot.png")
