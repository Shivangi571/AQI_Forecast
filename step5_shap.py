import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

# Load the 24-hour model made by step4
saved = joblib.load("models/xgb_24h.joblib")
model, features = saved["model"], saved["features"]

# Take 3000 random rows to explain
df = pd.read_csv("features.csv")
sample = df[features].sample(3000, random_state=42)

# SHAP calculates how much each clue pushed the prediction up or down
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(sample)

# Graph 1: detailed
shap.summary_plot(sv, sample, show=False)
plt.savefig("shap_detail.png", dpi=130, bbox_inches="tight")
plt.close()

# Graph 2: simple bar chart
shap.summary_plot(sv, sample, plot_type="bar", show=False)
plt.savefig("shap_bar.png", dpi=130, bbox_inches="tight")
plt.close()

# Print the top 10 most important clues
imp = pd.Series(abs(sv).mean(axis=0), index=features).sort_values(ascending=False)
print("Top 10 most important clues:")
print(imp.head(10).round(2))
print("Saved shap_detail.png and shap_bar.png")
