# ===============================
# 0. IMPORTS
# ===============================
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from metrics_module import (
    calculate_r2,
    calculate_mse,
    calculate_rmse,
    calculate_mae,
    calculate_correlation
)

# ===============================
# 1. LOAD DATA
# ===============================
personal = pd.read_excel("personal_data.xlsx")
survey = pd.read_excel("survey_data.xlsx")

# ===============================
# 2. CLEANING
# ===============================
personal.columns = personal.columns.str.strip().str.replace(" ", "")
survey.columns = survey.columns.str.strip().str.replace(" ", "")

cols = ["ScreenTime", "StudyHours", "SleepHours"]

for col in cols:
    personal[col] = pd.to_numeric(personal[col], errors='coerce')
    survey[col] = pd.to_numeric(survey[col], errors='coerce')

personal = personal.dropna()
survey = survey.dropna()

# ===============================
# 3. NORMALIZATION (SURVEY ONLY)
# ===============================
scaler = MinMaxScaler()

survey_scaled = pd.DataFrame(
    scaler.fit_transform(survey[cols]),
    columns=cols
)

# ===============================
# 4. MODEL (POLYNOMIAL)
# ===============================
X = survey_scaled[["ScreenTime", "SleepHours"]]
y = survey_scaled["StudyHours"]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)

# ===============================
# 5. EVALUATION
# ===============================
print("\n----- MODEL PERFORMANCE -----")

r2 = calculate_r2(y, y_pred)

mse = calculate_mse(y, y_pred)

rmse = calculate_rmse(y, y_pred)

mae = calculate_mae(y, y_pred)

correlation = calculate_correlation(
    survey_scaled["ScreenTime"],
    survey_scaled["StudyHours"]
)

print("R² Score :", round(r2, 3))

print("MSE      :", round(mse, 3))

print("RMSE     :", round(rmse, 3))

print("MAE      :", round(mae, 3))

print("Correlation :", round(correlation, 3))

# ===============================
# 6. KPI VALUES
# ===============================
p_screen = personal["ScreenTime"].mean()
p_study = personal["StudyHours"].mean()

s_screen = survey["ScreenTime"].mean()
s_study = survey["StudyHours"].mean()

# ===============================
# 7. DASHBOARD (FINAL CLEAN)
# ===============================
fig = plt.figure(figsize=(16,9), facecolor="#0f1c2e")
gs = fig.add_gridspec(3, 4)

# -------------------------------
# Chart 1: Grouped Study Hours
# -------------------------------
ax1 = fig.add_subplot(gs[0, 0:2])

survey["StudyBins"] = pd.cut(survey["StudyHours"], bins=5)
avg = survey.groupby("StudyBins")["ScreenTime"].mean()

ax1.bar(range(len(avg)), avg.values, color="orange")

ax1.set_title("Screen Time vs Study Hours (Grouped)", color="white")
ax1.set_xlabel("Study Hour Groups", color="white")
ax1.set_ylabel("Avg Screen Time", color="white")
ax1.set_facecolor("#1e2a3a")
ax1.tick_params(colors='white')

# -------------------------------
# Chart 2: Trend
# -------------------------------
ax2 = fig.add_subplot(gs[0, 2:4])

survey_numeric = survey.select_dtypes(include=[np.number])
grouped = survey_numeric.groupby("ScreenTime").mean()

ax2.plot(grouped.index, grouped.index, label="Screen Time", linewidth=2)
ax2.plot(grouped.index, grouped["StudyHours"], label="Study Hours", linewidth=2)
ax2.plot(grouped.index, grouped["SleepHours"], label="Sleep Hours", linewidth=2)

ax2.set_title("Trend Analysis", color="white")
ax2.set_xlabel("Screen Time", color="white")
ax2.set_ylabel("Hours", color="white")
ax2.set_facecolor("#1e2a3a")
ax2.legend()
ax2.tick_params(colors='white')

# -------------------------------
# Chart 3: ScreenTime vs Sleep
# -------------------------------
ax3 = fig.add_subplot(gs[1:, 0:2])

sleep_avg = survey.groupby("ScreenTime")["SleepHours"].mean()

ax3.bar(sleep_avg.index, sleep_avg.values, color="skyblue")

ax3.set_title("Screen Time vs Sleep Hours", color="white")
ax3.set_xlabel("Screen Time", color="white")
ax3.set_ylabel("Avg Sleep Hours", color="white")
ax3.set_facecolor("#1e2a3a")
ax3.tick_params(colors='white')

# -------------------------------
# KPI PANEL (LEFT SHIFTED)
# -------------------------------
ax4 = fig.add_subplot(gs[1:, 2:4])
ax4.axis('off')

kpi_text = f"""
PERSONAL DATA
Screen Time Avg : {p_screen:.2f}
Study Hours Avg : {p_study:.2f}

SURVEY DATA
Screen Time Avg : {s_screen:.2f}
Study Hours Avg : {s_study:.2f}
"""

ax4.text(0.4, 0.3, kpi_text,
         fontsize=14,
         color="white",
         bbox=dict(facecolor="#1e2a3a",
                   edgecolor="#00FFFF",
                   boxstyle="round,pad=1.5"))

# -------------------------------
# TITLE
# -------------------------------
plt.suptitle("Impact of Mobile Usage on Productivity and Sleep",
             fontsize=20, color="white")

plt.tight_layout()
plt.show()

# ===============================
# 8. PREDICTION
# ===============================
new_screen = float(input("\nEnter Screen Time: "))
new_sleep = float(input("Enter Sleep Hours: "))

input_df = pd.DataFrame([[new_screen, 0, new_sleep]],
                        columns=["ScreenTime","StudyHours","SleepHours"])

scaled = scaler.transform(input_df)
scaled_df = pd.DataFrame(scaled, columns=cols)

X_new = scaled_df[["ScreenTime","SleepHours"]]
X_new_poly = poly.transform(X_new)

prediction = model.predict(X_new_poly)

# Convert back
dummy = np.zeros((1, len(cols)))
dummy[0, 1] = prediction[0]

real_value = scaler.inverse_transform(dummy)[0, 1]
real_value = max(0, real_value)

print("Predicted Study Hours:", round(real_value, 2))