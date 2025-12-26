import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 1) Load smoothed COVID-19 data from CSV (Thailand)
csv_path = "SVM_Lab/LAB6_COVID19/data/owid-covid-data.csv"  
country = "Thailand"

df = pd.read_csv(csv_path)
df = df[df["location"] == country].copy()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

series = df["new_cases_smoothed"].fillna(0).values
dates = df["date"].values

# 2) Create sliding window features
window = 14
X, y, target_dates = [], [], []

for i in range(window, len(series)):
    X.append(series[i-window:i])
    y.append(series[i])
    target_dates.append(dates[i])

X = np.array(X)
y = np.array(y)
target_dates = np.array(target_dates)

# 3) Train / Test split (last 90 days for test)
test_horizon = 90
X_train, X_test = X[:-test_horizon], X[-test_horizon:]
y_train, y_test = y[:-test_horizon], y[-test_horizon:]
dates_train, dates_test = target_dates[:-test_horizon], target_dates[-test_horizon:]

# 4) Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 5) Train SVR models with different kernels
kernels = ["linear", "poly", "rbf"]
models = {}
predictions = {}

print("SVR performance:")
for k in kernels:
    model = SVR(kernel=k, C=10.0, gamma="scale")
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Kernel = {k:6s}: RMSE = {rmse:.2f}")
    models[k] = model
    predictions[k] = y_pred

# 6) Use RBF model for 3-month forecasting
best_model = models["rbf"]

last_window = series[-window:].copy()
future_steps = 90
future_preds = []

future_dates = pd.date_range(
    df["date"].iloc[-1] + pd.Timedelta(days=1),
    periods=future_steps,
    freq="D"
)

for _ in range(future_steps):
    x_win = scaler.transform(last_window.reshape(1, -1))
    next_val = best_model.predict(x_win)[0]
    future_preds.append(next_val)
    last_window = np.roll(last_window, -1)
    last_window[-1] = next_val

future_preds = np.array(future_preds)

# 7) Plot results
plt.figure(figsize=(12, 6))

plt.plot(dates, series, label="Actual (Smoothed)", color="black")
plt.plot(dates_test, predictions["rbf"], label="SVR (Test)", color="blue")
plt.plot(future_dates, future_preds, label="SVR Forecast (Next 3 months)", color="orange")

plt.title(f"SVR Forecasting of Smoothed COVID-19 Cases – {country}")
plt.xlabel("Date")
plt.ylabel("New Cases (Smoothed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
