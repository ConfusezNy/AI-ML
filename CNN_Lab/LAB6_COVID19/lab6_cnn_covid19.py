# CNN Lab 6 - COVID-19 Analysis
# =====================================================
# LEB 6: CNN with Smoothed COVID-19 Data (THA)
# Safe + Fast + Follow Assignment
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam


# -----------------------------------------------------
# 1) Load Smoothed COVID-19 Data (THA)
# -----------------------------------------------------
CSV_PATH = "shared_data/covid19/owid-covid-data.csv"   # แก้ path ให้ตรงเครื่อง
COUNTRY = "Thailand"

df = pd.read_csv(CSV_PATH)
df = df[df["location"] == COUNTRY].copy()
df["date"] = pd.to_datetime(df["date"])

# ใช้ new_cases_smoothed (ถ้ามี)
if "new_cases_smoothed" in df.columns:
    series = df["new_cases_smoothed"].fillna(0).values
else:
    # fallback: smooth เอง
    series = df["new_cases"].fillna(0).rolling(7, min_periods=1).mean().values

dates = df["date"].values


# -----------------------------------------------------
# 2) Normalize & Create Sliding Window
# -----------------------------------------------------
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

WINDOW = 14   # ใช้อดีต 14 วัน

def create_dataset(data, window):
    X, y, target_dates = [], [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
        target_dates.append(dates[i+window])
    return np.array(X), np.array(y), np.array(target_dates)

X, y, target_dates = create_dataset(series_scaled, WINDOW)

# reshape for CNN (samples, timesteps, channels)
X = X.reshape(X.shape[0], X.shape[1], 1)


# -----------------------------------------------------
# 3) Train / Test Split (Last 90 days = Test)
# -----------------------------------------------------
TEST_DAYS = 90
X_train, X_test = X[:-TEST_DAYS], X[-TEST_DAYS:]
y_train, y_test = y[:-TEST_DAYS], y[-TEST_DAYS:]
dates_train, dates_test = target_dates[:-TEST_DAYS], target_dates[-TEST_DAYS:]


# -----------------------------------------------------
# 4) CNN Model Builder (1D CNN)
# -----------------------------------------------------
def build_cnn(n_conv_layers=1, n_filters=32, lr=0.001):
    model = Sequential()
    model.add(Input(shape=(WINDOW, 1)))

    for _ in range(n_conv_layers):
        model.add(Conv1D(
            filters=n_filters,
            kernel_size=3,
            activation="relu",
            padding="same"
        ))

    model.add(Flatten())
    model.add(Dense(50, activation="relu"))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mse"
    )
    return model


# =====================================================
# PART A: Compare Network Sizes (Accuracy)
# =====================================================
layer_list  = [1, 3, 5]          # แทน 1–10 layers
filter_list = [10, 50, 100]      # แทน 10–1000 nodes

net_results = []

print("\n=== Network Size Comparison ===")
for layers in layer_list:
    for filters in filter_list:
        model = build_cnn(layers, filters, lr=0.001)
        model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

        y_pred = model.predict(X_test, verbose=0).flatten()

        # Accuracy แบบ direction (ขึ้น/ลง)
        acc = accuracy_score(
            (np.diff(y_test) > 0),
            (np.diff(y_pred) > 0)
        )

        net_results.append([layers, filters, acc])
        print(f"{layers} layers x {filters} filters -> Accuracy = {acc*100:.2f}%")

df_net = pd.DataFrame(
    net_results,
    columns=["Conv Layers", "Filters per Layer", "Accuracy"]
)

print("\n=== Network Size Results ===")
print(df_net.sort_values(by="Accuracy", ascending=False))


# =====================================================
# PART B: Train Best Model for Forecasting
# =====================================================
best_cfg = df_net.sort_values(by="Accuracy", ascending=False).iloc[0]

best_model = build_cnn(
    int(best_cfg["Conv Layers"]),
    int(best_cfg["Filters per Layer"]),
    lr=0.001
)

best_model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)


# -----------------------------------------------------
# 5) Forecast 3 Months & 6 Months
# -----------------------------------------------------
def forecast(model, last_window, steps):
    preds = []
    window = last_window.copy()
    for _ in range(steps):
        pred = model.predict(window.reshape(1, WINDOW, 1), verbose=0)[0,0]
        preds.append(pred)
        window = np.roll(window, -1)
        window[-1] = pred
    return np.array(preds)

last_window = X_test[-1].flatten()

pred_3m = forecast(best_model, last_window, 90)
pred_6m = forecast(best_model, last_window, 180)

# inverse scale
pred_3m = scaler.inverse_transform(pred_3m.reshape(-1,1)).flatten()
pred_6m = scaler.inverse_transform(pred_6m.reshape(-1,1)).flatten()


# -----------------------------------------------------
# 6) Plot: Actual vs Smoothed vs CNN Forecast
# -----------------------------------------------------
plt.figure(figsize=(12,6))

plt.plot(df["date"], series, label="Actual / Smoothed", color="black")
plt.plot(dates_test, scaler.inverse_transform(y_test.reshape(-1,1)),
         label="CNN Prediction (Test)", color="blue")

future_dates_3m = pd.date_range(df["date"].iloc[-1] + pd.Timedelta(days=1), periods=90)
future_dates_6m = pd.date_range(df["date"].iloc[-1] + pd.Timedelta(days=1), periods=180)

plt.plot(future_dates_3m, pred_3m, label="CNN Forecast (3 Months)", color="green")
plt.plot(future_dates_6m, pred_6m, label="CNN Forecast (6 Months)", color="red")

plt.title("CNN Forecasting of Smoothed COVID-19 Cases (Thailand)")
plt.xlabel("Date")
plt.ylabel("Cases")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
