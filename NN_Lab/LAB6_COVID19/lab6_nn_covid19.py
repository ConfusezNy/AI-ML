import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


csv_path = "shared_data/covid19/owid-covid-data.csv"
country = "Thailand"


df = pd.read_csv(csv_path)
df = df[df["location"] == country].copy()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

series = df["new_cases_smoothed"].fillna(0).values
dates = df["date"].values


scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()


window = 14
X, y = [], []

for i in range(window, len(series_scaled)):
    X.append(series_scaled[i - window:i])
    y.append(series_scaled[i])

X = np.array(X)
y = np.array(y)


test_horizon = 90
X_train, X_test = X[:-test_horizon], X[-test_horizon:]
y_train, y_test = y[:-test_horizon], y[-test_horizon:]
dates_test = dates[-test_horizon:]


def build_model():
    model = Sequential()
    model.add(Input(shape=(window,)))
    for _ in range(10):
        model.add(Dense(100, activation="relu"))
    model.add(Dense(1))
    model.compile(
        optimizer=Adam(0.001),
        loss="mse"
    )
    return model


model = build_model()
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)


y_pred_test = model.predict(X_test, verbose=0).flatten()


direction_true = np.sign(np.diff(y_test, prepend=y_test[0]))
direction_pred = np.sign(np.diff(y_pred_test, prepend=y_pred_test[0]))

accuracy = accuracy_score(direction_true > 0, direction_pred > 0)
print("Directional Accuracy:", accuracy)


last_window = series_scaled[-window:].copy()

def forecast(model, last_window, steps):
    preds = []
    win = last_window.copy()
    for _ in range(steps):
        next_val = model.predict(win.reshape(1, -1), verbose=0)[0, 0]
        preds.append(next_val)
        win = np.roll(win, -1)
        win[-1] = next_val
    return np.array(preds)


forecast_3m = forecast(model, last_window, 90)
forecast_6m = forecast(model, last_window, 180)


forecast_3m_inv = scaler.inverse_transform(forecast_3m.reshape(-1, 1)).flatten()
forecast_6m_inv = scaler.inverse_transform(forecast_6m.reshape(-1, 1)).flatten()


plt.figure(figsize=(12, 6))
plt.plot(df["date"], series, label="Actual (Smoothed)")
plt.plot(dates_test, scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten(),
         label="NN Test Prediction")
future_dates_3m = pd.date_range(df["date"].iloc[-1] + pd.Timedelta(days=1), periods=90)
future_dates_6m = pd.date_range(df["date"].iloc[-1] + pd.Timedelta(days=1), periods=180)
plt.plot(future_dates_3m, forecast_3m_inv, label="NN Forecast 3M")
plt.plot(future_dates_6m, forecast_6m_inv, label="NN Forecast 6M")
plt.xlabel("Date")
plt.ylabel("New Cases")
plt.title("NN Forecasting of Smoothed COVID-19 Cases (Thailand)")
plt.legend()
plt.tight_layout()
plt.show()
