import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

df = pd.read_csv("shared_data/iris/Iris.csv")

X = df.drop(columns=["Species"]).values
y = df["Species"].values

encoder = LabelEncoder()
y = encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def build_model(learning_rate):
    model = Sequential()
    model.add(Dense(50, activation="relu", input_shape=(X.shape[1],)))
    for _ in range(4):
        model.add(Dense(50, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    model.compile(
        optimizer=Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
results = []

for lr in learning_rates:
    model = build_model(lr)
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    results.append([lr, acc])

results_df = pd.DataFrame(
    results,
    columns=["Learning_Rate", "Accuracy"]
)

print(results_df)

best_lr = results_df.sort_values(by="Accuracy", ascending=False).iloc[0][0]

best_model = build_model(best_lr)
best_model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

idx = np.random.choice(len(X_test), 5, replace=False)
sample_X = X_test[idx]
sample_y = y_test[idx]
sample_pred = np.argmax(best_model.predict(sample_X, verbose=0), axis=1)

for i in range(5):
    true_label = encoder.inverse_transform([sample_y[i]])[0]
    pred_label = encoder.inverse_transform([sample_pred[i]])[0]
    print(f"True: {true_label} | Pred: {pred_label}")
