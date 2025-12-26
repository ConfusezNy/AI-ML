import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

digits = load_digits()
X = digits.data
y = digits.target

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

configs = [
    (1, 10), (1, 50), (1, 100),
    (2, 50), (2, 100),
    (3, 100),
    (5, 100),
    (5, 500),
    (10, 100)
]

results = []

def build_model(layers, nodes):
    model = Sequential()
    model.add(Dense(nodes, activation="relu", input_shape=(64,)))
    for _ in range(layers - 1):
        model.add(Dense(nodes, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.compile(
        optimizer=Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

for layers, nodes in configs:
    model = build_model(layers, nodes)
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    results.append([layers, nodes, acc])

results_df = pd.DataFrame(
    results,
    columns=["Layers", "Nodes_per_Layer", "Accuracy"]
)

print(results_df.sort_values(by="Accuracy", ascending=False))

best_layers, best_nodes, _ = results_df.sort_values(
    by="Accuracy", ascending=False
).iloc[0]

best_model = build_model(int(best_layers), int(best_nodes))
best_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

sample_idx = np.random.choice(len(X_test), 8, replace=False)
sample_X = X_test[sample_idx]
sample_y = y_test[sample_idx]
sample_pred = np.argmax(best_model.predict(sample_X, verbose=0), axis=1)

plt.figure(figsize=(12, 4))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(sample_X[i].reshape(8, 8), cmap="gray")
    plt.title(f"True: {sample_y[i]} / Pred: {sample_pred[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
