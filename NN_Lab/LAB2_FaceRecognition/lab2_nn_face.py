import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

lfw = fetch_lfw_people(min_faces_per_person=50, resize=0.4)

X = lfw.data
y = lfw.target
target_names = lfw.target_names
image_shape = lfw.images[0].shape

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

configs = [
    (1, 50), (1, 100),
    (2, 100),
    (3, 100),
    (5, 100),
    (10, 100)
]

results = []

def build_model(layers, nodes):
    model = Sequential()
    model.add(Dense(nodes, activation="relu", input_shape=(X.shape[1],)))
    for _ in range(layers - 1):
        model.add(Dense(nodes, activation="relu"))
    model.add(Dense(len(target_names), activation="softmax"))
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

idx = np.random.choice(len(X_test), 6, replace=False)
sample_X = X_test[idx]
sample_y = y_test[idx]
sample_pred = np.argmax(best_model.predict(sample_X, verbose=0), axis=1)

plt.figure(figsize=(12, 4))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(sample_X[i].reshape(image_shape), cmap="gray")
    plt.title(
        f"True: {target_names[sample_y[i]]}\nPred: {target_names[sample_pred[i]]}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()
