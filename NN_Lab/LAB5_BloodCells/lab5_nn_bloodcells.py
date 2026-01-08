import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from PIL import Image


base_path = "shared_data/bloodcells/bloodcells_dataset"
image_size = (32, 32)


def load_images(folder_path):
    X, y = [], []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    img = Image.open(os.path.join(label_path, file)).convert("L")
                    img = img.resize(image_size)
                    X.append(np.array(img).flatten())
                    y.append(label)
    return np.array(X), np.array(y)


X, y = load_images(base_path)


scaler = MinMaxScaler()
X = scaler.fit_transform(X)


encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

num_classes = len(np.unique(y_enc))
y_cat = to_categorical(y_enc, num_classes)


X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_enc
)


def train_nn(num_layers, num_nodes, lr):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(num_nodes, activation="relu"))

    for _ in range(num_layers - 1):
        model.add(Dense(num_nodes, activation="relu"))

    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy"
    )

    model.fit(
        X_train, y_train,
        epochs=3,
        batch_size=32,
        verbose=0
    )

    y_pred = model.predict(X_test, verbose=0)
    y_pred_label = np.argmax(y_pred, axis=1)
    y_true_label = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true_label, y_pred_label)
    return acc


acc_fixed = train_nn(5, 100, 0.001)
print("Accuracy (5 layers, 100 nodes):", acc_fixed)


layers_list = [1, 3, 5, 7, 10]
nodes_list = [10, 100, 1000]

results_structure = []

for l in layers_list:
    for n in nodes_list:
        acc = train_nn(l, n, 0.001)
        results_structure.append({
            "Layers": l,
            "Nodes": n,
            "Accuracy": acc
        })

df_structure = pd.DataFrame(results_structure)
print("\nCompare Network Sizes")
print(df_structure)


learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
results_lr = []

for lr in learning_rates:
    acc = train_nn(5, 100, lr)
    results_lr.append({
        "Learning Rate": lr,
        "Accuracy": acc
    })

df_lr = pd.DataFrame(results_lr)
print("\nCompare Learning Rates")
print(df_lr)
