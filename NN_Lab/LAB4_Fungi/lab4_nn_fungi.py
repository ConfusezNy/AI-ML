import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from PIL import Image


train_path = "NN_Lab/LAB4_Fungi/data/train"
valid_path = "NN_Lab/LAB4_Fungi/data/valid"
test_path  = "NN_Lab/LAB4_Fungi/data/test"


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


X_train, y_train = load_images(train_path)
X_test, y_test = load_images(test_path)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)

num_classes = len(np.unique(y_train_enc))


y_train_cat = to_categorical(y_train_enc, num_classes)
y_test_cat = to_categorical(y_test_enc, num_classes)


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
        X_train, y_train_cat,
        epochs=3,
        batch_size=32,
        verbose=0
    )

    y_pred = model.predict(X_test, verbose=0)
    y_pred_label = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_test_enc, y_pred_label)
    return acc


acc_fixed = train_nn(num_layers=5, num_nodes=100, lr=0.001)
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
