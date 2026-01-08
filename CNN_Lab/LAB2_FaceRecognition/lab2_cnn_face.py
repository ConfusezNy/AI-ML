# =====================================================
# LEB 2: CNN on Face Recognition
# Safe + Fast + Follow Assignment
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# -----------------------------------------------------
# 1) Load Face Recognition Dataset (Library)
# -----------------------------------------------------
lfw = fetch_lfw_people(
    min_faces_per_person=70,   # ลด class ที่ข้อมูลน้อย
    resize=0.4
)

X = lfw.images        # (n_samples, h, w)
y = lfw.target
class_names = lfw.target_names

n_classes = len(class_names)
img_h, img_w = X.shape[1], X.shape[2]

# Normalize & reshape
X = X / 255.0
X = X.reshape(-1, img_h, img_w, 1)
y_cat = to_categorical(y, n_classes)

print(f"Dataset: {X.shape[0]} images, {n_classes} classes")


# -----------------------------------------------------
# 2) Train / Test Split
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------------------------------
# 3) CNN Model Builder
# -----------------------------------------------------
def build_cnn(n_conv_layers=1, n_filters=32):
    model = Sequential()
    model.add(Input(shape=(img_h, img_w, 1)))

    for _ in range(n_conv_layers):
        model.add(Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        ))

    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# -----------------------------------------------------
# 4) Train CNN Models (Representative Sizes)
# -----------------------------------------------------
layer_list  = [1, 3, 5]        # แทน 1–10 layers
filter_list = [10, 50, 100]    # แทน 10–1000 nodes
epochs = 5                    # เร็ว แต่ยังเห็นผล

results = []

for layers in layer_list:
    for filters in filter_list:
        print(f"Training CNN: {layers} Conv layers, {filters} filters")

        model = build_cnn(layers, filters)
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            verbose=0
        )

        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        y_true = np.argmax(y_test, axis=1)

        acc = accuracy_score(y_true, y_pred)
        results.append([layers, filters, acc])

        print(f"Accuracy = {acc*100:.2f}%")


# -----------------------------------------------------
# 5) Results Table
# -----------------------------------------------------
df_results = pd.DataFrame(
    results,
    columns=["Conv Layers", "Filters per Layer", "Accuracy"]
).sort_values(by="Accuracy", ascending=False)

print("\n=== CNN Architecture Comparison (Face Recognition) ===")
print(df_results.reset_index(drop=True))


# -----------------------------------------------------
# 6) Display Sample Predictions (Best Model)
# -----------------------------------------------------
best_cfg = df_results.iloc[0]

best_model = build_cnn(
    int(best_cfg["Conv Layers"]),
    int(best_cfg["Filters per Layer"])
)

best_model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=32,
    verbose=0
)

y_pred_best = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)

plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(X_test[i].reshape(img_h, img_w), cmap='gray')

    true_name = class_names[y_true[i]]
    pred_name = class_names[y_pred_best[i]]

    color = 'green' if y_true[i] == y_pred_best[i] else 'red'
    plt.title(f"True: {true_name}\nPred: {pred_name}", color=color, fontsize=9)
    plt.axis('off')

plt.suptitle("Sample Predictions (CNN Face Recognition)")
plt.tight_layout()
plt.show()
