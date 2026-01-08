# ============================================
# LAB1: CNN Basic for Classify Digits
# Safe + Fast + Follow Assignment
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


# --------------------------------------------
# 1) Load Digits Dataset
# --------------------------------------------
digits = load_digits()
X = digits.images        # shape: (n_samples, 8, 8)
y = digits.target        # labels 0–9

# Normalize and reshape for CNN
X = X / 16.0
X = X.reshape(-1, 8, 8, 1)
y_cat = to_categorical(y, num_classes=10)


# --------------------------------------------
# 2) Train / Test Split
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------
# 3) CNN Model Builder
# --------------------------------------------
def build_cnn(n_conv_layers=1, n_filters=32):
    model = Sequential()
    model.add(Input(shape=(8, 8, 1)))

    for _ in range(n_conv_layers):
        model.add(Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        ))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# --------------------------------------------
# 4) Train CNN Models (FAST & SAFE CONFIG)
# --------------------------------------------
layer_list  = [1, 3, 5]       # representative depths
filter_list = [10, 50, 100]   # representative sizes
epochs =  5                 # fast training

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


# --------------------------------------------
# 5) Results Table
# --------------------------------------------
df_results = pd.DataFrame(
    results,
    columns=["Conv Layers", "Filters per Layer", "Accuracy"]
).sort_values(by="Accuracy", ascending=False)

print("\n=== CNN Architecture Comparison ===")
print(df_results.reset_index(drop=True))


# --------------------------------------------
# 6) Display Sample Predictions (Best Model)
# --------------------------------------------
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

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    color = 'green' if y_pred_best[i] == y_true[i] else 'red'
    plt.title(f"True: {y_true[i]}\nPred: {y_pred_best[i]}", color=color)
    plt.axis('off')

plt.suptitle("Sample Predictions (CNN Digits)")
plt.tight_layout()
plt.show()
