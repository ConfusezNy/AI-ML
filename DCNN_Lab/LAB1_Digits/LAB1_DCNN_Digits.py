# =====================================================
# LAB1: DCNN for Digit Classification
# Models: VGG16, ResNet50, DenseNet121, MobileNetV2
# Single-file | No extra folders
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import tensorflow as tf
from tensorflow.keras.applications import (
    VGG16, ResNet50, DenseNet121, MobileNetV2
)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


# -----------------------------------------------------
# 1) Load Digits Dataset
# -----------------------------------------------------
digits = load_digits()
X = digits.images          # (n, 8, 8)
y = digits.target          # 0–9

X = X / 16.0
y_cat = to_categorical(y, 10)


# -----------------------------------------------------
# 2) Resize images for DCNN (8x8 → 100x100 / 200x200)
# -----------------------------------------------------
def resize_images(X, size):
    X_resized = np.zeros((X.shape[0], size, size, 3))
    for i in range(X.shape[0]):
        img = tf.image.resize(X[i][..., np.newaxis], (size, size))
        img = tf.image.grayscale_to_rgb(img)
        X_resized[i] = img.numpy()
    return X_resized


# -----------------------------------------------------
# 3) DCNN Model Builder
# -----------------------------------------------------
def build_dcnn(base_model_fn, img_size):
    base = base_model_fn(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(10, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# -----------------------------------------------------
# 4) Training Configuration
# -----------------------------------------------------
IMAGE_SIZES = [100, 200]
MODELS = {
    "VGG16": VGG16,
    "ResNet50": ResNet50,
    "DenseNet121": DenseNet121,
    "MobileNetV2": MobileNetV2
}

EPOCHS = 5
BATCH_SIZE = 32

results = []


# -----------------------------------------------------
# 5) Train / Evaluate DCNNs
# -----------------------------------------------------
for img_size in IMAGE_SIZES:
    print(f"\n==============================")
    print(f"IMAGE SIZE: {img_size}x{img_size}")
    print(f"==============================")

    X_img = resize_images(X, img_size)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_img, y_cat, test_size=0.3, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42,
        stratify=np.argmax(y_temp, axis=1)
    )

    for name, model_fn in MODELS.items():
        print(f"\nTraining {name}")

        model = build_dcnn(model_fn, img_size)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        # ---- Evaluation ----
        y_train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
        y_val_pred   = np.argmax(model.predict(X_val, verbose=0), axis=1)
        y_test_pred  = np.argmax(model.predict(X_test, verbose=0), axis=1)

        y_train_true = np.argmax(y_train, axis=1)
        y_val_true   = np.argmax(y_val, axis=1)
        y_test_true  = np.argmax(y_test, axis=1)

        acc  = accuracy_score(y_test_true, y_test_pred)
        prec = precision_score(y_test_true, y_test_pred, average="macro")
        rec  = recall_score(y_test_true, y_test_pred, average="macro")

        results.append([
            name, f"{img_size}x{img_size}",
            acc, prec, rec
        ])

        print(f"Accuracy={acc:.3f} Precision={prec:.3f} Recall={rec:.3f}")

        # ---- Sample Prediction ----
        plt.figure(figsize=(8, 3))
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(X_test[i])
            color = "green" if y_test_pred[i] == y_test_true[i] else "red"
            plt.title(f"T:{y_test_true[i]} P:{y_test_pred[i]}", color=color)
            plt.axis("off")
        plt.suptitle(f"{name} ({img_size}x{img_size})")
        plt.tight_layout()
        plt.show()


# -----------------------------------------------------
# 6) Result Table
# -----------------------------------------------------
df_results = pd.DataFrame(
    results,
    columns=["Model", "Image Size", "Accuracy", "Precision", "Recall"]
)

print("\n=== DCNN Performance Comparison ===")
print(df_results)
