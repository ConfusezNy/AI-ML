# =========================================================
# LAB2: DCNN (VGG, ResNet, DenseNet, MobileNet)
# Face Recognition with Transfer Learning
# Display predictions COMPARATIVELY (subplot)
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    VGG16, ResNet50, DenseNet121, MobileNetV2
)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score, precision_score, recall_score


# =========================================================
# PATH CONFIG (robust, OS-independent)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "shared_data", "faces")
)

print("DATA_DIR:", DATA_DIR)
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError("Dataset not found: shared_data/faces")


# =========================================================
# EXPERIMENT CONFIG
# =========================================================
IMAGE_SIZES = [(50, 50), (100, 100)]
BATCH_SIZE = 16
EPOCHS = 10
MODEL_NAMES = ["VGG16", "ResNet50", "DenseNet121", "MobileNetV2"]

RESULTS = []


# =========================================================
# MODEL FACTORY
# =========================================================
def get_base_model(name, input_shape):
    if name == "VGG16":
        return VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    if name == "ResNet50":
        return ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    if name == "DenseNet121":
        return DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    if name == "MobileNetV2":
        return MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    raise ValueError("Unknown model name")


def build_model(base_model, num_classes):
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# =========================================================
# MAIN LOOP
# =========================================================
for img_size in IMAGE_SIZES:
    print(f"\n==============================")
    print(f"IMAGE SIZE: {img_size}")
    print(f"==============================")

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.3
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    num_classes = train_gen.num_classes
    class_names = list(train_gen.class_indices.keys())

    # ---- pick ONE sample image for comparison ----
    sample_img = val_gen[0][0][0]
    true_label = class_names[np.argmax(val_gen[0][1][0])]

    sample_predictions = []

    for model_name in MODEL_NAMES:
        print(f"\n--- Training {model_name} ---")

        base_model = get_base_model(
            model_name,
            input_shape=(img_size[0], img_size[1], 3)
        )

        model = build_model(base_model, num_classes)

        model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            verbose=1
        )

        # Save model
        model_filename = f"{model_name}_face_{img_size[0]}x{img_size[1]}.h5"
        model.save(model_filename)

        # Evaluation
        y_true = val_gen.classes
        y_pred = model.predict(val_gen, verbose=0)
        y_pred_cls = np.argmax(y_pred, axis=1)

        acc = accuracy_score(y_true, y_pred_cls)
        prec = precision_score(y_true, y_pred_cls, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred_cls, average="macro", zero_division=0)

        RESULTS.append([
            model_name,
            f"{img_size[0]}x{img_size[1]}",
            acc,
            prec,
            rec
        ])

        # Prediction for sample image
        sample_input = np.expand_dims(sample_img, axis=0)
        sample_pred = model.predict(sample_input, verbose=0)
        pred_label = class_names[np.argmax(sample_pred)]

        sample_predictions.append((model_name, pred_label))

    # =====================================================
    # DISPLAY ALL MODELS TOGETHER (subplot)
    # =====================================================
    fig, axes = plt.subplots(1, len(sample_predictions), figsize=(18, 4))

    for ax, (model_name, pred_label) in zip(axes, sample_predictions):
        ax.imshow(sample_img)
        ax.set_title(
            f"{model_name}\nPred: {pred_label}\nTrue: {true_label}",
            fontsize=10
        )
        ax.axis("off")

    plt.suptitle(f"Face Recognition Comparison ({img_size[0]}x{img_size[1]})")
    plt.tight_layout()
    plt.show()


# =========================================================
# RESULT TABLE
# =========================================================
df = pd.DataFrame(
    RESULTS,
    columns=["Model", "Image Size", "Accuracy", "Precision", "Recall"]
)

print("\n==============================")
print("PERFORMANCE SUMMARY")
print("==============================")
print(df)

df.to_csv("lab2_face_recognition_results.csv", index=False)
print("\nSaved: lab2_face_recognition_results.csv")
