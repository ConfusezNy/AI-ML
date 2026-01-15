# =========================================================
# LAB5: DCNN (VGG, ResNet, DenseNet, MobileNet)
# COVID-19 Image Recognition with Grad-CAM
# Dataset structure:
#   train/ (split into train + valid)
#   test/
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

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
# PATH CONFIG (MATCH YOUR STRUCTURE)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "shared_data", "Covid19-dataset")
)

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR  = os.path.join(DATASET_DIR, "test")

for d in [TRAIN_DIR, TEST_DIR]:
    if not os.path.exists(d):
        raise FileNotFoundError(f"Dataset folder not found: {d}")

print("DATASET DIR:", DATASET_DIR)


# =========================================================
# EXPERIMENT CONFIG
# =========================================================
IMAGE_SIZES = [(120, 120), (224, 224)]
BATCH_SIZE = 16
EPOCHS = 3
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
# GRAD-CAM (SAFE)
# =========================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return None

    heatmap /= max_val
    return heatmap.numpy()


def overlay_gradcam(img, heatmap):
    if heatmap is None:
        return img

    heatmap = np.nan_to_num(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)


# =========================================================
# MAIN LOOP
# =========================================================
for img_size in IMAGE_SIZES:
    print(f"\n==============================")
    print(f"IMAGE SIZE: {img_size}")
    print(f"==============================")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    valid_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    num_classes = train_gen.num_classes
    class_names = list(train_gen.class_indices.keys())

    sample_img = test_gen[0][0][0]
    true_label = class_names[np.argmax(test_gen[0][1][0])]

    sample_predictions = []
    gradcam_images = []

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
            validation_data=valid_gen,
            verbose=1
        )

        model_filename = f"{model_name}_covid19_{img_size[0]}x{img_size[1]}.h5"
        model.save(model_filename)

        # -------- TEST EVALUATION --------
        y_true = test_gen.classes
        y_pred = model.predict(test_gen, verbose=0)
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

        # -------- SAMPLE + GRAD-CAM --------
        sample_input = np.expand_dims(sample_img, axis=0)
        pred_label = class_names[np.argmax(model.predict(sample_input, verbose=0))]

        last_conv_layer = [
            layer.name for layer in model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ][-1]

        heatmap = make_gradcam_heatmap(sample_input, model, last_conv_layer)
        overlay = overlay_gradcam(
            (sample_img * 255).astype(np.uint8),
            heatmap
        )

        sample_predictions.append((model_name, pred_label))
        gradcam_images.append(overlay)

    # =====================================================
    # DISPLAY COMPARISON + GRAD-CAM
    # =====================================================
    fig, axes = plt.subplots(2, len(MODEL_NAMES), figsize=(20, 8))

    for i, (model_name, pred_label) in enumerate(sample_predictions):
        axes[0, i].imshow(sample_img)
        axes[0, i].set_title(f"{model_name}\nPred: {pred_label}\nTrue: {true_label}")
        axes[0, i].axis("off")

        axes[1, i].imshow(gradcam_images[i])
        axes[1, i].set_title("Grad-CAM")
        axes[1, i].axis("off")

    plt.suptitle(f"COVID-19 Recognition + Grad-CAM ({img_size[0]}x{img_size[1]})")
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
print("PERFORMANCE SUMMARY (TEST SET)")
print("==============================")
print(df)

df.to_csv("lab5_covid19_recognition_results.csv", index=False)
print("\nSaved: lab5_covid19_recognition_results.csv")
