# CNN Lab 5 - Blood Cells Classification
# =====================================================
# LEB 5: CNN for Blood Cells Classification
# Safe + Fast + Follow Assignment
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


# -----------------------------------------------------
# 1) Dataset Path (แก้ให้ตรงเครื่อง)
# -----------------------------------------------------
DATASET_DIR = "shared_data/bloodcells"


# -----------------------------------------------------
# 2) Config
# -----------------------------------------------------
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 5


# -----------------------------------------------------
# 3) Load Dataset from Directory
# -----------------------------------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

test_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

num_classes = train_gen.num_classes
class_names = list(train_gen.class_indices.keys())

print(f"Number of classes: {num_classes}")
print(f"Classes: {class_names}")


# -----------------------------------------------------
# 4) CNN Model Builder
# -----------------------------------------------------
def build_cnn(n_conv_layers=1, n_filters=32, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))

    for _ in range(n_conv_layers):
        model.add(Conv2D(
            filters=n_filters,
            kernel_size=(3,3),
            activation="relu",
            padding="same"
        ))
        model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# =====================================================
# PART A: Compare Network Sizes
# =====================================================
layer_list  = [1, 3, 5]          # representative of 1–10
filter_list = [10, 50, 100]      # representative of 10–1000

net_results = []

print("\n=== Network Size Comparison ===")
for layers in layer_list:
    for filters in filter_list:
        print(f"Training CNN: {layers} layers, {filters} filters")

        model = build_cnn(
            n_conv_layers=layers,
            n_filters=filters,
            learning_rate=0.001
        )

        model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=EPOCHS,
            verbose=0
        )

        _, acc = model.evaluate(test_gen, verbose=0)
        net_results.append([layers, filters, acc])

        print(f"Accuracy = {acc*100:.2f}%")


df_net = pd.DataFrame(
    net_results,
    columns=["Conv Layers", "Filters per Layer", "Accuracy"]
).sort_values(by="Accuracy", ascending=False)

print("\n=== Network Size Results ===")
print(df_net.reset_index(drop=True))


# =====================================================
# PART B: Compare Learning Rates
# =====================================================
learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
lr_results = []

print("\n=== Learning Rate Comparison ===")
for lr in learning_rates:
    print(f"Training with learning rate = {lr}")

    model = build_cnn(
        n_conv_layers=3,
        n_filters=50,
        learning_rate=lr
    )

    model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS,
        verbose=0
    )

    _, acc = model.evaluate(test_gen, verbose=0)
    lr_results.append([lr, acc])

    print(f"Accuracy = {acc*100:.2f}%")


df_lr = pd.DataFrame(
    lr_results,
    columns=["Learning Rate", "Accuracy"]
)

print("\n=== Learning Rate Results ===")
print(df_lr)


# =====================================================
# (Optional) Sample Prediction Images
# =====================================================
print("\n=== Display Sample Predictions ===")

best_cfg = df_net.iloc[0]
best_model = build_cnn(
    int(best_cfg["Conv Layers"]),
    int(best_cfg["Filters per Layer"]),
    0.001
)

best_model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=EPOCHS,
    verbose=0
)

x_batch, y_batch = next(test_gen)
preds = best_model.predict(x_batch, verbose=0)

plt.figure(figsize=(10,4))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(x_batch[i])
    true_label = class_names[np.argmax(y_batch[i])]
    pred_label = class_names[np.argmax(preds[i])]
    color = "green" if true_label == pred_label else "red"
    plt.title(f"T:{true_label}\nP:{pred_label}", color=color, fontsize=9)
    plt.axis("off")

plt.suptitle("Sample Predictions (CNN Blood Cells)")
plt.tight_layout()
plt.show()
