# =====================================================
# LEB 4: CNN on Microscopic Fungi Classification
# Version: With Sample Prediction Images
# =====================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


# -----------------------------------------------------
# 1) Dataset Path
# -----------------------------------------------------
DATASET_DIR = "shared_data/fungi"


# -----------------------------------------------------
# 2) Config
# -----------------------------------------------------
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 5


# -----------------------------------------------------
# 3) Load Dataset
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

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

num_classes = train_gen.num_classes
class_names = list(train_gen.class_indices.keys())


# -----------------------------------------------------
# 4) CNN Model Builder
# -----------------------------------------------------
def build_cnn(n_conv_layers=1, n_filters=32, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))

    for _ in range(n_conv_layers):
        model.add(Conv2D(n_filters, (3,3), activation="relu", padding="same"))
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
# PART A: Network Size Comparison
# =====================================================
layer_list  = [1, 3, 5]
filter_list = [10, 50, 100]

results = []

for layers in layer_list:
    for filters in filter_list:
        model = build_cnn(layers, filters, 0.001)
        model.fit(train_gen, validation_data=val_gen,
                  epochs=EPOCHS, verbose=0)
        _, acc = model.evaluate(val_gen, verbose=0)
        results.append([layers, filters, acc])

df_net = pd.DataFrame(
    results,
    columns=["Conv Layers", "Filters", "Accuracy"]
).sort_values(by="Accuracy", ascending=False)

print("\n=== Network Size Results ===")
print(df_net.reset_index(drop=True))


# =====================================================
# PART B: Learning Rate Comparison
# =====================================================
learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
lr_results = []

for lr in learning_rates:
    model = build_cnn(3, 50, lr)
    model.fit(train_gen, validation_data=val_gen,
              epochs=EPOCHS, verbose=0)
    _, acc = model.evaluate(val_gen, verbose=0)
    lr_results.append([lr, acc])

df_lr = pd.DataFrame(
    lr_results,
    columns=["Learning Rate", "Accuracy"]
)

print("\n=== Learning Rate Results ===")
print(df_lr)


# =====================================================
# PART C: Sample Prediction Images (ONLY ONCE)
# =====================================================
print("\n=== Display Sample Predictions ===")

best_cfg = df_net.iloc[0]
best_model = build_cnn(
    int(best_cfg["Conv Layers"]),
    int(best_cfg["Filters"]),
    0.001
)

best_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=0
)

# ดึง batch เดียว
x_batch, y_batch = next(val_gen)
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

plt.suptitle("Sample Predictions (CNN Fungi)")
plt.tight_layout()
plt.show()
