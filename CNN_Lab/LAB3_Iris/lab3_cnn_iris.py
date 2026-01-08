# =====================================================
# LEB 3: CNN on Iris.csv (Classification)
# Safe + Fast + Follow Assignment
# =====================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# -----------------------------------------------------
# 1) Load Iris.csv from directory
# -----------------------------------------------------
# แก้ path ให้ตรงเครื่องคุณ
data_path = "shared_data/iris/iris.csv"
df = pd.read_csv(data_path)

X = df.iloc[:, 0:4].values    # 4 features
y = df.iloc[:, 4].values     # class label

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape for CNN (samples, timesteps, channels)
X = X.reshape(X.shape[0], X.shape[1], 1)

n_classes = y_cat.shape[1]


# -----------------------------------------------------
# 2) Train / Test Split
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat,
    test_size=0.2,
    random_state=42
)


# -----------------------------------------------------
# 3) CNN Model Builder (1D CNN)
# -----------------------------------------------------
def build_cnn(n_conv_layers=1, n_filters=16, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(X.shape[1], 1)))

    for _ in range(n_conv_layers):
        model.add(Conv1D(
            filters=n_filters,
            kernel_size=2,
            activation='relu',
            padding='same'
        ))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# =====================================================
# PART A: Compare Learning Rates
# =====================================================
learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
lr_results = []

print("\n=== Learning Rate Comparison ===")
for lr in learning_rates:
    model = build_cnn(
        n_conv_layers=3,
        n_filters=32,
        learning_rate=lr
    )

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        verbose=0
    )

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    lr_results.append([lr, acc])

    print(f"Learning Rate = {lr:.0e} -> Accuracy = {acc*100:.2f}%")


df_lr = pd.DataFrame(
    lr_results,
    columns=["Learning Rate", "Accuracy"]
)

print("\nLearning Rate Results Table")
print(df_lr)


# =====================================================
# PART B: Compare Network Sizes
# =====================================================
layer_list  = [1, 3, 5]          # representative of 1–10
filter_list = [10, 50, 100]      # representative of 10–1000

net_results = []

print("\n=== Network Size Comparison ===")
for layers in layer_list:
    for filters in filter_list:
        model = build_cnn(
            n_conv_layers=layers,
            n_filters=filters,
            learning_rate=0.001
        )

        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=16,
            verbose=0
        )

        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        y_true = np.argmax(y_test, axis=1)

        acc = accuracy_score(y_true, y_pred)
        net_results.append([layers, filters, acc])

        print(f"{layers} layers x {filters} filters -> Accuracy = {acc*100:.2f}%")


df_net = pd.DataFrame(
    net_results,
    columns=["Conv Layers", "Filters per Layer", "Accuracy"]
).sort_values(by="Accuracy", ascending=False)

print("\nNetwork Size Results Table")
print(df_net.reset_index(drop=True))
