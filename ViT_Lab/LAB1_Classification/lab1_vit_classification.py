# ============================================
# LAB1: Vision Transformer for Image Classification
# Dataset: CIFAR-10
# Build ViT from scratch with Keras
# ============================================

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------
# 1) Hyperparameters
# --------------------------------------------
IMAGE_SIZE      = 32
PATCH_SIZE      = 8
NUM_PATCHES     = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 16
PROJECTION_DIM  = 64
NUM_HEADS       = 4
TRANSFORMER_LAYERS = 2
MLP_HEAD_UNITS  = [128, 64]
NUM_CLASSES     = 10
EPOCHS          = 10
BATCH_SIZE      = 64


# --------------------------------------------
# 2) Load & Prepare CIFAR-10
# --------------------------------------------
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_test_cat  = to_categorical(y_test,  NUM_CLASSES)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# --------------------------------------------
# 3) Patch Embedding Layer
# --------------------------------------------
class PatchEmbedding(layers.Layer):
    """Split image into patches and project to embedding dimension."""

    def __init__(self, patch_size, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection = layers.Dense(projection_dim)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        # Extract patches
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # Reshape: (batch, num_patches, patch_dim)
        patch_dim = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dim])
        return self.projection(patches)


# --------------------------------------------
# 4) Positional Encoding (Learnable)
# --------------------------------------------
class PositionalEncoding(layers.Layer):
    """Add learnable positional embeddings + CLS token."""

    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.cls_token = self.add_weight(
            shape=(1, 1, projection_dim),
            initializer="random_normal",
            trainable=True,
            name="cls_token",
        )
        self.position_embedding = layers.Embedding(
            input_dim=num_patches + 1,
            output_dim=projection_dim,
        )

    def call(self, patch_embeddings):
        batch_size = tf.shape(patch_embeddings)[0]
        # Repeat CLS token for every sample in batch
        cls_tokens = tf.broadcast_to(
            self.cls_token, [batch_size, 1, tf.shape(patch_embeddings)[-1]]
        )
        # Prepend CLS token
        patch_embeddings = tf.concat([cls_tokens, patch_embeddings], axis=1)
        # Add positional encoding (use tf.shape for dynamic shape in graph mode)
        seq_len = tf.shape(patch_embeddings)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        return patch_embeddings + self.position_embedding(positions)


# --------------------------------------------
# 5) Transformer Encoder Block
# --------------------------------------------
class TransformerBlock(layers.Layer):
    """Single Transformer Encoder: Multi-Head Self-Attention + FFN."""

    def __init__(self, projection_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(projection_dim * 2, activation="gelu"),
            layers.Dense(projection_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1   = layers.Dropout(0.1)
        self.dropout2   = layers.Dropout(0.1)

    def call(self, x, training=False):
        # Self-Attention + Residual
        attn_output = self.attention(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.layernorm1(x + attn_output)

        # Feed Forward + Residual
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.layernorm2(x + ffn_output)
        return x


# --------------------------------------------
# 6) Build Vision Transformer Model
# --------------------------------------------
def build_vit():
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # Patch Embedding
    patch_emb = PatchEmbedding(PATCH_SIZE, PROJECTION_DIM)(inputs)

    # Positional Encoding
    encoded = PositionalEncoding(NUM_PATCHES, PROJECTION_DIM)(patch_emb)

    # Transformer Encoder Blocks
    for _ in range(TRANSFORMER_LAYERS):
        encoded = TransformerBlock(PROJECTION_DIM, NUM_HEADS)(encoded)

    # Classification Head (use CLS token output)
    cls_output = encoded[:, 0]   # CLS token
    x = layers.Dense(128, activation="gelu")(cls_output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="gelu")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------
# 7) Train the Model
# --------------------------------------------
print("\n=== Building Vision Transformer ===")
vit_model = build_vit()
vit_model.summary()

print("\n=== Training ===")
history = vit_model.fit(
    X_train, y_train_cat,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
)


# --------------------------------------------
# 8) Evaluation
# --------------------------------------------
y_pred = np.argmax(vit_model.predict(X_test, verbose=0), axis=1)
y_true = y_test.flatten()
acc = accuracy_score(y_true, y_pred)

print(f"\n=== Test Accuracy: {acc*100:.2f}% ===")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))


# --------------------------------------------
# 9) Plot Training Curves
# --------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
axes[0].plot(history.history["val_accuracy"],  label="Val Accuracy")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"],     label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.suptitle("ViT Training on CIFAR-10")
plt.tight_layout()
plt.show()


# --------------------------------------------
# 10) Display Sample Predictions
# --------------------------------------------
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i])
    color = "green" if y_pred[i] == y_true[i] else "red"
    plt.title(
        f"True: {class_names[y_true[i]]}\nPred: {class_names[y_pred[i]]}",
        color=color, fontsize=8
    )
    plt.axis("off")

plt.suptitle("Sample Predictions — ViT on CIFAR-10")
plt.tight_layout()
plt.show()
