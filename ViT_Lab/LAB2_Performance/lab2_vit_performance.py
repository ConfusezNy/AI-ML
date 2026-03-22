# ============================================
# LAB2: Vision Transformer Performance Analysis
# Use pretrained ViT from HuggingFace
# Test on high-resolution images
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

import torch
from transformers import ViTForImageClassification, ViTImageProcessor


# --------------------------------------------
# 1) Load Pretrained ViT Model
# --------------------------------------------
MODEL_NAME = "google/vit-base-patch16-224"

print("Loading pretrained ViT model...")
feature_extractor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model = ViTForImageClassification.from_pretrained(MODEL_NAME, output_attentions=True)
model.eval()
print(f"Model loaded: {MODEL_NAME}")
print(f"Number of classes: {model.config.num_labels}")


# --------------------------------------------
# 2) Sample Test Images (from URLs)
# --------------------------------------------
test_images = {
    "Cat": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
    "Dog": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg",
    "Bird": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Eopsaltria_australis_-_Mogo_Campground.jpg/640px-Eopsaltria_australis_-_Mogo_Campground.jpg",
    "Car": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Fiat_500_in_Turin_%28cropped%29.jpg/640px-Fiat_500_in_Turin_%28cropped%29.jpg",
    "Flower": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Sunflower_from_Silesia2.jpg/800px-Sunflower_from_Silesia2.jpg",
}


def load_image_from_url(url):
    """Download an image from URL and return as PIL Image."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, timeout=10, headers=headers)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


# --------------------------------------------
# 3) Predict and Analyze
# --------------------------------------------
def predict_image(image, name="image"):
    """Run prediction on a single image, return top-5 results + attentions."""
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    # Top-5 predictions
    top5_probs, top5_indices = torch.topk(probs, 5)
    top5_labels = [model.config.id2label[idx.item()] for idx in top5_indices]
    top5_confs  = [p.item() * 100 for p in top5_probs]

    print(f"\n{'='*50}")
    print(f"Image: {name}")
    print(f"{'='*50}")
    for i, (label, conf) in enumerate(zip(top5_labels, top5_confs)):
        bar = "█" * int(conf / 2)
        print(f"  #{i+1}  {label:30s}  {conf:5.1f}%  {bar}")

    return top5_labels, top5_confs, outputs.attentions


# --------------------------------------------
# 4) Run Predictions on All Test Images
# --------------------------------------------
all_results = {}
all_images  = {}
all_attentions = {}

for name, url in test_images.items():
    try:
        img = load_image_from_url(url)
        all_images[name] = img
        labels, confs, attentions = predict_image(img, name)
        all_results[name] = (labels, confs)
        all_attentions[name] = attentions
    except Exception as e:
        print(f"\n[ERROR] Could not process '{name}': {e}")


# --------------------------------------------
# 5) Visualize Predictions
# --------------------------------------------
n_images = len(all_images)
if n_images == 0:
    print("[ERROR] No images were loaded successfully.")
else:
    fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 5))

    if n_images == 1:
        axes = [axes]

    for ax, (name, img) in zip(axes, all_images.items()):
        ax.imshow(img)
        labels, confs = all_results[name]
        ax.set_title(
            f"{name}\nPred: {labels[0]}\n({confs[0]:.1f}%)",
            fontsize=10, fontweight="bold"
        )
        ax.axis("off")

    plt.suptitle("ViT Predictions on High-Resolution Images", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# --------------------------------------------
# 6) Attention Heatmap Visualization
# --------------------------------------------
def visualize_attention(image, attentions, name="image"):
    """Visualize attention from the last Transformer layer."""
    # attentions: tuple of (batch, heads, seq_len, seq_len) per layer
    # Use last layer, average over heads
    last_layer_attn = attentions[-1]  # (1, heads, seq+1, seq+1)
    attn = last_layer_attn[0].mean(dim=0).numpy()  # (seq+1, seq+1)

    # CLS token attention to patches (exclude CLS-to-CLS)
    cls_attn = attn[0, 1:]  # attention from CLS to each patch
    num_patches = int(np.sqrt(cls_attn.shape[0]))
    cls_attn = cls_attn.reshape(num_patches, num_patches)

    # Resize attention map to image size
    img_resized = image.resize((224, 224))
    attn_resized = np.array(
        Image.fromarray(
            (cls_attn / cls_attn.max() * 255).astype(np.uint8)
        ).resize((224, 224), Image.BILINEAR)
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_resized)
    axes[0].set_title(f"Original: {name}")
    axes[0].axis("off")

    axes[1].imshow(attn_resized, cmap="hot")
    axes[1].set_title("Attention Map")
    axes[1].axis("off")

    axes[2].imshow(img_resized)
    axes[2].imshow(attn_resized, cmap="hot", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.suptitle(f"Attention Heatmap — {name}", fontweight="bold")
    plt.tight_layout()
    plt.show()


# Show attention heatmaps for all images
for name in all_images:
    if name in all_attentions:
        visualize_attention(all_images[name], all_attentions[name], name)

print("\n=== LAB2 Complete ===")
