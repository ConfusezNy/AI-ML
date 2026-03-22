# ============================================
# LAB3: Test Vision Transformer with Real Images
# Load real images from local folder or URLs
# Display predictions + attention heatmaps
# ============================================

import os
import sys
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


# --------------------------------------------
# 2) Helper: Load Image (file or URL)
# --------------------------------------------
def load_image(source):
    """Load image from file path or URL."""
    if source.startswith("http"):
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(source, timeout=10, headers=headers)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        return Image.open(source).convert("RGB")


# --------------------------------------------
# 3) Predict Single Image
# --------------------------------------------
def predict(image):
    """Return top-5 predictions and attention weights."""
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    top5_probs, top5_idx = torch.topk(probs, 5)

    results = []
    for prob, idx in zip(top5_probs, top5_idx):
        label = model.config.id2label[idx.item()]
        results.append((label, prob.item() * 100))

    return results, outputs.attentions


# --------------------------------------------
# 4) Attention Heatmap
# --------------------------------------------
def get_attention_map(attentions, image_size=224):
    """Extract CLS token attention map from last layer."""
    last_attn = attentions[-1][0].mean(dim=0).numpy()  # avg over heads
    cls_attn = last_attn[0, 1:]  # CLS -> patches
    grid = int(np.sqrt(len(cls_attn)))
    attn_map = cls_attn.reshape(grid, grid)

    # Normalize and resize
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    attn_resized = np.array(
        Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
            (image_size, image_size), Image.BILINEAR
        )
    )
    return attn_resized


# --------------------------------------------
# 5) Full Analysis for One Image
# --------------------------------------------
def analyze_image(source, title=None):
    """Full pipeline: load, predict, show results + attention."""
    if title is None:
        title = os.path.basename(source) if not source.startswith("http") else source[:50]

    print(f"\n{'='*60}")
    print(f"  Analyzing: {title}")
    print(f"{'='*60}")

    try:
        image = load_image(source)
    except Exception as e:
        print(f"  [ERROR] Cannot load image: {e}")
        return

    # Get predictions
    results, attentions = predict(image)
    print(f"\n  Top-5 Predictions:")
    for i, (label, conf) in enumerate(results):
        bar = "█" * int(conf / 2)
        print(f"    #{i+1}  {label:35s}  {conf:5.1f}%  {bar}")

    # Get attention map
    attn_map = get_attention_map(attentions)
    img_resized = image.resize((224, 224))

    # --- Plot ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Original Image
    axes[0].imshow(img_resized)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    # Top-5 Bar Chart
    labels = [r[0][:20] for r in results]
    confs  = [r[1] for r in results]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 5))
    axes[1].barh(labels[::-1], confs[::-1], color=colors)
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Confidence %")
    axes[1].set_title("Top-5 Predictions", fontsize=11)

    # Attention Heatmap
    axes[2].imshow(attn_map, cmap="hot")
    axes[2].set_title("Attention Map", fontsize=11)
    axes[2].axis("off")

    # Overlay
    axes[3].imshow(img_resized)
    axes[3].imshow(attn_map, cmap="hot", alpha=0.5)
    axes[3].set_title("Attention Overlay", fontsize=11)
    axes[3].axis("off")

    plt.suptitle(f"ViT Analysis — {title}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# --------------------------------------------
# 6) Main: Process Images
# --------------------------------------------
if __name__ == "__main__":

    # === Option A: Load from local folder ===
    TEST_DIR = os.path.join(os.path.dirname(__file__), "test_images")

    if os.path.isdir(TEST_DIR):
        print(f"\nLoading images from: {TEST_DIR}")
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for fname in sorted(os.listdir(TEST_DIR)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_ext:
                fpath = os.path.join(TEST_DIR, fname)
                analyze_image(fpath, title=fname)
    else:
        print(f"\n[INFO] No local test_images/ folder found at: {TEST_DIR}")
        print("[INFO] You can create it and put your images there.")
        print("[INFO] Using sample URLs instead...\n")

    # === Option B: Sample URLs (fallback) ===
    sample_urls = {
        "Cat": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        "Dog": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg",
        "Butterfly": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Papilio_machaon_-_Dovetail_Butterfly.jpg/800px-Papilio_machaon_-_Dovetail_Butterfly.jpg",
    }

    if not os.path.isdir(TEST_DIR):
        for name, url in sample_urls.items():
            analyze_image(url, title=name)

    print("\n=== LAB3 Complete ===")
    print("TIP: Put your own images in 'test_images/' folder and re-run!")
