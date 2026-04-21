from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
    (0, 128, 255), (255, 0, 128),
]

def _class_color(class_name, color_map):
    if class_name not in color_map:
        color_map[class_name] = _COLORS[len(color_map) % len(_COLORS)]
    return color_map[class_name]


def save_image_with_boxes(image_path, detections, output_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    color_map = {}

    for det in detections:
        color = _class_color(det["class"], color_map)
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{det['class']} {det['confidence']:.2f}"
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((x1, y1 - 18), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 18), label, fill=(255, 255, 255), font=font)

    img.save(output_path)


def save_change_visualization(before_path, after_path, diff_array, output_path):
    before = Image.open(before_path).convert("RGB")
    after = Image.open(after_path).convert("RGB")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(before)
    axes[0].set_title("Before")
    axes[0].axis("off")

    axes[1].imshow(after)
    axes[1].set_title("After")
    axes[1].axis("off")

    im = axes[2].imshow(diff_array, cmap="hot")
    axes[2].set_title("Change Heatmap")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_class_distribution(class_distribution, output_path):
    labels = list(class_distribution.keys())
    values = list(class_distribution.values())

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.5)))
    ax.barh(labels, values, color="steelblue")
    ax.set_xlabel("Count")
    ax.set_title("Detected Class Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
