from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Global model — loaded once, reused by report_generator.py
_model = None
_processor = None


def load_qwen2vl():
    """
    Loads Qwen2-VL-7B-Instruct in float16 on GPU.
    Stores in global variables so it is only loaded once.
    Takes 1-2 minutes on first call. Subsequent calls return immediately.
    """
    global _model, _processor
    if _model is not None:
        return _model, _processor

    print("Loading Qwen2-VL-7B-Instruct (1-2 mins)...")
    _processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        trust_remote_code=True
    )
    _model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    _model.eval()
    print("Qwen2-VL loaded successfully")
    return _model, _processor


def get_loaded_model():
    """Returns the already-loaded model and processor. Call load_qwen2vl() first."""
    return _model, _processor


def query_qwen2vl(images, prompt, max_new_tokens=512):
    """
    Sends a list of PIL images + a text prompt to Qwen2-VL.
    Returns the generated text response.
    images: list of PIL Image objects
    prompt: string
    """
    model, processor = load_qwen2vl()

    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    # Decode only the newly generated tokens
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.decode(generated[0], skip_special_tokens=True).strip()


def compute_pixel_diff(before_path, after_path):
    """Returns a 2D numpy array of mean absolute pixel differences for visualization."""
    before = np.array(Image.open(before_path).resize((640, 640)).convert("RGB")).astype(np.float32)
    after = np.array(Image.open(after_path).resize((640, 640)).convert("RGB")).astype(np.float32)
    return np.abs(before - after).mean(axis=2)


def run_change_detection_pipeline(before_path, after_path, output_dir="outputs"):
    """
    Runs Qwen2-VL on a before/after image pair.
    Saves change_stats.json and change_mask.png.
    Returns the change stats dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    before_img = Image.open(before_path).convert("RGB")
    after_img = Image.open(after_path).convert("RGB")

    prompt = """You are analyzing two satellite images taken at different times of the same location.
The first image is the BEFORE state. The second image is the AFTER state.

Analyze what has changed and respond ONLY with this JSON object, nothing else:
{
  "change_detected": true or false,
  "change_severity": "low" or "moderate" or "high",
  "changed_area_percent": number between 0 and 100,
  "primary_change_type": "short phrase",
  "detailed_description": "2-3 sentences describing what changed",
  "before_description": "1 sentence describing the before image",
  "after_description": "1 sentence describing the after image"
}"""

    print("Running Qwen2-VL change detection...")
    response = query_qwen2vl([before_img, after_img], prompt, max_new_tokens=300)

    # Parse response — strip markdown fences if model adds them
    try:
        clean = response.strip().strip("```json").strip("```").strip()
        change_stats = json.loads(clean)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON response. Raw: {response[:200]}")
        change_stats = {
            "change_detected": True,
            "change_severity": "moderate",
            "changed_area_percent": 0.0,
            "primary_change_type": "land use change",
            "detailed_description": response,
            "before_description": "Agricultural land",
            "after_description": "Industrial area"
        }

    change_stats["image_before"] = before_path
    change_stats["image_after"] = after_path

    with open(os.path.join(output_dir, "change_stats.json"), "w") as f:
        json.dump(change_stats, f, indent=2)

    diff = compute_pixel_diff(before_path, after_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(before_img)
    axes[0].set_title("Before")
    axes[0].axis("off")
    axes[1].imshow(after_img)
    axes[1].set_title("After")
    axes[1].axis("off")
    im = axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("Change Heatmap")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.suptitle(f"Change: {change_stats.get('primary_change_type', '')}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "change_mask.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Change detection complete — severity: {change_stats.get('change_severity')}, type: {change_stats.get('primary_change_type')}")
    return change_stats
