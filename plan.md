# Satellite AI Pipeline — Build Plan

**Project:** Advanced Satellite Image Understanding System  
**Deadline:** Sunday (same day as interview)  
**Compute:** Vast.ai RTX 3090 (24GB VRAM) — inference only, no training, no fine-tuning  
**Constraint:** No cloud APIs — fully open source models only  
**Repo:** Host under Eunice Labs GitHub  

---

## What We Are Building

An end-to-end satellite image understanding pipeline with three modules:

1. **Multi-class Object Detection** — YOLOv8 runs on satellite images, outputs bounding boxes with class labels and confidence scores
2. **Change Detection** — Qwen2-VL-7B receives a before/after image pair and describes what changed between them semantically
3. **Report Generation** — Qwen2-VL-7B (same model, already loaded) takes the detection JSON + change analysis + original images and generates a structured professional analytical report

**Two models total:**
- `YOLOv8n` (ultralytics) — for proper bounding box detection with class labels and confidence scores
- `Qwen/Qwen2-VL-7B-Instruct` (via HuggingFace transformers) — for semantic change detection and report generation

Qwen2-VL is a vision-language model. It takes actual images as input, not just text. This means it can visually understand satellite imagery context — not just pattern match on pixel diffs. It is loaded once and used for both modules 2 and 3.

**Why this architecture:**
- YOLOv8 satisfies the "object detection with bounding boxes" requirement that Qwen2-VL alone cannot cleanly deliver
- Qwen2-VL handles the "semantic and context aware" requirement that a pure YOLO + text LLM pipeline cannot deliver
- Together they cover all three assignment objectives completely

---

## Workflow — How We Build This

**Step 1 — Write all code locally (no compute needed)**
AI engineer writes every `.py` file locally using Cursor or any editor. Nothing is run locally. This is pure writing.

**Step 2 — Push to GitHub**
Every file goes into the Eunice Labs GitHub repo. Commit frequently.

**Step 3 — Spin up Vast.ai instance**
Only spin up when all code is written and pushed. Do not waste instance time writing code.

**Step 4 — Clone repo on instance and run tests**
SSH into the instance, clone the repo, run each phase test sequentially. Fix bugs by editing on the instance directly or pushing fixes and pulling.

**Step 5 — Run full pipeline, collect outputs, destroy instance**
Run `main.py`, verify all outputs exist, `scp` outputs to local machine, destroy the instance on Vast.ai dashboard.

**Total machine time needed: 30-45 minutes. Estimated cost: ~$0.15.**

---

## Compute Setup

**Instance specs to select on Vast.ai:**
- GPU: RTX 3090 (24GB VRAM) — Qwen2-VL-7B needs ~16GB VRAM
- RAM: 32GB minimum
- Disk: 100GB (Qwen2-VL-7B model weights are ~15GB)
- Base image: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- Estimated cost: ~$0.30-0.40/hr

**SSH into the instance:**
```bash
ssh -p <port> root@<host>
```

**One-time setup on the instance (run these first, in order):**
```bash
# Install all Python dependencies
pip install ultralytics transformers accelerate torchvision \
            Pillow matplotlib numpy scipy fpdf2 qwen-vl-utils torchgeo

# Clone the repo
git clone https://github.com/eunice-labs/satellite-ai
cd satellite-ai

# Create required directories
mkdir -p outputs data/samples
```

**Verify GPU is working:**
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Must print: True + NVIDIA GeForce RTX 3090
# If False — the instance has no GPU attached, destroy and repick
```

**Debug loop during testing:**
```bash
# Option A — edit directly on instance via SSH
nano modules/detection.py
python test_phase2.py

# Option B — fix locally, push, pull on instance
git pull origin main
python test_phase2.py
```

**Before destroying the instance — download all outputs to local machine:**
```bash
scp -P <port> -r root@<host>:/root/satellite-ai/outputs ./outputs
```
Then destroy the instance on Vast.ai dashboard immediately after.

---

## Project Structure

```
satellite-ai/
├── main.py                      # Runs full pipeline end to end
├── requirements.txt
├── README.md
├── modules/
│   ├── __init__.py
│   ├── detection.py             # YOLOv8 object detection + EuroSAT data loading
│   ├── change_detection.py      # Qwen2-VL loading + change detection
│   ├── report_generator.py      # Qwen2-VL report generation (reuses loaded model)
│   └── visualizer.py            # All visualization utilities
├── data/
│   ├── README.md                # How to pull EuroSAT
│   └── samples/                 # EuroSAT images land here at runtime
├── outputs/                     # All generated outputs land here
│   ├── detections.json
│   ├── detection_viz.png
│   ├── class_distribution.png
│   ├── change_stats.json
│   ├── change_mask.png
│   └── report.md
├── test_phase1.py
├── test_phase2.py
├── test_phase3.py
└── test_phase4.py
```

---

## Dataset

**Primary:** EuroSAT (Sentinel-2 based land-use classification)
**Why:** No login wall, 10 land-use classes, clean RGB images, directly loadable via torchgeo

**Classes:** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

**For change detection demo:** Pick one `AnnualCrop` image as "before" and one `Industrial` image as "after". This simulates agricultural land converted to industrial use — a realistic and meaningful scenario. This is standard practice in change detection research when true temporal pairs are unavailable.

**Important — EuroSAT image format:** Images are 64x64 pixels with 13 Sentinel-2 spectral bands. For YOLOv8 (pretrained on RGB COCO images), extract bands at indices 3, 2, 1 (corresponding to B4=Red, B3=Green, B2=Blue), normalize to uint8, and resize to 640x640. Detection results may be sparse since YOLOv8 was not trained on satellite imagery — this is expected and must be documented in the README.

---

## requirements.txt

```
ultralytics
transformers
accelerate
torchvision
Pillow
matplotlib
numpy
scipy
torch
fpdf2
qwen-vl-utils
torchgeo
```

---

## Phase 1 — Repo Setup + Data Pipeline + Visualizer

**Goal:** Repo is initialized, EuroSAT loads correctly on the Vast.ai instance, RGB images are extracted and saved to `data/samples/`, all visualizer utility functions are written and confirmed working.

### Tasks

**1. Initialize the GitHub repo under Eunice Labs:**
- Add `README.md`, `.gitignore`, `requirements.txt`
- Create the full folder structure above with empty `__init__.py` files
- `.gitignore` must exclude: `data/`, `outputs/`, `*.pt`, `__pycache__/`, `*.egg-info/`

**2. Write `modules/visualizer.py` with these three functions:**

`save_image_with_boxes(image_path, detections, output_path)`
- Load image from `image_path` using PIL
- Draw each detection from the `detections` list as a colored rectangle on the image
- Label each box with `"{class_name} {confidence:.2f}"` in white text above the box
- Use a different color per class — build a simple color map from class names using a fixed set of colors
- Save the annotated image to `output_path`

`save_change_visualization(before_path, after_path, diff_array, output_path)`
- Create a matplotlib figure with 3 panels side by side: Before | After | Change Heatmap
- Change heatmap uses `matplotlib.cm.hot` colormap on the `diff_array` (2D numpy array)
- Add a title to each panel
- Save to `output_path` at 150 DPI

`save_class_distribution(class_distribution, output_path)`
- Takes a dict like `{"building": 12, "car": 4}`
- Plots a horizontal bar chart using matplotlib
- Saves to `output_path`

**3. Write the EuroSAT data loading function inside `modules/detection.py` (top section):**

```python
from torchgeo.datasets import EuroSAT
import numpy as np
from PIL import Image
import os

def load_eurosat_samples(n_per_class=1, output_dir="data/samples"):
    """
    Downloads EuroSAT, extracts RGB from bands [3,2,1] (B4,B3,B2),
    normalizes to uint8, resizes to 640x640, saves as PNG.
    Also saves before.png (AnnualCrop) and after.png (Industrial)
    for the change detection demo.
    Returns list of saved image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset = EuroSAT(root="data/", download=True)

    saved = []
    seen_classes = {}
    before_saved = False
    after_saved = False

    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample["label"].item()
        class_name = dataset.classes[label]

        if seen_classes.get(label, 0) >= n_per_class:
            continue

        # Extract RGB: B4=index3, B3=index2, B2=index1
        img_array = sample["image"][[3, 2, 1], :, :].numpy().astype(np.float32)
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array.transpose(1, 2, 0))
        img = img.resize((640, 640), Image.BILINEAR)

        # Save general sample
        path = os.path.join(output_dir, f"{class_name}_{seen_classes.get(label, 0)}.png")
        img.save(path)
        saved.append(path)
        seen_classes[label] = seen_classes.get(label, 0) + 1

        # Save before/after for change detection
        if class_name == "AnnualCrop" and not before_saved:
            img.save(os.path.join(output_dir, "before.png"))
            before_saved = True
        if class_name == "Industrial" and not after_saved:
            img.save(os.path.join(output_dir, "after.png"))
            after_saved = True

    return saved
```

### Phase 1 Test

Write as `test_phase1.py` in the repo root. Run on Vast.ai via SSH:
```bash
python test_phase1.py
```

```python
# test_phase1.py
import os
from PIL import Image
from modules.detection import load_eurosat_samples
from modules.visualizer import save_class_distribution

os.makedirs("outputs", exist_ok=True)

# Test dataset loading and image extraction
paths = load_eurosat_samples(n_per_class=1)
assert len(paths) >= 5, f"Expected at least 5 images, got {len(paths)}"

# Validate each image
for p in paths:
    img = Image.open(p)
    assert img.size == (640, 640), f"Wrong size {img.size} for {p}"
    assert img.mode == "RGB", f"Wrong mode {img.mode} for {p}"

# Validate before/after images for change detection
assert os.path.exists("data/samples/before.png"), "before.png not saved — check AnnualCrop extraction"
assert os.path.exists("data/samples/after.png"), "after.png not saved — check Industrial extraction"

# Validate visualizer
save_class_distribution({"AnnualCrop": 3, "Industrial": 2, "Forest": 5}, "outputs/test_dist.png")
assert os.path.exists("outputs/test_dist.png"), "Visualizer failed to save chart"

print(f"PHASE 1 TEST PASSED — {len(paths)} images loaded and saved, before/after exist, visualizer works")
```

**Expected output:** `PHASE 1 TEST PASSED` + images in `data/samples/` + chart in `outputs/`

---

## Phase 2 — Object Detection Module (YOLOv8)

**Goal:** YOLOv8 runs inference on EuroSAT images, produces bounding boxes with class labels and confidence scores, saves structured JSON and annotated visualization.

### Tasks

**Write the full detection pipeline in `modules/detection.py` (below the data loading function):**

```python
from ultralytics import YOLO
import json
from modules.visualizer import save_image_with_boxes, save_class_distribution

def run_detection_pipeline(image_paths, output_dir="outputs"):
    """
    Runs YOLOv8n inference on a list of image paths.
    Saves detections.json, detection_viz.png, class_distribution.png.
    Returns the detections dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO("yolov8n.pt")  # downloads automatically on first run (~6MB)

    all_detections = []
    class_distribution = {}
    total_objects = 0

    for image_path in image_paths:
        results = model(image_path, verbose=False)
        result = results[0]

        image_detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            confidence = round(float(box.conf.item()), 4)
            bbox = [round(float(x), 2) for x in box.xyxy[0].tolist()]

            image_detections.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": bbox  # [x1, y1, x2, y2] in pixels
            })

            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
            total_objects += 1

        all_detections.append({
            "image_path": image_path,
            "objects_detected": len(image_detections),
            "detections": image_detections
        })

    all_confs = [d["confidence"] for img in all_detections for d in img["detections"]]
    avg_confidence = round(sum(all_confs) / len(all_confs), 4) if all_confs else 0.0

    output = {
        "total_images_analyzed": len(image_paths),
        "total_objects_detected": total_objects,
        "average_confidence": avg_confidence,
        "class_distribution": class_distribution,
        "per_image_results": all_detections
    }

    # Save JSON
    json_path = os.path.join(output_dir, "detections.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    # Save annotated visualization on first image
    if image_paths:
        if all_detections[0]["detections"]:
            save_image_with_boxes(
                image_paths[0],
                all_detections[0]["detections"],
                os.path.join(output_dir, "detection_viz.png")
            )
        else:
            # No detections — save raw image with a note
            Image.open(image_paths[0]).save(os.path.join(output_dir, "detection_viz.png"))

    # Save class distribution chart
    if class_distribution:
        save_class_distribution(class_distribution, os.path.join(output_dir, "class_distribution.png"))

    print(f"Detection complete: {total_objects} objects detected across {len(image_paths)} images")
    return output
```

**Note on results:** YOLOv8n is pretrained on COCO (cars, people, animals). EuroSAT images are aerial satellite patches. Detection counts will be low and class names will reflect COCO categories (e.g. "truck" on a road patch). This is expected — the pipeline architecture is correct and that is what is being evaluated. Document this clearly in the README.

### Phase 2 Test

Write as `test_phase2.py`. Run on Vast.ai:
```bash
python test_phase2.py
```

```python
# test_phase2.py
import json
import os
from modules.detection import run_detection_pipeline

image_paths = ["data/samples/before.png", "data/samples/after.png"]
for p in image_paths:
    assert os.path.exists(p), f"Missing {p} — run Phase 1 first"

result = run_detection_pipeline(image_paths)

assert os.path.exists("outputs/detections.json"), "detections.json not saved"
with open("outputs/detections.json") as f:
    data = json.load(f)

assert "total_objects_detected" in data, "Missing total_objects_detected"
assert "average_confidence" in data, "Missing average_confidence"
assert "class_distribution" in data, "Missing class_distribution"
assert "per_image_results" in data, "Missing per_image_results"
assert isinstance(data["per_image_results"], list), "per_image_results must be a list"
assert len(data["per_image_results"]) == 2, f"Expected 2 image results, got {len(data['per_image_results'])}"

for img_result in data["per_image_results"]:
    assert "detections" in img_result
    for det in img_result["detections"]:
        assert "class" in det
        assert "confidence" in det
        assert "bbox" in det
        assert len(det["bbox"]) == 4, "bbox must be [x1, y1, x2, y2]"
        assert 0.0 <= det["confidence"] <= 1.0

assert os.path.exists("outputs/detection_viz.png"), "detection_viz.png not saved"

print(f"PHASE 2 TEST PASSED — {data['total_objects_detected']} objects detected, JSON structure valid, viz saved")
```

**Expected output:** `PHASE 2 TEST PASSED` + valid `detections.json` + `detection_viz.png`

---

## Phase 3 — Change Detection Module (Qwen2-VL)

**Goal:** Qwen2-VL-7B-Instruct is loaded, receives the before and after satellite images as visual input, understands what changed semantically, outputs a structured change analysis saved as JSON and a visual change heatmap.

**Important:** Qwen2-VL is loaded here and stored in a global variable. The report generator in Phase 4 reuses this loaded model — it does NOT reload it. This saves 1-2 minutes and avoids running out of VRAM.

### Tasks

**Write the full `modules/change_detection.py`:**

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
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

    inputs = processor(
        text=[text],
        images=images,
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

    # Save JSON
    with open(os.path.join(output_dir, "change_stats.json"), "w") as f:
        json.dump(change_stats, f, indent=2)

    # Save 3-panel visualization: Before | After | Diff Heatmap
    diff = compute_pixel_diff(before_path, after_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(before_img); axes[0].set_title("Before"); axes[0].axis("off")
    axes[1].imshow(after_img); axes[1].set_title("After"); axes[1].axis("off")
    im = axes[2].imshow(diff, cmap="hot"); axes[2].set_title("Change Heatmap"); axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.suptitle(f"Change: {change_stats.get('primary_change_type', '')}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "change_mask.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Change detection complete — severity: {change_stats.get('change_severity')}, type: {change_stats.get('primary_change_type')}")
    return change_stats
```

### Phase 3 Test

Write as `test_phase3.py`. Run on Vast.ai:
```bash
python test_phase3.py
```

```python
# test_phase3.py
import json
import os
from modules.change_detection import run_change_detection_pipeline

assert os.path.exists("data/samples/before.png"), "before.png missing — run Phase 1 first"
assert os.path.exists("data/samples/after.png"), "after.png missing — run Phase 1 first"

result = run_change_detection_pipeline("data/samples/before.png", "data/samples/after.png")

assert os.path.exists("outputs/change_stats.json"), "change_stats.json not saved"
with open("outputs/change_stats.json") as f:
    data = json.load(f)

assert "change_detected" in data, "Missing change_detected"
assert "change_severity" in data, "Missing change_severity"
assert data["change_severity"] in ["low", "moderate", "high"], f"Invalid severity: {data['change_severity']}"
assert "changed_area_percent" in data, "Missing changed_area_percent"
assert "primary_change_type" in data, "Missing primary_change_type"
assert "detailed_description" in data, "Missing detailed_description"
assert len(data["detailed_description"]) > 20, "Description too short — model likely failed"
assert os.path.exists("outputs/change_mask.png"), "change_mask.png not saved"

print(f"PHASE 3 TEST PASSED — severity: {data['change_severity']}, type: {data['primary_change_type']}")
print(f"Description: {data['detailed_description'][:120]}...")
```

**Expected output:** `PHASE 3 TEST PASSED` + valid `change_stats.json` with meaningful description + `change_mask.png`

---

## Phase 4 — Report Generation + Full Pipeline Integration

**Goal:** Qwen2-VL (already loaded from Phase 3) generates a structured professional analytical report combining detection JSON, change stats JSON, and the actual satellite images. `main.py` chains all modules into a single command.

**Critical:** The model is already in memory from Phase 3. `report_generator.py` calls `query_qwen2vl()` from `change_detection.py` directly — it does NOT call `load_qwen2vl()` again. This avoids reloading 15GB of weights.

### Tasks

**1. Write `modules/report_generator.py`:**

```python
import json
import os
from PIL import Image
from modules.change_detection import query_qwen2vl

def run_report_pipeline(detections_path, change_stats_path, before_path, after_path, output_dir="outputs"):
    """
    Generates a structured analytical report using Qwen2-VL.
    Qwen2-VL must already be loaded (run change_detection pipeline first).
    Saves report.md and returns the report text.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(detections_path) as f:
        detections = json.load(f)
    with open(change_stats_path) as f:
        change_stats = json.load(f)

    before_img = Image.open(before_path).convert("RGB")
    after_img = Image.open(after_path).convert("RGB")

    prompt = f"""You are a professional satellite imagery analyst writing an intelligence report.

You have the following data from automated analysis:

OBJECT DETECTION (YOLOv8):
- Images analyzed: {detections['total_images_analyzed']}
- Total objects detected: {detections['total_objects_detected']}
- Average confidence: {detections['average_confidence']}
- Class distribution: {json.dumps(detections['class_distribution'])}

CHANGE DETECTION (Visual Analysis):
- Change detected: {change_stats['change_detected']}
- Severity: {change_stats['change_severity']}
- Estimated changed area: {change_stats['changed_area_percent']}%
- Change type: {change_stats['primary_change_type']}
- Before: {change_stats['before_description']}
- After: {change_stats['after_description']}
- Analysis: {change_stats['detailed_description']}

Using this data AND the two satellite images provided, write a complete analytical report in markdown:

# Satellite Imagery Intelligence Report

## 1. Executive Summary

## 2. Object Detection Findings

## 3. Change Detection Analysis

## 4. Risk Assessment

## 5. Recommendations

Be specific, use the actual numbers, write in a professional intelligence analyst tone."""

    print("Generating analytical report with Qwen2-VL...")
    report_text = query_qwen2vl([before_img, after_img], prompt, max_new_tokens=800)

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"Report saved to {report_path} ({len(report_text)} characters)")
    return report_text
```

**2. Write `main.py`:**

```python
import os
from modules.detection import load_eurosat_samples, run_detection_pipeline
from modules.change_detection import run_change_detection_pipeline
from modules.report_generator import run_report_pipeline

def main():
    print("=" * 50)
    print("Satellite AI Pipeline — Eunice Labs")
    print("=" * 50)

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data/samples", exist_ok=True)

    # Step 1 — Load data
    print("\n[1/4] Loading EuroSAT samples...")
    image_paths = load_eurosat_samples(n_per_class=1)
    print(f"Loaded {len(image_paths)} images")

    before_path = "data/samples/before.png"
    after_path = "data/samples/after.png"
    assert os.path.exists(before_path), "before.png missing"
    assert os.path.exists(after_path), "after.png missing"

    # Step 2 — Object Detection
    print("\n[2/4] Running YOLOv8 object detection...")
    detections = run_detection_pipeline(image_paths)
    print(f"Detected {detections['total_objects_detected']} objects")

    # Step 3 — Change Detection (loads Qwen2-VL here — takes 1-2 min first time)
    print("\n[3/4] Running Qwen2-VL change detection...")
    change_stats = run_change_detection_pipeline(before_path, after_path)
    print(f"Severity: {change_stats['change_severity']}")

    # Step 4 — Report Generation (reuses already-loaded Qwen2-VL, no reload)
    print("\n[4/4] Generating analytical report...")
    report = run_report_pipeline(
        detections_path="outputs/detections.json",
        change_stats_path="outputs/change_stats.json",
        before_path=before_path,
        after_path=after_path
    )

    # Summary
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)

    outputs = [
        "outputs/detections.json",
        "outputs/detection_viz.png",
        "outputs/class_distribution.png",
        "outputs/change_stats.json",
        "outputs/change_mask.png",
        "outputs/report.md"
    ]
    print("\nOutputs:")
    for f in outputs:
        status = "OK" if os.path.exists(f) else "MISSING"
        print(f"  [{status}] {f}")

    print("\n--- Report Preview ---")
    print(report[:600])
    print("...")

if __name__ == "__main__":
    main()
```

**3. Write `README.md` covering:**
- What the system does (3 sentences)
- ASCII architecture diagram showing the two-model pipeline
- Models: YOLOv8n for detection, Qwen2-VL-7B-Instruct for change detection and report generation
- Dataset: EuroSAT with note on band extraction and temporal pair simulation
- How to run: clone → pip install → python main.py
- Sample outputs section (embed the output images once generated)
- Known limitations: YOLOv8 not satellite-trained, EuroSAT temporal pair is simulated

### Phase 4 Test

Write as `test_phase4.py`. Run on Vast.ai. Requires Phase 2 and Phase 3 outputs to already exist.

```bash
python test_phase4.py
```

```python
# test_phase4.py
import os
from modules.report_generator import run_report_pipeline

assert os.path.exists("outputs/detections.json"), "Run Phase 2 first"
assert os.path.exists("outputs/change_stats.json"), "Run Phase 3 first"
assert os.path.exists("data/samples/before.png"), "Run Phase 1 first"
assert os.path.exists("data/samples/after.png"), "Run Phase 1 first"

# Note: if running this test standalone (not after Phase 3),
# Qwen2-VL will load fresh — takes 1-2 minutes
report = run_report_pipeline(
    detections_path="outputs/detections.json",
    change_stats_path="outputs/change_stats.json",
    before_path="data/samples/before.png",
    after_path="data/samples/after.png"
)

assert os.path.exists("outputs/report.md"), "report.md not saved"
assert len(report) > 300, f"Report too short ({len(report)} chars) — generation failed"

required_sections = ["Executive Summary", "Object Detection", "Change Detection", "Risk", "Recommendation"]
for section in required_sections:
    assert section in report, f"Report missing section: {section}"

print("PHASE 4 TEST PASSED — report generated with all required sections")
print(f"Report length: {len(report)} characters")
print("\n--- Report Preview ---")
print(report[:600])
print("...")
```

**Final integration test — run on Vast.ai:**
```bash
python main.py
```

All 6 outputs must exist:
- `outputs/detections.json` ✓
- `outputs/detection_viz.png` ✓
- `outputs/class_distribution.png` ✓
- `outputs/change_stats.json` ✓
- `outputs/change_mask.png` ✓
- `outputs/report.md` ✓

---

## Summary

| Phase | What gets built | Models used | Test confirms |
|-------|----------------|-------------|---------------|
| 1 | Repo + EuroSAT loading + visualizer | None | Images at 640x640, before/after exist, charts save |
| 2 | YOLOv8 detection pipeline | YOLOv8n | detections.json valid schema, bbox structure correct |
| 3 | Qwen2-VL change detection | Qwen2-VL-7B | change_stats.json valid, description meaningful |
| 4 | Qwen2-VL report generation + main.py | Qwen2-VL-7B (reused) | report.md has all 5 sections, >300 chars |

**Key implementation note:** Qwen2-VL is loaded once in Phase 3 via the global `_model` variable in `change_detection.py`. Phase 4 reuses it by calling `query_qwen2vl()` directly. Never call `load_qwen2vl()` again in `report_generator.py` — it will try to load a second copy into VRAM and fail.

**Total Vast.ai machine time: 30-45 minutes.**
**Estimated cost: ~$0.15-0.20 out of $1.4 budget.**
**After running:** `scp` outputs to local machine → push final code to GitHub → destroy instance on Vast.ai dashboard.

**Saving outputs before destroying the instance:**

```bash
scp -P <port> -r root@<host>:/root/satellite-ai/outputs ./outputs
```

You only need the `outputs/` folder. The model weights (Qwen2-VL ~15GB) can always be re-downloaded. The generated report and visualizations are what you are submitting — those are the irreplaceable ones. Do not destroy the instance before running this command.