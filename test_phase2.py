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
