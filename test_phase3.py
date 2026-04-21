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
