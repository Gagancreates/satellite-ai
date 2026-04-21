import os
from modules.report_generator import run_report_pipeline

assert os.path.exists("outputs/detections.json"), "Run Phase 2 first"
assert os.path.exists("outputs/change_stats.json"), "Run Phase 3 first"
assert os.path.exists("data/samples/before.png"), "Run Phase 1 first"
assert os.path.exists("data/samples/after.png"), "Run Phase 1 first"

# Note: if running standalone (not after Phase 3), Qwen2-VL will load fresh — takes 1-2 minutes
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
