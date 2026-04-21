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
