import os
from modules.detection import run_detection_pipeline
from modules.change_detection import run_change_detection_pipeline
from modules.report_generator import run_report_pipeline


def main():
    print("=" * 50)
    print("Satellite AI Pipeline — Eunice Labs")
    print("=" * 50)

    os.makedirs("outputs", exist_ok=True)

    before_path = "before.jpg"
    after_path = "after.png"
    assert os.path.exists(before_path), "before.jpg not found in repo root"
    assert os.path.exists(after_path), "after.png not found in repo root"

    # Step 1 — Object Detection on both images
    print("\n[1/3] Running YOLOv8-OBB object detection...")
    detections = run_detection_pipeline([before_path, after_path])
    print(f"Detected {detections['total_objects_detected']} objects")

    # Step 2 — Change Detection (loads Qwen2-VL — takes 1-2 min first time)
    print("\n[2/3] Running Qwen2-VL change detection...")
    change_stats = run_change_detection_pipeline(before_path, after_path)
    print(f"Severity: {change_stats['change_severity']}")

    # Step 3 — Report Generation (reuses already-loaded Qwen2-VL, no reload)
    print("\n[3/3] Generating analytical report...")
    report = run_report_pipeline(
        detections_path="outputs/detections.json",
        change_stats_path="outputs/change_stats.json",
        before_path=before_path,
        after_path=after_path
    )

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)

    outputs = [
        "outputs/annotated.jpg",
        "outputs/detections.json",
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
