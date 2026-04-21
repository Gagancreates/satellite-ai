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
