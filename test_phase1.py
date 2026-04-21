import os
from PIL import Image
from modules.detection import load_eurosat_samples
from modules.visualizer import save_class_distribution

os.makedirs("outputs", exist_ok=True)

paths = load_eurosat_samples(n_per_class=1)
assert len(paths) >= 5, f"Expected at least 5 images, got {len(paths)}"

for p in paths:
    img = Image.open(p)
    assert img.size == (640, 640), f"Wrong size {img.size} for {p}"
    assert img.mode == "RGB", f"Wrong mode {img.mode} for {p}"

assert os.path.exists("data/samples/before.png"), "before.png not saved — check AnnualCrop extraction"
assert os.path.exists("data/samples/after.png"), "after.png not saved — check Industrial extraction"

save_class_distribution({"AnnualCrop": 3, "Industrial": 2, "Forest": 5}, "outputs/test_dist.png")
assert os.path.exists("outputs/test_dist.png"), "Visualizer failed to save chart"

print(f"PHASE 1 TEST PASSED — {len(paths)} images loaded and saved, before/after exist, visualizer works")
