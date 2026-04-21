from torchgeo.datasets import EuroSAT
from ultralytics import YOLO
from PIL import Image
from modules.visualizer import save_image_with_boxes, save_class_distribution
import numpy as np
import json
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

        path = os.path.join(output_dir, f"{class_name}_{seen_classes.get(label, 0)}.png")
        img.save(path)
        saved.append(path)
        seen_classes[label] = seen_classes.get(label, 0) + 1

        if class_name == "AnnualCrop" and not before_saved:
            img.save(os.path.join(output_dir, "before.png"))
            before_saved = True
        if class_name == "Industrial" and not after_saved:
            img.save(os.path.join(output_dir, "after.png"))
            after_saved = True

        # Stop early once all classes have n_per_class samples and before/after are saved
        if len(seen_classes) == len(dataset.classes) and before_saved and after_saved:
            break

    return saved


def run_detection_pipeline(image_paths, output_dir="outputs"):
    """
    Runs YOLOv8n inference on a list of image paths.
    Saves detections.json, detection_viz.png, class_distribution.png.
    Returns the detections dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    # OBB model pretrained on DOTA — 15 aerial classes (plane, ship, vehicle, etc.)
    model = YOLO("yolov8n-obb.pt")

    all_detections = []
    class_distribution = {}
    total_objects = 0

    for image_path in image_paths:
        results = model(image_path, verbose=False)
        result = results[0]

        image_detections = []
        # OBB models use result.obb instead of result.boxes
        obb = result.obb
        if obb is not None:
            for i in range(len(obb)):
                class_id = int(obb.cls[i].item())
                class_name = model.names[class_id]
                confidence = round(float(obb.conf[i].item()), 4)
                # Convert OBB corners to axis-aligned bbox [x1, y1, x2, y2]
                corners = obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
                x1 = round(float(corners[:, 0].min()), 2)
                y1 = round(float(corners[:, 1].min()), 2)
                x2 = round(float(corners[:, 0].max()), 2)
                y2 = round(float(corners[:, 1].max()), 2)

                image_detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]  # [x1, y1, x2, y2] in pixels
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

    json_path = os.path.join(output_dir, "detections.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    if image_paths:
        if all_detections[0]["detections"]:
            save_image_with_boxes(
                image_paths[0],
                all_detections[0]["detections"],
                os.path.join(output_dir, "annotated.jpg")
            )
        else:
            Image.open(image_paths[0]).save(os.path.join(output_dir, "annotated.jpg"))

    if class_distribution:
        save_class_distribution(class_distribution, os.path.join(output_dir, "class_distribution.png"))

    print(f"Detection complete: {total_objects} objects detected across {len(image_paths)} images")
    return output
