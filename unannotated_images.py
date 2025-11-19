from ultralytics import YOLO
import os
import torch
import pandas as pd

# ==============================================================
# CONFIGURATION
# ==============================================================
MODEL_PATH = r"C:/Users/Amma.DESKTOP-4K4SV7F/Desktop/dl_code/runs/train/moon_detection_full_pipeline11/weights/best.pt"
UNANNOTATED_DIR = r"D:/DL/DATA/roboflow_upload_pngs"
OUTPUT_DIR = r"D:/DL/DATA/runs/detect/unannotated_predictions"
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "predictions.csv")
CONF_THRESHOLD = 0.4

device = 0 if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {device}")

# ==============================================================
# LOAD MODEL
# ==============================================================
model = YOLO(MODEL_PATH)

# ==============================================================
# RUN INFERENCE
# ==============================================================
if not os.path.exists(UNANNOTATED_DIR):
    print(f"⚠ Directory not found: {UNANNOTATED_DIR}")
else:
    print(f"📂 Running inference on all images in: {UNANNOTATED_DIR}")

    results = model.predict(
        source=UNANNOTATED_DIR,
        conf=CONF_THRESHOLD,
        save=True,
        save_txt=True,
        project="runs/detect",
        name="unannotated_predictions",
        device=device,
        show=False
    )

    print("\n✅ Inference complete! Now saving structured CSV...")

    # ==============================================================
    # SAVE PREDICTIONS TO CSV
    # ==============================================================
    all_detections = []

    for result in results:
        image_path = result.path
        image_name = os.path.basename(image_path)

        if result.boxes is None:
            continue

        boxes = result.boxes.xywh.cpu().numpy()        # x, y, w, h (normalized)
        confs = result.boxes.conf.cpu().numpy()        # confidence scores
        classes = result.boxes.cls.cpu().numpy()       # class indices

        for box, conf, cls in zip(boxes, confs, classes):
            all_detections.append({
                "image": image_name,
                "class_id": int(cls),
                "class_name": model.names[int(cls)],
                "x_center": box[0],
                "y_center": box[1],
                "width": box[2],
                "height": box[3],
                "confidence": float(conf)
            })

    # Create DataFrame and save
    df = pd.DataFrame(all_detections)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(CSV_OUTPUT, index=False)

    print(f"📄 Saved detailed predictions to: {CSV_OUTPUT}")
    print(f"🖼 Annotated images & YOLO .txt files are in: {OUTPUT_DIR}")