from ultralytics import YOLO
import torch
import os
import numpy as np
import multiprocessing


def main():
    # ---------------- CONFIG ----------------
    DATA_YAML = r"D:/DL/DATA/moon_ohrc_detection.v2i.yolov8-obb/data.yaml"  # Roboflow dataset YAML
    MODEL_NAME = "yolov8m.pt"                   # Detection model
    RUN_NAME = "moon_detection_full_pipeline"

    # Confirm CUDA (GPU) availability
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")

    # =====================================================================
    # 1. TRAIN THE MODEL
    # =====================================================================
    model = YOLO(MODEL_NAME)

    print("\n📌 Starting training...")
    results = model.train(
        data=DATA_YAML,
        epochs=200,          # Increase for better convergence
        imgsz=640,           # Resize images
        batch=8,             # Adjust for GPU VRAM
        name=RUN_NAME,
        lr0=0.002,
        patience=50,
        device=device,
        augment=True,
        hsv_h=0.015,  # color jitter hue
        hsv_s=0.7,    # saturation
        hsv_v=0.4,    # brightness
        degrees=5,    # rotation
        translate=0.1,  # shift
        scale=0.5,      # zoom
        shear=0.1,      # skew
        mosaic=1.0,     # multi-image mosaic
        mixup=0.2,      # combine two images
        project="runs/train"
    )
    print("\n✅ Training complete! Check runs/train for results & weights.\n")

    # =====================================================================
    # 2. VALIDATION METRICS
    # =====================================================================
    print("📊 Running evaluation on validation dataset...")
    val_metrics = model.val(data=DATA_YAML, split="val", device=device)

    print("\n✅ Validation metrics complete!")
    if hasattr(val_metrics, "box"):
        print(f"mAP50 (val): {val_metrics.box.map50:.4f}")
        print(f"mAP50-95 (val): {val_metrics.box.map:.4f}")

        precision_val = float(val_metrics.box.p.mean()) if isinstance(val_metrics.box.p, (np.ndarray, list)) else val_metrics.box.p
        recall_val = float(val_metrics.box.r.mean()) if isinstance(val_metrics.box.r, (np.ndarray, list)) else val_metrics.box.r

        print(f"Precision (val): {precision_val:.4f}")
        print(f"Recall (val): {recall_val:.4f}")
    else:
        print("⚠ Could not access box metrics. Raw metrics:")
        print(val_metrics)

    # =====================================================================
    # 3. TEST METRICS
    # =====================================================================
    print("\n📊 Running evaluation on test dataset...")
    test_metrics = model.val(data=DATA_YAML, split="test", device=device)

    print("\n✅ Test metrics complete!")
    if hasattr(test_metrics, "box"):
        print(f"mAP50 (test): {test_metrics.box.map50:.4f}")
        print(f"mAP50-95 (test): {test_metrics.box.map:.4f}")

        precision_test = float(test_metrics.box.p.mean()) if isinstance(test_metrics.box.p, (np.ndarray, list)) else test_metrics.box.p
        recall_test = float(test_metrics.box.r.mean()) if isinstance(test_metrics.box.r, (np.ndarray, list)) else test_metrics.box.r

        print(f"Precision (test): {precision_test:.4f}")
        print(f"Recall (test): {recall_test:.4f}")
    else:
        print("⚠ Could not access box metrics. Raw metrics:")
        print(test_metrics)

    # =====================================================================
    # 4. INFERENCE ON SINGLE IMAGE
    # =====================================================================
    print("\n🔍 Running inference on one sample test image...")
    TEST_IMG = r"D:/DL/DATA/moon_ohrc_detection.v2i.yolov8-obb/test/images/ohr_000_patch_3676_png.rf.1df89fcc93addd79b5597a697bfdf0d1.jpg"

    if os.path.exists(TEST_IMG):
        single_pred = model.predict(
            source=TEST_IMG,
            conf=0.5,
            save=True,
            show=True,    # Show image with predicted boxes
            device=device
        )
        print("✅ Single-image inference done! Check 'runs/detect/predict/' for output.")
    else:
        print("⚠ The sample image path is incorrect or missing.")

    # =====================================================================
    # 5. INFERENCE ON ENTIRE TEST SET
    # =====================================================================
    print("\n🧠 Running inference on all test images...")
    TEST_DIR = r"D:/DL/DATA/moon_ohrc_detection.v2i.yolov8-obb/test/images"

    if os.path.exists(TEST_DIR):
        all_preds = model.predict(
            source=TEST_DIR,
            conf=0.5,
            save=True,        # Saves predicted images with boxes
            device=device,
            stream=False
        )
        print("✅ Inference complete for all test images!")
        print("📂 Check 'runs/detect/predict/' or 'runs/detect/exp*/' for output.")
    else:
        print("⚠ Test directory not found. Please verify path.")

    print("\n🎉 Full detection pipeline complete!")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required on Windows
    main()

