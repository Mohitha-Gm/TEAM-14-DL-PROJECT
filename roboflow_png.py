import os
import shutil

# Root folder containing multiple subfolders (e.g., ohr_001, ohr_002, ...)
SRC_ROOT = "D:/DL/DATA/selected_patches"
DEST_DIR = "D:/DL/DATA/roboflow_upload_pngs"

# Make sure output folder exists
os.makedirs(DEST_DIR, exist_ok=True)

# Walk through all subdirectories
count = 0
for root, _, files in os.walk(SRC_ROOT):
    for file in files:
        if file.lower().endswith(".png"):
            src_path = os.path.join(root, file)
            dest_path = os.path.join(DEST_DIR, file)
            shutil.copy2(src_path, dest_path)
            count += 1

print(f"✅ Copied {count} PNG images from all subfolders of {SRC_ROOT} → {DEST_DIR}")
