import os
import numpy as np
from PIL import Image
import shutil

PATCH_DIR = "D:/DL/DATA/patches/ohrc"      # root OHRC directory
OUT_DIR = "D:/DL/DATA/patches/ohrc_png"    # output directory
PATCH_SIZE = 512                             # fallback patch size
os.makedirs(OUT_DIR, exist_ok=True)

# -------- RECURSIVE SEARCH --------
for root, _, files in os.walk(PATCH_DIR):
    for file in files:
        if file.lower().endswith(".img"):
            base = os.path.splitext(file)[0]
            img_path = os.path.join(root, file)
            json_path = os.path.join(root, base + ".json")

            # ---- Determine folder structure ----
            relative_folder = os.path.relpath(root, PATCH_DIR)
            dest_folder = os.path.join(OUT_DIR, relative_folder)
            os.makedirs(dest_folder, exist_ok=True)

            # ---- Convert .img → .png ----
            try:
                data = np.fromfile(img_path, dtype=np.uint8)
                data = data.reshape((PATCH_SIZE, PATCH_SIZE))
                Image.fromarray(data).save(os.path.join(dest_folder, base + ".png"))
            except Exception as e:
                print(f"❌ Failed to convert {img_path}: {e}")
                continue

            # ---- Copy matching JSON ----
            if os.path.exists(json_path):
                shutil.copy(json_path, os.path.join(dest_folder, base + ".json"))

print("✅ All .img → .png conversion done, folder structure preserved.")
