import os
import json
import shutil
import numpy as np
from PIL import Image
from skimage.measure import shannon_entropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# -------- CONFIG --------
ROOT_DIR = "D:/DL/DATA/patches/ohrc_png"      # root folder containing OHRC subfolders
DEST_ROOT = "D:/DL/DATA/selected_patches"     # destination root
NUM_SAMPLES_PER_FOLDER = 20                   # number of diverse patches per OHRC set

os.makedirs(DEST_ROOT, exist_ok=True)


def extract_metadata(json_path):
    """Read metadata values for diversity scoring."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        sun_elev = meta.get("sun_elevation", 0)
        sun_azim = meta.get("sun_azimuth", 0)
        yaw = meta.get("satellite_yaw", 0)
        roll = meta.get("satellite_roll", 0)
        pitch = meta.get("satellite_pitch", 0)
        lon = meta.get("longitude", 0)
        lat = meta.get("latitude", 0)
        return [sun_elev, sun_azim, yaw, roll, pitch, lon, lat]
    except Exception:
        return [0]*7


def extract_visual_features(img_path):
    """Compute brightness, contrast, entropy."""
    img = np.array(Image.open(img_path).convert("L"))
    entropy = shannon_entropy(img)
    brightness = np.mean(img)
    contrast = np.std(img)
    return [brightness, contrast, entropy]


def process_ohrc_folder(folder_path, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    patches = []

    # gather image + metadata features
    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            base = os.path.splitext(file)[0]
            json_path = os.path.join(folder_path, base + ".json")
            if not os.path.exists(json_path):
                continue

            img_path = os.path.join(folder_path, file)
            visual_feats = extract_visual_features(img_path)
            meta_feats = extract_metadata(json_path)
            features = visual_feats + meta_feats
            patches.append((base, features))

    print(f"📂 {os.path.basename(folder_path)} — {len(patches)} patches found")

    if len(patches) == 0:
        return

    X = np.array([f for _, f in patches])
    X_scaled = MinMaxScaler().fit_transform(X)
    k = min(NUM_SAMPLES_PER_FOLDER, len(X_scaled))

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # select one representative from each cluster
    selected = []
    for cluster_id in range(k):
        indices = np.where(labels == cluster_id)[0]
        cluster_points = X_scaled[indices]
        centroid = kmeans.cluster_centers_[cluster_id]
        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        best_idx = indices[np.argmin(dists)]
        selected.append(patches[best_idx][0])

    print(f"✅ Selected {len(selected)} diverse patches from {os.path.basename(folder_path)}")

    # copy selected files (preserving patch names)
    for base in selected:
        png_src = os.path.join(folder_path, base + ".png")
        json_src = os.path.join(folder_path, base + ".json")
        png_dst = os.path.join(dest_folder, base + ".png")
        json_dst = os.path.join(dest_folder, base + ".json")
        shutil.copy(png_src, png_dst)
        if os.path.exists(json_src):
            shutil.copy(json_src, json_dst)

    print(f"📁 Saved to {dest_folder}\n")

# -------- MAIN --------


for subfolder in os.listdir(ROOT_DIR):
    folder_path = os.path.join(ROOT_DIR, subfolder)
    if os.path.isdir(folder_path):
        dest_folder = os.path.join(DEST_ROOT, subfolder)
        process_ohrc_folder(folder_path, dest_folder)

print("🎉 Done! All diverse samples saved by subfolder.")
