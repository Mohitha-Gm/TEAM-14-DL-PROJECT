import os
import zipfile
import shutil
import re

# ----------- CONFIG PATHS ------------
RAW_ZIP_DIR = "D:/DL/DATA/raw_zips"
EXTRACTED_DIR = "D:/DL/DATA/extracted_zips"
TARGET_OHRC = "D:/DL/DATA/lunasurface_data/ohrc"
TARGET_TMC = "D:/DL/DATA/lunasurface_data/tmc"
TARGET_DTM = "D:/DL/DATA/lunasurface_data/dtm"

os.makedirs(EXTRACTED_DIR, exist_ok=True)
os.makedirs(TARGET_TMC, exist_ok=True)
os.makedirs(TARGET_DTM, exist_ok=True)
os.makedirs(TARGET_OHRC, exist_ok=True)

# ----------- STEP 1: EXTRACT ONLY NEW ZIPS ------------
for item in os.listdir(RAW_ZIP_DIR):
    path = os.path.join(RAW_ZIP_DIR, item)

    if item.endswith(".zip"):
        folder_name = os.path.splitext(item)[0]
        extract_path = os.path.join(EXTRACTED_DIR, folder_name)

        # Skip if already extracted
        if os.path.exists(extract_path):
            print(f"⏩ Already extracted: {item}")
            continue

        # Extract zip
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"✅ Extracted new ZIP: {item}")

    elif os.path.isdir(path):
        dest_path = os.path.join(EXTRACTED_DIR, item)
        if not os.path.exists(dest_path):
            shutil.copytree(path, dest_path)
            print(f"📁 Copied existing folder: {item}")
        else:
            print(f"⏩ Already exists: {item}")

# ----------- STEP 2: ORGANIZE TMC + DTM ------------


def extract_timestamp(filename):
    match = re.search(r'(\d{8}T\d+)', filename)
    return match.group(1) if match else None


tmc_map, dtm_map = {}, {}

for root, _, files in os.walk(EXTRACTED_DIR):
    for file in files:
        path = os.path.join(root, file)
        lower = file.lower()
        timestamp = extract_timestamp(file)

        if not timestamp:
            continue

        if 'tmc' in lower and 'oth' in lower and file.endswith('.tif'):
            tmc_map.setdefault(timestamp, {})['image'] = path
        elif 'tmc' in lower and ('dtm' in lower or 'dem' in lower) and file.endswith('.tif'):
            dtm_map.setdefault(timestamp, {})['image'] = path
        elif 'tmc' in lower and file.endswith('.xml'):
            if timestamp in tmc_map and 'meta' not in tmc_map[timestamp]:
                tmc_map[timestamp]['meta'] = path
            elif timestamp in dtm_map and 'meta' not in dtm_map[timestamp]:
                dtm_map[timestamp]['meta'] = path

# Determine next available index to continue numbering
existing_tmc_count = len([d for d in os.listdir(TARGET_TMC) if os.path.isdir(os.path.join(TARGET_TMC, d))])
existing_dtm_count = len([d for d in os.listdir(TARGET_DTM) if os.path.isdir(os.path.join(TARGET_DTM, d))])
start_idx = max(existing_tmc_count, existing_dtm_count)

matched_ts = sorted(set(tmc_map.keys()) & set(dtm_map.keys()))
for idx, ts in enumerate(matched_ts, start=start_idx):
    tmc_folder = os.path.join(TARGET_TMC, f"tmc_{idx:03d}")
    dtm_folder = os.path.join(TARGET_DTM, f"dtm_{idx:03d}")

    if os.path.exists(tmc_folder) and os.path.exists(dtm_folder):
        continue  # skip already copied pairs

    os.makedirs(tmc_folder, exist_ok=True)
    os.makedirs(dtm_folder, exist_ok=True)

    shutil.copy(tmc_map[ts]['image'], os.path.join(tmc_folder, "image.tif"))
    if 'meta' in tmc_map[ts]:
        shutil.copy(tmc_map[ts]['meta'], os.path.join(tmc_folder, "meta.xml"))

    shutil.copy(dtm_map[ts]['image'], os.path.join(dtm_folder, "image.tif"))
    if 'meta' in dtm_map[ts]:
        shutil.copy(dtm_map[ts]['meta'], os.path.join(dtm_folder, "meta.xml"))

print(f"✅ Added {len(matched_ts)} new matched TMC + DTM pairs.")

# ----------- STEP 3: ORGANIZE OHRC FILES ------------
count_ohrc = len([d for d in os.listdir(TARGET_OHRC) if os.path.isdir(os.path.join(TARGET_OHRC, d))])
ohrc_folders = {}

# Create folders for .img files
for root, _, files in os.walk(EXTRACTED_DIR):
    for file in files:
        if file.lower().endswith('.img') and 'ohr' in file.lower():
            timestamp = extract_timestamp(file)
            if not timestamp:
                continue
            folder = os.path.join(TARGET_OHRC, f"ohr_{count_ohrc:03d}")
            if os.path.exists(folder):
                continue
            os.makedirs(folder, exist_ok=True)
            shutil.copy(os.path.join(root, file), os.path.join(folder, "image.img"))
            ohrc_folders[timestamp] = folder
            count_ohrc += 1

# Move metadata
for root, _, files in os.walk(EXTRACTED_DIR):
    for file in files:
        if 'ohr' not in file.lower():
            continue
        filepath = os.path.join(root, file)
        timestamp = extract_timestamp(file)
        if not timestamp or timestamp not in ohrc_folders:
            continue
        matched_folder = ohrc_folders[timestamp]

        if file.endswith('.xml') and 'data' in root:
            shutil.copy(filepath, os.path.join(matched_folder, "meta.xml"))
        elif file.endswith('.csv') and 'geometry' in root:
            shutil.copy(filepath, os.path.join(matched_folder, "coords.csv"))
        elif file.endswith('.xml') and 'geometry' in root:
            shutil.copy(filepath, os.path.join(matched_folder, "coords.xml"))
        elif file.endswith('.spm'):
            shutil.copy(filepath, os.path.join(matched_folder, "sun.spm"))
        elif file.endswith('.oat'):
            shutil.copy(filepath, os.path.join(matched_folder, "orbit.oat"))

print(f"✅ Added {len(ohrc_folders)} new OHRC folders.")
print("\n🎉 Incremental lunar data organization complete.")
