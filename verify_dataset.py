"""
Quick integrity check on the dataset before training.
Checks every image listed in dataset_labels.csv can actually be opened,
and reports basic stats (file sizes, dimensions) to catch anomalies.

This does NOT modify your dataset - it only reports problems.
"""

import pandas as pd
from PIL import Image
import os

CSV_PATH = "dataset_labels.csv"

df = pd.read_csv(CSV_PATH)
print(f"Checking {len(df)} images...\n")

corrupted_files = []
missing_files = []
sizes = []

for idx, row in df.iterrows():
    path = row["image_path"]

    if not os.path.exists(path):
        missing_files.append(path)
        continue

    try:
        img = Image.open(path)
        img.verify()  # checks the file isn't corrupted
        sizes.append(img.size)  # (width, height)
    except Exception as e:
        corrupted_files.append((path, str(e)))

    # Progress indicator every 5000 images
    if (idx + 1) % 5000 == 0:
        print(f"Checked {idx + 1}/{len(df)}...")

print(f"\n{'='*50}")
print(f"RESULTS")
print(f"{'='*50}")
print(f"Total images checked: {len(df)}")
print(f"Missing files: {len(missing_files)}")
print(f"Corrupted files: {len(corrupted_files)}")

if sizes:
    unique_sizes = set(sizes)
    print(f"\nUnique image dimensions found: {len(unique_sizes)}")
    if len(unique_sizes) <= 5:
        print(f"Dimensions: {unique_sizes}")
    else:
        print(f"(too many unique sizes to list - this is fine, "
              f"since we resize everything to 224x224 anyway)")

if missing_files:
    print(f"\nFirst few missing files:")
    for f in missing_files[:5]:
        print(f"  - {f}")

if corrupted_files:
    print(f"\nFirst few corrupted files:")
    for f, err in corrupted_files[:5]:
        print(f"  - {f}: {err}")

if not missing_files and not corrupted_files:
    print(f"\nAll images verified successfully! Dataset is clean and ready for training.")
else:
    print(f"\nSome issues found - consider removing these rows from dataset_labels.csv before training.")