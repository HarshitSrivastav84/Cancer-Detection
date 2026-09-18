"""
Classification: Cancer (Tumor) vs Non-Cancer
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_DIR = "data_histopathology/HMU-GC-HE-30K/all_image"

CLASS_MAP = {
    "TUM": "cancer",
    "STR": "non_cancer",
    "NOR": "non_cancer",
    "MUS": "non_cancer",
    "MUC": "non_cancer",
    "LYM": "non_cancer",
    "DEB": "non_cancer",
    "ADI": "non_cancer",
}

records = []

for folder_name, label in CLASS_MAP.items():
    folder_path = os.path.join(DATASET_DIR, folder_name)

    if not os.path.exists(folder_path):
        print(f"WARNING: Folder not found, skipping: {folder_path}")
        continue

        # For image filtered list
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    ]

    print(f"{folder_name}: {len(image_files)} images -> labeled '{label}'")

    for img_file in image_files:
        full_path = os.path.join(folder_path, img_file)
        records.append({
            "image_path": full_path,
            "class_folder": folder_name,
            "label": label
        })

# Converts in pandas table (helps in structuring)
df = pd.DataFrame(records)
print(f"\nTotal images collected: {len(df)}")
print("\nClass balance (cancer vs non_cancer):")
print(df["label"].value_counts())

train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

output_csv = "dataset_labels.csv"
final_df.to_csv(output_csv, index=False)

print(f"\nSaved {output_csv} with {len(final_df)} total entries.")
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")