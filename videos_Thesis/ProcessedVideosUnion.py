import os
import shutil
import pandas as pd
import re

processed_root_dir = "ProcessedVideos"
all_images_dir = "AllImages"
all_labels_csv = "all_labels.csv"

os.makedirs(all_images_dir, exist_ok=True)

next_index = 1
all_rows = []

# Walk through all subdirectories in ProcessedVideos
for dirpath, dirnames, filenames in os.walk(processed_root_dir):
    # Skip the root directory itself
    if dirpath == processed_root_dir:
        continue

    # Find CSV in this folder
    csv_files = [f for f in filenames if f.lower().endswith(".csv")]
    if not csv_files:
        continue

    labels_csv_path = os.path.join(dirpath, csv_files[0])
    labels_df = pd.read_csv(labels_csv_path)

    # Go through images in this folder
    image_files = [f for f in filenames if f.lower().endswith(".jpg")]
    image_files.sort()  # optional, keep original order

    for img_file in image_files:
        old_path = os.path.join(dirpath, img_file)

        # New name with continuous numbering
        new_name = f"img_{next_index:06d}.jpg"
        new_path = os.path.join(all_images_dir, new_name)

        shutil.copy2(old_path, new_path)

        # Update CSV row
        row = labels_df[labels_df['id'] == img_file].iloc[0].to_dict()
        row['id'] = new_name
        all_rows.append(row)

        next_index += 1

# Save unified CSV
pd.DataFrame(all_rows).to_csv(all_labels_csv, index=False)
print(f"All images copied to {all_images_dir}, total images: {next_index-1}")
print(f"Unified CSV saved as {all_labels_csv}")
