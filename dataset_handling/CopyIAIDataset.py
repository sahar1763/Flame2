import os
import shutil
import csv


def get_next_index(destination_folder, prefix="img_", extension=".jpg"):
    """
    Scans the destination folder to find the highest current index
    and returns the next available index number.
    """
    max_index = -1

    # Create the folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        return 0

    files = os.listdir(destination_folder)
    for filename in files:
        if filename.startswith(prefix) and filename.endswith(extension):
            try:
                # Extract the number part: img_000002.jpg -> 000002 -> 2
                number_part = filename[len(prefix):-len(extension)]
                index = int(number_part)
                if index > max_index:
                    max_index = index
            except ValueError:
                continue

    return max_index + 1


def process_dataset(source_root, dest_root):
    # --- Define Subpaths Automatically ---
    fire_dest = os.path.join(dest_root, "Fire")
    nofire_dest = os.path.join(dest_root, "NoFire")

    print(f"Destination Fire:   {fire_dest}")
    print(f"Destination NoFire: {nofire_dest}")
    print("Scanning destination folders to determine starting indices...")

    # Initialize counters for both destinations
    fire_counter = get_next_index(fire_dest)
    nofire_counter = get_next_index(nofire_dest)

    print(f"Fire folder start index:   {fire_counter}")
    print(f"NoFire folder start index: {nofire_counter}")
    print("-" * 40)

    files_copied = 0

    # Recursively walk through the source directory
    for root, dirs, files in os.walk(source_root):
        if "labels.csv" in files:
            label_path = os.path.join(root, "labels.csv")
            print(f"Processing: {label_path}")

            try:
                with open(label_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)

                    for row in reader:
                        # Get filename and label from CSV columns
                        image_name = row.get('id') or row.get('filename')
                        fire_label = row.get('fire') or row.get('label')

                        if not image_name or fire_label is None:
                            continue

                        source_image_path = os.path.join(root, image_name)

                        # Only proceed if the image file actually exists
                        if os.path.exists(source_image_path):

                            # Determine destination based on label
                            if str(fire_label).strip() == "1":
                                current_dest = fire_dest
                                new_name = f"img_{fire_counter:06d}.jpg"
                                fire_counter += 1
                            else:
                                current_dest = nofire_dest
                                new_name = f"img_{nofire_counter:06d}.jpg"
                                nofire_counter += 1

                            # Perform the copy
                            dest_path = os.path.join(current_dest, new_name)
                            shutil.copy2(source_image_path, dest_path)
                            files_copied += 1

            except Exception as e:
                print(f"Error reading {label_path}: {e}")

    print("-" * 40)
    print(f"Success! Copied {files_copied} images.")


# --- CONFIGURATION ---
# The folder that contains all your subfolders (videos)
SOURCE_DIR = r"/videos_Thesis/ProcessedVideos"

# The PARENT folder where 'Fire' and 'NoFire' folders are located
DEST_PARENT_DIR = r"C:\Projects\Flame2\Datasets_FromDvir\Seperated_Dataset\IAI_Datasets"

if __name__ == "__main__":
    process_dataset(SOURCE_DIR, DEST_PARENT_DIR)