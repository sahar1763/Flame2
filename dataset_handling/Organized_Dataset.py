import os
import shutil
import pandas as pd
import csv
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def split_by_dataset_fire(source_images_dir, source_csv_path, output_root_dir):
    """
    Split images into folders by dataset and fire/no-fire, renumber them,
    and save a CSV for each dataset.

    Parameters:
    - source_images_dir: folder with all original images
    - source_csv_path: CSV with columns ["id", "dataset", "fire"]
    - output_root_dir: root folder to save dataset folders
    """

    os.makedirs(output_root_dir, exist_ok=True)

    # Load unified CSV
    df = pd.read_csv(source_csv_path)

    # Check required columns
    if "dataset" not in df.columns:
        raise ValueError("CSV must contain a 'dataset' column.")
    if "fire" not in df.columns:
        raise ValueError("CSV must contain a 'fire' column (0 / 1).")

    # Group by dataset
    grouped = df.groupby("dataset")

    for dataset, group_df in grouped:

        print(f"Processing dataset: {dataset}")

        dataset_dir = os.path.join(output_root_dir, dataset)
        new_rows = []
        next_index = 1

        for _, row in group_df.iterrows():

            old_image_name = row["id"]
            fire_value = row["fire"]
            old_image_path = os.path.join(source_images_dir, old_image_name)

            if not os.path.exists(old_image_path):
                print(f"Warning: image not found: {old_image_name}")
                continue

            # Decide class folder
            if fire_value == 1:
                class_folder = "Fire"
            elif fire_value == 0:
                class_folder = "NoFire"
            else:
                print(f"Warning: unexpected fire value ({fire_value}) in {old_image_name}")
                continue

            # New sequential name inside this dataset
            new_image_name = f"img_{next_index:06d}.jpg"
            class_dir = os.path.join(dataset_dir, class_folder)
            new_image_path = os.path.join(class_dir, new_image_name)

            # Ensure directory exists
            os.makedirs(class_dir, exist_ok=True)

            shutil.copy2(old_image_path, new_image_path)

            # Update row
            new_row = row.copy()
            new_row["id"] = new_image_name
            new_rows.append(new_row)

            next_index += 1


    print("Done splitting datasets.")

def sample_images_from_folder(source_images_dir, sample_every_n=10):
    """
    Sample images from a folder at a fixed interval, renumber them, and save to a new folder
    next to the original folder with suffix '_Sampled'.

    Parameters:
    - source_images_dir: folder containing original images
    - sample_every_n: sample one image every n images
    """

    # Prepare output folder next to original
    parent_dir = os.path.dirname(os.path.normpath(source_images_dir))
    folder_name = os.path.basename(os.path.normpath(source_images_dir))
    sampled_dir = os.path.join(parent_dir, folder_name + "_Sampled")
    os.makedirs(sampled_dir, exist_ok=True)

    # Get all image files, sorted
    all_images = sorted([f for f in os.listdir(source_images_dir)
                         if os.path.isfile(os.path.join(source_images_dir, f))])

    next_index = 1

    for i, img_name in enumerate(all_images):
        if i % sample_every_n != 0:
            continue

        old_image_path = os.path.join(source_images_dir, img_name)
        new_image_name = f"img_{next_index:06d}.jpg"
        new_image_path = os.path.join(sampled_dir, new_image_name)

        shutil.copy2(old_image_path, new_image_path)
        next_index += 1

    print(f"Sampled {next_index - 1} images to {sampled_dir}")

def sample_images_from_class_folders(source_class_dir, sample_every_n=10):
    """
    Sample images from a class folder (Fire / NoFire) at a fixed interval,
    renumber them, and save to a new folder next to the parent dataset folder
    with suffix '_Sampled', preserving class subfolder.

    Parameters:
    - source_class_dir: folder containing class images (Fire or NoFire)
    - sample_every_n: sample one image every n images
    """

    # Get parent dataset folder and class name
    parent_dataset_dir = os.path.dirname(os.path.normpath(source_class_dir))
    class_name = os.path.basename(os.path.normpath(source_class_dir))
    dataset_name = os.path.basename(parent_dataset_dir)

    # Create new sampled dataset folder
    sampled_dataset_dir = os.path.join(os.path.dirname(parent_dataset_dir), dataset_name + "_Sampled")
    sampled_class_dir = os.path.join(sampled_dataset_dir, class_name)
    os.makedirs(sampled_class_dir, exist_ok=True)

    # Get all image files, sorted
    all_images = sorted([f for f in os.listdir(source_class_dir)
                         if os.path.isfile(os.path.join(source_class_dir, f))])

    next_index = 1

    for i, img_name in enumerate(all_images):
        if i % sample_every_n != 0:
            continue

        old_image_path = os.path.join(source_class_dir, img_name)
        new_image_name = f"img_{next_index:06d}.jpg"
        new_image_path = os.path.join(sampled_class_dir, new_image_name)

        shutil.copy2(old_image_path, new_image_path)
        next_index += 1

    print(f"Sampled {next_index - 1} images to {sampled_class_dir}")

def renumber_images_in_subfolders(root_dir, img_ext=".jpg"):
    """
    Traverse all subfolders in root_dir.
    Renumber all images in each subfolder sequentially,
    keeping the current order, without copying.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter image files
        image_files = [f for f in filenames if f.lower().endswith(img_ext)]
        if not image_files:
            continue

        # Sort files to preserve current order
        image_files.sort()

        for idx, filename in enumerate(image_files, start=1):
            old_path = os.path.join(dirpath, filename)
            new_name = f"img_{idx:06d}{img_ext}"
            new_path = os.path.join(dirpath, new_name)

            if old_path != new_path:
                os.rename(old_path, new_path)

        print(f"Renumbered {len(image_files)} images in {dirpath}")

def get_next_index(destination_folder, prefix="img_", extension=".jpg"):
    """
    Scans the destination folder to find the highest current index
    and returns the next available index.
    """
    max_index = 0

    # Create the folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        return 1

    for filename in os.listdir(destination_folder):
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

def copy_and_rename(source_root, dest_folder):
    """
    Recursively scans source_root, finds files, and copies them to
    dest_folder with a sequential name (img_XXXXXX.jpg).
    """
    # Get the starting counter based on what is already there
    current_counter = get_next_index(dest_folder)

    files_copied = 0

    # os.walk recursively visits every subfolder
    for root, dirs, files in os.walk(source_root):
        for file in files:
            # You can filter for specific file types if needed
            # if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            #     continue

            source_path = os.path.join(root, file)

            # Determine file extension (keeps original extension)
            _, ext = os.path.splitext(file)

            # Create new filename: img_000001.jpg
            new_filename = f"img_{current_counter:06d}{ext}"
            dest_path = os.path.join(dest_folder, new_filename)

            # Copy the file
            shutil.copy2(source_path, dest_path)

            print(f"Copied: {source_path} -> {dest_path}")

            current_counter += 1
            files_copied += 1

    print(f"\nSuccess! Moved {files_copied} files.")

def get_next_index2(destination_folder, prefix="img_", extension=".jpg"):
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
    fire_counter = get_next_index2(fire_dest)
    nofire_counter = get_next_index2(nofire_dest)

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


def _process_single_image(args):
    """Internal helper function for parallel processing."""
    old_path, new_path, target_size = args
    try:
        if target_size is not None:
            with Image.open(old_path) as img:
                # High-quality Lanczos resize
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                # Convert to RGB (standard for ML) and save as JPEG
                resized_img.convert("RGB").save(new_path, "JPEG", quality=90)
        else:
            # If image_size is None, just copy the original file
            shutil.copy2(old_path, new_path)
        return True
    except Exception as e:
        return f"Error on {old_path}: {e}"


def create_unified_dataset(
        input_root_dir,
        output_root_dir,
        sample_ratio_csv=None,
        image_size=None
):
    os.makedirs(output_root_dir, exist_ok=True)

    # 1. Handle image_size input flexibility
    if isinstance(image_size, int):
        target_size = (image_size, image_size)
    elif isinstance(image_size, (tuple, list)) and len(image_size) == 2 and isinstance(image_size[0], int):
        target_size = tuple(image_size)
    else:
        target_size = None  # Covers None, (None, None), etc.

    # 2. Load sample ratios if provided
    sample_ratios = {}
    if sample_ratio_csv is not None and os.path.exists(sample_ratio_csv):
        df_ratio = pd.read_csv(sample_ratio_csv)
        for _, row in df_ratio.iterrows():
            dataset = row["dataset"]
            fire = int(row["fire"])
            sample_every = int(row["sample_every"])
            sample_ratios[(dataset, fire)] = sample_every

    tasks = []
    new_rows = []
    next_index = 1

    # 3. Walk through folders and plan the tasks
    print("Scanning directories and planning tasks...")
    for dataset_name in os.listdir(input_root_dir):
        dataset_path = os.path.join(input_root_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        for fire_folder_name, fire_value in [("Fire", 1), ("NoFire", 0)]:
            folder_path = os.path.join(dataset_path, fire_folder_name)
            if not os.path.exists(folder_path):
                continue

            sample_every = sample_ratios.get((dataset_name, fire_value), 1)
            images = sorted(os.listdir(folder_path))

            for idx, image_name in enumerate(images):
                if idx % sample_every != 0:
                    continue

                old_image_path = os.path.join(folder_path, image_name)
                if not os.path.exists(old_image_path):
                    continue

                new_image_name = f"img_{next_index:06d}.jpg"
                new_image_path = os.path.join(output_root_dir, new_image_name)

                # Queue the task for parallel execution
                tasks.append((old_image_path, new_image_path, target_size))

                new_rows.append({
                    "id": new_image_name,
                    "dataset": dataset_name,
                    "fire": fire_value
                })
                next_index += 1

    # 4. Execute processing in parallel using all CPU cores
    print(f"Starting processing of {len(tasks)} images...")
    with ProcessPoolExecutor() as executor:
        # Wrapped in tqdm for a nice progress bar
        results = list(tqdm(executor.map(_process_single_image, tasks), total=len(tasks)))

    # 5. Save unified CSV
    csv_path = os.path.join(output_root_dir, "labels.csv")
    pd.DataFrame(new_rows).to_csv(csv_path, index=False)

    # Error Reporting
    errors = [res for res in results if res is not True]
    if errors:
        print(f"Finished with {len(errors)} errors. Check console for details.")

    print(f"Success! Unified dataset with {len(new_rows)} images created at {output_root_dir}")

# -------------------
# Example usage
# -------------------
if __name__ == "__main__":

    # # =====================================================
    # # Split by dataset (fire - no fire)
    # # =====================================================
    # source_images_dir = r"C:\Projects\Flame2\Datasets_FromDvir\Datasets\rgb_images"
    # source_csv_path = r"C:\Projects\Flame2\Datasets_FromDvir\Datasets\labels.csv"
    # output_root_dir = "DatasetsBySource"
    # split_by_dataset_fire(source_images_dir, source_csv_path, output_root_dir)

    # =====================================================
    # Sample images
    # =====================================================
    # sample_images_from_class_folders(
    #     source_class_dir="DatasetsBySource/AA/NoFire",
    #     sample_every_n=2
    # )

    # =====================================================
    # Renumber images in subfolders
    # =====================================================
    # root_folder = "DatasetsBySource"
    # renumber_images_in_subfolders(root_folder)

    # =====================================================
    # Copy FireMan images to dataset
    # =====================================================
    # source_directory = r"C:\Projects\Flame2\Datasets_FromDvir\FireMan\No_Fire"
    # destination_directory = r"C:\Projects\Flame2\Datasets_FromDvir\Seperated_Dataset\FireMan\NoFire"
    # copy_and_rename(source_directory, destination_directory)

    # =====================================================
    # Reorganized new video images into Fire/No Fire folder format.
    # =====================================================
    # # The folder that contains all subfolders (videos)
    # SOURCE_DIR = r"C:\Projects\Flame2\videos_Thesis\ProcessedVideos"
    # # The PARENT folder where 'Fire' and 'NoFire' folders are located
    # DEST_PARENT_DIR = r"C:\Projects\Flame2\Datasets_FromDvir\Seperated_Dataset\IAI_Datasets"
    #
    # process_dataset(SOURCE_DIR, DEST_PARENT_DIR)



    # =====================================================
    # Create unified dataset
    # =====================================================
    input_root_dir = r"..\Seperated_Dataset"
    output_root_dir = r"..\UnifiedDataset"
    sample_ratio_csv = "SampleRatio.csv"
    image_size = (224,224)

    create_unified_dataset(input_root_dir, output_root_dir, sample_ratio_csv, image_size)