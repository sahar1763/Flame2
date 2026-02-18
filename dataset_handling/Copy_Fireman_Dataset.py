import os
import shutil


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


# --- CONFIGURATION ---
source_directory = r"C:\Projects\Flame2\Datasets_FromDvir\FireMan\No_Fire"
destination_directory = r"C:\Projects\Flame2\Datasets_FromDvir\Seperated_Dataset\FireMan\NoFire"

if __name__ == "__main__":
    copy_and_rename(source_directory, destination_directory)