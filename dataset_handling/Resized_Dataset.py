import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def resize_image(image_path):
    """
    Resizes a single image to 224x224 and overwrites the original file.
    """
    try:
        # Use Image.open in a way that ensures the file is closed properly
        with Image.open(image_path) as img:
            # Skip if already 224x224 to save time
            if img.size == (224, 224):
                return True

            img = img.convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
            img.save(image_path, "JPEG", quality=90)
        return True
    except Exception as e:
        return f"Error on {image_path}: {e}"


def main():
    # The directory we confirmed earlier
    root_dir = r"C:\Projects\Flame2\UnifiedDataset2"

    if not os.path.exists(root_dir):
        print(f"Directory NOT FOUND: {root_dir}")
        return

    print(f"Scanning directory: {root_dir} ...")
    all_images = []
    valid_extensions = ('.jpg', '.jpeg', '.png')

    # Using os.walk for better compatibility
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                all_images.append(os.path.join(root, file))

    total_images = len(all_images)
    if total_images == 0:
        print("Still no images found. Try running: ls -R /tmp/saharc/UnifiedDataset | head")
        return

    print(f"Found {total_images} images. Starting parallel resize...")

    # ProcessPoolExecutor uses all CPU cores
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(resize_image, all_images), total=total_images))

    # Error summary
    errors = [res for res in results if res is not True]
    if errors:
        print(f"\nFinished with {len(errors)} errors.")
    else:
        print("\nSuccess! All images are now 224x224.")


if __name__ == "__main__":
    main()