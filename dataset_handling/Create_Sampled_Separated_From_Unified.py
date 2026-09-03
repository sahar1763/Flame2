r"""
Create_Sampled_Separated_From_Unified.py
----------------------------------------

Reconstructs the EXACT original-image subset used to create a UnifiedDataset.

The UnifiedDataset labels.csv must contain:

    id
    dataset
    fire
    original_id
    original_relative_path

These fields are written by the modified create_unified_dataset() function.

The script reads every selected UnifiedDataset row and copies the corresponding
ORIGINAL image from Seperated_Dataset while preserving:

    - Original filename
    - Original resolution
    - Original image contents
    - Original dataset hierarchy
    - Fire / NoFire hierarchy
    - images/ hierarchy
    - All corresponding labels* files when available

Therefore:

    UnifiedDataset/img_012345.jpg

may correspond to:

    Seperated_Dataset/
        D_Fire/
            Fire/
                images/
                    frame_000123.jpg

The sampled separated output will contain:

    Sampled_Seperated_Dataset/
        D_Fire/
            Fire/
                images/
                    frame_000123.jpg

                labels_raw/
                    frame_000123.txt

                labels_biggest/
                    frame_000123.txt

                labels_unified/
                    frame_000123.txt


IMPORTANT
=========

This script DOES NOT perform sampling itself.

It uses the UnifiedDataset labels.csv as the source of truth.

Therefore the output contains exactly the original source images that were
actually selected when the UnifiedDataset was created.

It does NOT:

    - Resize images
    - Crop images
    - Rename images
    - Re-encode images
    - Move source images
    - Change labels


EXAMPLE USAGE
=============

Reconstruct all datasets:

    python Create_Sampled_Separated_From_Unified.py --input ..\Seperated_Dataset --unified_labels ..\UnifiedDataset\labels.csv --output ..\Sampled_Seperated_Dataset


Only selected datasets:

    python Create_Sampled_Separated_From_Unified.py --input ..\Seperated_Dataset --unified_labels ..\UnifiedDataset\labels.csv --output ..\Sampled_Seperated_Dataset --datasets FireSmokeDataset FireSmokeNEWdataset IAI_Datasets


Clear existing output first:

    python Create_Sampled_Separated_From_Unified.py --input ..\Seperated_Dataset --unified_labels ..\UnifiedDataset\labels.csv --output ..\Sampled_Seperated_Dataset --datasets FireSmokeDataset FireSmokeNEWdataset IAI_Datasets --clear_output
"""


import os
import shutil
import argparse
import pandas as pd


def create_sampled_separated_from_unified(
        input_root,
        unified_labels_csv,
        output_root,
        datasets_to_keep=None,
        clear_output=False,
):
    """
    Reconstruct the exact original source-image subset represented by a
    UnifiedDataset labels.csv.

    Parameters
    ----------
    input_root : str
        Original Seperated_Dataset directory.

    unified_labels_csv : str
        labels.csv created together with the UnifiedDataset.

    output_root : str
        Destination for the reconstructed original-resolution subset.

    datasets_to_keep : list[str] or None
        Optional datasets to reconstruct.
        If None, all datasets represented in labels.csv are copied.

    clear_output : bool
        Delete the existing OUTPUT directory before processing.
        The source dataset is never modified.
    """

    # ============================================================
    # Validate input
    # ============================================================

    if not os.path.isdir(input_root):
        raise FileNotFoundError(
            f"Original Seperated_Dataset not found:\n{input_root}"
        )

    if not os.path.isfile(unified_labels_csv):
        raise FileNotFoundError(
            f"UnifiedDataset labels.csv not found:\n"
            f"{unified_labels_csv}"
        )

    # ============================================================
    # Prepare output
    # ============================================================

    if clear_output and os.path.exists(output_root):

        print(
            f"Removing existing output folder:\n"
            f"{output_root}"
        )

        shutil.rmtree(output_root)

    os.makedirs(
        output_root,
        exist_ok=True,
    )

    # ============================================================
    # Load UnifiedDataset mapping
    # ============================================================

    df = pd.read_csv(
        unified_labels_csv
    )

    required_columns = {
        "id",
        "dataset",
        "fire",
        "original_id",
        "original_relative_path",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "UnifiedDataset labels.csv does not contain the "
            "required reverse-mapping columns.\n\n"
            f"Missing columns: {missing_columns}\n\n"
            "Regenerate the UnifiedDataset using the modified "
            "create_unified_dataset() function."
        )

    # ============================================================
    # Optional dataset filter
    # ============================================================

    if datasets_to_keep is not None:

        datasets_to_keep = set(
            datasets_to_keep
        )

        df = df[
            df["dataset"].isin(
                datasets_to_keep
            )
        ].copy()

    # Preserve the exact original UnifiedDataset CSV order.
    df = df.reset_index(drop=True)

    print()
    print("=" * 70)
    print("RECONSTRUCTING UNIFIED DATASET SOURCE IMAGES")
    print("=" * 70)

    print(
        f"Unified images selected: {len(df)}"
    )

    print()

    # ============================================================
    # Statistics
    # ============================================================

    images_copied = 0
    labels_copied = 0

    missing_images = []
    missing_mapping = []

    # ============================================================
    # Process exact UnifiedDataset rows
    # ============================================================

    for _, row in df.iterrows():

        unified_id = str(
            row["id"]
        )

        dataset_name = str(
            row["dataset"]
        )

        original_id = str(
            row["original_id"]
        )

        relative_path = str(
            row["original_relative_path"]
        )

        # Make CSV paths compatible with the current OS.
        relative_path_os = relative_path.replace(
            "/",
            os.sep,
        )

        # ========================================================
        # Exact original source image
        # ========================================================

        source_image = os.path.join(
            input_root,
            relative_path_os,
        )

        if not os.path.isfile(source_image):

            missing_images.append({
                "unified_id": unified_id,
                "dataset": dataset_name,
                "original_id": original_id,
                "expected_path": source_image,
            })

            continue

        # ========================================================
        # Recreate exact original hierarchy
        # ========================================================

        destination_image = os.path.join(
            output_root,
            relative_path_os,
        )

        destination_image_dir = os.path.dirname(
            destination_image
        )

        os.makedirs(
            destination_image_dir,
            exist_ok=True,
        )

        # Copy original image exactly.
        shutil.copy2(
            source_image,
            destination_image,
        )

        images_copied += 1

        # ========================================================
        # Determine label structure
        # ========================================================

        source_image_dir = os.path.dirname(
            source_image
        )

        image_stem = os.path.splitext(
            original_id
        )[0]

        label_name = (
            image_stem
            + ".txt"
        )

        # Typical detection structure:
        #
        # Fire/
        #   images/
        #   labels_raw/
        #   labels_biggest/
        #   labels_unified/
        #
        if (
            os.path.basename(source_image_dir).lower()
            == "images"
        ):

            source_class_dir = os.path.dirname(
                source_image_dir
            )

            relative_class_dir = os.path.relpath(
                source_class_dir,
                input_root,
            )

            # Find every labels* folder.
            for folder_name in sorted(
                os.listdir(source_class_dir)
            ):

                label_folder = os.path.join(
                    source_class_dir,
                    folder_name,
                )

                if not (
                    os.path.isdir(label_folder)
                    and folder_name.lower().startswith("labels")
                ):
                    continue

                source_label = os.path.join(
                    label_folder,
                    label_name,
                )

                # Not every label folder necessarily contains
                # a label for every image.
                if not os.path.isfile(source_label):
                    continue

                destination_label_folder = os.path.join(
                    output_root,
                    relative_class_dir,
                    folder_name,
                )

                os.makedirs(
                    destination_label_folder,
                    exist_ok=True,
                )

                destination_label = os.path.join(
                    destination_label_folder,
                    label_name,
                )

                shutil.copy2(
                    source_label,
                    destination_label,
                )

                labels_copied += 1

    # ============================================================
    # Save the exact Unified -> Original mapping in output
    # ============================================================

    mapping_output = os.path.join(
        output_root,
        "unified_to_original_mapping.csv",
    )

    df.to_csv(
        mapping_output,
        index=False,
    )

    # ============================================================
    # Save missing-image report if necessary
    # ============================================================

    if missing_images:

        missing_output = os.path.join(
            output_root,
            "missing_original_images.csv",
        )

        pd.DataFrame(
            missing_images
        ).to_csv(
            missing_output,
            index=False,
        )

    # ============================================================
    # Summary
    # ============================================================

    print()
    print("=" * 70)
    print("RECONSTRUCTION COMPLETE")
    print("=" * 70)

    print(
        f"Unified rows:          {len(df)}"
    )

    print(
        f"Original images copied:{images_copied}"
    )

    print(
        f"Labels copied:         {labels_copied}"
    )

    print(
        f"Missing source images: {len(missing_images)}"
    )

    print()

    print(
        f"Output:\n{output_root}"
    )

    print()

    print(
        "Unified -> original mapping:\n"
        f"{mapping_output}"
    )

    if missing_images:

        print()
        print(
            "WARNING:"
        )

        print(
            "Some original images referenced by the UnifiedDataset "
            "could not be found."
        )

        print(
            f"See:\n{missing_output}"
        )

    print("=" * 70)


# ================================================================
# Command-line interface
# ================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the exact original-resolution source-image "
            "subset represented by a UnifiedDataset."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Original Seperated_Dataset root."
        ),
    )

    parser.add_argument(
        "--unified_labels",
        required=True,
        help=(
            "labels.csv from the UnifiedDataset."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output Sampled_Seperated_Dataset directory."
        ),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Optional dataset names to reconstruct. "
            "If omitted, all datasets are reconstructed."
        ),
    )

    parser.add_argument(
        "--clear_output",
        action="store_true",
        help=(
            "Delete the existing OUTPUT directory first. "
            "The original dataset is never modified."
        ),
    )

    args = parser.parse_args()

    create_sampled_separated_from_unified(
        input_root=args.input,
        unified_labels_csv=args.unified_labels,
        output_root=args.output,
        datasets_to_keep=args.datasets,
        clear_output=args.clear_output,
    )