import os
import shutil
import argparse
import pandas as pd


def Extract_Datasets_From_UnifiedDataset(
        input_root,
        output_root,
        datasets_to_keep,
        labels_csv="labels.csv",
        clear_output=False,
):
    r"""
    Extract selected source datasets from a unified image dataset.

    The input dataset is expected to contain a labels.csv file describing
    the source dataset associated with every image.

    Required CSV columns:
        id       - image filename
        dataset  - source dataset name

    Any additional CSV columns, such as 'fire', are preserved unchanged.

    Example input:
        UnifiedDataset/
            img_000001.jpg
            img_000002.jpg
            img_000003.jpg
            labels.csv

    Example labels.csv:
        id,dataset,fire
        img_000001.jpg,D_Fire,1
        img_000002.jpg,FASDD,0
        img_000003.jpg,IAI_Datasets,1

    If datasets_to_keep = ["D_Fire", "IAI_Datasets"], the output becomes:

        SelectedDataset/
            img_000001.jpg
            img_000003.jpg
            labels.csv

    The output labels.csv contains only the rows corresponding to copied
    images while preserving all original columns and values.

    Parameters
    ----------
    input_root : str
        Path to the unified dataset containing the images and labels.csv.

    output_root : str
        Destination folder for the selected images and filtered labels.csv.

    datasets_to_keep : list[str]
        Dataset names to extract, matching values in the CSV 'dataset' column.

    labels_csv : str, optional
        Name of the CSV file. Default is "labels.csv".

    clear_output : bool, optional
        If True, delete the existing output directory before extraction.
        Default is False.

    Notes
    -----
    Images can be located either as:

        input_root/image.jpg

    or:

        input_root/images/image.jpg

    Missing images are skipped and reported.

    Images are copied without renaming, resizing, cropping, or modifying them.

    Usage:

    python Extract_Datasets_From_UnifiedDataset.py --input ..\UnifiedDataset --output ..\SelectedDataset --datasets D_Fire IAI_Datasets

    """

    csv_path = os.path.join(input_root, labels_csv)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Could not find labels CSV:\n{csv_path}"
        )

    # --------------------------------------------------------
    # Prepare output folder
    # --------------------------------------------------------
    if clear_output and os.path.exists(output_root):
        print(f"Removing existing output folder: {output_root}")
        shutil.rmtree(output_root)

    os.makedirs(output_root, exist_ok=True)

    # --------------------------------------------------------
    # Read labels CSV
    # --------------------------------------------------------
    df = pd.read_csv(csv_path)

    required_columns = {"id", "dataset"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Keep only selected datasets
    # --------------------------------------------------------
    filtered_df = df[
        df["dataset"].isin(datasets_to_keep)
    ].copy()

    print("\nDatasets requested:")

    for dataset in datasets_to_keep:
        count = (filtered_df["dataset"] == dataset).sum()
        print(f"  {dataset}: {count}")

    print(f"\nTotal rows selected: {len(filtered_df)}")

    # --------------------------------------------------------
    # Copy images
    # --------------------------------------------------------
    valid_rows = []
    missing_images = []

    for _, row in filtered_df.iterrows():

        image_name = str(row["id"])

        # Structure 1:
        # input_root/image.jpg
        source_image = os.path.join(
            input_root,
            image_name,
        )

        # Structure 2:
        # input_root/images/image.jpg
        if not os.path.isfile(source_image):

            alternative_path = os.path.join(
                input_root,
                "images",
                image_name,
            )

            if os.path.isfile(alternative_path):
                source_image = alternative_path

            else:
                missing_images.append(image_name)
                continue

        destination_image = os.path.join(
            output_root,
            image_name,
        )

        shutil.copy2(
            source_image,
            destination_image,
        )

        # Keep the complete original CSV row
        valid_rows.append(row)

    # --------------------------------------------------------
    # Write filtered labels.csv
    # --------------------------------------------------------
    output_df = pd.DataFrame(
        valid_rows,
        columns=df.columns,
    )

    output_csv = os.path.join(
        output_root,
        labels_csv,
    )

    output_df.to_csv(
        output_csv,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("DATASET EXTRACTION COMPLETE")
    print("=" * 60)

    print(f"Requested CSV rows: {len(filtered_df)}")
    print(f"Images copied:       {len(valid_rows)}")
    print(f"Missing images:      {len(missing_images)}")

    print("\nCopied by dataset:")

    if len(output_df) > 0:
        for dataset, count in output_df["dataset"].value_counts().items():
            print(f"  {dataset}: {count}")

    if missing_images:
        print("\nWARNING: Images referenced by CSV but not found:")

        for image_name in missing_images[:20]:
            print(f"  {image_name}")

        if len(missing_images) > 20:
            print(
                f"  ... and {len(missing_images) - 20} more"
            )

    print(f"\nOutput folder: {output_root}")
    print(f"Output CSV:    {output_csv}")

    print("=" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Extract selected source datasets from a unified dataset "
            "using the dataset column in labels.csv."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Unified dataset folder containing images and labels.csv.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output folder for the selected dataset.",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help=(
            "Dataset names to extract. "
            "Example: --datasets D_Fire IAI_Datasets"
        ),
    )

    parser.add_argument(
        "--labels",
        default="labels.csv",
        help="CSV filename. Default: labels.csv",
    )

    parser.add_argument(
        "--clear_output",
        action="store_true",
        help="Delete the output folder before extraction.",
    )

    args = parser.parse_args()

    Extract_Datasets_From_UnifiedDataset(
        input_root=args.input,
        output_root=args.output,
        datasets_to_keep=args.datasets,
        labels_csv=args.labels,
        clear_output=args.clear_output,
    )