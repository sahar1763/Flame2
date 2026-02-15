import os
import cv2
import pandas as pd
import re
import time


def get_next_image_index(output_dir):
    """
    Finds the highest numeric image index in folder and returns next index.
    Expected filename format: img_XXXXX.jpg
    """
    existing_files = os.listdir(output_dir)
    max_index = -1

    pattern = re.compile(r"img_(\d+)\.jpg")

    for file in existing_files:
        match = pattern.match(file)
        if match:
            idx = int(match.group(1))
            max_index = max(max_index, idx)

    return max_index + 1


def parse_time_range(time_range_str):
    """
    Expected format: "(20, 30)"
    """
    time_range_str = time_range_str.strip().replace("(", "").replace(")", "")
    start, end = time_range_str.split(",")
    return float(start.strip()), float(end.strip())


def process_video(
    video_path,
    ranges_csv_path,
    output_images_dir,
    labels_csv_path,
    fps_factor=1,
    dataset_name="video_dataset"
):

    os.makedirs(output_images_dir, exist_ok=True)

    ranges_df = pd.read_csv(ranges_csv_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_every_n_frames = int(fps_factor * fps)
    next_index = get_next_image_index(output_images_dir)

    # print(f"Starting from image index: {next_index}")

    # Buffered CSV writing
    buffer_rows = []
    buffer_size = 200
    file_exists = os.path.exists(labels_csv_path)

    for _, row in ranges_df.iterrows():

        start_sec, end_sec = parse_time_range(row["time_range"])
        fire_label = int(row["label"])

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        while current_frame <= end_frame:

            # Skip frames efficiently
            for _ in range(sample_every_n_frames - 1):
                if not cap.grab():
                    break
                current_frame += 1

            if current_frame > end_frame:
                break

            # Decode only the needed frame
            ret, frame = cap.retrieve()
            if not ret:
                break

            image_name = f"img_{next_index:06d}.jpg"
            image_path = os.path.join(output_images_dir, image_name)

            cv2.imwrite(image_path, frame)

            buffer_rows.append({
                "id": image_name,
                "dataset": dataset_name,
                "fire": fire_label
            })

            next_index += 1
            current_frame += 1

            # Batch write
            if len(buffer_rows) >= buffer_size:
                pd.DataFrame(buffer_rows).to_csv(
                    labels_csv_path,
                    mode='a',
                    header=not file_exists,
                    index=False
                )
                file_exists = True
                buffer_rows = []

    # Flush remaining rows
    if buffer_rows:
        pd.DataFrame(buffer_rows).to_csv(
            labels_csv_path,
            mode='a',
            header=not file_exists,
            index=False
        )

    cap.release()
    print("Finished processing video.")


if __name__ == "__main__":

    videos_root_dir = "Videos"           # root folder containing subfolders and videos
    processed_root_dir = "ProcessedVideos"  # root folder for processed outputs
    os.makedirs(processed_root_dir, exist_ok=True)

    all_videos = []
    videos_to_run = []
    videos_skipped = []

    # First pass: scan all videos and decide which to run / skip
    for dirpath, dirnames, filenames in os.walk(videos_root_dir):

        for video_file in filenames:
            if not video_file.lower().endswith(".mp4"):
                continue

            video_path = os.path.join(dirpath, video_file)
            ranges_csv_path = os.path.join(
                dirpath,
                os.path.splitext(video_file)[0] + ".csv"
            )

            if not os.path.exists(ranges_csv_path):
                # print(f"No ranges CSV for {video_path}, skipping...")
                continue

            video_name = os.path.splitext(video_file)[0]
            output_images_dir = os.path.join(processed_root_dir, video_name)
            labels_csv_path = os.path.join(output_images_dir, "labels.csv")

            all_videos.append(video_path)

            if os.path.exists(output_images_dir) and os.path.exists(labels_csv_path):
                videos_skipped.append(video_path)
            else:
                videos_to_run.append((video_path, ranges_csv_path, output_images_dir, labels_csv_path, video_name))

    # Overview
    print("=== OVERVIEW BEFORE RUNNING ===")
    print(f"Total videos found: {len(all_videos)}")
    print(f"Videos to run: {len(videos_to_run)}")
    for v in videos_to_run:
        print(f"  -> {v[0]}")
    print(f"Videos skipped (already processed): {len(videos_skipped)}")
    for v in videos_skipped:
        print(f"  x {v}")
    print("================================\n")

    # Run the videos
    for video_path, ranges_csv_path, output_images_dir, labels_csv_path, video_name in videos_to_run:

        os.makedirs(output_images_dir, exist_ok=True)

        start_time = time.perf_counter()
        process_video(
            video_path,
            ranges_csv_path,
            output_images_dir,
            labels_csv_path,
            fps_factor=1,
            dataset_name=video_name
        )
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"{video_path} took {elapsed:.2f} seconds")
