from torchvision import transforms, models
from PIL import Image
import yaml
import os
import importlib.resources as pkg_resources
import numpy as np
import time
import requests
from typing import List, Tuple, Optional, Iterable, Dict, Any
import httpx

from wildfire_detector.utils_Frame import *
from wildfire_detector.utils_phase2_flow import *
from wildfire_detector.TensorRT_infer import TRTInference

import subprocess
import shutil

class ScanManager:
    def __init__(self, config_path=None):
        # Load config.yaml from package
        if config_path is None:
            with pkg_resources.open_text(__package__, "config.yaml") as f:
                self.config = yaml.safe_load(f)
        else:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

        # Initialize DetectorClient
        self.detector_client = DetectorClient(self.config["detector"])

        # === Phase 0 ===
        self.frames = {}    # frame_id: frame
        self.corners = {}   # frame_id: corners. Format: [top-left, top-right, bottom-right, bottom-left]
        self.centers_pixels0 = {}
        self.cluster_info0 = {}
        self.cluster_descriptors0 = {}

        ir_height, ir_width = self.config['image']['ir_size']
        self.points0_arrange = generate_uniform_grid(ir_height, ir_width, points_num=self.config['grid']['points_per_frame'])

        # --- Phase 2: Load TensorRT Engine ---
        engine_path = self.load_or_build_trt_engine()

        print(f"[INFO] Loading TensorRT engine: {engine_path}")
        self.model = TRTInference(engine_path)  # <-- TensorRT wrapper
        self.is_trt = True

        # <<< Warm up >>>
        print("\033[1m\033[96m+++++ Start warmup +++++\033[0m")
        self._warmup_phase2()
        print("\033[1m\033[96m+++++ End warmup +++++\033[0m")

    def _warmup_phase2(self) -> None:
        """
        Run a full phase2 pass with dummy RGB image + bbox to warm pipelines:
        PIL transforms, tensor move, model forward, softmax, and postprocess.
        """
        try:
            image_rgb_size = self.config['image']['rgb_size']
            img_height, img_width = image_rgb_size[0], image_rgb_size[1]
            dummy_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

            # bbox at the center
            bbox_center_y, bbox_center_x, box_half_size = img_height // 2, img_width // 2, 140
            dummy_bbox = (bbox_center_y - box_half_size, bbox_center_x - box_half_size, 0, bbox_center_y + box_half_size, bbox_center_x + box_half_size, 0)

            # TODO: Insert rational values
            self.dummy_md = {
                "uav": {
                    "altitude_agl_meters": 2400.0,
                    "roll_deg": 0,
                    "pitch_deg": 0,
                    "yaw_deg": 0,
                },
                "payload": {
                    "elevation_deg": -90,
                    "azimuth_deg": 0,
                    "field_of_view_deg": 2.2,
                    "resolution_px": [1920, 1080],
                },
                "geolocation": {
                    "transformation_matrix": np.eye(4, dtype=float).ravel(order="C").tolist(),
                    "latitude": 31.0461, # NonUsed
                    "longitude": 34.8516, # NonUsed
                },
                "investigation_parameters": {
                    "detection_latitude": 31.0421,
                    "detection_longitude": 34.8516,
                    "detection_altitude": 0.0000,
                    "detected_bounding_box": [31.1, 34.8, 0.0, 31.0, 34.9, 0.0]
                },
                "scan_parameters": {
                    "current_scanned_frame_id": 35,
                    "total_scanned_frames": 173,
                },
                "timestamp": "2025-04-08T12:30:45.123Z",  # ISO 8601 format
            }

            self.dummy_md["investigation_parameters"]["detected_bounding_box"] = dummy_bbox

            self.warmup = True # BBox input in warmup is in pixels, skip using geo2pix conversion
            for i in range(self.config["warmup"]["num_iterations"]):
                _ = self.phase2(dummy_img, self.dummy_md)
            self.warmup = False

        except Exception as e:
            print(f"[phase2 warmup] skipped: {e}")

    def phase0(self, frame: np.ndarray, metadata: dict):
        """
        Store initial scan frame and its projected corners.
        """
        frame_id = metadata["scan_parameters"]["current_scanned_frame_id"]

        # Store the frame
        self.frames[frame_id] = frame.copy()

        # === Step 1: Create uniform pixel points ===
        pts_image = self.points0_arrange  # shape (N, 2), order: (y, x)

        # === Step 2: Normalize to [0, 1] for GeoReg ===
        ir_height, ir_width = self.config['image']['ir_size']
        normalized_pixels = np.stack([
            pts_image[:, 1] / (ir_width - 1),  # x normalized
            pts_image[:, 0] / (ir_height - 1),  # y normalized
        ], axis=1)  # shape (N, 2)

        # === Step 3: Prepare transformation matrix ===
        flatten_transformation_matrix = metadata["geolocation"]["transformation_matrix"]  # should be List Length=16

        # === Step 4: Call updated DetectorClient method ===
        ground_corners = self.detector_client.georeg_pixels_to_latlon_batch(
            transf16=flatten_transformation_matrix,
            pixels_xy_norm=normalized_pixels
        )   # lat, lon, alt

        # === Step 5: Save result ===
        self.corners[frame_id] = ground_corners  # ndarray(N, 3)

        # === Step 6: Clustering ===
        drone_height = metadata["uav"]["altitude_agl_meters"]  # [m]
        projection_angle = camera_angle_from_vertical(
            platform_roll_deg=metadata["uav"]["roll_deg"],
            platform_pitch_deg=metadata["uav"]["pitch_deg"],
            platform_yaw_deg=metadata["uav"]["yaw_deg"],
            sensor_azimuth_deg=metadata["payload"]["azimuth_deg"],
            sensor_elevation_deg=metadata["payload"]["elevation_deg"],
        )  # angle regarding the world

        # IF camera is almost horizontal
        if projection_angle > 89:
            return []

        hfov = metadata["payload"]["field_of_view_deg"]  # [deg]
        # Fire Max Size (length)
        fire_size = self.config['fire']['max_size_m']  # [m]
        # Important Calculation
        ir_height, ir_width = self.config['image']['ir_size']
        Slant_Range = drone_height / np.cos(np.deg2rad(projection_angle))  # Slant range from camera to ground (meters)
        IFOV = hfov / ir_width / 180 * np.pi  # Instantaneous Field of View [urad]
        GSD = Slant_Range * IFOV  # Ground Sampling Distance [meters per pixel]
        fire_length_pixel = np.max([np.floor(fire_size / GSD),1]) # if expected fire below 1 pixel search for fire of at least 1 pixel

        # Compute DBSCAN parameters based on estimated fire characteristics
        min_samples_factor = self.config['dbscan']['min_samples_factor']
        eps_distance_factor = self.config['dbscan']['eps_distance_factor']
        eps_distance = int(np.clip((np.floor((fire_length_pixel / 2)*np.sqrt(eps_distance_factor))),1,10)) # Need to verify that expected pixels are within the radius
        min_samples = int(np.floor(min_samples_factor * eps_distance ** 2))

        # Preprocess, compare, cluster, and score
        image0 = preprocess_images(frame, applying=self.config['preprocessing']['apply'])
        image0_centers_pixels, image0_label_map, image0_bboxes_pixels = find_cluster_centers_conditional(
            diff_map=image0,
            threshold=self.config['dbscan']['diff_threshold'],  # Only consider pixels with diff > diff_threshold
            eps=eps_distance,  # Clustering radius
            min_samples=min_samples,  # Minimum number of points in cluster
            min_contrast=self.config['dbscan']['min_contrast']  # Contrast-based center selection
        )

        if len(image0_centers_pixels) > 0:
            cluster_info_img0 = compute_cluster_size_maxval(image0_label_map, frame, GSD)
            # --- Compute cluster descriptors ---
            final_descriptors_img0, _ = extract_orb_descriptors(
                image=image0,
                cluster_centers=image0_centers_pixels,
                patch_size_px=fire_length_pixel
            )
        else:
            cluster_info_img0 = []
            final_descriptors_img0 = []

        self.centers_pixels0[frame_id] = image0_centers_pixels
        self.cluster_info0[frame_id] = cluster_info_img0
        self.cluster_descriptors0[frame_id] = final_descriptors_img0

    def phase1(self, image1: np.ndarray, metadata: dict):
        """
        Process a new IR frame using stored Scan0 reference.
        """
        frame_id = metadata["scan_parameters"]["current_scanned_frame_id"]  # []
        drone_height = metadata["uav"]["altitude_agl_meters"]  # [m]
        projection_angle = camera_angle_from_vertical(
            platform_roll_deg=metadata["uav"]["roll_deg"],
            platform_pitch_deg=metadata["uav"]["pitch_deg"],
            platform_yaw_deg=metadata["uav"]["yaw_deg"],
            sensor_azimuth_deg=metadata["payload"]["azimuth_deg"],
            sensor_elevation_deg=metadata["payload"]["elevation_deg"],
        )  # angle regarding the world

        # IF camera is almost horizontal
        if projection_angle > 89:
            return []

        hfov = metadata["payload"]["field_of_view_deg"]  # [deg]

        # Fire Max Size (length)
        fire_size = self.config['fire']['max_size_m']  # [m]
        # DB_Scan parameters
        min_samples_factor = self.config['dbscan']['min_samples_factor']
        eps_distance_factor = self.config['dbscan']['eps_distance_factor']
        # Important Calculation
        rgb_height, rgb_width = self.config['image']['rgb_size']  # [width, height]
        ir_height, ir_width = self.config['image']['ir_size']
        Slant_Range = drone_height / np.cos(np.deg2rad(projection_angle))  # Slant range from camera to ground (meters)
        IFOV = hfov / ir_width / 180 * np.pi  # Instantaneous Field of View [urad]
        GSD = Slant_Range * IFOV  # Ground Sampling Distance [meters per pixel]

        fire_length_pixel = np.max(
            [np.floor(fire_size / GSD), 1])  # if expected fire below 1 pixel search for fire of at least 1 pixel
        fire_num_pixel = fire_length_pixel ** 2

        # FOV calc for Phase 2
        ratio_image = self.config['fire']['ratio_in_rgb_image']  # fire ratio within the RGB image
        IR2RGB_ratio = rgb_width / ir_width  # resolution ratio between RGB and IR images
        min_fov = self.config['fov']['min_deg']  # degrees - minimal allowed FOV
        max_fov = self.config['fov']['max_deg']  # degrees - maximal allowed FOV

        # Prepare transformation matrix
        flatten_transformation_matrix = metadata["geolocation"]["transformation_matrix"]  # should be List Length=16

        # Load scan0 image and info
        image0 = self.frames[frame_id]
        corners_0 = self.corners[frame_id]  # at world coordinates  # lat, lon, alt
        centers_pixels0_org = self.centers_pixels0[frame_id]
        cluster_info_img0 = self.cluster_info0[frame_id]
        final_descriptors_img0 = self.cluster_descriptors0[frame_id]

        # Preprocess, compare, cluster, and score
        image1 = preprocess_images(image1, applying=self.config['preprocessing']['apply'])
        # Step 1: Compute DBSCAN parameters based on estimated fire characteristics
        eps_distance = int(np.clip((np.floor((fire_length_pixel / 2) * np.sqrt(eps_distance_factor))), 1,
                                   10))  # Need to verify that expected pixels are within the radius
        min_samples = int(np.floor(min_samples_factor * eps_distance ** 2))
        # Step 2: Run conditional DBSCAN clustering to identify potential fire regions
        # === Phase 1: Clustering ===
        image1_centers_pixels, image1_label_map, image1_bboxes_pixels = find_cluster_centers_conditional(
            diff_map=image1,
            threshold=self.config['dbscan']['diff_threshold'],  # Only consider pixels with diff > diff_threshold
            eps=eps_distance,  # Clustering radius
            min_samples=min_samples,  # Minimum number of points in cluster
            min_contrast=self.config['dbscan']['min_contrast']  # Contrast-based center selection
        )

        # IF no detection, return empty array
        if len(image1_centers_pixels) == 0:
            return []

        # --- Compute cluster sizes and max values ---
        cluster_info_img1 = compute_cluster_size_maxval(image1_label_map, image1, GSD)

        if len(centers_pixels0_org) > 0:
            # --- Compute image1 cluster descriptors ---
            final_descriptors_img1, _ = extract_orb_descriptors(
                image=image1,
                cluster_centers=image1_centers_pixels,
                patch_size_px=fire_length_pixel
            )

            # Get image resolution for normalization
            ir_height, ir_width = self.config['image']['ir_size']

            # --- Geo → normalized pixel coordinates in image-1 ---
            pixels_norm = self.detector_client.georeg_latlon_to_pixels_batch(
                transf16=flatten_transformation_matrix,
                coords_latlon_alt=corners_0
            )  # shape (N,2) -> [y_norm, x_norm]

            # --- Normalized → IR image pixel coordinates ---
            pixels_img0_at_img1 = np.zeros_like(pixels_norm, dtype=np.float32)
            pixels_img0_at_img1[:, 0] = pixels_norm[:, 0] * (ir_height - 1)  # y
            pixels_img0_at_img1[:, 1] = pixels_norm[:, 1] * (ir_width - 1)  # x

            pts_image = self.points0_arrange

            homography_mat = create_homography(pts_image, pixels_img0_at_img1)

            # Project image0 points to image1 coordinates
            centers_pixels0_array = np.array(centers_pixels0_org)
            image0_centers_pixels = project_points_with_homography(centers_pixels0_array, homography_mat).tolist()

            # --- Build cluster dictionaries with descriptors and metadata ---
            clusters_phase0 = {}
            for idx, center in enumerate(image0_centers_pixels):
                clusters_phase0[idx] = {
                    "center": center,
                    "descriptor": final_descriptors_img0[idx] if final_descriptors_img0 is not None else None,
                    "area": cluster_info_img0.get(idx, {}).get("size", 1),
                    "max_val": cluster_info_img0.get(idx, {}).get("max_val", 0.0)
                }

            clusters_phase1 = {}
            for idx, center in enumerate(image1_centers_pixels):
                clusters_phase1[idx] = {
                    "center": center,
                    "descriptor": final_descriptors_img1[idx] if final_descriptors_img1 is not None else None,
                    "area": cluster_info_img1.get(idx, {}).get("size", 1),
                    "max_val": cluster_info_img1.get(idx, {}).get("max_val", 0.0)
                }

            # --- Compute the cost matrix ---
            cost_matrix = compute_cluster_cost_matrix(
                clusters_phase1,
                clusters_phase0,
                gsd=GSD,
                config=self.config,
            )

            # --- Hungarian matching ---
            unmatched_mask, _ = match_clusters_hungarian(cost_matrix)

            # --- Filter phase1 data based on matches ---
            centers_pixels, bboxes_pixels, cluster_info_filtered = filter_unmatched_clusters(
                unmatched_mask,
                image1_centers_pixels,
                image1_bboxes_pixels,
                cluster_info_img1
            )

            # IF no detection, return empty array
            if len(centers_pixels) == 0:
                return []
        else:
            centers_pixels = image1_centers_pixels
            cluster_info_filtered = cluster_info_img1
            bboxes_pixels = image1_bboxes_pixels

        # === Compute scores ===
        scores = compute_cluster_scores(
            cluster_info_filtered,
            norm_size=self.config['scoring']['norm_size'],
            norm_intensity=self.config['scoring']['norm_intensity'],
            weights=self.config['scoring']['scaling_weights'],
        )  # TODO (Maayan) switch to intensity parameter according to assaf instructions

        # === Compute Required FOVs Based on Detected Cluster Bounding Boxes ===
        required_fov2 = []
        for bbox in bboxes_pixels:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            fire_size_IR = max(width, height)
            fire_size_RGB = fire_size_IR * IR2RGB_ratio
            fov = hfov / (ratio_image * rgb_height / fire_size_RGB)
            required_fov2.append(round(np.clip(fov, min_fov, max_fov), 2))

        required_fov2 = np.array(required_fov2, dtype=np.float32)

        # ============================================================
        # === PIXEL → GEO (single batched pix2geo call)
        # ============================================================

        N = centers_pixels.shape[0]

        # --- Build batched pixel array (3N, 2) ---
        pixels_batch = np.zeros((3 * N, 2), dtype=np.float32)

        for i in range(N):
            # center
            pixels_batch[3 * i + 0] = centers_pixels[i]  # [y, x]

            # bbox corners
            pixels_batch[3 * i + 1] = bboxes_pixels[i, 0:2]  # [y1, x1]
            pixels_batch[3 * i + 2] = bboxes_pixels[i, 2:4]  # [y2, x2]

        # --- Normalize pixels ---
        pixels_batch_norm = np.zeros_like(pixels_batch)
        pixels_batch_norm[:, 0] = pixels_batch[:, 0] / (ir_height - 1)  # y
        pixels_batch_norm[:, 1] = pixels_batch[:, 1] / (ir_width - 1)  # x

        # --- Single pix2geo call ---
        lla_batch = self.detector_client.georeg_pixels_to_latlon_batch(
            transf16=flatten_transformation_matrix,
            pixels_xy_norm=pixels_batch_norm
        )  # shape (3N,3) -> [lat, lon, alt]

        # ============================================================
        # === Reconstruct outputs
        # ============================================================

        centers_lla = np.zeros((N, 3), dtype=np.float32)
        bboxes_lla = np.zeros((N, 6), dtype=np.float32)

        for i in range(N):
            # center
            centers_lla[i] = lla_batch[3 * i + 0]

            # bbox corners
            p1 = lla_batch[3 * i + 1]  # lat1, lon1, alt1
            p2 = lla_batch[3 * i + 2]  # lat2, lon2, alt2
            bboxes_lla[i] = np.concatenate([p1, p2])

        # ============================================================
        # === Final structured results
        # ============================================================

        results = []
        for i in range(N):
            results.append({
                'latitude': centers_lla[i, 0],
                'longitude': centers_lla[i, 1],
                'altitude': centers_lla[i, 2],
                'bounding_box': bboxes_lla[i],  # [lat1, lon1, alt1, lat2, lon2, alt2]
                'confidence_pct': scores[i],
                # TODO (Maayan) switch to intensity parameter according to assaf instructions
                'required_fov_deg': required_fov2[i]
            })

        return results

    def phase2(self, image1: np.ndarray, metadata: dict):
        """
        Process a new RGB frame.
        """
        # Using transformation function to convert World coordinates to RGB image coordinates
        tt0 = time.perf_counter()

        # === 3. Define bbox
        # Prepare transformation matrix
        flatten_transformation_matrix = metadata["geolocation"]["transformation_matrix"]  # should be List Length=16

        # Convert bbox [lat1, lon1, lat2, lon2] to [[lat1, lon1], [lat2, lon2]]
        bbox = metadata["investigation_parameters"]["detected_bounding_box"]
        bbox_geo = np.array([
            [bbox[0], bbox[1], bbox[2]],  # top-left # lat lon alt
            [bbox[3], bbox[4], bbox[5]]  # bottom-right
        ])

        # Get image resolution for normalization
        rgb_height, rgb_width = self.config['image']['rgb_size']

        if self.warmup:
            bbox_pixels_array = np.array([
                [bbox[0], bbox[1]],
                [bbox[3], bbox[4]]
            ])
        else:
            # ============================================================
            # === GEO → PIXEL (bbox corners) — single geo2pixel call
            # ============================================================

            # --- Geo → normalized pixel coordinates ---
            bbox_pixels_norm = self.detector_client.georeg_latlon_to_pixels_batch(
                transf16=flatten_transformation_matrix,
                coords_latlon_alt=bbox_geo
            )  # shape (2,2) -> [y_norm, x_norm]

            # --- Normalized → RGB pixel coordinates ---
            bbox_pixels_array = np.zeros_like(bbox_pixels_norm, dtype=np.float32)
            bbox_pixels_array[:, 0] = bbox_pixels_norm[:, 0] * (rgb_height - 1) # y
            bbox_pixels_array[:, 1] = bbox_pixels_norm[:, 1] * (rgb_width - 1)  # x

        # ------------------------------------------------------------
        # === Build bbox from projected pixel points
        # ------------------------------------------------------------

        x_min = int(np.floor(np.min(bbox_pixels_array[:, 1]))) # TODO fixed bug, check for consistence
        y_min = int(np.floor(np.min(bbox_pixels_array[:, 0])))
        x_max = int(np.ceil(np.max(bbox_pixels_array[:, 1])))
        y_max = int(np.ceil(np.max(bbox_pixels_array[:, 0])))

        # Original (unfixed) bbox
        bbox_pixels_raw = (y_min, x_min, y_max, x_max)

        valid_scan, bbox_pixels = self._valid_phase2(bbox_pixels_raw, rgb_height, rgb_width)

        # === 4. Define crop factors and transformation
        crop_factors = self.config['phase2']['crop_factors']
        image_size = self.config['phase2']['net_image_size']

        # Convert and Resize INDIVIDUALLY to ensure they match
        resized_tensors = []
        cropped_images_np = []
        for crop_factor in crop_factors:

            cropped_np = crop_bbox_scaled(image1, bbox_pixels, crop_factor, min_cropsize=image_size)
            cropped_images_np.append(cropped_np)

            # NumPy [H, W, C] -> Tensor [C, H, W]
            t = torch.from_numpy(cropped_np).permute(2, 0, 1).float()

            # Resize all tensors (3, image_size, image_size)
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0),
                size=(image_size, image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            resized_tensors.append(t)

        test_tensors = torch.stack(resized_tensors)  # Shape: [3, 3, image_size, image_size]

        # Final Normalization (Vectorized and Fast)
        test_tensors.div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        test_tensors = (test_tensors - mean) / std

        total_time = time.perf_counter() - tt0
        print(f"\n=== Inference Timing for Preprocess and Cropping === {total_time * 1000:.2f} msec\n")

        final_label, avg_conf = predict_crops_majority_vote_RT(
            crops=test_tensors,
            model=self.model,
            bbox=bbox_pixels,
            original_image=image1,
            crops_np=cropped_images_np,
            plot=False
        )

        result = {
            "fire_existence": final_label,  # 1 - Fire, 0 - No Fire
            "confidence_pct": avg_conf,
            "valid_scan" : int(valid_scan),
            "latitude": metadata["investigation_parameters"]["detection_latitude"],
            "longitude": metadata["investigation_parameters"]["detection_longitude"],
            "altitude": metadata["investigation_parameters"]["detection_altitude"],
            "bounding_box": metadata["investigation_parameters"]["detected_bounding_box"], # returning same bbox as in phase1
        }

        print("Final Prediction:", result["fire_existence"])
        print("Confidence:", f"{result['confidence_pct']:.2f}")

        return result

    def _valid_phase2(self, bbox_pixels_raw, rgb_height, rgb_width):

        y_min, x_min, y_max, x_max = bbox_pixels_raw
        # ------------------------------------------------------------
        # === Determine bbox relation to image
        # ------------------------------------------------------------

        # Fully inside image
        fully_inside = (
                x_min >= 0 and y_min >= 0 and
                x_max < rgb_width and y_max < rgb_height
        )

        # Fully outside image (no overlap at all)
        fully_outside = (
                x_max < 0 or y_max < 0 or
                x_min >= rgb_width or y_min >= rgb_height
        )

        # ------------------------------------------------------------
        # === Apply policy
        # ------------------------------------------------------------

        if fully_inside:
            # Case 2: bbox fully inside
            valid_scan = 2
            bbox_pixels = bbox_pixels_raw

        elif fully_outside:
            # Case 0: bbox fully outside → take full image
            valid_scan = 0
            bbox_pixels = (0, 0, (rgb_height - 1), (rgb_width - 1))

        else:
            # Case 1: bbox partially inside → clip
            valid_scan = 1

            bbox_pixels = (
                max(0, y_min),
                max(0, x_min),
                min((rgb_height - 1), y_max),
                min((rgb_width - 1), x_max)
            )
        return valid_scan, bbox_pixels

    def load_or_build_trt_engine(self) -> str:
        """
        Loads the existing TensorRT engine file, or builds it from the ONNX file located
        inside the wildfire_detector package using trtexec.

        Returns:
            str: Path to the .trt engine file inside the package
        """
        package_root = pkg_resources.files("wildfire_detector")
        onnx_path = str(package_root / "best_model.onnx")
        engine_path = str(package_root / "best_model_fp16.trt")
        net_image_size = self.config['phase2']['net_image_size']

        if os.path.exists(engine_path):
            print(f"[TRT] Found existing engine at: {engine_path}")
            return engine_path

        print("[TRT] Building TensorRT engine...")

        build_cmd = [
            "/usr/src/tensorrt/bin/trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--fp16",
            f"--minShapes=input:1x3x{net_image_size}x{net_image_size}",
            f"--optShapes=input:3x3x{net_image_size}x{net_image_size}",
            f"--maxShapes=input:16x3x{net_image_size}x{net_image_size}",
            f"--shapes=input:1x3x{net_image_size}x{net_image_size}"
        ]

        start = time.perf_counter()
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        duration = time.perf_counter() - start

        if result.returncode != 0:
            raise RuntimeError(f"[TRT] Engine build failed:\n{result.stderr}")

        print(f"[TRT] Engine built in {duration:.1f}s and saved to: {engine_path}")
        return engine_path


# ---------------------------------------------------------------------------
# GeoReg server method: pixel → lat/lon via HTTP
# ---------------------------------------------------------------------------
class DetectorClient:
    def __init__(self, config):
        self.geo_server_base = config["geo_server_base"]
        self.endpoint_pixel2geo = config["endpoint_pixel2geo"]
        self.endpoint_geo2pixel = config["endpoint_geo2pixel"]
        self.server_id = int(config.get("server_id", 1))
        self.timeout_s = float(config.get("timeout_s", 5.0))

    def normalize_georeg_endpoint(self, ep: str | None) -> str:
        if not ep or ep.strip() == "":
            return "/api/pixel_to_geo"
        ep = ep.strip()
        if ep.startswith("/api/"):
            return ep
        if not ep.startswith("/"):
            ep = "/" + ep
        return "/api" + ep

    def georeg_pixels_to_latlon_batch(
        self,
        transf16: List[float],
        pixels_xy_norm: np.ndarray,  # shape (n, 2) for [x, y] normalized to [0, 1]
    ) -> np.ndarray:
        """
        Convert multiple normalized pixels to geographic coordinates using the GeoReg server.

        Returns:
            ndarray of shape (n, 3) containing (lat, lon, alt) per pixel.
        """

        assert pixels_xy_norm.ndim == 2 and pixels_xy_norm.shape[1] == 2, "Expected shape (n, 2) for pixel array"

        # Build request
        ep = self.normalize_georeg_endpoint(self.endpoint_pixel2geo)
        url = self.geo_server_base.rstrip("/") + ep

        pixels_dict = {
            f"ptc{i}": {"x": float(x), "y": float(y)}
            for i, (x, y) in enumerate(pixels_xy_norm)
        }

        payload = {
            "server_id": int(self.server_id),
            "body": {
                "transf": [float(v) for v in transf16],
                "pixels": pixels_dict
            },
        }

        # Send request
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Extract results to ndarray (n, 3)
        coords = data["data"]["res"]["coords"]
        results = []
        for i in range(len(pixels_xy_norm)):
            pt = coords.get(f"ptc{i}")
            results.append([pt["y"], pt["x"], pt["z"]])  # lat, lon, alt

        return np.array(results, dtype=np.float32)

    def georeg_latlon_to_pixels_batch(
            self,
            transf16: List[float],
            coords_latlon_alt: np.ndarray,  # shape (n, 3) for [lat, lon, alt]
    ) -> np.ndarray:
        """
        Convert multiple geographic coordinates to normalized pixel coordinates
        using the GeoReg server.

        Returns:
            ndarray of shape (n, 2) containing (y, x) per coordinate.
        """

        assert coords_latlon_alt.ndim == 2 and coords_latlon_alt.shape[1] == 3, \
            "Expected shape (n, 3) for geo array"

        # Build request
        ep = self.normalize_georeg_endpoint(self.endpoint_geo2pixel)
        url = self.geo_server_base.rstrip("/") + ep

        coords_dict = {
            f"ptc{i}": {
                "x": float(lon),  # lon -> x
                "y": float(lat),  # lat -> y
                "z": float(alt)
            }
            for i, (lat, lon, alt) in enumerate(coords_latlon_alt)
        }

        payload = {
            "server_id": int(self.server_id),
            "body": {
                "transf": [float(v) for v in transf16],
                "coords": coords_dict
            },
        }

        # Send request
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Extract results to ndarray (n, 2)
        pixels = data["data"]["res"]["pixels"]
        results = []
        for i in range(len(coords_latlon_alt)):
            pt = pixels.get(f"ptc{i}")
            results.append([pt["y"], pt["x"]])  # y, x

        return np.array(results, dtype=np.float32)
