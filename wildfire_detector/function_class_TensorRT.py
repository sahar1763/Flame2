from torchvision import transforms, models
from PIL import Image
import yaml
import os
import importlib.resources as pkg_resources

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
        self.detector_client = DetectorClient()

        # === Phase 0 ===
        self.frames = {}    # frame_id: frame
        self.corners = {}   # frame_id: corners. Format: [top-left, top-right, bottom-right, bottom-left]
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
            bbox_center_x, bbox_center_y, box_half_size = img_width // 2, img_height // 2, 140
            dummy_bbox = (bbox_center_x - box_half_size, bbox_center_y - box_half_size, bbox_center_x + box_half_size, bbox_center_y + box_half_size)

            # TODO: Insert rational values
            self.dummy_md = {
                "uav": {
                    "altitude_agl_meters": 2400.0,
                    "roll_deg": 0.5,
                    "pitch_deg": -1.2,
                    "yaw_deg": 45.0,
                },
                "payload": {
                    "pitch_deg": -12.0,
                    "azimuth_deg": 128.0,
                    "field_of_view_deg": 2.5,
                    "resolution_px": [1920, 1080],
                },
                "geolocation": {
                    "latitude": 31.0461,
                    "transformation_matrix": np.eye(4).tolist(),
                    "longitude": 34.8516,
                },
                "investigation_parameters": {
                    "detection_latitude": 31.0421,
                    "detection_longitude": 34.8516,
                    "detected_bounding_box": [31.1, 34.8, 31.0, 34.9]
                },
                "scan_parameters": {
                    "current_scanned_frame_id": 35,
                    "total_scanned_frames": 173,
                },
                "timestamp": "2025-04-08T12:30:45.123Z",  # ISO 8601 format
            }

            self.dummy_md["investigation_parameters"]["detected_bounding_box"] = dummy_bbox

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            for i in range(self.config["warmup"]["num_iterations"]):
                _ = self.phase2(dummy_img, self.dummy_md)

        except Exception as e:
            print(f"[phase2 warmup] skipped: {e}")

    def phase0(self, frame: np.ndarray, metadata: dict):
        """
        Store initial scan frame and its projected corners.
        """
        frame_id = metadata["scan_parameters"]["current_scanned_frame_id"]

        # Store the frame
        self.frames[frame_id] = frame.copy()

        # Create corners:
        pts_image = self.points0_arrange
        pixels = [[int(x), int(y)] for (y, x) in pts_image]

        # normalized pixel to [0 1] range
        ir_height, ir_width = self.config['image']['ir_size']
        normalized_pixels = [
            [x / (ir_width - 1), y / (ir_height - 1)]
            for (y, x) in pts_image
        ]

        transformation_matrix = np.array(metadata["geolocation"]["transformation_matrix"])  # should be shape (4, 4)
        flatten_transformation_matrix = transformation_matrix.astype(float).flatten().tolist()

        # Compute corners and store them
        geo_results = []

        for x_01, y_01 in normalized_pixels:  # TODO: verify the api for pixels in batch
            result = self.detector_client.georeg_pixel_to_latlon(
                geo_server_base=self.config['detector']['geo_server_base'],
                endpoint=self.config['detector']['endpoint_pixel2geo'],
                transf16=flatten_transformation_matrix,
                x_01=x_01,
                y_01=y_01,
                server_id=1
            )
            try:
                coords = result["data"]["res"]["coords"]["ptc"]
                geo_results.append([coords["y"], coords["x"]])  # [lat, lon]
            except Exception:
                geo_results.append(None)

        ground_corners = np.array([
            g if g is not None else [np.nan, np.nan]
            for g in geo_results
        ])

        self.corners[frame_id] = ground_corners
    
    def phase1(self, image1: np.ndarray, metadata: dict):
        """
        Process a new IR frame using stored Scan0 reference.
        """
        frame_id = metadata["scan_parameters"]["current_scanned_frame_id"] # []
        drone_height = metadata["uav"]["altitude_agl_meters"] # [m]
        projection_angle = camera_angle_from_vertical(
            platform_roll_deg=metadata["uav"]["roll_deg"],
            platform_pitch_deg=metadata["uav"]["pitch_deg"],
            platform_yaw_deg=metadata["uav"]["yaw_deg"],
            sensor_azimuth_deg=metadata["payload"]["azimuth_deg"],
            sensor_elevation_deg=metadata["payload"]["pitch_deg"],
        )  # angle regarding to world
        hfov = metadata["payload"]["field_of_view_deg"] # [deg]

        # Load scan0 image and corners
        image0 = self.frames[frame_id]
        corners_0 = self.corners[frame_id]

        # Fire Max Size (length)
        fire_size = self.config['fire']['max_size_m'] # [m]
        # DB_Scan parameters
        min_samples_factor = self.config['dbscan']['min_samples_factor']
        eps_distance_factor = self.config['dbscan']['eps_distance_factor']
        # Important Calculation
        rgb_height, rgb_width = self.config['image']['rgb_size'] # [width, height]
        ir_height, ir_width = self.config['image']['ir_size']
        Slant_Range = drone_height / np.cos(np.deg2rad(projection_angle))  # Slant range from camera to ground (meters)
        IFOV = hfov / rgb_width / 180 * np.pi  # Instantaneous Field of View [urad]
        GSD = Slant_Range * IFOV  # Ground Sampling Distance [meters per pixel]
    
        fire_length_pixel = np.floor(fire_size / GSD)
        fire_num_pixel = fire_length_pixel ** 2
    
        # FOV calc for Phase 2
        ratio_image = self.config['fire']['ratio_in_rgb_image']  # fire ratio within the RGB image
        IR2RGB_ratio = rgb_width / ir_width  # resolution ratio between RGB and IR images
        min_fov = self.config['fov']['min_deg']  # degrees - minimal allowed FOV
        max_fov = self.config['fov']['max_deg']  # degrees - maximal allowed FOV

        # Reproject and compute homography
        transformation_matrix = np.array(metadata["geolocation"]["transformation_matrix"])  # should be shape (4, 4)
        flatten_transformation_matrix = transformation_matrix.astype(float).flatten().tolist()

        # Convert Geo coordinates [lon, lat] -> [lat, lon] for API compatibility
        geo_coords = [
            [c[1], c[0]] if c is not None else None
            for c in corners_0
        ]

        # Get image resolution for normalization
        ir_height, ir_width = self.config['image']['ir_size']

        # Initialize results list
        pixels_img0_at_img1_list = []

        # Loop through each coordinate and convert
        for coord in geo_coords:  # TODO: verify the api for pixels in batch
            if coord is None:
                pixels_img0_at_img1_list.append(None)
                continue

            lat, lon = coord  # already [lat, lon] # TODO: missing alt?

            try:
                result = self.detector_client.georeg_latlon_to_pixel(
                    geo_server_base=self.config['detector']['geo_server_base'],
                    endpoint=self.config['detector']['endpoint_geo2pixel'],
                    transf16=flatten_transformation_matrix,
                    lat=lat,
                    lon=lon,  # TODO: missing alt?
                    server_id=1
                )
                # Get pixel coordinates and de-normalize
                px = result["data"]["res"]["coords"]["ptc"]["x"] * (ir_width - 1)
                py = result["data"]["res"]["coords"]["ptc"]["y"] * (ir_height - 1)
                pixels_img0_at_img1_list.append([px, py])

            except Exception:
                pixels_img0_at_img1_list.append(None)

        # Format final output
        pixels_img0_at_img1 = np.array([
            p if p is not None else [np.nan, np.nan]
            for p in pixels_img0_at_img1_list
        ])  # TODO: Verify the returned format

        pts_image = self.points0_arrange

        homography_mat = create_homography(pts_image, pixels_img0_at_img1)

        # Warp image0 to image1 frame
        image0_proj = cv2.warpPerspective(image0, homography_mat, (image1.shape[1], image1.shape[0]),
                                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=np.median(image0))

        # Preprocess, compare, cluster, and score
        image1, image0_proj = preprocess_images(image1, image0_proj, applying=self.config['preprocessing']['apply'])
        diff_map = compute_positive_difference(image0_proj, image1)
        diff_map = postprocess_difference_map(diff_map, image1, threshold=self.config['postprocessing']['threshold'],
                                              temp_threshold=self.config['postprocessing']['temp_threshold'])

        # Step 1: Compute DBSCAN parameters based on estimated fire characteristics
        min_samples = int(np.ceil(fire_num_pixel / min_samples_factor))
        eps_distance = int(np.floor(fire_length_pixel * eps_distance_factor))
        # Step 2: Run conditional DBSCAN clustering to identify potential fire regions
        centers, label_map, bboxes = find_cluster_centers_conditional(
            diff_map=diff_map,
            threshold=self.config['dbscan']['diff_threshold'],  # Only consider pixels with diff > diff_threshold
            eps=eps_distance,  # Clustering radius
            min_samples=min_samples,  # Minimum number of points in cluster
            min_contrast=self.config['dbscan']['min_contrast']  # Contrast-based center selection
        )
        # Compute scores
        scores = compute_cluster_scores(label_map, image1, GSD, norm_size=self.config['scoring']['norm_size'],
                                        norm_intensity=self.config['scoring']['norm_intensity']) # TODO (Maayan) switch to intensity parameter according to assaf instructions

        # === Compute Required FOVs Based on Detected Cluster Bounding Boxes ===
        # FOV calculation
        required_fov2 = []
        for bbox in bboxes:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            fire_size_IR = max(width, height)
            fire_size_RGB = fire_size_IR * IR2RGB_ratio
            fov = hfov / (ratio_image * rgb_height / fire_size_RGB)
            required_fov2.append(round(np.clip(fov, min_fov, max_fov), 2))

        # Return structured result
        results = []
        for i in range(len(centers)):
            results.append({
                'Frame_index': frame_id,
                'loc': centers[i],
                'bbox': bboxes[i],
                'confidence_pct': scores[i], # TODO (Maayan) switch to intensity parameter according to assaf instructions
                'required_fov2': required_fov2[i]
            })

        return results

    def phase2(self, image1: np.ndarray, metadata: dict):
        """
        Process a new RGB frame.
        """
        # Using transformation function to convert World coordinates to RGB image coordinates
        tt0 = time.perf_counter()

        # === 3. Define bbox
        # Reproject and compute homography
        transformation_matrix = np.array(metadata["geolocation"]["transformation_matrix"])  # should be shape (4, 4)
        flatten_transformation_matrix = transformation_matrix.astype(float).flatten().tolist()

        # Convert bbox [lat1, lon1, lat2, lon2] to [[lat1, lon1], [lat2, lon2]]
        bbox = metadata["investigation_parameters"]["detected_bounding_box"]
        bbox_geo = [
            [bbox[0], bbox[1]],  # top-left
            [bbox[2], bbox[3]]  # bottom-right
        ]

        # Get image resolution for normalization
        rgb_height, rgb_width = self.config['image']['rgb_size']

        # Initialize results list
        bbox_list_pixel = []

        # Loop through each coordinate and convert
        for coord in bbox_geo:  # TODO: verify the api for pixels in batch
            if coord is None:
                bbox_list_pixel.append(None)
                continue

            lat, lon = coord  # already [lat, lon] # TODO: missing alt?

            try:
                result = self.detector_client.georeg_latlon_to_pixel(
                    geo_server_base=self.config['detector']['geo_server_base'],
                    endpoint=self.config['detector']['endpoint_geo2pixel'],
                    transf16=flatten_transformation_matrix,
                    lat=lat,
                    lon=lon,  # TODO: missing alt?
                    server_id=1
                )
                # Get pixel coordinates and de-normalize
                px = result["data"]["res"]["coords"]["ptc"]["x"] * (rgb_width - 1)
                py = result["data"]["res"]["coords"]["ptc"]["y"] * (rgb_height - 1)
                bbox_list_pixel.append([px, py])

            except Exception:
                bbox_list_pixel.append(None)

        bbox_pixels_array = np.array([
            g if g is not None else [np.nan, np.nan]
            for g in bbox_list_pixel
        ])

        bbox_pixels = (
            int(np.floor(np.min(bbox_pixels_array[:, 0]))),  # x_min
            int(np.floor(np.min(bbox_pixels_array[:, 1]))),  # y_min
            int(np.ceil(np.max(bbox_pixels_array[:, 0]))),  # x_max
            int(np.ceil(np.max(bbox_pixels_array[:, 1])))  # y_max
        ) # TODO: Remove min/max implementation if not needed - based on the return from the detector_client

        # === 4. Define crop factors and transformation
        crop_factors = self.config['phase2']['crop_factors']
        image_size = self.config['phase2']['net_image_size']

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        cropped_images_np = []
        test_tensors = []

        for crop_factor in crop_factors:
            # Crop the image (NumPy RGB)
            cropped_np = crop_bbox_scaled(image1, bbox_pixels, crop_factor)
            cropped_images_np.append(cropped_np)  # Save for plotting

            # Convert to PIL and apply transforms
            pil_img = Image.fromarray(cropped_np)
            test_tensors.append(transform(pil_img))

        test_tensors = [t for t in test_tensors if t.numel() > 0]

        total_time = time.perf_counter() - tt0
        print(f"\n=== Inference Timing for Preprocess and Cropping === {total_time * 1000:.2f} msec\n")

        result = predict_crops_majority_vote_RT(
            crops=test_tensors,
            model=self.model,
            bbox=bbox_pixels,
            original_image=image1,
            crops_np=cropped_images_np,
            plot=False
        )

        print("Final Prediction:", result["final_prediction"])
        print("Confidence:", f"{result['confidence']:.2f}")
        print("BBox:", result["bbox"])

        return result

    def load_or_build_trt_engine(self) -> str:
        """
        Loads the existing TensorRT engine file, or builds it from the ONNX file located
        inside the wildfire_detector package using trtexec.

        Returns:
            str: Path to the .trt engine file inside the package
        """
        package_root = pkg_resources.files("wildfire_detector")
        onnx_path = str(package_root / "resnet_fire_classifier.onnx")
        engine_path = str(package_root / "resnet_fire_classifier_fp16.trt")

        if os.path.exists(engine_path):
            print(f"[TRT] Found existing engine at: {engine_path}")
            return engine_path

        print("[TRT] Building TensorRT engine...")

        build_cmd = [
            "/usr/src/tensorrt/bin/trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--fp16",
            "--minShapes=input:1x3x254x254",
            "--optShapes=input:3x3x254x254",
            "--maxShapes=input:16x3x254x254",
            "--shapes=input:1x3x254x254"
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
    def __init__(self):
        pass

    def normalize_georeg_endpoint(self, ep: str | None) -> str:
        """
        Normalize a GeoReg endpoint path so it always starts with '/api/'.

        Examples:
          None or ''        -> '/api/pixel_to_geo'
          'pixel_to_geo'    -> '/api/pixel_to_geo'
          '/pixel_to_geo'   -> '/api/pixel_to_geo'
          '/api/pixel_to_geo' -> '/api/pixel_to_geo'
        """
        default = "/api/pixel_to_geo"
        if not ep:
            return default
        ep = ep.strip()
        if not ep:
            return default
        if ep.startswith("/api/"):
            return ep
        if ep == "/api":
            return default
        if not ep.startswith("/"):
            ep = "/" + ep
        return "/api" + ep

    def georeg_pixel_to_latlon(
            self,
            geo_server_base: str,
            endpoint: str,
            transf16: List[float],
            x_01: float,
            y_01: float,
            server_id: int = 1,
            timeout_s: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Request pixel→geo conversion from the GeoReg server.

        INPUTS:
          - geo_server_base: e.g. "http://127.0.0.1:9000"
          - endpoint: usually "/api/pixel_to_geo" (normalized automatically).
          - transf16: same 16-element transform used in the manual method.
          - x_01, y_01: pixel coordinates normalized to [0, 1].
                        TL = (0,0), BR = (1,1).
          - server_id: GeoReg server id (as in the detector).

        The payload structure mirrors the detector server's query_pixel_to_geo().
        """
        ep = self.normalize_georeg_endpoint(endpoint)
        url = geo_server_base.rstrip("/") + ep

        payload = {
            "server_id": int(server_id),
            "body": {
                "transf": [float(v) for v in transf16],
                "pixels": {
                    "ptc": {"x": float(x_01), "y": float(y_01)}
                },
            },
        }

        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # The detector code expects:
        #   data["data"]["res"]["coords"]["ptc"] => {'x': lon, 'y': lat}
        return data

    def georeg_latlon_to_pixel(
            self,
            geo_server_base: str,
            endpoint: str,
            transf16: List[float],
            lon: float,
            lat: float,
            server_id: int = 1,
            timeout_s: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Request geo→pixel conversion from the GeoReg server.

        INPUTS:
          - geo_server_base: e.g. "http://127.0.0.1:9000"
          - endpoint: usually "/api/geo_to_pixel" (normalized automatically).
          - transf16: same 16-element transform used in the manual method.
          - lon, lat: geographic coordinates in degrees (WGS84).
          - server_id: GeoReg server id (as in the detector).

        The payload structure mirrors the detector server's query_geo_to_pixel().
        """
        ep = self.normalize_georeg_endpoint(endpoint)
        url = geo_server_base.rstrip("/") + ep

        payload = {
            "server_id": int(server_id),
            "body": {
                "transf": [float(v) for v in transf16],
                "coords": {
                    "ptc": {"x": float(lon), "y": float(lat)}
                },
            },
        }

        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # The detector code expects:
        #   data["data"]["res"]["pixels"]["ptc"] => {'x': x_01, 'y': y_01}
        return data

