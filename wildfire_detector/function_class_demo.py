from torchvision import transforms, models
from PIL import Image
import yaml
import importlib.resources as pkg_resources
import numpy as np

import requests
from typing import List, Tuple, Optional, Iterable, Dict, Any
import httpx

from wildfire_detector.utils_Frame import *
from wildfire_detector.utils_phase2_flow import *
from wildfire_detector.utils_simulation import *

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
        ir_height, ir_width = self.config['image']['ir_size']
        self.points0_arrange = generate_uniform_grid(ir_height, ir_width, points_num=self.config['grid']['points_per_frame'])

        # === Phase 2 - Loading Model ===
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load model from package
        with pkg_resources.files(__package__).joinpath("resnet_fire_classifier.pt").open("rb") as f:
            checkpoint = torch.load(f, map_location=torch.device('cpu'), weights_only=True)

        # Define model and load state
        num_classes = 2
        resnet = models.resnet18(weights=None) #(pretrained=False)
        num_ftrs = resnet.fc.in_features
        resnet.fc = torch.nn.Linear(num_ftrs, num_classes)
        resnet.load_state_dict(checkpoint["model_state"])
        resnet = resnet.to(self.device)
        resnet.eval()
        self.model = resnet

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

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            self.warmup = True # BBox input in warmup is in pixels, skip using geo2pix conversion
            for i in range(self.config["warmup"]["num_iterations"]):
                _ = self.phase2(dummy_img, self.dummy_md)
            self.warmup = False

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

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
        pts_image = self.points0_arrange

        # === Step 2: Normalize to [0, 1] for GeoReg ===
        # Not relevant for demo

        # === Step 3: Prepare transformation matrix ===
        flatten_transformation_matrix = metadata["geolocation"]["transformation_matrix"]  # should be List Length=16

        # === Step 4: Import parameters and using internal conversion function ===
        img_size = self.config["image"]["ir_size"]
        phi_deg = metadata["uav"]["pitch_deg"]
        theta_deg = metadata["uav"]["yaw_deg"]
        h = metadata["uav"]["altitude_agl_meters"]
        hfov_deg = metadata["payload"]["field_of_view_deg"]
        ground_corners = pixel2geo(theta_deg=theta_deg, phi_deg=phi_deg, h=h, hfov_deg=hfov_deg, img_size=img_size) # output in meters for demo

        # === Step 5: Save result ===
        self.corners[frame_id] = ground_corners  # ndarray(N, 3)

    
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
        )  # angle regarding to world
        hfov = metadata["payload"]["field_of_view_deg"]  # [deg]

        # Load scan0 image and corners
        image0 = self.frames[frame_id]
        corners_0 = self.corners[frame_id] # at world coordinates  # x y z in meters for demo

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

        fire_length_pixel = np.max([np.floor(fire_size / GSD),1]) # if expected fire below 1 pixel search for fire of at least 1 pixel
        fire_num_pixel = fire_length_pixel ** 2

        # FOV calc for Phase 2
        ratio_image = self.config['fire']['ratio_in_rgb_image']  # fire ratio within the RGB image
        IR2RGB_ratio = rgb_width / ir_width  # resolution ratio between RGB and IR images
        min_fov = self.config['fov']['min_deg']  # degrees - minimal allowed FOV
        max_fov = self.config['fov']['max_deg']  # degrees - maximal allowed FOV

        # Prepare transformation matrix
        flatten_transformation_matrix = metadata["geolocation"]["transformation_matrix"]  # should be List Length=16

        # Get parameters for geo2pixel conversion
        img_size = self.config["image"]["ir_size"]
        phi_deg = metadata["uav"]["pitch_deg"]
        theta_deg = metadata["uav"]["yaw_deg"]
        h = metadata["uav"]["altitude_agl_meters"]
        hfov_deg = metadata["payload"]["field_of_view_deg"]
        # --- Geo →  pixel coordinates in image-1 - internal function for demo---
        pixels_img0_at_img1, corners_1 = geo2pixel(corners_0=corners_0, theta1=theta_deg, phi1=phi_deg, h1=h, hfov1=hfov_deg, img_size=img_size)

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
        eps_distance = int(np.clip((np.floor((fire_length_pixel / 2)* np.sqrt(eps_distance_factor))),1,10)) # Need to verify that expected pixels are within the radius
        min_samples = int(np.floor(min_samples_factor * eps_distance ** 2))
        # Step 2: Run conditional DBSCAN clustering to identify potential fire regions
        # === Phase 1: Clustering ===
        centers_pixels, label_map, bboxes_pixels = find_cluster_centers_conditional(
            diff_map=diff_map,
            threshold=self.config['dbscan']['diff_threshold'],  # Only consider pixels with diff > diff_threshold
            eps=eps_distance,  # Clustering radius
            min_samples=min_samples,  # Minimum number of points in cluster
            min_contrast=self.config['dbscan']['min_contrast']  # Contrast-based center selection
        )

        # IF no detection, return empty array
        if len(centers_pixels) == 0:
            return []

        # === Compute scores ===
        scores = compute_cluster_scores(
            label_map,
            image1,
            GSD,
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
        # === in demo mode no need to convert back to lla, output is in pixels
        # ============================================================

        # ============================================================
        # === Final structured results
        # ============================================================

        # Return structured result
        # Output is different in demo mode according to debug results # TODO : optional, convert to geo and adjust return
        results = []
        for i in range(len(centers_pixels)):
            results.append({
                'Frame_index': frame_id,
                'loc': centers_pixels[i],
                'bbox': bboxes_pixels[i],
                'confidence_pct': scores[i],
                # TODO (Maayan) switch to intensity parameter according to assaf instructions
                'required_fov2': required_fov2[i]
            }) # Not similar to real output, ICD is different and output is in pixels and not geo

        # Plots of phase1 for debugging
        plot_phase1(diff_map, corners_0, corners_1, centers_pixels, bboxes_pixels, frame_id) # TODO: Delete later

        return results

    def phase2(self, image1: np.ndarray, metadata: dict):
        """
        Process a new RGB frame.
        """
        # Using transformation function to convert World coordinates to RGB image coordinates
        tt0 = time.perf_counter()

        # === 3. Define bbox
        # bbox_pixels = (960-140, 540-140, 960+140, 540+140)  # example bounding box
        bbox = metadata["investigation_parameters"]["detected_bounding_box"]# bbox is calculated based on pixels and not geo, 4 params instead of 6
        # bbox_pixels = bbox
        bbox_pixels_metadata = np.array([
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
        else: # keep similarity flow comparing to real code
            bbox_pixels_array = np.array([
                [bbox[0], bbox[1]],
                [bbox[3], bbox[4]]
            ])

        # ------------------------------------------------------------
        # === Build bbox from projected pixel points
        # ------------------------------------------------------------

        x_min = int(np.floor(np.min(bbox_pixels_array[:, 1]))) # TODO fixed bug, check for consistence
        y_min = int(np.floor(np.min(bbox_pixels_array[:, 0])))
        x_max = int(np.ceil(np.max(bbox_pixels_array[:, 1])))
        y_max = int(np.ceil(np.max(bbox_pixels_array[:, 0])))

        # Original (unfixed) bbox
        bbox_pixels_raw = (x_min, y_min, x_max, y_max)

        valid_scan, bbox_pixels = self._valid_phase2(bbox_pixels_raw, rgb_height, rgb_width)

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

        final_label, avg_conf = predict_crops_majority_vote(
            crops=test_tensors,
            model=self.model,
            bbox=bbox_pixels,
            device=self.device,
            original_image=image1,
            crops_np=cropped_images_np,
            plot=True
        )

        # Output is different in demo mode according to debug results # TODO : optional, convert to geo and adjust return
        result = {
            "final_prediction": final_label,
            "confidence": avg_conf,
            "bbox": bbox
        } # Not same as in the real code

        print("Final Prediction:", result["final_prediction"])
        print("Confidence:", f"{result['confidence']:.2f}")
        print("BBox:", result["bbox"])

        return result

    def _valid_phase2(self, bbox_pixels_raw, rgb_height, rgb_width):

        x_min, y_min, x_max, y_max = bbox_pixels_raw
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
            bbox_pixels = (0, 0, (rgb_width - 1), (rgb_height - 1))

        else:
            # Case 1: bbox partially inside → clip
            valid_scan = 1

            bbox_pixels = (
                max(0, x_min),
                max(0, y_min),
                min((rgb_width - 1), x_max),
                min((rgb_height - 1), y_max)
            )
        return valid_scan, bbox_pixels

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

