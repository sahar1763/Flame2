from torchvision import transforms, models
from PIL import Image
import yaml
import importlib.resources as pkg_resources

import requests
from typing import List, Tuple, Optional

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
        self.detector_client = DetectorClient(
            server_url=self.config["detector"]["server_url"],  # add to config.yaml
            geo_server_address=self.config["detector"]["geo_server_address"]
        )

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

        # Create corners:
        pts_image = self.points0_arrange

        transformation_matrix = np.array(metadata["geolocation"]["transformation_matrix"])  # should be shape (4, 4)
        flatten_transformation_matrix = transformation_matrix.astype(float).flatten().tolist()

        img_size = self.config["image"]["ir_size"]
        phi_deg = metadata["uav"]["pitch_deg"]
        theta_deg = metadata["uav"]["yaw_deg"]
        h = metadata["uav"]["altitude_agl_meters"]
        hfov_deg = metadata["payload"]["field_of_view_deg"]
        ground_corners = pixel2geo(theta_deg=theta_deg, phi_deg=phi_deg, h=h, hfov_deg=hfov_deg, img_size=img_size)

        # pixels = [[int(x), int(y)] for (y, x) in pts_image]
        #
        # # Compute corners and store them
        # geo_results = self.detector_client.pixels_to_geo(flatten_transformation_matrix, pixels)
        #
        # ground_corners = np.array([
        #     g if g is not None else [np.nan, np.nan]
        #     for g in geo_results
        # ])

        self.corners[frame_id] = ground_corners
    
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
            sensor_elevation_deg=metadata["payload"]["pitch_deg"],
        )  # angle regarding to world
        hfov = metadata["payload"]["field_of_view_deg"]  # [deg]

        # Load scan0 image and corners
        image0 = self.frames[frame_id]
        corners_0 = self.corners[frame_id]

        # Fire Max Size (length)
        fire_size = self.config['fire']['max_size_m']  # [m]
        # DB_Scan parameters
        min_samples_factor = self.config['dbscan']['min_samples_factor']
        eps_distance_factor = self.config['dbscan']['eps_distance_factor']
        # Important Calculation
        rgb_height, rgb_width = self.config['image']['rgb_size']  # [width, height]
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

        # # Convert Geo coordinates [lon, lat] -> [lat, lon] for API compatibility
        # geo_coords = [
        #     [c[1], c[0]] if c is not None else None
        #     for c in corners_0
        # ]
        #
        # # Get projected pixel coordinates in image1
        # pixels_img0_at_img1_list = self.detector_client.geos_to_pixels(flatten_transformation_matrix, geo_coords)
        #
        # pixels_img0_at_img1 = np.array([
        #     g if g is not None else [np.nan, np.nan]
        #     for g in pixels_img0_at_img1_list
        # ])  # TODO: Verify the returned format

        img_size = self.config["image"]["ir_size"]
        phi_deg = metadata["uav"]["pitch_deg"]
        theta_deg = metadata["uav"]["yaw_deg"]
        h = metadata["uav"]["altitude_agl_meters"]
        hfov_deg = metadata["payload"]["field_of_view_deg"]
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
                                        norm_intensity=self.config['scoring'][
                                            'norm_intensity'])  # TODO (Maayan) switch to intensity parameter according to assaf instructions

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
                'confidence_pct': scores[i],
                # TODO (Maayan) switch to intensity parameter according to assaf instructions
                'required_fov2': required_fov2[i]
            })

        # Plots of phase1 for debugging
        # plot_phase1(diff_map, corners_0, corners_1, centers, bboxes, frame_id) # TODO: Delete later

        return results

    def phase2(self, image1: np.ndarray, metadata: dict):
        """
        Process a new RGB frame.
        """
        # Using transformation function to convert World coordinates to RGB image coordinates
        tt0 = time.perf_counter()

        # === 3. Define bbox
        # bbox_pixels = (960-140, 540-140, 960+140, 540+140)  # example bounding box
        bbox = metadata["investigation_parameters"]["detected_bounding_box"]
        bbox_pixels = bbox

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

        result = predict_crops_majority_vote(
            crops=test_tensors,
            model=self.model,
            bbox=bbox_pixels,
            device=self.device,
            original_image=image1,
            crops_np=cropped_images_np,
            plot=False
        )

        print("Final Prediction:", result["final_prediction"])
        print("Confidence:", f"{result['confidence']:.2f}")
        print("BBox:", result["bbox"])

        return result



class DetectorClient:
    def __init__(self,
                 server_url: str,
                 geo_server_address: str = "http://localhost:8080",
                 server_id: int = 1):
        self.server_url = server_url.rstrip("/")
        self.geo_server_address = geo_server_address
        self.server_id = server_id

    def start(self, transf: List[float], pixel: List[float]) -> bool:
        if len(transf) != 16:
            raise ValueError("Transformation matrix must be 16 float values")

        payload = {
            "geo_server_address": self.geo_server_address,
            "endpoint": "/pixel_to_geo",
            "geo_request_message_body": {
                "server_id": self.server_id,
                "body": {
                    "transf": transf,
                    "pixels": {
                        "ptc": {
                            "x": pixel[0],
                            "y": pixel[1]
                        }
                    }
                }
            },
            "geo_response_message_body": {
                "geo_coordinates": {},
                "status": None,
                "timestamp": None
            }
        }

        try:
            response = requests.post(f"{self.server_url}/start", json=payload)
            response.raise_for_status()
            print("Detector started")
            return True
        except requests.RequestException as e:
            print(f"Start failed: {e}")
            return False

    def stop(self) -> bool:
        try:
            response = requests.post(f"{self.server_url}/stop")
            response.raise_for_status()
            print("Detector stopped")
            return True
        except requests.RequestException as e:
            print(f"Stop failed: {e}")
            return False

    def pixels_to_geo(self, transf: List[float], pixels: List[List[float]]) -> List[Optional[Tuple[float, float]]]:
        if len(transf) != 16:
            raise ValueError("Transformation matrix must be 16 float values")

        results = []

        for pixel in pixels:
            payload = {
                "geo_server_address": self.geo_server_address,
                "endpoint": "/pixel_to_geo",
                "geo_request_message_body": {
                    "server_id": self.server_id,
                    "body": {
                        "transf": transf,
                        "pixels": {
                            "ptc": {
                                "x": pixel[0],
                                "y": pixel[1]
                            }
                        }
                    }
                },
                "geo_response_message_body": {
                    "geo_coordinates": {},
                    "status": None,
                    "timestamp": None
                }
            }

            try:
                response = requests.post(f"{self.server_url}/start", json=payload)
                response.raise_for_status()
                data = response.json()
                coords = data.get("geo_coordinates", {}).get("ptc", None)

                if coords and "x" in coords and "y" in coords:
                    results.append((coords["x"], coords["y"]))
                else:
                    results.append(None)

            except requests.RequestException as e:
                print(f"pixel_to_geo failed: {e}")
                results.append(None)

        return results

    def geos_to_pixels(self, transf: List[float], geo_coords: List[List[float]]) -> List[Optional[List[float]]]:
        if len(transf) != 16:
            raise ValueError("Transformation matrix must be 16 float values")

        results = []

        for coord in geo_coords:
            if len(coord) != 3:
                results.append(None)
                continue

            lon, lat, alt = coord

            payload = {
                "geo_server_address": self.geo_server_address,
                "endpoint": "/geo_to_pixel",
                "geo_request_message_body": {
                    "server_id": self.server_id,
                    "body": {
                        "transf": transf,
                        "coords": {
                            "ptc": {
                                "x": lon,
                                "y": lat,
                                "z": alt
                            }
                        }
                    }
                },
                "geo_response_message_body": {
                    "geo_coordinates": {},
                    "status": None,
                    "timestamp": None
                }
            }

            try:
                response = requests.post(f"{self.server_url}/start", json=payload)
                response.raise_for_status()
                data = response.json()
                ptc = data.get("geo_coordinates", {}).get("ptc", None)

                if ptc and "x" in ptc and "y" in ptc:
                    results.append([ptc["x"], ptc["y"]])
                else:
                    results.append(None)

            except requests.RequestException as e:
                print(f"geo_to_pixel failed: {e}")
                results.append(None)

        return results

