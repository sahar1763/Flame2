from torchvision import transforms, models
from PIL import Image
import yaml
import importlib.resources as pkg_resources

import requests
from typing import List, Tuple, Optional

from wildfire_detector.utils_Frame import *
from wildfire_detector.utils_phase2_flow import *

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
        ir_width, ir_height = self.config['image']['ir_size']
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

    def phase0(self, frame: np.ndarray, metadata: dict):
        """
        Store initial scan frame and its projected corners.
        """
        frame_id = metadata["scan_parameters"]["current_scanned_frame_id"]

        # Store the frame
        self.frames[frame_id] = frame.copy()

        # Create corners:
        pts_image = self.points0_arrange

        matrix = np.array(metadata["transformation_matrix"])  # should be shape (4, 4)
        transf = matrix.astype(float).flatten().tolist()

        # Compute corners and store them
        geo_results = self.detector_client.pixels_to_geo(transf, pts_image)

        ground_corners = np.array([
            g if g is not None else [np.nan, np.nan]
            for g in geo_results
        ])

        self.corners[frame_id] = ground_corners

    # def get_frame(self, frame_id: int) -> np.ndarray:
    #     return self.frames.get(frame_id, None)
    #
    # def get_corners(self, frame_id: int) -> np.ndarray:
    #     return self.corners.get(frame_id, None)

    
    def phase1(self, image1: np.ndarray, metadata: dict):
        """
        Process a new IR frame using stored Scan0 reference.
        """
        frame_id = metadata["scan_parameters"]["current_scanned_frame_id"] # []
        h1 = metadata["uav"]["altitude_agl_meters"] # [m]
        phi1 = metadata["payload"]["pitch_deg"] # [deg] TODO: regarding to world or payload
        hfov1 = metadata["payload"]["field_of_view_deg"] # [deg]

        # Load scan0 image and corners
        image0 = self.frames[frame_id]
        corners_0 = self.corners[frame_id]

        # Fire Max Size (length)
        fire_size = self.config['fire']['max_size_m'] # [m]
        # DB_Scan parameters
        min_samples_factor = self.config['dbscan']['min_samples_factor']
        eps_distance_factor = self.config['dbscan']['eps_distance_factor']
        # Important Calculation
        rgb_width, rgb_height = self.config['image']['rgb_size'] # [width, height]
        ir_width, ir_height = self.config['image']['ir_size']
        Slant_Range = h1 / np.cos(np.deg2rad(phi1))  # Slant range from camera to ground (meters)
        IFOV = hfov1 / rgb_width / 180 * np.pi  # Instantaneous Field of View [urad]
        GSD = Slant_Range * IFOV  # Ground Sampling Distance [meters per pixel]
    
        fire_length_pixel = np.floor(fire_size / GSD)
        fire_num_pixel = fire_length_pixel ** 2
    
        # FOV calc for Phase 2
        ratio_image = self.config['fire']['ratio_in_rgb_image']  # fire ratio within the RGB image
        IR2RGB_ratio = rgb_width / ir_width  # resolution ratio between RGB and IR images
        min_fov = self.config['fov']['min_deg']  # degrees - minimal allowed FOV
        max_fov = self.config['fov']['max_deg']  # degrees - maximal allowed FOV

        # Reproject and compute homography
        matrix = np.array(metadata["transformation_matrix"])  # should be shape (4, 4)
        transf = matrix.astype(float).flatten().tolist()

        pixels_img0_at_img1_list = self.detector_client.geos_to_pixels(transf, corners_0)

        pixels_img0_at_img1 = np.array([
            g if g is not None else [np.nan, np.nan]
            for g in pixels_img0_at_img1_list
        ]) # TODO: Verify the returned format


        pts_image = self.points0_arrange
        
        H = create_homography(pts_image, pixels_img0_at_img1)

        # Warp image0 to image1 frame
        image0_proj = cv2.warpPerspective(image0, H, (image1.shape[1], image1.shape[0]),
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
            fov = hfov1 / (ratio_image * rgb_height / fire_size_RGB)
            required_fov2.append(round(np.clip(fov, min_fov, max_fov, 2)))

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


    def phase2(self, image1: np.ndarray, bbox, metadata: dict):
        """
        Process a new RGB frame.
        """
        # Using transformation function to convert World coordinates to RGB image coordinates
        tt0 = time.time()

        # === 3. Define bbox
        # Reproject and compute homography
        matrix = np.array(metadata["transformation_matrix"])  # should be shape (4, 4)
        transf = matrix.astype(float).flatten().tolist()

        # Convert bbox [lat1, lon1, lat2, lon2] to [[lon1, lat1], [lon2, lat2]]
        bbox_geo = [
            [bbox[1], bbox[0]],  # top-left:  [lon, lat]
            [bbox[3], bbox[2]]  # bottom-right: [lon, lat]
        ]
        bbox_list_pixel = self.detector_client.geos_to_pixels(transf, bbox_geo)

        bbox_pixels_array = np.array([
            g if g is not None else [np.nan, np.nan]
            for g in bbox_list_pixel
        ]) # TODO: Verify the returned format

        bbox_pixels = (
            np.min(bbox_pixels_array[:, 0]),  # x_min
            np.min(bbox_pixels_array[:, 1]),  # y_min
            np.max(bbox_pixels_array[:, 0]),  # x_max
            np.max(bbox_pixels_array[:, 1])  # y_max
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

        total_time = time.time() - tt0
        print(f"\n=== Inference Timing for Cropping === {total_time * 1000:.2f} msec\n")

        # tt1 = time.time()
        #
        #
        # total_time = time.time() - tt1

        print(f"\n=== Inference Timing For Loading the Model === {total_time * 1000:.2f} msec\n")

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
                 endpoint: str = "/pixel_to_geo",
                 server_id: int = 1):
        self.server_url = server_url.rstrip("/")
        self.geo_server_address = geo_server_address
        self.endpoint = endpoint
        self.server_id = server_id

    def start(self, transf: List[float], pixel: List[float]) -> bool:
        if len(transf) != 16:
            raise ValueError("Transformation matrix must be 16 float values")

        payload = {
            "geo_server_address": self.geo_server_address,
            "endpoint": self.endpoint,
            "geo_request_message_body": {
                "server_id": self.server_id,
                "body": {
                    "transf": transf,
                    "pixels": [pixel]
                }
            },
            "geo_response_message_body": {
                "geo_coordinates": [],
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
            print(f" Start failed: {e}")
            return False

    def stop(self) -> bool:
        try:
            response = requests.post(f"{self.server_url}/stop")
            response.raise_for_status()
            print(" Detector stopped")
            return True
        except requests.RequestException as e:
            print(f" Stop failed: {e}")
            return False

    def pixels_to_geo(self, transf: List[float], pixels: List[List[float]]) -> List[Optional[Tuple[float, float]]]:
        """
        Sends a list of pixel coordinates and returns list of corresponding (lon, lat) geo coordinates.
        """
        if len(transf) != 16:
            raise ValueError("Transformation matrix must be 16 float values")

        payload = {
            "geo_server_address": self.geo_server_address,
            "endpoint": self.endpoint,
            "geo_request_message_body": {
                "server_id": self.server_id,
                "body": {
                    "transf": transf,
                    "pixels": pixels  # send full list
                }
            },
            "geo_response_message_body": {
                "geo_coordinates": [],
                "status": None,
                "timestamp": None
            }
        }

        try:
            response = requests.post(f"{self.server_url}/start", json=payload)
            response.raise_for_status()
            data = response.json()

            coords = data.get("geo_coordinates", [])
            results = []
            for c in coords:
                if c is None or "longitude" not in c or "latitude" not in c:
                    results.append(None)
                else:
                    results.append((c["longitude"], c["latitude"]))

            return results

        except requests.RequestException as e:
            print(f"pixel_to_geo (batch) failed: {e}")
            return [None] * len(pixels)

    def geos_to_pixels(self, transf: List[float], geo_coords: List[List[float]]) -> List[Optional[List[float]]]:
        """
        Converts geo coordinates (lon, lat) to image pixels using the geo server.
        """
        if len(transf) != 16:
            raise ValueError("Transformation matrix must have 16 float values")

        payload = {
            "geo_server_address": self.geo_server_address,
            "endpoint": "/geo_to_pixel",  # TODO: update
            "geo_request_message_body": {
                "server_id": self.server_id,
                "body": {
                    "transf": transf,
                    "pixels": geo_coords
                }
            },
            "geo_response_message_body": {
                "geo_coordinates": [],
                "status": None,
                "timestamp": None
            }
        }

        try:
            response = requests.post(f"{self.server_url}/start", json=payload)
            response.raise_for_status()
            coords = response.json().get("geo_coordinates", [])

            return [
                [c["x"], c["y"]] if c and "x" in c and "y" in c else None
                for c in coords
            ]

        except requests.RequestException as e:
            print(f"geo_to_pixel failed: {e}")
            return [None] * len(geo_coords)

