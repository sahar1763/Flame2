from torchvision import transforms, models
from PIL import Image
import yaml
import importlib.resources as pkg_resources
import torch.backends.cudnn as cudnn

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

        # <<< Warm up >>>
        print("Start warmup")
        self._warmup_phase2()
        print("End warmup")

    def _warmup_phase2(self) -> None:
        """
        Run a full phase2 pass with dummy RGB image + bbox to warm pipelines:
        PIL transforms, tensor move, model forward, softmax, and postprocess.
        """
        try:
            image_rgb_size = self.config['image']['rgb_size']
            H, W = image_rgb_size[1], image_rgb_size[0]
            dummy_img = np.zeros((H, W, 3), dtype=np.uint8)

            # bbox at the center
            cx, cy, r = W // 2, H // 2, 140
            dummy_bbox = (cx - r, cy - r, cx + r, cy + r)

            dummy_md = {
                "uav": {
                    "altitude_agl_meters": 2400.0,
                    "roll_deg": 0.5,
                    "pitch_deg": -1.2,
                    "yaw_deg": 45.0,
                },
                "payload": {
                    "pitch_deg ": -12.0,
                    "azimuth_deg ": 128.0,
                    "field_of_view_deg ": 2.5,
                    "resolution_px": [1920, 1080],
                },
                "geolocation": {
                    "latitude": 31.0461,
                    "transformation_matrix": np.eye(4).tolist(),
                    "longitude": 34.8516,
                },
                "investigation_parameters": {
                    "detection_latitude": 31.0421,
                    "detection_longitude ": 34.8516,
                    "detected_bounding_box ": [31.1, 34.8, 31.0, 34.9]

                },
                "timestamp": "2025-04-08T12:30:45.123Z",  # ISO 8601 format
            }

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            _ = self.phase2(dummy_img, dummy_bbox, dummy_md)

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

        print(f"frame_id: {frame_id}")

    
    def phase1(self, image1: np.ndarray, metadata: dict):
        """
        Process a new IR frame using stored Scan0 reference.
        """
        frame_id = metadata["scan_parameters"]["current_scanned_frame_id"] # []
        h1 = metadata["uav"]["altitude_agl_meters"] # [m]
        phi1 = metadata["payload"]["pitch_deg"] # [deg] TODO: regarding to world or payload
        hfov1 = metadata["payload"]["field_of_view_deg"] # [deg]

        image0 = self.frames[frame_id]
        if np.array_equal(image0, image1):
            print("Images are identical. Test passed.")
        else:
            print("Images are different. Test failed.")

        if frame_id == 3:
            l = 3
        elif frame_id == 1:
            l = 1
        else:
            l = 0
        # Return structured result
        results = []
        for i in range(l):
            results.append({
                'latitude': 0,
                'longitude': 1,
                'bounding_box': [2, 3, 4, 5],
                'confidence_pct': 6, # TODO (Maayan) switch to intensity parameter according to assaf instructions
                'required_fov2': 7,
                'current_IR_fov': 8,
            })

        return results


    def phase2(self, image1: np.ndarray, bbox, metadata: dict):
        """
        Process a new RGB frame.
        """
        # Using transformation function to convert World coordinates to RGB image coordinates
        tt0 = time.perf_counter()

        # === 3. Define bbox
        bbox_pixels = (960-140, 540-140, 960+140, 540+140)  # example bounding box

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
        print(f"\n=== Inference Timing for Cropping === {total_time * 1000:.2f} msec\n")

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

