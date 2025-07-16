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

        frame_id = metadata["scan_parameters"]["current_scanned_frame_id"]  # []
        h1 = metadata["uav"]["altitude_agl_meters"]  # [m]
        phi1 = metadata["payload"]["pitch_deg"]  # [deg] TODO: regarding to world or payload
        hfov1 = metadata["payload"]["field_of_view_deg"]  # [deg]

        # Reproject and compute homography
        matrix = np.array(metadata["transformation_matrix"])  # should be shape (4, 4)
        transf = matrix.astype(float).flatten().tolist()

        if frame_id == 1:
            fire_existence = 1
        else:
            fire_existence = 0

        result = {
            "fire_existence": fire_existence,
            "latitude": 0,
            "longitude": 1,
            "bounding_box": bbox,
            "confidence_pct": 2,
        } # TODO: Update the return based on ICD

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

