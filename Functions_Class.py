from torchvision import transforms, models
from PIL import Image

from utils_Frame import *
from utils_phase2_flow import *

class RealScanManager:
    def __init__(self):
        self.frames = {}    # frame_id: frame
        self.corners = {}   # frame_id: corners
        # Format: [top-left, top-right, bottom-right, bottom-left]
        image_width = 1280
        image_height = 720
        self.points0_arrange = generate_uniform_grid(image_height, image_width, points_num=36)

        # Phase2 - Loading Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load checkpoint
        checkpoint = torch.load("resnet_fire_classifier.pt", map_location=self.device)

        # Define model and load state
        num_classes = 2
        resnet = models.resnet18(pretrained=False)
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
        
        # Compute corners and store them
        ground_corners = Reuven_Function(pts_image, metadata, pixel2world) # TBD to Reuven requirements
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
        fire_size = 1  # [m]
        # DB_Scan parameters
        min_samples_factor = 10
        eps_distance_factor = 1.5
        # Important Calculation
        # Calculations
        Slant_Range = h1 / np.cos(np.deg2rad(phi1))  # Slant range from camera to ground (meters)
        IFOV = hfov1 / 1920 / 180 * np.pi  # Instantaneous Field of View [urad]
        GSD = Slant_Range * IFOV  # Ground Sampling Distance [meters per pixel]
    
        fire_length_pixel = np.floor(fire_size / GSD)
        fire_num_pixel = fire_length_pixel ** 2
    
        # FOV calc for Phase 2
        patch_length = 224  # total patch size in pixels
        ratio_patch = 0.7  # fire ratio within the patch
        ratio_image = 0.25  # fire ratio within the RGB image
        IR2RGB_ratio = 1920 / 1280  # resolution ratio between RGB and IR images
        rgb_len = 1080
        min_fov = 2.2  # degrees - minimal allowed FOV
        max_fov = 60.0  # degrees - maximal allowed FOV
    
        # image_height = image1.shape[0]
        # image_width = image1.shape[1]

        

        # Reproject and compute homography
        pixels_img0_at_img1 = Reuven_Function(corners_0, metadata, world2pixel) # TBD to Reuven requirements
        pts_image = self.points0_arrange
        
        H = create_homography(pts_image, pixels_img0_at_img1)

        # Warp image0 to image1 frame
        image0_proj = cv2.warpPerspective(image0, H, (image1.shape[1], image1.shape[0]),
                                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=np.median(image0))

        # Preprocess, compare, cluster, and score
        image1, image0_proj = preprocess_images(image1, image0_proj, applying=0)
        diff_map = compute_positive_difference(image0_proj, image1)
        # diff_map = postprocess_difference_map(diff_map, image1, threshold=20, temp_threshold=None)

        # Step 1: Compute DBSCAN parameters based on estimated fire characteristics
        min_samples = int(np.ceil(fire_num_pixel / min_samples_factor))
        eps_distance = int(np.floor(fire_length_pixel * eps_distance_factor))
        # Step 2: Run conditional DBSCAN clustering to identify potential fire regions
        centers, label_map, bboxes = find_cluster_centers_conditional(
            diff_map=diff_map,
            threshold=10,  # Only consider pixels with diff > 10
            eps=eps_distance,  # Clustering radius
            min_samples=min_samples,  # Minimum number of points in cluster
            min_contrast=10  # Contrast-based center selection
        )
        # Compute scores
        scores = compute_cluster_scores(label_map, image1, GSD) # TODO (Maayan) switch to intentsity parameter according to assaf instrunctions

        # === Compute Required FOVs Based on Detected Cluster Bounding Boxes ===
        # Estimate the desired fire size in RGB pixel scale (target size for zoom decision)
        pixels_RGB_at_patch = patch_length * ratio_patch  # target fire size in RGB pixels

        # FOV calculation
        required_fov2 = []
        for bbox in bboxes:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            fire_size_IR = max(width, height)
            fire_size_RGB = fire_size_IR * IR2RGB_ratio
            fov = hfov1 / (ratio_image * rgb_len / fire_size_RGB)
            required_fov2.append(round(np.clip(fov, min_fov, max_fov, 2))

        # Return structured result
        results = []
        for i in range(len(centers)):
            results.append({
                'Frame_index': frame_id,
                'loc': centers[i],
                'bbox': bboxes[i],
                'confidence_pct': scores[i], # TODO (Maayan) switch to intentsity parameter according to assaf instrunctions
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
        bbox_pixels = Reuven_Function(bbox, metadata, world2pixel) # TODO: TBD to Reuven requirements

        # === 4. Define crop factors and transformation
        crop_factors = [1.5 ** 0.5, 2 ** 0.5, 2]
        image_size = 224

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
