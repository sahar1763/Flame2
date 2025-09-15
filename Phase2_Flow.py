import cv2
from torchvision import transforms, models
from PIL import Image

from wildfire_detector.utils_phase2_flow import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)



# def predict_crops_majority_vote(crops, model, bbox, device,
#                                  original_image=None,
#                                  crops_np=None,
#                                  plot=False):
#     """
#     Predicts fire presence using multiple crops of the same image.

#     Args:
#         crops (List[Tensor]): List of transformed image tensors [C, H, W]
#         model (nn.Module): Trained model
#         bbox (tuple): (x_min, y_min, x_max, y_max)
#         device (torch.device): cuda or cpu
#         original_image (np.ndarray): Full RGB image (for optional plotting)
#         crops_np (List[np.ndarray]): List of RGB crops (unprocessed)
#         plot (bool): Whether to show a plot of the prediction

#     Returns:
#         dict with prediction, confidence, and bbox
#     """

#     start_time = time.time()

#     model.eval()
#     batch = torch.stack(crops).to(device)

#     with torch.no_grad():
#         outputs = model(batch)
#         probs = F.softmax(outputs, dim=1)
#         preds = torch.argmax(probs, dim=1)

#     # Convert to labels and confidence values
#     label_names = {0: "No Fire", 1: "Fire"}
#     pred_labels = [label_names[p.item()] for p in preds]
#     confidence_scores = probs.max(dim=1).values.cpu().numpy().tolist()

#     # Voting logic
#     fire_votes = (preds == 1).sum().item()
#     vote_ratio = fire_votes / len(crops)
#     final_class = 1 if vote_ratio > 0.6 else 0
#     final_label = label_names[final_class]
#     avg_conf = probs[:, final_class].mean().item()

#     end_time = time.time()
#     elapsed = end_time - start_time
#     print(f"Inference time: {elapsed:.4f} seconds")

#     # Optional plot
#     if plot and original_image is not None and crops_np is not None:
#         plot_crops_with_predictions(
#             original_image,
#             crops_np,
#             pred_labels,
#             confidence_scores,
#             final_label,
#             avg_conf,
#             bbox=bbox
#         )

#     return {
#         "final_prediction": final_label,
#         "confidence": avg_conf,
#         "bbox": bbox
#     }



# # Input / Parameters:
# bbox = # Fire BBox in world coordinates
# image1 =
# IR2RGB_ratio =
# HFOV = # IR Fov in Phase1
# required_fov2 =
# ratio_patch =



# === 1. Load image (OpenCV loads as BGR, so convert to RGB)



# Load and convert to RGB

image_path = r"C:/Projects/Flame2/Datasets_FromDvir/Datasets/rgb_images/00039-fire-rgb-flame3.JPG"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Step 1: Resize image to 600x600
resized = cv2.resize(image_rgb, (600, 600), interpolation=cv2.INTER_LINEAR)

# Step 2: Create a black canvas (1080x1920)
image1 = np.zeros((1080, 1920, 3), dtype=np.uint8)

# Step 3: Compute top-left corner to paste the resized image in the center
y_offset = (1080 - 600) // 2
x_offset = (1920 - 600) // 2

# Step 4: Paste resized image into the center of the canvas
image1[y_offset:y_offset+600, x_offset:x_offset+600] = resized


# === 2. Check shape and dtype
print("Image shape:", image1.shape)
print("Dtype:", image1.dtype)        # should be uint8

# Using transformation function to convert World coordinates to RGB image coordinates
tt0 = time.time()

# === 3. Define bbox
bbox_pixels = (960-140, 540-140, 960+140, 540+140)  # example bounding box

# === 4. Define crop factors and transformation
crop_factors = [1.5**0.5, 2**0.5, 2]
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
print(f"\n=== Inference Timing for Cropping === {total_time*1000:.2f} msec\n")

tt1 = time.time()

# Load checkpoint
checkpoint = torch.load("resnet_fire_classifier.pt", map_location=device)

# Define model and load state
num_classes = 2
resnet = models.resnet18(pretrained=False)
num_ftrs = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_ftrs, num_classes)
resnet.load_state_dict(checkpoint["model_state"])
resnet = resnet.to(device)
resnet.eval()

total_time = time.time() - tt1
print(f"\n=== Inference Timing For Loading the Model === {total_time*1000:.2f} msec\n")

i = 0
while i <= 10:
    result = predict_crops_majority_vote(
        test_tensors,
        resnet,
        bbox_pixels,
        device,
        original_image=image1,
        crops_np=cropped_images_np,
        plot=False
    )

    i = i + 1


print("Final Prediction:", result["final_prediction"])
print("Confidence:", f"{result['confidence']:.2f}")
print("BBox:", result["bbox"])