import subprocess
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torchvision import models

def extract_subframes(image, subframe_res):
    subframe_height, subframe_width = subframe_res
    height, width, _ = image.shape
    num_subframes_y = height // subframe_height
    num_subframes_x = width // subframe_width
    subframes = []
    for y in range(num_subframes_y):
        for x in range(num_subframes_x):
            subframe = image[
                y * subframe_height : (y + 1) * subframe_height,
                x * subframe_width : (x + 1) * subframe_width,
                :
            ]
            subframes.append(subframe)
    return subframes

def convert_frames_to_video(frames_path, video_path, fps=12):
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite without asking
        '-framerate', str(fps),
        '-i', os.path.join(frames_path, 'tinted_frame_%04d.jpg'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        video_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)

def predict_subframes(model, subframes):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # typical size for ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    with torch.no_grad():
        for idx, subframe in enumerate(subframes):
            # Convert NumPy array to PIL Image
            image = Image.fromarray(subframe)

            # Preprocess and predict
            input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

            predictions.append({
                'index': idx,
                'predicted_class': predicted_class,
                'probabilities': probabilities.cpu().numpy()[0]
            })

    return predictions

def phase0(frame_generator, agg_frames, init_values):
    for i_frame, frame in enumerate(frame_generator):
        init_values = agg_frames(init_values, frame, i_frame)
        yield init_values

def phase1(frame_generator, init_values, map_to_cells):
    for frame in frame_generator:
        pass

def phase2(model, frame_generator, save_path, subframe_res=(120, 120), make_video=True):
    for frame_index, frame in enumerate(frame_generator):
        subframes = extract_subframes(frame, subframe_res)
        preds = predict_subframes(model, subframes)

        # Create a tinted copy of the original frame
        tinted_frame = frame.copy().astype(np.float32)

        subframe_height, subframe_width = subframe_res
        height, width, _ = frame.shape
        num_subframes_y = height // subframe_height
        num_subframes_x = width // subframe_width

        for pred in preds:
            idx = pred['index']
            prob_fire = pow(pred['probabilities'][0], 4)

            y = idx // num_subframes_x
            x = idx % num_subframes_x

            # Extract region
            region = tinted_frame[
                y * subframe_height : (y + 1) * subframe_height,
                x * subframe_width : (x + 1) * subframe_width
            ]

            # Apply red tint (blend with red)
            red_overlay = np.array([255, 0, 0], dtype=np.float32)
            region[:] = (1 - 0.5 * prob_fire) * region + (0.5 * prob_fire) * red_overlay

        # Clip values to valid range and convert to uint8
        tinted_frame = np.clip(tinted_frame, 0, 255).astype(np.uint8)
        result_image = Image.fromarray(tinted_frame)

        # Save image
        result_image.save(os.path.join(save_path, f'tinted_frame_{frame_index:04d}.jpg'))
    if make_video:
        convert_frames_to_video(".\\tmp", ".\\output.mp4")




def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reconstruct model architecture exactly as during training
    num_classes = 4
    resnet = models.resnet18(pretrained=False)  # pretrained=False because you'll load your own weights
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    resnet = resnet.to(device)

    # Load trained weights
    checkpoint = torch.load('resnet_fire_classifier.pt', map_location=device)

    # Checkpoint might be the state_dict directly or contain a key like 'model_state_dict'
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        resnet.load_state_dict(checkpoint['model_state'])
    else:
        resnet.load_state_dict(checkpoint)

    resnet.eval()  # Put the model in evaluation mode
    return resnet

def create_frame_generator(frames_path):
    # Get sorted list of image files
    frame_files = sorted([
        f for f in os.listdir(frames_path)
        if os.path.isfile(os.path.join(frames_path, f)) and
           f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    # Yield frames one by one as numpy arrays
    for frame_file in frame_files:
        frame = Image.open(os.path.join(frames_path, frame_file)).convert('RGB')
        frame_np = np.array(frame)  # shape: (height, width, 3)
        yield frame_np


if __name__ == "__main__":
    FRAMES_PATH = 'C:\\Users\\User\\OneDrive - Technion\\FireMan\\Fire\\dji_video_011\\dji_video_011_rgb'
    model = load_model()
    frames_generator = create_frame_generator(FRAMES_PATH)
    phase2(model, frames_generator, ".\\tmp")