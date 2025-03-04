import os
from PIL import Image
import lpips
import torch
import numpy as np

# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='vgg')


def load_gif_frames(gif_path):
    """Load a GIF image file as a set of frames."""
    with Image.open(gif_path) as img:
        frames = []
        for frame in range(img.n_frames):
            img.seek(frame)
            frames.append(img.convert('RGB'))
    return frames


def calculate_average_lpips(gif_path):
    """Calculate the average LPIPS score for a given GIF."""
    frames = load_gif_frames(gif_path)
    lpips_scores = []

    for i in range(len(frames) - 1):
        # Convert adjacent frames to tensors
        frame1 = torch.tensor((np.array(frames[i]).astype(np.float32) / 255.0) * 2 - 1).permute(2, 0, 1).unsqueeze(0)
        frame2 = torch.tensor((np.array(frames[i + 1]).astype(np.float32) / 255.0) * 2 - 1).permute(2, 0, 1).unsqueeze(0)

        # Calculate the LPIPS score
        lpips_score = lpips_model(frame1, frame2).item()
        lpips_scores.append(lpips_score)

    # Calculate the average LPIPS score for the GIF
    return sum(lpips_scores) / len(lpips_scores) if lpips_scores else 0


def calculate_folder_average_lpips(folder_path):
    """Calculate the average LPIPS score for all GIFs in a folder."""
    gif_files = [f for f in os.listdir(folder_path) if f.endswith('_sample.gif')]
    all_lpips_scores = []

    for gif_file in gif_files:
        gif_path = os.path.join(folder_path, gif_file)
        avg_lpips = calculate_average_lpips(gif_path)
        all_lpips_scores.append(avg_lpips)
        print(f"Processed {gif_file}, average LPIPS: {avg_lpips}")

    # Calculate the overall average LPIPS score for all GIFs
    overall_average_lpips = sum(all_lpips_scores) / len(all_lpips_scores) if all_lpips_scores else 0
    return overall_average_lpips


folder_path = '/output/samples/sample-27099/'   ### replace with your folder
overall_average_lpips_score = calculate_folder_average_lpips(folder_path)
print(f"Overall average LPIPS score for all GIFs: {overall_average_lpips_score}")
