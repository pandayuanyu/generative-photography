import os
import json
import torch
import clip
from PIL import Image, ImageSequence
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Parse the JSON file
def load_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

# Load frames from GIF and preprocess
def load_gif_frames(gif_path):
    gif = Image.open(gif_path)
    frames = [preprocess(frame.convert("RGB")).unsqueeze(0) for frame in ImageSequence.Iterator(gif)]


# Calculate the similarity between image and text features using cosine similarity
def compute_similarity(image_features, text_features):
    similarities = F.cosine_similarity(image_features, text_features, dim=1)
    return similarities.squeeze()


# Extract frames from GIF and compute the similarity with the given prompt text
def extract_frames_and_similarity(gif_path, prompt_text):
    gif = Image.open(gif_path)
    
    # Use ImageSequence to get all frames
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]

    # Get the text features
    text_input = clip.tokenize([prompt_text]).to(device)
    text_features = model.encode_text(text_input)

    # Store similarity for each frame
    frame_similarities = []

    for frame in frames:
        # Image preprocessing
        image_input = preprocess(frame).unsqueeze(0).to(device)
        image_features = model.encode_image(image_input)

        # Compute cosine similarity
        similarity = compute_similarity(image_features, text_features)
        frame_similarities.append(similarity.item())  # Store similarity for each frame

    # Return the frame similarities and the average similarity
    avg_similarity = np.mean(frame_similarities)
    return frame_similarities, avg_similarity


# Process all GIFs in a folder and compute their average similarity to the corresponding prompt
def compute_avg_similarity_for_folder(gif_folder, json_file_path):
    # Read the JSON data file
    with open(json_file_path, "r") as f:
        json_data = json.load(f)

    avg_similarities = []

    for idx, item in enumerate(json_data):
        gif_path = os.path.join(gif_folder, f"{idx}_sample.gif")  # Assume GIF filenames are indexed
        prompt_text = item["caption"]

        # Compute the similarity for the current GIF and prompt
        frame_similarities, avg_similarity = extract_frames_and_similarity(gif_path, prompt_text)

        # Print similarity for each GIF and prompt pair
        print(f"GIF {idx} - Similarities: {frame_similarities}")
        print(f"GIF {idx} - Average Similarity: {avg_similarity}")
        
        avg_similarities.append(avg_similarity)

    # Compute the overall average similarity for all GIFs
    overall_avg_similarity = np.mean(avg_similarities)
    print(f"Overall Average Similarity: {overall_avg_similarity}")
    return overall_avg_similarity

# Input folder path and JSON data file path
gif_folder = '/output/samples/sample-27199/'  # GIF folder path
json_file_path = '/camera_dataset/camera_bokehK/annotations/validation.json'  # JSON file path

# Calculate the average similarity between all GIFs and prompts
overall_avg_similarity = compute_avg_similarity_for_folder(gif_folder, json_file_path)
