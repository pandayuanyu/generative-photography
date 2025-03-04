import numpy as np
import os
from PIL import Image, ImageSequence
import cv2


# Load GIF and extract frames
def extract_frames(gif_path):
    gif = Image.open(gif_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    return frames


# Calculate the blur degree of each frame (using Laplacian operator)
def calculate_blur_degree(frame):
    frame = np.array(frame.convert('L'))  # Convert to grayscale
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)  # Apply Laplacian filter
    blur_degree = laplacian.var()  # The variance of Laplacian is a good indicator of sharpness
    return blur_degree


# Calculate the blur degree changes between frames
def calculate_blur_changes(frames):
    blur_values = [calculate_blur_degree(frame) for frame in frames]
    blur_changes = np.diff(blur_values)  # Calculate the change between consecutive frames
    return blur_changes


# Compare the similarity of blur degree change trends between two GIFs
def compare_blur_changes(gif1_path, gif2_path):
    frames1 = extract_frames(gif1_path)
    frames2 = extract_frames(gif2_path)

    # Calculate blur degree change trends for each GIF
    blur_changes1 = calculate_blur_changes(frames1)
    blur_changes2 = calculate_blur_changes(frames2)

    # Compute Pearson correlation coefficient
    correlation = np.corrcoef(blur_changes1, blur_changes2)[0, 1]

    return correlation


# Calculate the average similarity of blur degree change trends for all GIF pairs in a folder
def calculate_average_blur_similarity(folder_path):
    gif_pairs = [f for f in os.listdir(folder_path) if f.endswith('.gif')]

    # Assign reference and sample GIF pairs
    reference_gifs = sorted([f for f in gif_pairs if 'reference' in f])
    sample_gifs = sorted([f for f in gif_pairs if 'sample' in f])

    # Ensure the number of reference and sample GIFs is the same
    if len(reference_gifs) != len(sample_gifs):
        raise ValueError("Number of reference and sample GIFs must be the same.")

    total_similarity = 0
    for reference, sample in zip(reference_gifs, sample_gifs):
        reference_path = os.path.join(folder_path, reference)
        sample_path = os.path.join(folder_path, sample)

        # Compare the blur degree change similarity for each GIF pair
        similarity = compare_blur_changes(reference_path, sample_path)
        total_similarity += similarity
        print(f"Blur change similarity for {reference} and {sample}: {similarity}")

    # Compute average similarity
    average_similarity = total_similarity / len(reference_gifs)
    return average_similarity


# Input folder path
folder_path = '//output/samples/sample-27099/'  # Replace with your folder path

# Compute and output the average blur change similarity
average_similarity = calculate_average_blur_similarity(folder_path)
print(f"Average blur change similarity: {average_similarity}")
