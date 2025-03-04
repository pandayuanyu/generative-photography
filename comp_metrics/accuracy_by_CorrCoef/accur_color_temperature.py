import numpy as np
import os
from PIL import Image, ImageSequence


# Load the GIF and extract frames
def extract_frames(gif_path):
    gif = Image.open(gif_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    return frames


# Calculate the average color of each frame (AWB)
def calculate_average_color(frame):
    frame = np.array(frame.convert('RGB'))  # Convert to RGB
    return np.mean(frame, axis=(0, 1))  # Calculate average color for each channel (R, G, B)


# Calculate the color changes for each frame
def calculate_color_changes(frames):
    color_values = [calculate_average_color(frame) for frame in frames]
    color_changes = np.diff(color_values, axis=0)  # Calculate the change between consecutive frames for each channel
    return color_changes


# Calculate the similarity of color change trends between two GIFs
def compare_color_changes(gif1_path, gif2_path):
    frames1 = extract_frames(gif1_path)
    frames2 = extract_frames(gif2_path)

    # Calculate the color change trends for each GIF
    color_changes1 = calculate_color_changes(frames1)
    color_changes2 = calculate_color_changes(frames2)

    # Calculate the Pearson correlation coefficient for each channel (RGB)
    correlation_r = np.corrcoef(color_changes1[:, 0], color_changes2[:, 0])[0, 1]  # Red channel
    correlation_g = np.corrcoef(color_changes1[:, 1], color_changes2[:, 1])[0, 1]  # Green channel
    correlation_b = np.corrcoef(color_changes1[:, 2], color_changes2[:, 2])[0, 1]  # Blue channel

    # Return the average correlation coefficient for all three channels
    average_correlation = (correlation_r + correlation_g + correlation_b) / 3

    return average_correlation


# Calculate the color change trend similarity for all GIF pairs in the folder
def calculate_average_color_similarity(folder_path):
    gif_pairs = [f for f in os.listdir(folder_path) if f.endswith('.gif')]

    # Assign reference and sample GIF pairs
    reference_gifs = sorted([f for f in gif_pairs if 'reference' in f])
    sample_gifs = sorted([f for f in gif_pairs if 'sample' in f])

    # Ensure the number of reference and sample GIFs are the same
    if len(reference_gifs) != len(sample_gifs):
        raise ValueError("Number of reference and sample GIFs must be the same.")

    total_similarity = 0
    for reference, sample in zip(reference_gifs, sample_gifs):
        reference_path = os.path.join(folder_path, reference)
        sample_path = os.path.join(folder_path, sample)

        # Compare the color change trend similarity for each GIF pair
        similarity = compare_color_changes(reference_path, sample_path)
        total_similarity += similarity
        print(f"Color change similarity for {reference} and {sample}: {similarity}")

    # Calculate the average similarity
    average_similarity = total_similarity / len(reference_gifs)
    return average_similarity


# Input folder path
folder_path = '/output/samples/sample-41009/'  # Replace with your folder path

# Calculate and output the average color change trend similarity
average_similarity = calculate_average_color_similarity(folder_path)
print(f"Average color change similarity: {average_similarity}")
