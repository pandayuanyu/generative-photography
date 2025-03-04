import numpy as np
import os
from PIL import Image, ImageSequence


# Load the GIF and extract frames
def extract_frames(gif_path):
    gif = Image.open(gif_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    return frames


# Calculate the average brightness of each frame
def calculate_average_brightness(frame):
    frame = np.array(frame.convert('L'))  # Convert to grayscale
    return np.mean(frame)  # Calculate average brightness


# Calculate the brightness changes for each frame
def calculate_brightness_changes(frames):
    brightness_values = [calculate_average_brightness(frame) for frame in frames]
    brightness_changes = np.diff(brightness_values)  # Calculate the change between consecutive frames
    return brightness_changes


# Calculate the similarity of brightness changes between two GIFs
def compare_brightness_changes(gif1_path, gif2_path):
    frames1 = extract_frames(gif1_path)
    frames2 = extract_frames(gif2_path)

    # Calculate the brightness change trends for each GIF
    brightness_changes1 = calculate_brightness_changes(frames1)
    brightness_changes2 = calculate_brightness_changes(frames2)

    # Calculate the Pearson correlation coefficient
    correlation = np.corrcoef(brightness_changes1, brightness_changes2)[0, 1]

    return correlation


# Calculate the average similarity of brightness changes for all GIF pairs in the folder
def calculate_average_similarity(folder_path):
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

        # Compare the brightness change similarity for each GIF pair
        similarity = compare_brightness_changes(reference_path, sample_path)
        total_similarity += similarity
        print(f"Similarity for {reference} and {sample}: {similarity}")

    # Calculate the average similarity
    average_similarity = total_similarity / len(reference_gifs)
    return average_similarity


# Input folder path
folder_path = '/output/samples/sample-74009/'

# Calculate and output the average similarity
average_similarity = calculate_average_similarity(folder_path)
print(f"Average brightness change similarity: {average_similarity}")
