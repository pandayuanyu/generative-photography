import os
import cv2
import numpy as np
from PIL import Image, ImageSequence

# Extract frames from a GIF
def extract_frames(gif_path):
    gif = Image.open(gif_path)
    frames = [frame.convert('RGB') for frame in ImageSequence.Iterator(gif)]
    return frames

# Calculate the zoom scale between two consecutive frames
def calculate_zoom_scale(frame1, frame2):
    # Convert to grayscale
    gray1 = cv2.cvtColor(np.array(frame1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(frame2), cv2.COLOR_RGB2GRAY)
    
    # Use SIFT to detect keypoints
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # Feature matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Compute similarity transformation matrix
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    
    # Extract zoom scale from the transformation matrix
    if M is not None:
        scale_x = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
        scale_y = np.sqrt(M[1, 0] ** 2 + M[1, 1] ** 2)
        scale = (scale_x + scale_y) / 2
    else:
        scale = None

    return scale

# Calculate the zoom scale similarity between two GIFs
def calculate_zoom_similarity(gif1_path, gif2_path):
    # Extract frames from both GIFs
    frames1 = extract_frames(gif1_path)
    frames2 = extract_frames(gif2_path)

    # Ensure both GIFs have the same number of frames
    if len(frames1) != len(frames2):
        print("Error: The two GIFs have different numbers of frames.")
        return None

    # Compute zoom scales for each GIF
    zoom_scales1 = []
    zoom_scales2 = []
    
    for i in range(len(frames1) - 1):
        scale1 = calculate_zoom_scale(frames1[i], frames1[i + 1])
        scale2 = calculate_zoom_scale(frames2[i], frames2[i + 1])
        
        zoom_scales1.append(scale1)
        zoom_scales2.append(scale2)
    
    # Compute similarity between zoom scales (Pearson correlation coefficient)
    correlation = np.corrcoef(zoom_scales1, zoom_scales2)[0, 1]
    return correlation

# Compute zoom scale similarity for all GIF pairs in a folder and calculate the average
def calculate_average_zoom_similarity(folder_path):
    total_similarity = 0
    count = 0
    
    # Iterate through all GIF files in the folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith("_sample.gif"):
            # Find the corresponding _reference.gif file
            reference_filename = filename.replace("_sample.gif", "_reference.gif")
            reference_path = os.path.join(folder_path, reference_filename)
            sample_path = os.path.join(folder_path, filename)
            
            # If the corresponding _reference.gif file exists
            if os.path.exists(reference_path):
                similarity = calculate_zoom_similarity(reference_path, sample_path)
                if similarity is not None:
                    total_similarity += similarity
                    count += 1
                    print(f"Zoom scale similarity for {reference_filename} and {filename}: {similarity}")
    
    # Compute the average similarity
    if count > 0:
        average_similarity = total_similarity / count
        print(f"Average zoom scale similarity for all GIF pairs: {average_similarity}")
        return average_similarity
    else:
        print("No valid pairs found.")
        return None

# Example usage
folder_path = "/output/samples/sample-63109/"  # Provide your folder path
average_similarity = calculate_average_zoom_similarity(folder_path)

