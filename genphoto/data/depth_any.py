

from transformers import pipeline
from PIL import Image
import requests
import os

model_dir = ""  
os.makedirs(model_dir, exist_ok=True)

pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    cache_dir=model_dir
)

image = Image.open(".jpg")

# inference
depth = pipe(image)["depth"]



depth_image_path = ""  
depth.save(depth_image_path)


print(f"Depth map saved at: {depth_image_path}")

