import os
import random
import json
import torch
import math
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from torch.utils.data.dataset import Dataset
from packaging import version as pver
import cv2
from PIL import Image
from einops import rearrange
from transformers import pipeline, CLIPTextModel, CLIPTokenizer

import sys
sys.path.append('/Generative_Photography/genphoto/data/BokehMe/')
from classical_renderer.scatter import ModuleRenderScatter  



#### for shutter speed ####
def create_shutter_speed_embedding(shutter_speed_values, target_height, target_width, base_exposure=0.5):
    """
    Create an shutter_speed embedding tensor using a constant fwc value.
    Args:
    - shutter_speed_values: Tensor of shape [f, 1] containing shutter_speed values for each frame.
    - H: Height of the image.
    - W: Width of the image.
    - base_exposure: A base exposure value to normalize brightness (defaults to 0.18 as a common base exposure level).

    Returns:
    - shutter_speed_embedding: Tensor of shape [f, 1, H, W] where each pixel is scaled based on the shutter_speed values.
    """
    f = shutter_speed_values.shape[0]

    # Set a constant full well capacity (fwc)
    fwc = 32000  # Constant value for full well capacity

    # Calculate scale based on EV and sensor full well capacity (fwc)
    scales = (shutter_speed_values / base_exposure) * (fwc / (fwc + 0.0001))

    # Reshape and expand to match image dimensions
    scales = scales.unsqueeze(2).unsqueeze(3).expand(f, 3, target_height, target_width)

    # Use scales to create the final shutter_speed embedding
    shutter_speed_embedding = scales      # Shape [f, 3, H, W]
    return shutter_speed_embedding


def sensor_image_simulation_numpy(avg_PPP, photon_flux, fwc, Nbits, gain=1):
    min_val = 0
    max_val = 2 ** Nbits - 1
    theta = photon_flux * (avg_PPP / (np.mean(photon_flux) + 0.0001))
    theta = np.clip(theta, 0, fwc)
    theta = np.round(theta * gain * max_val / fwc)
    theta = np.clip(theta, min_val, max_val)
    theta = theta.astype(np.float32)
    return theta


class CameraShutterSpeed(Dataset):
    def __init__(
            self,
            root_path,
            annotation_json,
            sample_n_frames=5,
            sample_size=[256, 384],
            is_Train=True,
    ):
        self.root_path = root_path
        self.sample_n_frames = sample_n_frames
        self.dataset = json.load(open(os.path.join(root_path, annotation_json), 'r'))
        self.length = len(self.dataset)
        self.is_Train = is_Train
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size

        pixel_transforms = [transforms.Resize(sample_size),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]

        self.pixel_transforms = pixel_transforms
        self.tokenizer = CLIPTokenizer.from_pretrained("/home/yuan418/data/project/stable-diffusion-v1-5/", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("/home/yuan418/data/project/stable-diffusion-v1-5/", subfolder="text_encoder")

    def load_image_reader(self, idx):
        image_dict = self.dataset[idx]
        image_path = os.path.join(self.root_path, image_dict['base_image_path'])
        image_reader = cv2.imread(image_path)
        image_reader = cv2.cvtColor(image_reader, cv2.COLOR_BGR2RGB)
        image_caption = image_dict['caption']
        
        if self.is_Train:
            mean = 0.48
            std_dev = 0.25
            shutter_speed_values = [random.gauss(mean, std_dev) for _ in range(self.sample_n_frames)]
            shutter_speed_values = [max(0.1, min(1.0, ev)) for ev in shutter_speed_values]
            print('train shutter_speed values', shutter_speed_values)
           
        else:
            shutter_speed_list_str = image_dict['shutter_speed_list']
            shutter_speed_values = json.loads(shutter_speed_list_str)
            print('validation shutter_speed_values', shutter_speed_values)

        shutter_speed_values = torch.tensor(shutter_speed_values).unsqueeze(1)
        return image_path, image_reader, image_caption, shutter_speed_values


    def get_batch(self, idx):
        image_path, image_reader, image_caption, shutter_speed_values = self.load_image_reader(idx)

        total_frames = len(shutter_speed_values)
        if total_frames < 3:
            raise ValueError("less than 3 frames")

        # Generate prompts for each shutter speed value and append shutter speed information to caption
        prompts = []
        for ss in shutter_speed_values:
            prompt = f"<exposure: {ss.item()}>"
            prompts.append(prompt)

        # Tokenize prompts and encode to get embeddings
        with torch.no_grad():
            prompt_ids = self.tokenizer(
                prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            # print('tokenizer model_max_length', self.tokenizer.model_max_length)

            encoder_hidden_states = self.text_encoder(input_ids=prompt_ids).last_hidden_state  # Shape: (f, sequence_length, hidden_size)
        
        # print('encoder_hidden_states shape', encoder_hidden_states.shape)

        # Calculate differences between consecutive embeddings (ignoring sequence_length)
        differences = []
        for i in range(1, encoder_hidden_states.size(0)):
            diff = encoder_hidden_states[i] - encoder_hidden_states[i - 1]
            diff = diff.unsqueeze(0)
            differences.append(diff)  

        # Add the difference between the last and the first embedding
        final_diff = encoder_hidden_states[-1] - encoder_hidden_states[0]
        final_diff = final_diff.unsqueeze(0)
        differences.append(final_diff)

        # Concatenate differences along the batch dimension (f-1)
        concatenated_differences = torch.cat(differences, dim=0) 
        # print('concatenated_differences shape', concatenated_differences.shape) # f 77 768

        frame = concatenated_differences.size(0)

        concatenated_differences = torch.cat(differences, dim=0)

        # Current shape: (f, 77, 768) Pad the second dimension (77) to 128
        pad_length = 128 - concatenated_differences.size(1)
        if pad_length > 0:
        # Pad along the second dimension (77 -> 128), pad only on the right side
            concatenated_differences_padded = F.pad(concatenated_differences, (0, 0, 0, pad_length))

        ## ccl = constrative camera learning
        ccl_embedding = concatenated_differences_padded.reshape(frame, self.sample_size[0], self.sample_size[1])
        ccl_embedding = ccl_embedding.unsqueeze(1)  
        ccl_embedding = ccl_embedding.expand(-1, 3, -1, -1)

        # Now handle the sensor image simulation
        fwc = random.uniform(19000, 64000)
        pixel_values = []
        for ee in shutter_speed_values:
            avg_PPP = (0.6 * ee.item() + 0.1) * fwc
            img_sim = sensor_image_simulation_numpy(avg_PPP, image_reader, fwc, Nbits=8, gain=1)    
            pixel_values.append(img_sim)
        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous() / 255.

        # Create shutter_speed embedding and concatenate it with CCL embedding
        shutter_speed_embedding = create_shutter_speed_embedding(shutter_speed_values, self.sample_size[0], self.sample_size[1])

        camera_embedding = torch.cat((shutter_speed_embedding, ccl_embedding), dim=1) 
        # print('camera_embedding shape', camera_embedding.shape)

        return pixel_values, image_caption, camera_embedding, shutter_speed_values

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                video, video_caption, camera_embedding, shutter_speed_values = self.get_batch(idx)
                break
            except Exception as e:
                idx = random.randint(0, self.length - 1)

        for transform in self.pixel_transforms:
            video = transform(video)

        sample = dict(pixel_values=video, text=video_caption, camera_embedding=camera_embedding, shutter_speed_values=shutter_speed_values)

        return sample








#### for focal length ####
def crop_focal_length(image_path, base_focal_length, target_focal_length, target_height, target_width, sensor_height=24.0, sensor_width=36.0):
    img = Image.open(image_path)
    width, height = img.size

    # Calculate base and target FOV
    base_x_fov = 2.0 * math.atan(sensor_width * 0.5 / base_focal_length)
    base_y_fov = 2.0 * math.atan(sensor_height * 0.5 / base_focal_length)

    target_x_fov = 2.0 * math.atan(sensor_width * 0.5 / target_focal_length)
    target_y_fov = 2.0 * math.atan(sensor_height * 0.5 / target_focal_length)

    # Calculate crop ratio, use the smaller ratio to maintain aspect ratio
    crop_ratio = min(target_x_fov / base_x_fov, target_y_fov / base_y_fov)

    crop_width = int(round(crop_ratio * width))
    crop_height = int(round(crop_ratio * height))

    # Ensure crop dimensions are within valid bounds
    crop_width = max(1, min(width, crop_width))
    crop_height = max(1, min(height, crop_height))

    # Crop coordinates
    left = int((width - crop_width) / 2)
    top = int((height - crop_height) / 2)
    right = int((width + crop_width) / 2)
    bottom = int((height + crop_height) / 2)

    # Crop the image
    zoomed_img = img.crop((left, top, right, bottom))

    # Resize the cropped image to target resolution
    resized_img = zoomed_img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # Convert the PIL image to a numpy array
    resized_img_np = np.array(resized_img).astype(np.float32)

    return resized_img_np


def create_focal_length_embedding(focal_length_values, base_focal_length, target_height, target_width, sensor_height=24.0, sensor_width=36.0):
    device = 'cpu'
    focal_length_values = focal_length_values.to(device)

    f = focal_length_values.shape[0]  # Number of frames

    # Convert constants to tensors to perform operations with focal_length_values
    sensor_width = torch.tensor(sensor_width, device=device)
    sensor_height = torch.tensor(sensor_height, device=device)
    base_focal_length = torch.tensor(base_focal_length, device=device)

    # Calculate the FOV for the base focal length (min_focal_length)
    base_fov_x = 2.0 * torch.atan(sensor_width * 0.5 / base_focal_length)
    base_fov_y = 2.0 * torch.atan(sensor_height * 0.5 / base_focal_length)

    # Calculate the FOV for each focal length in focal_length_values
    target_fov_x = 2.0 * torch.atan(sensor_width * 0.5 / focal_length_values)
    target_fov_y = 2.0 * torch.atan(sensor_height * 0.5 / focal_length_values)

    # Calculate crop ratio: how much of the image is cropped at the current focal length
    crop_ratio_xs = target_fov_x / base_fov_x  # Crop ratio for horizontal axis
    crop_ratio_ys = target_fov_y / base_fov_y  # Crop ratio for vertical axis

    # Get the center of the image
    center_h, center_w = target_height // 2, target_width // 2

    # Initialize a mask tensor with zeros on CPU
    focal_length_embedding = torch.zeros((f, 3, target_height, target_width), dtype=torch.float32)  # Shape [f, 3, H, W]

    # Fill the center region with 1 based on the calculated crop dimensions
    for i in range(f):
        # Crop dimensions calculated using rounded float values
        crop_h = torch.round(crop_ratio_ys[i] * target_height).int().item()  # Rounded cropped height for the current frame
        crop_w = torch.round(crop_ratio_xs[i] * target_width).int().item()  # Rounded cropped width for the current frame

        # Ensure the cropped dimensions are within valid bounds
        crop_h = max(1, min(target_height, crop_h))
        crop_w = max(1, min(target_width, crop_w))

        # Set the center region of the focal_length embedding to 1 for the current frame
        focal_length_embedding[i, :,
        center_h - crop_h // 2: center_h + crop_h // 2,
        center_w - crop_w // 2: center_w + crop_w // 2] = 1.0

    return focal_length_embedding


class CameraFocalLength(Dataset):
    def __init__(
            self,
            root_path,
            annotation_json,
            sample_n_frames=5,
            sample_size=[256, 384],
            is_Train=True,
    ):
        self.root_path = root_path
        self.sample_n_frames = sample_n_frames
        self.dataset = json.load(open(os.path.join(root_path, annotation_json), 'r'))
        self.length = len(self.dataset)
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size
        pixel_transforms = [transforms.Resize(sample_size),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]

        self.pixel_transforms = pixel_transforms
        self.is_Train = is_Train
        self.tokenizer = CLIPTokenizer.from_pretrained("/home/yuan418/data/project/stable-diffusion-v1-5/", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("/home/yuan418/data/project/stable-diffusion-v1-5/", subfolder="text_encoder")


    def load_image_reader(self, idx):
        image_dict = self.dataset[idx]

        image_path = os.path.join(self.root_path, image_dict['base_image_path'])
        image_reader = cv2.imread(image_path)

        image_caption = image_dict['caption']

        if self.is_Train:
            focal_length_values = [random.uniform(24.0, 70.0) for _ in range(self.sample_n_frames)]
            print('train focal_length_values', focal_length_values)
        else:
            focal_length_list_str = image_dict['focal_length_list']
            focal_length_values = json.loads(focal_length_list_str)
            print('validation focal_length_values', focal_length_values)

        focal_length_values = torch.tensor(focal_length_values).unsqueeze(1)

        return image_path, image_reader, image_caption, focal_length_values


    def get_batch(self, idx):
        image_path, image_reader, image_caption, focal_length_values = self.load_image_reader(idx)

        total_frames = len(focal_length_values)
        if total_frames < 3:
            raise ValueError("less than 3 frames")

        # Generate prompts for each fl value and append fl information to caption
        prompts = []
        for fl in focal_length_values:
            prompt = f"<focal length: {fl.item()}>"
            prompts.append(prompt)
       
        # Tokenize prompts and encode to get embeddings
        with torch.no_grad():
            prompt_ids = self.tokenizer(
                prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids

            encoder_hidden_states = self.text_encoder(input_ids=prompt_ids).last_hidden_state  # Shape: (f, sequence_length, hidden_size)
        # print('encoder_hidden_states shape', encoder_hidden_states.shape)

        # Calculate differences between consecutive embeddings (ignoring sequence_length)
        differences = []
        for i in range(1, encoder_hidden_states.size(0)):
            diff = encoder_hidden_states[i] - encoder_hidden_states[i - 1]
            diff = diff.unsqueeze(0)
            differences.append(diff)  

        # Add the difference between the last and the first embedding
        final_diff = encoder_hidden_states[-1] - encoder_hidden_states[0]
        final_diff = final_diff.unsqueeze(0)
        differences.append(final_diff)

        # Concatenate differences along the batch dimension (f-1)
        concatenated_differences = torch.cat(differences, dim=0) 
        # print('concatenated_differences shape', concatenated_differences.shape) # f 77 768

        frame = concatenated_differences.size(0)

        # Concatenate differences along the batch dimension (f)
        concatenated_differences = torch.cat(differences, dim=0)

        # Current shape: (f, 77, 768), Pad the second dimension (77) to 128
        pad_length = 128 - concatenated_differences.size(1)
        if pad_length > 0:
        # Pad along the second dimension (77 -> 128), pad only on the right side
            concatenated_differences_padded = F.pad(concatenated_differences, (0, 0, 0, pad_length))
        
        ## CCL = constrative camera learning
        ccl_embedding = concatenated_differences_padded.reshape(frame, self.sample_size[0], self.sample_size[1])

        ccl_embedding = ccl_embedding.unsqueeze(1)  
        ccl_embedding = ccl_embedding.expand(-1, 3, -1, -1)
        # print('ccl_embedding shape', ccl_embedding.shape)

        pixel_values = []
        for ff in focal_length_values:
            img_sim = crop_focal_length(image_path=image_path, base_focal_length=24.0, target_focal_length=ff, target_height=self.sample_size[0], target_width=self.sample_size[1], sensor_height=24.0, sensor_width=36.0)
      
            pixel_values.append(img_sim)
            # save_path = os.path.join(self.root_path, f"simulated_img_focal_length_{fl.item():.2f}.png")
            # cv2.imwrite(save_path, img_sim)
            # print(f"Saved image: {save_path}")

        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous() / 255.

        focal_length_embedding = create_focal_length_embedding(focal_length_values, base_focal_length=24.0, target_height=self.sample_size[0], target_width=self.sample_size[1])
        # print('focal_length_embedding shape', focal_length_embedding.shape)

        camera_embedding = torch.cat((focal_length_embedding, ccl_embedding), dim=1) 
        # print('camera_embedding shape', camera_embedding.shape)

        return pixel_values, image_caption, camera_embedding, focal_length_values

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                video, video_caption, camera_embedding, focal_length_values = self.get_batch(idx)
                break
            except Exception as e:
                idx = random.randint(0, self.length - 1)

        for transform in self.pixel_transforms:
            video = transform(video)

        sample = dict(pixel_values=video, text=video_caption, camera_embedding=camera_embedding, focal_length_values=focal_length_values)

        return sample







#### for color temperature  ####
def kelvin_to_rgb(kelvin):
    temp = kelvin / 100.0

    if temp <= 66:
        red = 255
        green = 99.4708025861 * np.log(temp) - 161.1195681661 if temp > 0 else 0
        if temp <= 19:
            blue = 0
        else:
            blue = 138.5177312231 * np.log(temp - 10) - 305.0447927307

    elif 66<temp<=88:
        red = 0.5 * (255 + 329.698727446 * ((temp - 60) ** -0.19332047592))
        green = 0.5 * (288.1221695283 * ((temp - 60) ** -0.1155148492) + (99.4708025861 * np.log(temp) - 161.1195681661 if temp > 0 else 0))
        blue = 0.5 * (138.5177312231 * np.log(temp - 10) - 305.0447927307 + 255)

    else:
        red = 329.698727446 * ((temp - 60) ** -0.19332047592)
        green = 288.1221695283 * ((temp - 60) ** -0.1155148492)
        blue = 255

    return np.array([red, green, blue], dtype=np.float32) / 255.0



def create_color_temperature_embedding(color_temperature_values, target_height, target_width, min_color_temperature=2000, max_color_temperature=10000):
    """
    Create an color_temperature embedding tensor based on color temperature.
    Args:
    - color_temperature_values: Tensor of shape [f, 1] containing color_temperature values for each frame.
    - target_height: Height of the image.
    - target_width: Width of the image.
    - min_color_temperature: Minimum color_temperature value for normalization.
    - max_color_temperature: Maximum color_temperature value for normalization.
    Returns:
    - color_temperature_embedding: Tensor of shape [f, 3, target_height, target_width] for RGB channel scaling.
    """
    f = color_temperature_values.shape[0]
    rgb_factors = []

    # Compute RGB factors based on kelvin_to_rgb function
    for ct in color_temperature_values.squeeze():
        kelvin = min_color_temperature + (ct * (max_color_temperature - min_color_temperature))  # Map normalized color_temperature to actual Kelvin
        rgb = kelvin_to_rgb(kelvin)
        rgb_factors.append(rgb)
    
    # Convert to tensor and expand to target dimensions
    rgb_factors = torch.tensor(rgb_factors).float()  # [f, 3]
    rgb_factors = rgb_factors.unsqueeze(2).unsqueeze(3)  # [f, 3, 1, 1]
    color_temperature_embedding = rgb_factors.expand(f, 3, target_height, target_width)  # [f, 3, target_height, target_width]
    return color_temperature_embedding



def kelvin_to_rgb_smooth(kelvin):
    temp = kelvin / 100.0

    if temp <= 66:
        red = 255
        green = 99.4708025861 * np.log(temp) - 161.1195681661 if temp > 0 else 0
        if temp <= 19:
            blue = 0
        else:
            blue = 138.5177312231 * np.log(temp - 10) - 305.0447927307

    elif 66<temp<=88:
        red = 0.5 * (255 + 329.698727446 * ((temp - 60) ** -0.19332047592))
        green = 0.5 * (288.1221695283 * ((temp - 60) ** -0.1155148492) + (99.4708025861 * np.log(temp) - 161.1195681661 if temp > 0 else 0))
        blue = 0.5 * (138.5177312231 * np.log(temp - 10) - 305.0447927307 + 255)

    else:
        red = 329.698727446 * ((temp - 60) ** -0.19332047592)
        green = 288.1221695283 * ((temp - 60) ** -0.1155148492)
        blue = 255

    red = np.clip(red, 0, 255)
    green = np.clip(green, 0, 255)
    blue = np.clip(blue, 0, 255)
    balance_rgb = np.array([red, green, blue], dtype=np.float32)

    return balance_rgb


def interpolate_white_balance(image, kelvin):

    balance_rgb = kelvin_to_rgb_smooth(kelvin.item())
    image = image.astype(np.float32)

    r, g, b = cv2.split(image)
    r = r * (balance_rgb[0] / 255.0)
    g = g * (balance_rgb[1] / 255.0)
    b = b * (balance_rgb[2] / 255.0)

    balanced_image = cv2.merge([r,g,b])
    balanced_image = np.clip(balanced_image, 0, 255).astype(np.uint8)
    
    return balanced_image


class CameraColorTemperature(Dataset):
    def __init__(
            self,
            root_path,
            annotation_json,
            sample_n_frames=5,
            sample_size=[256, 384],
            is_Train=True,
    ):
        self.root_path = root_path
        self.sample_n_frames = sample_n_frames
        self.dataset = json.load(open(os.path.join(root_path, annotation_json), 'r'))

        self.length = len(self.dataset)
        self.is_Train = is_Train

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size

        pixel_transforms = [transforms.Resize(sample_size),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]

        self.pixel_transforms = pixel_transforms
        self.tokenizer = CLIPTokenizer.from_pretrained("/home/yuan418/data/project/stable-diffusion-v1-5/", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("/home/yuan418/data/project/stable-diffusion-v1-5/", subfolder="text_encoder")

    def load_image_reader(self, idx):
        image_dict = self.dataset[idx]

        image_path = os.path.join(self.root_path, image_dict['base_image_path'])
        image_reader = cv2.imread(image_path)
        image_reader = cv2.cvtColor(image_reader, cv2.COLOR_BGR2RGB)

        image_caption = image_dict['caption']
        
        if self.is_Train:
            color_temperature_values = [random.uniform(2000.0, 10000.0) for _ in range(self.sample_n_frames)]
            print('train color_temperature values', color_temperature_values)
           
        else:
            color_temperature_list_str = image_dict['color_temperature_list']
            color_temperature_values = json.loads(color_temperature_list_str)
            print('validation color_temperature_values', color_temperature_values)

        color_temperature_values = torch.tensor(color_temperature_values).unsqueeze(1)
        return image_path, image_reader, image_caption, color_temperature_values


    def get_batch(self, idx):
        image_path, image_reader, image_caption, color_temperature_values = self.load_image_reader(idx)

        total_frames = len(color_temperature_values)
        if total_frames < 3:
            raise ValueError("less than 3 frames")

        # Generate prompts for each color_temperature value and append color_temperature information to caption
        prompts = []
        for cc in color_temperature_values:
            prompt = f"<color temperature: {cc.item()}>"
            prompts.append(prompt)

        # Tokenize prompts and encode to get embeddings
        with torch.no_grad():
            prompt_ids = self.tokenizer(
                prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            # print('tokenizer model_max_length', self.tokenizer.model_max_length)

            encoder_hidden_states = self.text_encoder(input_ids=prompt_ids).last_hidden_state  # Shape: (f, sequence_length, hidden_size)
        
        # print('encoder_hidden_states shape', encoder_hidden_states.shape)

        # Calculate differences between consecutive embeddings (ignoring sequence_length)
        differences = []
        for i in range(1, encoder_hidden_states.size(0)):
            diff = encoder_hidden_states[i] - encoder_hidden_states[i - 1]
            diff = diff.unsqueeze(0)
            differences.append(diff)  

        # Add the difference between the last and the first embedding
        final_diff = encoder_hidden_states[-1] - encoder_hidden_states[0]
        final_diff = final_diff.unsqueeze(0)
        differences.append(final_diff)

        # Concatenate differences along the batch dimension (f-1)
        concatenated_differences = torch.cat(differences, dim=0)
        # print('concatenated_differences shape', concatenated_differences.shape) # f 77 768

        frame = concatenated_differences.size(0)

        concatenated_differences = torch.cat(differences, dim=0)

        # Current shape: (f, 77, 768), Pad the second dimension (77) to 128
        pad_length = 128 - concatenated_differences.size(1)
        if pad_length > 0:
        # Pad along the second dimension (77 -> 128), pad only on the right side
            concatenated_differences_padded = F.pad(concatenated_differences, (0, 0, 0, pad_length))

        ccl_embedding = concatenated_differences_padded.reshape(frame, self.sample_size[0], self.sample_size[1])
        ccl_embedding = ccl_embedding.unsqueeze(1)  
        ccl_embedding = ccl_embedding.expand(-1, 3, -1, -1)
        # print('ccl_embedding shape', ccl_embedding.shape)

        # Now handle the sensor image simulation
        pixel_values = []
        for aw in color_temperature_values:
            img_sim = interpolate_white_balance(image_reader, aw)    
            pixel_values.append(img_sim)
        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous() / 255.

        # Create color_temperature embedding and concatenate it with CCL embedding
        color_temperature_embedding = create_color_temperature_embedding(color_temperature_values, self.sample_size[0], self.sample_size[1])
        # print('color_temperature_embedding shape', color_temperature_embedding.shape)

        camera_embedding = torch.cat((color_temperature_embedding, ccl_embedding), dim=1) 
        # print('camera_embedding shape', camera_embedding.shape)

        return pixel_values, image_caption, camera_embedding, color_temperature_values

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                video, video_caption, camera_embedding, color_temperature_values = self.get_batch(idx)
                break
            except Exception as e:
                idx = random.randint(0, self.length - 1)

        for transform in self.pixel_transforms:
            video = transform(video)

        sample = dict(pixel_values=video, text=video_caption, camera_embedding=camera_embedding, color_temperature_values=color_temperature_values)

        return sample








#### for bokeh (K is the blur parameter) ####
def create_bokehK_embedding(bokehK_values, target_height, target_width):
    """
    Creates a Bokeh embedding based on the given K values. The larger the K value,
    the more the image is blurred.

    Args:
        bokehK_values (torch.Tensor): Tensor of K values for bokeh effect.
        target_height (int): Desired height of the output embedding.
        target_width (int): Desired width of the output embedding.
        base_K (float): Base K value to control the minimum blur level.

    Returns:
        torch.Tensor: Bokeh embedding tensor. [f 3 h w]
    """
    f = bokehK_values.shape[0]
    bokehK_embedding = torch.zeros((f, 3, target_height, target_width), dtype=bokehK_values.dtype)

    for i in range(f):
        K_value = bokehK_values[i].item() 

        kernel_size = max(K_value, 1)  
        sigma = K_value / 3.0  

        ax = np.linspace(-(kernel_size / 2), kernel_size / 2, int(np.ceil(kernel_size)))
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= np.sum(kernel)  

        scale = kernel[int(np.ceil(kernel_size) / 2), int(np.ceil(kernel_size) / 2)]
        bokehK_embedding[i] = scale

    return bokehK_embedding


def bokehK_simulation(image_path, depth_map_path, K, disp_focus, gamma=2.2):
    ## depth map image can be inferenced online using following code ##
    #  model_dir = "/home/modules/" 
    #  pipe = pipeline(
    #            task="depth-estimation",
    #           model="depth-anything/Depth-Anything-V2-Small-hf",
    #          cache_dir=model_dir,
    #            device=0
    #         )

    # image_raw = Image.open(image_path)

    # disp = pipe(image_raw)["depth"]
    # base_name = os.path.basename(image_path) 
    # file_name, ext = os.path.splitext(base_name)  

    # disp_file_name = f"{file_name}_disp.png"
    # disp.save(disp_file_name)
    
    # disp = np.array(disp)
    # disp = disp.astype(np.float32)
    # disp /= 255.0

    disp = np.float32(cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE))

    disp /= 255.0
    disp = (disp - disp.min()) / (disp.max() - disp.min())
    min_disp = np.min(disp)
    max_disp = np.max(disp)

    device = torch.device('cuda')
  
    # Initialize renderer
    classical_renderer = ModuleRenderScatter().to(device)

    # Load image and disparity
    image = cv2.imread(image_path).astype(np.float32) / 255.0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
    # Calculate defocus
    defocus = K * (disp - disp_focus) / 10.0

    # Convert to tensors and move to GPU if available
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)

    defocus = defocus.unsqueeze(0).unsqueeze(0).to(device)   
   
    bokeh_classical, defocus_dilate = classical_renderer(image**gamma, defocus*10.0)
    bokeh_pred = bokeh_classical ** (1/gamma)
    bokeh_pred = bokeh_pred.squeeze(0)
    bokeh_pred = bokeh_pred.permute(1, 2, 0)  # remove batch dim and change channle order
    bokeh_pred = (bokeh_pred * 255).cpu().numpy()
    bokeh_pred = np.round(bokeh_pred)
    bokeh_pred = bokeh_pred.astype(np.float32)

    return bokeh_pred




class CameraBokehK(Dataset):
    def __init__(
            self,
            root_path,
            annotation_json,
            sample_n_frames=5,
            sample_size=[256, 384],
            is_Train=True,
    ):
        self.root_path = root_path
        self.sample_n_frames = sample_n_frames
        self.dataset = json.load(open(os.path.join(root_path, annotation_json), 'r'))

        self.length = len(self.dataset)
        self.is_Train = is_Train
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size

        pixel_transforms = [transforms.Resize(sample_size),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]

        self.pixel_transforms = pixel_transforms
        self.tokenizer = CLIPTokenizer.from_pretrained("/home/yuan418/data/project/stable-diffusion-v1-5/", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("/home/yuan418/data/project/stable-diffusion-v1-5/", subfolder="text_encoder")

    def load_image_reader(self, idx):
        image_dict = self.dataset[idx]

        image_path = os.path.join(self.root_path, image_dict['base_image_path'])
        depth_map_path = os.path.join(self.root_path, image_dict['depth_map_path'])

        image_caption = image_dict['caption']
     
        
        if self.is_Train:
            bokehK_values = [random.uniform(1.0, 30.0) for _ in range(self.sample_n_frames)]
            print('train bokehK values', bokehK_values)
           
        else:
            bokehK_list_str = image_dict['bokehK_list']  
            bokehK_values = json.loads(bokehK_list_str)
            print('validation bokehK_values', bokehK_values)

        bokehK_values = torch.tensor(bokehK_values).unsqueeze(1)
        return image_path, depth_map_path, image_caption, bokehK_values


    def get_batch(self, idx):
        image_path, depth_map_path, image_caption, bokehK_values = self.load_image_reader(idx)

        total_frames = len(bokehK_values)
        if total_frames < 3:
            raise ValueError("less than 3 frames")

        # Generate prompts for each bokehK value and append bokehK information to caption
        prompts = []
        for bb in bokehK_values:
            prompt = f"<bokeh kernel size: {bb.item()}>"
            prompts.append(prompt)

        # Tokenize prompts and encode to get embeddings
        with torch.no_grad():
            prompt_ids = self.tokenizer(
                prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
        # print('tokenizer model_max_length', self.tokenizer.model_max_length)

            encoder_hidden_states = self.text_encoder(input_ids=prompt_ids).last_hidden_state  # Shape: (f, sequence_length, hidden_size)
        
        # print('encoder_hidden_states shape', encoder_hidden_states.shape)

        # Calculate differences between consecutive embeddings (ignoring sequence_length)
        differences = []
        for i in range(1, encoder_hidden_states.size(0)):
            diff = encoder_hidden_states[i] - encoder_hidden_states[i - 1]
            diff = diff.unsqueeze(0)
            differences.append(diff)  

        # Add the difference between the last and the first embedding
        final_diff = encoder_hidden_states[-1] - encoder_hidden_states[0]
        final_diff = final_diff.unsqueeze(0)
        differences.append(final_diff)

        # Concatenate differences along the batch dimension (f-1)
        concatenated_differences = torch.cat(differences, dim=0) 

        # print('concatenated_differences shape', concatenated_differences.shape) # f 77 768

        frame = concatenated_differences.size(0)

        # Concatenate differences along the batch dimension (f)
        concatenated_differences = torch.cat(differences, dim=0)

        # Current shape: (f, 77, 768), Pad the second dimension (77) to 128
        pad_length = 128 - concatenated_differences.size(1)
        if pad_length > 0:
        # Pad along the second dimension (77 -> 128), pad only on the right side
            concatenated_differences_padded = F.pad(concatenated_differences, (0, 0, 0, pad_length))

        ## ccl = contrastive camera learning ##
        ccl_embedding = concatenated_differences_padded.reshape(frame, self.sample_size[0], self.sample_size[1])  
        ccl_embedding = ccl_embedding.unsqueeze(1)  
        ccl_embedding = ccl_embedding.expand(-1, 3, -1, -1)
        # print('ccl_embedding shape', ccl_embedding.shape)

        pixel_values = []
        for bk in bokehK_values:
            img_sim = bokehK_simulation(image_path, depth_map_path, bk, disp_focus=0.96, gamma=2.2)    
            # save_path = os.path.join(self.root_path, f"simulated_img_bokeh_{bk.item():.2f}.png")
            # cv2.imwrite(save_path, img_sim)
            # print(f"Saved image: {save_path}")
            pixel_values.append(img_sim)

        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous() / 255.

        # Create bokehK embedding and concatenate it with CCL embedding
        bokehK_embedding = create_bokehK_embedding(bokehK_values, self.sample_size[0], self.sample_size[1])

        camera_embedding = torch.cat((bokehK_embedding, ccl_embedding), dim=1) 
        # print('camera_embedding shape', camera_embedding.shape)

        return pixel_values, image_caption, camera_embedding, bokehK_values

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                video, video_caption, camera_embedding, bokehK_values = self.get_batch(idx)
                break
            except Exception as e:
                idx = random.randint(0, self.length - 1)

        for transform in self.pixel_transforms:
            video = transform(video)

        sample = dict(pixel_values=video, text=video_caption, camera_embedding=camera_embedding, bokehK_values=bokehK_values)

        return sample



def test_camera_bokehK_dataset():
    root_path = '/home/yuan418/data/project/camera_dataset/camera_bokehK/'
    annotation_json = 'annotations/inference.json'

    print('------------------')
    dataset = CameraBokehK(
       root_path=root_path,
       annotation_json=annotation_json,
       sample_n_frames=4,
       sample_size=[256, 384],
       is_Train=False,
     )

    # choose one sample for testing
    idx = 1
    sample = dataset[idx]

    pixel_values = sample['pixel_values']
    text = sample['text']
    camera_embedding = sample['camera_embedding']
    print(f"Pixel values shape: {pixel_values.shape}")
    print(f"Text: {text}")
    print(f"camera embedding shape: {camera_embedding.shape}")


if __name__ == "__main__":
    test_camera_bokehK_dataset()
