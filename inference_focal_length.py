import os
import torch
import logging
import argparse
import json
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler
from einops import rearrange

from genphoto.pipelines.pipeline_animation import GenPhotoPipeline
from genphoto.models.unet import UNet3DConditionModelCameraCond
from genphoto.models.camera_adaptor import CameraCameraEncoder, CameraAdaptor
from genphoto.utils.util import save_videos_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_focal_length_embedding(focal_length_values, target_height, target_width, base_focal_length=24.0, sensor_height=24.0, sensor_width=36.0):
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
       # print('crop_h', crop_h)
        crop_w = torch.round(crop_ratio_xs[i] * target_width).int().item()  # Rounded cropped width for the current frame

        # Ensure the cropped dimensions are within valid bounds
        crop_h = max(1, min(target_height, crop_h))
        crop_w = max(1, min(target_width, crop_w))

        # Set the center region of the focal_length embedding to 1 for the current frame
        focal_length_embedding[i, :,
        center_h - crop_h // 2: center_h + crop_h // 2,
        center_w - crop_w // 2: center_w + crop_w // 2] = 1.0

    return focal_length_embedding


class Camera_Embedding(Dataset):
    def __init__(self, focal_length_values, tokenizer, text_encoder, device, sample_size=[256, 384]):
        self.focal_length_values = focal_length_values.to(device)                                                 
        self.tokenizer = tokenizer                 
        self.text_encoder = text_encoder       
        self.device = device               
        self.sample_size = sample_size

    def load(self):

        if len(self.focal_length_values) != 5:
            raise ValueError("Expected 5 focal_length values")

        # Generate prompts for each focal length value and append focal_length information to caption
        prompts = []
        for fl in self.focal_length_values:
            prompt = f"<focal length: {fl.item()}>"
            prompts.append(prompt)
        

        # Tokenize prompts and encode to get embeddings
        with torch.no_grad():
            prompt_ids = self.tokenizer(
                prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)

            encoder_hidden_states = self.text_encoder(input_ids=prompt_ids).last_hidden_state  # Shape: (f, sequence_length, hidden_size)
        

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
        frame = concatenated_differences.size(0)
        concatenated_differences = torch.cat(differences, dim=0)

        pad_length = 128 - concatenated_differences.size(1)
        if pad_length > 0:
        # Pad along the second dimension (77 -> 128), pad only on the right side
            concatenated_differences_padded = F.pad(concatenated_differences, (0, 0, 0, pad_length))


        ccl_embedding = concatenated_differences_padded.reshape(frame, self.sample_size[0], self.sample_size[1])
        ccl_embedding = ccl_embedding.unsqueeze(1)  
        ccl_embedding = ccl_embedding.expand(-1, 3, -1, -1)
        ccl_embedding = ccl_embedding.to(self.device)
        focal_length_embedding = create_focal_length_embedding(self.focal_length_values, self.sample_size[0], self.sample_size[1]).to(self.device)

        camera_embedding = torch.cat((focal_length_embedding, ccl_embedding), dim=1)
        return camera_embedding


def load_models(cfg):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(cfg.noise_scheduler_kwargs))
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_path, subfolder="text_encoder").to(device)
    text_encoder.requires_grad_(False)
    unet = UNet3DConditionModelCameraCond.from_pretrained_2d(
        cfg.pretrained_model_path,
        subfolder=cfg.unet_subfolder,
        unet_additional_kwargs=cfg.unet_additional_kwargs
    ).to(device)
    unet.requires_grad_(False)

    camera_encoder = CameraCameraEncoder(**cfg.camera_encoder_kwargs).to(device)
    camera_encoder.requires_grad_(False)
    camera_adaptor = CameraAdaptor(unet, camera_encoder)
    camera_adaptor.requires_grad_(False)
    camera_adaptor.to(device)

    logger.info("Setting the attention processors")
    unet.set_all_attn_processor(
        add_spatial_lora=cfg.lora_ckpt is not None,
        add_motion_lora=cfg.motion_lora_rank > 0,
        lora_kwargs={"lora_rank": cfg.lora_rank, "lora_scale": cfg.lora_scale},
        motion_lora_kwargs={"lora_rank": cfg.motion_lora_rank, "lora_scale": cfg.motion_lora_scale},
        **cfg.attention_processor_kwargs
    )

    if cfg.lora_ckpt is not None:
        print(f"Loading the lora checkpoint from {cfg.lora_ckpt}")
        lora_checkpoints = torch.load(cfg.lora_ckpt, map_location=unet.device)
        if 'lora_state_dict' in lora_checkpoints.keys():
            lora_checkpoints = lora_checkpoints['lora_state_dict']
        _, lora_u = unet.load_state_dict(lora_checkpoints, strict=False)
        assert len(lora_u) == 0
        print(f'Loading done')

    if cfg.motion_module_ckpt is not None:
        print(f"Loading the motion module checkpoint from {cfg.motion_module_ckpt}")
        mm_checkpoints = torch.load(cfg.motion_module_ckpt, map_location=unet.device)
        _, mm_u = unet.load_state_dict(mm_checkpoints, strict=False)
        assert len(mm_u) == 0
        print("Loading done")
    
    if cfg.camera_adaptor_ckpt is not None:
        logger.info(f"Loading camera adaptor from {cfg.camera_adaptor_ckpt}")
        camera_adaptor_checkpoint = torch.load(cfg.camera_adaptor_ckpt, map_location=device)
        camera_encoder_state_dict = camera_adaptor_checkpoint['camera_encoder_state_dict']
        attention_processor_state_dict = camera_adaptor_checkpoint['attention_processor_state_dict']
        camera_enc_m, camera_enc_u = camera_adaptor.camera_encoder.load_state_dict(camera_encoder_state_dict, strict=False)

        assert len(camera_enc_m) == 0 and len(camera_enc_u) == 0
        _, attention_processor_u = camera_adaptor.unet.load_state_dict(attention_processor_state_dict, strict=False)
        assert len(attention_processor_u) == 0
        
        logger.info("Camera Adaptor loading done")
    else:
        logger.info("No Camera Adaptor checkpoint used")

    pipeline = GenPhotoPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        camera_encoder=camera_encoder
    ).to(device)
    pipeline.enable_vae_slicing()

    return pipeline, device


def run_inference(pipeline, tokenizer, text_encoder, base_scene, focal_length_list, output_dir, device, video_length=5, height=256, width=384):
    os.makedirs(output_dir, exist_ok=True)

    focal_length_list_str = focal_length_list
    focal_length_values = json.loads(focal_length_list_str)
    focal_length_values = torch.tensor(focal_length_values).unsqueeze(1)

    # Ensure camera_embedding is on the correct device
    camera_embedding = Camera_Embedding(focal_length_values, tokenizer, text_encoder, device).load()
    camera_embedding = rearrange(camera_embedding.unsqueeze(0), "b f c h w -> b c f h w")

    with torch.no_grad():
        sample = pipeline(
            prompt=base_scene,
            camera_embedding=camera_embedding,
            video_length=video_length,
            height=height,
            width=width,
            num_inference_steps=25,
            guidance_scale=8.0
        ).videos[0]

    sample_save_path = os.path.join(output_dir, "sample.gif")
    save_videos_grid(sample[None, ...], sample_save_path)
    logger.info(f"Saved generated sample to {sample_save_path}")


def main(config_path, base_scene, focal_length_list):
    torch.manual_seed(42)
    cfg = OmegaConf.load(config_path)
    logger.info("Loading models...")
    pipeline, device = load_models(cfg)
    logger.info("Starting inference...")


    run_inference(pipeline, pipeline.tokenizer, pipeline.text_encoder, base_scene, focal_length_list, cfg.output_dir, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--base_scene", type=str, required=True, help="invariant scene caption as JSON string")
    parser.add_argument("--focal_length_list", type=str, required=True, help="focal_length values as JSON string")
    args = parser.parse_args()
    main(args.config, args.base_scene, args.focal_length_list)

    # usage example
    # python inference_focal_length.py --config configs/inference_genphoto/adv3_256_384_genphoto_relora_focal_length.yaml --base_scene "A cozy living room with a large, comfy sofa and a coffee table." --focal_length_list "[25.0, 35.0, 45.0, 55.0, 65.0]"

