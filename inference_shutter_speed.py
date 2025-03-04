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

def create_shutter_speed_embedding(shutter_speed_values, target_height, target_width, base_exposure=0.5):

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



class Camera_Embedding(Dataset):
    def __init__(self, shutter_speed_values, tokenizer, text_encoder, device, sample_size=[256, 384]):
        self.shutter_speed_values = shutter_speed_values.to(device)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device  
        self.sample_size = sample_size

    def load(self):

        if len(self.shutter_speed_values) != 5:
            raise ValueError("Expected 5 shutter_speed values")

        # Generate prompts for each shutter_speed value and append shutter_speed information to caption
        prompts = []
        for ss in self.shutter_speed_values:
            prompt = f"<exposure: {ss.item()}>"
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

            concatenated_differences_padded = F.pad(concatenated_differences, (0, 0, 0, pad_length))


        ccl_embedding = concatenated_differences_padded.reshape(frame, self.sample_size[0], self.sample_size[1])
        ccl_embedding = ccl_embedding.unsqueeze(1)  
        ccl_embedding = ccl_embedding.expand(-1, 3, -1, -1)
        ccl_embedding = ccl_embedding.to(self.device)
        shutter_speed_embedding = create_shutter_speed_embedding(self.shutter_speed_values, self.sample_size[0], self.sample_size[1]).to(self.device)
        camera_embedding = torch.cat((shutter_speed_embedding, ccl_embedding), dim=1)
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
    

    # 🔥 加载 Camera Adaptor Checkpoint
    if cfg.camera_adaptor_ckpt is not None:
        logger.info(f"Loading camera adaptor from {cfg.camera_adaptor_ckpt}")
        camera_adaptor_checkpoint = torch.load(cfg.camera_adaptor_ckpt, map_location=device)

        # 加载 Camera Encoder
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


def run_inference(pipeline, tokenizer, text_encoder, base_scene, shutter_speed_list, output_dir, device, video_length=5, height=256, width=384):

    os.makedirs(output_dir, exist_ok=True)

    shutter_speed_list_str = shutter_speed_list
    shutter_speed_values = json.loads(shutter_speed_list_str)
    shutter_speed_values = torch.tensor(shutter_speed_values).unsqueeze(1)

    # Ensure camera_embedding is on the correct device
    camera_embedding = Camera_Embedding(shutter_speed_values, tokenizer, text_encoder, device).load()
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


def main(config_path, base_scene, shutter_speed_list):
    torch.manual_seed(42)
    cfg = OmegaConf.load(config_path)
    logger.info("Loading models...")
    pipeline, device = load_models(cfg)
    logger.info("Starting inference...")


    run_inference(pipeline, pipeline.tokenizer, pipeline.text_encoder, base_scene, shutter_speed_list, cfg.output_dir, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--base_scene", type=str, required=True, help="invariant scene caption as JSON string")
    parser.add_argument("--shutter_speed_list", type=str, required=True, help="shutter_speed values as JSON string")
    args = parser.parse_args()
    main(args.config, args.base_scene, args.shutter_speed_list)

    # usage example 
    # python inference_shutter_speed.py --config configs/inference_genphoto/adv3_256_384_genphoto_relora_shutter_speed.yaml --base_scene "A modern bathroom with a mirror and soft lighting." --shutter_speed_list "[0.1, 0.3, 0.52, 0.7, 0.8]"

