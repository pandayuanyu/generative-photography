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

def create_bokehK_embedding(bokehK_values, target_height, target_width):
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



class Camera_Embedding(Dataset):
    def __init__(self, bokehK_values, tokenizer, text_encoder, device, sample_size=[256, 384]):
        self.bokehK_values = bokehK_values.to(device)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device  
        self.sample_size = sample_size

    def load(self):

        if len(self.bokehK_values) != 5:
            raise ValueError("Expected 5 bokehK values")

        # Generate prompts for each bokehK value and append bokehK information to caption
        prompts = []
        for bb in self.bokehK_values:
            prompt = f"<bokeh kernel size: {bb.item()}>"
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

        # Concatenate differences along the batch dimension (f)
        concatenated_differences = torch.cat(differences, dim=0)

        pad_length = 128 - concatenated_differences.size(1)

        if pad_length > 0:
            concatenated_differences_padded = F.pad(concatenated_differences, (0, 0, 0, pad_length))


        ccl_embedding = concatenated_differences_padded.reshape(frame, self.sample_size[0], self.sample_size[1])
        ccl_embedding = ccl_embedding.unsqueeze(1)  
        ccl_embedding = ccl_embedding.expand(-1, 3, -1, -1)
        ccl_embedding = ccl_embedding.to(self.device)
        bokehK_embedding = create_bokehK_embedding(self.bokehK_values, self.sample_size[0], self.sample_size[1]).to(self.device)
        camera_embedding = torch.cat((bokehK_embedding, ccl_embedding), dim=1)
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


def run_inference(pipeline, tokenizer, text_encoder, base_scene, bokehK_list, output_dir, device, video_length=5, height=256, width=384):
    os.makedirs(output_dir, exist_ok=True)

    bokehK_list_str = bokehK_list
    bokehK_values = json.loads(bokehK_list_str)
    bokehK_values = torch.tensor(bokehK_values).unsqueeze(1)

    # Ensure camera_embedding is on the correct device
    camera_embedding = Camera_Embedding(bokehK_values, tokenizer, text_encoder, device).load()
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


def main(config_path, base_scene, bokehK_list):
    torch.manual_seed(42)
    cfg = OmegaConf.load(config_path)
    logger.info("Loading models...")
    pipeline, device = load_models(cfg)
    logger.info("Starting inference...")


    run_inference(pipeline, pipeline.tokenizer, pipeline.text_encoder, base_scene, bokehK_list, cfg.output_dir, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--base_scene", type=str, required=True, help="invariant scene caption as JSON string")
    parser.add_argument("--bokehK_list", type=str, required=True, help="Bokeh K values as JSON string")
    args = parser.parse_args()
    main(args.config, args.base_scene, args.bokehK_list)

    # usage example
    # python inference_bokehK.py --config configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml --base_scene "A young boy wearing an orange jacket is standing on a crosswalk, waiting to cross the street." --bokehK_list "[2.44, 8.3, 10.1, 17.2, 24.0]"

