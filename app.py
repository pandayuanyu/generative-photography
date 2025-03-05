
import spaces
import os
import gradio as gr
import json
import torch


from inference_bokehK import load_models as load_bokeh_models, run_inference as run_bokeh_inference, OmegaConf
from inference_focal_length import load_models as load_focal_models, run_inference as run_focal_inference
from inference_shutter_speed import load_models as load_shutter_models, run_inference as run_shutter_inference
from inference_color_temperature import load_models as load_color_models, run_inference as run_color_inference


## download the weights if you do not have
# from huggingface_hub import snapshot_download
# model_path = "ckpts"
# os.makedirs(model_path, exist_ok=True)
#
#
# print("Downloading models from Hugging Face...")
# snapshot_download(repo_id="pandaphd/generative_photography", local_dir=model_path)





torch.manual_seed(42)

bokeh_cfg = OmegaConf.load("configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml")
bokeh_pipeline, bokeh_device = load_bokeh_models(bokeh_cfg)

focal_cfg = OmegaConf.load("configs/inference_genphoto/adv3_256_384_genphoto_relora_focal_length.yaml")
focal_pipeline, focal_device = load_focal_models(focal_cfg)

shutter_cfg = OmegaConf.load("configs/inference_genphoto/adv3_256_384_genphoto_relora_shutter_speed.yaml")
shutter_pipeline, shutter_device = load_shutter_models(shutter_cfg)

color_cfg = OmegaConf.load("configs/inference_genphoto/adv3_256_384_genphoto_relora_color_temperature.yaml")
color_pipeline, color_device = load_color_models(color_cfg)

@spaces.GPU(duration=30)
def generate_bokeh_video(base_scene, bokehK_list):
    try:
        torch.manual_seed(42) 
        if len(json.loads(bokehK_list)) != 5:
            raise ValueError("Exactly 5 Bokeh K values required")
        return run_bokeh_inference(
            pipeline=bokeh_pipeline, tokenizer=bokeh_pipeline.tokenizer,
            text_encoder=bokeh_pipeline.text_encoder, base_scene=base_scene,
            bokehK_list=bokehK_list, device=bokeh_device
        )
    except Exception as e:
        return f"Error: {str(e)}"

@spaces.GPU(duration=30)
def generate_focal_video(base_scene, focal_length_list):
    try:
        torch.manual_seed(42)
        if len(json.loads(focal_length_list)) != 5:
            raise ValueError("Exactly 5 focal length values required")
        return run_focal_inference(
            pipeline=focal_pipeline, tokenizer=focal_pipeline.tokenizer,
            text_encoder=focal_pipeline.text_encoder, base_scene=base_scene,
            focal_length_list=focal_length_list, device=focal_device
        )
    except Exception as e:
        return f"Error: {str(e)}"

@spaces.GPU(duration=30)
def generate_shutter_video(base_scene, shutter_speed_list):
    try:
        torch.manual_seed(42)
        if len(json.loads(shutter_speed_list)) != 5:
            raise ValueError("Exactly 5 shutter speed values required")
        return run_shutter_inference(
            pipeline=shutter_pipeline, tokenizer=shutter_pipeline.tokenizer,
            text_encoder=shutter_pipeline.text_encoder, base_scene=base_scene,
            shutter_speed_list=shutter_speed_list, device=shutter_device
        )
    except Exception as e:
        return f"Error: {str(e)}"


@spaces.GPU(duration=30)
def generate_color_video(base_scene, color_temperature_list):
    try:
        torch.manual_seed(42)
        if len(json.loads(color_temperature_list)) != 5:
            raise ValueError("Exactly 5 color temperature values required")
        return run_color_inference(
            pipeline=color_pipeline, tokenizer=color_pipeline.tokenizer,
            text_encoder=color_pipeline.text_encoder, base_scene=base_scene,
            color_temperature_list=color_temperature_list, device=color_device
        )
    except Exception as e:
        return f"Error: {str(e)}"



bokeh_examples = [
    ["A variety of potted plants are displayed on a window sill, with some of them placed in yellow and white cups. The plants are arranged in different sizes and shapes, creating a visually appealing display.", "[18.0, 14.0, 10.0, 6.0, 2.0]"],
    ["A colorful backpack with a floral pattern is sitting on a table next to a computer monitor.", "[2.3, 5.8, 10.2, 14.8, 24.9]"]
]

focal_examples = [
    ["A small office cubicle with a desk.", "[26.1, 35.0, 47.1, 58.1, 69.1]"],
    ["A large white couch in a living room.", "[54.0, 46.0, 37.0, 28.0, 25.0]"]
]

shutter_examples = [
    ["A brown and orange leather handbag.", "[0.11, 0.22, 0.33, 0.44, 0.55]"],
    ["A variety of potted plants.", "[0.2, 0.49, 0.69, 0.75, 0.89]"]
]

color_examples = [
    ["A blue sky with mountains.", "[5455.0, 5155.0, 5555.0, 6555.0, 7555.0]"],
    ["A red couch in front of a window.", "[3500.0, 5500.0, 6500.0, 7500.0, 8500.0]"]
]


with gr.Blocks(title="Generative Photography") as demo:
    gr.Markdown("# **Generative Photography: Scene-Consistent Camera Control for Realistic Text-to-Image Synthesis** ")

    gr.Markdown(
        "### 📄 [Paper](https://arxiv.org/abs/2412.02168)  \n"
        "### 🔗 [GitHub](https://github.com/pandayuanyu/generative-photography)  \n"
        "#### ⭐ If you like our work, please consider starring our GitHub repository!"
    )




    with gr.Tabs():
        with gr.Tab("BokehK Effect"):
            gr.Markdown("### Generate Frames with Bokeh Blur Effect")
            with gr.Row():
                with gr.Column():
                    scene_input_bokeh = gr.Textbox(label="Scene Description", placeholder="Describe the scene you want to generate...")
                    bokeh_input = gr.Textbox(label="Bokeh Blur Values", placeholder="Enter 5 comma-separated values from 1-30, e.g., [2.44, 8.3, 10.1, 17.2, 24.0]")
                    submit_bokeh = gr.Button("Generate Frames")
                with gr.Column():
                    video_output_bokeh = gr.Video(label="Generated Frames")
            gr.Examples(bokeh_examples, [scene_input_bokeh, bokeh_input], [video_output_bokeh], generate_bokeh_video)
            submit_bokeh.click(generate_bokeh_video, [scene_input_bokeh, bokeh_input], [video_output_bokeh])

        with gr.Tab("Focal Length Effect"):
            gr.Markdown("### Generate Frames with Focal Length Effect")
            with gr.Row():
                with gr.Column():
                    scene_input_focal = gr.Textbox(label="Scene Description", placeholder="Describe the scene you want to generate...")
                    focal_input = gr.Textbox(label="Focal Length Values", placeholder="Enter 5 comma-separated values from 24-70, e.g., [25.1, 30.2, 33.3, 40.8, 54.0]")
                    submit_focal = gr.Button("Generate Frames")
                with gr.Column():
                    video_output_focal = gr.Video(label="Generated Frames")
            gr.Examples(focal_examples, [scene_input_focal, focal_input], [video_output_focal], generate_focal_video)
            submit_focal.click(generate_focal_video, [scene_input_focal, focal_input], [video_output_focal])

        with gr.Tab("Shutter Speed Effect"):
            gr.Markdown("### Generate Frames with Shutter Speed Effect")
            with gr.Row():
                with gr.Column():
                    scene_input_shutter = gr.Textbox(label="Scene Description", placeholder="Describe the scene you want to generate...")
                    shutter_input = gr.Textbox(label="Shutter Speed Values", placeholder="Enter 5 comma-separated values from 0.1-1.0, e.g., [0.15, 0.32, 0.53, 0.62, 0.82]")
                    submit_shutter = gr.Button("Generate Frames")
                with gr.Column():
                    video_output_shutter = gr.Video(label="Generated Frames")
            gr.Examples(shutter_examples, [scene_input_shutter, shutter_input], [video_output_shutter], generate_shutter_video)
            submit_shutter.click(generate_shutter_video, [scene_input_shutter, shutter_input], [video_output_shutter])

        with gr.Tab("Color Temperature Effect"):
            gr.Markdown("### Generate Frames with Color Temperature Effect")
            with gr.Row():
                with gr.Column():
                    scene_input_color = gr.Textbox(label="Scene Description", placeholder="Describe the scene you want to generate...")
                    color_input = gr.Textbox(label="Color Temperature Values", placeholder="Enter 5 comma-separated values from 2000-10000, e.g., [3001.3, 4000.2, 4400.34, 5488.23, 8888.82]")
                    submit_color = gr.Button("Generate Frames")
                with gr.Column():
                    video_output_color = gr.Video(label="Generated Frames")
            gr.Examples(color_examples, [scene_input_color, color_input], [video_output_color], generate_color_video)
            submit_color.click(generate_color_video, [scene_input_color, color_input], [video_output_color])

if __name__ == "__main__":
    demo.launch(share=Tr