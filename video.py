import gradio as gr
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float32 
).to("cpu")

def text_to_video(prompt):
    video_frames = pipe(prompt=prompt, num_inference_steps=25).frames
    return video_frames 

gr.Interface(
    fn=text_to_video,
    inputs=gr.Textbox(),
    outputs=gr.Video(label="Video generado")
).launch()
