import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 
    if torch.cuda.is_available() else torch.float32)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generar_imagen(text):
    image = pipe(text).images[0]
    return image

image = gr.Interface(
    fn=generar_imagen,
    inputs=gr.Textbox(label="Descripción de la imagen"),
    outputs=gr.Image(label="Imagen generada")
)

image.launch()
