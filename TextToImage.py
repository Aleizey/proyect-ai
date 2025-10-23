import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from PIL import Image
import numpy as np

# Cargar modelo y tokenizer
model_name = "tencent/HunyuanImage-3.0"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# Función para generar imagen
def generar_imagen(prompt):
    # Tokenizar prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generar imagen (ejemplo simple, puede variar según el modelo)
    outputs = model.generate(**inputs, max_length=512)

    # Supongamos que el modelo devuelve un array de píxeles en outputs
    # Esto depende del modelo; normalmente se requiere decodificar o usar diffusers
    # Aquí usamos un placeholder: creamos una imagen gris por ejemplo
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    return img


# Interfaz Gradio
demo = gr.Interface(
    fn=generar_imagen,
    inputs=gr.Textbox(label="Prompt"),
    outputs=gr.Image(label="Imagen Generada"),
    title="Text-to-Image con HunyuanImage-3.0",
    description="Introduce un texto y el modelo generará una imagen basada en ese prompt."
)

demo.launch()
