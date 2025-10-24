import gradio as gr
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline

# model = pipeline("text-to-speech", model="microsoft/VibeVoice-1.5B")
asr = pipeline(task="automatic-speech-recognition",model="openai/whisper-small",
    device=0 if torch.cuda.is_available() else -1)
pipe = pipeline(task="translation",model="Helsinki-NLP/opus-mt-es-en")

#imagen 
model_id = "runwayml/stable-diffusion-v1-5"
pipe2 = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 
    if torch.cuda.is_available() else torch.float32)
pipe2 = pipe2.to("cuda" if torch.cuda.is_available() else "cpu")

def transcribe(audio):

    if audio is None:
        return "Por favor, sube un archivo de audio."
    
    audio_a_texto = asr(audio)["text"]

    if not audio_a_texto.strip():
        return "Introduce algún texto."
    
    resultado = pipe(audio_a_texto)

    # texto_a_audio = model(resultado[0]["translation_text"])

    image = pipe2(resultado[0]["translation_text"]).images[0]
    
    return image


gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Image(label="Imagen generada")
).launch()