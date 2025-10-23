import gradio as gr
import torch
from transformers import pipeline

asr = pipeline(task="automatic-speech-recognition",model="openai/whisper-small",
    device=0 if torch.cuda.is_available() else -1)
pipe = pipeline(task="translation",model="Helsinki-NLP/opus-mt-es-en")

def transcribe(audio):

    if audio is None:
        return "Por favor, sube un archivo de audio."
    
    audio_a_texto = asr(audio)["text"]

    if not audio_a_texto.strip():
        return "Introduce algún texto."
    
    resultado = pipe(audio_a_texto)
    
    return resultado[0]["translation_text"]


gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.TextArea(label="Traducción")
).launch()