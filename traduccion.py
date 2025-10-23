import gradio as gr
from transformers import pipeline

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

# Función de traducción
def traducir_es_a_en(txt):
    if not txt.strip():
        return "Introduce algún texto."
    resultado = pipe(txt)
    return resultado[0]["translation_text"]

# Interfaz con Gradio
demo = gr.Interface(
    fn=traducir_es_a_en,
    inputs=gr.Textbox(label="Texto en español"),
    outputs=gr.Textbox(label="Traducción al inglés"),
    title="Traductor Español → Inglés (Helsinki-NLP)",
    description="Introduce texto en español y obtén su traducción al inglés usando el modelo especializado Helsinki-NLP."
)

demo.launch()


