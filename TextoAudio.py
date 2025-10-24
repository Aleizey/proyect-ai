from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/VibeVoice-1.5B", torch_dtype="auto")