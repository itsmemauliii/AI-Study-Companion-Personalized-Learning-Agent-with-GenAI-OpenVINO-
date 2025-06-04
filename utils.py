import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from openvino.runtime import Core

MODEL_NAME = "t5-small"  # You can replace with your model

def load_model():
    # Load model with OpenVINO optimization
    core = Core()
    # For demo, loading standard transformers model (you can replace with OpenVINO IR model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    # In real use case, convert and load OpenVINO IR model here
    return model, tokenizer

def generate_response(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**inputs, max_length=200)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded
