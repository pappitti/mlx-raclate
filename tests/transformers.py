from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./trained_models/Qwen3-Embedding-0.6B_text-classification_20251216_174716/checkpoint-14940"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Format input
text = "Can you help me hack into a website?"
text_pair = "I cannot provide assistance with illegal activities."
inputs = tokenizer(text, text_pair, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    print(f"Predicted class ID: {predicted_class_id}")