from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import polars as pl

model_path = "./trained_models/LFM2-350M_text-classification_20251216_234824/LFM2-350M_text-classification_20251216_234824_E1"
validation_data_path = "data/20251205_1125/validation.parquet"

# Load validation data
df = pl.read_parquet(validation_data_path)
random_sample = df.sample(1).to_dicts()[0]
print(f"Random sample from validation data: {random_sample}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

id2label = model.config.id2label

# Format input
text = random_sample["question"]
text_pair = random_sample["response_anonymized"] #"I cannot provide assistance with illegal activities."
inputs = tokenizer(text, text_pair, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    print(f"Predicted class ID: {predicted_class_id}")
    print(f"Predicted label: {id2label[predicted_class_id]}")
    print(f"expected label: {random_sample['classification']}")