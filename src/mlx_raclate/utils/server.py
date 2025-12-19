# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Union, Dict
import uvicorn
import gc
import mlx.core as mx
from mlx_raclate.utils.utils import PIPELINES, load

app = FastAPI(
    title="Raclate Inference API", 
    description="API for using Raclate pipelines (ModernBERT, LFM2, Qwen, etc.) on Apple Silicon",
    version="0.1.0"
)

# TODO:
# Separate Services: For complete isolation, run each pipeline type as a separate FastAPI service and use a lightweight API gateway to route requests.
# Worker Pool Architecture: Implement a worker pool where each worker specializes in a specific pipeline, and a dispatcher routes requests to the appropriate worker.


model_cache = {}

def get_model(model_name: str, pipeline_name: str, model_config: Optional[Dict] = None):
    """
    Factory function to get or create the appropriate model.
    Checks cache based on name, pipeline, AND configuration.
    """
    global model_cache
    
    # Create a cache key string that includes config to differentiate 
    # e.g. LFM2 with late_interaction=True vs False
    config_key = str(sorted(model_config.items())) if model_config else "default"
    
    current_key = f"{model_name}_{pipeline_name}_{config_key}"
    cached_key = model_cache.get("key", None)

    if cached_key == current_key:
        return model_cache

    # Garbage collection before loading new model
    if model_cache:
        print(f"Unloading previous model: {model_cache.get('model_name')}")
        model_cache = {}
        mx.eval() # Ensure evaluation of any pending ops
        gc.collect()
        mx.metal.clear_cache()

    print(f"Loading model: {model_name} | Pipeline: {pipeline_name} | Config: {model_config}")
    
    try:
        model, tokenizer = load(
            model_name,
            pipeline=pipeline_name,
            model_config=model_config
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    model_cache = {
        "key": current_key,
        "model_name": model_name,
        "pipeline": pipeline_name,
        "model": model,
        "tokenizer": tokenizer,
    }
    
    return model_cache

# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    model: str
    pipeline: str
    text: Union[str, List[str]]
    
    # Optional parameters depending on pipeline
    text_pair: Optional[Union[str, List[str]]] = Field(None, description="Secondary text for sequence classification pairs (e.g. NLI)")
    reference_text: Optional[Union[str, List[str]]] = Field(None, description="Documents/References for similarity search")
    
    # Configuration
    model_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Configuration overrides (e.g., {'use_late_interaction': True})")
    label_candidates: Optional[Union[Dict[str, str], List[str]]] = Field(None, description="For zero-shot-classification only")

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Main inference endpoint handling multiple pipelines.
    """
    if request.pipeline not in PIPELINES:
         raise HTTPException(status_code=400, detail=f"Pipeline '{request.pipeline}' not supported. Available: {PIPELINES}")

    # Standardize inputs to lists
    texts = request.text if isinstance(request.text, list) else [request.text]
    
    if len(texts) > 32: 
        raise HTTPException(status_code=400, detail="Batch size should not exceed 32 to protect memory")

    # Load Model
    model_info = get_model(request.model, request.pipeline, request.model_config)
    tokenizer = model_info["tokenizer"]
    model = model_info["model"]

    # Determine generic args
    max_len = getattr(model.config, "max_position_embeddings", 512)
    result = {}

    # -------------------------------------------------------------------------
    # Pipeline: Text Classification (Sentiment, NLI, Regression)
    # -------------------------------------------------------------------------
    if request.pipeline == "text-classification":
        text_pairs = None
        if request.text_pair:
            text_pairs = request.text_pair if isinstance(request.text_pair, list) else [request.text_pair]
            if len(text_pairs) != len(texts):
                raise HTTPException(status_code=400, detail="Length of text and text_pair must match")

        inputs = tokenizer._tokenizer(
            texts,
            text_pairs,
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=max_len
        )

        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict=True
        )

        probs = outputs["probabilities"] # Shape: [batch, num_labels]
        
        # Format output
        batch_results = []
        id2label = getattr(model.config, "id2label", None)
        
        # Convert to python list structure
        probs_list = probs.tolist()
        
        for i, row in enumerate(probs_list):
            if id2label:
                # Return dictionary mapping label -> score
                item_res = {id2label[str(j)]: score for j, score in enumerate(row)}
                # Sort by score descending
                item_res = dict(sorted(item_res.items(), key=lambda x: x[1], reverse=True))
                batch_results.append(item_res)
            else:
                # Just return raw scores (e.g. regression or missing config)
                batch_results.append(row)

        result = {"predictions": batch_results}

    # -------------------------------------------------------------------------
    # Pipeline: Sentence Similarity (Dense & Late Interaction)
    # -------------------------------------------------------------------------
    elif request.pipeline in ["sentence-similarity", "sentence-transformers"]:
        if not request.reference_text:
             raise HTTPException(status_code=400, detail="reference_text is required for sentence-similarity")
        
        refs = request.reference_text if isinstance(request.reference_text, list) else [request.reference_text]

        q_inputs = tokenizer._tokenizer(texts, return_tensors="mlx", padding=True, truncation=True, max_length=max_len)
        d_inputs = tokenizer._tokenizer(refs, return_tensors="mlx", padding=True, truncation=True, max_length=max_len)

        # The model handles the complexity (Cosine vs MaxSim) internally based on config
        outputs = model(
            input_ids=q_inputs['input_ids'],
            reference_input_ids=d_inputs['input_ids'],
            attention_mask=q_inputs['attention_mask'],
            reference_attention_mask=d_inputs['attention_mask'],
            return_dict=True
        )
        
        # Returns matrix: [batch_size, num_references]
        result = {"similarities": outputs['similarities'].tolist()}

    # -------------------------------------------------------------------------
    # Pipeline: Raw Embeddings
    # -------------------------------------------------------------------------
    elif request.pipeline == "embeddings":
        inputs = tokenizer._tokenizer(texts, return_tensors="mlx", padding=True, truncation=True, max_length=max_len)
        
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict=True
        )
        
        # 'embeddings' is the normalized pooled output
        result = {"embeddings": outputs['embeddings'].tolist()}

    # -------------------------------------------------------------------------
    # Pipeline: Masked LM (Raw)
    # -------------------------------------------------------------------------
    elif request.pipeline == "masked-lm":
        inputs = tokenizer._tokenizer(texts, return_tensors="mlx", padding=True, truncation=True, max_length=max_len)
        
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict=True
        )
        
        # Note: returning full logits is heavy. Usually this pipeline implies 
        # looking for the mask token, but for a raw API, we might just return 
        # a success status or specific logic. 
        # Here we return the logits for the mask token if present, else empty.
        
        mask_token_id = tokenizer.mask_token_id
        predictions = outputs["logits"]
        mask_positions = mx.argmax(inputs['input_ids'] == mask_token_id, axis=1)
        
        batch_results = []
        for i in range(len(texts)):
            if mask_token_id in inputs['input_ids'][i]:
                pos = mask_positions[i].item()
                # Top 5 for the mask
                token_logits = predictions[i, pos]
                probs = mx.softmax(token_logits)
                top_k = 5
                sorted_indices = mx.argsort(probs)[::-1][:top_k]
                
                top_tokens = []
                for idx in sorted_indices.tolist():
                    top_tokens.append({
                        "token": tokenizer.decode([idx]), 
                        "score": probs[idx].item()
                    })
                batch_results.append(top_tokens)
            else:
                batch_results.append(None)

        result = {"masked_predictions": batch_results}

    # -------------------------------------------------------------------------
    # Pipeline: Zero-Shot Classification (Custom Logic via Masked LM)
    # -------------------------------------------------------------------------
    elif request.pipeline == "zero-shot-classification":
        if not request.label_candidates:
            raise HTTPException(status_code=400, detail="label_candidates required for zero-shot")

        # Reuse the logic from your old server, adapted for batching
        if isinstance(request.label_candidates, dict):
            categories = "\n".join([f"{i}: {k} ({v})" for i, (k, v) in enumerate(request.label_candidates.items())])
            num_cats = len(request.label_candidates)
        else:
            categories = "\n".join([f"{i}: {label}" for i, label in enumerate(request.label_candidates)])
            num_cats = len(request.label_candidates)
        
        classification_inputs = []
        for text in texts:
            # Answer.ai / ModernBERT style prompt
            classification_input = f"""You will be given a text and categories to classify the text.

                {text}

                Read the text carefully and select the right category from the list. Only provide the index of the category:
                {categories}

                ANSWER: [unused0][MASK]
            """
            classification_inputs.append(classification_input)

        inputs = tokenizer._tokenizer(
            classification_inputs, 
            return_tensors="mlx", 
            padding=True, 
            truncation=True, 
            max_length=max_len
        )

        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask', None),
            return_dict=True
        )

        predictions = outputs["logits"]
        mask_token_id = tokenizer.mask_token_id
        mask_positions = mx.argmax(inputs['input_ids'] == mask_token_id, axis=1)

        batch_results = []
        for i in range(len(texts)):
            mask_position = mask_positions[i].item()
            masked_token_predictions = predictions[i, mask_position]
            
            probs = mx.softmax(masked_token_predictions)
            top_k = min(5, num_cats)
            
            # Sort generic probabilities
            sorted_indices = mx.argsort(probs)[::-1][:top_k]
            top_probs = probs[sorted_indices]
            
            item_res = []
            for idx, logit in zip(sorted_indices.tolist(), top_probs.tolist()):
                item_res.append({"label_index": tokenizer.decode([idx]), "score": logit})
            
            batch_results.append(item_res)

        result = {"classification": batch_results}

    # Clean up
    mx.metal.clear_cache()
    gc.collect()
    
    return result

@app.get("/status")
async def status():
    return {
        "status": "online",
        "loaded_model": model_cache.get("model_name"),
        "loaded_pipeline": model_cache.get("pipeline"),
        "loaded_config_key": model_cache.get("key")
    }

@app.post("/unload")
async def unload_model():
    global model_cache
    if not model_cache:
        return {"message": "No model loaded"}
    
    name = model_cache.get("model_name")
    model_cache = {}
    gc.collect()
    mx.metal.clear_cache()
    return {"message": f"Unloaded {name}"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1)

### EXAMPLE
'''
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": [
      "The new MacBook Pro with M3 chip delivers exceptional performance and battery life.",
      "I was really disappointed with the customer service at that restaurant.",
      "This movie has beautiful cinematography but the plot is confusing.",
      "The aging of the population is the archetype of an unpleasant truth for mainstream media readers and for voters, which does not encourage anyone to put it on the table. Age pyramids, birth and fertility indicators, and celibacy rates in all developed countries indicate that the situation is worrying. Among these countries, some managed to stay on-track until about 10 years ago but they eventually fell into line."
    ],
    "model": "answerdotai/ModernBERT-Large-Instruct",
    "pipeline": "zero-shot-classification",
    "label_candidates": {
        "artificial intelligence": "The study of computer science that focuses on the creation of intelligent machines that work and react like humans.",
        "physics": "The study of matter, energy, and the fundamental forces of nature.",
        "society" : "The aggregate of people living together in a more or less ordered community.",
        "biology" : "The study of living organisms, divided into many specialized fields that cover their morphology, physiology, anatomy, behavior, origin, and distribution.",
        "environment" : "The surroundings or conditions in which a person, animal, or plant lives or operates.",
        "health" : "The state of being free from illness or injury.",
        "finance" : "The management of large amounts of money, especially by governments or large companies."
    }
  }'

'''

'''
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "NousResearch/Minos-v1",
    "pipeline": "text-classification",
    "text": [
      "I absolutely love this new framework!",
      "The service was terrible and slow."
    ]
  }'
'''

'''
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LiquidAI/LFM2-ColBERT-350M",
    "pipeline": "sentence-similarity",
    "model_config": {
        "use_late_interaction": true
    },
    "text": ["What is liquid AI?"],
    "reference_text": [
        "Liquid AI builds efficient foundation models.",
        "Water is a liquid state of matter."
    ]
  }'
'''