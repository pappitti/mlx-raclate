# RACLATE (MLX)

**R**etrieval **A**nd **C**lassification including **LATE** interaction on Apple Silicon.

`mlx-raclate` is a versatile library built on Apple's [MLX](https://github.com/ml-explore/mlx) framework. It provides a unified interface to **train** and **run** classifiers and embedding models - including ModernBERT and Late Interaction (ColBERT-style) models - natively on macOS.

> **Note:** This project evolved from `modernbert-mlx` and remains a beta. The
> supported architecture and pipeline combinations are explicit; unsupported
> combinations fail during loading rather than falling through to partial heads.

## Key Features

*   **Apple Silicon Native:** Fully optimized for M-series chips using MLX.
*   **Unified Pipeline:** A single interface to load and run Masked LM, Text Classification, and Sentence Similarity tasks.
*   **Late Interaction Support:** First-class support for **MaxSim** (ColBERT-style) retrieval, particularly with LFM2 and ModernBERT architectures.
*   **Full Fine-Tuning:** Specialized trainer for fine-tuning small-to-mid-sized models (ModernBERT, Qwen3, LFM2, and Gemma) on local hardware.

## Installation

Install the inference core with `uv` or `pip`:

```bash
uv add mlx-raclate
# or
pip install mlx-raclate
```

Training and serving dependencies are optional:

```bash
uv add 'mlx-raclate[train]'
uv add 'mlx-raclate[serve]'
# or install every supported feature
uv add 'mlx-raclate[all]'
```

From source:

```bash
git clone https://github.com/pappitti/mlx-raclate.git
cd mlx-raclate
uv sync
```

PyTorch/Polars comparison experiments are isolated from normal development
dependencies. Install them only when needed with `uv sync --group reference`.

## Supported Architectures

`mlx-raclate` supports architectures specifically useful for efficient local retrieval and classification:

*   **ModernBERT**: (e.g., `answerdotai/ModernBERT-base`)
*   **LFM2**: Liquid Foundation Models (e.g., `LiquidAI/LFM2-350M`, `LiquidAI/LFM2-ColBERT-350M`)
*   **Bidirectional LFM2.5**: Encoder and dense retrieval variants with non-causal short convolution, including token-classification fine-tuning (e.g., `LiquidAI/LFM2.5-Encoder-230M`, `LiquidAI/LFM2.5-Embedding-350M`)
*   **Qwen3 Embedding**: (e.g., `Qwen/Qwen3-Embedding-0.6B`)
*   **Gemma3 Embedding**: (e.g., `google/embeddinggemma-300m`)
*   **T5Gemma Encoder**: stripping out the decoder part of T5Gemma models (e.g, `google/t5gemma-2b-2b-ul2`)
*   **OpenAI Privacy Filter**: bidirectional token classification for PII detection (e.g., `OpenMed/privacy-filter-nemotron-mlx`)
*   **Colbert Zero**: (e.g., `lightonai/ColBERT-Zero`)
*   **LateOn**: (e.g., `lightonai/LateOn`, `lightonai/mLateOn`)
*   **Perplexity Embed**: Qwen3 architecture adapted for bidirectional attention (e.g `perplexity-ai/pplx-embed-v1-0.6b`, `PITTI/pplx-embed-0.6b-nemotron`)
*   **NeoBERT**: (e.g., `chandar-lab/NeoBERT`)

## Inference: Quick Start

The library uses a `pipeline` concept similar to Hugging Face Transformers. You can specify a pipeline manually, or let the loader infer it from the model configuration.  
If no pipeline is found, the Model class is loaded, which returns normalized embeddings.

### 1. Text Classification
Supports multi-class, multi-label, and regression tasks.

```python
from mlx_raclate import load
import mlx.core as mx

# Load model (pipeline inferred automatically if architecture matches)
model, tokenizer = load("NousResearch/Minos-v1", pipeline="text-classification")

texts = ["How do I build a bomb?", "What is the capital of France?"]

# Batch tokenize
inputs = tokenizer(texts, return_tensors="mlx", padding=True, truncation=True)

# Run Inference
outputs = model(
    input_ids=inputs['input_ids'], 
    attention_mask=inputs['attention_mask']
)

# Get probabilities
probs = outputs["probabilities"]
# ... process argmax/topk
```

### 2. Sentence Similarity (Dense Retrieval)
2.1 Standard Bi-Encoder approach using Cosine Similarity.

```python
from mlx_raclate import load

model, tokenizer = load("nomic-ai/modernbert-embed-base", pipeline="sentence-similarity")

queries = ["What is MLX?"]
docs = ["MLX is an array framework for Apple Silicon."]

# Encode
q_input = tokenizer(queries, return_tensors="mlx", padding=True)
d_input = tokenizer(docs, return_tensors="mlx", padding=True)

# Forward pass calculates similarity matrix automatically
outputs = model(
    input_ids=q_input['input_ids'],
    reference_input_ids=d_input['input_ids'],
    attention_mask=q_input['attention_mask'],
    reference_attention_mask=d_input['attention_mask']
)

print(outputs['similarities']) # Cosine similarity matrix
```

2.2. Late Interaction (ColBERT / MaxSim)
By enabling `use_late_interaction`, the model computes **MaxSim** scores (interaction between all token embeddings) instead of standard Cosine similarity of pooled embeddings.

This is ideal for models like **LFM2-ColBERT**, but it works with any model.

```python
from mlx_raclate import load

# Load a ColBERT-style model
model, tokenizer = load(
    "LiquidAI/LFM2-ColBERT-350M", 
    pipeline="sentence-similarity",
    model_config={"use_late_interaction": True} # <--- Enables MaxSim
)

queries = ["Who creates liquid neural networks?"]
docs = ["Liquid AI is a company founded by researchers from MIT..."]

# Tokenize
q_input = tokenizer(queries, return_tensors="mlx", padding=True)
d_input = tokenizer(docs, return_tensors="mlx", padding=True)

# The model keeps embeddings unpooled and computes MaxSim
outputs = model(
    input_ids=q_input['input_ids'],
    reference_input_ids=d_input['input_ids'],
    attention_mask=q_input['attention_mask'],
    reference_attention_mask=d_input['attention_mask']
)

print("MaxSim Scores:", outputs['similarities'])
```

## Pipelines Reference

When using `load()`, the `pipeline` argument determines the class and return types. If not provided, `mlx-raclate` attempts to infer it from the `config.json`.

| Pipeline | Class | Output | Use Case |
| :--- | :--- | :--- | :--- |
| `embeddings` | `Model` | Raw Embeddings (normalized)| Feature extraction |
| `text-classification` | `ModelForSequenceClassification` | Logits/Probs | Sentiment, Intent, Regression |
| `sentence-similarity` | `ModelForSentenceSimilarity` | Embeddings & Similarity | Semantic Search, RAG |
| `sentence-transformers` | `ModelForSentenceTransformers` | Embeddings & Similarity | Same as `sentence-similarity` but  different sanitization strategy for Sentence Transformers weights |
| `masked-lm` | `ModelForMaskedLM` | Token Logits | Domain adaptation, MLM training |
| `token-classification` | `ModelForTokenClassification` | Token Logits | NER tasks | 
| `zero-shot-classification` | `ModelForMaskedLM` | Token Logits | Implementation of [this AnswerAI paper](https://arxiv.org/html/2502.03793v2) |

Runnable templates used by both tests and generated model cards live in
`mlx_raclate.inference_examples`.

## Server

Install the `serve` extra and start the single-worker FastAPI service with:

```bash
raclate serve --host 127.0.0.1 --port 8000
```

The token-classification model itself does not accept an `operating_point`;
its forward pass only produces logits and probabilities. Direct library
inference decodes those outputs with the checkpoint's `default` Viterbi
calibration.

The HTTP server adds a separate decoding layer after model inference. That
server layer can select any named operating point stored in the checkpoint's
`viterbi_calibration.json`:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "trained_models/finance_ner/checkpoint-4000",
    "pipeline": "token-classification",
    "text": "Email alice@example.com",
    "decode_mode": "viterbi",
    "operating_point": "high_precision"
  }'
```

`operating_point` is therefore a server request option, not a model argument.
Use `"decode_mode": "argmax"` for unstructured greedy decoding. Named points
require a corresponding calibration-file entry; unavailable names return HTTP
400 rather than silently falling back.

## NER evaluation and decoder calibration

These commands are a post-training workflow for labelled token-classification
data; they are not required for ordinary model inference. Evaluate a checkpoint
and write a metrics report with:

```bash
raclate ner evaluate \
  --model trained_models/my_ner_run/checkpoint-4000 \
  --dataset data/my_ner_dataset \
  --output eval/my_ner_run.json
```

For BIO/BIOES models, calibration can search server-side Viterbi decoder
profiles directly from labelled calibration data in one command:

```bash
raclate ner calibrate \
  --model trained_models/my_ner_run/checkpoint-4000 \
  --dataset data/my_ner_calibration
```

The command uses temporary logits internally and writes
`viterbi_calibration.json` into the local checkpoint. It does not modify model
weights or the model-forward API. It adds named decoder profiles for the HTTP
server while leaving `default` unchanged unless `--set-default` is supplied.
The advanced persistent-cache workflow and guidance on keeping calibration and
final test data separate are documented in
[TUNER.md](src/mlx_raclate/tuner/TUNER.md#post-training-bio-viterbi-calibration).

## Training (Tuner)

`mlx-raclate` includes a robust training engine specifically designed for fine-tuning these architectures on Apple Silicon.

Install the `train` extra, then inspect the supported CLI options with
`raclate train --help`.

It supports:
*   **Full Fine-tuning** (LoRA is not currently supported/needed for these model sizes).
*   **Tasks:** Text Classification, Sentence Similarity (Bi-Encoder & Late Interaction), and Masked LM.
*   **Efficiency:** Gradient Accumulation, Gradient Checkpointing, and Smart Collation.

For detailed training documentation, supported datasets, and CLI usage, please see [TUNER.md](src/mlx_raclate/tuner/TUNER.md).

### Quick Training Snippet

```python
from mlx_raclate.tuner.trainer import Trainer, TrainingArgs
from mlx_raclate import load

# Load model
model, tokenizer = load("Qwen/Qwen3-Embedding-0.6B", pipeline="text-classification", train=True)

# Define Args
args = TrainingArgs(
    output_dir="outputs/my_classifier",
    learning_rate=1e-5,
    num_train_epochs=3,
    batch_size=4
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    task_type="text-classification",
    training_args=args,
    train_dataset=train_dataset, # See TUNER.md for dataset formatting
    eval_dataset=eval_dataset
)

trainer.train()
```

## Acknowledgements

*   [MLX](https://github.com/ml-explore/mlx) team for the framework.
*   [Transformers](https://github.com/huggingface/transformers) for the configuration standards.
*   [MLX-Embeddings](https://github.com/Blaizzy/mlx-embeddings) for inspiration on broader embedding architectures. MLX-Raclate focuses on longer-context models but you should definitely look there for BERT, XLM_RoBERTa and image embeddings.
*   [PyLate](https://github.com/lightonai/pylate) for inspiration on Late Interaction mechanics.
