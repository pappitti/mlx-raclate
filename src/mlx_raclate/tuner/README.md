# RACLATE Tuner

The `tuner` module is the training engine of RACLATE (**R**etrieval **A**nd **C**lassification including with **LATE** interaction models). Built entirely on Apple's [MLX](https://github.com/ml-explore/mlx) framework, it provides a highly efficient, unified interface for fine-tuning [small] Transformer-based classifiers on Apple Silicon.

This trainer supports standard dense retrieval, classification, and masked language modeling, as well as **Late Interaction (ColBERT-style)** training patterns.

## Key Features

*   **Apple Silicon Native:** Fully optimized for M-series chips using MLX.
*   **Full Training:**  Pretraining (not recommended) and full fine-tuning of pretrained models (_see supported architectures below_). LORA fine-tuning is not supported (yet). 
*   **Memory Efficiency:** Built-in support for **Gradient Accumulation** and **Gradient Checkpointing** to train larger batches/models on limited Unified Memory.
*   **Flexible Schedulers:** Linear, Cosine, and Constant learning rate schedules with warmup.
*   **Smart Collators:** Task-specific data collators that handle padding, masking, and chat templates automatically.
*   **Embedding Freezing:** Option to freeze embedding layers to speed up fine-tuning or prevent catastrophic forgetting.
*   **HF Hub Integration (TODO):** Seamless saving and pushing of checkpoints to the Hugging Face Hub.

## Supported Architectures

The trainer supports a variety of modern architectures supporting long context (relative to BERT models). As these models are meant to be trained and run on local machines, model implementations are specifically optimized for small-to-mid-sized models:

*   **ModernBERT**: MLX implementation of `answerdotai/ModernBERT-base` (encoder-only). Long context (8k) and high efficiency.
*   **Qwen 3**: MLX implementation of `Qwen/Qwen3-Embedding-0.6B` (32k context window) which leverages the qwen3 
*   **Gemma 3**: MLX implementation of `google/embeddinggemma-300m` (2k context window) which leverages the gemma3 text variant architecture with a few tweaks. As per the official embeddingggemma3 architecture, the attention mask is set to causal or bi-directional based on a config parameter (`use_bidirectional_attn` or `use_bidirectional_attention`). Therefore, it is possible to switch between encoder and decoder mode, and standard gemma3_text models (32k context window) are also supported. 
architecture.
*   **T5Gemma-Encoder**: MLX implementation of `google/t5gemma-b-b-ul2`, but only keeping the encoder weights at initialization (the encoder config is merged into the main model config)
*   **LFM2 (Lightweight Foundation Model 2)**: (Causal/AR).



## 🛠 Supported Tasks & Pipelines

The `Trainer` adapts its logic based on the `task_type` and the specific model class initialized.

### 1. Sentence Similarity (Embedding & Retrieval)
Train models for semantic search, clustering, or RAG.
*   **Task Type:** `sentence-similarity` or `sentence-transformers`
*   **Training Modes:**
    *   **Bi-Encoder (Dense):** Standard cosine similarity optimization.
    *   **Late Interaction (MaxSim):** ColBERT-style interaction where fine-grained token-level similarities are computed (requires `use_late_interaction=True`).
*   **Loss Functions:** Automatically selects between **MNRL (Multiple Negatives Ranking Loss)** for triplets/pairs or **MSE/Cosine Loss** for scored pairs.

### 2. Sequence Classification
Train discriminative models for sentiment analysis, intent detection, etc.
*   **Task Type:** `text-classification`
*   **Features:**
    *   Supports Multi-class and Binary classification.
    *   Supports Regression (if `is_regression=True`).
    *   Native support for Chat Templates in tokenizer.

### 3. Masked Language Modeling (MLM)
Perform domain adaptation on raw text.
*   **Task Type:** `masked-lm`
*   **Features:** Implements the standard 80% mask / 10% random / 10% original masking strategy dynamically during training.

### 4. Token Classification (NER/POS)
Named Entity Recognition and Part-of-Speech tagging.
*   **Task Type:** `token-classification`
*   **Features:** Handles label alignment for sub-word tokens automatically.

## 📊 Data Preparation

The `datasets.py` module handles loading (JSONL, Parquet, CSV, HF Hub) and column standardization.

### Column Mapping
The trainer looks for specific column names. You can map your custom dataset fields via `DatasetArgs`.

| Task | Required Columns | Description |
| :--- | :--- | :--- |
| **Classification** | `text`, `label` | Input text and target class/score. |
| **Pairs (Sim.)** | `text`, `text_pair` | Anchor and Positive/Candidate. |
| **Triplets** | `text`, `text_pair`, `negative` | Anchor, Positive, Hard Negative. |
| **MLM** | `text` | Raw text for masking. |
| **NER** | `tokens`, `labels` | Pre-tokenized words and aligned tags. |

*Note: For Sentence Similarity, if a `label` column is present with floats, the trainer switches to Regression/MSE loss (e.g., for Cross-Encoders or scored Bi-Encoders).*

## 🚀 Usage Example

Here is a simplified example of how to set up a training run programmatically:

```python
from tuner.trainer import Trainer, TrainingArgs
from tuner.datasets import load_dataset, DatasetArgs
from tuner.modernbert import ModelForSentenceSimilarity, ModelArgs
from transformers import AutoTokenizer

# 1. Setup Data
data_args = DatasetArgs(
    data="my_org/my_dataset",
    task_type="sentence-similarity",
    train=True,
    test=True
)
train_ds, val_ds, test_ds, _, _ = load_dataset(data_args)

# 2. Setup Model & Tokenizer
model_config = ModelArgs(
    model_type="modernbert",
    use_late_interaction=False # Set True for ColBERT style
)
model = ModelForSentenceSimilarity(model_config)
# Load weights... (custom logic usually required here to load from safetensors)

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

# 3. Setup Training Arguments
args = TrainingArgs(
    output_dir="modernbert-retrieval-v1",
    batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=3,
    gradient_accumulation_steps=4, # Crucial for Mac memory management
    grad_checkpoint=True,          # Saves memory at cost of compute
    push_to_hub=False
)

# 4. Train
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    task_type="sentence-similarity",
    training_args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds
)

trainer.train()
```

## 🧠 Advanced Concepts

### Late Interaction
RACLATE treats Late Interaction as a first-class citizen. In `base.py`, the `compute_similarity_and_loss` function handles the "MaxSim" operation:
$$ S(Q, D) = \sum_{i \in Q} \max_{j \in D} (q_i \cdot d_j) $$
This allows for cheap indexing with high-precision retrieval. To enable this, set `use_late_interaction=True` in your model config.

### Gradient Accumulation & Checkpointing
To train models like Gemma-3 or ModernBERT on MacBooks with 16GB or 24GB of RAM:
1.  **Gradient Checkpointing:** Re-computes the forward pass during backprop to save activation memory.
2.  **Gradient Accumulation:** Simulates a larger batch size by accumulating gradients over multiple steps before updating weights.

Both are enabled via `TrainingArgs`.