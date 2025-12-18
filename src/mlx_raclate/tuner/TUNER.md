# RACLATE Tuner

The `tuner` module is the training engine of RACLATE (**R**etrieval **A**nd **C**lassification including with **LATE** interaction models). Built entirely on Apple's [MLX](https://github.com/ml-explore/mlx) framework, it provides a highly efficient, unified interface for fine-tuning [small] Transformer-based classifiers on Apple Silicon.

This trainer supports standard dense retrieval, classification, and masked language modeling, as well as **Late Interaction (ColBERT-style)** training patterns.

## Key Features

*   **Apple Silicon Native:** Fully optimized for M-series chips using MLX.
*   **Full Training:**  Full fine-tuning of pretrained models (_see supported architectures below_). LORA fine-tuning is not supported (yet). The library allows transfer learning, meaning that existing heads can be stripped out of pretrained models (and new heads can be added to base models for specific tasks)
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
*   **T5Gemma-Encoder**: MLX implementation of `google/t5gemma-b-b-ul2`, but only keeping the encoder weights at initialization (the encoder config is merged into the main model config)
*   **LFM2**: MLX implementation of `LiquidAI/LFM2-350M` (Causal/AR) which also supports `LiquidAI/LFM2-ColBERT-350M` when model config file includes `use_late_interaction=True`. These models have a context window of 128k tokens. In training mode, 128k tokens exceeds the RAM capacity of most Apple hardware. _See parameters below to cap sequences to a more reasonable length during training_


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

The `datasets.py` module handles loading (JSONL, Parquet, CSV, HF Hub) and column standardization. If is built on top of HuggingFace's datasets.

### Column Mapping
The trainer looks for specific column names. 

| Task | Required Columns | Description |
| :--- | :--- | :--- |
| **Classification** | `text`, `label` | Input text and target class/score. |
| **Pairs (Sim.)** | `text`, `text_pair` | Anchor and Positive/Candidate. |
| **Triplets** | `text`, `text_pair`, `negative` | Anchor, Positive, Hard Negative. |
| **MLM** | `text` | Raw text for masking. |
| **NER** | `tokens`, `labels` | Pre-tokenized words and aligned tags. |

*Note: For Sentence Similarity, if a `label` column is present with floats, the trainer switches to Regression/MSE loss (e.g., for Cross-Encoders or scored Bi-Encoders).*

You can map your custom dataset fields via `DatasetArgs`.
```
# Load datasets
dataset_args = DatasetArgs(
    data=dataset, # dataset path
    task_type=task_type, 
    text_field="question", # maps column 'question' to 'text'
    text_pair_field="response", # maps column 'response' to 'text_pair'
    negative_field="semantically_different_response" # maps column 'semantically_different_response' to 'negative'
    label_field="classification" # maps column 'classification to 'label'
    test=True # creates a test split, if not already present in the dataset, out of the training set (validation set not affected).
)
```  
See _standardize_column_names() in `datasets.py` for more information on column mapping.

### Text Pairs and Chat Template

For certain tasks like text-classification, you may want to classify how two token sequences (text and text_pair) relate to each other.  

For bi-encoders, it is highly recommended to let the tokenizer combine the text and the text_pair rather than aggregating them manually. This ensures that the correct separation token is used.

```
batch = self.tokenizer(
    texts,
    text_pairs,
    padding="longest",
    truncation=True,
    max_length=self.max_length,
    return_tensors="mlx"
)
```

For some models, you may want to use the chat template that was used to train the model you intend to finetune. For example, LFM2-350M recommends using a chat template.  
If `use_chat_template` is set to True when initializing the training (default False) and if a chat template is available in the tokenizer (do check!), the text and the text_pair values will be combined and text_pair will be set to None.  

You can also force a specific string as separator

```
if text_pairs is not None:
    if getattr(self.tokenizer, "chat_template", None) and self.use_chat_template:
        # This ensures the model sees exactly what it expects for Q&A
        formatted_texts = []
        for prompt, response in zip(texts, text_pairs):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            formatted_texts.append(
                self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
        texts = formatted_texts
        text_pairs = None # Handled by template

    elif self.force_separator is not None:
        # Use the forced separator for decoder models
        texts = [
            f"{t}{self.force_separator}{p}" 
            for t, p in zip(texts, text_pairs)
        ]
        text_pairs = None

```

See DataCollatorForSequenceClassification in `collators.py` for more information on text_pair handling for text-classification.

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