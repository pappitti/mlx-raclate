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


## Supported Tasks & Pipelines

The `Trainer` adapts its logic based on the `task_type` and the specific model class initialized.

### 1. Sentence Similarity (Embedding & Retrieval)
Train models for semantic search, clustering, or RAG.
*   **Task Type:** `sentence-similarity` 
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

### 1. Column Mapping
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

### 2. Text Pairs and Chat Template

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


## Quick Start (Programmatic)

Below is a simplified example of how to set up a training run programmatically without using the CLI.

```python
from mlx_raclate.utils.utils import load
from mlx_raclate.tuner.datasets import load_dataset, DatasetArgs
from mlx_raclate.tuner.trainer import Trainer, TrainingArgs

# 1. Configuration variables
model_path = "Qwen/Qwen3-Embedding-0.6B"
dataset_path = "data/wines"
task_type = "text-classification"

# 2. Load and Prepare Dataset
dataset_args = DatasetArgs(
    data=dataset_path, 
    task_type=task_type,
    # Optional: override field names if your data isn't standard
    # text_field="text",
    # label_field="label"
)
train_ds, valid_ds, test_ds, id2label, label2id = load_dataset(dataset_args)

# 3. Load Model and Tokenizer
# Pass label mappings to model config for classification tasks
model_config = {"id2label": id2label, "label2id": label2id} if id2label else {}

model, tokenizer = load(
    model_path, 
    model_config=model_config, 
    pipeline=task_type,
    train=True
)

# 4. Define Training Arguments
args = TrainingArgs(
    output_dir="outputs/my_run",
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=3,
    lr_scheduler_type="cosine_decay",
    warmup_ratio=0.03,
    save_steps=500,
    logging_steps=10,
    max_length=2048,
    freeze_embeddings=False
)

# 5. Initialize Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    task_type=task_type,
    training_args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    label2id=label2id,
    # For decoder models doing classification on pairs:
    use_chat_template=False 
)

# 6. Run Training
trainer.train()

# 7. Evaluate on Test Set (Optional)
if test_ds:
    trainer.test(test_ds)
```

---

## API Reference

### DatasetArgs

Used to configure how data is loaded and mapped.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `str` | *Required* | Local path or HF identifier of the dataset. |
| `task_type` | `str` | *Required* | The type of task (e.g., `text-classification`). |
| `text_field` | `str` | `None` | Name of the text input column. |
| `text_pair_field`| `str` | `None` | Name of the second text input column (for pairs). |
| `label_field` | `str` | `None` | Name of the label/target column. |
| `negative_field`| `str` | `None` | Name of the negative samples column. |
| `test` | `bool` | `False` | If True, creates a test split from the training set if one doesn't exist. |. 

Note : use load_dataset("dataset_path") from `datasets.py` to fetch the dataset splits and label2id

### TrainingArgs

Controls the hyperparameters and runtime configuration.

#### Hyperparameters
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `batch_size` | `2` | The batch size per device/step. |
| `gradient_accumulation_steps` | `8` | Number of steps to accumulate gradients before updating weights. |
| `num_train_epochs` | `2` | Total number of training epochs. |
| `max_length` | `512` | Max sequence length. If `None`, uses model's default config. |
| `freeze_embeddings` | `False` | If `True`, freezes the embedding layer to save memory/compute. |

#### Optimizer & Scheduler
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `learning_rate` | `3e-5` | Initial learning rate (Peak LR). |
| `weight_decay` | `0.01` | Weight decay factor for AdamW. |
| `lr_scheduler_type` | `"constant"` | Scheduler type: `"cosine_decay"`, `"linear_schedule"`, or `"constant"`. |
| `min_lr` | `0.0` | Minimum learning rate at the end of the schedule. |
| `warmup_ratio` | `0.0` | Ratio of total training steps used for warmup. |
| `warmup_steps` | `0` | Absolute number of warmup steps (overrides `warmup_ratio` if set). |
| `max_grad_norm` | `1.0` | Gradient clipping threshold. |

#### Checkpointing & Logging
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `output_dir` | `None` | Directory to save checkpoints and logs. Defaults to a timestamped folder. |
| `save_steps` | `1000` | Frequency of saving model checkpoints (in steps). |
| `logging_steps` | `16` | Frequency of logging metrics to console/files. |
| `eval_batch_size` | `4` | Batch size used during evaluation/testing. |
| `resume_from_step`| `0` | Step to resume training from. |  

Gradient checkpointing is enabled by default

### Model Config

When loading a pretrained model, you can create a model_config dictionary with new parameters and pass it to the load() function. Examples 

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `is_regression` | `bool` | `False` | For text-classification tasks, whether the classification is a regression |
| `use_late_interaction`| `bool` | `False` | For sentence similarity tasks, whether late interaction (MaxSim) should be used instead of cosine similarity |

## Trainer

The main class that orchestrates the training.

**Constructor Parameters:**

*   **`model`**: The loaded MLX model.
*   **`tokenizer`**: The loaded tokenizer. If you want to use a chat template, make sure that the tokenizer includes the chat template. If not, add it manually before instantiating the Trainer.
*   **`task_type`**: String identifier for the pipeline (e.g., "text-classification").
*   **`training_args`**: Instance of `TrainingArgs`.
*   **`train_dataset`**: The processed training dataset.
*   **`eval_dataset`**: (Optional) The processed validation dataset.
*   **`label2id`**: (Optional) Dictionary mapping labels to IDs (required for classification metrics).
*   **`use_chat_template`** *(bool)*: If `True`, applies the tokenizer's chat template to inputs. Useful for decoder models (like Qwen/Llama) performing classification on text pairs.
*   **`force_separator`** (Optional *str*): If not using a chat template, this string is used to join text pairs for decoder models.
*   **`optimizer`** *(Optional *mlx.optimizer*): If no optimizer is passed, AdamW will be used with the hyper parameters set in TrainingArgs 

**Methods:**

*   `train()`: Starts the training loop.
*   `test(dataset)`: Runs evaluation on the provided dataset.

