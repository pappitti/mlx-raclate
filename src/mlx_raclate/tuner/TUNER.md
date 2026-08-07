# RACLATE Tuner

The `tuner` module is the training engine of RACLATE (**R**etrieval **A**nd **C**lassification including **LATE** interaction models). Built entirely on Apple's [MLX](https://github.com/ml-explore/mlx) framework, it provides a highly efficient, unified interface for fine-tuning *small* Transformer-based classifiers on Apple Silicon.

This trainer supports standard dense retrieval, classification, and masked language modeling, as well as **Late Interaction (ColBERT-style)** training patterns.

## Key Features

*   **Apple Silicon Native:** Fully optimized for M-series chips using MLX.
*   **Full Training:**  Full fine-tuning of pretrained models (_see supported architectures below_). LORA fine-tuning is not supported (yet). The library allows transfer learning, meaning that existing heads can be stripped out of pretrained models (and new heads can be added to base models for specific tasks)
*   **Memory Efficiency:** Built-in support for **Gradient Accumulation**, **Gradient Checkpointing**, and configurable MLX cache clearing to train larger batches/models on limited Unified Memory.
*   **Flexible Schedulers:** Linear, Cosine, and Constant learning rate schedules with warmup.
*   **Smart Collators:** Task-specific data collators that handle padding, masking, and chat templates automatically.
*   **Embedding Freezing:** Option to freeze embedding layers to speed up fine-tuning or prevent catastrophic forgetting.
*   **HF Hub Integration (TODO):** Seamless saving and pushing of checkpoints to the Hugging Face Hub.

## Supported Architectures

The trainer supports a variety of modern architectures supporting long context (relative to BERT models). As these models are meant to be trained and run on local machines, model implementations are specifically optimized for small-to-mid-sized models:

*   **ModernBERT**: MLX implementation of `answerdotai/ModernBERT-base` (encoder-only). Long context (8k) and high efficiency.
*   **Qwen 3**: MLX implementation of `Qwen/Qwen3-Embedding-0.6B` (32k context window) which leverages the qwen3 architecture.
*   **Gemma 3**: MLX implementation of `google/embeddinggemma-300m` (2k context window) which leverages the gemma3 text variant architecture with a few tweaks. As per the official embeddinggemma3 architecture, the attention mask is set to causal or bi-directional based on a config parameter (`use_bidirectional_attn` or `use_bidirectional_attention`). Therefore, it is possible to switch between encoder and decoder mode, and standard gemma3_text models (32k context window) are also supported. 
*   **T5Gemma-Encoder**: MLX implementation of `google/t5gemma-b-b-ul2`, but only keeping the encoder weights at initialization (the encoder config is merged into the main model config)
*   **LFM2**: MLX implementation of `LiquidAI/LFM2-350M` (Causal/AR) which also supports `LiquidAI/LFM2-ColBERT-350M` when model config file includes `use_late_interaction=True`. These models have a context window of 128k tokens. In training mode, 128k tokens exceeds the RAM capacity of most Apple hardware. _See parameters below to cap sequences to a more reasonable length during training_
*   **Bidirectional LFM2.5**: MLX implementation of `LiquidAI/LFM2.5-Encoder-350M` for masked-LM training and `LiquidAI/LFM2.5-Embedding-350M` for CLS-pooled dense retrieval. The loader detects LiquidAI's `Lfm2Bidirectional*` architecture metadata automatically.


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
*   **Features:** Handles label alignment for sub-word tokens automatically. The trainer is label-scheme agnostic: BIO, BIOES/BILOU, flat tags, or integer class ids all work as long as there is one label per input word.


## Data Preparation

The `datasets.py` module handles loading (JSONL, Parquet, CSV, HF Hub) and column standardization. It is built on top of HuggingFace's datasets.

### 1. Column Mapping
The trainer looks for specific column names. 

| Task | Required Columns | Description |
| :--- | :--- | :--- |
| **Classification** | `text`, `label` | Input text and target class/score. |
| **Pairs (Sim.)** | `text`, `text_pair` | Anchor and Positive/Candidate. |
| **Triplets** | `text`, `text_pair`, `negative` | Anchor, Positive, Hard Negative. |
| **MLM** | `text` | Raw text for masking. |
| **NER** | `text`, `labels` | Pre-tokenized words and aligned tags. HF-style `tokens`/`ner_tags` columns are automatically renamed to `text`/`labels`. |

*Note: For Sentence Similarity, if a `label` column is present with floats, the trainer switches to Regression/MSE loss (e.g., for scored Bi-Encoders).*

You can map your custom dataset fields via `DatasetArgs`.
```
# Load datasets
dataset_args = DatasetArgs(
    data=dataset, # dataset path
    task_type=task_type, 
    text_field="question", # maps column 'question' to 'text'
    text_pair_field="response", # maps column 'response' to 'text_pair'
    negative_field="semantically_different_response", # maps column 'semantically_different_response' to 'negative'
    label_field="classification", # maps column 'classification to 'label'
    test=True # creates a test split, if not already present in the dataset, out of the training set (validation set not affected).
)
```  
Anchor is automatically mapped to `text` and Positive is automatically mapped to `text_pair`. See _standardize_column_names() in `datasets.py` for more information on column mapping. 

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

In some cases, you may want to use the chat template that was used to train the model you intend to finetune. For example, LFM2-350M recommends using a chat template.  

If `use_chat_template` is set to True when initializing the training (default False) and if a chat template is available in the tokenizer (do check!), the text and the text_pair values will be combined and text_pair will be set to None.  

You can also force a specific string as separator.  

This is how it works under the hood:

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

Below is a simplified example of how to set up a training run programmatically.

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
    # text_field="question",
    # text_pair_field="response",
    # label_field="classification"
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

## CLI usage 

An example of CLI tool including **all** parameters to train a model is available in `mlx_raclate.utils.train.py`.  

WARNING : this example includes default values that override the default values of the DatasetArgs, TrainingArgs and Trainer classes presented below.

### NER evaluation

Use `tools/evaluate_ner_models.py` to compare saved token-classification
checkpoints. It reports token accuracy plus macro/micro precision, recall, and
F1 at both token level and exact entity-boundary level. Entity true-positive,
false-positive, false-negative, support, precision, recall, and F1 values are
also included per entity type in the JSON output.

For a normal dataset directory containing `validation.parquet`:

```bash
uv run python tools/evaluate_ner_models.py \
  --model trained_models/my_ner_run/checkpoint-4000 \
  --dataset data/my_ner_dataset \
  --output eval/my_ner_run.json
```

A multi-phase root is discovered automatically. The following evaluates both
`generalization/validation.parquet` and
`lexical_calibration/validation.parquet` for every checkpoint:

```bash
uv run python tools/evaluate_ner_models.py \
  --model \
    trained_models/finance_ner/checkpoint-4000 \
    trained_models/finance_ner/checkpoint-6000 \
  --dataset data/NER-Finance-CRM/bioes_pplx_embed_v1_0.6b \
  --decode both \
  --output eval/finance_ner_checkpoints.json
```

`--decode viterbi` is the default and enforces valid BIO/BIOES transitions,
using the checkpoint's `viterbi_calibration.json` when present. Use
`--decode argmax` for raw greedy scores or `--decode both` to compare them.
Add `--limit N` for a quick smoke test. Existing output files are protected
unless `--overwrite` is passed. Dataset labels are resolved from ClassLabel
metadata or a nearby `labels.json` and remapped to model IDs by name; an
incompatible dataset/model ontology fails before inference rather than
silently producing invalid scores.

## API Reference

### DatasetArgs

Used to configure how data is loaded and mapped.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `str` | *Required* | Local path or HF identifier of the dataset. |
| `task_type` | `str` | *Required* | The type of task (e.g., `text-classification`). |
| `text_field` | `str` | `"text"` | Name of the text input column. |
| `text_pair_field`| `str` | `None` | Name of the second text input column (for pairs). |
| `label_field` | `str` | `"label"` | Name of the label/target column. |
| `negative_field`| `str` | `None` | Name of the negative samples column. |
| `train_limit` | `int` | `None` | Optional cap on training examples after loading and split creation. |
| `validation_limit` | `int` | `None` | Optional cap on validation examples after loading and split creation. |
| `test_limit` | `int` | `None` | Optional cap on test examples after loading and split creation. |
| `test` | `bool` | `False` | If True, creates a test split from the training set if one doesn't exist. |   

Note : use load_dataset("dataset_path") from `datasets.py` to fetch the dataset splits and the label2id dictionary.

### TrainingArgs

Controls the hyperparameters and runtime configuration.

#### Hyperparameters
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `batch_size` | `2` | The physical batch size per device/step. Reduce to 1 on Macbooks with limited RAM if training on long context. |
| `gradient_accumulation_steps` | `8` | Number of steps to accumulate gradients before updating weights. |
| `gradient_accumulation_dtype` | `"float32"` | Accumulation precision: `"float32"` is recommended for FP16/BF16 models; `"model"` uses the gradient dtype. |
| `multi_precision` | `True` | Keep FP32 master weights and AdamW moments when fully training an FP16/BF16 model. |
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
| `output_dir` | `"outputs"` | Directory name under `trained_models/` for checkpoints and logs. |
| `save_steps` | `1000` | Frequency of saving model checkpoints (in steps). |
| `logging_steps` | `100` | Frequency of logging metrics to console/files. |
| `eval_batch_size` | `4` | Batch size used during evaluation/testing. |
| `eval_strategy` | `"end"` | When to run validation: `"end"` for the final epoch only, `"epoch"` after every epoch, or `"no"` to skip validation during training. |
| `resume_from_checkpoint` | `None` | Complete checkpoint containing model, optimizer, scheduler step, and trainer position. This is the recommended resume mechanism. `num_train_epochs` remains the total target. |
| `resume_from_step`| `0` | Legacy data-skipping resume without Adam moments; use only when a complete checkpoint is unavailable. |
| `grad_checkpoint` | `True` | Whether to wrap transformer layers in MLX gradient checkpointing to reduce activation memory. |
| `cache_clear_steps` | `1` | Clear MLX cache every N optimizer updates. `1` is safest for large/MoE models; higher values can improve speed but may increase process footprint. |

Gradient checkpointing is enabled by default due to RAM constraints of consumer hardware. It is applied at each backbone block call through `GradientCheckpointingMixin`, using MLX's module-aware `mlx.nn.utils.checkpoint` helper so the block's trainable parameters participate in recomputation. A new backbone must inherit that mixin and invoke blocks through `_call_checkpointed_layer`; the trainer fails explicitly if no compatible backbone is present. MLX cache clearing is also aggressive by default because unified-memory footprint can be much larger than `mx.get_active_memory()` for some architectures.

### Model Config

When loading a pretrained model, you can create a model_config dictionary with new parameters and pass it to the load() function. Common examples :

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `is_regression` | `bool` | `False` | For text-classification tasks, whether the classification is a regression |
| `use_late_interaction`| `bool` | `False` | For sentence similarity tasks, whether late interaction (MaxSim) should be used instead of cosine similarity |
| `dtype` | `str` | `"auto"` | Persisted model parameter dtype: `"auto"`, `"float32"`, `"float16"`, or `"bfloat16"`. Canonical `dtype` takes precedence over legacy `torch_dtype`. |

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
*   **`optimizer`** (Optional *mlx.optimizer*): If no optimizer is passed, AdamW will be used with the hyper parameters set in TrainingArgs 

**Methods:**

*   `train()`: Starts the training loop.
*   `test(dataset)`: Runs evaluation on the provided dataset.
