import time
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from textwrap import dedent
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers
from datasets import Dataset as HFDataset

from mlx.utils import tree_flatten, tree_map

from .collators import DataCollator

@dataclass
class TrainingArgs:

    def __init__(
        self,
        batch_size: int = 32,
        eval_batch_size: int = 16,
        max_length: int = 512,
        num_train_epochs: int = 3,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 1,
        eval_steps: int = 500,
        save_steps: int = 1000,
        logging_steps: int = 100,
        output_dir: str = "outputs",
        save_total_limit: Optional[int] = None,
        grad_checkpoint: bool = True,
        push_to_hub: bool = False,
    ):
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_length = max_length
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio ### not used here but kept for later (see scheduler in utils)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.output_dir = output_dir
        self.save_total_limit = save_total_limit
        self.grad_checkpoint = grad_checkpoint ### mat not be necessary but helps anticipating hardware constraints
        self.push_to_hub = push_to_hub 

class Trainer:
    """
    A trainer for ModernBERT that adapts to the model's training objective.
    The training logic is determined by the model's class implementation.
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        task_type: str,
        training_args: TrainingArgs,
        train_dataset: HFDataset,
        use_chat_template: bool = False, # for decoder-based models, you may want to use chat templates when preparing the data
        force_separator: Optional[str] = None, # for decoder-based models, you may want to force a specific separator when preparing the data
        eval_dataset: Optional[HFDataset] = None,
        optimizer = None
    ):
        self.model = model
        self.tokenizer = tokenizer._tokenizer ### tokenizer is a wrapper around the HF tokenizer (see utils/tokenizer_utils.py)
        self.task_type = task_type
        self.args = training_args
        self.train_dataset = train_dataset
        self.use_chat_template = use_chat_template
        self.force_separator = force_separator
        self.eval_dataset = eval_dataset
        self.data_collator = self._get_collator()
        
        # Initialize optimizer
        self.optimizer = optimizer or mlx.optimizers.AdamW(
            learning_rate=training_args.learning_rate,
            weight_decay=training_args.weight_decay
        )

        # Setup output directory
        self.output_dir = Path(training_args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Capture state that needs updating (random state for Dropout, etc.)
        self.state = [self.model.state, self.optimizer.state, mx.random.state]
        
        # Setup training state and output directory
        self.global_step = 0
        self.epoch = 0

        # Capture state that needs updating (random state for Dropout, etc.)
        self.state = [self.model.state, self.optimizer.state, mx.random.state]
        
        # Enable gradient checkpointing if requested
        if training_args.grad_checkpoint:
            self._apply_grad_checkpointing()
            
        def loss_fn(model, batch):
            outputs = model(**batch)
            return mx.mean(outputs["loss"])
        
        grad_fn = nn.value_and_grad(self.model, loss_fn)

        @partial(mx.compile, inputs=self.state, outputs=self.state)
        def step_calc(batch):
            loss, grads = grad_fn(self.model, batch)
            return loss, grads

        self.step_calc = step_calc

        # Optimizer Update Function
        # We define a function that takes the model and ACCUMULATED grads
        @partial(mx.compile, inputs=self.state, outputs=self.state)
        def update_fn(grads):
            self.optimizer.update(self.model, grads)
        
        self.step_update = update_fn
        self.push_to_hub = training_args.push_to_hub 
        
        print(f"Training {model.__class__.__name__}")
        # Log model type and config           
        self._save_config()

    def _apply_grad_checkpointing(self):
        """
        Apply gradient checkpointing to the model's forward pass to reduce memory usage.
        Uses MLX's checkpoint mechanism to save memory during backpropagation.
        """
        def checkpoint_fn(module):
            original_call = module.__call__

            def checkpointed_call(self, **kwargs):
                # Let MLX handle the parameter management, just checkpoint the function call
                return mx.checkpoint(original_call)(self, **kwargs)

            module.__call__ = checkpointed_call
        
        # Checkpoint transformer layers - these are the main memory users
        for layer in self.model.model.layers:
            checkpoint_fn(layer)

        ### TODO : optionally checkpoint other layers  (head, classifier) 


    def _compute_loss(self, batch_inputs): 
        """Compute the loss for training"""
        outputs = self.model(**batch_inputs)
        return mx.mean(outputs["loss"])
    
    def _get_collator(self) -> DataCollator:
        if self.task_type == "masked-lm":
            from .collators import DataCollatorForMaskedLanguageModeling
            return DataCollatorForMaskedLanguageModeling(
                tokenizer=self.tokenizer, 
                max_length=self.args.max_length
            )
        elif self.task_type == "text-classification":
            from .collators import DataCollatorForSequenceClassification
            # For decoder-based models:
            # the collator will apply chat template in priority if specified
            # if not, it will force the separator if specified
            # if not, it will use the tokenizer default
            return DataCollatorForSequenceClassification(
                tokenizer=self.tokenizer, 
                max_length=self.args.max_length,
                use_chat_template=self.use_chat_template,
                force_separator=self.force_separator
            )
        elif self.task_type == "token-classification":
            from .collators import DataCollatorForTokenClassification
            return DataCollatorForTokenClassification(
                tokenizer=self.tokenizer, 
                max_length=self.args.max_length
            )
        elif self.task_type == "sentence-similarity" or self.task_type == "sentence-transformers":
            from .collators import DataCollatorForSentenceSimilarity
            return DataCollatorForSentenceSimilarity(
                tokenizer=self.tokenizer, 
                max_length=self.args.max_length
            )
        # TODO : Add other tasks & collators if needed
        raise ValueError(f"No collator defined for {self.task_type}")


    def _create_batches(self, dataset, batch_size, shuffle=False, seed=42):
        """
        Iterates over HF dataset, slices it, and passes to collator.
        """
        data_len = len(dataset)
        
        # Use HF dataset's efficient shuffle which works with memory mapping
        if shuffle:
            dataset = dataset.shuffle(seed=seed) 
            
        # Standard iteration
        for start_idx in range(0, data_len, batch_size):
            end_idx = min(start_idx + batch_size, data_len)
            batch_slice = dataset[start_idx:end_idx]
            # HF Dataset slicing returns a Dict of lists: {'text': ['a', 'b'], 'label': [0, 1]}
            # Convert HF Columnar batch (Dict[str, List]) to MLX batch (Dict[str, mx.array])
            yield self.data_collator(batch_slice)

    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        for epoch in range(self.args.num_train_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.args.num_train_epochs}")
            self._train_epoch()
            
            if self.eval_dataset is not None:
                print(f"Evaluating after epoch {self.epoch + 1}...")
                metrics = self.evaluate()
                self._save_checkpoint(metrics)
            else:
                # Save checkpoint even if no eval dataset is provided
                print(f"Saving checkpoint after epoch {self.epoch + 1} without evaluation...")
                self._save_checkpoint({})

    def _train_epoch(self):
        """Training logic for one epoch."""
        self.model.train()
        running_loss = 0
        n_steps = 0
        start_time = time.time()
        
        # Accumulation container
        accumulated_grads = None
        steps_to_accumulate = self.args.gradient_accumulation_steps
        scale_factor = 1.0 / steps_to_accumulate if steps_to_accumulate > 1 else 1.0

        # ensures different shuffling each epoch
        current_seed = 42 + self.epoch 

        for batch in self._create_batches(self.train_dataset, self.args.batch_size, shuffle=True, seed=current_seed):
            
            # Calculate Grads
            loss, grads = self.step_calc(batch)

            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(lambda x, y: x + y, accumulated_grads, grads)
            
            # depending on hardware and model size, we may want to avoid syncing here
            running_loss += loss.item() # running_loss += loss to avoid sync
            n_steps += 1
            self.global_step += 1

            # Update Optimizer if Accumulation Done
            if n_steps % steps_to_accumulate == 0:

                # Scale Grads for Accumulation (only once per accumulation cycle)
                if steps_to_accumulate > 1:
                    accumulated_grads = tree_map(lambda g: g * scale_factor, accumulated_grads)

                # Apply updates
                self.step_update(accumulated_grads)

                ## can be used for debugging/logging (expensive)
                # grads_for_logging = accumulated_grads if (self.global_step + 1) % self.args.logging_steps == 0 else None

                # Reset
                accumulated_grads = None
                
                # Eval state to actually trigger the computation graph
                mx.eval(self.model.state, self.optimizer.state)
            
                if self.global_step % self.args.logging_steps == 0:
                    # if running_loss is mx.array (see comment on hardware above), convert to float
                    if isinstance(running_loss, mx.array):
                        running_loss = running_loss.item()
                    avg_loss = running_loss / n_steps

                    # Handle both static float and dynamic schedule
                    current_lr = self.optimizer.learning_rate
                    if isinstance(current_lr, mx.array):
                        current_lr = current_lr.item()

                    ## can be used for debugging/logging (expensive)
                    # grad_norm = 0.0
                    # if grads_for_logging is not None:
                    #     # Flatten the tree of gradients
                    #     flat_grads = tree_flatten(grads_for_logging)
                    #     # Concatenate and compute norm
                    #     # Note: This is an extra computation, but useful for debugging
                    #     grad_vec = mx.concatenate([g.flatten() for g in flat_grads])
                    #     grad_norm = mx.linalg.norm(grad_vec).item()

                    mem_gb = mx.get_active_memory() / 1e9

                    elapsed = time.time() - start_time
                    steps_per_sec = n_steps / elapsed
                    
                    print(
                        f"Step {self.global_step} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Mem: {mem_gb:.1f}GB | Speed: {steps_per_sec:.2f} it/s"
                    )
                    
                    # Reset window counters
                    running_loss = 0.0
                    n_steps = 0
                    start_time = time.time()
        
        return 0.0 # placeholder if we want to use the average loss for anything
    
    def evaluate(self):
        """Evaluation loop."""
        self.model.eval()
        total_loss = 0
        n_steps = 0
        
        for batch in self._create_batches(self.eval_dataset, self.args.eval_batch_size):
            outputs = self.model(**batch)
            loss = mx.mean(outputs["loss"])
            total_loss += loss.item()
            n_steps += 1
        metrics = {"eval_loss": total_loss / n_steps}
        
        print(f"\nEvaluation metrics: {metrics}")
        return metrics
    
    def test(self, test_dataset=None):
        """
        Evaluate the model on the test set after training is complete.
        Args: test_dataset: Optional test dataset. If None, uses self.eval_dataset
        """
        print("\nPerforming final evaluation on test set...")
        
        # Save the model's training state
        training = self.model.training
        self.model.eval()
        total_loss = 0
        n_steps = 0
        
        # Use provided test dataset or fall back to eval dataset
        dataset_to_test = test_dataset or self.eval_dataset
        if dataset_to_test is None:
            raise ValueError("No test dataset provided")
        
        # Perform evaluation
        for batch in self._create_batches(dataset_to_test, self.args.eval_batch_size):
            outputs = self.model(**batch)
            loss = mx.mean(outputs["loss"])
            total_loss += loss.item()
            n_steps += 1
        metrics = {"eval_loss": total_loss / n_steps}
        
        # Save test results
        results_path = self.output_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Test results: {metrics}")
        
        # Restore model's training state
        self.model.train(training)
        
        return metrics
    
    def _save_checkpoint(self, metrics: Dict[str, float]):
        save_path = self.output_dir / f"checkpoint-{self.global_step}"
        save_path.mkdir(exist_ok=True)

        hf_transformers_arch = self.model.get_hf_transformers_arch()
        if hf_transformers_arch:
            self.model.config.architectures = [hf_transformers_arch]

        with open(save_path / "config.json", "w") as f:
            json.dump(self.model.config.__dict__, f, indent=2)

        self.tokenizer.save_pretrained(save_path)
        
        weights = dict(tree_flatten(self.model.trainable_parameters()))
        mx.save_safetensors(str(save_path / "model.safetensors"), weights)
        
        with open(save_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Push to Hub
        if self.args.push_to_hub:
            print("Warning, push to hub is untested")
            # Assumes args.output_dir is the repo name or you have a separate arg
            repo_id = self.args.output_dir.split("/")[-1] # Simple heuristic
            print(f"Pushing to hub: {repo_id}")
            upload_to_hub(
                path=str(save_path),
                upload_repo=repo_id,
                hf_path=self.model.config.model_type, # Or base model name
                task_type=self.task_type
                # TODO : map architecture to HF pipeline for full compatibility
            )
        
        # Manage checkpoint rotation
        if self.args.save_total_limit:
            ### TODO
            raise NotImplementedError("Checkpoint rotation not implemented yet")
            self._rotate_checkpoints()
    
    def _save_config(self):
        """Save training configuration."""
        config = {
            "model_type": self.model.__class__.__name__,
            "training_args": vars(self.args)
        }
        with open(self.output_dir / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)

def upload_to_hub(
        path: str, 
        upload_repo: str, 
        hf_path: str,
        task_type: str,
        ):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (str): Local path to the model.
        upload_repo (str): Name of the HF repo to upload to.
        hf_path (str): Path to the original Hugging Face model.
        task_type (str): Type of task the model was trained on.
    """
    import os

    from huggingface_hub import HfApi, ModelCard, logging

    from . import __version__

    model_path = Path(path)

    card = ModelCard.load(hf_path) if ModelCard.exist_in_hub(hf_path) else ModelCard()
    card.data.tags = ["mlx"] if card.data.tags is None else card.data.tags + ["mlx"] 
    card.data.base_model = hf_path
    card.data.task_type = task_type
    card.text = dedent(
        f"""
        # {upload_repo}

        This model was trained using [MLX Raclate](https://github.com/pappitti/mlx-raclate) for {task_type}.

        ## Usage

        ```python
        TODO
       
        ```
        """
    )
    # Save the model card
    card.save(model_path / "README.md")

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
        multi_commits=True,
        multi_commits_verbose=True,
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")

## COMMENTED OUT FOR NOW (Sharding not needing for small models)
# def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
#     """
#     Splits the weights into smaller shards.

#     Args:
#         weights (dict): Model weights.
#         max_file_size_gb (int): Maximum size of each shard in gigabytes.

#     Returns:
#         list: List of weight shards.
#     """
#     max_file_size_bytes = max_file_size_gb << 30
#     shards = []
#     shard, shard_size = {}, 0
#     for k, v in weights.items():
#         if shard_size + v.nbytes > max_file_size_bytes:
#             shards.append(shard)
#             shard, shard_size = {}, 0
#         shard[k] = v
#         shard_size += v.nbytes
#     shards.append(shard)
#     return shards