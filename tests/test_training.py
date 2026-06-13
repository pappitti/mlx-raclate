"""
Training Tests for mlx-raclate

These tests verify that training works correctly for each (model_family, pipeline)
combination. Tests use dummy datasets and run minimal training steps.

Run with: pytest tests/test_training.py -v --run-slow
"""

import pytest
from datasets import Dataset as HFDataset

from .model_registry import MODEL_FAMILIES, BASE_MODELS, get_trainable_pipelines
from .test_fixtures import (
    generate_classification_dataset,
    generate_similarity_dataset,
    generate_triplet_dataset,
    generate_mlm_dataset,
    generate_ner_dataset,
    generate_id2label,
    generate_label2id,
)


def _get_training_params():
    """Generate pytest parameters for training tests."""
    params = []
    for family in MODEL_FAMILIES:
        for pipeline in get_trainable_pipelines(family):
            params.append(
                pytest.param(
                    family,
                    pipeline,
                    id=f"{family}-{pipeline}",
                    marks=[pytest.mark.slow]
                )
            )
    return params


def _get_dataset_for_pipeline(pipeline: str, triplet: bool = False) -> tuple:
    """Get dummy dataset and label mappings for a pipeline."""
    if pipeline == "text-classification":
        data = generate_classification_dataset(n_samples=16, n_classes=3)
        id2label = generate_id2label(3)
        label2id = generate_label2id(3)
        return HFDataset.from_dict(data), id2label, label2id
    
    elif pipeline == "sentence-similarity":
        # Use triplet data for cross-entropy training
        if triplet:
            data = generate_triplet_dataset(n_samples=16)
        else:
            data = generate_similarity_dataset(n_samples=16)
        return HFDataset.from_dict(data), None, None
    
    elif pipeline == "masked-lm":
        data = generate_mlm_dataset(n_samples=16)
        return HFDataset.from_dict(data), None, None
    
    elif pipeline == "token-classification":
        data = generate_ner_dataset(n_samples=16, n_tags=5)
        id2label = {str(i): f"TAG_{i}" for i in range(5)}
        label2id = {v: int(k) for k, v in id2label.items()}
        return HFDataset.from_dict(data), id2label, label2id
    
    else:
        raise ValueError(f"Unknown trainable pipeline: {pipeline}")


@pytest.mark.parametrize("model_family,pipeline", _get_training_params())
def test_training_step(model_family: str, pipeline: str):
    """Test that a single training step completes without error."""
    from mlx_raclate.utils.utils import load
    from mlx_raclate.tuner.trainer import Trainer, TrainingArgs
    
    def _test_training_step(model_family: str, pipeline: str, triplet: bool = False):
        """Test that a single training step completes without error."""
        # Get base model and dataset
        base_model = BASE_MODELS[model_family]
        train_dataset, id2label, label2id = _get_dataset_for_pipeline(pipeline, triplet)
    
        # Build model config
        model_config = {}
        if id2label:
            model_config["id2label"] = id2label
            model_config["label2id"] = label2id
    
        # Load model in training mode
        model, tokenizer = load(
            base_model,
            model_config=model_config,
            pipeline=pipeline,
            train=True,
        )

        if pipeline == "masked-lm" and getattr(tokenizer, "mask_token_id", None) is None:
             # 1. Try to find common mask tokens
             if "<mask >" in tokenizer.vocab:
                 tokenizer.mask_token = "<mask >"
                 tokenizer.mask_token_id = tokenizer.vocab["<mask >"]
             elif "[MASK]" in tokenizer.vocab:
                 tokenizer.mask_token = "[MASK]"
                 tokenizer.mask_token_id = tokenizer.vocab["[MASK]"]
             elif "<mask>" in tokenizer.vocab:
                 tokenizer.mask_token = "<mask>"
                 tokenizer.mask_token_id = tokenizer.vocab["<mask>"]
             
             # 2. If not found, add it and resize
             else:
                 tokenizer.add_tokens(["[MASK]"], special_tokens=True)
                 tokenizer.mask_token = "[MASK]"
                 tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
                 
                 # Resize using the new base model method
                 if hasattr(model, "resize_token_embeddings"):
                    model.resize_token_embeddings(len(tokenizer))

        # Configure minimal training
        args = TrainingArgs(
            batch_size=2,
            num_train_epochs=1,
            max_length=64,  # Short for speed
            learning_rate=1e-5,
            logging_steps=1,
            save_steps=10000,  # Don't save during test
            output_dir=f"/tmp/test_training_{model_family}_{pipeline}",
        )
    
        # Initialize trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            task_type=pipeline,
            training_args=args,
            train_dataset=train_dataset,
            label2id=label2id,
        )

        # Just verify the training loop executes without error
        try:
            # Access the internal training method to run just a few batches
            batches = list(trainer._create_batches(
                train_dataset, 
                args.batch_size,
                shuffle=True
            ))[:2]  # Just 2 batches
        
            for batch in batches:
                # Verify we can compute loss
                outputs = model(**trainer.data_collator(batch))
                assert "loss" in outputs or outputs.get("loss") is not None, \
                    f"Model should return loss during training for {pipeline}"
            
        except Exception as e:
            pytest.fail(f"Training step failed for {model_family}/{pipeline}: {e}")

    _test_training_step(model_family, pipeline)
    # second test for triplet
    if pipeline == "sentence-similarity":
        _test_training_step(model_family, pipeline, triplet=True)


def test_classification_dataset_generation():
    """Test classification dataset generator."""
    data = generate_classification_dataset(n_samples=10, n_classes=3)
    assert "text" in data and "label" in data
    assert len(data["text"]) == 10
    assert len(data["label"]) == 10
    assert all(label in [0, 1, 2] for label in data["label"])

def test_similarity_dataset_generation():
    """Test similarity dataset generator."""
    data = generate_similarity_dataset(n_samples=10)
    assert "text" in data and "text_pair" in data and "label" in data
    assert len(data["text"]) == 10
    assert all(0.0 <= score <= 1.0 for score in data["label"])

def test_triplet_dataset_generation():
    """Test triplet dataset generator."""
    data = generate_triplet_dataset(n_samples=10)
    assert "text" in data and "text_pair" in data and "negative" in data
    assert len(data["text"]) == 10

def test_mlm_dataset_generation():
    """Test MLM dataset generator."""
    data = generate_mlm_dataset(n_samples=10)
    assert "text" in data
    assert len(data["text"]) == 10

def test_ner_dataset_generation():
    """Test NER dataset generator."""
    data = generate_ner_dataset(n_samples=10, n_tags=5)
    assert "text" in data and "labels" in data
    assert len(data["text"]) == 10
    assert all(len(tokens) == len(labels) for tokens, labels in zip(data["text"], data["labels"]))


def test_token_classification_dataset_columns_are_standardized():
    """Test HF-style token-classification columns map to trainer columns."""
    from mlx_raclate.tuner.datasets import DatasetArgs, _standardize_column_names

    dataset = HFDataset.from_dict(
        {
            "tokens": [["John", "lives", "there"]],
            "ner_tags": [[1, 0, 0]],
        }
    )
    args = DatasetArgs(data="unused", task_type="token-classification")

    standardized = _standardize_column_names(dataset, args)

    assert "text" in standardized.column_names
    assert "labels" in standardized.column_names
    assert standardized[0]["text"] == ["John", "lives", "there"]
    assert standardized[0]["labels"] == [1, 0, 0]


def test_token_classification_pretokenized_columns_are_preserved():
    """Test pretokenized token-classification columns survive standardization."""
    from mlx_raclate.tuner.datasets import DatasetArgs, _standardize_column_names

    dataset = HFDataset.from_dict(
        {
            "input_ids": [[10, 11]],
            "attention_mask": [[1, 1]],
            "labels": [[0, 1]],
            "unused": ["drop me"],
        }
    )
    args = DatasetArgs(data="unused", task_type="token-classification")

    standardized = _standardize_column_names(dataset, args)

    assert set(standardized.column_names) == {"input_ids", "attention_mask", "labels"}
    assert standardized[0]["input_ids"] == [10, 11]
    assert standardized[0]["labels"] == [0, 1]


def test_pretokenized_token_classification_collator_pads_inputs():
    """Test pretokenized token-classification batches are padded directly."""
    from mlx_raclate.tuner.collators import DataCollatorForTokenClassification

    class DummyTokenizer:
        pad_token_id = 0
        eos_token_id = 2

    collator = DataCollatorForTokenClassification(
        tokenizer=DummyTokenizer(),
        max_length=4,
    )

    batch = collator(
        {
            "input_ids": [[10, 11, 12], [20]],
            "attention_mask": [[1, 1, 1], [1]],
            "labels": [[1, 2, 3], [4]],
        }
    )

    assert batch["input_ids"].tolist() == [[10, 11, 12], [20, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[1, 2, 3], [4, -100, -100]]


def test_token_classification_checkpoint_saves_viterbi_calibration(tmp_path):
    """Test token-classification checkpoints include Viterbi calibration metadata."""
    import json
    from types import SimpleNamespace

    import mlx.core as mx

    from mlx_raclate.tuner.trainer import Trainer

    class DummyModel:
        config = SimpleNamespace(model_type="dummy")
        viterbi_calibration = {
            "operating_points": {
                "default": {
                    "biases": {
                        "transition_bias_background_stay": 0.0,
                        "transition_bias_background_to_start": 2.0,
                        "transition_bias_inside_to_continue": 0.0,
                        "transition_bias_inside_to_end": 0.0,
                        "transition_bias_end_to_background": 0.0,
                        "transition_bias_end_to_start": 0.0,
                    }
                }
            }
        }

        def get_hf_transformers_arch(self):
            return "DummyForTokenClassification"

        def parameters(self):
            return {"weight": mx.zeros((1,), dtype=mx.float32)}

    class DummyTokenizer:
        def save_pretrained(self, path):
            return None

    trainer = Trainer.__new__(Trainer)
    trainer.output_dir = tmp_path
    trainer.global_step = 7
    trainer.model = DummyModel()
    trainer.tokenizer = DummyTokenizer()
    trainer.task_type = "token-classification"
    trainer.args = SimpleNamespace(push_to_hub=False, save_total_limit=None)

    trainer._save_checkpoint({"loss": 1.0})

    calibration_path = tmp_path / "checkpoint-7" / "viterbi_calibration.json"
    with open(calibration_path) as f:
        calibration = json.load(f)

    assert calibration["operating_points"]["default"]["biases"]["transition_bias_background_to_start"] == 2.0


def test_label_mappings_generation():
    """Test label mapping generators."""
    id2label = generate_id2label(3)
    label2id = generate_label2id(3)
    
    assert len(id2label) == 3
    assert len(label2id) == 3
    assert all(str(k) in id2label for k in range(3))
    assert all(label2id[v] == int(k) for k, v in id2label.items())


def test_training_args_runtime_memory_options():
    """Test runtime memory knobs are exposed with expected defaults."""
    from mlx_raclate.tuner.trainer import TrainingArgs

    default_args = TrainingArgs()
    assert default_args.grad_checkpoint is True
    assert default_args.cache_clear_steps == 1

    custom_args = TrainingArgs(
        grad_checkpoint=False,
        cache_clear_steps=8,
    )
    assert custom_args.grad_checkpoint is False
    assert custom_args.cache_clear_steps == 8
