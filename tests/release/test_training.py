"""Lean training, checkpoint, and dataset contracts for publishing."""

import pytest
from datasets import Dataset as HFDataset

pytestmark = pytest.mark.release


def _checkpoint_only_trainer(tmp_path, args, *, global_step=3):
    from types import SimpleNamespace

    import mlx.core as mx

    from mlx_raclate.tuner.trainer import Trainer

    class DummyModel:
        config = SimpleNamespace(model_type="dummy")

        def get_hf_transformers_arch(self):
            return None

        def parameters(self):
            return {"weight": mx.zeros((1,), dtype=mx.float32)}

    class DummyTokenizer:
        def save_pretrained(self, path):
            return None

    trainer = Trainer.__new__(Trainer)
    trainer.output_dir = tmp_path
    trainer.global_step = global_step
    trainer.model = DummyModel()
    trainer.tokenizer = DummyTokenizer()
    trainer.task_type = "text-classification"
    trainer.args = args
    return trainer


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


def test_token_classification_checkpoint_survives_model_card_failure(
    tmp_path, monkeypatch
):
    """Test token-classification checkpoints include Viterbi calibration metadata."""
    import json
    from types import SimpleNamespace

    import mlx.core as mx

    from mlx_raclate.tuner import trainer as trainer_module
    from mlx_raclate.tuner.trainer import Trainer

    def fail_model_card(**kwargs):
        raise ImportError("missing inference template")

    monkeypatch.setattr(trainer_module, "get_code_for_trained_model", fail_model_card)

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
    trainer.args = SimpleNamespace(push_to_hub=False, save_total_limit=1)
    (tmp_path / "checkpoint-1").mkdir()

    with pytest.warns(UserWarning, match="README generation failed"):
        trainer._save_checkpoint({"loss": 1.0})

    calibration_path = tmp_path / "checkpoint-7" / "viterbi_calibration.json"
    with open(calibration_path) as f:
        calibration = json.load(f)

    assert calibration["operating_points"]["default"]["biases"]["transition_bias_background_to_start"] == 2.0
    assert (tmp_path / "checkpoint-7" / "model.safetensors").is_file()
    assert (tmp_path / "checkpoint-7" / "config.json").is_file()
    assert (tmp_path / "checkpoint-7" / "metrics.json").is_file()
    assert not (tmp_path / "checkpoint-1").exists()


def test_local_checkpoint_card_uses_local_path_and_explicit_base_model(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from mlx_raclate.tuner import trainer as trainer_module

    captured = {}

    def capture_model_card(**kwargs):
        captured.update(kwargs)
        return "local card"

    monkeypatch.setattr(
        trainer_module,
        "get_code_for_trained_model",
        capture_model_card,
    )
    trainer = _checkpoint_only_trainer(
        tmp_path,
        SimpleNamespace(
            push_to_hub=False,
            hub_repo_id=None,
            base_model="acme/base-model",
            save_total_limit=None,
        ),
    )

    trainer._save_checkpoint({"loss": 1.0})

    checkpoint = tmp_path / "checkpoint-3"
    assert captured["model_path"] == str(checkpoint)
    assert captured["base_model"] == "acme/base-model"
    assert (checkpoint / "README.md").read_text() == "local card"


def test_hub_upload_failure_preserves_checkpoint_and_uses_explicit_ids(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from mlx_raclate.tuner import trainer as trainer_module

    captured = {}

    def capture_model_card(**kwargs):
        captured["model_card"] = kwargs
        return "hub card"

    def fail_upload(**kwargs):
        captured["upload"] = kwargs
        raise RuntimeError("Hub unavailable")

    monkeypatch.setattr(
        trainer_module,
        "get_code_for_trained_model",
        capture_model_card,
    )
    monkeypatch.setattr(trainer_module, "upload_to_hub", fail_upload)
    trainer = _checkpoint_only_trainer(
        tmp_path,
        SimpleNamespace(
            push_to_hub=True,
            hub_repo_id="acme/fine-tuned-model",
            base_model="acme/base-model",
            save_total_limit=None,
        ),
    )

    with pytest.warns(UserWarning, match="Hub upload failed"):
        trainer._save_checkpoint({"loss": 1.0})

    checkpoint = tmp_path / "checkpoint-3"
    assert captured["model_card"]["model_path"] == "acme/fine-tuned-model"
    assert captured["model_card"]["base_model"] == "acme/base-model"
    assert captured["upload"] == {
        "path": checkpoint,
        "hub_repo_id": "acme/fine-tuned-model",
    }
    assert (checkpoint / "README.md").read_text() == "hub card"
    assert (checkpoint / "model.safetensors").is_file()


def test_upload_to_hub_creates_and_uploads_explicit_repo(tmp_path, monkeypatch):
    import huggingface_hub

    from mlx_raclate.tuner.trainer import upload_to_hub

    calls = []

    class FakeApi:
        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload_folder", kwargs))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    checkpoint = tmp_path / "checkpoint-11"
    checkpoint.mkdir()

    upload_to_hub(checkpoint, "acme/fine-tuned-model")

    assert calls == [
        (
            "create_repo",
            {
                "repo_id": "acme/fine-tuned-model",
                "repo_type": "model",
                "exist_ok": True,
            },
        ),
        (
            "upload_folder",
            {
                "folder_path": checkpoint,
                "repo_id": "acme/fine-tuned-model",
                "repo_type": "model",
                "commit_message": "Upload checkpoint-11",
            },
        ),
    ]


def test_checkpoint_rotation_uses_numeric_steps_and_ignores_other_paths(tmp_path):
    """Checkpoint retention removes only the oldest numbered directories."""
    from types import SimpleNamespace

    from mlx_raclate.tuner.trainer import Trainer

    for name in ("checkpoint-2", "checkpoint-10", "checkpoint-100"):
        checkpoint = tmp_path / name
        checkpoint.mkdir()
        (checkpoint / "marker").write_text(name)
    (tmp_path / "checkpoint-latest").mkdir()
    (tmp_path / "checkpoint-200").write_text("not a directory")

    trainer = Trainer.__new__(Trainer)
    trainer.output_dir = tmp_path
    trainer.args = SimpleNamespace(save_total_limit=2)

    trainer._rotate_checkpoints()

    assert not (tmp_path / "checkpoint-2").exists()
    assert (tmp_path / "checkpoint-10").is_dir()
    assert (tmp_path / "checkpoint-100").is_dir()
    assert (tmp_path / "checkpoint-latest").is_dir()
    assert (tmp_path / "checkpoint-200").is_file()


def test_partial_accumulation_window_is_flushed_with_actual_divisor(
    monkeypatch,
):
    """A short final window produces one correctly averaged optimizer update."""
    from types import SimpleNamespace

    from mlx_raclate.tuner import trainer as trainer_module
    from mlx_raclate.tuner.trainer import Trainer

    cache_clears = []
    monkeypatch.setattr(
        trainer_module,
        "mx",
        SimpleNamespace(
            array=lambda value: value,
            stop_gradient=lambda value: value,
            eval=lambda *values: None,
            clear_cache=lambda: cache_clears.append(True),
        ),
    )
    monkeypatch.setattr(trainer_module.gc, "collect", lambda: None)

    class DummyModel:
        state = {}

        def train(self):
            return None

    trainer = Trainer.__new__(Trainer)
    trainer.model = DummyModel()
    trainer.optimizer = SimpleNamespace(state={})
    trainer.optimizer_step = 0
    trainer.args = SimpleNamespace(
        batch_size=1,
        gradient_accumulation_steps=3,
        gradient_accumulation_dtype="model",
        cache_clear_steps=2,
    )
    trainer.train_dataset = object()
    trainer.epoch = 0
    trainer.global_step = 0
    trainer.next_log_step = 100
    trainer.next_save_step = 100
    trainer.logging_steps = 100
    trainer.save_steps = 100
    trainer.data_collator = lambda batch: batch
    trainer._create_batches = lambda *args, **kwargs: iter(
        {"gradient": gradient} for gradient in (1.0, 2.0, 3.0, 8.0, 10.0)
    )
    trainer.step_calc = lambda batch: (
        1.0,
        {"weight": batch["gradient"]},
    )
    applied_gradients = []

    def record_update(grads):
        applied_gradients.append(grads["weight"])
        return 1.0

    trainer.step_update = record_update

    trainer._train_epoch()

    assert applied_gradients == [2.0, 9.0]
    assert trainer.global_step == 5
    assert trainer.microbatches_in_epoch == 5
    assert trainer.optimizer_step == 2
    assert len(cache_clears) == 1


def test_checkpoint_restores_optimizer_and_completed_epoch(tmp_path):
    """A complete checkpoint resumes at the next epoch with Adam state intact."""
    import json
    from types import SimpleNamespace

    import mlx.core as mx
    import mlx.nn as nn

    from mlx_raclate.tuner.trainer import Trainer
    from mlx_raclate.tuner.utils import MultiPrecisionAdamW

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.array([1.0], dtype=mx.bfloat16)
            self.config = SimpleNamespace(model_type="dummy")

        def get_hf_transformers_arch(self):
            return None

    class DummyTokenizer:
        def save_pretrained(self, path):
            return None

    model = TinyModel()
    optimizer = MultiPrecisionAdamW(learning_rate=1.5e-5, weight_decay=0.01)
    optimizer.update(
        model,
        {"weight": mx.array([0.1], dtype=mx.bfloat16)},
    )
    mx.eval(model.parameters(), optimizer.state)
    expected_master = optimizer.state["weight"]["master"]

    trainer = Trainer.__new__(Trainer)
    trainer.output_dir = tmp_path
    trainer.global_step = 8
    trainer.epoch = 0
    trainer.microbatches_in_epoch = 8
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.tokenizer = DummyTokenizer()
    trainer.task_type = "text-classification"
    trainer.args = SimpleNamespace(push_to_hub=False, save_total_limit=None)
    trainer._save_checkpoint({"loss": 1.0}, completed_epoch=True)

    checkpoint = tmp_path / "checkpoint-8"
    with open(checkpoint / "trainer_state.json") as file:
        state = json.load(file)
    assert state["completed_epochs"] == 1
    assert state["optimizer_step"] == 1

    resumed = Trainer.__new__(Trainer)
    resumed.model = TinyModel()
    resumed.optimizer = MultiPrecisionAdamW(
        learning_rate=1.5e-5,
        weight_decay=0.01,
    )
    resumed._restore_checkpoint_state(checkpoint)

    assert resumed.global_step == 8
    assert resumed.start_epoch == 1
    assert resumed.resume_microbatches_in_epoch == 0
    assert resumed.optimizer.step.item() == 1
    assert mx.allclose(
        resumed.optimizer.state["weight"]["master"],
        expected_master,
    )


def test_train_continues_from_completed_epoch():
    """The total epoch target is retained when resuming a phase."""
    from types import SimpleNamespace

    from mlx_raclate.tuner.trainer import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(num_train_epochs=3, eval_strategy="no")
    trainer.start_epoch = 1
    trainer.resume_microbatches_in_epoch = 0
    trainer.eval_dataset = None
    trained_epochs = []
    saved_epochs = []
    trainer._train_epoch = lambda skip_microbatches=0: trained_epochs.append(
        (trainer.epoch, skip_microbatches)
    )
    trainer._save_checkpoint = lambda metrics, completed_epoch=False: saved_epochs.append(
        (trainer.epoch, completed_epoch)
    )

    trainer.train()

    assert trained_epochs == [(1, 0), (2, 0)]
    assert saved_epochs == [(1, True), (2, True)]


def test_evaluation_failure_saves_completed_epoch_before_reraising():
    """A broken evaluator must not discard a fully trained epoch."""
    from types import SimpleNamespace

    from mlx_raclate.tuner.trainer import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(num_train_epochs=1, eval_strategy="epoch")
    trainer.start_epoch = 0
    trainer.resume_microbatches_in_epoch = 0
    trainer.eval_dataset = object()
    trainer._train_epoch = lambda skip_microbatches=0: None
    trainer.evaluate = lambda: (_ for _ in ()).throw(RuntimeError("bad metrics"))
    saved = []
    trainer._save_checkpoint = lambda metrics, completed_epoch=False: saved.append(
        (metrics, completed_epoch)
    )

    with pytest.warns(UserWarning, match="saving a resumable checkpoint"):
        with pytest.raises(RuntimeError, match="bad metrics"):
            trainer.train()

    assert saved == [
        (
            {
                "eval_failed": True,
                "eval_error_type": "RuntimeError",
                "eval_error": "bad metrics",
            },
            True,
        )
    ]


def test_training_args_runtime_memory_options():
    """Test runtime memory knobs are exposed with expected defaults."""
    from mlx_raclate.tuner.trainer import TrainingArgs

    default_args = TrainingArgs()
    assert default_args.grad_checkpoint is True
    assert default_args.cache_clear_steps == 1
    assert default_args.multi_precision is True
    assert default_args.gradient_accumulation_dtype == "float32"
    assert default_args.push_to_hub is False
    assert default_args.hub_repo_id is None
    assert default_args.base_model is None

    custom_args = TrainingArgs(
        grad_checkpoint=False,
        cache_clear_steps=8,
        multi_precision=False,
        gradient_accumulation_dtype="model",
    )
    assert custom_args.grad_checkpoint is False
    assert custom_args.cache_clear_steps == 8
    assert custom_args.multi_precision is False
    assert custom_args.gradient_accumulation_dtype == "model"

    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        TrainingArgs(gradient_accumulation_steps=0)
    with pytest.raises(ValueError, match="save_total_limit"):
        TrainingArgs(save_total_limit=0)
    with pytest.raises(ValueError, match="cache_clear_steps"):
        TrainingArgs(cache_clear_steps=-1)
    with pytest.raises(ValueError, match="hub_repo_id"):
        TrainingArgs(push_to_hub=True, base_model="acme/base-model")
    with pytest.raises(ValueError, match="base_model"):
        TrainingArgs(push_to_hub=True, hub_repo_id="acme/model")
