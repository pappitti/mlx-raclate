"""Tests for installable model-card inference templates."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_raclate.tuner.model_card_utils import (
    _load_inference_example_module,
    get_code_for_trained_model,
    get_inference_code,
    get_transformers_inference_code,
)


pytestmark = pytest.mark.release


def test_token_classification_template_is_loaded_from_runtime_package():
    module = _load_inference_example_module("token-classification")

    assert module.__name__ == (
        "mlx_raclate.inference_examples.token_classification"
    )


def test_token_classification_model_card_code_is_generated_without_tests_package():
    code = get_inference_code(
        pipeline="token-classification",
        model_path="org/model",
    )

    assert 'model_path = "org/model"' in code
    assert "postprocess_token_classification_output" in code


def test_trained_model_card_contains_hub_metadata_and_repo_usage_path():
    from huggingface_hub import ModelCard

    content = get_code_for_trained_model(
        pipeline="text-classification",
        model_path="acme/fine-tuned-model",
        base_model="acme/base-model",
    )
    card = ModelCard(content)

    assert card.data.base_model == "acme/base-model"
    assert card.data.library_name == "mlx-raclate"
    assert card.data.pipeline_tag == "text-classification"
    assert set(card.data.tags) == {"mlx", "mlx-raclate"}
    assert '    "acme/fine-tuned-model",' in content
    assert "https://huggingface.co/acme/base-model" in content


def test_trained_model_card_omits_unknown_base_model():
    from huggingface_hub import ModelCard

    content = get_code_for_trained_model(
        pipeline="embeddings",
        model_path="local/checkpoint-1",
    )
    card = ModelCard(content)

    assert card.data.base_model is None
    assert card.data.pipeline_tag == "feature-extraction"
    assert "**Base Model**" not in content


class _FakeTokenizer:
    mask_token_id = 7

    def __init__(self):
        self._tokenizer = self

    def __call__(self, texts, *args, return_offsets_mapping=False, **kwargs):
        batch_size = len(texts)
        result = {
            "input_ids": mx.array([[1, 2, 3]] * batch_size),
            "attention_mask": mx.ones((batch_size, 3), dtype=mx.int32),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = mx.array(
                [[[0, 1], [1, 2], [2, 3]]] * batch_size
            )
        return result

    def encode(self, text, return_tensors=None, **kwargs):
        token_ids = [1, self.mask_token_id, 2]
        if return_tensors == "mlx":
            return mx.array([token_ids])
        return token_ids

    def decode(self, token_ids):
        labels = {
            0: "artificial intelligence",
            1: "physics",
            2: "biology",
        }
        return labels.get(int(token_ids[0]), f"token-{int(token_ids[0])}")


class _FakeModel:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.config = SimpleNamespace(
            max_position_embeddings=32,
            id2label={"0": "O", "1": "S-PERSON"},
        )
        self.viterbi_calibration = None

    def __call__(self, **inputs):
        input_ids = inputs["input_ids"]
        batch_size, sequence_length = input_ids.shape

        if self.pipeline == "text-classification":
            return {
                "probabilities": mx.broadcast_to(
                    mx.array([[0.25, 0.75]]),
                    (batch_size, 2),
                )
            }
        if self.pipeline == "embeddings":
            return {
                "embeddings": mx.broadcast_to(
                    mx.array([[1.0, 0.0, 0.0]]),
                    (batch_size, 3),
                )
            }
        if self.pipeline == "sentence-similarity":
            document_count = inputs["reference_input_ids"].shape[0]
            return {"similarities": mx.ones((batch_size, document_count))}
        if self.pipeline == "token-classification":
            logits = mx.zeros((batch_size, sequence_length, 2))
            return {
                "logits": logits,
                "probabilities": mx.softmax(logits, axis=-1),
            }

        return {"logits": mx.zeros((batch_size, sequence_length, 8))}


@pytest.mark.parametrize(
    "pipeline,kwargs",
    [
        (
            "text-classification",
            {"texts": ["A test sentence."], "is_regression": False},
        ),
        (
            "sentence-similarity",
            {
                "queries": ["A query"],
                "documents": ["A document"],
                "use_late_interaction": False,
            },
        ),
        (
            "sentence-similarity",
            {
                "queries": ["A query"],
                "documents": ["A document"],
                "use_late_interaction": True,
            },
        ),
        ("embeddings", {"texts": ["First", "Second"]}),
        ("masked-lm", {"text": "The [MASK] is blue."}),
        (
            "token-classification",
            {"texts": ["Alice works here."]},
        ),
        (
            "zero-shot-classification",
            {
                "text": "MLX is a framework.",
                "label_candidates": ["technology", "sports"],
            },
        ),
    ],
)
def test_generated_mlx_model_card_code_executes(monkeypatch, pipeline, kwargs):
    """Execute the exact Python snippet emitted into the checkpoint README."""
    from mlx_raclate.utils import utils

    loaded_pipelines = []

    def fake_load(model_path, *, pipeline, **model_kwargs):
        loaded_pipelines.append(pipeline)
        return _FakeModel(pipeline), _FakeTokenizer()

    monkeypatch.setattr(utils, "load", fake_load)
    code = get_inference_code(
        pipeline=pipeline,
        model_path="local/checkpoint",
        **kwargs,
    )

    exec(compile(code, "<generated-model-card>", "exec"), {})

    assert loaded_pipelines == [pipeline]


def test_generated_transformers_token_classification_code_executes(monkeypatch):
    """Execute the exact Transformers section emitted into the NER README."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "local/checkpoint"
            return cls()

        def __call__(self, texts, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "offset_mapping": torch.tensor([[[0, 1], [1, 2], [2, 3]]]),
            }

    class FakeAutoModel:
        config = SimpleNamespace(id2label={0: "O", 1: "S-PERSON"})

        @classmethod
        def from_pretrained(cls, model_id):
            assert model_id == "local/checkpoint"
            return cls()

        def eval(self):
            return self

        def __call__(self, **inputs):
            batch_size, sequence_length = inputs["input_ids"].shape
            return SimpleNamespace(
                logits=torch.zeros((batch_size, sequence_length, 2))
            )

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(
        transformers,
        "AutoModelForTokenClassification",
        FakeAutoModel,
    )
    code = get_transformers_inference_code(
        pipeline="token-classification",
        model_path="local/checkpoint",
        texts=["Alice works here."],
    )

    assert code is not None
    exec(compile(code, "<generated-transformers-model-card>", "exec"), {})
