import inspect

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_raclate.models import base
from mlx_raclate.models.base import GradientCheckpointingMixin, RaclateBaseModel
from mlx_raclate.models.gemma3_text import Gemma3Model
from mlx_raclate.models.lfm2 import Lfm2Model
from mlx_raclate.models.modernbert import ModernBertModel
from mlx_raclate.models.neobert import NeoBERTModel
from mlx_raclate.models.openai_privacy_filter import OpenAIPrivacyFilterModel
from mlx_raclate.models.qwen3 import Qwen3Model
from mlx_raclate.models.t5gemma_encoder import T5GemmaEncoder
from mlx_raclate.tuner.trainer import Trainer


BACKBONE_CLASSES = (
    Qwen3Model,
    ModernBertModel,
    Gemma3Model,
    T5GemmaEncoder,
    Lfm2Model,
    NeoBERTModel,
    OpenAIPrivacyFilterModel,
)


class _IdentityLayer(nn.Module):
    def __call__(self, value):
        return value


class _CheckpointableBackbone(GradientCheckpointingMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _IdentityLayer()

    def __call__(self, value):
        return self._call_checkpointed_layer(self.layer, value)


class _CheckpointableModel(RaclateBaseModel):
    def __init__(self):
        super().__init__()
        self.model = _CheckpointableBackbone()


class _LinearBackbone(GradientCheckpointingMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 2)

    def __call__(self, value):
        return self._call_checkpointed_layer(self.layer, value)


class _DifferentiableModel(RaclateBaseModel):
    def __init__(self):
        super().__init__()
        self.model = _LinearBackbone()

    def __call__(self, value):
        return self.model(value)


class _UnsupportedModel(RaclateBaseModel):
    pass


@pytest.mark.parametrize("backbone_class", BACKBONE_CLASSES)
def test_backbones_use_shared_gradient_checkpointing(backbone_class):
    assert issubclass(backbone_class, GradientCheckpointingMixin)
    assert "_call_checkpointed_layer" in inspect.getsource(backbone_class.__call__)


def test_checkpointing_is_enabled_for_compatible_backbones(monkeypatch):
    checkpointed_layers = []

    def fake_checkpoint(layer):
        checkpointed_layers.append(layer)
        return layer

    monkeypatch.setattr(base, "checkpoint_module", fake_checkpoint)
    model = _CheckpointableModel()

    assert model.set_gradient_checkpointing(True) == 1
    assert model.model.gradient_checkpointing is True

    model.train()
    assert model.model("training") == "training"
    assert checkpointed_layers == [model.model.layer]

    model.eval()
    assert model.model("evaluation") == "evaluation"
    assert checkpointed_layers == [model.model.layer]


def test_checkpointed_gradients_match_regular_gradients():
    model = _DifferentiableModel()
    inputs = mx.ones((2, 3))
    grad_fn = nn.value_and_grad(model, lambda value: mx.sum(model(value)))

    model.set_gradient_checkpointing(False)
    regular_loss, regular_grads = grad_fn(inputs)

    model.set_gradient_checkpointing(True)
    checkpointed_loss, checkpointed_grads = grad_fn(inputs)

    regular_grads = dict(tree_flatten(regular_grads))
    checkpointed_grads = dict(tree_flatten(checkpointed_grads))
    mx.eval(
        regular_loss,
        checkpointed_loss,
        regular_grads,
        checkpointed_grads,
    )

    assert mx.allclose(regular_loss, checkpointed_loss)
    assert regular_grads.keys() == checkpointed_grads.keys()
    for name in regular_grads:
        assert mx.allclose(regular_grads[name], checkpointed_grads[name])


def test_trainer_activates_supported_checkpointing(capsys):
    trainer = Trainer.__new__(Trainer)
    trainer.model = _CheckpointableModel()

    trainer._apply_grad_checkpointing()

    assert trainer.model.model.gradient_checkpointing is True
    assert "1 backbone(s)" in capsys.readouterr().out


def test_trainer_rejects_unsupported_checkpointing():
    trainer = Trainer.__new__(Trainer)
    trainer.model = _UnsupportedModel()

    with pytest.raises(ValueError, match="no checkpoint-compatible backbone"):
        trainer._apply_grad_checkpointing()
