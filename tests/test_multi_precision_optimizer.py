import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers

from mlx_raclate.tuner.utils import MultiPrecisionAdamW


class _ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.array([1.0], dtype=mx.bfloat16)


def test_multi_precision_adamw_keeps_fp32_master_state():
    model = _ScalarModel()
    optimizer = MultiPrecisionAdamW(learning_rate=1.5e-5, weight_decay=0.0)
    gradient = {"weight": mx.array([0.1], dtype=mx.bfloat16)}

    optimizer.update(model, gradient)
    mx.eval(model.parameters(), optimizer.state)

    state = optimizer.state["weight"]
    assert model.weight.dtype == mx.bfloat16
    assert state["master"].dtype == mx.float32
    assert state["m"].dtype == mx.float32
    assert state["v"].dtype == mx.float32
    assert state["master"].item() < 1.0


def test_master_weight_accumulates_updates_below_bfloat16_resolution():
    model = _ScalarModel()
    optimizer = MultiPrecisionAdamW(learning_rate=1.5e-5, weight_decay=0.0)
    gradient = {"weight": mx.array([0.1], dtype=mx.bfloat16)}

    for _ in range(512):
        optimizer.update(model, gradient)
    mx.eval(model.parameters(), optimizer.state)

    assert model.weight.item() < 1.0


def test_master_weight_matches_float32_adamw():
    low_precision_model = _ScalarModel()
    reference_model = _ScalarModel()
    reference_model.set_dtype(mx.float32)
    low_precision_optimizer = MultiPrecisionAdamW(
        learning_rate=1.5e-5,
        weight_decay=0.01,
    )
    reference_optimizer = optimizers.AdamW(
        learning_rate=1.5e-5,
        weight_decay=0.01,
    )

    for _ in range(8):
        low_precision_optimizer.update(
            low_precision_model,
            {"weight": mx.array([0.1], dtype=mx.bfloat16)},
        )
        reference_optimizer.update(
            reference_model,
            {"weight": mx.array([0.1], dtype=mx.float32)},
        )
    mx.eval(
        low_precision_optimizer.state,
        reference_model.parameters(),
    )

    master = low_precision_optimizer.state["weight"]["master"]
    assert mx.allclose(master, reference_model.weight)
