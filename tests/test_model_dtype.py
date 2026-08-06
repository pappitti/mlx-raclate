import mlx.core as mx

from mlx_raclate.utils.utils import (
    _cast_floating_weights,
    _determine_model_dtype,
)


def test_bfloat16_config_override_selects_bfloat16():
    assert _determine_model_dtype({"torch_dtype": "bfloat16"}, {}) == mx.bfloat16


def test_cast_floating_weights_preserves_integer_arrays():
    weights = {
        "weight": mx.ones((2, 2), dtype=mx.float32),
        "indices": mx.array([0, 1], dtype=mx.int32),
    }

    result = _cast_floating_weights(weights, mx.bfloat16)
    mx.eval(result)

    assert result is weights
    assert result["weight"].dtype == mx.bfloat16
    assert result["indices"].dtype == mx.int32
