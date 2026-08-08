import mlx.core as mx

from mlx_raclate.models.base import mean_pooling
from mlx_raclate.tuner.collators import DataCollatorForMaskedLanguageModeling


class _MaskedLMTokenizer:
    def __init__(
        self,
        input_ids,
        *,
        cls_token_id=1,
        sep_token_id=2,
        pad_token_id=0,
    ):
        self._input_ids = mx.array(input_ids, dtype=mx.int32)
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = 3
        self.vocab_size = 32

    def __call__(self, texts, **kwargs):
        del texts, kwargs
        return {"input_ids": self._input_ids}


def test_mean_pooling_preserves_fp32_semantics_without_expanded_mask():
    token_embeddings = mx.array(
        [
            [[1.0, 2.0], [3.0, 6.0], [100.0, 200.0]],
            [[4.0, 8.0], [10.0, 20.0], [16.0, 32.0]],
        ],
        dtype=mx.float16,
    )
    attention_mask = mx.array([[1, 1, 0], [1, 0, 1]], dtype=mx.int32)

    pooled = mean_pooling(token_embeddings, attention_mask)
    mx.eval(pooled)

    expected = mx.array([[2.0, 4.0], [10.0, 20.0]], dtype=mx.float32)
    assert pooled.shape == (2, 2)
    assert pooled.dtype == mx.float32
    assert mx.allclose(pooled, expected).item()


def test_mean_pooling_returns_zero_for_fully_masked_rows():
    token_embeddings = mx.ones((1, 3, 2), dtype=mx.float16)
    attention_mask = mx.zeros((1, 3), dtype=mx.int32)

    pooled = mean_pooling(token_embeddings, attention_mask)
    mx.eval(pooled)

    assert mx.array_equal(pooled, mx.zeros((1, 2), dtype=mx.float32)).item()


def test_mlm_collator_excludes_special_tokens_from_labels():
    tokenizer = _MaskedLMTokenizer([[1, 7, 2, 0], [1, 8, 9, 2]])
    collator = DataCollatorForMaskedLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=1.0,
    )

    batch = collator({"text": ["first", "second"]})
    mx.eval(batch["input_ids"], batch["labels"])

    expected_labels = mx.array(
        [[-100, 7, -100, -100], [-100, 8, 9, -100]],
        dtype=mx.int32,
    )
    assert mx.array_equal(batch["labels"], expected_labels).item()


def test_mlm_collator_ignores_missing_optional_special_token_ids():
    tokenizer = _MaskedLMTokenizer(
        [[4, 5, 6]],
        cls_token_id=None,
        sep_token_id=None,
        pad_token_id=None,
    )
    collator = DataCollatorForMaskedLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=1.0,
    )

    batch = collator({"text": ["text"]})
    mx.eval(batch["input_ids"], batch["labels"])

    expected_labels = mx.array([[4, 5, 6]], dtype=mx.int32)
    assert mx.array_equal(batch["labels"], expected_labels).item()
