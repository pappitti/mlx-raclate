import json
from unittest.mock import Mock, patch

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

from mlx_raclate.utils.tokenizer_utils import (
    _FIXED_MISTRAL_REGEX,
    _ensure_mistral_regex_fixed,
    load_tokenizer,
)


def test_load_tokenizer_fixes_mistral_regex_by_default(tmp_path):
    tokenizer = Mock()
    tokenizer.decode.return_value = ""

    with patch(
        "mlx_raclate.utils.tokenizer_utils.AutoTokenizer.from_pretrained",
        return_value=tokenizer,
    ) as from_pretrained:
        load_tokenizer(tmp_path)

    from_pretrained.assert_called_once_with(
        tmp_path,
        trust_remote_code=False,
        fix_mistral_regex=True,
    )


def test_load_tokenizer_allows_explicit_regex_compatibility_override(tmp_path):
    tokenizer = Mock()
    tokenizer.decode.return_value = ""

    with patch(
        "mlx_raclate.utils.tokenizer_utils.AutoTokenizer.from_pretrained",
        return_value=tokenizer,
    ) as from_pretrained:
        load_tokenizer(tmp_path, {"fix_mistral_regex": False})

    assert from_pretrained.call_args.kwargs["fix_mistral_regex"] is False


def test_load_tokenizer_forces_fix_when_transformers_skips_requested_patch():
    incorrect_regex = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
        r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
        r"\s+(?!\S)|\s+"
    )
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                pattern=Regex(incorrect_regex),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    backend.decoder = decoders.ByteLevel()
    tokenizer = Mock(backend_tokenizer=backend, _tokenizer=backend)

    assert _ensure_mistral_regex_fixed(tokenizer) is True

    state = json.loads(tokenizer._tokenizer.to_str())
    assert (
        state["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        == _FIXED_MISTRAL_REGEX
    )
