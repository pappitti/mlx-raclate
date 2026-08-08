"""Published-checkpoint validation for generated model-card code."""

import pytest

from mlx_raclate.tuner.model_card_utils import get_inference_code


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_generated_token_classification_card_runs_against_real_checkpoint():
    code = get_inference_code(
        pipeline="token-classification",
        model_path="mlx-community/openai-privacy-filter-bf16",
        texts=["Email Alice at alice@example.com after 5pm."],
    )

    exec(compile(code, "<generated-real-model-card>", "exec"), {})
