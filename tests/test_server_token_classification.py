import asyncio

import pytest
from fastapi.testclient import TestClient

from mlx_raclate.utils import server


class _FakeConfig:
    max_position_embeddings = 64
    id2label = {"0": "O", "1": "S-private_email"}


class _FakeModel:
    config = _FakeConfig()

    def __call__(self, **kwargs):
        return {
            "logits": [
                [
                    [5.0, 0.0],
                    [5.0, 0.0],
                    [0.0, 5.0],
                    [5.0, 0.0],
                ]
            ],
            "probabilities": [
                [
                    [0.99, 0.01],
                    [0.99, 0.01],
                    [0.01, 0.99],
                    [0.99, 0.01],
                ]
            ],
        }


class _FakeProbabilitiesOnlyModel:
    config = _FakeConfig()

    def __call__(self, **kwargs):
        return {
            "probabilities": [
                [
                    [0.99, 0.01],
                    [0.99, 0.01],
                    [0.01, 0.99],
                    [0.99, 0.01],
                ]
            ],
        }


class _FakeTokenizer:
    def __init__(self):
        self._tokenizer = self

    def __call__(self, texts, **kwargs):
        return {
            "input_ids": [[0, 1, 2, 3]],
            "attention_mask": [[1, 1, 1, 1]],
            "offset_mapping": [[(0, 0), (0, 5), (5, 23), (0, 0)]],
        }

    def decode(self, token_ids):
        return f"tok-{token_ids[0]}"


def test_predict_token_classification_returns_grouped_spans(monkeypatch):
    monkeypatch.setattr(
        server,
        "get_model",
        lambda model_name, pipeline_name, config_file: {
            "model": _FakeModel(),
            "tokenizer": _FakeTokenizer(),
        },
    )
    monkeypatch.setattr(server.mx.metal, "clear_cache", lambda: None)
    monkeypatch.setattr(server.gc, "collect", lambda: None)

    result = asyncio.run(
        server.predict(
            server.PredictionRequest(
                model="fake-privacy-filter",
                pipeline="token-classification",
                text="Email alice@example.com",
            )
        )
    )

    assert result["grouped_spans"][0][0]["entity_group"] == "private_email"
    assert result["grouped_spans"][0][0]["word"] == "alice@example.com"
    assert result["entities"] == result["grouped_spans"]
    assert result["predictions"][0][2]["label"] == "S-private_email"
    assert result["predictions"][0][2]["token"] == "tok-2"
    assert result["predictions"][0][2]["start"] == 5
    assert result["predictions"][0][2]["end"] == 23


def test_predict_token_classification_accepts_probabilities_without_logits(monkeypatch):
    monkeypatch.setattr(
        server,
        "get_model",
        lambda model_name, pipeline_name, config_file: {
            "model": _FakeProbabilitiesOnlyModel(),
            "tokenizer": _FakeTokenizer(),
        },
    )
    monkeypatch.setattr(server.mx.metal, "clear_cache", lambda: None)
    monkeypatch.setattr(server.gc, "collect", lambda: None)

    result = asyncio.run(
        server.predict(
            server.PredictionRequest(
                model="fake-pplx-token-classifier",
                pipeline="token-classification",
                text="Email alice@example.com",
            )
        )
    )

    assert result["grouped_spans"][0][0]["entity_group"] == "private_email"
    assert result["predictions"][0][2]["label"] == "S-private_email"


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://0.0.0.0:5173",
        "http://localhost:3000",
        "null",
    ],
)
def test_predict_cors_preflight_allows_local_demo_origins(origin):
    response = TestClient(server.app).options(
        "/predict",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
