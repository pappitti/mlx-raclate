import pytest

from mlx_raclate.utils import hub


pytestmark = pytest.mark.release


def test_get_model_path_returns_existing_local_path(tmp_path):
    assert hub.get_model_path(str(tmp_path)) == tmp_path


def test_get_model_path_forwards_narrow_download_contract(monkeypatch, tmp_path):
    downloaded_path = tmp_path / "downloaded"
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(downloaded_path)

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)

    assert hub.get_model_path(
        "organization/model",
        revision="revision",
        allow_patterns=hub.TOKENIZER_ALLOW_PATTERNS,
    ) == downloaded_path
    assert calls == [
        {
            "repo_id": "organization/model",
            "revision": "revision",
            "allow_patterns": list(hub.TOKENIZER_ALLOW_PATTERNS),
        }
    ]


def test_get_model_path_wraps_hub_errors(monkeypatch):
    def snapshot_download(**kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)

    with pytest.raises(hub.ModelNotFoundError, match="organization/model"):
        hub.get_model_path("organization/model")
