import json

import pytest

from dev_tests.train_finance_crm_ner import (
    LABELS,
    _validate_checkpoint_labels,
    _validate_dataset_manifest,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_dataset_manifest_requires_matching_bio_length(tmp_path):
    _write_json(
        tmp_path / "manifest.json",
        {
            "task_type": "token-classification",
            "label_scheme": "BIO",
            "labels": list(LABELS),
            "max_length": 16384,
        },
    )

    assert _validate_dataset_manifest(tmp_path, 16384) == {
        index: label for index, label in enumerate(LABELS)
    }
    with pytest.raises(ValueError, match="silently truncate"):
        _validate_dataset_manifest(tmp_path, 8192)


def test_bioes_checkpoint_is_rejected_for_bio_training(tmp_path):
    expected = {index: label for index, label in enumerate(LABELS)}
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_json(
        checkpoint / "config.json",
        {"id2label": {str(index): label for index, label in expected.items()}},
    )
    _validate_checkpoint_labels(checkpoint, expected, "--checkpoint")

    _write_json(
        checkpoint / "config.json",
        {"id2label": {"0": "O", "1": "S-PERSON"}},
    )
    with pytest.raises(ValueError, match="BIOES checkpoint"):
        _validate_checkpoint_labels(checkpoint, expected, "--checkpoint")
