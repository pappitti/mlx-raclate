from pathlib import Path

import pytest

from tools.prepare_finance_crm_ner_dataset import (
    _align_entities,
    _chunk_ranges,
    labels_for_scheme,
)


def test_bio_label_ontology_has_nine_labels():
    assert labels_for_scheme("BIO") == [
        "O",
        "B-PERSON",
        "I-PERSON",
        "B-ORGANIZATION",
        "I-ORGANIZATION",
        "B-FUND",
        "I-FUND",
        "B-EVENT",
        "I-EVENT",
    ]


def test_bio_alignment_uses_b_for_single_token_entities():
    labels = labels_for_scheme("BIO")
    label2id = {label: index for index, label in enumerate(labels)}

    aligned, token_entities, partial_boundaries = _align_entities(
        offsets=[(0, 5), (6, 10), (11, 15), (16, 20)],
        entities=[
            {"start": 0, "end": 5, "label": "PERSON"},
            {"start": 11, "end": 20, "label": "FUND"},
        ],
        source=Path("fixture.jsonl"),
        line_number=1,
        label2id=label2id,
        label_scheme="BIO",
    )

    assert [labels[label_id] for label_id in aligned] == [
        "B-PERSON",
        "O",
        "B-FUND",
        "I-FUND",
    ]
    assert token_entities == [0, None, 1, 1]
    assert partial_boundaries == 0


def test_default_bioes_alignment_remains_unchanged():
    labels = labels_for_scheme("BIOES")

    aligned, _, _ = _align_entities(
        offsets=[(0, 5), (6, 10), (11, 15), (16, 20)],
        entities=[
            {"start": 0, "end": 5, "label": "PERSON"},
            {"start": 11, "end": 20, "label": "FUND"},
        ],
        source=Path("fixture.jsonl"),
        line_number=1,
    )

    assert [labels[label_id] for label_id in aligned] == [
        "S-PERSON",
        "O",
        "B-FUND",
        "E-FUND",
    ]


def test_entity_safe_chunks_do_not_split_an_entity():
    assert _chunk_ranges([None, None, 0, 0, None], max_length=3) == [
        (0, 2),
        (2, 5),
    ]


def test_unknown_label_scheme_is_rejected():
    with pytest.raises(ValueError, match="BIO"):
        labels_for_scheme("BILOU")
