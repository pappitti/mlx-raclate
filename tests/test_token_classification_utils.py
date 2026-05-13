import pytest

from mlx_raclate.utils.token_classification import (
    decode_bioes_spans,
    decode_token_classification_batch,
    postprocess_token_classification_output,
    viterbi_decode_bioes_ids,
)


def test_decode_bioes_spans_groups_multi_token_entity():
    text = "Call Alice Smith tomorrow"
    offsets = [(0, 4), (5, 10), (11, 16), (17, 25)]
    labels = ["O", "B-private_person", "E-private_person", "O"]
    scores = [0.1, 0.9, 0.8, 0.2]

    spans = decode_bioes_spans(text, offsets, labels, scores)

    assert len(spans) == 1
    assert spans[0]["entity_group"] == "private_person"
    assert spans[0]["word"] == "Alice Smith"
    assert spans[0]["start"] == 5
    assert spans[0]["end"] == 16
    assert spans[0]["score"] == pytest.approx(0.85)


def test_decode_bioes_spans_handles_singleton_and_special_tokens():
    text = "Email alice@example.com"
    offsets = [(0, 0), (0, 5), (5, 23), (0, 0)]
    labels = ["O", "O", "S-private_email", "O"]
    scores = [0.0, 0.2, 0.95, 0.0]

    spans = decode_bioes_spans(text, offsets, labels, scores)

    assert len(spans) == 1
    assert spans[0]["entity_group"] == "private_email"
    assert spans[0]["word"] == "alice@example.com"
    assert spans[0]["start"] == 6
    assert spans[0]["end"] == 23
    assert spans[0]["score"] == pytest.approx(0.95)


def test_decode_token_classification_batch_decodes_multiple_texts():
    texts = ["A B", "C D"]
    offsets = [[(0, 1), (2, 3)], [(0, 1), (2, 3)]]
    labels = [["B-x", "E-x"], ["S-y", "O"]]
    scores = [[0.8, 0.6], [0.9, 0.1]]

    decoded = decode_token_classification_batch(texts, offsets, labels, scores)

    assert decoded[0][0]["entity_group"] == "x"
    assert decoded[0][0]["word"] == "A B"
    assert decoded[1][0]["entity_group"] == "y"
    assert decoded[1][0]["word"] == "C"


def test_viterbi_decode_bioes_ids_rejects_invalid_start_state():
    label_names = ["O", "B-name", "I-name", "E-name", "S-name"]
    emissions = [
        [0.0, 4.8, 5.0, 1.0, 4.0],
        [0.0, 0.5, 1.0, 5.0, 0.5],
    ]

    decoded = viterbi_decode_bioes_ids(emissions, label_names)

    assert decoded == [1, 3]


def test_viterbi_decode_bioes_ids_enforces_matching_entity_type():
    label_names = [
        "O",
        "B-x",
        "I-x",
        "E-x",
        "S-x",
        "B-y",
        "I-y",
        "E-y",
        "S-y",
    ]
    emissions = [
        [0.0, 9.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 9.0, 0.0],
    ]

    decoded = viterbi_decode_bioes_ids(emissions, label_names)

    assert decoded == [5, 7]


def test_viterbi_decode_bioes_ids_forces_special_tokens_to_outside():
    label_names = ["O", "B-email", "I-email", "E-email", "S-email"]
    emissions = [
        [0.0, 10.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 9.0],
    ]

    decoded = viterbi_decode_bioes_ids(
        emissions,
        label_names,
        special_tokens_mask=[True, False],
    )

    assert decoded == [0, 4]


def test_viterbi_decode_bioes_ids_applies_transition_biases():
    label_names = ["O", "B-email", "E-email", "S-email"]
    emissions = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]

    decoded = viterbi_decode_bioes_ids(
        emissions,
        label_names,
        transition_biases={"transition_bias_background_to_start": 3.0},
    )

    assert decoded == [0, 3]


def test_postprocess_token_classification_output_decodes_predictions_and_spans():
    id2label = {
        "0": "O",
        "1": "B-private_person",
        "2": "E-private_person",
        "3": "S-private_email",
    }
    logits = [
        [
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
        ]
    ]
    text = "Hi Alice Smith"
    offsets = [[(0, 2), (2, 8), (8, 14), (0, 0)]]

    result = postprocess_token_classification_output(
        logits=logits,
        id2label=id2label,
        texts=[text],
        offsets=offsets,
    )

    assert result["decoded_label_ids"] == [[0, 1, 2, 0]]
    assert result["predictions"][0][1]["label"] == "B-private_person"
    assert result["grouped_spans"][0][0]["entity_group"] == "private_person"
    assert result["grouped_spans"][0][0]["word"] == "Alice Smith"
    assert result["grouped_spans"][0][0]["start"] == 3
    assert result["grouped_spans"][0][0]["end"] == 14
