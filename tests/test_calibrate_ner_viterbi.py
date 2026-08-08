import json

import numpy as np
import pytest

from mlx_raclate.utils.token_classification import viterbi_decode_bioes_ids
from mlx_raclate.utils.token_classification_calibration import (
    LogitsCacheBuilder,
    decode_cached_bio_sequence,
    evaluate_bio_transition_biases,
    load_bio_calibration_corpus,
    load_logits_cache,
    search_bio_transition_biases,
    write_calibration_operating_points,
)


LABEL_NAMES = ("O", "B-email", "I-email")


def _write_fixture_cache(tmp_path):
    builder = LogitsCacheBuilder(
        LABEL_NAMES,
        metadata={"dataset": "fixture"},
    )
    builder.append(
        logits=[[[3.0, 0.0, 0.0], [1.0, 1.5, 0.0]]],
        references=[[0, 0]],
        attention_mask=[[1, 1]],
    )
    builder.append(
        logits=[
            [
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        ],
        references=[[0, 1, 2]],
        attention_mask=[[1, 1, 1]],
    )
    path = tmp_path / "fixture.npz"
    builder.save(path)
    return path


def test_logits_cache_roundtrip_preserves_variable_length_sequences(tmp_path):
    path = _write_fixture_cache(tmp_path)

    cache = load_logits_cache(path)

    assert cache.label_names == LABEL_NAMES
    assert cache.sequence_count == 2
    assert cache.token_count == 5
    assert cache.metadata["dataset"] == "fixture"
    assert cache.sequence(0)[0].shape == (2, 3)
    assert cache.sequence(1)[0].shape == (3, 3)


def test_calibration_rejects_caches_from_different_models(tmp_path):
    paths = []
    for model_name in ("model-a", "model-b"):
        builder = LogitsCacheBuilder(LABEL_NAMES, metadata={"model": model_name})
        builder.append(
            logits=[[[1.0, 0.0, 0.0]]],
            references=[[0]],
            attention_mask=[[1]],
        )
        path = tmp_path / f"{model_name}.npz"
        builder.save(path)
        paths.append(path)

    with pytest.raises(ValueError, match="same model checkpoint"):
        load_bio_calibration_corpus(paths)


def test_cached_numpy_decoder_matches_runtime_viterbi(tmp_path):
    builder = LogitsCacheBuilder(("O", "B-x", "I-x", "B-y", "I-y"))
    emissions = np.asarray(
        [
            [2.0, 1.0, 0.0, 0.5, 0.0],
            [0.2, 0.3, 1.4, 0.4, 1.2],
            [1.1, 0.8, 0.9, 0.7, 0.6],
            [0.4, 0.5, 1.0, 0.9, 1.3],
        ],
        dtype=np.float32,
    )
    builder.append(
        logits=emissions[None, :, :],
        references=[[0, 1, 2, 0]],
        attention_mask=[[1, 1, 1, 1]],
    )
    cache_path = tmp_path / "decoder.npz"
    builder.save(cache_path)
    corpus = load_bio_calibration_corpus([cache_path])
    biases = {
        "transition_bias_background_to_start": -0.25,
        "transition_bias_inside_to_continue": 0.3,
        "transition_bias_inside_to_background": 0.1,
        "transition_bias_inside_to_start": -0.4,
    }

    cached = decode_cached_bio_sequence(corpus, corpus.sequences[0], biases)
    runtime = viterbi_decode_bioes_ids(
        emissions,
        list(corpus.label_names),
        transition_biases=biases,
    )

    assert cached.tolist() == runtime


def test_bias_search_removes_false_positive_entity_start(tmp_path):
    cache_path = _write_fixture_cache(tmp_path)
    corpus = load_bio_calibration_corpus([cache_path])

    baseline = evaluate_bio_transition_biases(corpus, {})
    result = search_bio_transition_biases(
        corpus,
        grid=(-1.0, 0.0),
        passes=1,
        refine_step=0.0,
        search_max_sequences=None,
        finalists=2,
    )
    balanced = result["operating_points"]["balanced"]

    assert baseline["boundary_f1_micro"] == pytest.approx(2 / 3)
    assert balanced["calibration_metrics"]["boundary_f1_micro"] == 1.0
    assert balanced["biases"]["transition_bias_background_to_start"] == -1.0


def test_writer_merges_named_operating_points_with_existing_default(tmp_path):
    cache_path = _write_fixture_cache(tmp_path)
    corpus = load_bio_calibration_corpus([cache_path])
    result = search_bio_transition_biases(
        corpus,
        grid=(-1.0, 0.0),
        passes=1,
        refine_step=0.0,
        search_max_sequences=None,
        finalists=1,
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    calibration_path = checkpoint / "viterbi_calibration.json"
    calibration_path.write_text(
        json.dumps({"operating_points": {"default": {"biases": {}}}}),
        encoding="utf-8",
    )

    written = write_calibration_operating_points(checkpoint, result)
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert set(payload["operating_points"]) == {
        "default",
        "balanced",
        "high_precision",
        "high_recall",
    }
    assert "calibration_search" in payload


def test_writer_protects_existing_named_operating_points(tmp_path):
    cache_path = _write_fixture_cache(tmp_path)
    corpus = load_bio_calibration_corpus([cache_path])
    result = search_bio_transition_biases(
        corpus,
        grid=(0.0,),
        passes=1,
        refine_step=0.0,
        search_max_sequences=None,
        finalists=1,
    )
    output = tmp_path / "viterbi_calibration.json"
    write_calibration_operating_points(output, result)

    with pytest.raises(FileExistsError, match="already exist"):
        write_calibration_operating_points(output, result)
