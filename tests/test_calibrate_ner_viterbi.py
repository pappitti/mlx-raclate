import json
from types import SimpleNamespace

import numpy as np
import pytest

from mlx_raclate.ner import calibrate as calibration_cli
from mlx_raclate.ner import evaluate as evaluation_cli
from mlx_raclate.ner.calibrate import (
    load_bio_calibration_corpus,
    search_bio_transition_biases,
    write_calibration_operating_points,
)
from mlx_raclate.utils.token_classification import viterbi_decode_bioes_ids
from mlx_raclate.utils.token_classification_calibration import (
    LogitsCacheBuilder,
    decode_cached_bio_sequence,
    evaluate_bio_transition_biases,
    load_logits_cache,
)


LABEL_NAMES = ("O", "B-email", "I-email")


def _fixture_metrics():
    return {
        "boundary_precision_micro": 1.0,
        "boundary_recall_micro": 1.0,
        "boundary_f1_micro": 1.0,
        "boundary_f1_macro": 1.0,
        "predicted_spans": 1,
        "reference_spans": 1,
    }


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


def test_writer_preserves_a_default_decoder_for_new_calibration(tmp_path):
    output = tmp_path / "viterbi_calibration.json"
    write_calibration_operating_points(
        output,
        {
            "operating_points": {
                "balanced": {
                    "biases": {"transition_bias_background_to_start": -1.0}
                }
            }
        },
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["operating_points"]["default"]["biases"] == {
        "transition_bias_background_stay": 0.0,
        "transition_bias_background_to_start": 0.0,
        "transition_bias_inside_to_continue": 0.0,
        "transition_bias_inside_to_end": 0.0,
        "transition_bias_inside_to_background": 0.0,
        "transition_bias_inside_to_start": 0.0,
        "transition_bias_end_to_background": 0.0,
        "transition_bias_end_to_start": 0.0,
    }


def test_one_step_calibration_generates_temporary_caches_and_updates_checkpoint(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    captured = {}

    def fake_run_evaluation(request, *, emit_progress, emit_json):
        captured["request"] = request
        captured["emit_progress"] = emit_progress
        captured["emit_json"] = emit_json
        cache_paths = tuple(
            request.logits_cache_dir / f"dataset-{index}.npz" for index in range(2)
        )
        for path in cache_paths:
            path.touch()
        captured["cache_dir"] = request.logits_cache_dir
        return evaluation_cli.EvaluationRun(
            benchmark={"results": []},
            logits_cache_paths=cache_paths,
        )

    metrics = _fixture_metrics()
    search_result = {
        "operating_points": {
            name: {"biases": {}, "calibration_metrics": metrics}
            for name in ("balanced", "high_precision", "high_recall")
        },
        "search": {"zero_bias_metrics": metrics},
    }
    monkeypatch.setattr(evaluation_cli, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(
        calibration_cli,
        "load_bio_calibration_corpus",
        lambda paths: SimpleNamespace(
            sequences=[object()],
            label_names=LABEL_NAMES,
            cache_metadata=[{"dataset": str(path)} for path in paths],
        ),
    )
    monkeypatch.setattr(
        calibration_cli,
        "search_bio_transition_biases",
        lambda *args, **kwargs: search_result,
    )

    written_path = calibration_cli.main(
        [
            "--model",
            str(checkpoint),
            "--dataset",
            "dataset-a",
            "dataset-b",
        ]
    )

    assert written_path == checkpoint / "viterbi_calibration.json"
    assert written_path.is_file()
    assert captured["request"].model == (str(checkpoint),)
    assert captured["request"].dataset == ("dataset-a", "dataset-b")
    assert captured["emit_progress"] is False
    assert captured["emit_json"] is False
    assert not captured["cache_dir"].exists()
    payload = json.loads(written_path.read_text(encoding="utf-8"))
    assert payload["calibration_search"]["generated_from"]["model"] == str(
        checkpoint
    )
    assert "cache_files" not in payload["calibration_search"]


def test_one_step_calibration_requires_a_dataset(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    with pytest.raises(SystemExit, match="2"):
        calibration_cli.main(["--model", str(checkpoint)])


def test_remote_model_requires_an_explicit_calibration_output():
    with pytest.raises(SystemExit, match="2"):
        calibration_cli.main(
            ["--model", "org/model", "--dataset", "org/calibration-data"]
        )


def test_cache_mode_remains_available_for_repeated_searches(tmp_path):
    cache_path = _write_fixture_cache(tmp_path)
    output = tmp_path / "reused-cache-calibration.json"

    written_path = calibration_cli.main(
        [
            "--cache",
            str(cache_path),
            "--output-calibration",
            str(output),
            "--grid",
            "0",
            "--passes",
            "1",
            "--refine-step",
            "0",
            "--search-max-sequences",
            "1",
            "--finalists",
            "1",
        ]
    )

    assert written_path == output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["calibration_search"]["cache_files"] == [str(cache_path)]
