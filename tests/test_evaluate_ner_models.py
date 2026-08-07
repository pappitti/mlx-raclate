import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "tools" / "evaluate_ner_models.py"
SPEC = importlib.util.spec_from_file_location("evaluate_ner_models", SCRIPT_PATH)
evaluate_ner_models = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = evaluate_ner_models
SPEC.loader.exec_module(evaluate_ner_models)


def test_discovers_all_phase_validation_files(tmp_path):
    for phase in ("generalization", "lexical_calibration"):
        phase_dir = tmp_path / phase
        phase_dir.mkdir()
        (phase_dir / "validation.parquet").touch()

    sources = evaluate_ner_models.discover_evaluation_sources(
        [str(tmp_path)], "validation"
    )

    assert [source.name for source in sources] == [
        "generalization",
        "lexical_calibration",
    ]


def test_remaps_dataset_label_ids_by_semantic_name():
    remapped = evaluate_ner_models.remap_reference_sequences(
        [[0, 1, 2, -100]],
        dataset_labels=["O", "B-PERSON", "E-PERSON"],
        model_label2id={"B-PERSON": 0, "E-PERSON": 1, "O": 2},
    )

    assert remapped == [[2, 0, 1, -100]]


def test_remapping_rejects_labels_the_model_cannot_predict():
    with pytest.raises(ValueError, match="missing from the model"):
        evaluate_ner_models.remap_reference_sequences(
            [[1]],
            dataset_labels=["O", "S-FUND"],
            model_label2id={"O": 0, "S-PERSON": 1},
        )
