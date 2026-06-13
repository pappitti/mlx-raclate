import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


_INVALID_PATH_SCORE = -1e12
VITERBI_CALIBRATION_FILE = "viterbi_calibration.json"
VITERBI_BIAS_KEYS = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)


def zero_viterbi_transition_biases() -> Dict[str, float]:
    return {key: 0.0 for key in VITERBI_BIAS_KEYS}


def default_viterbi_calibration(operating_point: str = "default") -> Dict[str, Any]:
    return {
        "operating_points": {
            operating_point: {
                "biases": zero_viterbi_transition_biases(),
            }
        }
    }


def viterbi_transition_biases_from_calibration(
    calibration: Optional[Mapping[str, Any]],
    operating_point: str = "default",
) -> Dict[str, float]:
    if calibration is None:
        return zero_viterbi_transition_biases()

    operating_points = calibration.get("operating_points")
    if not isinstance(operating_points, Mapping):
        raise ValueError("Viterbi calibration must contain an 'operating_points' mapping")

    if operating_point not in operating_points:
        raise ValueError(f"Viterbi calibration operating point not found: {operating_point}")

    operating_point_config = operating_points[operating_point]
    if not isinstance(operating_point_config, Mapping):
        raise ValueError(f"Viterbi calibration operating point {operating_point!r} must be a mapping")

    biases = operating_point_config.get("biases", {})
    if not isinstance(biases, Mapping):
        raise ValueError(f"Viterbi calibration operating point {operating_point!r} has invalid biases")

    return _resolve_viterbi_transition_biases(biases)


def load_viterbi_calibration(path_or_hf_repo: str | Path) -> Optional[Dict[str, Any]]:
    from mlx_raclate.utils.utils import get_model_path

    model_path = get_model_path(str(path_or_hf_repo))
    calibration_path = model_path / VITERBI_CALIBRATION_FILE
    if not calibration_path.exists():
        return None

    with open(calibration_path) as f:
        return json.load(f)


def save_viterbi_calibration(
    output_dir: str | Path,
    calibration: Optional[Mapping[str, Any]] = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(calibration or default_viterbi_calibration())
    with open(output_dir / VITERBI_CALIBRATION_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def _resolve_viterbi_transition_biases(
    transition_biases: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    resolved = zero_viterbi_transition_biases()
    if transition_biases is None:
        return resolved

    for key, value in transition_biases.items():
        if key not in resolved:
            raise ValueError(f"Unknown Viterbi transition bias: {key}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Viterbi transition bias {key} must be numeric")
        resolved[key] = float(value)
    return resolved


def _split_bioes_label(label: str) -> Tuple[Optional[str], str]:
    if label == "O":
        return None, "O"
    if "-" not in label:
        return None, label
    prefix, entity_group = label.split("-", 1)
    if prefix not in {"B", "I", "O", "E", "S"}:
        return None, label
    return prefix, entity_group


def _aggregate_scores(scores: Sequence[float], strategy: str = "mean") -> float:
    if not scores:
        return 0.0
    if strategy == "max":
        return max(scores)
    if strategy == "min":
        return min(scores)
    return sum(scores) / len(scores)


def _argmax_index(scores: Sequence[float]) -> int:
    return max(range(len(scores)), key=lambda index: float(scores[index]))


def _infer_tag_scheme(label_names: Sequence[str]) -> Optional[str]:
    prefixes = set()
    for label in label_names:
        if label == "O":
            continue
        prefix, _ = _split_bioes_label(label)
        if prefix is None:
            return None
        prefixes.add(prefix)

    if not prefixes:
        return None
    if prefixes & {"E", "S"}:
        return "BIOES"
    if prefixes & {"B", "I"}:
        return "BIO"
    return None


def _is_valid_start_label(label: str, scheme: str) -> bool:
    if label == "O":
        return True

    prefix, _ = _split_bioes_label(label)
    if prefix is None:
        return True

    if scheme == "BIOES":
        return prefix in {"B", "S"}
    return prefix == "B"


def _is_valid_end_label(label: str, scheme: str) -> bool:
    if label == "O":
        return True

    prefix, _ = _split_bioes_label(label)
    if prefix is None:
        return True

    if scheme == "BIOES":
        return prefix in {"E", "S"}
    return prefix in {"B", "I"}


def _is_valid_transition(previous_label: str, current_label: str, scheme: str) -> bool:
    previous_prefix, previous_entity = _split_bioes_label(previous_label)
    current_prefix, current_entity = _split_bioes_label(current_label)

    if current_label == "O" or current_prefix is None:
        return scheme != "BIOES" or previous_prefix not in {"B", "I"}

    if previous_label == "O" or previous_prefix is None:
        if scheme == "BIOES":
            return current_prefix in {"B", "S"}
        return current_prefix == "B"

    if scheme == "BIOES":
        if previous_prefix in {"B", "I"}:
            return current_prefix in {"I", "E"} and previous_entity == current_entity
        return current_prefix in {"B", "S"}

    if current_prefix == "I":
        return previous_prefix in {"B", "I"} and previous_entity == current_entity
    return current_prefix == "B"


def _transition_bias(
    previous_label: str,
    current_label: str,
    scheme: str,
    transition_biases: Mapping[str, float],
) -> float:
    if scheme != "BIOES":
        return 0.0

    previous_prefix, previous_entity = _split_bioes_label(previous_label)
    current_prefix, current_entity = _split_bioes_label(current_label)
    previous_is_background = previous_label == "O" or previous_prefix is None
    current_is_background = current_label == "O" or current_prefix is None

    if previous_is_background:
        if current_is_background:
            return transition_biases["transition_bias_background_stay"]
        if current_prefix in {"B", "S"}:
            return transition_biases["transition_bias_background_to_start"]
        return 0.0

    if previous_prefix in {"B", "I"} and previous_entity == current_entity:
        if current_prefix == "I":
            return transition_biases["transition_bias_inside_to_continue"]
        if current_prefix == "E":
            return transition_biases["transition_bias_inside_to_end"]
        return 0.0

    if previous_prefix in {"E", "S"}:
        if current_is_background:
            return transition_biases["transition_bias_end_to_background"]
        if current_prefix in {"B", "S"}:
            return transition_biases["transition_bias_end_to_start"]

    return 0.0


def viterbi_decode_bioes_ids(
    emissions: Sequence[Sequence[float]],
    label_names: Sequence[str],
    special_tokens_mask: Optional[Sequence[bool]] = None,
    outside_label: str = "O",
    transition_biases: Optional[Mapping[str, float]] = None,
) -> List[int]:
    """Decode a single label sequence with BIO/BIOES transition constraints."""
    if not emissions:
        return []

    scheme = _infer_tag_scheme(label_names)
    outside_index = label_names.index(outside_label) if outside_label in label_names else None
    greedy_ids = [_argmax_index(step_scores) for step_scores in emissions]

    if scheme is None:
        if outside_index is None or special_tokens_mask is None:
            return greedy_ids
        return [
            outside_index if index < len(special_tokens_mask) and special_tokens_mask[index] else greedy_id
            for index, greedy_id in enumerate(greedy_ids)
        ]

    num_labels = len(label_names)
    num_tokens = len(emissions)
    special_mask = list(special_tokens_mask or [])
    biases = _resolve_viterbi_transition_biases(transition_biases)

    scores = [[_INVALID_PATH_SCORE] * num_labels for _ in range(num_tokens)]
    backpointers = [[-1] * num_labels for _ in range(num_tokens)]

    for label_index, label_name in enumerate(label_names):
        if special_mask[:1] and special_mask[0] and outside_index is not None and label_index != outside_index:
            continue
        if not _is_valid_start_label(label_name, scheme):
            continue
        scores[0][label_index] = float(emissions[0][label_index])

    for token_index in range(1, num_tokens):
        force_outside = (
            token_index < len(special_mask)
            and special_mask[token_index]
            and outside_index is not None
        )

        for current_index, current_label in enumerate(label_names):
            if force_outside and current_index != outside_index:
                continue

            emission_score = float(emissions[token_index][current_index])
            best_score = _INVALID_PATH_SCORE
            best_previous_index = -1

            for previous_index, previous_label in enumerate(label_names):
                previous_score = scores[token_index - 1][previous_index]
                if previous_score <= _INVALID_PATH_SCORE / 2:
                    continue
                if not _is_valid_transition(previous_label, current_label, scheme):
                    continue

                candidate_score = (
                    previous_score
                    + _transition_bias(previous_label, current_label, scheme, biases)
                    + emission_score
                )
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_previous_index = previous_index

            scores[token_index][current_index] = best_score
            backpointers[token_index][current_index] = best_previous_index

    best_final_score = _INVALID_PATH_SCORE
    best_final_index = -1
    for label_index, label_name in enumerate(label_names):
        if scores[-1][label_index] <= _INVALID_PATH_SCORE / 2:
            continue
        if not _is_valid_end_label(label_name, scheme):
            continue
        if scores[-1][label_index] > best_final_score:
            best_final_score = scores[-1][label_index]
            best_final_index = label_index

    if best_final_index < 0:
        if outside_index is None or special_tokens_mask is None:
            return greedy_ids
        return [
            outside_index if index < len(special_tokens_mask) and special_tokens_mask[index] else greedy_id
            for index, greedy_id in enumerate(greedy_ids)
        ]

    decoded_ids = [best_final_index]
    for token_index in range(num_tokens - 1, 0, -1):
        best_final_index = backpointers[token_index][best_final_index]
        if best_final_index < 0:
            return greedy_ids
        decoded_ids.append(best_final_index)

    decoded_ids.reverse()
    return decoded_ids


def viterbi_decode_bioes_batch(
    emissions: Sequence[Sequence[Sequence[float]]],
    label_names: Sequence[str],
    special_tokens_mask: Optional[Sequence[Sequence[bool]]] = None,
    outside_label: str = "O",
    transition_biases: Optional[Mapping[str, float]] = None,
) -> List[List[int]]:
    """Decode a batch of label sequences with BIO/BIOES transition constraints."""
    decoded: List[List[int]] = []
    for index, sequence_emissions in enumerate(emissions):
        decoded.append(
            viterbi_decode_bioes_ids(
                emissions=sequence_emissions,
                label_names=label_names,
                special_tokens_mask=None if special_tokens_mask is None else special_tokens_mask[index],
                outside_label=outside_label,
                transition_biases=transition_biases,
            )
        )
    return decoded


def ordered_label_names(id2label: Mapping[Any, str], num_labels: int) -> List[str]:
    """Return label names ordered by numeric class id."""
    ordered = []
    for index in range(num_labels):
        if index in id2label:
            ordered.append(id2label[index])
        elif str(index) in id2label:
            ordered.append(id2label[str(index)])
        else:
            raise KeyError(f"id2label is missing label id {index}")
    return ordered


def decode_bioes_spans(
    text: str,
    offsets: Sequence[Sequence[int] | Tuple[int, int]],
    labels: Sequence[str],
    scores: Optional[Sequence[float]] = None,
    score_aggregation: str = "mean",
    trim_span_whitespace: bool = True,
    discard_overlapping: bool = False,
) -> List[Dict[str, Any]]:
    """
    Decode BIOES token labels into grouped entity spans.

    Returns spans in a format close to the Hugging Face token-classification
    pipeline and OpenMed's privacy-filter output.
    """
    spans: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    def append_span(entity_group: str, start: int, end: int, span_scores: Sequence[float]):
        if trim_span_whitespace:
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1

        if end <= start:
            return

        spans.append(
            {
                "entity_group": entity_group,
                "score": _aggregate_scores(span_scores, score_aggregation),
                "word": text[start:end],
                "start": start,
                "end": end,
            }
        )

    def flush_current():
        nonlocal current
        if current is None:
            return
        append_span(
            entity_group=current["entity_group"],
            start=current["start"],
            end=current["end"],
            span_scores=current.pop("_scores"),
        )
        current = None

    for index, raw_offset in enumerate(offsets):
        if index >= len(labels):
            break

        start, end = int(raw_offset[0]), int(raw_offset[1])
        label = labels[index]
        score = float(scores[index]) if scores is not None and index < len(scores) else None

        # Skip special tokens and empty spans.
        if end <= start:
            continue

        prefix, entity_group = _split_bioes_label(label)

        if label == "O":
            flush_current()
            continue

        if prefix is None:
            flush_current()
            append_span(entity_group, start, end, [] if score is None else [score])
            continue

        if prefix == "S":
            flush_current()
            append_span(entity_group, start, end, [] if score is None else [score])
            continue

        if prefix == "B":
            flush_current()
            current = {
                "entity_group": entity_group,
                "start": start,
                "end": end,
                "_scores": [] if score is None else [score],
            }
            continue

        if current is None or current["entity_group"] != entity_group:
            flush_current()
            current = {
                "entity_group": entity_group,
                "start": start,
                "end": end,
                "_scores": [] if score is None else [score],
            }
        else:
            current["end"] = end
            if score is not None:
                current["_scores"].append(score)

        if prefix == "E":
            flush_current()

    flush_current()
    if discard_overlapping:
        spans = _discard_overlapping_spans(spans)
    return spans


def _discard_overlapping_spans(spans: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(spans, key=lambda span: (span["start"], -(span["end"] - span["start"]), span["entity_group"]))
    kept: List[Dict[str, Any]] = []
    cursor = 0
    for span in ordered:
        if span["start"] < cursor or span["end"] <= span["start"]:
            continue
        kept.append(dict(span))
        cursor = span["end"]
    return kept


def decode_token_classification_batch(
    texts: Sequence[str],
    offsets: Sequence[Sequence[Sequence[int] | Tuple[int, int]]],
    labels: Sequence[Sequence[str]],
    scores: Optional[Sequence[Sequence[float]]] = None,
    score_aggregation: str = "mean",
    trim_span_whitespace: bool = True,
    discard_overlapping: bool = False,
) -> List[List[Dict[str, Any]]]:
    """Decode a batch of token-classification predictions into grouped spans."""
    decoded: List[List[Dict[str, Any]]] = []
    for index, text in enumerate(texts):
        decoded.append(
            decode_bioes_spans(
                text=text,
                offsets=offsets[index],
                labels=labels[index],
                scores=None if scores is None else scores[index],
                score_aggregation=score_aggregation,
                trim_span_whitespace=trim_span_whitespace,
                discard_overlapping=discard_overlapping,
            )
        )
    return decoded


def bioes_tags_to_spans(
    labels: Sequence[str],
    outside_label: str = "O",
) -> List[Dict[str, Any]]:
    """Decode BIO/BIOES token labels into token-indexed entity spans."""
    spans: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    def flush_current():
        nonlocal current
        if current is None:
            return
        spans.append(current)
        current = None

    for token_index, label in enumerate(labels):
        if label == outside_label:
            flush_current()
            continue

        prefix, entity_group = _split_bioes_label(label)
        if prefix is None:
            flush_current()
            spans.append(
                {
                    "entity_group": label,
                    "start": token_index,
                    "end": token_index + 1,
                }
            )
            continue

        if prefix == "S":
            flush_current()
            spans.append(
                {
                    "entity_group": entity_group,
                    "start": token_index,
                    "end": token_index + 1,
                }
            )
            continue

        if prefix == "B":
            flush_current()
            current = {
                "entity_group": entity_group,
                "start": token_index,
                "end": token_index + 1,
            }
            continue

        if current is None or current["entity_group"] != entity_group:
            flush_current()
            current = {
                "entity_group": entity_group,
                "start": token_index,
                "end": token_index + 1,
            }
        else:
            current["end"] = token_index + 1

        if prefix == "E":
            flush_current()

    flush_current()
    return spans


def compute_token_classification_metrics(
    predictions,
    references,
    *,
    id2label: Optional[Mapping[Any, str]] = None,
    ignore_index: int = -100,
    outside_label: str = "O",
    labels: Optional[Sequence[Any]] = None,
    include_outside_in_macro: bool = False,
) -> Dict[str, Any]:
    """
    Compute token and exact-boundary metrics for token classification.

    `predictions` and `references` should be batched sequences of label ids or
    label names. Positions with `reference == ignore_index` are skipped.
    Boundary F1 decodes BIO/BIOES labels and requires entity type plus token
    start/end boundaries to match exactly.
    """
    prediction_sequences = _to_list(predictions)
    reference_sequences = _to_list(references)
    _validate_parallel_label_sequences(prediction_sequences, reference_sequences)

    token_pairs = _filtered_token_label_pairs(
        prediction_sequences,
        reference_sequences,
        id2label=id2label,
        ignore_index=ignore_index,
    )
    total_tokens = len(token_pairs)
    correct_tokens = sum(1 for predicted, expected in token_pairs if predicted == expected)
    token_accuracy = correct_tokens / total_tokens if total_tokens else 0.0

    token_labels = _metric_label_set(
        token_pairs=token_pairs,
        labels=labels,
        id2label=id2label,
        outside_label=outside_label,
        include_outside=include_outside_in_macro,
    )
    token_scores = _precision_recall_f1_from_pairs(token_pairs, token_labels)

    predicted_spans, reference_spans = _span_sets_from_label_sequences(
        prediction_sequences,
        reference_sequences,
        id2label=id2label,
        ignore_index=ignore_index,
        outside_label=outside_label,
    )
    boundary_labels = _boundary_metric_label_set(
        predicted_spans,
        reference_spans,
        labels=labels,
        id2label=id2label,
        outside_label=outside_label,
    )
    boundary_scores = _precision_recall_f1_from_spans(
        predicted_spans,
        reference_spans,
        boundary_labels,
    )

    return {
        "token_accuracy": token_accuracy,
        "token_correct": correct_tokens,
        "token_total": total_tokens,
        "token_precision_macro": token_scores["precision_macro"],
        "token_recall_macro": token_scores["recall_macro"],
        "token_f1_macro": token_scores["f1_macro"],
        "token_precision_micro": token_scores["precision_micro"],
        "token_recall_micro": token_scores["recall_micro"],
        "token_f1_micro": token_scores["f1_micro"],
        "token_f1_by_label": token_scores["f1_by_label"],
        "boundary_precision_macro": boundary_scores["precision_macro"],
        "boundary_recall_macro": boundary_scores["recall_macro"],
        "boundary_f1_macro": boundary_scores["f1_macro"],
        "boundary_precision_micro": boundary_scores["precision_micro"],
        "boundary_recall_micro": boundary_scores["recall_micro"],
        "boundary_f1_micro": boundary_scores["f1_micro"],
        "boundary_f1_by_label": boundary_scores["f1_by_label"],
        "macro_bf1": boundary_scores["f1_macro"],
        "micro_bf1": boundary_scores["f1_micro"],
        "predicted_spans": len(predicted_spans),
        "reference_spans": len(reference_spans),
    }


def _validate_parallel_label_sequences(predictions, references) -> None:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same batch size")

    for index, (prediction, reference) in enumerate(zip(predictions, references)):
        if len(prediction) != len(reference):
            raise ValueError(
                f"predictions and references differ in sequence length at batch index {index}"
            )


def _is_ignored_label(label, ignore_index: int) -> bool:
    if label is None:
        return True
    return label == ignore_index or str(label) == str(ignore_index)


def _label_to_name(label, id2label: Optional[Mapping[Any, str]]) -> str:
    if id2label is None:
        return str(label)

    if label in id2label:
        return id2label[label]

    label_key = str(label)
    if label_key in id2label:
        return id2label[label_key]

    try:
        int_key = int(label)
    except (TypeError, ValueError):
        return str(label)

    if int_key in id2label:
        return id2label[int_key]

    return str(label)


def _filtered_token_label_pairs(
    predictions,
    references,
    *,
    id2label: Optional[Mapping[Any, str]],
    ignore_index: int,
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for prediction_sequence, reference_sequence in zip(predictions, references):
        for predicted, expected in zip(prediction_sequence, reference_sequence):
            if _is_ignored_label(expected, ignore_index):
                continue
            pairs.append((
                _label_to_name(predicted, id2label),
                _label_to_name(expected, id2label),
            ))
    return pairs


def _metric_label_set(
    *,
    token_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    span_sets: Optional[Tuple[set, set]] = None,
    labels: Optional[Sequence[Any]],
    id2label: Optional[Mapping[Any, str]],
    outside_label: str,
    include_outside: bool,
) -> List[str]:
    if labels is not None:
        label_set = {_label_to_name(label, id2label) for label in labels}
    else:
        label_set = set()
        if token_pairs is not None:
            for predicted, expected in token_pairs:
                label_set.add(predicted)
                label_set.add(expected)
        if span_sets is not None:
            for spans in span_sets:
                label_set.update(span[1] for span in spans)

    if not include_outside:
        label_set.discard(outside_label)
    return sorted(label_set)


def _boundary_metric_label_set(
    predicted_spans: set,
    reference_spans: set,
    *,
    labels: Optional[Sequence[Any]],
    id2label: Optional[Mapping[Any, str]],
    outside_label: str,
) -> List[str]:
    if labels is None:
        label_set = {
            span[1]
            for spans in (predicted_spans, reference_spans)
            for span in spans
        }
        return sorted(label_set)

    label_set = set()
    for label in labels:
        label_name = _label_to_name(label, id2label)
        if label_name == outside_label:
            continue
        prefix, entity_group = _split_bioes_label(label_name)
        label_set.add(entity_group if prefix is not None else label_name)
    return sorted(label_set)


def _precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _average_prf(per_label: Mapping[str, Tuple[int, int, int]]) -> Dict[str, Any]:
    precisions = []
    recalls = []
    f1s = []
    f1_by_label = {}
    total_tp = total_fp = total_fn = 0

    for label, (tp, fp, fn) in per_label.items():
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        f1_by_label[label] = f1
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_precision, micro_recall, micro_f1 = _precision_recall_f1(
        total_tp,
        total_fp,
        total_fn,
    )
    label_count = len(per_label)
    return {
        "precision_macro": sum(precisions) / label_count if label_count else 0.0,
        "recall_macro": sum(recalls) / label_count if label_count else 0.0,
        "f1_macro": sum(f1s) / label_count if label_count else 0.0,
        "precision_micro": micro_precision,
        "recall_micro": micro_recall,
        "f1_micro": micro_f1,
        "f1_by_label": f1_by_label,
    }


def _precision_recall_f1_from_pairs(
    token_pairs: Sequence[Tuple[str, str]],
    labels: Sequence[str],
) -> Dict[str, Any]:
    per_label = {label: [0, 0, 0] for label in labels}
    for predicted, expected in token_pairs:
        if predicted == expected:
            if expected in per_label:
                per_label[expected][0] += 1
            continue
        if predicted in per_label:
            per_label[predicted][1] += 1
        if expected in per_label:
            per_label[expected][2] += 1

    return _average_prf({
        label: (counts[0], counts[1], counts[2])
        for label, counts in per_label.items()
    })


def _span_sets_from_label_sequences(
    predictions,
    references,
    *,
    id2label: Optional[Mapping[Any, str]],
    ignore_index: int,
    outside_label: str,
) -> Tuple[set, set]:
    predicted_spans = set()
    reference_spans = set()

    for sequence_index, (prediction_sequence, reference_sequence) in enumerate(zip(predictions, references)):
        predicted_labels = []
        reference_labels = []
        for predicted, expected in zip(prediction_sequence, reference_sequence):
            if _is_ignored_label(expected, ignore_index):
                continue
            predicted_labels.append(_label_to_name(predicted, id2label))
            reference_labels.append(_label_to_name(expected, id2label))

        for span in bioes_tags_to_spans(predicted_labels, outside_label=outside_label):
            predicted_spans.add((
                sequence_index,
                span["entity_group"],
                span["start"],
                span["end"],
            ))
        for span in bioes_tags_to_spans(reference_labels, outside_label=outside_label):
            reference_spans.add((
                sequence_index,
                span["entity_group"],
                span["start"],
                span["end"],
            ))

    return predicted_spans, reference_spans


def _precision_recall_f1_from_spans(
    predicted_spans: set,
    reference_spans: set,
    labels: Sequence[str],
) -> Dict[str, Any]:
    per_label = {}
    for label in labels:
        predicted_for_label = {span for span in predicted_spans if span[1] == label}
        reference_for_label = {span for span in reference_spans if span[1] == label}
        true_positive = len(predicted_for_label & reference_for_label)
        false_positive = len(predicted_for_label - reference_for_label)
        false_negative = len(reference_for_label - predicted_for_label)
        per_label[label] = (true_positive, false_positive, false_negative)

    return _average_prf(per_label)


def _to_list(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    max_value = max(float(value) for value in values)
    exps = [math.exp(float(value) - max_value) for value in values]
    total = sum(exps)
    if total == 0.0:
        return [0.0 for _ in exps]
    return [value / total for value in exps]


def _softmax_batch(logits: Sequence[Sequence[Sequence[float]]]) -> List[List[List[float]]]:
    return [[_softmax(token_logits) for token_logits in sequence] for sequence in logits]


def postprocess_token_classification_output(
    *,
    logits,
    id2label: Optional[Mapping[Any, str]] = None,
    probabilities=None,
    texts: Optional[Sequence[str]] = None,
    offsets: Optional[Sequence[Sequence[Sequence[int] | Tuple[int, int]]]] = None,
    special_tokens_mask: Optional[Sequence[Sequence[bool]]] = None,
    decode_mode: str = "viterbi",
    transition_biases: Optional[Mapping[str, float]] = None,
    score_aggregation: str = "mean",
    trim_span_whitespace: bool = True,
    discard_overlapping: bool = False,
) -> Dict[str, Any]:
    """
    Convert token-classification logits into token predictions and grouped spans.

    `logits` and `probabilities` can be MLX arrays, NumPy-like arrays, or nested
    Python sequences with shape `[batch, sequence, labels]`.
    """
    logits_list = _to_list(logits)
    if not logits_list:
        return {
            "predictions": [],
            "grouped_spans": [] if texts is not None and offsets is not None else None,
            "decoded_label_ids": [],
            "label_names": [],
        }

    probabilities_list = _to_list(probabilities) if probabilities is not None else _softmax_batch(logits_list)
    num_labels = len(logits_list[0][0]) if logits_list and logits_list[0] else 0
    label_names = (
        ordered_label_names(id2label, num_labels)
        if id2label is not None
        else [str(index) for index in range(num_labels)]
    )

    if special_tokens_mask is None and offsets is not None:
        special_tokens_mask = [
            [int(start) >= int(end) for start, end in sequence_offsets]
            for sequence_offsets in offsets
        ]

    if decode_mode == "viterbi" and id2label is not None:
        decoded_label_ids = viterbi_decode_bioes_batch(
            emissions=logits_list,
            label_names=label_names,
            special_tokens_mask=special_tokens_mask,
            transition_biases=transition_biases,
        )
    elif decode_mode == "argmax" or id2label is None:
        decoded_label_ids = [
            [_argmax_index(token_logits) for token_logits in sequence_logits]
            for sequence_logits in logits_list
        ]
    else:
        raise ValueError(f"Unsupported decode_mode: {decode_mode!r}")

    predictions: List[List[Dict[str, Any]]] = []
    grouped_labels: List[List[str]] = []
    grouped_scores: List[List[float]] = []
    for sequence_index, sequence_ids in enumerate(decoded_label_ids):
        sequence_predictions: List[Dict[str, Any]] = []
        sequence_labels: List[str] = []
        sequence_scores: List[float] = []

        for token_index, label_id in enumerate(sequence_ids):
            label = label_names[int(label_id)]
            score = float(probabilities_list[sequence_index][token_index][int(label_id)])
            sequence_predictions.append(
                {
                    "label": label,
                    "label_id": int(label_id),
                    "score": score,
                }
            )
            sequence_labels.append(label)
            sequence_scores.append(score)

        predictions.append(sequence_predictions)
        grouped_labels.append(sequence_labels)
        grouped_scores.append(sequence_scores)

    grouped_spans = None
    if texts is not None and offsets is not None:
        grouped_spans = decode_token_classification_batch(
            texts=texts,
            offsets=offsets,
            labels=grouped_labels,
            scores=grouped_scores,
            score_aggregation=score_aggregation,
            trim_span_whitespace=trim_span_whitespace,
            discard_overlapping=discard_overlapping,
        )

    return {
        "predictions": predictions,
        "grouped_spans": grouped_spans,
        "decoded_label_ids": decoded_label_ids,
        "label_names": label_names,
    }
