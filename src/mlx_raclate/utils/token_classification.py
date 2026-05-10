from typing import Any, Dict, List, Optional, Sequence, Tuple


_INVALID_PATH_SCORE = -1e12


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


def viterbi_decode_bioes_ids(
    emissions: Sequence[Sequence[float]],
    label_names: Sequence[str],
    special_tokens_mask: Optional[Sequence[bool]] = None,
    outside_label: str = "O",
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

                candidate_score = previous_score + emission_score
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
            )
        )
    return decoded


def decode_bioes_spans(
    text: str,
    offsets: Sequence[Sequence[int] | Tuple[int, int]],
    labels: Sequence[str],
    scores: Optional[Sequence[float]] = None,
    score_aggregation: str = "mean",
) -> List[Dict[str, Any]]:
    """
    Decode BIOES token labels into grouped entity spans.

    Returns spans in a format close to the Hugging Face token-classification
    pipeline and OpenMed's privacy-filter output.
    """
    spans: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    def flush_current():
        nonlocal current
        if current is None:
            return
        current["score"] = _aggregate_scores(current.pop("_scores"), score_aggregation)
        current["word"] = text[current["start"] : current["end"]]
        spans.append(current)
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
            spans.append(
                {
                    "entity_group": entity_group,
                    "score": 0.0 if score is None else score,
                    "word": text[start:end],
                    "start": start,
                    "end": end,
                }
            )
            continue

        if prefix == "S":
            flush_current()
            spans.append(
                {
                    "entity_group": entity_group,
                    "score": 0.0 if score is None else score,
                    "word": text[start:end],
                    "start": start,
                    "end": end,
                }
            )
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
    return spans


def decode_token_classification_batch(
    texts: Sequence[str],
    offsets: Sequence[Sequence[Sequence[int] | Tuple[int, int]]],
    labels: Sequence[Sequence[str]],
    scores: Optional[Sequence[Sequence[float]]] = None,
    score_aggregation: str = "mean",
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
            )
        )
    return decoded