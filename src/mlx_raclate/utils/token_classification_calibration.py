"""Cached-logit calibration utilities for BIO Viterbi decoding."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .token_classification import (
    VITERBI_CALIBRATION_FILE,
    VITERBI_BIAS_KEYS,
    zero_viterbi_transition_biases,
)


LOGITS_CACHE_SCHEMA_VERSION = 1
IGNORE_INDEX = -100
BIO_TUNABLE_BIAS_KEYS = (
    "transition_bias_background_to_start",
    "transition_bias_inside_to_background",
    "transition_bias_inside_to_start",
)


def _as_numpy(value: Any, dtype: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=dtype)
    except TypeError:
        if not hasattr(value, "tolist"):
            raise
        return np.asarray(value.tolist(), dtype=dtype)


@dataclass(frozen=True)
class TokenClassificationLogitsCache:
    """Flattened variable-length token-classification emissions and references."""

    logits: np.ndarray
    references: np.ndarray
    special_tokens_mask: np.ndarray
    sequence_offsets: np.ndarray
    label_names: tuple[str, ...]
    metadata: Mapping[str, Any]

    @property
    def sequence_count(self) -> int:
        return len(self.sequence_offsets) - 1

    @property
    def token_count(self) -> int:
        return len(self.references)

    def sequence(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = int(self.sequence_offsets[index])
        end = int(self.sequence_offsets[index + 1])
        return (
            self.logits[start:end],
            self.references[start:end],
            self.special_tokens_mask[start:end],
        )


class LogitsCacheBuilder:
    """Accumulate model batches before writing a compact ``.npz`` cache."""

    def __init__(
        self,
        label_names: Sequence[str],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        logits_dtype: str = "float32",
    ) -> None:
        if logits_dtype not in {"float16", "float32"}:
            raise ValueError("logits_dtype must be 'float16' or 'float32'")
        self.label_names = tuple(str(label) for label in label_names)
        self.metadata = dict(metadata or {})
        self.logits_dtype = np.dtype(logits_dtype)
        self._logits: list[np.ndarray] = []
        self._references: list[np.ndarray] = []
        self._special_tokens_mask: list[np.ndarray] = []

    def append(self, logits, references, attention_mask) -> None:
        batch_logits = _as_numpy(logits, self.logits_dtype)
        batch_references = _as_numpy(references, np.int32)
        batch_attention = _as_numpy(attention_mask, np.bool_)

        if batch_logits.ndim != 3:
            raise ValueError("Cached logits must have shape [batch, sequence, labels]")
        if batch_references.shape != batch_logits.shape[:2]:
            raise ValueError("Cached references do not match the logits batch shape")
        if batch_attention.shape != batch_logits.shape[:2]:
            raise ValueError("Cached attention mask does not match the logits batch shape")
        if batch_logits.shape[2] != len(self.label_names):
            raise ValueError("Cached logits label dimension does not match label_names")

        for index in range(batch_logits.shape[0]):
            self._logits.append(np.array(batch_logits[index], copy=True))
            self._references.append(np.array(batch_references[index], copy=True))
            self._special_tokens_mask.append(
                np.logical_not(np.array(batch_attention[index], copy=True))
            )

    def build(self) -> TokenClassificationLogitsCache:
        if not self._logits:
            raise ValueError("Cannot build an empty logits cache")

        lengths = np.asarray([len(sequence) for sequence in self._references], dtype=np.int64)
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        metadata = {
            **self.metadata,
            "schema_version": LOGITS_CACHE_SCHEMA_VERSION,
            "logits_dtype": self.logits_dtype.name,
            "sequence_count": len(lengths),
            "token_count": int(offsets[-1]),
        }
        return TokenClassificationLogitsCache(
            logits=np.concatenate(self._logits, axis=0),
            references=np.concatenate(self._references, axis=0),
            special_tokens_mask=np.concatenate(self._special_tokens_mask, axis=0),
            sequence_offsets=offsets,
            label_names=self.label_names,
            metadata=metadata,
        )

    def save(self, path: str | Path) -> TokenClassificationLogitsCache:
        cache = self.build()
        save_logits_cache(path, cache)
        return cache


def save_logits_cache(
    path: str | Path,
    cache: TokenClassificationLogitsCache,
) -> None:
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("A logits cache path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        logits=cache.logits,
        references=cache.references,
        special_tokens_mask=cache.special_tokens_mask,
        sequence_offsets=cache.sequence_offsets,
        label_names=np.asarray(cache.label_names),
        metadata_json=np.asarray(json.dumps(dict(cache.metadata), sort_keys=True)),
    )


def load_logits_cache(path: str | Path) -> TokenClassificationLogitsCache:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "logits",
            "references",
            "special_tokens_mask",
            "sequence_offsets",
            "label_names",
            "metadata_json",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(
                f"Logits cache {path} is missing arrays: {', '.join(sorted(missing))}"
            )
        logits = np.array(payload["logits"], copy=True)
        references = np.asarray(payload["references"], dtype=np.int32)
        special_tokens_mask = np.asarray(payload["special_tokens_mask"], dtype=np.bool_)
        sequence_offsets = np.asarray(payload["sequence_offsets"], dtype=np.int64)
        label_names = tuple(str(label) for label in payload["label_names"].tolist())
        metadata = json.loads(str(payload["metadata_json"].item()))

    if metadata.get("schema_version") != LOGITS_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported logits cache schema: {metadata.get('schema_version')!r}"
        )
    if logits.ndim != 2 or logits.shape[1] != len(label_names):
        raise ValueError(f"Logits cache {path} has an invalid logits shape")
    if references.shape != (len(logits),):
        raise ValueError(f"Logits cache {path} has invalid references")
    if special_tokens_mask.shape != references.shape:
        raise ValueError(f"Logits cache {path} has an invalid special-token mask")
    if (
        sequence_offsets.ndim != 1
        or len(sequence_offsets) < 2
        or sequence_offsets[0] != 0
        or sequence_offsets[-1] != len(logits)
        or np.any(sequence_offsets[1:] < sequence_offsets[:-1])
    ):
        raise ValueError(f"Logits cache {path} has invalid sequence offsets")

    return TokenClassificationLogitsCache(
        logits=logits,
        references=references,
        special_tokens_mask=special_tokens_mask,
        sequence_offsets=sequence_offsets,
        label_names=label_names,
        metadata=metadata,
    )


@dataclass(frozen=True)
class CalibrationSequence:
    logits: np.ndarray
    references: np.ndarray
    special_tokens_mask: np.ndarray
    reference_spans: frozenset[tuple[str, int, int]]
    source_index: int


@dataclass(frozen=True)
class BIOCalibrationCorpus:
    label_names: tuple[str, ...]
    prefixes: tuple[str, ...]
    entity_groups_by_label: tuple[str, ...]
    entity_groups: tuple[str, ...]
    outside_index: int
    sequences: tuple[CalibrationSequence, ...]
    cache_metadata: tuple[Mapping[str, Any], ...]

    def subset(self, indices: Iterable[int]) -> "BIOCalibrationCorpus":
        return BIOCalibrationCorpus(
            label_names=self.label_names,
            prefixes=self.prefixes,
            entity_groups_by_label=self.entity_groups_by_label,
            entity_groups=self.entity_groups,
            outside_index=self.outside_index,
            sequences=tuple(self.sequences[index] for index in indices),
            cache_metadata=self.cache_metadata,
        )


def _bio_label_metadata(
    label_names: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], int]:
    if label_names.count("O") != 1:
        raise ValueError("BIO calibration requires exactly one O label")

    prefixes = []
    groups_by_label = []
    entity_groups = set()
    for label in label_names:
        if label == "O":
            prefixes.append("O")
            groups_by_label.append("O")
            continue
        if "-" not in label:
            raise ValueError(f"BIO calibration cannot parse label {label!r}")
        prefix, entity_group = label.split("-", 1)
        if prefix not in {"B", "I"} or not entity_group:
            raise ValueError(
                "Cached labels are not a BIO ontology; only O, B-*, and I-* are supported"
            )
        prefixes.append(prefix)
        groups_by_label.append(entity_group)
        entity_groups.add(entity_group)

    return (
        tuple(prefixes),
        tuple(groups_by_label),
        tuple(sorted(entity_groups)),
        label_names.index("O"),
    )


def _bio_spans_from_ids(
    label_ids: Sequence[int],
    prefixes: Sequence[str],
    entity_groups_by_label: Sequence[str],
) -> frozenset[tuple[str, int, int]]:
    spans: set[tuple[str, int, int]] = set()
    current_group: Optional[str] = None
    current_start = 0

    def flush(end: int) -> None:
        nonlocal current_group
        if current_group is not None:
            spans.add((current_group, current_start, end))
            current_group = None

    for token_index, raw_label_id in enumerate(label_ids):
        label_id = int(raw_label_id)
        prefix = prefixes[label_id]
        entity_group = entity_groups_by_label[label_id]
        if prefix == "O":
            flush(token_index)
        elif prefix == "B":
            flush(token_index)
            current_group = entity_group
            current_start = token_index
        elif current_group != entity_group:
            flush(token_index)
            current_group = entity_group
            current_start = token_index

    flush(len(label_ids))
    return frozenset(spans)


def load_bio_calibration_corpus(
    cache_paths: Sequence[str | Path],
) -> BIOCalibrationCorpus:
    if not cache_paths:
        raise ValueError("At least one logits cache is required")

    caches = [load_logits_cache(path) for path in cache_paths]
    label_names = caches[0].label_names
    if any(cache.label_names != label_names for cache in caches[1:]):
        raise ValueError("All logits caches must use the same ordered label ontology")
    model_ids = {
        str(cache.metadata["model"])
        for cache in caches
        if cache.metadata.get("model") is not None
    }
    if len(model_ids) > 1:
        raise ValueError("All logits caches must come from the same model checkpoint")

    prefixes, groups_by_label, entity_groups, outside_index = _bio_label_metadata(
        label_names
    )
    sequences = []
    for source_index, cache in enumerate(caches):
        for sequence_index in range(cache.sequence_count):
            logits, references, special_tokens_mask = cache.sequence(sequence_index)
            valid_reference = references != IGNORE_INDEX
            reference_ids = references[valid_reference]
            if np.any(reference_ids < 0) or np.any(reference_ids >= len(label_names)):
                raise ValueError("A logits cache contains an invalid reference label ID")
            sequences.append(
                CalibrationSequence(
                    logits=logits,
                    references=references,
                    special_tokens_mask=special_tokens_mask,
                    reference_spans=_bio_spans_from_ids(
                        reference_ids,
                        prefixes,
                        groups_by_label,
                    ),
                    source_index=source_index,
                )
            )

    return BIOCalibrationCorpus(
        label_names=label_names,
        prefixes=prefixes,
        entity_groups_by_label=groups_by_label,
        entity_groups=entity_groups,
        outside_index=outside_index,
        sequences=tuple(sequences),
        cache_metadata=tuple(cache.metadata for cache in caches),
    )


def _resolved_biases(biases: Mapping[str, float]) -> dict[str, float]:
    resolved = zero_viterbi_transition_biases()
    for key, value in biases.items():
        if key not in VITERBI_BIAS_KEYS:
            raise ValueError(f"Unknown Viterbi transition bias: {key}")
        resolved[key] = float(value)
    return resolved


def _bio_transition_matrix(
    corpus: BIOCalibrationCorpus,
    biases: Mapping[str, float],
) -> np.ndarray:
    resolved = _resolved_biases(biases)
    label_count = len(corpus.label_names)
    transitions = np.full((label_count, label_count), -np.inf, dtype=np.float64)

    for previous_index in range(label_count):
        previous_prefix = corpus.prefixes[previous_index]
        previous_group = corpus.entity_groups_by_label[previous_index]
        for current_index in range(label_count):
            current_prefix = corpus.prefixes[current_index]
            current_group = corpus.entity_groups_by_label[current_index]

            if previous_prefix == "O":
                if current_prefix == "O":
                    transitions[previous_index, current_index] = resolved[
                        "transition_bias_background_stay"
                    ]
                elif current_prefix == "B":
                    transitions[previous_index, current_index] = resolved[
                        "transition_bias_background_to_start"
                    ]
                continue

            if current_prefix == "O":
                transitions[previous_index, current_index] = resolved[
                    "transition_bias_inside_to_background"
                ]
            elif current_prefix == "B":
                transitions[previous_index, current_index] = resolved[
                    "transition_bias_inside_to_start"
                ]
            elif current_group == previous_group:
                transitions[previous_index, current_index] = resolved[
                    "transition_bias_inside_to_continue"
                ]

    return transitions


def decode_cached_bio_sequence(
    corpus: BIOCalibrationCorpus,
    sequence: CalibrationSequence,
    biases: Mapping[str, float],
    *,
    transition_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vectorized-label Viterbi decode for one cached BIO sequence."""
    emissions = np.asarray(sequence.logits, dtype=np.float64)
    if len(emissions) == 0:
        return np.empty(0, dtype=np.int16)

    label_count = len(corpus.label_names)
    transitions = (
        _bio_transition_matrix(corpus, biases)
        if transition_matrix is None
        else transition_matrix
    )
    scores = np.array(emissions[0], copy=True)
    for label_index, prefix in enumerate(corpus.prefixes):
        if prefix == "I":
            scores[label_index] = -np.inf
    if sequence.special_tokens_mask[0]:
        scores[:] = -np.inf
        scores[corpus.outside_index] = emissions[0, corpus.outside_index]

    backpointers = np.empty((len(emissions), label_count), dtype=np.int16)
    backpointers[0] = -1
    label_indices = np.arange(label_count)
    for token_index in range(1, len(emissions)):
        candidates = scores[:, None] + transitions
        best_previous = np.argmax(candidates, axis=0)
        scores = emissions[token_index] + candidates[best_previous, label_indices]
        if sequence.special_tokens_mask[token_index]:
            scores[:] = -np.inf
            scores[corpus.outside_index] = (
                emissions[token_index, corpus.outside_index]
                + candidates[best_previous[corpus.outside_index], corpus.outside_index]
            )
        backpointers[token_index] = best_previous

    final_index = int(np.argmax(scores))
    decoded = np.empty(len(emissions), dtype=np.int16)
    decoded[-1] = final_index
    for token_index in range(len(emissions) - 1, 0, -1):
        final_index = int(backpointers[token_index, final_index])
        decoded[token_index - 1] = final_index
    return decoded


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def evaluate_bio_transition_biases(
    corpus: BIOCalibrationCorpus,
    biases: Mapping[str, float],
) -> dict[str, Any]:
    counts = {entity_group: [0, 0, 0] for entity_group in corpus.entity_groups}
    predicted_span_count = 0
    reference_span_count = 0
    transition_matrix = _bio_transition_matrix(corpus, biases)

    for sequence in corpus.sequences:
        decoded = decode_cached_bio_sequence(
            corpus,
            sequence,
            biases,
            transition_matrix=transition_matrix,
        )
        valid_reference = sequence.references != IGNORE_INDEX
        predicted_spans = _bio_spans_from_ids(
            decoded[valid_reference],
            corpus.prefixes,
            corpus.entity_groups_by_label,
        )
        reference_spans = sequence.reference_spans
        predicted_span_count += len(predicted_spans)
        reference_span_count += len(reference_spans)

        for entity_group in corpus.entity_groups:
            predicted_for_group = {
                span for span in predicted_spans if span[0] == entity_group
            }
            reference_for_group = {
                span for span in reference_spans if span[0] == entity_group
            }
            counts[entity_group][0] += len(
                predicted_for_group.intersection(reference_for_group)
            )
            counts[entity_group][1] += len(
                predicted_for_group.difference(reference_for_group)
            )
            counts[entity_group][2] += len(
                reference_for_group.difference(predicted_for_group)
            )

    precision_by_label = {}
    recall_by_label = {}
    f1_by_label = {}
    counts_by_label = {}
    total_tp = total_fp = total_fn = 0
    for entity_group, (tp, fp, fn) in counts.items():
        precision, recall, f1 = _prf(tp, fp, fn)
        precision_by_label[entity_group] = precision
        recall_by_label[entity_group] = recall
        f1_by_label[entity_group] = f1
        counts_by_label[entity_group] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "support": tp + fn,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_precision, micro_recall, micro_f1 = _prf(total_tp, total_fp, total_fn)
    label_count = len(corpus.entity_groups)
    return {
        "boundary_precision_macro": (
            sum(precision_by_label.values()) / label_count if label_count else 0.0
        ),
        "boundary_recall_macro": (
            sum(recall_by_label.values()) / label_count if label_count else 0.0
        ),
        "boundary_f1_macro": (
            sum(f1_by_label.values()) / label_count if label_count else 0.0
        ),
        "boundary_precision_micro": micro_precision,
        "boundary_recall_micro": micro_recall,
        "boundary_f1_micro": micro_f1,
        "boundary_precision_by_label": precision_by_label,
        "boundary_recall_by_label": recall_by_label,
        "boundary_f1_by_label": f1_by_label,
        "boundary_counts_by_label": counts_by_label,
        "predicted_spans": predicted_span_count,
        "reference_spans": reference_span_count,
    }


def _candidate_to_biases(candidate: Sequence[float]) -> dict[str, float]:
    biases = zero_viterbi_transition_biases()
    for key, value in zip(BIO_TUNABLE_BIAS_KEYS, candidate):
        biases[key] = float(value)
    return biases


def _candidate_norm(candidate: Sequence[float]) -> float:
    return sum(abs(float(value)) for value in candidate)


def _metric_keys(objective: str) -> tuple[str, str, str]:
    if objective not in {"micro", "macro"}:
        raise ValueError("objective must be 'micro' or 'macro'")
    return (
        f"boundary_precision_{objective}",
        f"boundary_recall_{objective}",
        f"boundary_f1_{objective}",
    )


def _sample_corpus(
    corpus: BIOCalibrationCorpus,
    maximum_sequences: Optional[int],
    seed: int,
) -> BIOCalibrationCorpus:
    if maximum_sequences is None or maximum_sequences >= len(corpus.sequences):
        return corpus
    if maximum_sequences <= 0:
        raise ValueError("search_max_sequences must be greater than zero")
    generator = random.Random(seed)
    indices = sorted(generator.sample(range(len(corpus.sequences)), maximum_sequences))
    return corpus.subset(indices)


def search_bio_transition_biases(
    corpus: BIOCalibrationCorpus,
    *,
    grid: Sequence[float] = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
    passes: int = 2,
    refine_step: float = 0.25,
    search_max_sequences: Optional[int] = 128,
    finalists: int = 4,
    retention: float = 0.95,
    objective: str = "micro",
    seed: int = 42,
) -> dict[str, Any]:
    """Coordinate-search BIO biases and return three practical operating points."""
    if not grid:
        raise ValueError("grid must contain at least one bias value")
    if passes <= 0:
        raise ValueError("passes must be greater than zero")
    if finalists <= 0:
        raise ValueError("finalists must be greater than zero")
    if not 0.0 < retention <= 1.0:
        raise ValueError("retention must be in (0, 1]")
    if refine_step < 0.0:
        raise ValueError("refine_step cannot be negative")

    precision_key, recall_key, f1_key = _metric_keys(objective)
    search_corpus = _sample_corpus(corpus, search_max_sequences, seed)
    search_scores: dict[tuple[float, ...], dict[str, Any]] = {}

    def score_search(candidate: tuple[float, ...]) -> dict[str, Any]:
        normalized = tuple(float(value) for value in candidate)
        if normalized not in search_scores:
            search_scores[normalized] = evaluate_bio_transition_biases(
                search_corpus,
                _candidate_to_biases(normalized),
            )
        return search_scores[normalized]

    def balanced_key(item: tuple[tuple[float, ...], Mapping[str, Any]]):
        candidate, metrics = item
        return (
            metrics[f1_key],
            metrics[precision_key] + metrics[recall_key],
            -_candidate_norm(candidate),
        )

    current = tuple(0.0 for _ in BIO_TUNABLE_BIAS_KEYS)
    score_search(current)
    absolute_grid = tuple(sorted(set(float(value) for value in grid)))
    for _ in range(passes):
        for dimension in range(len(BIO_TUNABLE_BIAS_KEYS)):
            options = []
            for value in absolute_grid:
                candidate = list(current)
                candidate[dimension] = value
                candidate_tuple = tuple(candidate)
                options.append((candidate_tuple, score_search(candidate_tuple)))
            current = max(options, key=balanced_key)[0]

    if refine_step:
        for dimension in range(len(BIO_TUNABLE_BIAS_KEYS)):
            options = []
            for offset in (-2, -1, 0, 1, 2):
                candidate = list(current)
                candidate[dimension] += offset * refine_step
                candidate_tuple = tuple(candidate)
                options.append((candidate_tuple, score_search(candidate_tuple)))
            current = max(options, key=balanced_key)[0]

    search_items = list(search_scores.items())
    search_balanced = max(search_items, key=balanced_key)
    search_balanced_metrics = search_balanced[1]
    search_recall_floor = search_balanced_metrics[recall_key] * retention
    search_precision_floor = search_balanced_metrics[precision_key] * retention

    rankings = [
        sorted(search_items, key=balanced_key, reverse=True),
        sorted(
            search_items,
            key=lambda item: (
                item[1][precision_key],
                item[1][f1_key],
                -_candidate_norm(item[0]),
            ),
            reverse=True,
        ),
        sorted(
            search_items,
            key=lambda item: (
                item[1][recall_key],
                item[1][f1_key],
                -_candidate_norm(item[0]),
            ),
            reverse=True,
        ),
        sorted(
            [item for item in search_items if item[1][recall_key] >= search_recall_floor],
            key=lambda item: (
                item[1][precision_key],
                item[1][f1_key],
                -_candidate_norm(item[0]),
            ),
            reverse=True,
        ),
        sorted(
            [item for item in search_items if item[1][precision_key] >= search_precision_floor],
            key=lambda item: (
                item[1][recall_key],
                item[1][f1_key],
                -_candidate_norm(item[0]),
            ),
            reverse=True,
        ),
    ]

    finalist_candidates = {tuple(0.0 for _ in BIO_TUNABLE_BIAS_KEYS), current}
    for ranking in rankings:
        finalist_candidates.update(candidate for candidate, _ in ranking[:finalists])

    full_scores = {
        candidate: evaluate_bio_transition_biases(
            corpus,
            _candidate_to_biases(candidate),
        )
        for candidate in sorted(finalist_candidates)
    }
    full_items = list(full_scores.items())
    balanced_candidate, balanced_metrics = max(full_items, key=balanced_key)
    recall_floor = balanced_metrics[recall_key] * retention
    precision_floor = balanced_metrics[precision_key] * retention

    high_precision_items = [
        item for item in full_items if item[1][recall_key] >= recall_floor
    ]
    high_recall_items = [
        item for item in full_items if item[1][precision_key] >= precision_floor
    ]
    high_precision_candidate, high_precision_metrics = max(
        high_precision_items,
        key=lambda item: (
            item[1][precision_key],
            item[1][f1_key],
            -_candidate_norm(item[0]),
        ),
    )
    high_recall_candidate, high_recall_metrics = max(
        high_recall_items,
        key=lambda item: (
            item[1][recall_key],
            item[1][f1_key],
            -_candidate_norm(item[0]),
        ),
    )

    def operating_point(candidate, metrics):
        return {
            "biases": _candidate_to_biases(candidate),
            "calibration_metrics": metrics,
        }

    return {
        "operating_points": {
            "balanced": operating_point(balanced_candidate, balanced_metrics),
            "high_precision": operating_point(
                high_precision_candidate,
                high_precision_metrics,
            ),
            "high_recall": operating_point(high_recall_candidate, high_recall_metrics),
        },
        "search": {
            "method": "coordinate_grid_with_full_corpus_finalists",
            "objective": f"boundary_f1_{objective}",
            "precision_recall_retention": retention,
            "bias_keys": list(BIO_TUNABLE_BIAS_KEYS),
            "grid": list(absolute_grid),
            "passes": passes,
            "refine_step": refine_step,
            "seed": seed,
            "search_sequences": len(search_corpus.sequences),
            "full_sequences": len(corpus.sequences),
            "search_candidates": len(search_scores),
            "full_corpus_finalists": len(full_scores),
            "zero_bias_metrics": full_scores[
                tuple(0.0 for _ in BIO_TUNABLE_BIAS_KEYS)
            ],
        },
    }


def write_calibration_operating_points(
    path_or_directory: str | Path,
    search_result: Mapping[str, Any],
    *,
    overwrite_operating_points: bool = False,
    set_default: bool = False,
) -> Path:
    path = Path(path_or_directory)
    if path.is_dir() or not path.suffix:
        path = path / VITERBI_CALIBRATION_FILE

    if path.exists():
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = {"operating_points": {}}

    operating_points = payload.setdefault("operating_points", {})
    if not isinstance(operating_points, dict):
        raise ValueError("Existing calibration has an invalid operating_points value")
    operating_points.setdefault(
        "default",
        {"biases": zero_viterbi_transition_biases()},
    )

    new_points = search_result.get("operating_points")
    if not isinstance(new_points, Mapping) or not new_points:
        raise ValueError("Search result contains no operating points")
    conflicts = set(new_points).intersection(operating_points)
    if conflicts and not overwrite_operating_points:
        raise FileExistsError(
            "Calibration operating points already exist: "
            + ", ".join(sorted(conflicts))
        )

    operating_points.update({name: dict(value) for name, value in new_points.items()})
    if set_default:
        operating_points["default"] = dict(new_points["balanced"])
    payload["calibration_search"] = dict(search_result.get("search", {}))

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary_path.replace(path)
    return path
