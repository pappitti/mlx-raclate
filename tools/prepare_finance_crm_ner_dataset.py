import argparse
import hashlib
import json
from bisect import bisect_left, bisect_right
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from datasets import ClassLabel, Dataset, Features, Sequence as DatasetSequence, Value
from huggingface_hub import snapshot_download

from mlx_raclate.utils.tokenizer_utils import load_tokenizer


ENTITY_TYPES = ("PERSON", "ORGANIZATION", "FUND", "EVENT")
BIO_PREFIXES = ("B", "I")
BIOES_PREFIXES = ("B", "I", "E", "S")
LABELS = ["O"] + [
    f"{prefix}-{entity_type}"
    for entity_type in ENTITY_TYPES
    for prefix in BIOES_PREFIXES
]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}

PHASE_FILES = {
    "generalization": "train_generalization.jsonl",
    "lexical_calibration": "train_lexical_calibration.jsonl",
}


def labels_for_scheme(label_scheme: str) -> List[str]:
    normalized_scheme = label_scheme.upper()
    if normalized_scheme == "BIO":
        prefixes = BIO_PREFIXES
    elif normalized_scheme == "BIOES":
        prefixes = BIOES_PREFIXES
    else:
        raise ValueError("label_scheme must be either 'BIO' or 'BIOES'")
    return ["O"] + [
        f"{prefix}-{entity_type}"
        for entity_type in ENTITY_TYPES
        for prefix in prefixes
    ]


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be strictly between 0 and 1")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _canonical_document_id(
    row: Mapping[str, Any], source: Path, line_number: int
) -> str:
    canonical_id = (
        row.get("metadata", {})
        .get("training_selection", {})
        .get("canonical_document_id")
    )
    if not isinstance(canonical_id, str) or not canonical_id:
        raise ValueError(
            f"{source}:{line_number}: missing "
            "metadata.training_selection.canonical_document_id"
        )
    return canonical_id


def _read_phase(path: Path, phase: str) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error

            if not row.get("training_ready", False):
                raise ValueError(
                    f"{path}:{line_number}: row is not marked training_ready"
                )
            if row.get("unresolved_entities"):
                raise ValueError(f"{path}:{line_number}: row has unresolved entities")

            row["_phase"] = phase
            row["_source_line"] = line_number
            row["_canonical_document_id"] = _canonical_document_id(
                row, path, line_number
            )
            rows.append(row)
    return rows


def _stable_split(canonical_id: str, validation_fraction: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}\0{canonical_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if bucket < validation_fraction else "train"


def _read_split_manifest(path: Path) -> Dict[str, str]:
    assignments = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            item = json.loads(line)
            canonical_id = item.get("canonical_document_id")
            split = item.get("split")
            if not isinstance(canonical_id, str) or split not in {
                "train",
                "validation",
            }:
                raise ValueError(f"{path}:{line_number}: invalid split manifest row")
            if canonical_id in assignments and assignments[canonical_id] != split:
                raise ValueError(
                    f"{path}:{line_number}: conflicting assignment for {canonical_id!r}"
                )
            assignments[canonical_id] = split
    return assignments


def _make_split_assignments(
    canonical_ids: Iterable[str],
    validation_fraction: float,
    seed: int,
    input_manifest: Optional[Path],
) -> Dict[str, str]:
    canonical_ids = sorted(set(canonical_ids))
    if input_manifest is not None:
        assignments = _read_split_manifest(input_manifest)
        missing = set(canonical_ids).difference(assignments)
        if missing:
            raise ValueError(
                f"Split manifest {input_manifest} is missing {len(missing)} canonical IDs"
            )
        return {
            canonical_id: assignments[canonical_id] for canonical_id in canonical_ids
        }

    assignments = {
        canonical_id: _stable_split(canonical_id, validation_fraction, seed)
        for canonical_id in canonical_ids
    }
    split_counts = Counter(assignments.values())
    if not split_counts["train"] or not split_counts["validation"]:
        raise ValueError(
            "The document-level split produced an empty train or validation set"
        )
    return assignments


def _validate_entities(row: Mapping[str, Any], source: Path) -> List[Dict[str, Any]]:
    line_number = row["_source_line"]
    text = row.get("text")
    entities = row.get("entities")
    if not isinstance(text, str) or not isinstance(entities, list):
        raise ValueError(
            f"{source}:{line_number}: expected string text and list entities"
        )

    validated = []
    for entity in entities:
        start = entity.get("start")
        end = entity.get("end")
        entity_text = entity.get("text")
        entity_type = str(entity.get("label", "")).upper()
        if entity_type not in ENTITY_TYPES:
            raise ValueError(
                f"{source}:{line_number}: unsupported entity label {entity.get('label')!r}"
            )
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"{source}:{line_number}: non-integer entity offsets")
        if not 0 <= start < end <= len(text):
            raise ValueError(
                f"{source}:{line_number}: invalid entity range [{start}, {end})"
            )
        if entity_text != text[start:end]:
            raise ValueError(
                f"{source}:{line_number}: entity text does not match [{start}, {end})"
            )
        validated.append({"start": start, "end": end, "label": entity_type})

    validated.sort(key=lambda entity: (entity["start"], entity["end"]))
    for previous, current in zip(validated, validated[1:]):
        if current["start"] < previous["end"]:
            raise ValueError(
                f"{source}:{line_number}: overlapping entity ranges "
                f"[{previous['start']}, {previous['end']}) and "
                f"[{current['start']}, {current['end']})"
            )
    return validated


def _apply_bioes(
    labels: List[int], token_indices: Sequence[int], entity_type: str
) -> None:
    if len(token_indices) == 1:
        labels[token_indices[0]] = LABEL2ID[f"S-{entity_type}"]
        return

    labels[token_indices[0]] = LABEL2ID[f"B-{entity_type}"]
    for token_index in token_indices[1:-1]:
        labels[token_index] = LABEL2ID[f"I-{entity_type}"]
    labels[token_indices[-1]] = LABEL2ID[f"E-{entity_type}"]


def _apply_entity_labels(
    labels: List[int],
    token_indices: Sequence[int],
    entity_type: str,
    *,
    label2id: Mapping[str, int],
    label_scheme: str,
) -> None:
    if label_scheme == "BIO":
        labels[token_indices[0]] = label2id[f"B-{entity_type}"]
        for token_index in token_indices[1:]:
            labels[token_index] = label2id[f"I-{entity_type}"]
        return

    if label_scheme != "BIOES":
        raise ValueError("label_scheme must be either 'BIO' or 'BIOES'")

    if len(token_indices) == 1:
        labels[token_indices[0]] = label2id[f"S-{entity_type}"]
        return

    labels[token_indices[0]] = label2id[f"B-{entity_type}"]
    for token_index in token_indices[1:-1]:
        labels[token_index] = label2id[f"I-{entity_type}"]
    labels[token_indices[-1]] = label2id[f"E-{entity_type}"]


def _align_entities(
    offsets: Sequence[Tuple[int, int]],
    entities: Sequence[Mapping[str, Any]],
    source: Path,
    line_number: int,
    *,
    label2id: Optional[Mapping[str, int]] = None,
    label_scheme: str = "BIOES",
) -> Tuple[List[int], List[Optional[int]], int]:
    normalized_scheme = label_scheme.upper()
    resolved_label2id = LABEL2ID if label2id is None else label2id
    labels = [resolved_label2id["O"]] * len(offsets)
    token_entities: List[Optional[int]] = [None] * len(offsets)
    token_starts = [start for start, _ in offsets]
    token_ends = [end for _, end in offsets]
    partial_boundaries = 0

    for entity_index, entity in enumerate(entities):
        first = bisect_right(token_ends, entity["start"])
        last = bisect_left(token_starts, entity["end"])
        token_indices = [
            index
            for index in range(first, last)
            if offsets[index][0] < entity["end"] and offsets[index][1] > entity["start"]
        ]
        if not token_indices:
            raise ValueError(
                f"{source}:{line_number}: no tokenizer tokens overlap entity "
                f"[{entity['start']}, {entity['end']})"
            )
        if offsets[token_indices[0]][0] < entity["start"]:
            partial_boundaries += 1
        if offsets[token_indices[-1]][1] > entity["end"]:
            partial_boundaries += 1

        conflicting = [
            index for index in token_indices if token_entities[index] is not None
        ]
        if conflicting:
            raise ValueError(
                f"{source}:{line_number}: tokenizer token {conflicting[0]} overlaps "
                "more than one entity"
            )

        for token_index in token_indices:
            token_entities[token_index] = entity_index
        _apply_entity_labels(
            labels,
            token_indices,
            entity["label"],
            label2id=resolved_label2id,
            label_scheme=normalized_scheme,
        )

    return labels, token_entities, partial_boundaries


def _chunk_ranges(
    token_entities: Sequence[Optional[int]], max_length: int
) -> List[Tuple[int, int]]:
    ranges = []
    token_count = len(token_entities)
    start = 0
    while start < token_count:
        end = min(start + max_length, token_count)
        if end < token_count:
            left_entity = token_entities[end - 1]
            right_entity = token_entities[end]
            if left_entity is not None and left_entity == right_entity:
                entity_start = end - 1
                while (
                    entity_start > start
                    and token_entities[entity_start - 1] == left_entity
                ):
                    entity_start -= 1
                if entity_start == start:
                    raise ValueError(
                        f"An entity requires more than the max_length of {max_length} tokens"
                    )
                end = entity_start
        ranges.append((start, end))
        start = end
    return ranges


def _convert_row(
    row: Mapping[str, Any],
    tokenizer,
    max_length: int,
    source: Path,
    *,
    label2id: Optional[Mapping[str, int]] = None,
    label_scheme: str = "BIOES",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    entities = _validate_entities(row, source)
    encoded = tokenizer(
        row["text"],
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", [1] * len(input_ids))
    offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
    if not input_ids:
        raise ValueError(
            f"{source}:{row['_source_line']}: tokenizer produced no tokens"
        )
    labels, token_entities, partial_boundaries = _align_entities(
        offsets,
        entities,
        source,
        row["_source_line"],
        label2id=label2id,
        label_scheme=label_scheme,
    )
    ranges = _chunk_ranges(token_entities, max_length)

    records = []
    provenance = []
    for chunk_index, (start, end) in enumerate(ranges):
        records.append(
            {
                "input_ids": input_ids[start:end],
                "attention_mask": attention_mask[start:end],
                "labels": labels[start:end],
            }
        )
        provenance.append(
            {
                "document_id": row["document_id"],
                "canonical_document_id": row["_canonical_document_id"],
                "dataset": row["dataset"],
                "language": row["language"],
                "phase": row["_phase"],
                "source_line": row["_source_line"],
                "chunk_index": chunk_index,
                "chunk_count": len(ranges),
                "token_start": start,
                "token_end": end,
                "char_start": offsets[start][0] if start < len(offsets) else 0,
                "char_end": offsets[end - 1][1] if end else 0,
                "source_token_count": len(input_ids),
                "entity_count": len(entities),
            }
        )

    return (
        records,
        provenance,
        {
            "documents": 1,
            "sequences": len(records),
            "tokens": len(input_ids),
            "entities": len(entities),
            "partial_entity_boundaries": partial_boundaries,
            "documents_chunked": int(len(ranges) > 1),
            "extra_chunks": len(ranges) - 1,
        },
    )


def _add_stats(total: Counter, addition: Mapping[str, int]) -> None:
    for key, value in addition.items():
        total[key] += value


def _training_features(labels: Optional[Sequence[str]] = None) -> Features:
    label_names = list(LABELS if labels is None else labels)
    return Features(
        {
            "input_ids": DatasetSequence(Value("int32")),
            "attention_mask": DatasetSequence(Value("int8")),
            "labels": DatasetSequence(ClassLabel(names=label_names)),
        }
    )


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _output_paths(output_dir: Path) -> List[Path]:
    paths = [output_dir / "labels.json", output_dir / "manifest.json"]
    paths.append(output_dir / "split_manifest.jsonl")
    for phase in PHASE_FILES:
        for split in ("train", "validation"):
            paths.extend(
                [
                    output_dir / phase / f"{split}.parquet",
                    output_dir / phase / f"{split}_provenance.jsonl",
                ]
            )
    return paths


def _model_context_length(model_path: Path) -> Optional[int]:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    value = config.get("max_position_embeddings")
    return value if isinstance(value, int) else None


def _get_model_path(path_or_hf_repo: str) -> Path:
    model_path = Path(path_or_hf_repo)
    if model_path.exists():
        return model_path
    return Path(
        snapshot_download(
            repo_id=path_or_hf_repo,
            allow_patterns=[
                "*.json",
                "*.model",
                "*.py",
                "*.tiktoken",
                "*.txt",
            ],
        )
    )


def _split_statistics(
    stats: Mapping[str, int],
    records: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
) -> Dict[str, Any]:
    label_token_counts = Counter()
    for record in records:
        label_token_counts.update(labels[label_id] for label_id in record["labels"])

    entity_counts = Counter()
    for label, count in label_token_counts.items():
        if label.startswith(("B-", "S-")):
            entity_counts[label.split("-", 1)[1]] += count

    token_count = sum(label_token_counts.values())
    return {
        **dict(sorted(stats.items())),
        "outside_token_fraction": (
            label_token_counts["O"] / token_count if token_count else 0.0
        ),
        "label_token_counts": {
            label: label_token_counts[label] for label in labels
        },
        "entity_counts": dict(sorted(entity_counts.items())),
    }


def prepare_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    model: str,
    max_length: int,
    validation_fraction: float,
    split_seed: int,
    split_manifest: Optional[Path],
    overwrite: bool,
    label_scheme: str = "BIOES",
) -> Dict[str, Any]:
    normalized_scheme = label_scheme.upper()
    labels = labels_for_scheme(normalized_scheme)
    label2id = {label: index for index, label in enumerate(labels)}

    source_paths = {
        phase: input_dir / filename for phase, filename in PHASE_FILES.items()
    }
    missing_sources = [str(path) for path in source_paths.values() if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(f"Missing input files: {', '.join(missing_sources)}")

    existing_outputs = [path for path in _output_paths(output_dir) if path.exists()]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            f"{existing_outputs[0]} already exists; pass --overwrite to replace outputs"
        )

    rows_by_phase = {
        phase: _read_phase(source_paths[phase], phase) for phase in PHASE_FILES
    }
    canonical_ids = {
        row["_canonical_document_id"] for rows in rows_by_phase.values() for row in rows
    }
    assignments = _make_split_assignments(
        canonical_ids,
        validation_fraction,
        split_seed,
        split_manifest,
    )

    model_path = _get_model_path(model)
    model_context_length = _model_context_length(model_path)
    if model_context_length is not None and max_length > model_context_length:
        raise ValueError(
            f"max_length={max_length} exceeds the model context window "
            f"of {model_context_length}"
        )
    tokenizer = load_tokenizer(model_path)._tokenizer
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required to align character offsets")

    features = _training_features(labels)
    phase_stats = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for phase, rows in rows_by_phase.items():
        phase_dir = output_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        split_records = {"train": [], "validation": []}
        split_provenance = {"train": [], "validation": []}
        split_stats = {"train": Counter(), "validation": Counter()}

        for row in rows:
            split = assignments[row["_canonical_document_id"]]
            records, provenance, stats = _convert_row(
                row,
                tokenizer,
                max_length,
                source_paths[phase],
                label2id=label2id,
                label_scheme=normalized_scheme,
            )
            split_records[split].extend(records)
            split_provenance[split].extend(provenance)
            _add_stats(split_stats[split], stats)

        for split in ("train", "validation"):
            if not split_records[split]:
                raise ValueError(f"{phase} produced an empty {split} split")
            dataset = Dataset.from_list(split_records[split], features=features)
            dataset.to_parquet(str(phase_dir / f"{split}.parquet"))
            _write_jsonl(
                phase_dir / f"{split}_provenance.jsonl",
                split_provenance[split],
            )

        phase_stats[phase] = {
            split: _split_statistics(split_stats[split], split_records[split], labels)
            for split in ("train", "validation")
        }

    split_membership = {canonical_id: set() for canonical_id in canonical_ids}
    for phase, rows in rows_by_phase.items():
        for row in rows:
            split_membership[row["_canonical_document_id"]].add(phase)
    split_manifest_rows = [
        {
            "canonical_document_id": canonical_id,
            "split": assignments[canonical_id],
            "phases": sorted(split_membership[canonical_id]),
        }
        for canonical_id in sorted(canonical_ids)
    ]
    _write_jsonl(output_dir / "split_manifest.jsonl", split_manifest_rows)
    _write_json(output_dir / "labels.json", labels)

    shared_ids = set(
        row["_canonical_document_id"] for row in rows_by_phase["generalization"]
    ).intersection(
        row["_canonical_document_id"] for row in rows_by_phase["lexical_calibration"]
    )
    manifest = {
        "schema_version": 1,
        "task_type": "token-classification",
        "label_scheme": normalized_scheme,
        "entity_types": list(ENTITY_TYPES),
        "labels": labels,
        "model": model,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "add_special_tokens": False,
        "max_length": max_length,
        "model_context_length": model_context_length,
        "split": {
            "group_field": "metadata.training_selection.canonical_document_id",
            "strategy": "provided_manifest" if split_manifest else "sha256",
            "seed": split_seed,
            "validation_fraction": validation_fraction,
            "canonical_documents": len(canonical_ids),
            "shared_phase_documents": len(shared_ids),
            "counts": dict(Counter(assignments.values())),
        },
        "columns": {
            "input_ids": "Tokenizer token IDs, chunked without cutting entities",
            "attention_mask": "One for every non-padding token",
            "labels": (
                f"{normalized_scheme} ClassLabel IDs aligned one-to-one with input_ids"
            ),
        },
        "phases": phase_stats,
    }
    _write_json(output_dir / "manifest.json", manifest)

    print(f"Saved Finance/CRM {normalized_scheme} datasets to {output_dir}")
    print(f"Tokenizer: {type(tokenizer).__name__}; labels: {len(labels)}")
    print(
        "Canonical split: "
        f"{manifest['split']['counts'].get('train', 0)} train / "
        f"{manifest['split']['counts'].get('validation', 0)} validation"
    )
    for phase in PHASE_FILES:
        train_stats = phase_stats[phase]["train"]
        validation_stats = phase_stats[phase]["validation"]
        print(
            f"{phase}: {train_stats['sequences']} train sequences / "
            f"{validation_stats['sequences']} validation sequences"
        )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create model-tokenized BIOES datasets for the two-phase Finance/CRM "
            "named-entity training run."
        )
    )
    parser.add_argument("--input-dir", default="data/NER-Finance-CRM")
    parser.add_argument("--model", default="perplexity-ai/pplx-embed-v1-0.6b")
    parser.add_argument(
        "--output-dir",
        default="data/NER-Finance-CRM/bioes_pplx_embed_v1_0.6b",
    )
    parser.add_argument("--max-length", type=_positive_int, default=32768)
    parser.add_argument("--validation-fraction", type=_fraction, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="Reuse canonical-document assignments from a prior split_manifest.jsonl.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite this script's output files if they already exist.",
    )
    args = parser.parse_args()
    prepare_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model=args.model,
        max_length=args.max_length,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        split_manifest=args.split_manifest,
        overwrite=args.overwrite,
        label_scheme="BIOES",
    )


if __name__ == "__main__":
    main()
