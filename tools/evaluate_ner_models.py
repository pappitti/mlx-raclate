"""Evaluate one or more MLX Raclate NER checkpoints.

The evaluator reports token metrics and exact-boundary entity metrics. It can
discover a conventional ``<split>.<extension>`` file in a dataset directory,
or all such files one level below a multi-phase dataset root.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from datasets import ClassLabel, Dataset, load_dataset

from mlx_raclate.utils.token_classification import (
    compute_token_classification_metrics,
    ordered_label_names,
    viterbi_decode_bioes_batch,
    viterbi_transition_biases_from_calibration,
)


SUPPORTED_DATA_EXTENSIONS = ("parquet", "jsonl", "json", "csv")
IGNORE_INDEX = -100


@dataclass(frozen=True)
class EvaluationSource:
    name: str
    location: str
    local_file: Optional[Path] = None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _split_file(directory: Path, split: str) -> Optional[Path]:
    matches = [
        directory / f"{split}.{extension}"
        for extension in SUPPORTED_DATA_EXTENSIONS
        if (directory / f"{split}.{extension}").is_file()
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple files found for split {split!r} in {directory}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0] if matches else None


def discover_evaluation_sources(
    dataset_specs: Sequence[str], split: str
) -> list[EvaluationSource]:
    """Resolve files, split directories, multi-phase roots, and Hub datasets."""
    sources: list[EvaluationSource] = []
    for spec in dataset_specs:
        path = Path(spec)
        if path.is_file():
            sources.append(
                EvaluationSource(
                    name=path.parent.name or path.stem,
                    location=str(path),
                    local_file=path,
                )
            )
            continue

        if path.is_dir():
            direct_split = _split_file(path, split)
            if direct_split is not None:
                sources.append(
                    EvaluationSource(
                        name=path.name,
                        location=str(direct_split),
                        local_file=direct_split,
                    )
                )
                continue

            phase_sources = []
            for child in sorted(item for item in path.iterdir() if item.is_dir()):
                phase_split = _split_file(child, split)
                if phase_split is not None:
                    phase_sources.append(
                        EvaluationSource(
                            name=child.name,
                            location=str(phase_split),
                            local_file=phase_split,
                        )
                    )
            if not phase_sources:
                raise FileNotFoundError(
                    f"No {split}.{{{','.join(SUPPORTED_DATA_EXTENSIONS)}}} found "
                    f"in {path} or its immediate subdirectories"
                )
            sources.extend(phase_sources)
            continue

        # Non-local identifiers are delegated to Hugging Face Datasets.
        sources.append(EvaluationSource(name=spec, location=spec))

    duplicate_names = {
        source.name for source in sources if sum(s.name == source.name for s in sources) > 1
    }
    if duplicate_names:
        raise ValueError(
            "Dataset result names must be unique; duplicate names: "
            + ", ".join(sorted(duplicate_names))
        )
    return sources


def _load_local_dataset(path: Path) -> Dataset:
    extension = path.suffix.lower().lstrip(".")
    if extension == "parquet":
        return Dataset.from_parquet(str(path))
    if extension in {"json", "jsonl"}:
        return Dataset.from_json(str(path))
    if extension == "csv":
        return Dataset.from_csv(str(path))
    raise ValueError(f"Unsupported dataset extension: {path.suffix}")


def load_evaluation_dataset(source: EvaluationSource, split: str) -> Dataset:
    if source.local_file is not None:
        dataset = _load_local_dataset(source.local_file)
    else:
        dataset = load_dataset(source.location, split=split)

    renames = {}
    if "tokens" in dataset.column_names and "text" not in dataset.column_names:
        renames["tokens"] = "text"
    if "ner_tags" in dataset.column_names and "labels" not in dataset.column_names:
        renames["ner_tags"] = "labels"
    if renames:
        dataset = dataset.rename_columns(renames)

    if "labels" not in dataset.column_names:
        raise ValueError(
            f"Dataset {source.name!r} has no labels or ner_tags column: "
            f"{dataset.column_names}"
        )
    if "input_ids" not in dataset.column_names and "text" not in dataset.column_names:
        raise ValueError(
            f"Dataset {source.name!r} needs input_ids, text, or tokens: "
            f"{dataset.column_names}"
        )
    return dataset


def _read_label_file(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list) and all(isinstance(label, str) for label in payload):
        return payload
    if isinstance(payload, Mapping):
        try:
            return [str(payload[str(index)]) for index in range(len(payload))]
        except KeyError as error:
            raise ValueError(
                f"Label mapping in {path} must have contiguous stringified integer keys"
            ) from error
    raise ValueError(f"Expected a JSON label list or id-to-label mapping in {path}")


def _labels_from_feature(dataset: Dataset) -> Optional[list[str]]:
    feature = dataset.features.get("labels")
    item_feature = getattr(feature, "feature", None)
    if isinstance(item_feature, ClassLabel):
        return list(item_feature.names)
    return None


def _nearby_label_file(source: EvaluationSource) -> Optional[Path]:
    if source.local_file is None:
        return None
    for parent in (source.local_file.parent, *source.local_file.parents):
        candidate = parent / "labels.json"
        if candidate.is_file():
            return candidate
    return None


def resolve_dataset_labels(
    dataset: Dataset,
    source: EvaluationSource,
    explicit_labels: Optional[Path],
) -> tuple[Optional[list[str]], str]:
    if explicit_labels is not None:
        return _read_label_file(explicit_labels), str(explicit_labels)

    feature_labels = _labels_from_feature(dataset)
    if feature_labels is not None:
        return feature_labels, "dataset ClassLabel metadata"

    nearby_labels = _nearby_label_file(source)
    if nearby_labels is not None:
        return _read_label_file(nearby_labels), str(nearby_labels)

    return None, "model id2label (assumed matching IDs)"


def _model_label_mapping(model) -> tuple[list[str], dict[str, int]]:
    id2label = getattr(model.config, "id2label", None)
    if not isinstance(id2label, Mapping) or not id2label:
        raise ValueError("The model config must contain a non-empty id2label mapping")
    label_names = ordered_label_names(id2label, len(id2label))
    if len(set(label_names)) != len(label_names):
        raise ValueError("The model id2label mapping contains duplicate label names")
    return label_names, {label: index for index, label in enumerate(label_names)}


def remap_reference_sequences(
    sequences: Sequence[Sequence[Any]],
    dataset_labels: Optional[Sequence[str]],
    model_label2id: Mapping[str, int],
) -> list[list[int]]:
    """Map dataset label IDs/names to model IDs before loss and metrics."""
    remapped = []
    model_label_ids = set(model_label2id.values())
    for sequence_index, sequence in enumerate(sequences):
        remapped_sequence = []
        for token_index, raw_label in enumerate(sequence):
            if raw_label is None or raw_label == IGNORE_INDEX or str(raw_label) == str(IGNORE_INDEX):
                remapped_sequence.append(IGNORE_INDEX)
                continue

            if isinstance(raw_label, str) and raw_label in model_label2id:
                label_name = raw_label
            else:
                try:
                    label_index = int(raw_label)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid label {raw_label!r} at sequence {sequence_index}, "
                        f"token {token_index}"
                    ) from error
                if dataset_labels is None:
                    if label_index not in model_label_ids:
                        raise ValueError(
                            f"Dataset label ID {label_index} is missing from the model "
                            "id2label mapping"
                        )
                    remapped_sequence.append(label_index)
                    continue
                if not 0 <= label_index < len(dataset_labels):
                    raise ValueError(
                        f"Dataset label ID {label_index} is outside the label mapping"
                    )
                label_name = dataset_labels[label_index]

            if label_name not in model_label2id:
                raise ValueError(
                    f"Dataset label {label_name!r} is missing from the model id2label mapping"
                )
            remapped_sequence.append(model_label2id[label_name])
        remapped.append(remapped_sequence)
    return remapped


def _decode_modes(requested: str) -> tuple[str, ...]:
    return ("argmax", "viterbi") if requested == "both" else (requested,)


def _decode_predictions(
    logits,
    attention_mask,
    label_names: Sequence[str],
    decode_mode: str,
    transition_biases: Mapping[str, float],
) -> list[list[int]]:
    import mlx.core as mx

    if decode_mode == "argmax":
        return mx.argmax(logits, axis=-1).tolist()

    logits_list = logits.tolist()
    attention = attention_mask.tolist()
    special_tokens_mask = [
        [not bool(value) for value in sequence] for sequence in attention
    ]
    return viterbi_decode_bioes_batch(
        emissions=logits_list,
        label_names=label_names,
        special_tokens_mask=special_tokens_mask,
        transition_biases=transition_biases,
    )


def evaluate_dataset(
    *,
    model,
    tokenizer,
    dataset: Dataset,
    dataset_labels: Optional[Sequence[str]],
    label_names: Sequence[str],
    model_label2id: Mapping[str, int],
    batch_size: int,
    max_length: int,
    decode_modes: Sequence[str],
    transition_biases: Mapping[str, float],
    include_outside_in_macro: bool,
) -> dict[str, Any]:
    import mlx.core as mx

    from mlx_raclate.tuner.collators import DataCollatorForTokenClassification

    if dataset_labels is not None:
        missing_labels = set(dataset_labels).difference(model_label2id)
        if missing_labels:
            raise ValueError(
                "Dataset labels are missing from the model id2label mapping: "
                + ", ".join(sorted(missing_labels))
            )

    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer._tokenizer,
        max_length=max_length,
    )
    predictions = {decode_mode: [] for decode_mode in decode_modes}
    references = []
    total_loss = 0.0
    loss_batches = 0
    started_at = time.perf_counter()

    model.eval()
    for start in range(0, len(dataset), batch_size):
        raw_batch = dataset[start : min(start + batch_size, len(dataset))]
        raw_batch["labels"] = remap_reference_sequences(
            raw_batch["labels"], dataset_labels, model_label2id
        )
        batch = collator(raw_batch)
        outputs = model(**batch)
        logits = outputs.get("logits")
        if logits is None:
            raise ValueError("Token-classification model output did not contain logits")

        loss = outputs.get("loss")
        if loss is not None:
            total_loss += mx.mean(loss).item()
            loss_batches += 1

        references.extend(batch["labels"].tolist())
        for decode_mode in decode_modes:
            predictions[decode_mode].extend(
                _decode_predictions(
                    logits,
                    batch["attention_mask"],
                    label_names,
                    decode_mode,
                    transition_biases,
                )
            )
        mx.clear_cache()

    elapsed = time.perf_counter() - started_at
    id2label = {index: label for index, label in enumerate(label_names)}
    metrics_by_decoder = {}
    for decode_mode in decode_modes:
        metrics_by_decoder[decode_mode] = compute_token_classification_metrics(
            predictions[decode_mode],
            references,
            id2label=id2label,
            include_outside_in_macro=include_outside_in_macro,
        )

    return {
        "examples": len(dataset),
        "evaluated_tokens": next(iter(metrics_by_decoder.values()))["token_total"],
        "mean_batch_loss": total_loss / loss_batches if loss_batches else None,
        "elapsed_seconds": elapsed,
        "examples_per_second": len(dataset) / elapsed if elapsed else None,
        "metrics": metrics_by_decoder,
    }


def _print_summary(result: Mapping[str, Any]) -> None:
    for decoder, metrics in result["metrics"].items():
        print(
            f"{result['model']} | {result['dataset']} | {decoder} | "
            f"entity micro P/R/F1 "
            f"{metrics['boundary_precision_micro']:.4f}/"
            f"{metrics['boundary_recall_micro']:.4f}/"
            f"{metrics['boundary_f1_micro']:.4f} | "
            f"entity macro F1 {metrics['boundary_f1_macro']:.4f} | "
            f"token accuracy {metrics['token_accuracy']:.4f}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or more NER checkpoints with token-level and exact-boundary "
            "precision, recall, and F1 metrics."
        )
    )
    parser.add_argument(
        "--model",
        nargs="+",
        required=True,
        help="Local checkpoint paths or Hugging Face model IDs.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help=(
            "Dataset files, split directories, multi-phase roots, or Hugging Face "
            "dataset IDs."
        ),
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument(
        "--max-length",
        type=_positive_int,
        help="Override the model context length used by the evaluation collator.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Evaluate only the first N examples from each dataset (useful for smoke tests).",
    )
    parser.add_argument(
        "--decode",
        choices=("argmax", "viterbi", "both"),
        default="viterbi",
        help="Token decoding strategy; Viterbi enforces valid BIO/BIOES transitions.",
    )
    parser.add_argument(
        "--operating-point",
        default="default",
        help="Viterbi calibration operating point stored with the checkpoint.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="Explicit labels.json for datasets without ClassLabel metadata.",
    )
    parser.add_argument(
        "--include-outside-in-macro",
        action="store_true",
        help="Include the O label in macro token metrics (excluded by default).",
    )
    parser.add_argument("--output", type=Path, help="Write the complete benchmark as JSON.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing --output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output is not None and args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
    if args.labels is not None and not args.labels.is_file():
        raise FileNotFoundError(f"Labels file not found: {args.labels}")

    sources = discover_evaluation_sources(args.dataset, args.split)
    loaded_datasets = []
    for source in sources:
        dataset = load_evaluation_dataset(source, args.split)
        if args.limit is not None:
            dataset = dataset.select(range(min(args.limit, len(dataset))))
        if not len(dataset):
            raise ValueError(f"Dataset {source.name!r} is empty")
        dataset_labels, label_source = resolve_dataset_labels(
            dataset, source, args.labels
        )
        loaded_datasets.append((source, dataset, dataset_labels, label_source))

    benchmark = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "decode": args.decode,
        "operating_point": args.operating_point,
        "results": [],
    }

    import mlx.core as mx

    from mlx_raclate.utils.utils import load

    for model_path in args.model:
        print(f"Loading {model_path}...")
        model, tokenizer = load(model_path, pipeline="token-classification")
        label_names, model_label2id = _model_label_mapping(model)
        transition_biases = {}
        if args.decode in {"viterbi", "both"}:
            transition_biases = viterbi_transition_biases_from_calibration(
                getattr(model, "viterbi_calibration", None),
                operating_point=args.operating_point,
            )
        max_length = (
            args.max_length
            or getattr(model.config, "max_position_embeddings", None)
            or 512
        )

        for source, dataset, dataset_labels, label_source in loaded_datasets:
            print(f"Evaluating {source.name} ({len(dataset)} examples)...")
            result = evaluate_dataset(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                dataset_labels=dataset_labels,
                label_names=label_names,
                model_label2id=model_label2id,
                batch_size=args.batch_size,
                max_length=max_length,
                decode_modes=_decode_modes(args.decode),
                transition_biases=transition_biases,
                include_outside_in_macro=args.include_outside_in_macro,
            )
            result.update(
                {
                    "model": model_path,
                    "dataset": source.name,
                    "dataset_location": source.location,
                    "dataset_label_source": label_source,
                    "dataset_labels": (
                        list(dataset_labels) if dataset_labels is not None else None
                    ),
                    "model_labels": label_names,
                    "max_length": max_length,
                    "batch_size": args.batch_size,
                }
            )
            benchmark["results"].append(result)
            _print_summary(result)

        del model
        del tokenizer
        gc.collect()
        mx.clear_cache()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(benchmark, handle, indent=2)
        print(f"Wrote complete metrics to {args.output}")
    else:
        print(json.dumps(benchmark, indent=2))


if __name__ == "__main__":
    main()
