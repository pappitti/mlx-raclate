"""Tune BIO Viterbi transition biases for a token-classification checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_raclate.utils.token_classification_calibration import (
    load_bio_calibration_corpus,
    search_bio_transition_biases,
    write_calibration_operating_points,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _retention(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return parsed


def _bias_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be comma-separated numbers") from error
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one number")
    return values


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="raclate ner calibrate",
        description=(
            "Run calibration inference for a checkpoint, search BIO Viterbi "
            "transition biases, and write balanced, high-precision, and high-recall "
            "operating points. Existing logits caches can also be reused."
        )
    )
    inputs = parser.add_argument_group("calibration inputs")
    source = inputs.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model",
        help=(
            "Checkpoint to calibrate. With no --output-calibration, this must be a "
            "local checkpoint directory and is updated in place."
        ),
    )
    source.add_argument(
        "--cache",
        nargs="+",
        type=Path,
        help="Advanced: reuse one or more .npz caches from 'raclate ner evaluate'.",
    )
    inputs.add_argument(
        "--dataset",
        nargs="+",
        help=(
            "Dataset files, split directories, multi-phase roots, or Hugging Face "
            "dataset IDs. Required with --model."
        ),
    )
    inputs.add_argument("--split", default="validation")
    inputs.add_argument("--batch-size", type=_positive_int, default=1)
    inputs.add_argument(
        "--max-length",
        type=_positive_int,
        help="Override the model context length used for calibration inference.",
    )
    inputs.add_argument(
        "--limit",
        type=_positive_int,
        help="Use only the first N examples from each calibration dataset.",
    )
    inputs.add_argument(
        "--labels",
        type=Path,
        help="Explicit labels.json for datasets without ClassLabel metadata.",
    )
    inputs.add_argument(
        "--logits-cache-dtype",
        choices=("float16", "float32"),
        default="float32",
        help="Precision of the temporary logits caches used by one-step calibration.",
    )
    output = parser.add_argument_group("calibration output")
    output.add_argument(
        "--output-calibration",
        type=Path,
        help=(
            "Destination viterbi_calibration.json or checkpoint directory. Defaults "
            "to the local --model checkpoint."
        ),
    )
    search = parser.add_argument_group("advanced search options")
    search.add_argument(
        "--objective",
        choices=("micro", "macro"),
        default="micro",
        help="Boundary F1 averaging method optimized by the balanced point.",
    )
    search.add_argument(
        "--grid",
        type=_bias_grid,
        default=_bias_grid("-2,-1.5,-1,-0.5,0,0.5,1,1.5,2"),
        help="Comma-separated absolute bias values used by coordinate search.",
    )
    search.add_argument("--passes", type=_positive_int, default=2)
    search.add_argument(
        "--refine-step",
        type=_nonnegative_float,
        default=0.25,
        help="Local refinement step after the coarse coordinate search; zero disables it.",
    )
    search.add_argument(
        "--search-max-sequences",
        type=_positive_int,
        default=128,
        help=(
            "Deterministic sequence sample used for the broad search. Finalists are "
            "always rescored on every cached sequence."
        ),
    )
    search.add_argument(
        "--finalists",
        type=_positive_int,
        default=4,
        help="Candidates retained per balanced/precision/recall ranking for full scoring.",
    )
    search.add_argument(
        "--retention",
        type=_retention,
        default=0.95,
        help=(
            "High-precision keeps at least this fraction of balanced recall; high-recall "
            "keeps the same fraction of balanced precision."
        ),
    )
    search.add_argument("--seed", type=int, default=42)
    output.add_argument(
        "--set-default",
        action="store_true",
        help="Also replace the checkpoint's default point with the balanced result.",
    )
    output.add_argument(
        "--overwrite-operating-points",
        action="store_true",
        help="Allow replacing existing balanced/high_precision/high_recall entries.",
    )
    args = parser.parse_args(argv)
    if args.model is not None:
        if not args.dataset:
            parser.error("--dataset is required with --model")
        if args.output_calibration is None:
            checkpoint = Path(args.model)
            if not checkpoint.is_dir():
                parser.error(
                    "--output-calibration is required when --model is not a local "
                    "checkpoint directory"
                )
            args.output_calibration = checkpoint
    else:
        if args.dataset:
            parser.error("--dataset can only be used with --model")
        if args.labels is not None:
            parser.error("--labels can only be used with --model")
        if args.output_calibration is None:
            parser.error("--output-calibration is required with --cache")
    return args


def _format_metrics(metrics) -> str:
    return (
        f"P/R/F1 {metrics['boundary_precision_micro']:.4f}/"
        f"{metrics['boundary_recall_micro']:.4f}/"
        f"{metrics['boundary_f1_micro']:.4f}; "
        f"macro F1 {metrics['boundary_f1_macro']:.4f}; "
        f"spans {metrics['predicted_spans']}/{metrics['reference_spans']}"
    )


def _generate_logits_caches(args, cache_dir: Path) -> tuple[Path, ...]:
    from mlx_raclate.ner.evaluate import EvaluationRequest, run_evaluation

    print(
        f"Generating calibration logits for {len(args.dataset)} dataset source(s)..."
    )
    evaluation = run_evaluation(
        EvaluationRequest(
            model=(args.model,),
            dataset=tuple(args.dataset),
            split=args.split,
            batch_size=args.batch_size,
            max_length=args.max_length,
            limit=args.limit,
            decode="argmax",
            labels=args.labels,
            logits_cache_dir=cache_dir,
            logits_cache_dtype=args.logits_cache_dtype,
        ),
        emit_progress=False,
        emit_json=False,
    )
    if not evaluation.logits_cache_paths:
        raise RuntimeError("Calibration inference did not produce any logits caches")
    return evaluation.logits_cache_paths


def _search_and_write(
    args,
    cache_paths: Sequence[Path],
    *,
    generated_from: dict[str, object] | None = None,
) -> Path:
    missing = [str(path) for path in cache_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing logits caches: " + ", ".join(missing))

    print(f"Loading {len(cache_paths)} cached-logit dataset(s)...")
    corpus = load_bio_calibration_corpus(cache_paths)
    print(
        f"Searching {len(corpus.sequences)} sequences with labels "
        f"{', '.join(corpus.label_names)}"
    )
    result = search_bio_transition_biases(
        corpus,
        grid=args.grid,
        passes=args.passes,
        refine_step=args.refine_step,
        search_max_sequences=args.search_max_sequences,
        finalists=args.finalists,
        retention=args.retention,
        objective=args.objective,
        seed=args.seed,
    )
    search_metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_metadata": [dict(metadata) for metadata in corpus.cache_metadata],
    }
    if generated_from is None:
        search_metadata["cache_files"] = [str(path) for path in cache_paths]
    else:
        search_metadata["generated_from"] = generated_from
    result["search"].update(search_metadata)

    print("zero_bias:", _format_metrics(result["search"]["zero_bias_metrics"]))
    for name, operating_point in result["operating_points"].items():
        active_biases = {
            key: value
            for key, value in operating_point["biases"].items()
            if value != 0.0
        }
        print(f"{name}: {_format_metrics(operating_point['calibration_metrics'])}")
        print(f"  biases: {active_biases or {'all': 0.0}}")

    written_path = write_calibration_operating_points(
        args.output_calibration,
        result,
        overwrite_operating_points=args.overwrite_operating_points,
        set_default=args.set_default,
    )
    print(f"Wrote Viterbi operating points to {written_path}")
    return written_path


def main(argv: Sequence[str] | None = None) -> Path:
    args = _parse_args(argv)
    if args.model is None:
        return _search_and_write(args, args.cache)

    with TemporaryDirectory(prefix="mlx-raclate-ner-calibration-") as directory:
        cache_paths = _generate_logits_caches(args, Path(directory))
        return _search_and_write(
            args,
            cache_paths,
            generated_from={
                "model": args.model,
                "datasets": list(args.dataset),
                "split": args.split,
                "max_length": args.max_length,
                "limit": args.limit,
                "logits_cache_dtype": args.logits_cache_dtype,
            },
        )


if __name__ == "__main__":
    main()
