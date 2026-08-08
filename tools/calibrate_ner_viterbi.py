"""Tune BIO Viterbi transition biases from cached token-classification logits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search BIO Viterbi transition biases from one or more cached-logit "
            "datasets and write balanced, high-precision, and high-recall operating "
            "points. No model inference or training is performed."
        )
    )
    parser.add_argument(
        "--cache",
        nargs="+",
        type=Path,
        required=True,
        help="One or more .npz caches written by evaluate_ner_models.py.",
    )
    parser.add_argument(
        "--output-calibration",
        type=Path,
        required=True,
        help=(
            "Destination viterbi_calibration.json or checkpoint directory. Existing "
            "unrelated operating points are preserved."
        ),
    )
    parser.add_argument(
        "--objective",
        choices=("micro", "macro"),
        default="micro",
        help="Boundary F1 averaging method optimized by the balanced point.",
    )
    parser.add_argument(
        "--grid",
        type=_bias_grid,
        default=_bias_grid("-2,-1.5,-1,-0.5,0,0.5,1,1.5,2"),
        help="Comma-separated absolute bias values used by coordinate search.",
    )
    parser.add_argument("--passes", type=_positive_int, default=2)
    parser.add_argument(
        "--refine-step",
        type=_nonnegative_float,
        default=0.25,
        help="Local refinement step after the coarse coordinate search; zero disables it.",
    )
    parser.add_argument(
        "--search-max-sequences",
        type=_positive_int,
        default=128,
        help=(
            "Deterministic sequence sample used for the broad search. Finalists are "
            "always rescored on every cached sequence."
        ),
    )
    parser.add_argument(
        "--finalists",
        type=_positive_int,
        default=4,
        help="Candidates retained per balanced/precision/recall ranking for full scoring.",
    )
    parser.add_argument(
        "--retention",
        type=_retention,
        default=0.95,
        help=(
            "High-precision keeps at least this fraction of balanced recall; high-recall "
            "keeps the same fraction of balanced precision."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--set-default",
        action="store_true",
        help="Also replace the checkpoint's default point with the balanced result.",
    )
    parser.add_argument(
        "--overwrite-operating-points",
        action="store_true",
        help="Allow replacing existing balanced/high_precision/high_recall entries.",
    )
    return parser.parse_args()


def _format_metrics(metrics) -> str:
    return (
        f"P/R/F1 {metrics['boundary_precision_micro']:.4f}/"
        f"{metrics['boundary_recall_micro']:.4f}/"
        f"{metrics['boundary_f1_micro']:.4f}; "
        f"macro F1 {metrics['boundary_f1_macro']:.4f}; "
        f"spans {metrics['predicted_spans']}/{metrics['reference_spans']}"
    )


def main() -> None:
    args = _parse_args()
    missing = [str(path) for path in args.cache if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing logits caches: " + ", ".join(missing))

    print(f"Loading {len(args.cache)} cached-logit dataset(s)...")
    corpus = load_bio_calibration_corpus(args.cache)
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
    result["search"].update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cache_files": [str(path) for path in args.cache],
            "cache_metadata": [dict(metadata) for metadata in corpus.cache_metadata],
        }
    )

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


if __name__ == "__main__":
    main()
