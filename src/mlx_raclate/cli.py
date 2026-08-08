"""Unified command-line entry point for mlx-raclate."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence


Command = tuple[str, Callable[[Sequence[str] | None], object]]


def _load_command(name: str) -> Command:
    if name == "train":
        from .utils.train import main

        return "Train or evaluate a Raclate model.", main
    if name == "serve":
        return "Run the Raclate inference API.", _run_serve
    if name == "ner":
        return "Evaluate and calibrate NER checkpoints.", _run_ner
    raise ValueError(f"Unknown command: {name}")


def _run_serve(argv: Sequence[str] | None = None) -> object:
    parser = argparse.ArgumentParser(
        prog="raclate serve",
        description="Run the Raclate inference API.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args(argv)

    try:
        from .utils.server import run
    except ModuleNotFoundError as error:
        if error.name in {"fastapi", "pydantic", "uvicorn"}:
            raise SystemExit(
                "Serving dependencies are not installed. Install "
                "'mlx-raclate[serve]'."
            ) from None
        raise
    return run(host=arguments.host, port=arguments.port)


def _run_ner(argv: Sequence[str] | None = None) -> object:
    parser = argparse.ArgumentParser(
        prog="raclate ner",
        description="Evaluate NER checkpoints or calibrate Viterbi decoding.",
        epilog=(
            "evaluate: report metrics on labelled data\n"
            "calibrate: tune server-side decoder profiles from labelled data\n\n"
            "Recommended calibration: raclate ner calibrate --model CHECKPOINT "
            "--dataset DATASET"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", choices=("evaluate", "calibrate"))
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command = arguments.pop(0)
    if command not in {"evaluate", "calibrate"}:
        parser.error(f"invalid choice: {command!r} (choose from 'evaluate', 'calibrate')")

    try:
        if command == "evaluate":
            from .ner.evaluate import main
        else:
            from .ner.calibrate import main
        return main(arguments)
    except ModuleNotFoundError as error:
        if error.name == "datasets":
            raise SystemExit(
                "NER evaluation dependencies are not installed. Install "
                "'mlx-raclate[train]'."
            ) from None
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raclate",
        description=(
            "Train, evaluate, run, and serve retrieval and classification models "
            "with MLX."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("train", "serve", "ner"),
        help="Command to run. Use 'raclate COMMAND --help' for command options.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> object:
    parser = _parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command_name = arguments.pop(0)
    if command_name not in {"train", "serve", "ner"}:
        parser.error(
            f"invalid choice: {command_name!r} (choose from 'train', 'serve', 'ner')"
        )

    _, command = _load_command(command_name)
    return command(arguments)


if __name__ == "__main__":
    main()
