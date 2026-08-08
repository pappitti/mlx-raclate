import subprocess
import sys

import pytest

from mlx_raclate import cli


pytestmark = pytest.mark.release


def test_main_forwards_command_help(monkeypatch):
    received = []

    def command(arguments):
        received.extend(arguments)
        return 17

    monkeypatch.setattr(cli, "_load_command", lambda name: ("description", command))

    assert cli.main(["train", "--help"]) == 17
    assert received == ["--help"]


def test_main_without_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "{train,serve,ner}" in capsys.readouterr().out


def test_main_rejects_unknown_command():
    with pytest.raises(SystemExit, match="2"):
        cli.main(["unknown"])


def test_ner_help_reaches_selected_command(capsys):
    with pytest.raises(SystemExit, match="0"):
        cli._run_ner(["evaluate", "--help"])

    assert "usage: raclate ner evaluate" in capsys.readouterr().out


def test_serve_help_does_not_import_mlx():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from mlx_raclate.cli import main; "
            "\ntry: main(['serve', '--help'])"
            "\nexcept SystemExit as error: assert error.code == 0"
            "\nassert not any(name == 'mlx' or name.startswith('mlx.') "
            "for name in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_train_hub_flags_require_explicit_repo_and_base_model():
    from mlx_raclate.utils.train import init_args

    required = ["--dataset", "data/train", "--model_path", "acme/base-model"]

    with pytest.raises(SystemExit, match="2"):
        init_args([*required, "--push_to_hub"])
    with pytest.raises(SystemExit, match="2"):
        init_args(
            [
                *required,
                "--push_to_hub",
                "--hub_repo_id",
                "acme/fine-tuned-model",
            ]
        )

    args = init_args(
        [
            *required,
            "--push_to_hub",
            "--hub_repo_id",
            "acme/fine-tuned-model",
            "--base_model",
            "acme/base-model",
        ]
    )

    assert args.push_to_hub is True
    assert args.hub_repo_id == "acme/fine-tuned-model"
    assert args.base_model == "acme/base-model"
