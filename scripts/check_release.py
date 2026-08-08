#!/usr/bin/env python3
"""Build and validate the distributions that would be uploaded to PyPI."""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "mlx_raclate"
FORBIDDEN_PARTS = {
    ".DS_Store",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "data",
    "dev_tests",
    "dist",
    "trained_models",
}


class ReleaseCheckError(RuntimeError):
    """Raised when a built distribution is not ready to publish."""


def _run(command: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseCheckError(message)


def _source_package_files() -> set[str]:
    return {
        path.relative_to(PACKAGE_ROOT.parent).as_posix()
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file()
        and not FORBIDDEN_PARTS.intersection(path.relative_to(ROOT).parts)
        and path.suffix not in {".pyc", ".pyo"}
    }


def _metadata(data: bytes, source: str) -> Message:
    message = BytesParser(policy=policy.default).parsebytes(data)
    _require(not message.defects, f"Malformed metadata in {source}: {message.defects}")
    return message


def _compact_requirement(requirement: str) -> str:
    """Normalize insignificant formatting without needing ``packaging``."""
    return re.sub(r"\s+", "", requirement).replace('"', "'").lower()


def _requirements_by_extra(
    requirements: Iterable[str],
) -> dict[str | None, Counter[str]]:
    grouped: dict[str | None, Counter[str]] = defaultdict(Counter)
    for requirement in requirements:
        dependency, _, marker = requirement.partition(";")
        extra_match = re.search(r"\bextra\s*==\s*['\"]([^'\"]+)['\"]", marker)
        grouped[extra_match.group(1) if extra_match else None][
            _compact_requirement(dependency)
        ] += 1
    return dict(grouped)


def _check_metadata(metadata: Message, project: dict[str, object]) -> None:
    name = project["name"]
    version = project["version"]
    _require(metadata["Name"] == name, f"Metadata name is not {name!r}")
    _require(metadata["Version"] == version, f"Metadata version is not {version!r}")
    _require(
        metadata["Summary"] == project["description"],
        "Wheel summary does not match pyproject.toml",
    )
    _require(
        metadata["Requires-Python"] == project["requires-python"],
        "Requires-Python does not match pyproject.toml",
    )
    _require(
        metadata["Description-Content-Type"] == "text/markdown",
        "README is not published as Markdown",
    )
    _require(metadata.get_payload().strip(), "Published long description is empty")
    _require("LICENSE" in metadata.get_all("License-File", []), "LICENSE is missing")

    optional = project.get("optional-dependencies", {})
    _require(
        isinstance(optional, dict), "project.optional-dependencies must be a table"
    )
    expected_extras = set(optional)
    actual_extras = set(metadata.get_all("Provides-Extra", []))
    _require(
        actual_extras == expected_extras,
        f"Published extras differ: expected {expected_extras}, got {actual_extras}",
    )

    expected_requirements: dict[str | None, Counter[str]] = {
        None: Counter(
            _compact_requirement(requirement.split(";", 1)[0])
            for requirement in project.get("dependencies", [])
        )
    }
    for extra, requirements in optional.items():
        expected_requirements[extra] = Counter(
            _compact_requirement(requirement.split(";", 1)[0])
            for requirement in requirements
        )
    actual_requirements = _requirements_by_extra(metadata.get_all("Requires-Dist", []))
    _require(
        actual_requirements == expected_requirements,
        "Published core or optional dependencies differ from pyproject.toml",
    )


def _check_record(archive: zipfile.ZipFile, record_name: str) -> None:
    rows = list(csv.reader(archive.read(record_name).decode().splitlines()))
    _require(all(len(row) == 3 for row in rows), "Wheel RECORD has malformed rows")
    recorded = {row[0]: (row[1], row[2]) for row in rows}
    names = set(archive.namelist())
    _require(set(recorded) == names, "Wheel RECORD does not list every archive member")

    for name, (digest, size) in recorded.items():
        if name == record_name:
            _require(not digest and not size, "RECORD must not hash itself")
            continue
        contents = archive.read(name)
        encoded = base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(
            b"="
        )
        _require(digest == f"sha256={encoded.decode()}", f"Bad RECORD hash for {name}")
        _require(size == str(len(contents)), f"Bad RECORD size for {name}")


def _check_wheel(wheel: Path, project: dict[str, object]) -> Message:
    expected_files = _source_package_files()
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        _require(len(names) == len(set(names)), "Wheel contains duplicate members")
        _require(
            not any(
                FORBIDDEN_PARTS.intersection(PurePosixPath(name).parts)
                for name in names
            ),
            "Wheel contains cache, development, data, or generated files",
        )
        packaged_files = {name for name in names if name.startswith("mlx_raclate/")}
        _require(
            packaged_files == expected_files,
            "Wheel package files do not exactly match src/mlx_raclate",
        )

        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        entry_point_names = [
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        ]
        wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
        record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
        _require(len(metadata_names) == 1, "Wheel must contain one METADATA file")
        _require(len(entry_point_names) == 1, "Wheel must contain one entry_points.txt")
        _require(len(wheel_names) == 1, "Wheel must contain one WHEEL file")
        _require(len(record_names) == 1, "Wheel must contain one RECORD file")

        metadata = _metadata(archive.read(metadata_names[0]), metadata_names[0])
        _check_metadata(metadata, project)

        entries = configparser.ConfigParser(interpolation=None)
        entries.read_string(archive.read(entry_point_names[0]).decode())
        _require(
            entries.get("console_scripts", "raclate", fallback=None)
            == "mlx_raclate.cli:main",
            "The raclate console entry point is missing or incorrect",
        )

        wheel_metadata = _metadata(archive.read(wheel_names[0]), wheel_names[0])
        _require(
            wheel_metadata["Root-Is-Purelib"] == "true", "Wheel is not pure Python"
        )
        _require(
            "py3-none-any" in wheel_metadata.get_all("Tag", []),
            "Wheel does not declare the expected portable tag",
        )
        _check_record(archive, record_names[0])
    return metadata


def _check_sdist(
    sdist: Path, project: dict[str, object], wheel_metadata: Message
) -> None:
    expected_root = f"mlx_raclate-{project['version']}"
    expected_package = {
        f"{expected_root}/src/{name}" for name in _source_package_files()
    }
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
        _require(
            len(names) == len(set(names)), "Source distribution has duplicate members"
        )
        roots = {PurePosixPath(name).parts[0] for name in names if name}
        _require(roots == {expected_root}, f"Unexpected sdist root directory: {roots}")
        _require(
            not any(
                FORBIDDEN_PARTS.intersection(PurePosixPath(name).parts)
                for name in names
            ),
            "Source distribution contains local data, caches, or generated files",
        )
        for required in ("LICENSE", "README.md", "pyproject.toml", "PKG-INFO"):
            _require(
                f"{expected_root}/{required}" in names,
                f"Source distribution is missing {required}",
            )
        packaged_source = {
            name
            for name in names
            if name.startswith(f"{expected_root}/src/mlx_raclate/")
        }
        _require(
            packaged_source == expected_package,
            "Source distribution package files do not exactly match src/mlx_raclate",
        )
        pkg_info_member = archive.extractfile(f"{expected_root}/PKG-INFO")
        _require(pkg_info_member is not None, "Could not read sdist PKG-INFO")
        pkg_info = _metadata(pkg_info_member.read(), "sdist PKG-INFO")
        _check_metadata(pkg_info, project)
        for key in ("Name", "Version", "Requires-Python"):
            _require(
                pkg_info.get_all(key) == wheel_metadata.get_all(key),
                f"sdist and wheel disagree on {key}",
            )


def _smoke_test_installed_wheel(
    wheel: Path, project: dict[str, object], work: Path
) -> None:
    venv = work / "installed"
    _run([sys.executable, "-m", "venv", str(venv)])
    scripts = venv / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    raclate = scripts / ("raclate.exe" if os.name == "nt" else "raclate")
    smoke_env = os.environ.copy()
    smoke_env.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    smoke_env.pop("PYTHONPATH", None)
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            str(wheel),
        ],
        env=smoke_env,
    )
    smoke = (
        "import importlib.util, sys; "
        "assert importlib.util.find_spec('mlx') is None; "
        "import mlx_raclate; "
        f"assert mlx_raclate.__version__ == {project['version']!r}; "
        "assert mlx_raclate.__all__ == ['__version__', 'load']; "
        "assert 'mlx' not in sys.modules"
    )
    _run([str(python), "-I", "-c", smoke], env=smoke_env)
    for arguments in (
        ("--help",),
        ("train", "--help"),
        ("serve", "--help"),
        ("ner", "--help"),
    ):
        _run([str(raclate), *arguments], env=smoke_env)


def _artifacts(directory: Path) -> tuple[Path, Path]:
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    _require(len(wheels) == 1, f"Expected one wheel, found {len(wheels)}")
    _require(len(sdists) == 1, f"Expected one sdist, found {len(sdists)}")
    return wheels[0], sdists[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and validate the exact sdist and wheel intended for PyPI."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require uv to use only cached build dependencies.",
    )
    arguments = parser.parse_args(argv)

    _require(shutil.which("uv") is not None, "uv is required to build distributions")
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]

    try:
        with tempfile.TemporaryDirectory(prefix="mlx-raclate-release-") as temporary:
            work = Path(temporary)
            artifacts = work / "artifacts"
            command = [
                "uv",
                "build",
                "--no-sources",
                "--out-dir",
                str(artifacts),
                str(ROOT),
            ]
            if arguments.offline:
                command.insert(2, "--offline")
            _run(command)
            wheel, sdist = _artifacts(artifacts)
            wheel_metadata = _check_wheel(wheel, project)
            _check_sdist(sdist, project, wheel_metadata)
            _smoke_test_installed_wheel(wheel, project, work)
    except (
        OSError,
        subprocess.CalledProcessError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as error:
        raise ReleaseCheckError(str(error)) from error

    print(
        f"Release gate passed for {project['name']} {project['version']}: "
        "sdist, wheel, metadata, RECORD, entry point, extras, and isolated smoke tests."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseCheckError as error:
        raise SystemExit(f"release check failed: {error}") from None
