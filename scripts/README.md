# Release gate

Run the release gate before publishing a new Raclate version:

```bash
python3 scripts/check_release.py
```

This builds a source distribution and wheel from the current source tree in a
temporary directory. It validates their contents, core and optional dependency
metadata, console entry point, wheel integrity record, and agreement on package
identity. It then installs the wheel into a fresh virtual environment using
`--no-deps --no-index` and smoke-tests the public import and CLI help commands.
That last step ensures metadata and command discovery do not eagerly require
MLX or any other runtime dependency.

Use `--offline` when the Hatchling build dependency is already present in uv's
cache and network access must be prohibited. Successful validation means the
artifacts are structurally ready for upload; it does not upload them or reserve
the version on PyPI.
