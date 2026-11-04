"""Smoke test: the package is installed into the venv and importable."""

import faulttrack


def test_import_exposes_version() -> None:
    assert isinstance(faulttrack.__version__, str)
    assert faulttrack.__version__
