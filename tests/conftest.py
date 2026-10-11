"""Shared fixtures and paths for the test suite.

The suite is offline by default. Coordinate files and one entry record are
committed under `fixtures/`, so a fresh clone can run everything with nothing
but numpy, scipy and pytest. Tests that genuinely need the network carry
`@pytest.mark.network` and are deselected by the `addopts` in pyproject.toml;
run them with `pytest -m network`.
"""

from __future__ import annotations

import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXTURES = os.path.join(HERE, "fixtures")

#: Entries with a committed CIF file.
ENTRIES = ("1CRN", "1UBQ", "2MBW", "1L2Y", "4HHB")


def fixture_path(name: str) -> str:
    path = os.path.join(FIXTURES, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing fixture: {path}")
    return path


def cif_path(entry: str) -> str:
    return fixture_path(f"{entry.upper()}.cif")


@pytest.fixture(scope="session")
def fixtures_dir() -> str:
    return FIXTURES


@pytest.fixture(scope="session")
def repo_root() -> str:
    return ROOT


@pytest.fixture(scope="session")
def sasa_reference() -> dict:
    """FreeSASA totals committed by tests/gen_reference.py."""
    with open(fixture_path("sasa_reference.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def entry_1crn() -> dict:
    with open(fixture_path("entry_1CRN.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def crambin():
    """1CRN as a selected structure: first model, no waters, no hydrogens."""
    from pdbpeek import parse_file, select

    return select(parse_file(cif_path("1CRN")), drop_hetatm=True)
