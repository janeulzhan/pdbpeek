"""Fetching coordinate files and entry metadata.

`curl` is shelled out to rather than using `urllib`, because a Python install
without `certifi` raises CERTIFICATE_VERIFY_FAILED on these hosts while the
system `curl` validates against the OS trust store and works. Shelling out
also means there is no third-party HTTP dependency to install.

Nothing in the test suite needs this module: fixtures are committed, and the
handful of tests that do hit the network are marked `network` and deselected
by default.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

RCSB_FILE_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
PDBE_SUMMARY_URL = "https://www.ebi.ac.uk/pdbe/api/pdb/entry/summary/{pdb_id}"

TIMEOUT = 30


class FetchError(RuntimeError):
    """Raised when a remote resource could not be retrieved."""


def valid_pdb_id(pdb_id: str) -> bool:
    """True for the four-character PDB id form, e.g. `1CRN`."""
    return len(pdb_id) == 4 and pdb_id[0].isdigit() and pdb_id.isalnum()


def _curl(url: str) -> bytes:
    exe = shutil.which("curl")
    if exe is None:
        raise FetchError("curl is not on PATH, so pdbpeek.fetch cannot run")
    proc = subprocess.run(
        [exe, "-sSL", "--fail", "--max-time", str(TIMEOUT), url],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise FetchError(f"GET {url} failed: {detail or f'curl exit {proc.returncode}'}")
    return proc.stdout


def fetch_cif(pdb_id: str, cache_dir: str | None = None) -> str:
    """Return the mmCIF text for `pdb_id`, caching it when `cache_dir` is set."""
    pdb_id = pdb_id.strip().upper()
    if not valid_pdb_id(pdb_id):
        raise ValueError(f"not a PDB id: {pdb_id!r}")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{pdb_id}.cif")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read()

    text = _curl(RCSB_FILE_URL.format(pdb_id=pdb_id)).decode("utf-8", "replace")
    if cache_dir:
        with open(os.path.join(cache_dir, f"{pdb_id}.cif"), "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


def fetch_entry(pdb_id: str) -> dict:
    """Return the decoded RCSB entry record for `pdb_id`."""
    pdb_id = pdb_id.strip().upper()
    if not valid_pdb_id(pdb_id):
        raise ValueError(f"not a PDB id: {pdb_id!r}")
    raw = _curl(RCSB_ENTRY_URL.format(pdb_id=pdb_id))
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FetchError(f"entry record for {pdb_id} was not JSON") from exc


def fetch_pdbe_summary(pdb_id: str) -> dict:
    """Return the decoded EBI PDBe summary record for `pdb_id`."""
    pdb_id = pdb_id.strip().lower()
    raw = _curl(PDBE_SUMMARY_URL.format(pdb_id=pdb_id))
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FetchError(f"PDBe summary for {pdb_id} was not JSON") from exc
