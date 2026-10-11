"""Tests for fetching, split into offline and network halves.

The offline tests cover id validation and error handling, which is where the
bugs actually are. The network tests confirm the three endpoints the demo page
depends on are still shaped the way the template expects. They carry
`@pytest.mark.network` and are deselected by default, so a fresh clone runs
without touching the internet:

    pytest              # offline, hermetic
    pytest -m network   # the live endpoint checks too
"""

from __future__ import annotations

import pytest

from pdbpeek.fetch import (
    PDBE_SUMMARY_URL,
    RCSB_ENTRY_URL,
    RCSB_FILE_URL,
    FetchError,
    fetch_cif,
    fetch_entry,
    fetch_pdbe_summary,
    valid_pdb_id,
)
from pdbpeek.mmcif import parse_mmcif, select
from pdbpeek.sasa import compute_sasa
from pdbpeek.summary import summarize_entry, summarize_pdbe

# --------------------------------------------------------------------------
# offline
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pdb_id", ["1CRN", "1ubq", "4HHB", "1l2y", "2mbw", "9ZZZ"])
def test_valid_ids_are_accepted(pdb_id):
    assert valid_pdb_id(pdb_id.upper())


@pytest.mark.parametrize(
    "pdb_id",
    ["", "1CR", "1CRNN", "ABCD", "1-CR", "1 CR", "crambin", "../etc/passwd", "1CR."],
)
def test_invalid_ids_are_rejected(pdb_id):
    assert not valid_pdb_id(pdb_id)


@pytest.mark.parametrize("bad", ["", "nope", "1CR", "../../etc/passwd"])
def test_fetch_refuses_a_bad_id_before_touching_the_network(bad):
    # A rejected id must raise ValueError, not reach out and fail later. This
    # is also what stops a crafted id becoming a path or URL injection.
    with pytest.raises(ValueError, match="not a PDB id"):
        fetch_cif(bad)
    with pytest.raises(ValueError, match="not a PDB id"):
        fetch_entry(bad)


def test_urls_are_https_and_point_at_the_expected_hosts():
    assert RCSB_FILE_URL.startswith("https://files.rcsb.org/")
    assert RCSB_ENTRY_URL.startswith("https://data.rcsb.org/")
    assert PDBE_SUMMARY_URL.startswith("https://www.ebi.ac.uk/")
    for template in (RCSB_FILE_URL, RCSB_ENTRY_URL, PDBE_SUMMARY_URL):
        assert "{pdb_id}" in template


def test_cache_is_used_before_the_network(tmp_path, fixtures_dir):
    # Seed the cache with the committed fixture, then fetch. If the cache were
    # ignored this would hit the network, and it must not.
    import shutil

    cache = tmp_path / "cache"
    cache.mkdir()
    shutil.copy(f"{fixtures_dir}/1CRN.cif", cache / "1CRN.cif")

    text = fetch_cif("1CRN", cache_dir=str(cache))
    assert "data_1CRN" in text
    assert len(parse_mmcif(text).atoms) == 327


def test_cache_lookup_is_case_insensitive_on_the_id(tmp_path, fixtures_dir):
    import shutil

    cache = tmp_path / "cache"
    cache.mkdir()
    shutil.copy(f"{fixtures_dir}/1CRN.cif", cache / "1CRN.cif")
    assert "data_1CRN" in fetch_cif("1crn", cache_dir=str(cache))


def test_fetch_error_is_raised_for_an_unreachable_host(monkeypatch):
    monkeypatch.setattr("pdbpeek.fetch.RCSB_FILE_URL", "https://localhost:1/{pdb_id}")
    with pytest.raises(FetchError, match="failed"):
        fetch_cif("1CRN")


def test_missing_curl_is_reported_clearly(monkeypatch):
    monkeypatch.setattr("pdbpeek.fetch.shutil.which", lambda _: None)
    with pytest.raises(FetchError, match="curl is not on PATH"):
        fetch_entry("1CRN")


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------


@pytest.mark.network
def test_live_cif_download_parses_and_matches_the_fixture(fixtures_dir):
    # If RCSB ever revises 1CRN the committed fixture would go stale, and
    # every SASA reference number with it. This is the canary for that.
    live = fetch_cif("1CRN")
    with open(f"{fixtures_dir}/1CRN.cif", "r", encoding="utf-8") as fh:
        committed = fh.read()
    assert len(parse_mmcif(live).atoms) == len(parse_mmcif(committed).atoms) == 327


@pytest.mark.network
def test_live_entry_record_has_the_fields_the_template_reads():
    entry = fetch_entry("1CRN")
    assert entry["struct"]["title"]
    assert entry["exptl"][0]["method"] == "X-RAY DIFFRACTION"
    assert entry["rcsb_entry_info"]["resolution_combined"][0] == 1.5
    assert entry["rcsb_entry_info"]["polymer_entity_count"] == 1
    assert entry["struct_keywords"]["pdbx_keywords"]
    assert entry["rcsb_primary_citation"]["year"] == 1984


@pytest.mark.network
def test_live_summary_matches_the_committed_golden(fixtures_dir):
    with open(f"{fixtures_dir}/summary_1CRN.golden.txt", "r", encoding="utf-8") as fh:
        expected = fh.read()
    assert summarize_entry(fetch_entry("1CRN")) == expected


@pytest.mark.network
def test_live_pdbe_summary_renders():
    text = summarize_pdbe(fetch_pdbe_summary("1CRN"), "1CRN")
    assert text.startswith("1CRN: ")
    assert "None" not in text


@pytest.mark.network
def test_end_to_end_from_a_live_download():
    structure = select(parse_mmcif(fetch_cif("1CRN")), drop_hetatm=True)
    result = compute_sasa(structure, n_points=256, reference=False)
    assert len(structure.atoms) == 327
    # Matches the FreeSASA reference total to within sampling error.
    assert 2900 < result.total < 3050


@pytest.mark.network
def test_unknown_entry_raises_rather_than_returning_junk():
    with pytest.raises(FetchError):
        fetch_cif("9ZZZ")
