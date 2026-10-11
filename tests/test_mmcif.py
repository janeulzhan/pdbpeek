"""Tests for the hand-written mmCIF reader."""

from __future__ import annotations

import pytest

from pdbpeek.mmcif import (
    MmcifError,
    parse_file,
    parse_mmcif,
    select,
)
from pdbpeek.mmcif import _split_line as split_line

from .conftest import cif_path

# Atom counts below are the number of ATOM plus HETATM rows in each committed
# file, confirmed with: grep -c '^ATOM\|^HETATM' tests/fixtures/<id>.cif
RAW_ATOM_COUNTS = {
    "1CRN": 327,
    "1UBQ": 660,
    "2MBW": 1410,
    "1L2Y": 11552,
    "4HHB": 4779,
}


@pytest.mark.parametrize("entry,expected", sorted(RAW_ATOM_COUNTS.items()))
def test_parses_every_atom_row(entry, expected):
    structure = parse_file(cif_path(entry))
    assert len(structure.atoms) == expected
    assert structure.entry_id.upper() == entry


def test_entry_id_and_kept_items():
    structure = parse_file(cif_path("1CRN"))
    assert structure.entry_id.upper() == "1CRN"
    assert structure.items["_exptl.method"] == "X-RAY DIFFRACTION"
    assert "CRAMBIN" in structure.items["_struct.title"].upper()


def test_coordinates_match_the_file():
    # First row of the 1CRN _atom_site loop:
    # ATOM 1 N N . THR A 1 1 ? 17.047 14.099 3.625 1.00 13.79 ? 1 THR A N 1
    first = parse_file(cif_path("1CRN")).atoms[0]
    assert first.serial == 1
    assert first.element == "N"
    assert first.atom_name == "N"
    assert first.res_name == "THR"
    assert first.chain == "A"
    assert first.res_seq == "1"
    assert (first.x, first.y, first.z) == (17.047, 14.099, 3.625)
    assert first.occupancy == 1.0
    assert first.b_factor == 13.79
    assert first.model == 1
    assert first.is_hetatm is False


def test_hetatm_flag_and_water_filtering():
    raw = parse_file(cif_path("1UBQ"))
    waters = [a for a in raw.atoms if a.is_hetatm]
    assert waters, "1UBQ has crystallographic waters"
    assert all(a.res_name == "HOH" for a in waters)
    assert len(select(raw, drop_hetatm=True).atoms) == len(raw.atoms) - len(waters)


# --------------------------------------------------------------------------
# multiple models
# --------------------------------------------------------------------------


def test_multi_model_file_reports_every_model():
    raw = parse_file(cif_path("1L2Y"))
    assert len(raw.models) == 38
    assert raw.models == list(range(1, 39))


def test_model_selection_takes_first_by_default():
    raw = parse_file(cif_path("1L2Y"))
    default = select(raw)
    explicit = select(raw, model=1)
    assert default.models == [1]
    assert [a.serial for a in default.atoms] == [a.serial for a in explicit.atoms]
    assert all(a.model == 1 for a in default.atoms)


def test_every_model_has_the_same_atom_count():
    raw = parse_file(cif_path("1L2Y"))
    counts = {m: len(select(raw, model=m).atoms) for m in raw.models}
    assert len(set(counts.values())) == 1, counts


def test_models_differ_in_coordinates():
    raw = parse_file(cif_path("1L2Y"))
    one = select(raw, model=1).atoms
    two = select(raw, model=2).atoms
    assert [a.atom_name for a in one] == [a.atom_name for a in two]
    assert any((a.x, a.y, a.z) != (b.x, b.y, b.z) for a, b in zip(one, two))


def test_unknown_model_raises():
    raw = parse_file(cif_path("1L2Y"))
    with pytest.raises(MmcifError, match="model 99 not in file"):
        select(raw, model=99)


# --------------------------------------------------------------------------
# alternate conformers
# --------------------------------------------------------------------------


def test_fixture_actually_has_altlocs():
    raw = parse_file(cif_path("2MBW"))
    assert {a.alt_id for a in raw.atoms if a.alt_id} == {"A", "B"}


def test_altloc_occupancy_policy_keeps_one_atom_per_name():
    raw = parse_file(cif_path("2MBW"))
    selected = select(raw, altloc="occupancy")
    seen = [(a.residue_key, a.atom_name) for a in selected.atoms]
    assert len(seen) == len(set(seen)), "a conformer slipped through"
    assert len(selected.atoms) < len(raw.atoms)


def test_altloc_occupancy_picks_the_higher_occupancy_conformer():
    raw = parse_file(cif_path("2MBW"))
    groups: dict[tuple, list] = {}
    for atom in raw.atoms:
        if atom.alt_id:
            groups.setdefault((atom.residue_key, atom.atom_name), []).append(atom)
    contested = {k: v for k, v in groups.items() if len(v) > 1}
    assert contested, "fixture should have at least one contested atom"

    chosen = {
        (a.residue_key, a.atom_name): a
        for a in select(raw, altloc="occupancy").atoms
    }
    for key, candidates in contested.items():
        best = max(candidates, key=lambda a: (a.occupancy, a.alt_id or ""))
        assert chosen[key].alt_id == best.alt_id


def test_altloc_first_policy_keeps_file_order():
    raw = parse_file(cif_path("2MBW"))
    selected = select(raw, altloc="first")
    seen = [(a.residue_key, a.atom_name) for a in selected.atoms]
    assert len(seen) == len(set(seen))
    first_b = next(a for a in raw.atoms if a.alt_id == "B")
    kept = next(
        a
        for a in selected.atoms
        if (a.residue_key, a.atom_name) == (first_b.residue_key, first_b.atom_name)
    )
    assert kept.alt_id == "A", "first-seen conformer should win under 'first'"


def test_altloc_all_policy_keeps_every_conformer():
    raw = parse_file(cif_path("2MBW"))
    kept = select(raw, altloc="all", drop_hydrogens=False)
    assert len(kept.atoms) == len(raw.atoms)


def test_unknown_altloc_policy_raises():
    raw = parse_file(cif_path("1CRN"))
    with pytest.raises(ValueError, match="unknown altloc policy"):
        select(raw, altloc="whatever")


# --------------------------------------------------------------------------
# chains
# --------------------------------------------------------------------------


def test_chain_splitting_on_a_four_chain_structure():
    raw = parse_file(cif_path("4HHB"))
    selected = select(raw)
    assert selected.chains == ["A", "B", "C", "D"]

    by_chain = selected.by_chain()
    assert sum(len(v) for v in by_chain.values()) == len(selected.atoms)
    assert set(by_chain) == set(selected.chains)


def test_chain_restriction_matches_the_grouping():
    raw = parse_file(cif_path("4HHB"))
    whole = select(raw)
    only_b = select(raw, chain="B")
    assert all(a.chain == "B" for a in only_b.atoms)
    assert len(only_b.atoms) == len(whole.by_chain()["B"])


def test_missing_chain_raises():
    raw = parse_file(cif_path("4HHB"))
    with pytest.raises(MmcifError, match="chain 'Z' not in model"):
        select(raw, chain="Z")


def test_residue_grouping_is_contiguous_and_complete():
    selected = select(parse_file(cif_path("1CRN")), drop_hetatm=True)
    groups = selected.residues()
    assert len(groups) == 46, "crambin is a 46 residue peptide"
    assert sum(len(atoms) for _, atoms in groups) == len(selected.atoms)
    keys = [k for k, _ in groups]
    assert len(keys) == len(set(keys)), "a residue was split into two groups"


def test_insertion_codes_separate_residues():
    text = _loop_header() + "\n".join(
        [
            "ATOM 1 C CA . ALA A 1 1 A 0.0 0.0 0.0 1.00 0.0 ? 52 ALA A CA 1",
            "ATOM 2 C CA . GLY A 1 2 B 3.8 0.0 0.0 1.00 0.0 ? 52 GLY A CA 1",
        ]
    )
    structure = parse_mmcif(text)
    assert [a.ins_code for a in structure.atoms] == ["A", "B"]
    assert len(structure.residues()) == 2


# --------------------------------------------------------------------------
# tokenizing
# --------------------------------------------------------------------------


def test_split_line_handles_quotes_and_comments():
    assert split_line("ATOM 1 C 'CA ' A") == ["ATOM", "1", "C", "CA ", "A"]
    assert split_line("""_x "a b" c""") == ["_x", "a b", "c"]
    assert split_line("a b # trailing comment") == ["a", "b"]
    assert split_line("   ") == []


def test_split_line_keeps_primed_atom_names():
    # A quote mid-token is not a delimiter, which is what keeps nucleic acid
    # atom names such as C1' and O5' intact.
    assert split_line("ATOM C1' O5' .") == ["ATOM", "C1'", "O5'", "."]


def test_semicolon_text_field_is_read():
    text = (
        "data_TEST\n"
        "_struct.title\n"
        ";A title that runs\n"
        "across two lines\n"
        ";\n"
        + _loop_header()
        + "ATOM 1 C CA . ALA A 1 1 ? 0.0 0.0 0.0 1.00 0.0 ? 1 ALA A CA 1\n"
    )
    structure = parse_mmcif(text)
    assert structure.items["_struct.title"] == "A title that runs\nacross two lines"


# --------------------------------------------------------------------------
# malformed input
# --------------------------------------------------------------------------


def _loop_header() -> str:
    return (
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_alt_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_entity_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.pdbx_PDB_ins_code\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.occupancy\n"
        "_atom_site.B_iso_or_equiv\n"
        "_atom_site.pdbx_formal_charge\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.auth_comp_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_atom_id\n"
        "_atom_site.pdbx_PDB_model_num\n"
    )


GOOD_ROW = "ATOM 1 N N . THR A 1 1 ? 17.047 14.099 3.625 1.00 13.79 ? 1 THR A N 1\n"


def test_valid_synthetic_file_parses():
    structure = parse_mmcif("data_TEST\n" + _loop_header() + GOOD_ROW)
    assert len(structure.atoms) == 1
    assert structure.atoms[0].res_name == "THR"


@pytest.mark.parametrize("text", ["", "   \n\n  ", "\t"])
def test_empty_input_raises(text):
    with pytest.raises(MmcifError, match="empty input"):
        parse_mmcif(text)


def test_input_without_an_atom_site_loop_raises():
    with pytest.raises(MmcifError, match="no _atom_site loop"):
        parse_mmcif("data_TEST\n_struct.title  'no coordinates here'\n")


def test_header_only_atom_site_loop_raises():
    with pytest.raises(MmcifError, match="contained no atoms"):
        parse_mmcif("data_TEST\n" + _loop_header())


def test_short_loop_row_raises():
    truncated = "ATOM 1 N N . THR A 1 1 ?\n"
    with pytest.raises(MmcifError, match="loop row ended early"):
        parse_mmcif("data_TEST\n" + _loop_header() + truncated)


def test_overlong_loop_row_raises():
    too_long = GOOD_ROW.rstrip("\n") + " EXTRA EXTRA2\n"
    with pytest.raises(MmcifError, match="loop row has 23 values"):
        parse_mmcif("data_TEST\n" + _loop_header() + too_long)


def test_non_numeric_coordinate_raises():
    bad = GOOD_ROW.replace("17.047", "seventeen")
    with pytest.raises(MmcifError, match="non-numeric x"):
        parse_mmcif("data_TEST\n" + _loop_header() + bad)


def test_null_coordinate_raises():
    bad = GOOD_ROW.replace("14.099", "?")
    with pytest.raises(MmcifError, match="has no y coordinate"):
        parse_mmcif("data_TEST\n" + _loop_header() + bad)


def test_unterminated_quote_raises():
    bad = GOOD_ROW.replace("THR A 1 1", "'THR A 1 1")
    with pytest.raises(MmcifError, match="unterminated quoted value"):
        parse_mmcif("data_TEST\n" + _loop_header() + bad)


def test_unterminated_semicolon_field_raises():
    text = "data_TEST\n_struct.title\n;never closed\nstill going\n"
    with pytest.raises(MmcifError, match="unterminated semicolon"):
        parse_mmcif(text)


def test_loop_with_no_headers_raises():
    with pytest.raises(MmcifError, match="no column headers"):
        parse_mmcif("data_TEST\nloop_\nnot_a_tag 1 2 3\n")


def test_atom_site_loop_without_coordinates_raises():
    header = (
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_seq_id\n"
    )
    with pytest.raises(MmcifError, match="missing _atom_site.cartn_x"):
        parse_mmcif("data_TEST\n" + header + "ATOM 1 ALA A 1\n")


def test_atom_site_loop_without_residue_identity_raises():
    header = (
        "loop_\n"
        "_atom_site.id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
    )
    with pytest.raises(MmcifError, match="lacks residue or chain identity"):
        parse_mmcif("data_TEST\n" + header + "1 0.0 0.0 0.0\n")


def test_selection_that_removes_everything_raises():
    # A file of only hydrogens, with hydrogen stripping on, leaves nothing.
    rows = "".join(
        f"ATOM {i} H H{i} . ALA A 1 1 ? {i}.0 0.0 0.0 1.00 0.0 ? 1 ALA A H{i} 1\n"
        for i in (1, 2)
    )
    structure = parse_mmcif("data_TEST\n" + _loop_header() + rows)
    assert len(structure.atoms) == 2
    with pytest.raises(MmcifError, match="selection left no atoms"):
        select(structure, drop_hydrogens=True)


# --------------------------------------------------------------------------
# fallbacks
# --------------------------------------------------------------------------


def test_element_is_inferred_when_type_symbol_is_absent():
    header = (
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
    )
    rows = (
        "ATOM 1 CA ALA A 1 0.0 0.0 0.0\n"
        "ATOM 2 ZN ZN B 1 5.0 0.0 0.0\n"
        "ATOM 3 OG SER A 2 9.0 0.0 0.0\n"
    )
    structure = parse_mmcif("data_TEST\n" + header + rows)
    # CA in a protein is the alpha carbon, so a single-letter read is right;
    # ZN is a two-letter element and must not be read as nitrogen.
    assert [a.element for a in structure.atoms] == ["C", "ZN", "O"]


def test_null_type_symbol_falls_back_to_the_atom_name():
    bad = GOOD_ROW.replace("ATOM 1 N N .", "ATOM 1 ? N .")
    structure = parse_mmcif("data_TEST\n" + _loop_header() + bad)
    assert structure.atoms[0].element == "N"
