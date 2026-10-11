"""Tests for the residue contact map."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pdbpeek.contacts import DEFAULT_CUTOFF, contact_map, contact_pairs
from pdbpeek.mmcif import parse_file, parse_mmcif, select

from .conftest import cif_path
from .test_mmcif import _loop_header


def test_map_shape_and_residue_count(crambin):
    cmap = contact_map(crambin)
    assert cmap.n_residues == 46, "crambin is a 46 residue peptide"
    assert cmap.contacts.shape == (46, 46)
    assert cmap.distances.shape == (46, 46)
    assert cmap.cutoff == DEFAULT_CUTOFF


def test_contact_matrix_is_symmetric(crambin):
    cmap = contact_map(crambin)
    assert np.array_equal(cmap.contacts, cmap.contacts.T)
    np.testing.assert_allclose(cmap.distances, cmap.distances.T, atol=0.0)


def test_diagonal_is_zero(crambin):
    cmap = contact_map(crambin)
    assert not cmap.contacts.diagonal().any(), "a residue must not contact itself"
    assert np.all(cmap.distances.diagonal() == 0.0)


def test_symmetry_holds_on_a_multi_chain_structure():
    selected = select(parse_file(cif_path("4HHB")), drop_hetatm=True)
    cmap = contact_map(selected)
    assert np.array_equal(cmap.contacts, cmap.contacts.T)
    assert not cmap.contacts.diagonal().any()
    assert cmap.n_residues == sum(
        1 for _, atoms in selected.residues() if not all(a.is_hetatm for a in atoms)
    )


def test_distances_agree_with_direct_computation(crambin):
    cmap = contact_map(crambin)
    # Recompute one entry the long way, straight from the CB (or CA) atoms.
    residues = dict(crambin.residues())
    def rep(key):
        atoms = residues[key]
        by_name = {a.atom_name.upper(): a for a in atoms}
        return by_name.get("CB") or by_name["CA"]

    a, b = cmap.residue_keys[3], cmap.residue_keys[9]
    pa, pb = rep(a), rep(b)
    expected = math.dist((pa.x, pa.y, pa.z), (pb.x, pb.y, pb.z))
    assert math.isclose(cmap.distances[3, 9], expected, rel_tol=1e-12)


def test_consecutive_residues_are_always_in_contact(crambin):
    # Sequential CB atoms in a polypeptide sit roughly 5 to 6 A apart, well
    # inside the 8 A cutoff, so the first off-diagonal must be all contacts.
    cmap = contact_map(crambin)
    neighbours = [cmap.contacts[i, i + 1] for i in range(cmap.n_residues - 1)]
    assert all(neighbours), "a break in the chain would mean a parsing error"


def test_known_crambin_contact_is_present(crambin):
    # Crambin's two cystine bridges are CYS3-CYS40 and CYS4-CYS32, and a
    # disulfide puts the two CB atoms far closer than 8 A. Finding both of
    # them is a check that residue indexing lines up with the file.
    cmap = contact_map(crambin)
    index = {int(key[1]): i for i, key in enumerate(cmap.residue_keys)}

    for left, right in ((3, 40), (4, 32)):
        i, j = index[left], index[right]
        assert cmap.residue_keys[i][2] == "CYS"
        assert cmap.residue_keys[j][2] == "CYS"
        assert cmap.contacts[i, j], f"CYS{left}-CYS{right} should be in contact"
        assert cmap.distances[i, j] < 5.0, cmap.distances[i, j]


def test_distant_residues_are_not_in_contact(crambin):
    cmap = contact_map(crambin)
    index = {int(key[1]): i for i, key in enumerate(cmap.residue_keys)}
    i, j = index[1], index[25]
    assert not cmap.contacts[i, j]
    assert cmap.distances[i, j] > DEFAULT_CUTOFF


def test_contact_count_matches_the_distance_matrix(crambin):
    cmap = contact_map(crambin)
    expected = int((cmap.distances[np.triu_indices(cmap.n_residues, k=1)] < cmap.cutoff).sum())
    assert cmap.n_contacts == expected


def test_cutoff_widens_the_map_monotonically(crambin):
    counts = [contact_map(crambin, cutoff=c).n_contacts for c in (4.0, 6.0, 8.0, 12.0)]
    assert counts == sorted(counts), counts
    assert counts[0] < counts[-1]


def test_tiny_cutoff_gives_no_contacts(crambin):
    cmap = contact_map(crambin, cutoff=0.5)
    assert cmap.n_contacts == 0
    assert not cmap.contacts.any()


def test_huge_cutoff_connects_everything(crambin):
    cmap = contact_map(crambin, cutoff=1000.0)
    n = cmap.n_residues
    assert cmap.n_contacts == n * (n - 1) // 2


@pytest.mark.parametrize("cutoff", [0.0, -1.0])
def test_non_positive_cutoff_raises(crambin, cutoff):
    with pytest.raises(ValueError, match="cutoff must be positive"):
        contact_map(crambin, cutoff=cutoff)


# --------------------------------------------------------------------------
# glycine
# --------------------------------------------------------------------------


def test_glycine_uses_ca_because_it_has_no_cb(crambin):
    glycines = [
        (key, atoms)
        for key, atoms in crambin.residues()
        if key[2] == "GLY"
    ]
    assert glycines, "crambin contains glycine"
    for key, atoms in glycines:
        names = {a.atom_name.upper() for a in atoms}
        assert "CB" not in names, "glycine has no beta carbon"
        assert "CA" in names

    # Every glycine still appears in the map, at its CA position.
    cmap = contact_map(crambin)
    for key, atoms in glycines:
        assert key in cmap.residue_keys
        i = cmap.residue_keys.index(key)
        ca = next(a for a in atoms if a.atom_name.upper() == "CA")
        j = (i + 1) % cmap.n_residues
        other_atoms = dict(crambin.residues())[cmap.residue_keys[j]]
        by_name = {a.atom_name.upper(): a for a in other_atoms}
        other = by_name.get("CB") or by_name["CA"]
        expected = math.dist((ca.x, ca.y, ca.z), (other.x, other.y, other.z))
        assert math.isclose(cmap.distances[i, j], expected, rel_tol=1e-12)


def test_cb_is_preferred_over_ca():
    # A synthetic residue with both atoms: the CB must be the one used.
    rows = (
        "ATOM 1 C CA . ALA A 1 1 ? 0.0 0.0 0.0 1.00 0.0 ? 1 ALA A CA 1\n"
        "ATOM 2 C CB . ALA A 1 1 ? 0.0 0.0 1.5 1.00 0.0 ? 1 ALA A CB 1\n"
        "ATOM 3 C CA . ALA A 1 2 ? 7.5 0.0 0.0 1.00 0.0 ? 2 ALA A CA 1\n"
        "ATOM 4 C CB . ALA A 1 2 ? 7.5 0.0 1.5 1.00 0.0 ? 2 ALA A CB 1\n"
    )
    structure = parse_mmcif("data_TEST\n" + _loop_header() + rows)
    cmap = contact_map(structure)
    # CB-CB separation is 7.5, CA-CA is also 7.5 here, so distinguish by
    # offsetting: distance must be the CB pair, exactly 7.5.
    assert math.isclose(cmap.distances[0, 1], 7.5, rel_tol=1e-12)


def test_residue_without_ca_or_cb_is_skipped_not_invented():
    rows = (
        "ATOM 1 C CA . ALA A 1 1 ? 0.0 0.0 0.0 1.00 0.0 ? 1 ALA A CA 1\n"
        "HETATM 2 O O . HOH A 1 2 ? 3.0 0.0 0.0 1.00 0.0 ? 2 HOH A O 1\n"
        "ATOM 3 N N . ALA A 1 3 ? 6.0 0.0 0.0 1.00 0.0 ? 3 ALA A N 1\n"
    )
    structure = parse_mmcif("data_TEST\n" + _loop_header() + rows)
    cmap = contact_map(structure, include_hetatm=True)
    assert cmap.n_residues == 1, "only the residue with a CA should be mapped"
    assert len(cmap.skipped) == 2
    assert ("A", "2", "HOH") in cmap.skipped
    assert ("A", "3", "ALA") in cmap.skipped


def test_waters_never_enter_the_map():
    # A water is a lone oxygen, so it has neither CB nor CA and the
    # representative-atom rule drops it whatever include_hetatm says. 1UBQ
    # ships 58 waters on top of its 76 residue chain; the map must be 76.
    selected = select(parse_file(cif_path("1UBQ")))
    assert sum(1 for key, _ in selected.residues() if key[2] == "HOH") == 58

    for include in (True, False):
        cmap = contact_map(selected, include_hetatm=include)
        assert cmap.n_residues == 76, include
        assert all(key[2] != "HOH" for key in cmap.residue_keys)
        assert sum(1 for key in cmap.skipped if key[2] == "HOH") == 58


def test_include_hetatm_admits_a_hetero_residue_that_has_a_backbone():
    # Modified residues such as selenomethionine are deposited as HETATM but
    # do carry CA and CB, so the flag is what decides whether they count.
    rows = (
        "ATOM 1 C CA . ALA A 1 1 ? 0.0 0.0 0.0 1.00 0.0 ? 1 ALA A CA 1\n"
        "ATOM 2 C CB . ALA A 1 1 ? 0.0 0.0 1.5 1.00 0.0 ? 1 ALA A CB 1\n"
        "HETATM 3 C CA . MSE A 1 2 ? 5.0 0.0 0.0 1.00 0.0 ? 2 MSE A CA 1\n"
        "HETATM 4 C CB . MSE A 1 2 ? 5.0 0.0 1.5 1.00 0.0 ? 2 MSE A CB 1\n"
    )
    structure = parse_mmcif("data_TEST\n" + _loop_header() + rows)

    excluded = contact_map(structure, include_hetatm=False)
    assert excluded.n_residues == 1
    assert ("A", "2", "MSE") in excluded.skipped

    included = contact_map(structure, include_hetatm=True)
    assert included.n_residues == 2
    assert included.n_contacts == 1
    assert math.isclose(included.distances[0, 1], 5.0, rel_tol=1e-12)


def test_empty_structure_gives_an_empty_map():
    rows = "HETATM 1 O O . HOH A 1 1 ? 0.0 0.0 0.0 1.00 0.0 ? 1 HOH A O 1\n"
    structure = parse_mmcif("data_TEST\n" + _loop_header() + rows)
    cmap = contact_map(structure, include_hetatm=False)
    assert cmap.n_residues == 0
    assert cmap.n_contacts == 0
    assert cmap.contacts.shape == (0, 0)
    assert cmap.contact_order() == 0.0


# --------------------------------------------------------------------------
# derived quantities
# --------------------------------------------------------------------------


def test_neighbour_counts_match_the_matrix(crambin):
    cmap = contact_map(crambin)
    counts = cmap.neighbour_counts()
    assert counts.shape == (cmap.n_residues,)
    assert int(counts.sum()) == 2 * cmap.n_contacts
    assert counts.min() >= 1, "no residue in a folded protein is fully isolated"


def test_contact_order_is_positive_and_bounded(crambin):
    cmap = contact_map(crambin)
    order = cmap.contact_order()
    assert 1.0 < order < cmap.n_residues


def test_contact_order_distinguishes_two_real_folds():
    # Trp-cage is 20 residues held together by local contacts; crambin is a
    # 46 residue fold with a beta hairpin pairing distant strands. The second
    # must therefore have the longer mean sequence separation.
    trp_cage = select(parse_file(cif_path("1L2Y")), drop_hetatm=True)
    crambin = select(parse_file(cif_path("1CRN")), drop_hetatm=True)
    assert contact_map(trp_cage).contact_order() < contact_map(crambin).contact_order()


def test_contact_order_ignores_cross_chain_pairs():
    selected = select(parse_file(cif_path("4HHB")), drop_hetatm=True)
    cmap = contact_map(selected)
    # Index distance across a chain boundary is large and meaningless; if
    # those pairs leaked in, the mean would be inflated past the chain length.
    longest_chain = max(len(v) for v in selected.by_chain().values())
    assert 0.0 < cmap.contact_order() < longest_chain


# --------------------------------------------------------------------------
# dense map against the KD-tree form
# --------------------------------------------------------------------------


def test_sparse_pairs_agree_with_the_dense_map(crambin):
    cmap = contact_map(crambin)
    pairs = contact_pairs(crambin)
    assert len(pairs) == cmap.n_contacts
    for i, j, d in pairs:
        assert cmap.contacts[i, j], (i, j)
        assert math.isclose(d, cmap.distances[i, j], rel_tol=1e-12)


def test_sparse_pairs_agree_on_a_large_structure():
    selected = select(parse_file(cif_path("4HHB")), drop_hetatm=True)
    cmap = contact_map(selected)
    pairs = contact_pairs(selected)
    assert len(pairs) == cmap.n_contacts
    assert all(i < j for i, j, _ in pairs)
    assert pairs == sorted(pairs)


def test_sparse_pairs_respect_the_cutoff(crambin):
    for cutoff in (5.0, 8.0, 11.0):
        pairs = contact_pairs(crambin, cutoff=cutoff)
        assert len(pairs) == contact_map(crambin, cutoff=cutoff).n_contacts
        assert all(d < cutoff for _, _, d in pairs)


def test_sparse_pairs_on_an_empty_selection():
    rows = "HETATM 1 O O . HOH A 1 1 ? 0.0 0.0 0.0 1.00 0.0 ? 1 HOH A O 1\n"
    structure = parse_mmcif("data_TEST\n" + _loop_header() + rows)
    assert contact_pairs(structure, include_hetatm=False) == []
