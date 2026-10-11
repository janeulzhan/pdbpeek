"""Residue contact map from beta-carbon positions.

The convention is the standard one in structure prediction evaluation: two
residues are in contact when their CB atoms are within a cutoff, commonly
8 A, and glycine substitutes CA because it has no CB. See for example the
contact definition used in the CASP assessments.

The map is built on real atoms only. A residue with neither CB nor CA, which
happens for waters, ions and some ligands, is reported in `skipped` rather
than given a fabricated position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .mmcif import Structure

DEFAULT_CUTOFF = 8.0


@dataclass
class ContactMap:
    """A symmetric boolean contact matrix and the residues it indexes."""

    residue_keys: list[tuple[str, str, str]]
    distances: np.ndarray
    contacts: np.ndarray
    cutoff: float
    skipped: list[tuple[str, str, str]]

    @property
    def n_residues(self) -> int:
        return len(self.residue_keys)

    @property
    def n_contacts(self) -> int:
        """Contacting pairs, counted once each and excluding self-contacts."""
        return int(np.triu(self.contacts, k=1).sum())

    def contact_order(self) -> float:
        """Mean sequence separation of contacting pairs within one chain.

        A low value means the fold is dominated by local contacts, a helix
        bundle for instance; a high value means long-range pairing such as a
        beta sheet formed from distant strands. Pairs on different chains are
        excluded because sequence separation is undefined across chains.
        """
        seps: list[int] = []
        rows, cols = np.nonzero(np.triu(self.contacts, k=1))
        for a, b in zip(rows, cols):
            if self.residue_keys[a][0] == self.residue_keys[b][0]:
                seps.append(abs(int(b) - int(a)))
        return float(np.mean(seps)) if seps else 0.0

    def neighbour_counts(self) -> np.ndarray:
        """Number of contacts per residue, excluding the diagonal."""
        return self.contacts.sum(axis=1).astype(np.int64)


def _representative(atoms) -> object | None:
    """CB if present, else CA. Glycine has no CB so it lands on CA."""
    by_name = {a.atom_name.upper(): a for a in atoms}
    return by_name.get("CB") or by_name.get("CA")


def contact_map(
    structure: Structure,
    cutoff: float = DEFAULT_CUTOFF,
    include_hetatm: bool = False,
) -> ContactMap:
    """Build the CB-CB contact map for a selected `Structure`."""
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    keys: list[tuple[str, str, str]] = []
    points: list[tuple[float, float, float]] = []
    skipped: list[tuple[str, str, str]] = []

    for key, atoms in structure.residues():
        if not include_hetatm and all(a.is_hetatm for a in atoms):
            skipped.append(key)
            continue
        rep = _representative(atoms)
        if rep is None:
            skipped.append(key)
            continue
        keys.append(key)
        points.append((rep.x, rep.y, rep.z))

    n = len(keys)
    if n == 0:
        return ContactMap(
            residue_keys=[],
            distances=np.zeros((0, 0)),
            contacts=np.zeros((0, 0), dtype=bool),
            cutoff=cutoff,
            skipped=skipped,
        )

    coords = np.array(points, dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(distances, 0.0)

    contacts = distances < cutoff
    np.fill_diagonal(contacts, False)

    return ContactMap(
        residue_keys=keys,
        distances=distances,
        contacts=contacts,
        cutoff=cutoff,
        skipped=skipped,
    )


def contact_pairs(
    structure: Structure,
    cutoff: float = DEFAULT_CUTOFF,
    include_hetatm: bool = False,
) -> list[tuple[int, int, float]]:
    """Contacting (i, j, distance) with i < j, found with a KD-tree.

    Equivalent to the dense map above but without the N^2 matrix, which is
    what you want once a structure runs to thousands of residues.
    """
    keys: list[tuple[str, str, str]] = []
    points: list[tuple[float, float, float]] = []
    for key, atoms in structure.residues():
        if not include_hetatm and all(a.is_hetatm for a in atoms):
            continue
        rep = _representative(atoms)
        if rep is None:
            continue
        keys.append(key)
        points.append((rep.x, rep.y, rep.z))
    if not points:
        return []
    coords = np.array(points, dtype=np.float64)
    tree = cKDTree(coords)
    out: list[tuple[int, int, float]] = []
    for i, j in tree.query_pairs(cutoff):
        a, b = (i, j) if i < j else (j, i)
        out.append((a, b, float(np.linalg.norm(coords[a] - coords[b]))))
    out.sort()
    return out
