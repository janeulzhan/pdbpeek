"""Shrake-Rupley solvent-accessible surface area, written from scratch.

The method is from A. Shrake & J. A. Rupley, "Environment and exposure to
solvent of protein atoms. Lysozyme and insulin", J. Mol. Biol. 79:351-371
(1973). Each atom is given an extended radius r_i + probe. Test points are
spread over that sphere; a point is accessible if it lies outside every other
atom's extended sphere. The atom's area is then

    A_i = 4 * pi * (r_i + probe)^2 * (accessible points / total points)

Two deliberate choices:

* Test points come from a golden-spiral (Fibonacci) lattice rather than the
  original paper's latitude bands. The spiral has no polar clustering, so the
  variance at a given point count is lower, and it is cheap to generate
  identically in another language, which is what the JS/Python parity test in
  this repo relies on.
* Neighbour search uses a scipy KD-tree. That is an exact query, not an
  approximation: it returns every atom within the cutoff, so the result is
  identical to the naive all-pairs version, just faster.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .mmcif import Atom, Structure
from .radii import DEFAULT_RADIUS, PROBE_RADIUS, radius_for, unknown_elements

DEFAULT_N_POINTS = 256

#: Neighbours tested per numpy call. Chosen by measurement, not by reasoning:
#: see the comment in `sasa_from_arrays`. Sweeping 1, 2, 4, 8, 16, 32, 64 and
#: 128 on 1CRN and 4HHB put the optimum at 8 on both.
_NEIGHBOUR_BLOCK = 8


def golden_spiral_points(n: int) -> np.ndarray:
    """`n` roughly equidistant unit vectors on a sphere, shape (n, 3).

    Point k sits at height y = 1 - (2k + 1)/n with azimuth k * pi * (3 - sqrt 5).
    The half-step offset in y keeps the points off the exact poles, which
    removes the single worst-placed point of the naive version.
    """
    if n < 1:
        raise ValueError("n_points must be at least 1")
    k = np.arange(n, dtype=np.float64)
    y = 1.0 - (2.0 * k + 1.0) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = k * (math.pi * (3.0 - math.sqrt(5.0)))
    return np.column_stack((np.cos(theta) * r, y, np.sin(theta) * r))


@dataclass
class SasaResult:
    """Per-atom SASA plus the residue-level roll-ups built from it."""

    atom_sasa: np.ndarray
    radii: np.ndarray
    probe: float
    n_points: int
    residue_keys: list[tuple[str, str, str]]
    residue_sasa: np.ndarray
    residue_reference: np.ndarray | None = None
    unknown_elements: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        """Total SASA in square angstroms."""
        return float(self.atom_sasa.sum())

    @property
    def relative_residue_sasa(self) -> np.ndarray | None:
        """Residue SASA divided by the same residue measured in isolation.

        `None` unless the reference pass was requested. Values can exceed 1.0
        slightly for a residue whose neighbours push solvent toward it, which
        is expected and not clipped.
        """
        if self.residue_reference is None:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(
                self.residue_reference > 0.0,
                self.residue_sasa / self.residue_reference,
                0.0,
            )
        return rel

    def burial(self, threshold: float = 0.25) -> list[str] | None:
        """Classify each residue "buried" or "exposed" at `threshold`.

        The 0.25 default is the conventional cutoff for relative solvent
        accessibility (Rost & Sander, J. Mol. Biol. 232:584-599, 1993). It is
        a convention, not a measurement, so it is a parameter here.
        """
        rel = self.relative_residue_sasa
        if rel is None:
            return None
        return ["buried" if v < threshold else "exposed" for v in rel]


def _extended_radii(elements, probe: float) -> np.ndarray:
    return np.array([radius_for(e) + probe for e in elements], dtype=np.float64)


def sasa_from_arrays(
    coords: np.ndarray,
    radii: np.ndarray,
    probe: float = PROBE_RADIUS,
    n_points: int = DEFAULT_N_POINTS,
) -> np.ndarray:
    """Per-atom SASA for raw coordinates and van der Waals radii.

    `coords` is (N, 3) in angstroms, `radii` is (N,) van der Waals radii
    *without* the probe added. Returns (N,) areas in square angstroms.
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    radii = np.ascontiguousarray(radii, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3), got {coords.shape}")
    if radii.shape != (coords.shape[0],):
        raise ValueError(f"radii must be ({coords.shape[0]},), got {radii.shape}")
    n_atoms = coords.shape[0]
    if n_atoms == 0:
        return np.zeros(0, dtype=np.float64)

    ext = radii + probe
    sphere = golden_spiral_points(n_points)
    const = 4.0 * math.pi / n_points

    # Two extended spheres can only overlap within max_ext * 2 of each other,
    # so one tree query radius covers every atom's neighbour list.
    tree = cKDTree(coords)
    cutoff = float(ext.max()) * 2.0
    neighbour_lists = tree.query_ball_point(coords, cutoff)

    out = np.empty(n_atoms, dtype=np.float64)
    for i in range(n_atoms):
        ri = ext[i]
        js = [j for j in neighbour_lists[i] if j != i]
        if js:
            js = np.asarray(js, dtype=np.int64)
            rj = ext[js]
            d = np.linalg.norm(coords[js] - coords[i], axis=1)
            keep = d < (ri + rj)
            js, rj = js[keep], rj[keep]

        if len(js) == 0:
            out[i] = const * ri * ri * n_points
            continue

        # Sort neighbours near to far: the closest spheres bury the most
        # points, so the early exit below fires sooner.
        order = np.argsort(np.linalg.norm(coords[js] - coords[i], axis=1))
        js, rj = js[order], rj[order]

        test = coords[i] + sphere * ri
        alive = np.ones(n_points, dtype=bool)
        n_alive = n_points

        # Test a small block of neighbours per numpy call rather than one at
        # a time. One call per (atom, neighbour) pair is about 175k calls on
        # a 4384 atom structure, and at that size numpy's per-call dispatch
        # cost dominates the arithmetic.
        #
        # The block has to stay small. Neighbours are sorted nearest first
        # precisely so the early exit below fires after two or three of them,
        # and a large block throws that away by evaluating all of its
        # neighbours before checking. Measured on 4HHB at 256 points: the
        # per-neighbour loop took 0.62 s, a block of 64 was worse at 0.54 s,
        # and a block of 8 was best at 0.31 s, two times faster than where
        # this started. 1CRN picked the same optimum.
        #
        # The arithmetic is unchanged: still a subtraction followed by a sum
        # of three squares, in the same order, so the result is bit for bit
        # what the per-neighbour version produced. That matters because
        # tests/test_parity.py asserts the per-atom areas match the
        # JavaScript implementation exactly.
        for start in range(0, len(js), _NEIGHBOUR_BLOCK):
            block = js[start : start + _NEIGHBOUR_BLOCK]
            r_block = rj[start : start + _NEIGHBOUR_BLOCK]

            live_idx = np.flatnonzero(alive)
            delta = test[live_idx, None, :] - coords[block][None, :, :]
            d2 = np.einsum("pkc,pkc->pk", delta, delta)
            inside = (d2 < (r_block * r_block)[None, :]).any(axis=1)

            if inside.any():
                alive[live_idx[inside]] = False
                n_alive -= int(inside.sum())
                if n_alive == 0:
                    break

        out[i] = const * ri * ri * n_alive

    return out


def compute_sasa(
    structure: Structure,
    probe: float = PROBE_RADIUS,
    n_points: int = DEFAULT_N_POINTS,
    reference: bool = True,
) -> SasaResult:
    """Run SASA over a selected `Structure` and roll it up per residue.

    With `reference=True` each residue is also measured on its own, with the
    rest of the structure removed. Dividing the in-context value by that gives
    a relative accessibility with no table of per-residue maxima involved: the
    denominator is produced by this same code path, on these same atoms.
    """
    atoms: list[Atom] = structure.atoms
    coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
    vdw = np.array([radius_for(a.element) for a in atoms], dtype=np.float64)
    atom_sasa = sasa_from_arrays(coords, vdw, probe=probe, n_points=n_points)

    keys: list[tuple[str, str, str]] = []
    index: dict[tuple[str, str, str], int] = {}
    members: list[list[int]] = []
    for pos, a in enumerate(atoms):
        key = a.residue_key
        if key not in index:
            index[key] = len(keys)
            keys.append(key)
            members.append([])
        members[index[key]].append(pos)

    residue_sasa = np.array([atom_sasa[m].sum() for m in members], dtype=np.float64)

    residue_reference = None
    if reference:
        residue_reference = np.array(
            [
                sasa_from_arrays(
                    coords[m], vdw[m], probe=probe, n_points=n_points
                ).sum()
                for m in members
            ],
            dtype=np.float64,
        )

    return SasaResult(
        atom_sasa=atom_sasa,
        radii=vdw,
        probe=probe,
        n_points=n_points,
        residue_keys=keys,
        residue_sasa=residue_sasa,
        residue_reference=residue_reference,
        unknown_elements=unknown_elements(a.element for a in atoms),
    )


# --------------------------------------------------------------------------
# analytic references used by the test suite
# --------------------------------------------------------------------------


def analytic_single_sphere(radius: float, probe: float = PROBE_RADIUS) -> float:
    """Exact SASA of one isolated atom: the full extended sphere."""
    return 4.0 * math.pi * (radius + probe) ** 2


def analytic_two_spheres(
    r1: float, r2: float, d: float, probe: float = PROBE_RADIUS
) -> float:
    """Exact SASA of two atoms whose centres are `d` apart.

    For extended radii R1, R2 the intersection removes one spherical cap from
    each sphere. A cap of height h on a sphere of radius R has area 2*pi*R*h,
    and the cap height follows from the radical plane of the two spheres:

        h1 = R1 - (d^2 + R1^2 - R2^2) / (2d)

    This is exact, so it is a real check on the sampling error of the
    Shrake-Rupley estimate rather than a comparison against another estimate.
    """
    R1, R2 = r1 + probe, r2 + probe
    if d >= R1 + R2:
        return 4.0 * math.pi * (R1 * R1 + R2 * R2)
    if d <= abs(R1 - R2):
        # One sphere is swallowed by the other; only the larger is exposed.
        big = max(R1, R2)
        return 4.0 * math.pi * big * big
    h1 = R1 - (d * d + R1 * R1 - R2 * R2) / (2.0 * d)
    h2 = R2 - (d * d + R2 * R2 - R1 * R1) / (2.0 * d)
    return (
        4.0 * math.pi * R1 * R1
        - 2.0 * math.pi * R1 * h1
        + 4.0 * math.pi * R2 * R2
        - 2.0 * math.pi * R2 * h2
    )


__all__ = [
    "DEFAULT_N_POINTS",
    "DEFAULT_RADIUS",
    "PROBE_RADIUS",
    "SasaResult",
    "analytic_single_sphere",
    "analytic_two_spheres",
    "compute_sasa",
    "golden_spiral_points",
    "sasa_from_arrays",
]
