"""Regenerate tests/fixtures/sasa_reference.json.

The reference numbers the test suite checks against are produced by FreeSASA
(Mitternacht, "FreeSASA: An open source C library for solvent accessible
surface area calculations", F1000Research 5:189, 2016), an independent
peer-reviewed implementation of the same algorithm.

The comparison is made fair by construction: `freesasa.calcCoord` takes raw
coordinates and radii, so FreeSASA never sees the mmCIF file and never applies
its own radius classifier. Both implementations therefore run Shrake-Rupley on
byte-identical geometry with byte-identical radii, and any disagreement is
purely sampling: FreeSASA uses latitude-band test points, pdbpeek uses a
golden spiral.

FreeSASA is a development-time dependency only. The generated JSON is
committed, so the test suite runs offline with nothing but numpy and scipy.

Usage:
    python3 -m pip install --user freesasa
    python3 tests/gen_reference.py
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import freesasa  # noqa: E402

from pdbpeek.mmcif import parse_file, select  # noqa: E402
from pdbpeek.radii import PROBE_RADIUS, radius_for  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
ENTRIES = ["1CRN", "1UBQ", "2MBW", "4HHB"]
N_POINTS = 1000


def main() -> None:
    freesasa.setVerbosity(freesasa.silent)
    params = freesasa.Parameters(
        {
            "algorithm": freesasa.ShrakeRupley,
            "probe-radius": PROBE_RADIUS,
            "n-points": N_POINTS,
            "n-threads": 1,
        }
    )

    out = {
        "generator": "freesasa " + importlib.metadata.version("freesasa"),
        "algorithm": "Shrake-Rupley",
        "probe_radius": PROBE_RADIUS,
        "freesasa_n_points": N_POINTS,
        "radii": "pdbpeek.radii (Bondi 1964), passed to freesasa.calcCoord",
        "selection": "first model, highest-occupancy altloc, hydrogens removed",
        "python": platform.python_version(),
        "entries": {},
    }

    for entry in ENTRIES:
        sel = select(parse_file(os.path.join(FIXTURES, f"{entry}.cif")))
        coords: list[float] = []
        radii: list[float] = []
        for a in sel.atoms:
            coords.extend((a.x, a.y, a.z))
            radii.append(radius_for(a.element))
        result = freesasa.calcCoord(coords, radii, params)
        out["entries"][entry] = {
            "n_atoms": len(sel.atoms),
            "total_sasa": round(result.totalArea(), 4),
        }
        print(f"{entry}: {len(sel.atoms)} atoms, total {result.totalArea():.2f} A^2")

    path = os.path.join(FIXTURES, "sasa_reference.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
