"""Measure JS/Python divergence so the parity test tolerance is a real number.

Not part of the test suite. Run it when the SASA code changes on either side
to check that the tolerance asserted in tests/test_parity.py still has margin.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pdbpeek import compute_sasa, contact_map, parse_file, select  # noqa: E402

RUNNER = os.path.join(ROOT, "docs", "pdbpeek", "parity_runner.js")


def run_js(path: str, n_points: int) -> dict:
    proc = subprocess.run(
        ["node", RUNNER, path, "--n-points", str(n_points), "--per-atom"],
        capture_output=True,
        check=True,
        cwd=ROOT,
    )
    return json.loads(proc.stdout)


def main() -> None:
    print(f"{'entry':6} {'n_pts':>6} {'atoms':>6} {'py total':>12} {'js total':>12} "
          f"{'rel diff':>10} {'max atom':>10} {'res match':>10}")
    for entry in ("1CRN", "1UBQ", "2MBW", "1L2Y", "4HHB"):
        path = os.path.join(ROOT, "tests", "fixtures", f"{entry}.cif")
        for n in (64, 256, 1000):
            js = run_js(path, n)
            sel = select(parse_file(path), drop_hetatm=True)
            py = compute_sasa(sel, n_points=n)
            cmap = contact_map(sel)

            rel = abs(py.total - js["sasa"]["total"]) / py.total
            per_atom = js["per_atom_sasa"]
            max_atom = max(
                abs(a - b) for a, b in zip(py.atom_sasa.tolist(), per_atom)
            )
            res_ok = (
                len(py.residue_keys) == js["n_residues"]
                and len(sel.atoms) == js["n_atoms"]
                and cmap.n_contacts == js["contacts"]["n_contacts"]
            )
            print(
                f"{entry:6} {n:6d} {len(sel.atoms):6d} {py.total:12.6f} "
                f"{js['sasa']['total']:12.6f} {rel:10.2e} {max_atom:10.2e} "
                f"{str(res_ok):>10}"
            )


if __name__ == "__main__":
    main()
