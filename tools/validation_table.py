"""Print the validation tables that go in the README.

Run this rather than copying numbers by hand, so what the README claims is
what the code produced.

    python3 tools/validation_table.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pdbpeek.mmcif import parse_file, select  # noqa: E402
from pdbpeek.sasa import (  # noqa: E402
    analytic_single_sphere,
    analytic_two_spheres,
    compute_sasa,
    sasa_from_arrays,
)

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
RUNNER = os.path.join(ROOT, "docs", "pdbpeek", "parity_runner.js")


def freesasa_table() -> None:
    with open(os.path.join(FIXTURES, "sasa_reference.json"), encoding="utf-8") as fh:
        reference = json.load(fh)

    print("## against FreeSASA (1000 points per atom)\n")
    print("| entry | atoms | pdbpeek | FreeSASA | relative difference |")
    print("| --- | --- | --- | --- | --- |")
    worst = 0.0
    for entry in ("1CRN", "1UBQ", "2MBW", "4HHB"):
        expected = reference["entries"][entry]
        selected = select(parse_file(os.path.join(FIXTURES, f"{entry}.cif")))
        total = compute_sasa(selected, n_points=1000, reference=False).total
        diff = abs(total - expected["total_sasa"]) / expected["total_sasa"]
        worst = max(worst, diff)
        print(
            f"| {entry} | {expected['n_atoms']} | {total:.2f} | "
            f"{expected['total_sasa']:.2f} | {diff:.1e} |"
        )
    print(f"\nworst relative difference: {worst:.2e}")
    print(f"generator: {reference['generator']}\n")


def analytic_table() -> None:
    print("## against closed-form geometry\n")

    print("single isolated atom, SASA = 4*pi*(r+probe)^2")
    worst_single = 0.0
    for radius in (1.2, 1.52, 1.70, 1.80, 2.27):
        for n in (8, 64, 256, 1024, 4096):
            got = sasa_from_arrays(
                np.zeros((1, 3)), np.array([radius]), n_points=n
            )[0]
            exact = analytic_single_sphere(radius)
            worst_single = max(worst_single, abs(got - exact) / exact)
    print(f"  worst relative error across radii and point counts: {worst_single:.2e}")

    print("\ntwo overlapping atoms, SASA from the spherical cap each loses")
    print("| separation (A) | pdbpeek | exact | relative error |")
    print("| --- | --- | --- | --- |")
    worst_pair = 0.0
    for d in (1.5, 2.0, 2.5, 3.0, 3.8, 4.5, 5.0, 5.5, 6.0, 6.2):
        coords = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
        got = sasa_from_arrays(coords, np.array([1.70, 1.52]), n_points=4096).sum()
        exact = analytic_two_spheres(1.70, 1.52, d)
        err = abs(got - exact) / exact
        worst_pair = max(worst_pair, err)
        print(f"| {d} | {got:.3f} | {exact:.3f} | {err:.1e} |")
    print(f"\nworst relative error: {worst_pair:.2e}\n")


def convergence_table() -> None:
    print("## convergence with sampling density (1CRN)\n")
    selected = select(parse_file(os.path.join(FIXTURES, "1CRN.cif")))
    dense = compute_sasa(selected, n_points=20000, reference=False).total
    print("| points per atom | total SASA | relative difference from 20000 |")
    print("| --- | --- | --- |")
    for n in (8, 16, 64, 256, 1024, 4096):
        total = compute_sasa(selected, n_points=n, reference=False).total
        print(f"| {n} | {total:.2f} | {abs(total - dense) / dense:.1e} |")
    print(f"\n20000 point value used as the converged reference: {dense:.2f}\n")


def parity_table() -> None:
    print("## JavaScript against Python\n")
    print("| entry | points | atoms | per-atom max difference | total relative difference |")
    print("| --- | --- | --- | --- | --- |")
    worst_total = 0.0
    worst_atom = 0.0
    for entry in ("1CRN", "1UBQ", "2MBW", "1L2Y", "4HHB"):
        for n in (64, 256, 1000):
            proc = subprocess.run(
                [
                    "node",
                    RUNNER,
                    os.path.join(FIXTURES, f"{entry}.cif"),
                    "--n-points",
                    str(n),
                    "--per-atom",
                ],
                capture_output=True,
                cwd=ROOT,
                check=True,
            )
            js = json.loads(proc.stdout)
            selected = select(
                parse_file(os.path.join(FIXTURES, f"{entry}.cif")), drop_hetatm=True
            )
            py = compute_sasa(selected, n_points=n, reference=False)
            max_atom = max(
                abs(a - b) for a, b in zip(py.atom_sasa.tolist(), js["per_atom_sasa"])
            )
            rel = abs(py.total - js["sasa"]["total"]) / py.total
            worst_atom = max(worst_atom, max_atom)
            worst_total = max(worst_total, rel)
            if n == 256:
                print(
                    f"| {entry} | {n} | {len(selected.atoms)} | "
                    f"{max_atom:.1e} | {rel:.1e} |"
                )
    print(f"\nacross all 15 combinations tested:")
    print(f"  worst per-atom difference: {worst_atom:.2e}")
    print(f"  worst total relative difference: {worst_total:.2e}\n")


def main() -> None:
    freesasa_table()
    analytic_table()
    convergence_table()
    parity_table()


if __name__ == "__main__":
    main()
