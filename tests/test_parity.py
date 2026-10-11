"""Cross-language parity: the browser JavaScript against the Python package.

The demo page computes SASA and contacts client-side, so there are two
independent implementations of the same algorithms in this repository and
nothing stops them drifting apart except a test. This is that test.

It shells out to `node docs/pdbpeek/parity_runner.js`, which imports the very
modules `docs/index.html` loads. No test-only JavaScript is involved, so a
pass means the code shipped to the browser is the code that was checked.

What the numbers actually are, measured on this machine across five fixtures
at 64, 256 and 1000 points per atom:

* per-atom areas are bit-identical, every atom, every structure
* whole-structure totals differ by at most 3.3e-15 relative, which is
  summation order alone: numpy adds pairwise, the JavaScript loop adds left
  to right
* atom, residue, chain and contact counts match exactly

Bit-identical per-atom areas are possible because both sides build the
golden-spiral point set from the same closed form and both use an exact
neighbour search, a scipy KD-tree in Python and a uniform cell grid in
JavaScript. Neither is an approximation the other does not also make. Run
`python3 tools/measure_parity.py` to re-measure after changing either side.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess

import numpy as np
import pytest

from pdbpeek.contacts import contact_map
from pdbpeek.mmcif import parse_file, select
from pdbpeek.sasa import compute_sasa

from .conftest import ROOT, cif_path

RUNNER = os.path.join(ROOT, "docs", "pdbpeek", "parity_runner.js")

#: Totals differ only by floating-point summation order. Measured worst case
#: was 3.3e-15 relative, so this leaves roughly three orders of magnitude.
TOTAL_RTOL = 1e-12

#: Per-atom areas came out exactly equal on every fixture tested. Asserting a
#: small absolute tolerance rather than equality leaves room for a different
#: platform's libm to round a sin or cos differently, which could flip a
#: borderline test point and move one atom's area by one point's worth.
PER_ATOM_ATOL = 1e-6

node = shutil.which("node")
requires_node = pytest.mark.skipif(
    node is None, reason="node is not installed; the parity test needs it"
)


def run_js(entry: str, n_points: int, per_atom: bool = False, extra=()) -> dict:
    """Run the browser modules under node and return the parsed JSON."""
    cmd = [node, RUNNER, cif_path(entry), "--n-points", str(n_points)]
    if per_atom:
        cmd.append("--per-atom")
    cmd.extend(extra)
    proc = subprocess.run(cmd, capture_output=True, cwd=ROOT, check=False)
    if proc.returncode != 0:
        raise AssertionError(
            f"parity runner failed on {entry}: "
            f"{proc.stderr.decode('utf-8', 'replace')}"
        )
    return json.loads(proc.stdout)


def run_py(entry: str, n_points: int):
    selected = select(parse_file(cif_path(entry)), drop_hetatm=True)
    return selected, compute_sasa(selected, n_points=n_points, reference=False)


@requires_node
def test_node_is_available_and_the_runner_works():
    payload = run_js("1CRN", 64)
    assert payload["entry_id"].upper() == "1CRN"
    assert payload["sasa"]["total"] > 0


@requires_node
def test_runner_rejects_bad_arguments():
    proc = subprocess.run(
        [node, RUNNER, cif_path("1CRN"), "--n-points", "0"],
        capture_output=True,
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode != 0
    assert b"positive integer" in proc.stderr


# --------------------------------------------------------------------------
# the parity assertions
# --------------------------------------------------------------------------

PARITY_CASES = [
    ("1CRN", 64),
    ("1CRN", 256),
    ("1CRN", 1000),
    ("1UBQ", 256),
    ("2MBW", 256),
    ("1L2Y", 256),
    ("4HHB", 128),
]


@requires_node
@pytest.mark.parametrize("entry,n_points", PARITY_CASES)
def test_total_sasa_agrees_across_languages(entry, n_points):
    js = run_js(entry, n_points)
    _, py = run_py(entry, n_points)
    assert math.isclose(py.total, js["sasa"]["total"], rel_tol=TOTAL_RTOL), (
        entry,
        n_points,
        py.total,
        js["sasa"]["total"],
    )


@requires_node
@pytest.mark.parametrize("entry,n_points", PARITY_CASES)
def test_per_atom_sasa_agrees_across_languages(entry, n_points):
    js = run_js(entry, n_points, per_atom=True)
    _, py = run_py(entry, n_points)
    js_atoms = np.asarray(js["per_atom_sasa"], dtype=np.float64)
    assert js_atoms.shape == py.atom_sasa.shape
    np.testing.assert_allclose(js_atoms, py.atom_sasa, rtol=0.0, atol=PER_ATOM_ATOL)


@requires_node
@pytest.mark.parametrize("entry,n_points", PARITY_CASES)
def test_counts_agree_exactly(entry, n_points):
    js = run_js(entry, n_points)
    selected, py = run_py(entry, n_points)
    cmap = contact_map(selected)

    assert js["n_atoms"] == len(selected.atoms)
    assert js["n_residues"] == len(py.residue_keys)
    assert js["n_chains"] == len(selected.chains)
    assert js["chains"] == selected.chains
    assert js["contacts"]["n_residues"] == cmap.n_residues
    assert js["contacts"]["n_contacts"] == cmap.n_contacts


@requires_node
def test_per_atom_areas_are_exactly_equal_on_the_small_fixture():
    # The strongest form of the claim, asserted without any tolerance at all
    # on the structure where it is cheapest to check.
    js = run_js("1CRN", 256, per_atom=True)
    _, py = run_py("1CRN", 256)
    assert js["per_atom_sasa"] == py.atom_sasa.tolist()


@requires_node
def test_residue_level_results_agree():
    js = run_js("1CRN", 256)
    selected, py = run_py("1CRN", 256)
    py_rel = py.relative_residue_sasa
    assert py_rel is None  # reference pass is off in run_py

    full = compute_sasa(selected, n_points=256, reference=True)
    rel = full.relative_residue_sasa
    burial = full.burial()

    assert len(js["residues"]) == len(full.residue_keys)
    for i, row in enumerate(js["residues"]):
        chain, seq, name = full.residue_keys[i]
        assert (row["chain"], row["name"]) == (chain, name)
        assert row["seq"] == seq
        assert math.isclose(row["sasa"], float(full.residue_sasa[i]), rel_tol=1e-12)
        assert math.isclose(row["relative_sasa"], float(rel[i]), rel_tol=1e-12)
        assert row["burial"] == burial[i]


@requires_node
def test_contact_order_agrees():
    js = run_js("1CRN", 64)
    selected, _ = run_py("1CRN", 64)
    expected = contact_map(selected).contact_order()
    assert math.isclose(js["contacts"]["mean_contact_order"], expected, rel_tol=1e-12)


@requires_node
def test_model_handling_agrees_on_a_multi_model_file():
    # Both sides must default to the first of 1L2Y's 38 models, and both must
    # honour an explicit model number.
    js_default = run_js("1L2Y", 128)
    assert js_default["models_in_file"] == list(range(1, 39))

    for model in (1, 2, 38):
        js = run_js("1L2Y", 128, extra=("--model", str(model)))
        selected = select(
            parse_file(cif_path("1L2Y")), model=model, drop_hetatm=True
        )
        py = compute_sasa(selected, n_points=128, reference=False)
        assert math.isclose(py.total, js["sasa"]["total"], rel_tol=TOTAL_RTOL), model

    assert js_default["sasa"]["total"] == pytest.approx(
        run_js("1L2Y", 128, extra=("--model", "1"))["sasa"]["total"]
    )


@requires_node
def test_altloc_resolution_agrees():
    # 2MBW carries A and B conformers, so the two implementations have to make
    # the same choice or their atom counts diverge.
    js = run_js("2MBW", 128)
    selected, py = run_py("2MBW", 128)
    assert js["n_atoms"] == len(selected.atoms)
    assert math.isclose(py.total, js["sasa"]["total"], rel_tol=TOTAL_RTOL)


@requires_node
def test_water_handling_agrees():
    js = run_js("1UBQ", 128, extra=("--keep-waters",))
    selected = select(parse_file(cif_path("1UBQ")), drop_hetatm=False)
    py = compute_sasa(selected, n_points=128, reference=False)
    assert js["n_atoms"] == len(selected.atoms) == 660
    assert math.isclose(py.total, js["sasa"]["total"], rel_tol=TOTAL_RTOL)


@requires_node
def test_unknown_element_reporting_agrees():
    js = run_js("1CRN", 64)
    _, py = run_py("1CRN", 64)
    assert js["sasa"]["unknown_elements"] == py.unknown_elements == []


@requires_node
def test_probe_radius_and_point_count_are_reported_identically():
    js = run_js("1CRN", 333)
    _, py = run_py("1CRN", 333)
    assert js["sasa"]["probe"] == py.probe == 1.4
    assert js["sasa"]["n_points"] == py.n_points == 333


@requires_node
def test_the_two_implementations_share_the_radius_table():
    # A divergence here would be invisible in the totals for structures made
    # only of C, N, O and S, and then surface as a silent error on a metal.
    proc = subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            "import {VDW_RADII, DEFAULT_RADIUS, PROBE_RADIUS} from "
            "'./docs/pdbpeek/sasa.js';"
            "process.stdout.write(JSON.stringify("
            "{radii: VDW_RADII, def: DEFAULT_RADIUS, probe: PROBE_RADIUS}));",
        ],
        capture_output=True,
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    payload = json.loads(proc.stdout)

    from pdbpeek.radii import DEFAULT_RADIUS, PROBE_RADIUS, VDW_RADII

    assert payload["radii"] == VDW_RADII
    assert payload["def"] == DEFAULT_RADIUS
    assert payload["probe"] == PROBE_RADIUS


@requires_node
def test_sphere_point_sets_are_identical():
    proc = subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            "import {goldenSpiralPoints} from './docs/pdbpeek/sasa.js';"
            "process.stdout.write(JSON.stringify("
            "Array.from(goldenSpiralPoints(257))));",
        ],
        capture_output=True,
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    js_points = np.asarray(json.loads(proc.stdout), dtype=np.float64).reshape(-1, 3)

    from pdbpeek.sasa import golden_spiral_points

    py_points = golden_spiral_points(257)
    assert js_points.shape == py_points.shape
    # Allow a 1 ulp difference in sin and cos between the two libm
    # implementations, and report the worst case if it ever grows.
    np.testing.assert_allclose(js_points, py_points, rtol=0.0, atol=1e-15)


@requires_node
def test_summary_template_agrees_across_languages(entry_1crn):
    # The demo page renders the summary in JavaScript, so the template has to
    # produce the same bytes there as the Python one does.
    proc = subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            "import {summarizeEntry} from './docs/pdbpeek/summary.js';"
            "import {readFileSync} from 'node:fs';"
            "const e = JSON.parse(readFileSync("
            "'tests/fixtures/entry_1CRN.json','utf8'));"
            "process.stdout.write(summarizeEntry(e));",
        ],
        capture_output=True,
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")

    from pdbpeek.summary import summarize_entry

    assert proc.stdout.decode("utf-8") == summarize_entry(entry_1crn)


@requires_node
def test_measured_sentence_agrees_across_languages(entry_1crn):
    measured_py = {
        "n_atoms": 327,
        "n_residues": 46,
        "n_chains": 1,
        "total_sasa": 2962.7214,
        "buried_fraction": 0.4782,
    }
    proc = subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            "import {summarizeEntry} from './docs/pdbpeek/summary.js';"
            "import {readFileSync} from 'node:fs';"
            "const e = JSON.parse(readFileSync("
            "'tests/fixtures/entry_1CRN.json','utf8'));"
            "process.stdout.write(summarizeEntry(e, {nAtoms: 327, nResidues: 46,"
            " nChains: 1, totalSasa: 2962.7214, buriedFraction: 0.4782}));",
        ],
        capture_output=True,
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")

    from pdbpeek.summary import summarize_entry

    assert proc.stdout.decode("utf-8") == summarize_entry(
        entry_1crn, measured=measured_py
    )
