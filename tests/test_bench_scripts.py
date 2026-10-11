"""Tests for the benchmark generators behind the README's numbers.

The README quotes timings and speedups. Those come from scripts in `bench/`,
and a script that has quietly rotted produces a number nobody can reproduce,
which is worse than no number. These tests keep the generators runnable and,
more importantly, keep the comparison honest.

The one that matters is `test_baseline_matches_the_shipping_implementation`.
`bench_block_size.sasa_per_neighbour_baseline` is a copy of the pre-blocking
inner loop, kept only so the "before" timing stays reproducible. Nothing in
the package imports it, so without a test it could drift into a strawman that
makes the optimisation look better than it is.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.bench_block_size import (  # noqa: E402
    BLOCK_SIZES,
    neighbour_pair_count,
    sasa_per_neighbour_baseline,
)
from pdbpeek.mmcif import parse_file, select  # noqa: E402
from pdbpeek.radii import PROBE_RADIUS, radius_for  # noqa: E402
from pdbpeek.sasa import sasa_from_arrays  # noqa: E402

from .conftest import ROOT, cif_path  # noqa: E402


def arrays(entry: str):
    selected = select(parse_file(cif_path(entry)), drop_hetatm=True)
    coords = np.array([[a.x, a.y, a.z] for a in selected.atoms], dtype=np.float64)
    radii = np.array([radius_for(a.element) for a in selected.atoms], dtype=np.float64)
    return coords, radii


# --------------------------------------------------------------------------
# the baseline must be a fair comparison, not a strawman
# --------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ["1CRN", "1UBQ"])
@pytest.mark.parametrize("n_points", [64, 256])
def test_baseline_matches_the_shipping_implementation(entry, n_points):
    # Bit for bit, not merely close. Both do a subtraction and a sum of three
    # squares in the same order, so any difference would mean the baseline is
    # computing something else and the speedup is measured against fiction.
    coords, radii = arrays(entry)
    baseline = sasa_per_neighbour_baseline(coords, radii, n_points=n_points)
    shipping = sasa_from_arrays(coords, radii, n_points=n_points)
    assert baseline.tolist() == shipping.tolist()


def test_block_size_does_not_change_the_answer():
    # The whole optimisation rests on this. If blocking changed the result,
    # the timing comparison would be meaningless.
    import pdbpeek.sasa as sasa_module

    coords, radii = arrays("1CRN")
    original = sasa_module._NEIGHBOUR_BLOCK
    reference = None
    try:
        for block in BLOCK_SIZES:
            sasa_module._NEIGHBOUR_BLOCK = block
            areas = sasa_module.sasa_from_arrays(coords, radii, n_points=128)
            if reference is None:
                reference = areas.tolist()
            else:
                assert areas.tolist() == reference, f"block {block} changed the result"
    finally:
        sasa_module._NEIGHBOUR_BLOCK = original


def test_shipping_block_size_is_the_measured_optimum():
    # Guards against someone editing the constant without re-running the
    # sweep. 8 is what bench/block_size.csv found on both test structures.
    import pdbpeek.sasa as sasa_module

    assert sasa_module._NEIGHBOUR_BLOCK == 8
    assert 8 in BLOCK_SIZES


# --------------------------------------------------------------------------
# the pair count quoted in the README
# --------------------------------------------------------------------------


def test_neighbour_pair_count_is_the_number_the_readme_quotes():
    # The README says the per-neighbour loop issues about 178k numpy calls on
    # 4HHB. This is where that comes from.
    coords, radii = arrays("4HHB")
    pairs = neighbour_pair_count(coords, radii, PROBE_RADIUS)
    assert 170_000 < pairs < 185_000, pairs


def test_pair_count_grows_with_structure_size():
    counts = [
        neighbour_pair_count(*arrays(entry), PROBE_RADIUS)
        for entry in ("1CRN", "1UBQ", "4HHB")
    ]
    assert counts == sorted(counts), counts


# --------------------------------------------------------------------------
# the generators still run
# --------------------------------------------------------------------------


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, script, *args],
        capture_output=True,
        cwd=ROOT,
        check=False,
    )


@pytest.mark.slow
def test_block_size_sweep_runs():
    proc = run("bench/bench_block_size.py", "--runs", "1", "--warmup", "0")
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    out = proc.stdout.decode("utf-8")
    assert "per-neighbour loop, the original" in out
    assert "identical" in out


@pytest.mark.slow
def test_cross_language_bench_runs():
    proc = run("bench/bench_cross_language.py", "--runs", "1", "--warmup", "0")
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    out = proc.stdout.decode("utf-8")
    assert "match" in out
    assert "DIFFER" not in out


@pytest.mark.slow
def test_main_bench_runs():
    proc = run("bench/bench_sasa.py", "--runs", "1", "--warmup", "0", "--entries", "1CRN")
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert b"1CRN" in proc.stdout


# --------------------------------------------------------------------------
# the committed CSVs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["results.csv", "block_size.csv", "cross_language.csv"]
)
def test_committed_csv_exists_and_records_its_hardware(name):
    path = os.path.join(ROOT, "bench", name)
    assert os.path.exists(path), f"{name} should be committed"
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    assert "# cpu" in text, "a timing CSV must say what machine produced it"
    assert "not gated in CI" in text, "timings must be labelled as ungated"
    assert len(text.splitlines()) > 8
