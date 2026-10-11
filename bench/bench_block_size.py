"""Sweep the neighbour block size in the SASA inner loop.

This is the measurement behind `_NEIGHBOUR_BLOCK = 8` in pdbpeek/sasa.py and
behind the block-size numbers quoted in the README. It exists so those
numbers point at a command rather than at a claim.

What is being traded off. The inner loop tests an atom's test points against
its neighbours, nearest neighbour first, and stops as soon as every point is
buried. Testing one neighbour per numpy call makes that early exit maximally
effective but pays numpy's per-call dispatch cost once per (atom, neighbour)
pair. Testing a block of neighbours per call amortises the dispatch but
evaluates the whole block before checking, so a large block throws the early
exit away. The optimum is therefore an empirical question, and guessing it
wrong is easy: 64 is slower than not blocking at all.

Usage:
    python3 bench/bench_block_size.py
    python3 bench/bench_block_size.py --csv bench/block_size.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pdbpeek.sasa as sasa_module  # noqa: E402
from pdbpeek.mmcif import parse_file, select  # noqa: E402
from pdbpeek.radii import radius_for  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")

BLOCK_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
ENTRIES = ["1CRN", "4HHB"]
N_POINTS = 256
RUNS = 5
WARMUP = 1


def load(entry: str):
    selected = select(parse_file(os.path.join(FIXTURES, f"{entry}.cif")), drop_hetatm=True)
    coords = np.array([[a.x, a.y, a.z] for a in selected.atoms], dtype=np.float64)
    radii = np.array([radius_for(a.element) for a in selected.atoms], dtype=np.float64)
    return coords, radii


def sasa_per_neighbour_baseline(
    coords: np.ndarray,
    radii: np.ndarray,
    probe: float = 1.4,
    n_points: int = N_POINTS,
) -> np.ndarray:
    """The pre-optimisation inner loop, kept so its timing stays reproducible.

    This is exactly what pdbpeek/sasa.py did before blocking was introduced:
    one numpy call per (atom, neighbour) pair, with the early exit checked
    after every single neighbour. It is dead code as far as the package is
    concerned and lives here only so the "before" number in the README points
    at something runnable rather than at a memory.

    `test_bench_scripts.py` asserts this returns the same areas as the
    shipping implementation, so it cannot silently drift into a strawman.
    """
    from scipy.spatial import cKDTree

    coords = np.ascontiguousarray(coords, dtype=np.float64)
    radii = np.ascontiguousarray(radii, dtype=np.float64)
    n_atoms = coords.shape[0]
    ext = radii + probe
    sphere = sasa_module.golden_spiral_points(n_points)
    const = 4.0 * np.pi / n_points

    tree = cKDTree(coords)
    neighbour_lists = tree.query_ball_point(coords, float(ext.max()) * 2.0)

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

        order = np.argsort(np.linalg.norm(coords[js] - coords[i], axis=1))
        js, rj = js[order], rj[order]

        test = coords[i] + sphere * ri
        alive = np.ones(n_points, dtype=bool)
        n_alive = n_points
        for j, r in zip(js, rj):
            delta = test[alive] - coords[j]
            inside = np.einsum("ij,ij->i", delta, delta) < r * r
            if inside.any():
                idx = np.flatnonzero(alive)
                alive[idx[inside]] = False
                n_alive -= int(inside.sum())
                if n_alive == 0:
                    break
        out[i] = const * ri * ri * n_alive

    return out


def time_baseline(coords, radii, runs: int, warmup: int) -> tuple[list[float], float]:
    for _ in range(warmup):
        sasa_per_neighbour_baseline(coords, radii)
    timings = []
    total = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        areas = sasa_per_neighbour_baseline(coords, radii)
        timings.append(time.perf_counter() - start)
        total = float(areas.sum())
    return timings, total


def neighbour_pair_count(coords: np.ndarray, radii: np.ndarray, probe: float) -> int:
    """How many (atom, neighbour) pairs the inner loop actually visits.

    This is the number of numpy calls a per-neighbour loop would issue, which
    is the quantity the blocking optimisation is trying to reduce.
    """
    from scipy.spatial import cKDTree

    ext = radii + probe
    tree = cKDTree(coords)
    total = 0
    for i, neighbours in enumerate(tree.query_ball_point(coords, float(ext.max()) * 2.0)):
        others = [j for j in neighbours if j != i]
        if not others:
            continue
        js = np.asarray(others, dtype=np.int64)
        d = np.linalg.norm(coords[js] - coords[i], axis=1)
        total += int((d < (ext[i] + ext[js])).sum())
    return total


def time_block(coords, radii, block: int, runs: int, warmup: int) -> tuple[list[float], float]:
    original = sasa_module._NEIGHBOUR_BLOCK
    sasa_module._NEIGHBOUR_BLOCK = block
    try:
        for _ in range(warmup):
            sasa_module.sasa_from_arrays(coords, radii, n_points=N_POINTS)
        timings = []
        total = 0.0
        for _ in range(runs):
            start = time.perf_counter()
            areas = sasa_module.sasa_from_arrays(coords, radii, n_points=N_POINTS)
            timings.append(time.perf_counter() - start)
            total = float(areas.sum())
        return timings, total
    finally:
        sasa_module._NEIGHBOUR_BLOCK = original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", default=None)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    args = parser.parse_args()

    def sysctl(key: str) -> str:
        try:
            return subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    cpu = sysctl("machdep.cpu.brand_string") or platform.processor()
    print(f"neighbour block size sweep, {N_POINTS} test points per atom")
    print(f"  {cpu}, Python {platform.python_version()}, numpy {np.__version__}")
    print(f"  {args.warmup} warmup, {args.runs} timed runs per cell")
    print(f"  default in pdbpeek/sasa.py is _NEIGHBOUR_BLOCK = "
          f"{sasa_module._NEIGHBOUR_BLOCK}")
    print()

    loaded = {entry: load(entry) for entry in ENTRIES}
    for entry, (coords, radii) in loaded.items():
        pairs = neighbour_pair_count(coords, radii, sasa_module.PROBE_RADIUS)
        print(f"  {entry}: {len(coords)} atoms, {pairs} (atom, neighbour) pairs "
              f"visited per SASA call")
    print()

    rows = []
    header = f"{'block':>6} " + " ".join(f"{e:>24}" for e in ENTRIES)
    print(header)
    print("-" * len(header))

    totals_seen: dict[str, set] = {e: set() for e in ENTRIES}

    # The historical per-neighbour loop, for the "before" number.
    baseline_cells = []
    for entry in ENTRIES:
        coords, radii = loaded[entry]
        timings, total = time_baseline(coords, radii, args.runs, args.warmup)
        mean = statistics.mean(timings)
        stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0
        totals_seen[entry].add(round(total, 6))
        baseline_cells.append(f"{mean:.4f}s {len(coords) / mean:8.0f}/s")
        rows.append({
            "entry": entry,
            "n_atoms": len(coords),
            "n_points": N_POINTS,
            "block_size": "per_neighbour_baseline",
            "mean_seconds": mean,
            "stdev_seconds": stdev,
            "atoms_per_second": len(coords) / mean,
            "total_sasa": round(total, 6),
        })
    print(f"{'none':>6} " + " ".join(f"{c:>24}" for c in baseline_cells)
          + "   <- per-neighbour loop, the original")

    for block in BLOCK_SIZES:
        cells = []
        for entry in ENTRIES:
            coords, radii = loaded[entry]
            timings, total = time_block(coords, radii, block, args.runs, args.warmup)
            mean = statistics.mean(timings)
            stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0
            rate = len(coords) / mean
            totals_seen[entry].add(round(total, 6))
            cells.append(f"{mean:.4f}s {rate:8.0f}/s")
            rows.append({
                "entry": entry,
                "n_atoms": len(coords),
                "n_points": N_POINTS,
                "block_size": block,
                "mean_seconds": mean,
                "stdev_seconds": stdev,
                "atoms_per_second": rate,
                "total_sasa": round(total, 6),
            })
        print(f"{block:>6} " + " ".join(f"{c:>24}" for c in cells))

    print()
    # Blocking must not change the answer. If it did, the speed comparison
    # would be meaningless.
    for entry, totals in totals_seen.items():
        status = "identical" if len(totals) == 1 else f"DIFFERENT: {sorted(totals)}"
        print(f"  {entry} total SASA across every block size: {status}")
    if any(len(t) != 1 for t in totals_seen.values()):
        print("\nblock size changed the result, which is a bug", file=sys.stderr)
        return 1

    print()
    for entry in ENTRIES:
        entry_rows = [r for r in rows if r["entry"] == entry]
        sweep = [r for r in entry_rows if r["block_size"] != "per_neighbour_baseline"]
        best = min(sweep, key=lambda r: r["mean_seconds"])
        baseline = next(
            r for r in entry_rows if r["block_size"] == "per_neighbour_baseline"
        )
        block64 = next(r for r in sweep if r["block_size"] == 64)
        print(f"  {entry}:")
        print(f"    per-neighbour baseline  {baseline['mean_seconds']:.4f}s")
        print(f"    block 64                {block64['mean_seconds']:.4f}s"
              f"   ({baseline['mean_seconds'] / block64['mean_seconds']:.2f}x "
              f"vs baseline)")
        print(f"    block {best['block_size']:<18}{best['mean_seconds']:.4f}s"
              f"   ({baseline['mean_seconds'] / best['mean_seconds']:.2f}x "
              f"vs baseline)  fastest")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["# neighbour block size sweep"])
            writer.writerow(["# cpu", cpu])
            writer.writerow(["# python", platform.python_version()])
            writer.writerow(["# numpy", np.__version__])
            writer.writerow(["# n_points", N_POINTS])
            writer.writerow(["# runs", args.runs, "warmup", args.warmup])
            writer.writerow(["# note", "timings are machine dependent, not gated in CI"])
            writer.writerow(rows[0].keys())
            for row in rows:
                writer.writerow(row.values())
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
