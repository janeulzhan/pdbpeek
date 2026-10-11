"""Time the JavaScript and Python SASA against each other.

This is the measurement behind the README's claim that the JavaScript
implementation is substantially faster than the numpy one, which is what
prompted the blocking optimisation in pdbpeek/sasa.py. It exists so that
claim points at a command.

Both sides run `sasa_from_arrays` / `sasaFromArrays` on the same structures,
the same coordinates and radii, the same point count, and the same warmup and
run counts. Neither side does the per-residue reference pass. The totals are
compared as well as the timings, so a speed number can never come from one
side quietly doing less work.

Timings are machine dependent and are never gated in CI.

Usage:
    python3 bench/bench_cross_language.py
    python3 bench/bench_cross_language.py --csv bench/cross_language.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pdbpeek.mmcif import parse_file, select  # noqa: E402
from pdbpeek.radii import radius_for  # noqa: E402
from pdbpeek.sasa import sasa_from_arrays  # noqa: E402

# The pre-optimisation Python inner loop, kept in bench_block_size.py. Timing
# it here is what makes the README's "before" comparison reproducible from a
# single artifact instead of arithmetic across two CSVs.
from bench.bench_block_size import sasa_per_neighbour_baseline  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
JS_HARNESS = os.path.join(ROOT, "bench", "bench_js_sasa.mjs")

ENTRIES = ["1CRN", "1UBQ", "2MBW", "4HHB"]
N_POINTS = 256
RUNS = 10
WARMUP = 2


def time_python(entry: str, n_points: int, runs: int, warmup: int) -> dict:
    selected = select(
        parse_file(os.path.join(FIXTURES, f"{entry}.cif")), drop_hetatm=True
    )
    coords = np.array([[a.x, a.y, a.z] for a in selected.atoms], dtype=np.float64)
    radii = np.array([radius_for(a.element) for a in selected.atoms], dtype=np.float64)

    for _ in range(warmup):
        sasa_from_arrays(coords, radii, n_points=n_points)

    timings = []
    total = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        areas = sasa_from_arrays(coords, radii, n_points=n_points)
        timings.append(time.perf_counter() - start)
        total = float(areas.sum())

    mean = statistics.mean(timings)
    return {
        "entry": entry,
        "n_atoms": len(selected.atoms),
        "mean_seconds": mean,
        "stdev_seconds": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "atoms_per_second": len(selected.atoms) / mean,
        "total_sasa": round(total, 6),
    }


def time_python_baseline(entry: str, n_points: int, runs: int, warmup: int) -> dict:
    selected = select(
        parse_file(os.path.join(FIXTURES, f"{entry}.cif")), drop_hetatm=True
    )
    coords = np.array([[a.x, a.y, a.z] for a in selected.atoms], dtype=np.float64)
    radii = np.array([radius_for(a.element) for a in selected.atoms], dtype=np.float64)

    for _ in range(warmup):
        sasa_per_neighbour_baseline(coords, radii, n_points=n_points)

    timings = []
    total = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        areas = sasa_per_neighbour_baseline(coords, radii, n_points=n_points)
        timings.append(time.perf_counter() - start)
        total = float(areas.sum())

    mean = statistics.mean(timings)
    return {
        "mean_seconds": mean,
        "atoms_per_second": len(selected.atoms) / mean,
        "total_sasa": round(total, 6),
    }


def time_javascript(entries, n_points: int, runs: int, warmup: int) -> dict:
    proc = subprocess.run(
        [
            "node", JS_HARNESS,
            "--n-points", str(n_points),
            "--runs", str(runs),
            "--warmup", str(warmup),
            "--entries", ",".join(entries),
        ],
        capture_output=True,
        cwd=ROOT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "JS harness failed: " + proc.stderr.decode("utf-8", "replace")
        )
    return json.loads(proc.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", default=None)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--n-points", type=int, default=N_POINTS)
    args = parser.parse_args()

    def sysctl(key: str) -> str:
        try:
            return subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    cpu = sysctl("machdep.cpu.brand_string") or platform.processor()
    js = time_javascript(ENTRIES, args.n_points, args.runs, args.warmup)
    js_by_entry = {r["entry"]: r for r in js["rows"]}

    print(f"JavaScript against Python, {args.n_points} test points per atom")
    print(f"  {cpu}")
    print(f"  Python {platform.python_version()}, numpy {np.__version__}")
    print(f"  {js['runtime']}, V8 {js['v8']}")
    print(f"  {args.warmup} warmup, {args.runs} timed runs")
    print()
    header = (
        f"{'entry':6} {'atoms':>6} {'py before s':>12} {'py now s':>10} "
        f"{'js s':>9} {'py speedup':>11} {'js vs before':>13} "
        f"{'js vs now':>10} {'totals':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for entry in ENTRIES:
        before = time_python_baseline(entry, args.n_points, args.runs, args.warmup)
        py = time_python(entry, args.n_points, args.runs, args.warmup)
        j = js_by_entry[entry]
        assert py["n_atoms"] == j["n_atoms"], (entry, py["n_atoms"], j["n_atoms"])

        ratio = py["mean_seconds"] / j["mean_seconds"]
        ratio_before = before["mean_seconds"] / j["mean_seconds"]
        py_speedup = before["mean_seconds"] / py["mean_seconds"]
        # All three must agree on the answer, or the timings mean nothing.
        agree = (
            abs(py["total_sasa"] - j["total_sasa"]) / py["total_sasa"] < 1e-12
            and abs(before["total_sasa"] - py["total_sasa"]) / py["total_sasa"] < 1e-12
        )
        print(
            f"{entry:6} {py['n_atoms']:6d} {before['mean_seconds']:12.4f} "
            f"{py['mean_seconds']:10.4f} {j['mean_seconds']:9.4f} "
            f"{py_speedup:10.2f}x {ratio_before:12.1f}x {ratio:9.1f}x "
            f"{'match' if agree else 'DIFFER':>8}"
        )
        rows.append({
            "entry": entry,
            "n_atoms": py["n_atoms"],
            "n_points": args.n_points,
            "python_before_mean_seconds": before["mean_seconds"],
            "python_before_atoms_per_second": before["atoms_per_second"],
            "python_mean_seconds": py["mean_seconds"],
            "python_stdev_seconds": py["stdev_seconds"],
            "python_atoms_per_second": py["atoms_per_second"],
            "python_speedup_from_blocking": py_speedup,
            "js_mean_seconds": j["mean_seconds"],
            "js_stdev_seconds": j["stdev_seconds"],
            "js_atoms_per_second": j["atoms_per_second"],
            "js_faster_than_python_before": ratio_before,
            "js_faster_than_python_now": ratio,
            "python_before_total_sasa": before["total_sasa"],
            "python_total_sasa": py["total_sasa"],
            "js_total_sasa": j["total_sasa"],
            "totals_agree": agree,
        })

    if not all(r["totals_agree"] for r in rows):
        print("\nthe two implementations disagree on the answer", file=sys.stderr)
        return 1

    now = [r["js_faster_than_python_now"] for r in rows]
    before = [r["js_faster_than_python_before"] for r in rows]
    speedups = [r["python_speedup_from_blocking"] for r in rows]
    print()
    print(f"  Blocking made Python {min(speedups):.2f}x to {max(speedups):.2f}x "
          f"faster")
    print(f"  JavaScript was {min(before):.1f}x to {max(before):.1f}x faster than "
          f"Python before that change")
    print(f"  JavaScript is now {min(now):.1f}x to {max(now):.1f}x faster")
    print("  All three produce identical totals, so these are like-for-like.")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["# JavaScript against Python SASA"])
            writer.writerow(["# cpu", cpu])
            writer.writerow(["# python", platform.python_version()])
            writer.writerow(["# numpy", np.__version__])
            writer.writerow(["# node", js["runtime"], "v8", js["v8"]])
            writer.writerow(["# n_points", args.n_points])
            writer.writerow(["# runs", args.runs, "warmup", args.warmup])
            writer.writerow(["# note", "timings are machine dependent, not gated in CI"])
            writer.writerow(rows[0].keys())
            for row in rows:
                writer.writerow(row.values())
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
