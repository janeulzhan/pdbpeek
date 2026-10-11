"""Benchmark harness for the SASA implementation.

Two kinds of number come out of this, and the README keeps them apart on
purpose.

Deterministic: total SASA, atom counts, residue counts, contact counts. These
depend only on the input and the algorithm, so they are reproducible anywhere
and `tools/benchlock.py` gates them in CI.

Timing: wall-clock seconds and atoms per second. These depend on the machine,
so they are reported with the hardware stated and are never gated in CI. A
shared CI runner's timings would mean nothing.

Usage:
    python3 bench/bench_sasa.py                     # human readable table
    python3 bench/bench_sasa.py --csv bench/results.csv
    python3 bench/bench_sasa.py --json bench/results.json
    python3 bench/bench_sasa.py --runs 10 --warmup 2
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pdbpeek.contacts import contact_map  # noqa: E402
from pdbpeek.mmcif import parse_file, select  # noqa: E402
from pdbpeek.sasa import compute_sasa, sasa_from_arrays  # noqa: E402

import numpy as np  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")

#: Structures in increasing size, so the scaling is visible.
ENTRIES = ["1CRN", "1UBQ", "2MBW", "4HHB"]

DEFAULT_N_POINTS = 256
DEFAULT_RUNS = 10
DEFAULT_WARMUP = 2


def hardware() -> dict:
    """Describe the machine, read from the system rather than hardcoded."""

    def sysctl(key: str) -> str:
        try:
            return subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    memsize = sysctl("hw.memsize")
    gib = f"{int(memsize) / (1024 ** 3):.0f} GiB" if memsize.isdigit() else ""

    return {
        "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "memory": gib,
        "memory_bytes": memsize,
        "platform": f"{platform.system()} {platform.release()}",
        "os_version": platform.mac_ver()[0] or platform.version(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def bench_entry(entry: str, n_points: int, runs: int, warmup: int) -> dict:
    """Time SASA on one entry and record the deterministic values alongside."""
    path = os.path.join(FIXTURES, f"{entry}.cif")

    parse_start = time.perf_counter()
    raw = parse_file(path)
    parse_seconds = time.perf_counter() - parse_start

    selected = select(raw, drop_hetatm=True)
    coords = np.array([[a.x, a.y, a.z] for a in selected.atoms], dtype=np.float64)
    from pdbpeek.radii import radius_for

    radii = np.array([radius_for(a.element) for a in selected.atoms], dtype=np.float64)

    # Warmup runs are discarded: they pay the numpy and scipy import and
    # first-touch costs that a steady-state measurement should not include.
    for _ in range(warmup):
        sasa_from_arrays(coords, radii, n_points=n_points)

    timings: list[float] = []
    total = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        areas = sasa_from_arrays(coords, radii, n_points=n_points)
        timings.append(time.perf_counter() - start)
        total = float(areas.sum())

    result = compute_sasa(selected, n_points=n_points, reference=True)
    burial = result.burial() or []
    cmap = contact_map(selected)

    mean = statistics.mean(timings)
    stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0
    n_atoms = len(selected.atoms)

    return {
        "entry": entry,
        # deterministic
        "n_atoms": n_atoms,
        "n_residues": len(result.residue_keys),
        "n_chains": len(selected.chains),
        "n_contacts": cmap.n_contacts,
        "n_buried": burial.count("buried"),
        "total_sasa": round(total, 4),
        "n_points": n_points,
        "file_atoms": len(raw.atoms),
        "models_in_file": len(raw.models),
        # timing, machine dependent
        "runs": runs,
        "warmup": warmup,
        "mean_seconds": mean,
        "stdev_seconds": stdev,
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "atoms_per_second": n_atoms / mean,
        "atom_points_per_second": n_atoms * n_points / mean,
        "parse_seconds": parse_seconds,
        "parse_atoms_per_second": len(raw.atoms) / parse_seconds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark pdbpeek SASA.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS)
    parser.add_argument("--entries", nargs="*", default=ENTRIES)
    parser.add_argument("--csv", default=None, help="write the raw rows here")
    parser.add_argument("--json", default=None, help="write nested JSON here")
    parser.add_argument(
        "--stdout-json",
        action="store_true",
        help="emit the JSON payload on stdout, for tools/benchlock.py",
    )
    args = parser.parse_args(argv)

    rows = [
        bench_entry(entry, args.n_points, args.runs, args.warmup)
        for entry in args.entries
    ]
    info = hardware()
    payload = {"hardware": info, "n_points": args.n_points, "rows": rows}

    if args.stdout_json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    print(f"pdbpeek SASA benchmark, {args.n_points} test points per atom")
    print(f"  {info['cpu']}, {info['memory']}, {info['platform']} {info['os_version']}")
    print(f"  Python {info['python']}, numpy {info['numpy']}")
    print(f"  {args.warmup} warmup runs discarded, {args.runs} timed runs")
    print()
    header = (
        f"{'entry':6} {'atoms':>6} {'residues':>9} {'total SASA':>12} "
        f"{'mean s':>9} {'stddev s':>9} {'atoms/s':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['entry']:6} {row['n_atoms']:6d} {row['n_residues']:9d} "
            f"{row['total_sasa']:12.1f} {row['mean_seconds']:9.4f} "
            f"{row['stdev_seconds']:9.4f} {row['atoms_per_second']:10.0f}"
        )

    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["# pdbpeek SASA benchmark"])
            writer.writerow(["# cpu", info["cpu"]])
            writer.writerow(["# memory", info["memory"]])
            writer.writerow(["# platform", f"{info['platform']} {info['os_version']}"])
            writer.writerow(["# python", info["python"]])
            writer.writerow(["# numpy", info["numpy"]])
            writer.writerow(
                ["# note", "timings are machine dependent and not gated in CI"]
            )
            writer.writerow(rows[0].keys())
            for row in rows:
                writer.writerow(row.values())
        print(f"\nwrote {args.csv}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
