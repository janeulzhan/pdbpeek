"""Keep the numbers in the README honest.

Every measured number in README.md lives inside a marker span:

    <!--bench:sasa.1crn.total_sasa-->2962.7214<!--/bench-->

This tool recomputes those values and either rewrites the spans or fails if a
committed number has drifted. CI runs it in `--check` mode, so a change that
moves a SASA total cannot be merged with a README that still claims the old
one. The README stops being prose that happens to contain numbers and becomes
something the build verifies.

Only deterministic quantities may be locked. SASA totals, atom counts,
residue counts, contact counts: these depend on the input and the algorithm,
nothing else, so they are reproducible on any machine. Wall-clock timings are
refused outright, by name, in `FORBIDDEN_KEYS` below. A timing locked here
would turn a slow CI runner into a red build and teach everyone to ignore the
gate. Timings belong in the README's own table, labelled with the hardware
they came from.

Usage:
    python3 tools/benchlock.py --list            # what can be locked
    python3 tools/benchlock.py --check           # CI mode, exits non-zero on drift
    python3 tools/benchlock.py --write           # update README.md in place
    python3 tools/benchlock.py --check --readme other.md
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_README = os.path.join(ROOT, "README.md")
BENCH = os.path.join(ROOT, "bench", "bench_sasa.py")

MARKER = re.compile(r"<!--bench:([A-Za-z0-9_.\-]+)-->(.*?)<!--/bench-->")
OPEN_MARKER = re.compile(r"<!--bench:([A-Za-z0-9_.\-]*)-->")
CLOSE_MARKER = re.compile(r"<!--/bench-->")

#: Substrings that mark a value as machine dependent. A key containing any of
#: these is refused even if it exists in the benchmark output, so a timing can
#: never become a CI gate by accident.
FORBIDDEN_KEYS = (
    "second",
    "seconds",
    "per_second",
    "time",
    "elapsed",
    "wall",
    "duration",
    "throughput",
)

#: (key glob, relative tolerance, decimal places). First match wins.
#: Integer counts are exact. SASA totals get 0.1%, which covers a different
#: platform's libm rounding a sin or cos differently and flipping a borderline
#: test point, while still catching any real change in the algorithm.
TOLERANCES: tuple[tuple[str, float, int], ...] = (
    ("*.n_atoms", 0.0, 0),
    ("*.n_residues", 0.0, 0),
    ("*.n_chains", 0.0, 0),
    ("*.n_contacts", 0.0, 0),
    ("*.n_buried", 0.0, 0),
    ("*.n_points", 0.0, 0),
    ("*.file_atoms", 0.0, 0),
    ("*.models_in_file", 0.0, 0),
    ("*.total_sasa", 1e-3, 4),
    ("*.mean_relative_error", 5e-2, 5),
)


class BenchlockError(RuntimeError):
    """Raised for a malformed README or an unlockable key."""


def tolerance_for(key: str) -> tuple[float, int]:
    for pattern, rel_tol, places in TOLERANCES:
        if fnmatch.fnmatch(key, pattern):
            return rel_tol, places
    raise BenchlockError(
        f"no tolerance declared for {key!r}; add one to TOLERANCES in "
        f"tools/benchlock.py before locking it"
    )


def is_forbidden(key: str) -> bool:
    lowered = key.lower()
    return any(bad in lowered for bad in FORBIDDEN_KEYS)


# --------------------------------------------------------------------------
# collecting values
# --------------------------------------------------------------------------


def run_bench(runs: int, n_points: int) -> dict:
    """Run the benchmark harness and return its JSON payload."""
    proc = subprocess.run(
        [
            sys.executable,
            BENCH,
            "--stdout-json",
            "--runs",
            str(runs),
            "--n-points",
            str(n_points),
        ],
        capture_output=True,
        cwd=ROOT,
        check=False,
    )
    if proc.returncode != 0:
        raise BenchlockError(
            "bench/bench_sasa.py failed: " + proc.stderr.decode("utf-8", "replace")
        )
    return json.loads(proc.stdout)


def flatten(payload: dict) -> dict[str, float]:
    """Flatten the benchmark payload to dotted keys.

    Timing fields are dropped here rather than filtered later, so they are
    never even offered as lockable values.
    """
    out: dict[str, float] = {}
    for row in payload["rows"]:
        entry = str(row["entry"]).lower()
        for field, value in row.items():
            if field in ("entry", "runs", "warmup"):
                continue
            key = f"sasa.{entry}.{field}"
            if is_forbidden(key):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = float(value)
    return out


def collect_derived(values: dict[str, float]) -> dict[str, float]:
    """Add cross-entry values the README quotes, computed from the same run."""
    totals = {k: v for k, v in values.items() if k.endswith(".total_sasa")}
    atoms = {k: v for k, v in values.items() if k.endswith(".n_atoms")}
    derived = {
        "sasa.all.total_sasa": sum(totals.values()),
        "sasa.all.n_atoms": sum(atoms.values()),
    }
    return {**values, **derived}


# --------------------------------------------------------------------------
# the README
# --------------------------------------------------------------------------


def validate_markers(text: str) -> None:
    """Fail loudly on an unbalanced or empty marker span."""
    opens = list(OPEN_MARKER.finditer(text))
    closes = list(CLOSE_MARKER.finditer(text))

    for match in opens:
        if not match.group(1):
            line = text[: match.start()].count("\n") + 1
            raise BenchlockError(f"line {line}: bench marker with an empty key")

    if len(opens) != len(closes):
        raise BenchlockError(
            f"unbalanced bench markers: {len(opens)} opening, {len(closes)} closing"
        )

    # Spans must be well nested and in order: open, close, open, close.
    positions = [(m.start(), "open") for m in opens] + [
        (m.start(), "close") for m in closes
    ]
    positions.sort()
    depth = 0
    for offset, kind in positions:
        depth += 1 if kind == "open" else -1
        line = text[:offset].count("\n") + 1
        if depth > 1:
            raise BenchlockError(f"line {line}: nested bench markers")
        if depth < 0:
            raise BenchlockError(f"line {line}: closing bench marker with no opening")

    # Every balanced span must actually match the full pattern. A span whose
    # body contains a newline or another comment would slip past the count
    # check above but not this one.
    if len(MARKER.findall(text)) != len(opens):
        raise BenchlockError(
            "a bench span is malformed: expected "
            "<!--bench:key-->value<!--/bench--> on one line"
        )


def format_value(value: float, places: int) -> str:
    if places == 0:
        return str(int(round(value)))
    return f"{value:.{places}f}"


def process(text: str, values: dict[str, float]) -> tuple[str, list[str], list[str]]:
    """Return the rewritten text, the drifted keys and the locked keys."""
    validate_markers(text)

    drifted: list[str] = []
    locked: list[str] = []

    def replace(match: re.Match) -> str:
        key, committed = match.group(1), match.group(2).strip()
        locked.append(key)

        if is_forbidden(key):
            raise BenchlockError(
                f"{key!r} looks like a wall-clock measurement. Timings are "
                f"machine dependent, so they are never gated. Put it in the "
                f"README timing table instead."
            )
        if key not in values:
            raise BenchlockError(
                f"unknown bench key {key!r}. Run "
                f"`python3 tools/benchlock.py --list` to see what is available."
            )

        rel_tol, places = tolerance_for(key)
        fresh = values[key]
        formatted = format_value(fresh, places)

        committed_clean = committed.replace(",", "")
        try:
            committed_value = float(committed_clean)
        except ValueError:
            drifted.append(
                f"{key}: committed {committed!r} is not a number, measured {formatted}"
            )
            return f"<!--bench:{key}-->{formatted}<!--/bench-->"

        denominator = abs(fresh) if fresh else 1.0
        delta = abs(committed_value - fresh) / denominator
        if delta > rel_tol:
            drifted.append(
                f"{key}: committed {committed_value:g}, measured {formatted}, "
                f"relative drift {delta:.2e} exceeds tolerance {rel_tol:.0e}"
            )
        return f"<!--bench:{key}-->{formatted}<!--/bench-->"

    return MARKER.sub(replace, text), drifted, locked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="fail if a committed number drifted beyond its tolerance",
    )
    mode.add_argument("--write", action="store_true", help="rewrite README.md in place")
    mode.add_argument("--list", action="store_true", help="list lockable keys")
    parser.add_argument("--readme", default=DEFAULT_README)
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="timed runs per entry; only affects timing, which is not gated",
    )
    parser.add_argument("--n-points", type=int, default=256)
    args = parser.parse_args(argv)

    try:
        values = collect_derived(flatten(run_bench(args.runs, args.n_points)))

        if args.list:
            print(f"{len(values)} lockable keys:")
            for key in sorted(values):
                rel_tol, places = tolerance_for(key)
                print(
                    f"  <!--bench:{key}-->{format_value(values[key], places)}"
                    f"<!--/bench-->   (tolerance {rel_tol:.0e})"
                )
            print(
                "\nRefused by name, because they are machine dependent: "
                + ", ".join(FORBIDDEN_KEYS)
            )
            return 0

        with open(args.readme, "r", encoding="utf-8") as fh:
            text = fh.read()

        updated, drifted, locked = process(text, values)

        if not locked:
            print(f"benchlock: no bench markers found in {args.readme}")
            return 0

        if args.write:
            if updated != text:
                with open(args.readme, "w", encoding="utf-8") as fh:
                    fh.write(updated)
                print(f"benchlock: rewrote {len(locked)} values in {args.readme}")
            else:
                print(f"benchlock: {len(locked)} values already current")
            return 0

        # Default and --check both verify.
        if drifted:
            print(
                f"benchlock: {len(drifted)} of {len(locked)} committed numbers "
                f"drifted",
                file=sys.stderr,
            )
            for line in drifted:
                print(f"  {line}", file=sys.stderr)
            print(
                "\nRe-measure and update with: python3 tools/benchlock.py --write",
                file=sys.stderr,
            )
            return 1

        print(f"benchlock: {len(locked)} committed numbers verified")
        return 0

    except BenchlockError as exc:
        print(f"benchlock: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
