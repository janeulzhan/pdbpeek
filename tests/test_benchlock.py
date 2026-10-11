"""Tests for tools/benchlock.py, the thing that keeps the README honest.

A gate nobody tests is a gate nobody can trust, and this one has a specific
failure mode worth guarding: silently passing. If a malformed marker were
skipped rather than reported, or a drifted number compared against itself, the
README could claim anything and CI would stay green.

These tests drive the pure functions against fixture README text and a fake
set of measured values, so they do not run the benchmark and stay fast.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.benchlock import (  # noqa: E402
    FORBIDDEN_KEYS,
    TOLERANCES,
    BenchlockError,
    collect_derived,
    flatten,
    format_value,
    is_forbidden,
    process,
    tolerance_for,
    validate_markers,
)

from .conftest import ROOT  # noqa: E402

# A fixed set of measured values, so these tests never run the benchmark.
VALUES = {
    "sasa.1crn.n_atoms": 327.0,
    "sasa.1crn.n_residues": 46.0,
    "sasa.1crn.n_contacts": 197.0,
    "sasa.1crn.total_sasa": 2962.7214,
    "sasa.4hhb.n_atoms": 4384.0,
    "sasa.4hhb.total_sasa": 26111.3534,
}


def span(key: str, value: str) -> str:
    return f"<!--bench:{key}-->{value}<!--/bench-->"


# --------------------------------------------------------------------------
# values in tolerance
# --------------------------------------------------------------------------


def test_exactly_correct_values_pass_unchanged():
    text = (
        f"Crambin has {span('sasa.1crn.n_atoms', '327')} atoms and "
        f"{span('sasa.1crn.total_sasa', '2962.7214')} square angstroms.\n"
    )
    updated, drifted, locked = process(text, VALUES)
    assert drifted == []
    assert sorted(locked) == ["sasa.1crn.n_atoms", "sasa.1crn.total_sasa"]
    assert updated == text


def test_value_inside_tolerance_passes_but_is_normalised():
    # 2962.7220 is 2e-7 away from the measured value, well inside the 1e-3
    # tolerance for a SASA total, so it passes. The rewrite still normalises
    # it to the measured value so --write converges.
    text = span("sasa.1crn.total_sasa", "2962.7220") + "\n"
    updated, drifted, _ = process(text, VALUES)
    assert drifted == []
    assert updated == span("sasa.1crn.total_sasa", "2962.7214") + "\n"


def test_thousands_separators_are_accepted():
    text = span("sasa.4hhb.n_atoms", "4,384") + "\n"
    updated, drifted, _ = process(text, VALUES)
    assert drifted == []
    assert updated == span("sasa.4hhb.n_atoms", "4384") + "\n"


def test_readme_with_no_markers_is_fine():
    updated, drifted, locked = process("Plain prose, no numbers locked.\n", VALUES)
    assert (drifted, locked) == ([], [])
    assert updated == "Plain prose, no numbers locked.\n"


# --------------------------------------------------------------------------
# drift
# --------------------------------------------------------------------------


def test_drifted_sasa_total_is_caught():
    # 1% off, ten times the declared 1e-3 tolerance.
    text = span("sasa.1crn.total_sasa", "2992.0") + "\n"
    updated, drifted, _ = process(text, VALUES)
    assert len(drifted) == 1
    assert "sasa.1crn.total_sasa" in drifted[0]
    assert "exceeds tolerance" in drifted[0]
    # The rewrite still carries the corrected value, so --write fixes it.
    assert "2962.7214" in updated


def test_integer_counts_are_gated_exactly():
    # Atom counts have a zero tolerance, so even one atom out must fail.
    text = span("sasa.1crn.n_atoms", "328") + "\n"
    _, drifted, _ = process(text, VALUES)
    assert len(drifted) == 1
    assert "committed 328, measured 327" in drifted[0]


def test_a_correct_count_passes_the_zero_tolerance():
    _, drifted, _ = process(span("sasa.1crn.n_atoms", "327") + "\n", VALUES)
    assert drifted == []


def test_several_drifts_are_all_reported():
    text = (
        span("sasa.1crn.n_atoms", "1")
        + span("sasa.1crn.n_residues", "2")
        + span("sasa.4hhb.total_sasa", "3")
        + "\n"
    )
    _, drifted, locked = process(text, VALUES)
    assert len(drifted) == 3
    assert len(locked) == 3


def test_non_numeric_committed_value_is_reported_not_crashed():
    text = span("sasa.1crn.n_atoms", "many") + "\n"
    updated, drifted, _ = process(text, VALUES)
    assert len(drifted) == 1
    assert "is not a number" in drifted[0]
    assert "327" in updated


def test_empty_committed_value_is_reported():
    text = span("sasa.1crn.n_atoms", "") + "\n"
    _, drifted, _ = process(text, VALUES)
    assert len(drifted) == 1
    assert "is not a number" in drifted[0]


# --------------------------------------------------------------------------
# malformed markers
# --------------------------------------------------------------------------


def test_unclosed_marker_raises():
    text = "The total is <!--bench:sasa.1crn.n_atoms-->327 and that is that.\n"
    with pytest.raises(BenchlockError, match="unbalanced bench markers"):
        process(text, VALUES)


def test_stray_closing_marker_raises():
    text = "Nothing opened this. <!--/bench-->\n"
    with pytest.raises(BenchlockError, match="unbalanced bench markers"):
        process(text, VALUES)


def test_closing_before_opening_raises():
    text = "<!--/bench-->327<!--bench:sasa.1crn.n_atoms-->\n"
    with pytest.raises(BenchlockError, match="closing bench marker with no opening"):
        process(text, VALUES)


def test_nested_markers_raise():
    text = (
        "<!--bench:sasa.1crn.n_atoms-->"
        "<!--bench:sasa.1crn.n_residues-->46<!--/bench-->"
        "<!--/bench-->\n"
    )
    with pytest.raises(BenchlockError, match="nested bench markers"):
        process(text, VALUES)


def test_empty_key_raises():
    with pytest.raises(BenchlockError, match="empty key"):
        process("<!--bench:-->327<!--/bench-->\n", VALUES)


def test_span_crossing_a_newline_raises():
    text = "<!--bench:sasa.1crn.n_atoms-->\n327\n<!--/bench-->\n"
    with pytest.raises(BenchlockError, match="malformed"):
        process(text, VALUES)


def test_marker_error_reports_a_line_number():
    text = "line one\nline two\n<!--bench:-->327<!--/bench-->\n"
    with pytest.raises(BenchlockError, match="line 3"):
        validate_markers(text)


def test_unknown_key_raises():
    text = span("sasa.9zzz.total_sasa", "100") + "\n"
    with pytest.raises(BenchlockError, match="unknown bench key"):
        process(text, VALUES)


def test_key_without_a_declared_tolerance_raises():
    text = span("sasa.1crn.something_new", "1") + "\n"
    with pytest.raises(BenchlockError, match="no tolerance declared"):
        process(text, {**VALUES, "sasa.1crn.something_new": 1.0})


# --------------------------------------------------------------------------
# timings must never be gated
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "sasa.1crn.mean_seconds",
        "sasa.1crn.atoms_per_second",
        "sasa.1crn.parse_seconds",
        "sasa.1crn.wall_clock",
        "sasa.1crn.elapsed",
        "sasa.1crn.total_time",
        "sasa.1crn.duration",
        "sasa.1crn.throughput",
    ],
)
def test_timing_keys_are_refused(key):
    assert is_forbidden(key)
    text = span(key, "0.0433") + "\n"
    with pytest.raises(BenchlockError, match="wall-clock"):
        process(text, {**VALUES, key: 0.0433})


def test_deterministic_keys_are_not_refused():
    for key in VALUES:
        assert not is_forbidden(key), key


def test_timing_fields_never_reach_the_lockable_set():
    # flatten() must drop them at the source, so --list cannot even offer one.
    payload = {
        "rows": [
            {
                "entry": "1CRN",
                "n_atoms": 327,
                "total_sasa": 2962.7214,
                "mean_seconds": 0.0433,
                "stdev_seconds": 0.0032,
                "atoms_per_second": 7553.0,
                "parse_seconds": 0.01,
                "parse_atoms_per_second": 1000.0,
                "atom_points_per_second": 1.9e6,
                "min_seconds": 0.04,
                "max_seconds": 0.05,
                "runs": 10,
                "warmup": 2,
            }
        ]
    }
    keys = set(flatten(payload))
    assert keys == {"sasa.1crn.n_atoms", "sasa.1crn.total_sasa"}
    assert not any(is_forbidden(k) for k in keys)


def test_every_declared_tolerance_is_for_a_deterministic_quantity():
    for pattern, _, _ in TOLERANCES:
        assert not is_forbidden(pattern), pattern


def test_forbidden_list_is_not_empty():
    assert FORBIDDEN_KEYS
    assert "seconds" in FORBIDDEN_KEYS


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def test_tolerance_lookup_uses_first_match():
    assert tolerance_for("sasa.1crn.n_atoms") == (0.0, 0)
    assert tolerance_for("sasa.1crn.total_sasa") == (1e-3, 4)


def test_format_value_rounds_to_the_declared_places():
    assert format_value(327.0, 0) == "327"
    assert format_value(326.6, 0) == "327"
    assert format_value(2962.72139, 4) == "2962.7214"


def test_derived_totals_are_sums_of_the_parts():
    derived = collect_derived(VALUES)
    assert derived["sasa.all.n_atoms"] == 327.0 + 4384.0
    assert derived["sasa.all.total_sasa"] == pytest.approx(2962.7214 + 26111.3534)


def test_flatten_skips_bookkeeping_fields():
    payload = {"rows": [{"entry": "1CRN", "n_atoms": 1, "runs": 10, "warmup": 2}]}
    assert set(flatten(payload)) == {"sasa.1crn.n_atoms"}


# --------------------------------------------------------------------------
# the real README
# --------------------------------------------------------------------------


def test_readme_markers_are_well_formed():
    # Cheap structural check that runs offline. The value check needs the
    # benchmark, which is what CI's `benchlock --check` step does.
    with open(os.path.join(ROOT, "README.md"), "r", encoding="utf-8") as fh:
        text = fh.read()
    validate_markers(text)


def test_readme_locks_at_least_a_few_numbers():
    with open(os.path.join(ROOT, "README.md"), "r", encoding="utf-8") as fh:
        text = fh.read()
    from tools.benchlock import MARKER

    keys = [m.group(1) for m in MARKER.finditer(text)]
    assert len(keys) >= 5, "the README should lock its measured numbers"
    assert not any(is_forbidden(k) for k in keys), "a timing is being gated"
    for key in keys:
        tolerance_for(key)  # raises if undeclared


def test_cli_list_mode_runs():
    proc = subprocess.run(
        [sys.executable, "tools/benchlock.py", "--list", "--runs", "1", "--n-points", "8"],
        capture_output=True,
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    out = proc.stdout.decode("utf-8")
    assert "lockable keys" in out
    assert "sasa.1crn.n_atoms" in out
    assert "Refused by name" in out


def test_cli_reports_drift_with_a_non_zero_exit(tmp_path):
    fake = tmp_path / "DRIFTED.md"
    fake.write_text(span("sasa.1crn.n_atoms", "99999") + "\n", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "tools/benchlock.py",
            "--check",
            "--readme",
            str(fake),
            "--runs",
            "1",
            "--n-points",
            "8",
        ],
        capture_output=True,
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode == 1
    assert b"drifted" in proc.stderr


def test_cli_write_mode_fixes_a_drifted_file(tmp_path):
    fake = tmp_path / "DRIFTED.md"
    fake.write_text(span("sasa.1crn.n_atoms", "99999") + "\n", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "tools/benchlock.py",
            "--write",
            "--readme",
            str(fake),
            "--runs",
            "1",
            "--n-points",
            "8",
        ],
        capture_output=True,
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert fake.read_text(encoding="utf-8") == span("sasa.1crn.n_atoms", "327") + "\n"

    # And the file now passes --check, so --write converges in one pass.
    recheck = subprocess.run(
        [
            sys.executable,
            "tools/benchlock.py",
            "--check",
            "--readme",
            str(fake),
            "--runs",
            "1",
            "--n-points",
            "8",
        ],
        capture_output=True,
        cwd=ROOT,
        check=False,
    )
    assert recheck.returncode == 0, recheck.stderr.decode("utf-8", "replace")


def test_cli_malformed_file_exits_two(tmp_path):
    fake = tmp_path / "BROKEN.md"
    fake.write_text("<!--bench:sasa.1crn.n_atoms-->327\n", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "tools/benchlock.py",
            "--check",
            "--readme",
            str(fake),
            "--runs",
            "1",
            "--n-points",
            "8",
        ],
        capture_output=True,
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode == 2
    assert b"unbalanced" in proc.stderr
